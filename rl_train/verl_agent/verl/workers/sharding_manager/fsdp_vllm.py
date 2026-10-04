# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
import os
import time
from collections import OrderedDict
from typing import List

import torch
from peft import PeftModel
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from dataclasses import asdict

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.device import get_torch_device
from verl.utils.fsdp_utils import fsdp_version, layered_summon_lora_params, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.torch_functional import check_cuda_is_available
from verl.utils.vllm_utils import TensorLoRARequest, VLLMHijack, is_version_ge, patch_vllm_moe_model_weight_loader

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))



class FSDPVLLMShardingManager(BaseShardingManager):
    @check_cuda_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
        load_format: str = 'dummy_hf',
        layered_summon: bool = True
    ):
        self.module = module
        # For AsyncLLM, inference_engine and model_runner are defer intialized in vLLMAsyncRollout.load_model
        self.inference_engine = inference_engine
        # self.model_runner = inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if inference_engine else None

        if "vllm_v_0_6_3" in str(type(self.inference_engine)) or "vllm_v_0_5_4" in str(type(self.inference_engine)):
            # vLLM <= v0.6.3
            self.model_runner = self.inference_engine.llm_engine.model_executor.worker.model_runner if self.inference_engine else None
        else:
            # vLLM > v0.6.3
            self.model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if self.inference_engine else None

        self.model_config = model_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.load_format = load_format
        self.layered_summon = layered_summon

        # Full params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig())
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = get_torch_device().get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

        self.base_sync_done: bool = 'dummy' not in load_format
        if is_version_ge(pkg='vllm', minver='0.7.3'):
            VLLMHijack.hijack()

        # ⚠ 10-04（grpo_h29 两个 8 卡任务分别在 step 10 / step 25 静默卡死 19~45h）：
        # verl-agent 的多轮 rollout 每走一步环境就调一次 generate_sequences，每次都
        # 完整进出本 context manager——上游原版语义是"每次 generate 都把 FSDP 权重
        # 重新 state_dict() → 逐参数 DTensor.full_tensor()（8 卡 NCCL all-gather）→
        # vLLM load_weights，结束再 sleep(level=1)"。单轮对话 RL 一个训练步只 generate
        # 一次，没问题；我们 200 步 episode 就变成**每个训练步 200 次全量权重同步 +
        # 200 次 sleep/wake**（验证再加 200 次），权重在 rollout 期间根本没变过：
        #   - 性能：smoke16 单帧 prompt 实测 6.7s/环境步，真正生成 <1s，其余几乎全是
        #     这套同步；h29 正式训练 14s/环境步里同样有大头是它
        #   - 稳定性：每个训练步 ~1.3 万轮跨 8 卡集合通信 + CuMemAllocator 反复搬运，
        #     是两次静默卡死最可疑的位置（无超时的 driver ray.get 会一直等）
        # 开关 VERL_ROLLOUT_KEEP_AWAKE=1 时：
        #   ① 权重未变（自上次同步后没有 update_actor/load_checkpoint）就不再同步，
        #      sleep(level=1) 只是把权重挪到 CPU，wake_up 原样恢复，数值与同步后一致
        #   ② 连续多次 generate 之间保持 vLLM 唤醒，不 sleep；直到 worker 要跑别的 GPU
        #      计算（log_prob / ref / update / 存档）时由 worker 调 release() 再 sleep
        # 显存峰值不变：两次 generate 之间 GPU 上的东西与 generate 进行中完全相同。
        # 默认关闭（=上游原行为），由训练脚本显式打开。
        self.keep_awake: bool = os.environ.get("VERL_ROLLOUT_KEEP_AWAKE", "0") == "1"
        self._weights_dirty: bool = True
        self._awake: bool = False
        self._n_full_sync = 0
        if self.keep_awake:
            print("[fsdp_vllm] VERL_ROLLOUT_KEEP_AWAKE=1：权重未变时跳过同步，rollout 期间保持唤醒", flush=True)

    def mark_weights_dirty(self):
        """FSDP 权重发生变化（update_actor / load_checkpoint）后调用。"""
        self._weights_dirty = True

    def release(self):
        """把保持唤醒的 vLLM 送回 sleep，给接下来的训练侧 GPU 计算腾显存。"""
        if not self._awake:
            return
        self.inference_engine.sleep(level=1)
        self._awake = False
        get_torch_device().empty_cache()

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __enter__(self):
        if self.keep_awake and vllm_version not in ("0.5.4", "0.6.3"):
            if self._awake and not self._weights_dirty:
                # 上一次 generate 之后没 sleep、权重也没变：什么都不用做
                if self.device_mesh is not None:
                    self.torch_random_states = get_torch_device().get_rng_state()
                    get_torch_device().set_rng_state(self.gen_random_states)
                return
            if (not self._awake and not self._weights_dirty and self.base_sync_done
                    and not isinstance(self.module._fsdp_wrapped_module, PeftModel)):
                # 只是被 release() 送去睡了、权重没变：直接唤醒，跳过全量同步
                get_torch_device().empty_cache()
                self.inference_engine.wake_up()
                self._awake = True
                if self.device_mesh is not None:
                    self.torch_random_states = get_torch_device().get_rng_state()
                    get_torch_device().set_rng_state(self.gen_random_states)
                return
            if self._awake:
                # 权重变了但 vLLM 还醒着（理论上 worker 会先 release，这里兜底）
                self.release()
        self._enter_full_sync()
        self._weights_dirty = False
        self._awake = True
        self._n_full_sync += 1
        if self.keep_awake and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"[fsdp_vllm] 全量权重同步 #{self._n_full_sync}", flush=True)

    def _enter_full_sync(self):
        def __collect_lora_params()->OrderedDict:
            """
            collect lora params or full params if base model is not ready in vllm
            work with if isinstance(self.module._fsdp_wrapped_module, PeftModel)
            """
            from peft.utils.save_and_load import get_peft_model_state_dict

            lora_params = OrderedDict()
            if fsdp_version(self.module) > 0:
                if self.layered_summon:
                    if not self.base_sync_done:
                        raise ValueError("To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let rollout.load_format=safetensors")
                    lora_params = layered_summon_lora_params(self.module)
                else:
                    with FSDP.summon_full_params(self.module, writeback=False):
                        if self.base_sync_done:
                            lora_params = get_peft_model_state_dict(self.module._fsdp_wrapped_module)
                            lora_params = {name: param.full_tensor().detach().cpu() if hasattr(param, 'full_tensor') else param.detach().cpu() 
                                        for name, param in lora_params.items()}
                        else:
                            model = self.module._fsdp_wrapped_module.base_model.model
                            orig_dev = 'cpu' if 'cpu' in next(model.parameters()).device.type else 'cuda'
                            model = model.to('cpu')
                            for name, param in model.state_dict().items():
                                if any(x in name for x in ['_flat_param', 'lora_']):
                                    continue
                                name = name.replace("_fsdp_wrapped_module.","").replace(".base_layer","")
                                lora_params[name] = param.full_tensor().detach().cpu() if hasattr(param, 'full_tensor') else param.detach().cpu()
                            model = model.to(orig_dev)
                    torch.cuda.empty_cache()
            else:
                if self.base_sync_done:
                    lora_params = get_peft_model_state_dict(self.module._fsdp_wrapped_module)
                else:
                    model = self.module._fsdp_wrapped_module.base_model.model
                    orig_dev = 'cpu' if 'cpu' in next(model.parameters()).device.type else 'cuda'
                    model = model.to('cpu')
                    for name, param in model.state_dict().items():
                        if any(x in name for x in ['_flat_param', 'lora_']):
                            continue
                        name = name.replace("_fsdp_wrapped_module.","").replace(".base_layer","")
                        lora_params[name] = param.detach().cpu()
                    model = model.to(orig_dev)
            return lora_params

        # NOTE: Basically, we only need `get_torch_device().empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        get_torch_device().empty_cache()

        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)

        peft_config = None
        if isinstance(self.module._fsdp_wrapped_module, PeftModel):
            peft_config = self.module._fsdp_wrapped_module.peft_config.get('default', None)
            params = __collect_lora_params()
        else:
            params = self.module.state_dict()
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)

        # Copy, not share memory
        load_format = "hf" if self.full_params else "dtensor"

        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self.inference_engine.sync_model_weights(params, load_format=load_format)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
        else:
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["weights"])
            else:
                self.inference_engine.wake_up()

            # update model params
            self.update_params(params, peft_config=peft_config)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
            if self.offload_param:
                offload_fsdp_model_to_cpu(self.module)
            get_torch_device().empty_cache()

            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["kv_cache"])

        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        # TODO(ZSL): check this
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self.inference_engine.offload_model_weights()
        elif self.keep_awake and exc_type is None:
            pass  # 保持唤醒，留给下一次 generate；worker 在别的 GPU 计算前调 release()
        else:
            self.inference_engine.sleep(level=1)
            self._awake = False

        self.module.train()

        # add empty cache after each compute
        get_torch_device().empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            group = vllm_ps.get_tensor_model_parallel_group()
        else:
            group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]

    def update_params(self, updated_params, peft_config=None):
        model = self.model_runner.model
        if peft_config:
            if self.base_sync_done:
                lora_int_id=int(time.time_ns() % 0x7FFFFFFF)
                lora_reqest = TensorLoRARequest(
                    lora_name=f"{lora_int_id}",
                    lora_int_id=lora_int_id,
                    lora_path="simon_lora_path",
                    peft_config=asdict(peft_config),
                    lora_tensors=updated_params,
                )
                self.inference_engine.llm_engine.add_lora(lora_reqest)
                logger.info(f"vLLM load weights, loaded_params: {len(updated_params)}")
                return
            else:
                def replace_lora_wrapper(k):
                    stacked_params = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
                    if any([k.endswith(f"{s}.weight") for s in stacked_params]):
                        return k.replace(".weight", ".base_layer.weight")
                    if any([k.endswith(f"{s}.bias") for s in stacked_params]):
                        return k.replace(".bias", ".base_layer.bias")
                    return k
                updated_params = {replace_lora_wrapper(k): v for k, v in updated_params.items()}

        patch_vllm_moe_model_weight_loader(model)
        device = get_torch_device().current_device()  # used when fsdp2 set cpu_offload_policy
        loaded_params = model.load_weights(((name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param) for name, param in updated_params.items()))

        self.base_sync_done = True
        logger.info(f"vLLM load weights, loaded_params: {len(loaded_params) if loaded_params else -1}")
