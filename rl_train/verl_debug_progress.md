# verl 迁移调试进展汇总（截至 10-02）

> 本文档是"把 verl 调通"这条主线的执行记录，按时间顺序梳理每一层验收状态、
> 踩过的坑和修复方式。更偏顶层的迁移设计/规划见 `verl_migration.md`；本文档
> 侧重"实际调试过程中发生了什么、为什么坏、怎么修的"，供后续排错和写作复盘用。

## 总览：五层验收状态

| 层 | 内容 | 状态 |
|---|---|---|
| **1. 适配器** | verl-agent + qwen3_5 模型后端，真实 9B 权重前向 | ✅ 通过（`verl-c4`，09-27） |
| **2. 环境接入** | Minecraft/Malmo 环境包 + `MinecraftEnvironmentManager` | ✅ 通过（`verl-env-d5`，09-28） |
| **2.5 训练/推理共存可行性** | HF 训练 + vLLM 推理能否同进程、权重能否同步 | ✅ 可行（`colocate-p2`，09-28） |
| **3. GRPO 冒烟** | `verl.trainer.main_ppo` 端到端跑通 rollout→advantage→update，连续 ≥2 步 | ✅ **通过**（`grpo-smoke16`，10-01，4×GPU，2/2 步；共 16 轮、修 12 个根因） |
| **4. 正式训练（h29 多图历史 + 真实任务池）** | 解决单图 vs h29 结构性差距，8 卡跑真实 100 步 GRPO | ⏳ **两个并行实例在跑**：`grpo_h29_mixed88_train2`（无 wandb，10-02 起已稳定跑完 7/100 步）+ `grpo_h29_mixed88_wandb`（同配置 + wandb，10-02 新开，见下） |
| **5. 吞吐对比自研线** | 吞吐对比自研 HTTP eval-harness，决定是否切换主线 | ❌ 未开始 |

---

## 第 1 层：适配器验收（09-27，`verl-c4`，已通过）

验证 verl-agent 内嵌的 qwen3_5 适配器能否用真实 stage3/v2 权重跑通 PPO 前向。

```
>>>>>>>>>> A+B（同进程）
[OK] ulysses_sp_size>1 被正确拒绝
[OK] torch backend 前向: log_probs(1, 11)
SMOKE_RESULT: A,B (2/2) ✅ 通过
>>>>>>>>>> C（独立进程，真实9B权重）
model_type=qwen3_5
[OK] 真实权重 PPO 前向（fla 内核 + 9B bf16）: log_probs(1, 90), entropy(1, 90)
SMOKE_RESULT: C (1/1) ✅ 通过
```

C 层此前失败四轮，**全部是冒烟脚本自身的 bug，与 verl 无关**：

| 轮次 | 失败原因 | 修复 |
|---|---|---|
| `c` | `AutoModelForImageTextToText` 加载方式错 | 改用正确的 Auto 类 + `trust_remote_code` |
| `c2` | `isfinite(None)`：`apply_monkey_patch` 是**类级**补丁，check_b 调用后 check_c 的"原生前向"其实已是 PPO 路径（`logits=None`） | 改为校验实际存在的字段（log_probs/entropy/logits 任一） |
| `c3` | **根本没入队**（提交输出被 watch 机制吞掉，误以为在跑） | 提交后必须核实 S3 日志目录存在 |
| `c4` | — | **通过**。`--only` 让 C 层在独立进程跑，从机制上消除类级补丁泄漏；同时改为失败即 `exit(1)` |

关键事实：`forward_with_torch_backend` 在 `labels=None` 时回退到 `torch.roll(input_ids, -1)`，
因此不传 labels 也能算出 log_probs——这正是 RL 训练算 `old_log_probs` 时的真实调用形态。

---

## 第 2 层：Minecraft 环境接入（09-28，`verl-env-d5`，已通过）

### 起点问题：之前"已注册"的 minecraft 分支从未真正跑过一步

`envs.py`（从 CrossAgent 抢救资产迁入，对着另一版 openagents 写的）有 7 个致命问题：

1. **import 断裂**：`render_video_enable`/`InitialActionCallback`/`craft_item` 三个符号在当前
   openagents 里都不存在 → 模块一导入就 `ImportError`
2. **调用契约不符**：旧代码 `self.env, extra_info = env_init(**kwargs)`，传
   `group_n`/`action_type`/`task_name` 等关键字；当前 `env_init` 签名是
   `(task_config, rollout_path, args, ...)` 且只返回 env → 必然 `TypeError`
3. 上面的 `TypeError` 被 `while True: try/except: retry` 包着 → **无限重试挂死**
4. `reset()` 里 `time.sleep(random.randint(0, 180))` → 每次 reset 平均白等 90s
5. `self.tasks = env_kwargs["tasks"]`，但 env_manager 从不传 `tasks` → `KeyError`
6. `save_render_videos` 里有 `breakpoint()` → Ray worker 里会永久挂起
7. 同组（GRPO group）各 worker 各自调 `choose_available_task` → 各自随机种子，
   组内初始条件不同，**组内相对优势失去可比性**

而 `env_manager.py` 里 minecraft 分支直接复用 `WebshopEnvironmentManager`：
`reset()` 对观测做 `obs.split(" [SEP] ")`（我们的观测是图像 ndarray，第一步就
`AttributeError`），且返回 `'image': None`（视觉策略拿不到任何画面）。

`projection.py` 的语义 bug：`TextActionTokenizer.decode()` 在文本里找不到
`Action:` 时不抛异常，而是静默返回空动作，原代码靠 try/except 判定合法性，于是
**任何垃圾输出都被记为 valid=1，RL 永远不会因格式错误受罚**——这很可能与
iter1 自举训练的格式崩溃直接相关。

### 修复

- **重写 `envs.py`**：不经 `env_init`（它强制要求 `rollout_path`+`args.fps`），
  直接复用 `openagents/envs/env.py` 同款 callback 组装 `MinecraftSim`；任务配置
  **组级采样一次**，同组 worker 共享同一个 task_config（同 seed/出生点/初始物品）；
  所有重试有上限；观测取 `info["pov"]`（与 OpenHA 一致，而非 224×224 的低清 `obs["image"]`）
- **重写 `projection.py`**：合法性判定改为要求匹配到真实动作原语
  （`move`/`press`/`click`/`no_op`），比"有 `Action:` 前缀"更严——像
  `"Action: Walk forward across the terrain"`（iter1 崩溃样例）这种有前缀无语法的
  自然语言现在会被正确判 `valid=0`
- **新增 `MinecraftEnvironmentManager`**：图像观测 + 与 SFT/评测逐字对齐的 prompt
  （`system_prompt(text_action) + instruction + <image>`）
- **补 `setup_openha_env.sh`**：openha env 不是镜像预装、是评测脚本运行时创建的；
  第一版漏抄了 `run_backbone_eval.sh` 里的 `cuda-python` 版本钉死和 Malmo 引擎下载
  两段，导致 Ray worker 在 `MinecraftSim.__init__` 里弹交互式 "download engine
  (Y/N)?" 直接 `EOFError`

### 验收结果（`verl-env-d5`）

```
=== D. projection ===
[OK] projection 合法/非法判定全部正确
=== E. Ray + Malmo reset（1 组 × 2 worker） ===
reset 耗时 45.9s
worker0/1: pov shape=(360, 640, 3) dtype=uint8 task=mine_block:oak_log
[OK] reset 通过，同组共享 task_config
=== F. step 闭环 × 3 ===
step0-2: valids=[1, 0] reward=[0.0, 0.0] done=[False, False] (0.02-0.03s)
[OK] step 契约成立（非法动作按 noop 执行、不崩溃）
=== G. MinecraftEnvironmentManager（verl 训练入口） ===
prompt 长度=1891 结尾='...Break the tree to get oak logs.<image>'
step: reward=[0.] done=[False] is_action_valid=1
SMOKE2_RESULT: D,E,F,G (4/4) ✅ 通过
```

⚠️ **已知限制**：`MinecraftEnvironmentManager` 每样本只放 1 张图（受限于
`rollout_loop.preprocess_single_sample` 的单图设计），而评测是 h29 多图历史。
当前接起来的版本等于"每一步都当作轨迹第一帧"——对 SFT 是分布内、RL 能跑通，
但策略训练时看不到历史，训练条件与评测条件不一致，**这是后续必须解决的
结构性差距，不是小改动**。

---

## 第 2.5 层：训练/推理共存可行性（09-28，`colocate-p2`，已确认可行）

### 背景：版本矩阵冲突

`trainenv-p1` 探针发现：

- `vllm35` env（评测已在用）：torch 2.10 / transformers 4.57.6 / vllm 0.17 ——
  **transformers 里没有 `Qwen3_5ForConditionalGeneration`**，FSDP 训练侧加载不了模型
- `sft` env（训练已在用）：torch 2.6 / transformers 5.15 —— 有 Qwen3_5，但
  vLLM 0.17 需要 torch 2.10
- vendored verl 的 vLLM glue 代码是按 vLLM 0.8-0.11 写的：`vllm_async_server`
  （用到 `vllm.entrypoints.openai.protocol`）和 `fsdp_vllm`（用到
  `vllm.lora.models`）在 0.17 下 import 失败

`hf_rollout.py`（HF 原生 generate 路径）只传 `input_ids`，不传
`pixel_values`/`image_grid_thw`——视觉策略会**瞎生成**，所以 vLLM rollout 是
必需项，不能绕开版本冲突改用 HF 生成。

### 探针结果（V1-V5 全部通过）

用 `torch 2.10 + transformers 5.15 + vllm 0.17` 的组合验证：

```
V1 版本: torch 2.10.0+cu128 | transformers 5.15.0 | vllm 0.17.0，三者可共存
V2 vLLM 进程内加载 9B 权重 + 带图生成: greedy 'Action: move(0, 0) and press() and click()'
V3 进程内 model 对象属性链: 命中 llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
V4 HF 侧前向（fla 内核）: HF logits (1, 5, 248320)，fla + causal_conv1d present
V5 HF→vLLM 权重同步回路: state_dict 760 张量 → load_weights 加载 664 → 同步后 greedy 输出与同步前逐字一致
```

由此确定第 3 层训练环境的技术路线：以 `vllm35` 为基座克隆出 `vllmtrain` env，
装 transformers 5.15 补 Qwen3_5 支持。

---

## 第 3 层：GRPO 冒烟（09-28~10-01，`grpo-smoke16` 通过）

### 环境组装：`setup_vllmtrain_env.sh`

一个 env 要同时满足训练侧（原 sft env 的一半）和环境侧（原 openha env）两件事。
核心教训：**openagents 声明 `vllm==0.8.5`/`transformers==4.54.0`/`numpy==1.26.4`，
minestudio 也有自己一整套依赖——任何一个不带 `--no-deps` 的安装都可能静默把
torch/torchvision/numpy 换掉**，冲垮 colocate-p2 刚验证过的组合（实测：装
`minestudio` 时漏了 `--no-deps`，torch 从 2.10.0 被换成 2.8.0 且带 ABI 不匹配的
torchaudio，import 直接崩）。

因此该脚本的设计原则：**所有第三方安装一律 `--no-deps`**，缺什么用
`_install_missing_no_deps` 自动探测报错类型（`ModuleNotFoundError` /
`ImportError: cannot import name` / 版本门禁）逐个补最小依赖，绝不让 pip
resolver 联带升降级 torch 系；每个大步骤后打印 torch 版本并断言未变化，
一旦漂移立即 `FATAL` 退出。

分层验证顺序（① Qwen3.5 支持 → ② fla/causal-conv1d → ③ verl 训练依赖 →
④ openagents+minestudio → ⑤ 真实起一次 Malmo(捕获仅运行时触发的依赖，如
Pyro4/xmltodict) → ⑥ 逐个 import 复核），是在多轮实测中被迫加出来的：
一些依赖（如 `Pyro4`）只在真正启动 JVM 时才 import，任何静态探针都探不到，
只能靠真实跑一次 Malmo reset 才能暴露。

### GRPO 冒烟脚本：`run_grpo_minecraft_smoke.sh`，六轮迭代

| 轮次 | 现象 | 根因 | 修复 |
|---|---|---|---|
| `smoke1` | 从未入队 | 代码同步（`aws s3 sync`）与任务提交并行执行，任务启动时代码还没传完 | 改为先 sync 确认完成，再提交 |
| `smoke2` | `ImportError: cannot import name AutoModelForVision2Seq`（此前）；`ModuleNotFoundError: No module named 'pyarrow'`/`'multiprocess'` | `verl.trainer.main_ppo` 真正的训练入口会往下拉一条完全不同的深链（`ray_trainer.py` → `multi_turn_rollout` → `rl_dataset` → `datasets`），此前的探针只测了 `import verl`，测不到这条链 | setup 脚本里补装 `pyarrow`/`multiprocess`，并显式加 `_install_missing_no_deps "import verl.trainer.main_ppo"` 探针 |
| `smoke3` | `Could not override 'data.apply_chat_template_kwargs'`（Hydra struct 拒绝新键） | Hydra struct 模式下，即使目标是空字典 `{}`，字面量整体替换也按"合并"语义逐键校验，`enable_thinking` 不在原 schema 里就直接拒绝 | 改用 `+data.apply_chat_template_kwargs.enable_thinking=false`（Hydra 报错里建议的写法，显式声明"新增键"） |
| `smoke3`（续） | `AssertionError: real_train_batch_size (1) must be divisible by total n_gpus (2)` | `TRAIN_BATCH`（任务组数）默认写死 1，与 `N_GPUS=2` 不整除 | `TRAIN_BATCH` 默认改为跟随 `N_GPUS` |
| `smoke4` | `ModuleNotFoundError: No module named 'flash_attn'`（发生在 `ref_init_model`） | `verl/workers/actor/dp_actor.py` 模块**顶层**无条件 `from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input`——这与 `attn_implementation` 无关（纯 pad/unpad 工具函数），且即使 `use_remove_padding=False`（唯一会调用这几个符号的分支被跳过）也照样在 import 阶段炸 | 加 try/except，无 flash_attn 时用纯 PyTorch 等价实现（已用本地单测验证 unpad→pad 往返一致、`cu_seqlens` 正确、`index_first_axis` 语义正确） |
| `smoke5` | `ImportError: cannot import name 'AutoModelForVision2Seq' from 'transformers'` | transformers 5.x 把 `AutoModelForVision2Seq` 重命名/合并进了 `AutoModelForImageTextToText`（`fsdp_workers.py` 和 `fsdp_checkpoint_manager.py` 两处硬编码旧名） | 两处都加 try/except 别名兜底 |
| `smoke6` | **首次跑过全部 setup**（fla/minestudio/真实 Malmo 冒烟全绿），到达 vLLM engine 初始化才报错：`pydantic ValidationError: Chunked prefill is required for mamba cache mode 'align'` | Qwen3.5 的 fla 混合注意力（mamba/gated-delta-net cache）在当前 vLLM 版本下要求 chunked prefill 开启，脚本里写死了 `enable_chunked_prefill=False` | 改为 `True` |
| `smoke7` | `setup_vllmtrain_env.sh` 装完 openjdk 立即退出：`JAVA_HOME_CONDA_BACKUP: unbound variable` | openjdk 的 conda 钩子（`deactivate.d/openjdk_deactivate.sh`）不兼容 `set -u`；这个坑此前只在 `setup_openha_env.sh` 里修过，新脚本漏搬同一处防护（间歇性——同一脚本在 `smoke6` 那次没触发） | 装 openjdk 前后临时关/开 `set -u`（第一版局部修法） |
| `smoke8` | **setup 完整跑完**（`SETUP_VLLMTRAIN_DONE` 打印），但 job 脚本自己**多余的第二次** `conda activate vllmtrain` 又炸：`activate.d/openjdk_activate.sh: target_platform: unbound variable` | 同一类坑的第二次复现：`smoke7` 的局部修法只在"安装那一刻"关 `set -u`，装完立刻恢复，治不了脚本 source 结束后、调用方自己再 activate 一次的场景 | 精简 job 脚本去掉重复 activate；同时把 setup 脚本的修法从"局部关闭"改为"检查 java 之前关闭、之后不再恢复"，一次性兜住所有后续 activate/deactivate |
| `smoke9` | 提交失败——CLI 升级到 2.4.0 后 `--s3-log` 参数被移除 | 用法过时，不是代码问题 | 改用 `smoke10`（去掉 `--s3-log`） |
| `smoke10` | 排队 **8 小时+**（集群资源被同 namespace 其他任务占满，配额本身充足）后终于跑起来，**第一次完整通过 setup 和所有此前 7 个坑，真正进入 GRPO 训练循环（Ray dataloader 开始跑）**，撞上第 8 个坑：`AttributeError: '_io.BytesIO' object has no attribute 'startswith'` | verl 自带 `vision_utils.process_image()` 把 `{"bytes": png}` 转成 `image["image"]=BytesIO(...)` 再调 `qwen_vl_utils.fetch_image()`——但 `fetch_image` 只认 `PIL.Image` 或 `str`（http(s):// / file:// / data:image base64 / 本地路径），**完全不支持 BytesIO**，这是 verl-agent 自带代码与当前 `qwen_vl_utils` 版本的不匹配，不是我们的适配层问题 | `prepare_minecraft_data.py` 改为把占位图写成真实 PNG 文件，`images` 列用 `{"image": "file://<path>"}` 代替 `{"bytes": ...}`，绕开这条有 bug 的分支（`file://` 是 `fetch_image` 明确支持的格式，本地已用 pandas/pyarrow round-trip 验证格式不变） |
| `smoke11` | 提交时撞上 `koala update` 强制版本门禁（见下），绕开后**跨过了 dataloader**（`file://` 修复生效），真正推进到 `MinecraftWorker.reset()`（rollout 的真实入口，第一次创建 Malmo 环境），撞上第 9 个坑：`ModuleNotFoundError: No module named 'xmltodict'` | `setup_vllmtrain_env.sh` 第一步 `apt-get install xvfb` 没有先 `apt-get update`，在干净容器里静默失败（输出被 `>/dev/null 2>&1` 吞掉）；缺 xvfb 导致 step⑤ 真实起 Malmo 的探针因 `xvfb-run: command not found` 报错，而这类报错不被 `_install_missing_no_deps` 的三种已知模式（ModuleNotFoundError/ImportError/版本门禁）识别，判定为"无法识别的错误"后放弃——**整个 Malmo 运行时依赖捕获环节被静默跳过**，xmltodict 这种只在真正起 JVM 时才 import 的依赖一路漏到 GRPO 训练才暴露 | 两处修复：① `apt-get update` 前置 + 真实错误输出（不再吞掉），失败有明确 WARN；② 不再把已知运行时依赖的捕获全部押在 step⑤ 的动态探针上，直接前置装好 `xmltodict`/`lxml`/`psutil` 等清单；③ step⑤ 本身在缺 xvfb 时退化为裸 `python`（仍能捕获 import 期错误）而不是静默跳过 |
| `smoke12` | **rollout（4 个真实 Malmo 环境）→ reward → advantage → log_prob → backward 全部走通**，在 actor `optimizer.step()`（Adam 首次 `_init_group`）CUDA OOM | 9B 全参数 Adam 的 step 峰值 ≈ fp32 主参数 36G + 梯度 36G + 两份动量 72G ≈ 144G，单张 140G 卡物理放不下，vLLM 还占一份 | 尝试 CPU offload（见下两行） |
| `smoke13` | vLLM 初始化即 `AssertionError: Expandable segments are not compatible with memory pool` | **我引入的错误**：为缓解碎片加了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，与 vLLM sleep-mode 的 `CuMemAllocator`（`free_cache_engine=True` 依赖它）不兼容（pytorch#147851） | 删掉该环境变量并注释原因 |
| `smoke14` | Hydra 覆盖参数里确认 `param_offload=True`/`optimizer_offload=True` 已生效，**仍在 Adam `_init_group` OOM** | 读 `fsdp_workers.update_actor` 才明白：verl 的 offload 不是 CPU 计算，`update_actor` 开头会 `load_fsdp_model_to_gpu` + `load_fsdp_optimizer` 把参数和优化器状态全搬回 GPU 再 step，只降低空闲/rollout 阶段占用，**降不了 step 峰值**。单卡无论如何都跑不通 | 默认改为 **4 卡 FSDP 分片**（每卡约 36G）；`GROUP_SIZE` 默认 4→2（GRPO 最小值，减少 Malmo 实例数）；<4 卡时打印明确 WARN |
| `smoke15` | ✅ **第一个 GRPO update step 完整跑通**（见下方指标），之后进程**正常退出**，Progress 停在 1/2，无任何报错 | 占位 parquet 只有 `TRAIN_BATCH` 行 = 恰好 1 个 batch，`total_epochs=1` 时 dataloader 一轮就耗尽；`total_training_steps=2` 只截断步数、不补数据 | `total_epochs` 跟随 `TOTAL_STEPS`（每 epoch 1 步；真实 prompt/图像来自环境，重复占位行无副作用） |
| `smoke16` | ✅ **`Training Progress: 2/2`，冒烟验收通过**。两个完整 GRPO update step 连续跑完，第 2 步 rollout 用的是第 1 步更新后的权重（FSDP→vLLM 权重同步在真实训练循环里走通）。结尾的 `DataLoader worker ... killed by signal` 出现在 `Final validation metrics` 和 `2/2` **之后**，是 Ray 关闭时杀掉 dataloader 预取进程产生的收尾噪声，不影响结果 | — | — |

**`smoke15` 第一个完整 GRPO step 的指标**（4×GPU，train_batch=4 组 × group_n=2，`max_steps=16`）：

| 指标 | 值 | 解读 |
|---|---|---|
| `actor/pg_loss` | −0.055 | 有限值，梯度在流动 |
| `actor/kl_loss` | 0.003 | 第一步 actor≈ref，符合预期 |
| `actor/grad_norm` | 4.06 | 正常量级，无爆炸 |
| `actor/entropy_loss` | 0.462 | |
| `episode/valid_action_ratio` | **99.2%** | 新 projection 的合法性判定在真实 rollout 中工作正常 |
| `episode/success_rate` | 0.0 | 16 步 episode 砍不完树，符合预期（冒烟只验链路） |
| `critic/score/mean` | −0.001 | 只有少量无效动作惩罚 |
| `perf/max_memory_allocated_gb` | 100.9 | 4 卡分片后每卡峰值，140G 卡有余量 |
| `timing_s/gen` | 215s | **rollout 占单步 66%**——吞吐优化的主战场 |
| `timing_s/update_actor` | 65s | |
| `timing_s/step` | 327s | |
| `perf/throughput` | 70.8 token/s/GPU | 吞吐对比的第一个基线数 |

**`smoke16` 两步对照**（同配置）：

| 指标 | step 1 | step 2 | 解读 |
|---|---|---|---|
| `actor/pg_loss` | −0.038 | 0.000 | step 2 组内 reward 全相同 → GRPO 优势全 0 |
| `actor/grad_norm` | 1.709 | 0.061 | 同上，几乎只剩 KL 项梯度 |
| `actor/kl_loss` | 0.001 | 0.001 | |
| `episode/valid_action_ratio` | 99.2% | 100% | |
| `episode/success_rate` | 0 | 0 | |
| `perf/max_memory_allocated_gb` | 100.9 | 100.9 | 显存不随步数增长，无泄漏 |
| `timing_s/gen` | 219s | **107s** | step 1 含 Malmo 首次 reset（~110s）；step 2 才是稳态 rollout 耗时 |
| `timing_s/update_actor` | 75s | 67s | |
| `timing_s/step` | 334s | **200s** | |
| `perf/throughput` | 69.6 | **115.2** token/s/GPU | 稳态吞吐基线 |

⚠️ step 2 的 `pg_loss=0` 不是 bug，而是 **reward 太稀疏**：16 步砍不完树，所有轨迹
reward 都≈0，GRPO 组内没有方差、学习信号为零。冒烟只验链路，这可以接受；但真正训练
前必须解决（加长 episode / 换更容易的任务 / 加 shaping），否则 RL 等于空转。

前 10 轮修复均已提交并同步到 S3/GitHub；每轮都是"改代码 → 语法检查 → 提交 →
拉日志定位下一个问题"的循环，累计暴露 9 个独立根因（6 个环境/框架兼容性问题 +
openjdk 的 `set -u` 坑复现两次 + verl 自带图像加载代码的版本不匹配 bug +
xvfb 静默安装失败连带 Malmo 运行时依赖捕获被跳过）。
**`smoke10`/`smoke11` 是连续两个里程碑**：`smoke10` 第一次跑进训练循环
（Ray dataloader 开始工作），`smoke11` 又往前跨了一大步——第一次真正走到
`MinecraftWorker.reset()`、真实创建 Malmo 环境这一步。问题域已经从
"环境/配置能不能起来"彻底转移到"训练逻辑内部的数据/环境细节"——这是全新的、
更接近终点的问题层级。`smoke7`/`smoke8` 是同一个 openjdk 坑的两次不同触发
路径，已改为从根上（不恢复 `set -u`）解决，理论上不会再犯第三次。

### 静态审查：提前排查 dataloader 之后的未执行代码（09-30）

每轮真实冒烟要等 ~8 小时集群排队，逐个试错不可接受，因此对
`smoke10` 崩点之后的整条链路（rollout loop → env manager → projection →
reward → advantage → update）做了系统性静态审查。结果如下。

**① 确凿 bug，已修**：`env_manager.py` 的 minecraft 分支是**唯一没有透传
`resources_per_worker` 的 env 分支**（sokoban/alfworld/gymcards/webshop 都传了），
导致启动脚本里的 `env.resources_per_worker.num_cpus` 是死参数，
`MinecraftWorker` 一直用写死的 `@ray.remote(num_cpus=2)`。CPU 预留随
`train_batch_size × group_n + 验证集` 线性增长，很容易超过 koala 的每卡 CPU
配额；**超了之后 Ray actor 不报错、只是永久 PENDING**，最终在 `reset()` 的
`ray.get(..., timeout=900)` 抛 `GetTimeoutError`——现象（卡 15 分钟后超时）
和根因（CPU 预留超配额）完全不像，极难定位。已改为仿 sokoban 的
`ray.remote(**resources_per_worker)(MinecraftWorker)` 动态包装，默认 0.5 CPU
（真实计算在独立的 Malmo JVM 进程里，Ray actor 只做 RPC 转发）。

**② 两个"必崩"结论经核实是误报**（值得记下来，避免以后重复怀疑）：
审查认为 Qwen3.5 的 processor 类名会是 `Qwen3_5*`，从而
`rl_dataset.py:235` 的门控 `"Qwen2VLImageProcessor" in image_processor 类名`
会失配、使 dataloader 与 rollout 产出不同形状的 `position_ids`。但实际读
`processor_config.json`：

```
image_processor_type = "Qwen2VLImageProcessor"   ← dataloader 门控命中
processor_class      = "Qwen3VLProcessor"        ← 选 qwen3_vl.get_rope_index
merge_size           = 2
```

且 `config.json` 的 `text_config.rope_parameters` 为
`{mrope_section: [11,11,10], mrope_interleaved: true}`——**Qwen3.5 确实用
mrope**，所以 4 行 `position_ids`（1 文本 + 3 视觉）是**正确约定而非 bug**，
两处门控实际都落在同一分支。

**③ 剩余的真正未知（静态读不出来，只能实跑）**：`qwen2_vl` 适配器在前向里调
`process_position_ids()` 校验/裁剪 `(4,bs,seq)`，而 `qwen3_5_base_forward`
**完全没有这一步**、直接 `**kwargs` 透传给 `language_model`。4 行 mrope 能否
被 transformers 5.15 的 Qwen3.5 接受，取决于其内部实现，读代码无法判定。

**对策——`smoke_step3.py`**：把所有剩余未知压进**一个** 1-GPU 任务，每层
独立 try/except（一层失败不遮挡其余），一次排队拿到全部答案：

| 层 | 验什么 |
|---|---|
| H | processor / image_processor 真实类名 → 两处 mrope 门控各走哪个分支 |
| I | rollout 侧硬编码的 `processor.image_token`、`merge_size`、`image_grid_thw` 键、以及 `get_rope_index` 需要的 `image/video/vision_start_token_id` 是否都存在 |
| J | 用真实 processor + 真实图实跑 `get_rope_index`，确认输出行数 |
| **K** | **★ 4 行 mrope 能否走通 Qwen3_5 的 PPO 前向**（复刻 `dp_actor` transpose 后的真实形态，最关键） |
| L | 真实 prompt 的 token 数 vs `max_prompt_length=1024`（`truncation='error'` 下超限即每步必崩；`text_action.txt` 1.8KB + 640×360 POV 的 vision token 估算已逼近上限） |

**④ 已核对确认无问题的项**（不必再查）：reward 接线（`EpisodeRewardManager`
无条件启用，不需注册 custom reward；`use_invalid_action_penalty` 依赖的
`is_action_valid` 键名两侧一致）、`data_source="visual"` 不参与任何分发分支
（只用于日志计数）、GRPO 的 batch 整除断言（相关断言已被上游注释掉，且
`adjust_batch` 会按 lcm 补齐）、`multi_modal_data` 的 vLLM 约定、
`success_evaluator` 读的 `info['won']`、`compute_data_metrics` 所需的全部键。

**⑤ 已知但暂不处理**：占位 parquet 的 `prompt`/`images` 在 rollout 里会被
完全丢弃（`rollout_loop.py:78-87` 把 `obs_content` 重置为环境产出），所以
`prepare_minecraft_data.py` 里那张占位图的**内容**无所谓——它只影响
dataloader 阶段算出、随后被覆盖的 `input_ids`。记下来避免以后误以为要对齐。

---

## 通用教训（贯穿全程，写入 `verl_migration.md` 的坑清单）

1. **提交 ≠ 入队**：koala 提交输出偶尔被吞、`koala ls` 偶尔返回空——判定标准
   一律是 S3 日志目录 `axiomjin/.koala-logs/<job>-*/` 是否存在
2. **不带 `--no-deps` 的安装极其危险**：pip resolver 会静默联带换掉 torch/
   transformers/numpy，冲垮之前验证过的版本组合
3. **静态 import 探针有盲区**：类级 monkey patch 的污染、只在运行时触发的
   深层依赖（Pyro4）、只在真正调用某个函数分支时才触发的顶层 import
   （flash_attn.bert_padding），都测不出来，只能靠真实跑一次完整链路暴露
4. **Hydra struct 模式的"新增键 vs 覆盖键"语义不同**，报错信息本身给出了
   正确写法（`+` 前缀），照做即可
5. **版本演进导致的 API 改名**（`AutoModelForVision2Seq` →
   `AutoModelForImageTextToText`）需要用 try/except 做双向兼容，而不是
   非此即彼地升级或钉住版本
6. **`source` 脚本改的 shell 选项会持续影响调用方后续代码**：局部关闭/
   重新开启某个选项（如 `set -u`）只能兜住"这一刻"，若调用方后面还有
   代码路径会触发同一类不兼容操作（如再调一次 `conda activate`），必须
   在 sourced 脚本里"从此不再恢复"才能彻底兜住，不能只治第一次触发点

## 调试过程记录（09-30~10-01：集群排队、CLI 故障与绕行）

`smoke10` 排队超过 8 小时（#21→#19→#15→#12，确认是集群资源被同 namespace
其他任务占满，非自己的配额/代码问题）后终于开始执行，**第一次完整跑过全部
setup 和此前 7 个坑，真正进入了 GRPO 训练循环**（Ray dataloader 开始工作），
在读取占位图片数据时撞上第 8 个坑（verl 自带 `vision_utils.py` 与
`qwen_vl_utils` 版本不匹配，见上表），已修复并提交 `smoke11`。

`smoke11` 又往前跨了一大步：**跨过了 dataloader，真正推进到
`MinecraftWorker.reset()`——第一次实际创建 Malmo 环境**，撞上第 9 个坑
（`xmltodict` 缺失，根因是 xvfb 静默安装失败连带整个 Malmo 运行时依赖捕获
环节被跳过，见上表）。已修复并提交 `smoke12`，**这次直接 `Running`，
集群排队压力已解除，不再需要等待数小时**。

**排队期间探索过的两条绕行方案，均已确认不可行**（记录供以后参考，不必
重复尝试）：

1. **迁移到 H20 集群**（完全空闲）——跨云拉 AWS ECR 镜像卡在
   `ContainerCreating` 超 5 分钟，代码/权重都在 AWS S3，暂不具备迁移条件
2. **debug 模式 + `koala exec`/`koala ssh`**——pod 本身健康，但 cvm→pod
   隧道建立失败（基础设施问题，非本地配置可修），交互式调试路径当前不可用

**期间还处理了三次 koala CLI 相关的意外阻塞**：

1. **2.3.6→2.4.0**：默认集群改为 H20（AWS 集群需显式加 `--cluster aws`）；
   **`--s3-log` 参数被移除**，此前全程依赖的 `.koala-logs` S3 拉日志方式
   失效，改用 `koala logs <job> --cluster aws --all`
2. **2.4.0→2.4.1**：平台设了版本门禁，`koala update` 因缺
   `OSS_ACCESS_KEY_ID/SECRET` 失败——绕开方式：直接从
   `s3://arcwm-code-us-west-2/tools/koala/koala-2.4.1-darwin-arm64.tar.gz`
   下载（sha256 校验通过），解压后手动替换 `~/.local/bin/koala` 的软链接
   目标，不依赖 OSS 凭据
3. **本地 `koala` 命令一度全面报错**`❌ 初始化失败，请联系管理员`——
   **不是缓存损坏，是残留的 git 锁文件**：`~/.cache/koala-cli/koala-config/`
   是本地克隆的一个 git 仓库，`.git/refs/heads/master.lock` 残留导致
   `git fsck` 报 `badRefContent`、完整性校验失败。此前 11 次同类故障都是
   靠整体删除 `~/.cache/koala-cli` 重建解决（见 `koala-cli.bak*` 的大量
   备份目录），这次改为精确诊断：`koala doctor` 先定位到
   `CACHE_INTEGRITY_FAILED`，再用 `git fsck` 找到具体坏在哪个 ref，发现
   只是锁文件残留，`find .git -name "*.lock" -delete` 即解决，**不需要
   删除重建整个缓存**——以后遇到同类报错，先试这个更轻量的修法

H20 集群的路径/存储约定也变了（个人持久目录 `/cpfs/<企业ID>`，代码走东京
OSS）——只有在真正决定迁移 H20 时才需要处理，目前不适用。

## 静态分析成果的实跑验证（`smoke_step3`，10-01 全部通过）

09-30 对 dataloader 之后的整条未执行链路做了系统性静态审查（见上文"静态
审查"小节），修了一个真 bug（`resources_per_worker` 未透传），推翻了两个
误判（mrope 形状不一致），剩余的唯一真正未知（4 行 mrope 能否走通 Qwen3_5
前向）打包进 `smoke_step3.py` 一次性验证。**结果：H/I/J/K/L 五层全部通过**，
最关键的 K 层（`(4,1,231)` 形状的 position_ids 走通 PPO 前向，产出
`log_probs(1,231)`/`entropy(1,231)`）证实 transformers 5.15 的 Qwen3.5
确实能接受 4 行 mrope，此前的担忧已清零。

## 当前状态（10-01）：冒烟验收通过

`grpo-smoke16` 在 4×GPU 上**连续跑完 2 个完整 GRPO update step**：真实 Malmo
rollout → reward → GRPO advantage → old/ref log_prob → backward → optimizer
step → FSDP→vLLM 权重同步 → 第 2 步 rollout。适配器、环境、训练循环、权重
同步四个环节都在真实训练里走通了。

**复现命令**（4 卡，koala CLI ≥2.4.1）：

```bash
koala submit -m normal -j axiomjin-verl-grpo -g 4 --cluster aws --large-ssd -y \
  --code "s3://arcwm-code-us-west-2/axiom/code:/data/work/run_codes" \
  -c "set -euo pipefail; cd /data/work/run_codes/Minecraft-CoT; \
      source rl_train/verl_jobs/setup_vllmtrain_env.sh; \
      export MODEL_PATH=/local-ssd/model_cache N_GPUS=4 TOTAL_STEPS=2; \
      aws s3 cp s3://arcwm-code-us-west-2/axiom/model/minecraft-sft-stage4-cot-pilot-v2/checkpoint-520/ \
        /local-ssd/model_cache/ --recursive --exclude 'global_step*' --only-show-errors; \
      bash rl_train/verl_jobs/run_grpo_minecraft_smoke.sh"
```

## 第 4 层：正式训练——h29 多图历史对齐 + 真实任务池（10-01~10-02，进行中）

### 解决了"下一步"里悬而未决的两项：单图 vs h29、reward 稀疏

**单图 → h29 多图历史**（`MinecraftEnvironmentManager(env.history_length=H>0)`）：
`verl-agent` 原生 rollout 每步只拼一条"一张图"的 user message，`history_length`
参数也只回放**文本**观测，两者都对不上评测的"h29 = 30 张图 + 29 条历史回复"多图
上下文。改为在 `MinecraftEnvironmentManager` 里按 env 维护最近 H 组
`(frame, response)`，产出结构化 context，`rollout_loop` 按与 `openha.gen_response`
逐字相同的方式渲染（首轮 user = system+instruction+image，随后每轮
`(image, response)` 对、文本置空，最后追加当前帧）。已用真实 tokenizer/processor
在 h=0/1/5/29 四档本地验证渲染结果与评测侧的 `[text, image]` 列表形式逐字节
一致。

**reward 稀疏 → 换真实任务池**：不再用冒烟的单任务，改用 SFT 起点 v2-e4 在
easy-h29 评测里"组内有成有败"的 88 个任务（`tasks/mixed_v2e4_easy_h29.txt`）——
只有组内既有成功又有失败的轨迹，GRPO 优势才非零，这是让冒烟期"pg_loss 退化
为 0"不再发生的关键前提（而非单纯拉长 episode）。

### 正式训练脚本：`run_grpo_minecraft_train.sh`

在 `run_grpo_minecraft_smoke.sh` 之上包一层，职责分离：

- `HISTORY_LENGTH=29`、真实任务池、`MAX_STEPS_ENV=200`（=评测步数）、
  `TOTAL_STEPS=100`、`GROUP_SIZE=8 × TRAIN_BATCH=8` = 64 个训练环境 + 16 验证
- 每条轨迹只抽 `TRAIN_STEPS_PER_TRAJ=4` 步参与梯度（h29 每样本 pixel_values
  约 155MB，64×200 步全量进显存不现实）
- 后台每 `UPLOAD_INTERVAL_S=600` 秒把已完整落盘的 HF 格式存档 + `train.log`
  同步到 S3（`upload_ready_ckpts`，靠 `latest_checkpointed_iteration.txt` 判断
  "完整"，避免上传半成品；容器意外终止也不会丢训练产物/日志）

### 当前运行状态（`grpo_h29_mixed88_train2`，8×GPU，10-02）

```
koala submit -m normal -j axiomjin-verl-grpo-h29-train2 -g 8 --cluster aws --large-ssd -y \
  --code "s3://arcwm-code-us-west-2/axiom/code:/data/work/run_codes" \
  -c "set -euo pipefail; cd /data/work/run_codes/Minecraft-CoT; \
      source rl_train/verl_jobs/setup_vllmtrain_env.sh; \
      export MODEL_PATH=/local-ssd/model_cache; \
      aws s3 cp s3://arcwm-code-us-west-2/axiom/model/minecraft-sft-stage4-cot-pilot-v2/checkpoint-520/ \
        /local-ssd/model_cache/ --recursive --exclude 'global_step*' --exclude 'rng_state_*' --exclude 'zero_to_fp32.py' --only-show-errors; \
      export EXPERIMENT_NAME=grpo_h29_mixed88_train2; \
      bash rl_train/verl_jobs/run_grpo_minecraft_train.sh"
```

提交后立即 `Running`（排队问题已随集群负载消退），截至最后检查**已稳定运行
9h+、连续跑完 7/100 步、无中断、无需人工介入**：

| step | pg_loss | grad_norm | kl_loss | success_rate | valid_action_ratio |
|---|---|---|---|---|---|
| 1 | 0.166 | 3.42 | 0.000 | 0.391 | 1.000 |
| 2 | −0.165 | 4.35 | 0.000 | 0.438 | 1.000 |
| 3 | 0.069 | 1.25 | 0.000 | 0.438 | 1.000 |
| 4 | −0.223 | 2.40 | 0.000 | 0.406 | 1.000 |
| 5 | 0.139 | 2.27 | 0.001 | 0.391 | 1.000 |
| 6 | −0.110 | 2.17 | 0.000 | 0.359 | 1.000 |
| 7 | 0.019 | 3.97 | 0.001 | 0.250 | 0.996 |

**这是本次迁移以来第一次看到持续、非零的 GRPO 学习信号**：`pg_loss` 在 7 步内
正负交替且无爆炸，`success_rate` 在 0.25~0.44 区间波动（而非冒烟期的恒 0）——
证实换真实任务池解决了 reward 稀疏问题。`timing_s/gen≈2900s`、
`timing_s/update_actor≈495s`、`timing_s/step≈3600s`（8 卡、h29 多图、200 步/episode、
64×4=256 梯度样本——规模远大于冒烟，不能直接与 `smoke16` 的吞吐基线比较）。

### 深入分析：worker 侧间歇性报错（已有兜底，未中断训练，非新 bug）

训练日志里出现 7 次 `[MinecraftWorker N] sim 创建失败 1/3`，完整链路：

```
TimeoutError: timed out                                    ← comms.recv_message 等 JVM 回包超时
  ↳ TypeError: Text.sample() takes 1 positional argument    ← minestudio 兜底："造一个随机观测"，
    but 2 were given                                          调 Dict.sample(bs)→Text.sample(bs)，
                                                                但当前 gym 版本 Text.sample() 不收参数
    ↳ TypeError: Cannot convert np.dtype into a dtype.        ← 兜底的兜底：改成不传 bs 的 sample()，
                                                                却在 spaces.py:497 写成
                                                                np.array(strings, np.dtype)——把 np.dtype
                                                                这个**类**当 dtype 用，必炸
```

**根因定位**：最外层是真实的资源竞争——64 训练 + 16 验证 = 80 个 Malmo JVM
实例同时起/同时 reset，socket 等 JVM 回包时偶发超时（`minerl.env.comms`）。
**内层两个 `TypeError` 都是 `minestudio`（第三方 vendored 库）自己的异常兜底代码
在新版 gym/numpy 下的真 bug**（`herobraine/hero/spaces.py`），不是我们代码的问题，
也不是本次新发现——`envs.py` 里已有对应注释（"10-01 grpo-small2"），说明这条
坏兜底链此前就踩过，并已针对性加固：

- `reset()`（创建 sim 阶段）：外层 `try/except` 捕获任意异常，重试最多
  `MAX_SIM_CREATE_ATTEMPTS=3` 次，3 次全失败才真正抛 `RuntimeError`
- `step()`（episode 进行中）：同样捕获任意异常，不重试，直接把**本局**按失败
  结束（`reward=0, done=True, info.env_error=True`）、关掉这个模拟器，下一轮
  `reset()` 重建——绝不让单个卡住的 JVM 拖垂整个训练进程

**实际影响核实**：本次运行中，7 次触发全部发生在 `reset()` 的第 1 次尝试，
全部在重试后成功（`sim 创建连续失败` 出现 0 次，即没有 worker 用完 3 次重试）；
`step()` 阶段的兜底分支（`step 失败`）出现 **0 次**。即约 9%（7/80）的 worker
在建组时需要一次重试，无一次真正失败，训练 100% 连续推进到第 7 步，**不是
需要立即处理的问题**。如果后续这个比例明显上升（例如同时起更多组、换更小的
机型），才需要考虑降 `GROUP_SIZE`/`TRAIN_BATCH` 或给 Malmo JVM 更长超时。

### 接入 wandb + 16 卡探索（10-02）

**16 卡（跨 2 节点）暂缓**：AWS 集群每节点最多 8 卡，16 卡必须 `-n 2 -g 8`
跨机。排查发现代码库里**没有任何多机 Ray 配置**——`verl.trainer.main_ppo`
的 `ray.init()` 是本地初始化（`main_ppo.py:36-48`，无 `address="auto"`），
`trainer.nnodes` 在 `ppo_trainer.yaml` 和两个训练脚本里都硬编码 `1`，整个
`rl_train/` 下没有 `ray start --head`/`--address` 的跨机引导脚本。直接提交
`-n 2` 会让两个节点各自独立跑一份完整训练、互不协同，是全新未验证路径、
有端口/资源冲突风险。与用户确认后**暂缓多机，改为 8 卡单节点（已验证配置）
+ 接入 wandb**。

**wandb 接入**：`run_grpo_minecraft_smoke.sh` 改为按 `WANDB_API_KEY` 是否
非空自动决定 `trainer.logger` 是 `['console']` 还是 `['console','wandb']`
（`tracking.py` 的 `WandbLogger` 直接拿 `trainer.project_name`/
`experiment_name` 做 `wandb.init(project=, name=)`，不需要额外变量），不传
key 时旧调用方式完全不受影响。复用 `trl_sft/common.sh` 里现成的 key（同一
wandb 账号/组织）。

新开任务 `grpo_h29_mixed88_wandb`（8×GPU，与 `train2` 完全同配置，仅
`EXPERIMENT_NAME`/`WANDB_API_KEY` 不同），立即 `Running`（配额 24 卡，与
`train2` 的 8 卡并行占用 16 卡，仍有余量），确认日志里 wandb 正常连接：

```
wandb: 🚀 View run at https://wandb.ai/eter1118-peking-university/verl_minecraft/runs/f4lzg2nn
```

现在 `grpo_h29_mixed88_train2`（无 wandb）和 `grpo_h29_mixed88_wandb`
（有 wandb）**两个同配置实例在并行跑**——后者的训练曲线可以直接在上面的
wandb 链接里看。

---

## 下一步（从"能跑"到"能用于实验"）

1. **等 `grpo_h29_mixed88_train2` 跑完 100 步**：持续盯 `success_rate`/`pg_loss`
   走势（是否随训练上升、还是在当前任务池难度下趋于平台）、显存是否随步数增长
   （目前 100.9G 量级档位是冒烟期数字，h29 多图+更长 episode 下需要重新确认）、
   以及 worker 侧 `sim 创建失败`/`step 失败` 的频率是否随时间上升
2. **吞吐对比自研线**：`grpo_h29_mixed88_train2` 的 `timing_s/*` 是在 h29 真实
   规模下的首个数字（见上），需要在相同 episode 长度/任务难度下对比自研
   HTTP eval-harness rollout 吞吐，这是切换主线的硬指标（目标 ≥3×）
3. **训练产物验证闭环**：`global_step_N` 存档已自动上传 S3，下一步应该拿一个
   存档跑一次 `run_backbone_eval.sh`（和评测用同一套 h29 设置），确认 verl 训出
   的策略在评测 harness 里分数正常、没有因为两套 rollout 实现的细节差异掉分
4. ~~单图 vs h29 多图历史~~ ✅ 已解决（见上）
5. ~~reward 稀疏~~ ✅ 已解决（真实任务池替代单任务冒烟）
