# verl 迁移调试进展汇总

> 本文档是"把 verl 调通"这条主线的执行记录，按时间顺序梳理每一层验收状态、
> 踩过的坑和修复方式。更偏顶层的迁移设计/规划见 `verl_migration.md`；本文档
> 侧重"实际调试过程中发生了什么、为什么坏、怎么修的"，供后续排错和写作复盘用。

## 总览：五层验收状态

| 层 | 内容 | 状态 |
|---|---|---|
| **1. 适配器** | verl-agent + qwen3_5 模型后端，真实 9B 权重前向 | ✅ 通过（`verl-c4`） |
| **2. 环境接入** | Minecraft/Malmo 环境包 + `MinecraftEnvironmentManager` | ✅ 通过（`verl-env-d5`） |
| **2.5 训练/推理共存可行性** | HF 训练 + vLLM 推理能否同进程、权重能否同步 | ✅ 可行（`colocate-p2`） |
| **3. GRPO 冒烟** | `verl.trainer.main_ppo` 端到端跑通 rollout→advantage→update，连续 ≥2 步 | ✅ **通过**（`grpo-smoke16`，4×GPU，2/2 步；共 16 轮、修 12 个根因） |
| **4. 正式训练（h29 多图历史 + 真实任务池）** | 解决单图 vs h29 结构性差距，8 卡跑真实 100 步 GRPO | ⏳ 续训 `resume20` 已到 **75/80**（总第 95 步），预计还需约 4 小时。卡死根因已缩小到"单卡 vLLM 前向 kernel 永不结束"（见"看门狗首次触发"），看门狗兜底有效 |
| **4.5 RL 是否真的提升成绩** | 在固定任务集上和 SFT 起点（28.9%）对比 | ✅ **显著提升**：总第 70 步的存档 **45.5%**，两次基线合并为 29.0%，**+16.5pp**；训练外任务也 +14.0pp（见"评测结果"）。t20/t40 还在跑 |
| **5. 吞吐对比自研线** | 吞吐对比自研 HTTP eval-harness，决定是否切换主线 | ❌ 未开始（verl 侧已有数据：每步约 47~53 分钟，rollout 占约 77%） |

## 当前状态

| 任务 | 状态 | 说明 |
|---|---|---|
| `axiomjin-grpo-h29-resume20` | 运行中，**75/80**，约 50 分钟/步，剩约 4h | 自那次看门狗重启之后没再卡死。`global_step_10`~`70` 都已上传 S3 |
| `axiomjin-eval-vg-t20` | 运行中，已跑约 12h | 首轮 `global_step_20`，总第 20 步 |
| `axiomjin-eval-vg-t40` | 运行中，已跑约 12h | 续训 `global_step_20`，总第 40 步 |
| `axiomjin-eval-vg-t70` | ✅ **已完成**（约 12h） | 续训 `global_step_50`，总第 70 步，结果见下 |

- 一次评测（202 任务 × 3 局，单卡）约需 12 小时，t20/t40 应该很快出结果
- `koala logs` 只返回日志最后约 1000 行，没法统计完成了多少局，要等 `summary.json` 上传到 `s3://.../eval_results/verl-*-easy-h29/` 才能看结果

**续训最新训练指标**（step 62~75）：
- 训练集成功率均值 **0.68**。首轮均值 0.51，续训 step 3~24 约 0.45、step 40~61 约 0.66~0.70
- 熵 0.21~0.39，比 step 50 前后更低、更稳定
- `val/success_rate` 全程：0.812 → 0.562 → 0.250 → 0.812 → 0.812 → 0.500 → 0.625 → **0.812**（step 0~70）。只有 16 个环境，看不出趋势，以固定任务集评测为准
- 已兜底的环境异常：`step 失败` 共 24 次，`sim 创建连续失败` 0 次

## 评测结果：RL 显著提升成绩（固定任务集 easy-h29）

**设置**：202 个非 GUI 任务（Embodied 153 + Combat 49）× 3 局，h29 上下文，温度 0.8 / top_p 0.99，每局最多 200 步。

**可比性核查**：
- t70 的任务清单（含每个任务的难度）与基线 `cal-v2e4` 的**逐字节相同**
- 基线之后，评测链路的代码（`run_backbone_eval.sh` / `rollout_openha.py` / `openha.py` / `openagents/envs` / `build_task_list.py` / `aggregate_results.py`）**没有任何改动**
- 对局记录里的实际参数和基线一致：`maximum_history_length=29`、`max_steps_num=200`、`difficulty=easy`、`temperature=0.8`、`top_p=0.99`
- 输出格式正常：抽查 4 个任务，空动作的比例 t70 为 5.7%（99/1747 步），基线为 6.5%（124/1919 步），t70 没有更多格式崩坏

| 模型 | 总体 | Embodied | Combat | 训练池 88 个任务 | 训练外 114 个任务 |
|---|---|---|---|---|---|
| 基线：原测 `cotv2-e4` | 177/606 = 29.2% | 31.2% | 23.1% | 47.7% | 14.9% |
| 基线：校准 `cal-v2e4` | 174/603 = 28.9% | 32.7% | 17.0% | 42.0% | 18.6% |
| **verl 总第 70 步** | **273/600 = 45.5%** | **51.5%** | **26.4%** | **64.5%** | **30.8%** |
| 与两次基线合并相比 | **+16.5pp**（z=+6.9） | | | +19.6pp（z=+5.2） | **+14.0pp**（z=+5.1） |

**解读**：
- **提升远超噪声**：评测噪声约 ±1.7pp，SFT 同配置重训的差异约 5pp，+16.5pp 远在两者之上
- **不只是训练任务变好了**：114 个没参与训练的任务也从 16.7% 涨到 30.8%，说明学到的东西能泛化
- **按任务数统计**（对比校准基线）：训练池 88 个任务中 43 个提升、9 个下降、36 个持平；训练外 114 个任务中 32 个提升、8 个下降、74 个持平。提升的任务数远多于下降的，不是靠少数任务撑起来的
- **训练池对比要用校准基线**：88 个训练任务是按"原测 `cotv2-e4` 里有成有败"挑出来的，用原测当基线会受"回归均值"影响。校准基线是独立重测的，不受这个选择偏差影响，相比之下仍然 +22.5pp
- **成功得更快**：Embodied 成功局的平均帧数从 82.6 降到 69.0。Combat 从 104 升到 121.7，但 Combat 的成功局多了不少，可能是一些原来打不赢的难局现在能打赢了
- **提升最多的任务**：`mine_block:poppy/dirt/diorite/coal_ore/acacia_stairs/acacia_pressure_plate` 都从基线的 0/6 涨到 3/3，而且全都是训练外任务
- **下降最多的任务**：`kill_entity:salmon`（4/6 → 0/3）、`mine_block:torch`（4/6 → 0/3）、`kill_entity:parrot`、`mine_block:gold_block`（3/6 → 0/3）。每个任务只有 3 局，单个任务的波动很大

**注意事项**：
- 只有一个训练种子。SFT 阶段同配置重训能差约 5pp，RL 的种子方差还没测
- t70 只评测了 600 局（基线 603/606）：有几局没生成结果文件，比例很小，不影响结论
- t20 / t40 / t100 出来后才能看成绩随步数的变化曲线，判断是否已经饱和

---

## 第 1 层：适配器验收（`verl-c4`，已通过）

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

## 第 2 层：Minecraft 环境接入（`verl-env-d5`，已通过）

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

## 第 2.5 层：训练/推理共存可行性（`colocate-p2`，已确认可行）

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

## 第 3 层：GRPO 冒烟（`grpo-smoke16` 通过）

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

### 静态审查：提前排查 dataloader 之后的未执行代码

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
7. **长任务必须有"无进展"检测**：koala 的 `Running` 不代表在训练，日志文件被
   周期性上传也不代表有新内容。判断进度要看 `step:N` 行数和 tqdm 时间戳；
   normal 任务进不了容器，卡死时的现场只能靠任务自己抓（看门狗 + py-spy）
8. **上游框架的调用频率假设要核对**：verl 的 sharding manager 按"一个训练步
   generate 一次"设计；verl-agent 的多轮 rollout 改成每个环境步 generate
   一次，开销就放大了 episode 长度倍（200×）。接别人的多轮框架时，要先数清楚
   每个训练步里各个重操作各执行了几次

## 调试过程记录（集群排队、CLI 故障与绕行）

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

## 静态分析成果的实跑验证（`smoke_step3`，全部通过）

此前对 dataloader 之后的整条未执行链路做了系统性静态审查（见上文"静态
审查"小节），修了一个真 bug（`resources_per_worker` 未透传），推翻了两个
误判（mrope 形状不一致），剩余的唯一真正未知（4 行 mrope 能否走通 Qwen3_5
前向）打包进 `smoke_step3.py` 一次性验证。**结果：H/I/J/K/L 五层全部通过**，
最关键的 K 层（`(4,1,231)` 形状的 position_ids 走通 PPO 前向，产出
`log_probs(1,231)`/`entropy(1,231)`）证实 transformers 5.15 的 Qwen3.5
确实能接受 4 行 mrope，此前的担忧已清零。

## 冒烟验收通过

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

## 第 4 层：正式训练——h29 多图历史对齐 + 真实任务池（进行中）

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

### 当前运行状态（`grpo_h29_mixed88_train2`，8×GPU）

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
也不是本次新发现——`envs.py` 里已有对应注释（"grpo-small2"），说明这条
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

### 接入 wandb + 16 卡探索

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

### 卡死排查：两个任务都在 rollout 中途静默挂起

**现象**

| 任务 | 最后完成的 step | 卡住时长 | 共同特征 |
|---|---|---|---|
| `grpo_h29_mixed88_train2` | 9（卡在 step 10） | ~45h | 日志停在下一轮 rollout 的环境 reset 回调之后，再无任何输出 |
| `grpo_h29_mixed88_wandb` | 24（卡在 step 25） | ~19.5h | 同上 |

- 没有任何 Traceback / Timeout / OOM，koala 一直显示 `Running`
- 日志文件仍在被周期性上传，但内容不再增长；某次检查时看到"到了 step 24"，实际已经卡住
- koala 显示的 GPU 利用率是运行期平均值，一直往下掉；按卡住前约 33% 倒推，卡住后实际约 12~14%，接近 1/8，像是只有一张卡还在跑
- 用 tqdm 时间戳核实：`train2` 的 step 9 在启动后 8h55m 完成，此后 23h+ 再没有 step 10

**已排除**

- **环境 RPC 卡住**：`envs.py` 里 reset/step 的 `ray.get` 都带超时（900s/300s），超时会抛 `GetTimeoutError` 让任务崩掉，不会静默卡几十小时；worker 侧的 `sim 创建连续失败` 一次都没出现
- **CPU 内存溢出**：`perf/cpu_memory_used_gb` 一直稳定在 770~800G（上限 1760Gi），没有增长
- **NCCL 死锁 watchdog 应该能兜住**：按 PyTorch 默认设置约 10 分钟会被终止。这一条只是按默认值推断，没有在现场确认

**根因：未能直接确认**。`koala exec` 只支持 debug 任务，进不了 normal 任务的容器抓调用栈。但读代码找到了一个结构性问题，是头号嫌疑点：

- verl-agent 的多轮 rollout（`rollout_loop.py:482-515`）**每走一步环境**就调一次 `actor_rollout_wg.generate_sequences`
- 每次调用都会完整进出一遍 `FSDPVLLMShardingManager`（`fsdp_workers.py:668`）。上游的设计是：每次 generate 都执行 FSDP `state_dict()` → 对每个参数做 `DTensor.full_tensor()`（8 卡 NCCL all-gather）→ vLLM `load_weights`，结束后再 `sleep(level=1)`
- 单轮对话 RL 一个训练步只 generate 一次，没问题。我们是 200 步 episode，**一个训练步要做约 200 次全量 9B 权重同步和 200 次 sleep/wake**（每 10 步的验证再加 200 次），而 rollout 期间权重根本没变
  - **稳定性**：每个训练步约上万轮跨 8 卡集合通信，加上 vLLM `CuMemAllocator` 反复搬运显存。driver 端等 `generate_sequences` 的结果**没有超时**，只要一张卡卡住，整个训练就会一直等下去。这和"只剩一张卡在忙"的现象吻合
  - **性能**：`smoke16` 单帧短 prompt 实测每个环境步 6.7s，真正的生成不到 1s，其余基本都是这套同步。h29 正式训练每个环境步约 14s，其中同样有一大块是它

**修复**（commit `71b8582`）

1. **`fsdp_vllm.py`**：新增开关 `VERL_ROLLOUT_KEEP_AWAKE=1`（默认关闭，关闭时行为与上游一致）
   - 权重没变（上次同步之后没有 `update_actor`/`load_checkpoint`）就跳过同步。`sleep(level=1)` 只是把权重挪到 CPU，`wake_up` 后原样恢复，数值与同步后一致
   - 连续多次 generate 之间保持 vLLM 唤醒，不 sleep
   - 显存峰值不变：两次 generate 之间 GPU 上的内容和 generate 进行中完全一样
   - rank0 打印 `[fsdp_vllm] 全量权重同步 #N`，用于实跑核对。预期每个训练步只同步一次（有验证的步是两次），而不是 200 次
2. **`fsdp_workers.py`**：在 `compute_log_prob` / `compute_ref_log_prob` / `update_actor` / `save_checkpoint` / `load_checkpoint` 开头调 `release()`，先让 vLLM sleep 腾出显存；在 `update_actor` / `load_checkpoint` 结束时标记权重已变，下一次 generate 就只做一次全量同步。ref 一定在 `old_log_prob` 之后执行（`ray_trainer.py:1143` → `1181`），所以 ref 计算时 vLLM 已经释放
3. **`run_grpo_minecraft_train.sh` 卡死看门狗**（不管上面的修复是否真正命中根因，都能兜底）
   - 超过 `STALL_TIMEOUT_S`（默认 3h，约为正常一步的 3 倍）没有出现新的 `step:N` 行，就判定卡死
   - 先抓现场：用 `py-spy dump --native` 抓所有 `WorkerDict`/`TaskRunner`/`MinecraftWorker` 的调用栈；抓不到就发 SIGABRT，配合 `PYTHONFAULTHANDLER=1` 把栈打进 Ray 日志。连同 `nvidia-smi`、`ps`、Ray 日志压缩包、日志末尾 3000 行，一起传到 S3 的 `hang_diag_<时间>/`
   - 然后杀掉整套进程（main_ppo / Ray / Malmo JVM / Xvfb），用同一个 `CKPT_DIR` 重新拉起。`trainer.resume_mode=auto` 会从本地最新的 FSDP 存档（含优化器状态）续跑，最多丢 `SAVE_FREQ`=10 步
   - 重启时跳过训练前验证；固定 `WANDB_RUN_ID`，重启后续写同一条 wandb 曲线
   - 连续 3 次重启仍卡死，以退出码 124 结束任务，不再空转
4. **`setup_vllmtrain_env.sh`**：装上 `py-spy`，供看门狗使用

**没做到的验证**：看门狗原计划先在本地用假训练进程模拟"卡死 → 抓现场 → 重启"全流程，两次都因为本地命令审批超时没跑成，只做了逐行静态复核。`KEEP_AWAKE` 的效果也要等实跑日志里的同步次数和 `timing_s/gen` 才能确认。

**卡死前的训练效果**（`grpo_h29_mixed88_wandb`，24 步）

- 验证集 `val/success_rate`：**0.312（训练前）→ 0.375（step 10）→ 0.625（step 20）**。验证集只有 16 个环境，噪声大，但涨幅明显
- 训练集 `episode/success_rate`：前 5 步均值 0.475 → 后 5 步均值 0.610。对 step 线性回归的斜率是 +0.011/步，但标准差 0.165，24 步内统计上不显著
- `grad_norm` 从 6.35 降到 1.3~3.3，`kl_loss` 基本在 0.00~0.03，`entropy_loss` 从 0.34 降到约 0.20，`valid_action_ratio` 全程 100%，没有发散迹象
- S3 上保留了两份 HF 存档：`grpo_h29_mixed88_wandb/global_step_10/`、`global_step_20/`（每份 35GB，fp32）。本地的 FSDP 分片存档已随容器回收

**GPU 利用率低的原因**（与卡死无关，正常运行时也只有约 30%）：rollout（`timing_s/gen`，约 2850s）占单步总时长的 78%，真正吃满 GPU 的 `update_actor`（约 508s）只占 14%。rollout 慢有三个原因：
- Malmo 每一步都要等 JVM 物理模拟和渲染完成
- h29 下 prompt 平均 6722 token、每步只生成 15.5 token（prefill:decode ≈ 435:1），而且 `mm_processor_cache_gb=0`，每步都要完整重新 prefill
- `enforce_eager=True` 关掉了 CUDA graph

上面的权重同步问题，可能是这 78% 里比例最大的一块。

### 续训：`grpo_h29_mixed88_resume20`

- koala 任务名 `axiomjin-grpo-h29-resume20-normal-20261004-210836`（8×GPU）。原定的 `-j axiomjin-verl-grpo-h29-resume20` 超过了 koala 29 字符的前缀上限，已缩短
- 起点：`MODEL_PATH` = S3 上的 `grpo_h29_mixed88_wandb/global_step_20/`（HF 格式）。配置与首轮一致，`TOTAL_STEPS=80`，凑满总共 100 步。新任务的 step 编号从 1 重新开始，对应首轮的 step 21 起
- 代价：丢掉了优化器状态（Adam 动量），本地 FSDP 分片已随容器回收，无法恢复。lr=1e-6 下影响不大
- 已开启 `VERL_ROLLOUT_KEEP_AWAKE=1`、卡死看门狗和 wandb（run id 为 `grpoh29mixed88resume20`）
- 第一步要看的东西：
  1. `[fsdp_vllm] 全量权重同步 #N` 的增长速度：每个训练步应该只 +1（有验证的步 +2）
  2. `timing_s/gen` 是否明显低于首轮的约 2850s
  3. 训练前验证的 `val/success_rate` 是否接近 step 20 时的 0.625

**前两步实测结果**（启动后 2h37m）：

| 检查项 | 结果 |
|---|---|
| 全量权重同步次数 | `#1`（训练前验证）→ `#2`（step 1 后）→ `#3`（step 2 后），**每个训练步正好 1 次**，原来约 200 次。修复按预期生效 |
| `timing_s/gen` | 2180s / 2160s，首轮平均约 2851s，**降了约 24%** |
| `timing_s/step` | 2862s / 2813s，首轮平均约 3667s，降了约 23% |
| 训练前验证 `val/success_rate` | **0.812**。比首轮 step 20 的 0.625 还高，但验证集只有 16 个环境、温度 0.8 采样，噪声很大，只能说明续训起点没有退化 |
| 训练集 `success_rate` | 0.609 / 0.766（首轮最后 5 步均值 0.610） |
| 显存 | 88.37G，与首轮一致，`KEEP_AWAKE` 没有带来额外显存 |
| 异常 | `sim 创建失败` 1 次（重试成功），`step 失败` 0 次，看门狗未触发 |

**实际提速比预期小**：rollout 循环跑满 200 个环境步（最长的轨迹决定），每个环境步从约 14.3s 降到约 10.9s。去掉权重同步后，剩下的时间主要来自 h29 长 prompt 每步完整重新 prefill（`mm_processor_cache_gb=0`）、`enforce_eager`，以及 Malmo 自身的步进。所以同步并不是 rollout 慢的大头，前面"性能"部分的估计偏高了。

**卡死是否解决**：两个首轮任务分别跑了 9 步、24 步才卡住，目前只跑了 2 步，还不能下结论。但即使再次卡死，看门狗也会在 3 小时内抓现场并自动续跑。

**更新：已跑到 19/80 步，没有卡死**（`train2` 当初在 step 10 就卡住了，这次已经越过；首轮 `wandb` 卡在 step 25，还没到）。权重同步次数 = `#20`，与 19 个训练步 + 1 次验证完全对应。`step 失败` 4 次、`sim 创建失败` 12 次，都已被兜底，看门狗未触发。`global_step_10` 存档已上传 S3。

**⚠ 训练质量出现与首轮相反的趋势（原因待查）**：

| 指标 | 首轮 `wandb` step 1→24 | 续训 `resume20` step 1→19 |
|---|---|---|
| `actor/entropy_loss` | 0.34 → 约 0.20，逐渐下降 | **0.20 → 约 0.57，持续上升** |
| `response_length/mean` | 17.6 → 约 15 | **14.7 → 约 27** |
| `response_length/max` | 21~76 | 多次撞到 **128 上限**（被截断） |
| `actor/grad_norm` | 1.3~3.3 | 2.0~5.4 |
| 训练集 `success_rate` | 均值 0.51 | step 1~2 为 0.61/0.77，step 3~19 均值约 0.45 |
| 验证集 `val/success_rate` | 0.312 → 0.375 → 0.625 | **0.812（起点）→ 0.562（step 10）** |
| `rollout_probs_diff_mean` | 0.002 | 0.002 |

- 最后一行说明 vLLM 生成时用的权重和训练侧一致，`KEEP_AWAKE` 跳过同步**没有**造成 off-policy
- 还不能判定是退化：验证集只有 16 个环境，噪声大。但熵上升、回复变长、撞上长度上限是常见的 GRPO 不稳定信号
- 与首轮相比可能有关的差异：
  1. 续训的参考模型 = `MODEL_PATH` = step 20 的策略，而不是 SFT 起点，KL 的锚点变了
  2. Adam 状态重新初始化。首轮也是从零开始的，所以单凭这一点解释不了
  3. 同样的配置，在已经训练过 20 步的策略上继续训练。需要等 step 20 的验证结果，以及 wandb 曲线再判断

### 方案 B 提速实验

**改动**（commit `89ffbbb`，默认关闭，不影响正在跑的续训）：

1. **`VERL_ROLLOUT_SKIP_DONE=1`**（`rollout_loop.py`）：每个环境步只把未结束的环境送进 vLLM 生成，结果再按原位置拼回。已结束的行复用第一条活跃行的输出占位，prompt/response 都补齐到固定长度，形状一致。这些行 `active_masks=False`，不参与训练、不计 reward；环境侧对已结束的局直接回放终态，所以占位内容不影响结果。拼回时 batch 和生成输出没有重叠的 key，不会触发 `union` 的一致性断言
2. **rollout 分段计时**：每次 rollout 结束打印一行 `[rollout-timing] train|val env_steps= prep= gen= env= gen_rows=已生成行数/全量行数`，把 `timing_s/gen` 拆成拼 prompt / vLLM 生成 / Malmo 步进三段
3. **GPU 利用率采样**：训练脚本后台每 30s 记一次各卡 `utilization.gpu` 和显存，写入 `gpu_util.csv`，随 `train.log` 一起上传 S3

**方案 B 第三项（关闭 `enforce_eager`）做不了**：`vllm_rollout_spmd.py:107` 断言 `enforce_eager=False` 时必须 `free_cache_engine=False`，也就是不能让 vLLM sleep。那样 vLLM 会在训练阶段一直占着约 56G 显存（`gpu_memory_utilization=0.4`），加上训练峰值 88G，超过 140G，会 OOM。

**对比实验**：两个 8 卡任务，都从 `global_step_20` 起步，配置与 `resume20` 相同，`env.seed=0` 也相同，所以 step 1 抽到的任务和 `resume20` 的 step 1 一致，可以直接对比。各跑 3 步，不做验证、不存档：

| 任务 | koala 名 | 区别 | 对照 |
|---|---|---|---|
| `exp_skipdone` | `axiomjin-grpo-skipdone-normal-20261005-145442` | `SKIP_DONE=1` | `resume20` step 1~3：gen 2180/2160/2323s |
| `exp_skipdone_mmcache` | `axiomjin-grpo-skipmmc-normal-20261005-145515` | `SKIP_DONE=1` + `MM_PROCESSOR_CACHE_GB=4`，重新打开多模态缓存 | 同上；首轮曾因该缓存失步报 `Expected a cached item for mm_hash`，这次验证 `KEEP_AWAKE` 下是否仍会出现 |

**实验结果**（两个任务都跑完 3 步、正常退出）：

| 指标（3 步平均） | 基线 `resume20` step 1~3 | `skipdone` | `skipdone + mmcache` |
|---|---|---|---|
| `timing_s/gen` | 2221s | 1766s（**−20%**） | 1465s（**−34%**） |
| `timing_s/step` | 2890s | 2456s（−15%） | 2141s（**−26%**，约 36 分钟/步） |
| 实际生成行数 / 全量行数 | 12800/12800 | 59~72% | 57~70% |
| `update_actor` / `old_log_prob` / `ref` | 483 / 94 / 91s | 497 / 97 / 95s | 488 / 96 / 92s |
| `rollout_probs_diff_mean` | 0.002 | 0.002 | 0.002 |
| `mm_hash` 断言 / OOM | 0 / 0 | 0 / 0 | **0** / 0 |

- 三组的 `rollout_probs_diff_mean` 相同，熵和回复长度也在同一水平，说明两项改动都没有改变生成结果的分布
- 多模态缓存在训练 rollout 中没再报错，但**验证路径还没测**：这次实验关了验证，而首轮的 `mm_hash` 报错正是在训练前验证的第一批生成就出现的。正式启用前，要先开着验证跑一次

**rollout 分段计时**（`[rollout-timing]`，单次训练 rollout，`skipdone + mmcache`）：

| 段 | 耗时 | 说明 |
|---|---|---|
| `prep`（拼 prompt） | 396~430s | 主进程单线程，每个环境步约 2s：为 64 个环境渲染 h29 对话、分词约 6000 token、计算 rope 位置。**已结束的环境也照样处理** |
| `gen`（vLLM 生成） | 591~746s | 含 driver ↔ 8 个 worker 的数据传输 |
| `env`（Malmo 步进） | **29~34s** | 每个环境步只有约 0.15s |
| 其他（`timing_s/gen` 减去以上三段） | 约 320s | 轨迹整理、抽中的 256 个样本物化 pixel_values 等 |

**更正之前的判断**：之前认为 GPU 在等 Minecraft，实测 Malmo 一整次 rollout 只花约 30s，**不是瓶颈**。GPU 空等的主要原因是主进程上的 CPU 单线程工作（`prep` + 其他，合计约 740s，占 rollout 的一半）。GPU 采样也印证了这一点：每 30s 一次、共约 2000 个样本，**平均利用率 28~29%，中位数 0%**，只有约 30% 的样本超过 50%。

**下一步可做的提速**（还没实现）：
- `prep` 也跳过已结束的环境：预计 `prep` 再省约 40%
- 按环境增量缓存已渲染的对话和 token：h29 每一步只新增一轮，不必整段重新渲染、重新分词
- 多线程或多进程并行拼 prompt

### ⚠ 续训验证集成功率持续下滑

| | step 0（起点 = 首轮 step 20） | step 10 | step 20 |
|---|---|---|---|
| `val/success_rate` | 0.812 | 0.562 | **0.250** |
| `actor/entropy_loss`（前后几步） | 0.20 | 约 0.55 | 约 0.34~0.71 |
| `response_length/mean` | 14.7 | 约 29 | 约 20~27（多次撞 128 上限） |

- **训练集成功率没有崩**：step 3~24 均值约 0.47，首轮均值 0.51。`grad_norm` 2~5.4，`kl_loss` ≤0.05
- **不能确定是退化**：验证集只有 16 个环境，而且每次验证随机抽不同任务，难度差异很大。同一份 step 20 权重，首轮测出 0.625、续训起点测出 0.812，可见噪声在 ±0.2 量级。但 0.81→0.56→0.25 连续下滑，加上熵和回复长度同时上升，都是需要警惕的信号
- **排查过的可能原因**：
  1. `KEEP_AWAKE` 导致 vLLM 用了旧权重：可能性低。`rollout_probs_diff_mean` 始终是 0.002，同步次数与训练步数一一对应
  2. 跨环境步复用前缀缓存：可能性低。`KEEP_AWAKE` 之前，每次 generate 前都会重置 KV cache；现在一次 rollout 内的前缀缓存可以跨环境步复用（这可能也是提速的来源之一）。Qwen3.5 的混合注意力在 vLLM 里要求 `mamba cache mode='align'`；但如果缓存状态有错，vLLM 的 logprob 应该会和训练侧明显不一致，而实测并没有
  3. 续训的参考模型变成了 step 20 的策略、Adam 状态重新初始化：两者都会改变训练动态，单凭这些数据分不清
- **最有说服力的判断方法**：在固定任务集上用评测 harness（`run_backbone_eval.sh`）对比 SFT 起点 v2-e4、首轮 `global_step_20`、续训 `global_step_10/20` 这几个存档，不再依赖 16 个随机验证环境

**更新：下滑没有持续，主要是验证噪声**。`val/success_rate` 依次为 0.812 → 0.562 → 0.250 → **0.812 → 0.812** → 0.500（step 0~50）。`entropy_loss` 从 step 30 的约 0.68 回落到 step 40~51 的 0.28~0.47，`response_length/mean` 回到约 16~17，`response_length/max` 不再撞 128 上限。训练集成功率 step 40~51 均值约 0.66，首轮均值 0.51。step 20 的 0.250 更像是抽到了一批难任务。不过仍建议用固定任务集评测来确认，见"下一步"。

### 看门狗首次触发：拿到了卡死现场

续训任务在 step 27 的 rollout 中卡死。看门狗按设计处理：

1. 无进展 10814s（阈值 3h）后判定卡死
2. 抓现场并上传 `grpo_h29_mixed88_resume20/hang_diag_1005_205704/`
3. 杀掉进程，从本地 `global_step_20` 的 FSDP 存档（含优化器状态）自动续跑，跳过训练前验证

代价：step 21~26 重跑了一遍，加上 3h 判定期，共损失约 8.5h。之后一直正常推进到 step 51。

**现场证据**：
- **主进程**（`TaskRunner`）：卡在 `rollout_loop.py:504`，`ray.get` 等 `generate_sequences` 返回，和之前推测的位置一致
- **8 个 rollout worker**：7 个空闲，只有 rank 7（pid 20345）还在 `generate_sequences` 里
- **nvidia-smi**：GPU 0~6 利用率 **0%**，**GPU 7 利用率 100%**，显存 73.7G，和其他卡相差不大
- **rank 7 的调用栈**：`vllm LLM.generate → engine step → gpu_model_runner.execute_model → qwen3_5.forward → qwen3_next.forward → attention.forward → vllm_flash_attn.flash_attn_varlen_func → torch.zero_ → cudaLaunchKernel → libcuda`。线程状态 active，阻塞在 **kernel 发射**上
- **该 worker 的 Ray stderr/stdout**：只有启动期的信息，没有任何报错或警告

**解读**：
- 主机线程卡在 `cuLaunchKernel`、GPU 却持续 100% 忙碌、3 小时没有进展，这是典型的 **GPU 上某个 kernel 永不结束**，导致 CUDA 发射队列被占满。不是 Python 层面的死循环：如果 vLLM 的引擎循环还在推进，`max_tokens=128` 早就让请求结束了
- 也不是 NCCL 死锁：rollout 用 TP=1，每张卡独立生成，不涉及跨卡通信
- 栈里看到的是被阻塞的那次发射，真正不结束的是它前面已经发射的某个 kernel。候选：Qwen3.5 混合注意力（`qwen3_next` 的 GatedDeltaNet，属于 fla/mamba 类内核）或 flash-attn varlen，在 **prefix caching + mamba cache `align` 模式 + chunked prefill** 组合下（vLLM 0.17 对混合模型的这条路径支持还比较新），拿到异常的元数据后死循环
- 只有一次快照，**还不能确定是哪个 kernel**

**这也解释了首轮两次卡死**：现象完全相同（只剩约 1/8 算力在跑）。`enable_prefix_caching=True` 硬编码在 `vllm_rollout_spmd.py:201`，所有任务都开着，`KEEP_AWAKE` 开没开都卡过。所以 `KEEP_AWAKE` 不是根因，但它仍然有效地去掉了每个环境步一次的全量同步。

**频率**：3 次卡死分别发生在约第 10、25、27 个训练步，约每 20 步一次，每次都是 8 张卡里的某一张。

**后续可选方案**：
1. **关掉 prefix caching 做对照**：在 `engine_kwargs` 里覆盖 `enable_prefix_caching=False`，确认卡死是否消失。代价：h29 同一 rollout 内历史前缀的复用没了，rollout 会变慢。另外需要确认关掉之后，mamba cache 模式和 chunked prefill 的约束是否随之放宽
2. **给单次 generate 加超时**：driver 端等 `generate_sequences` 改成带超时的 `ray.get`，超时直接让任务退出，交给看门狗重启。比现在先等 3 小时再处理快得多，但仍会丢掉自上次存档以来的进度
3. **现在的看门狗能兜底**：约每 20 步卡一次，每次损失约 3~8 小时。对 100 步规模的训练可以接受，长期训练就必须根治

### 准备：固定任务集评测 + 下一轮训练的改动

#### ① 固定任务集评测（已提交）

协议和 SFT 起点 v2-e4 的基线完全一致，结果可以直接对比：
- `EVAL_BENCHMARK=easy`：202 个非 GUI 任务（Embodied 153 + Combat 49），seed=42 固定
- 每个任务 3 次 rollout，共约 606 次
- h29 上下文（`MAXIMUM_HISTORY_LENGTH=29`、`LIMIT_MM_IMAGE=30`），采样温度 0.8、top_p 0.99

基线：`cal-v2e4-ckpt520-easy-h29` = **28.9%**（174/603；Embodied 32.7%，Combat 17.0%），原测为 29.2%。评测噪声约 ±1.7pp。

202 个任务里 **88 个是训练任务池**，另外 **114 个没参与训练**，可以从 `summary.json` 的 `per_task` 分开算两组成功率，区分"学会了训练任务"和"泛化到新任务"。

| 评测任务 | 存档 | 从 SFT 起点算起的总步数 |
|---|---|---|
| `axiomjin-eval-vg-t20` | `grpo_h29_mixed88_wandb/global_step_20` | 20 |
| `axiomjin-eval-vg-t40` | `grpo_h29_mixed88_resume20/global_step_20` | 40 |
| `axiomjin-eval-vg-t70` | `grpo_h29_mixed88_resume20/global_step_50` | 70 |
| 待训练结束后提交 | `grpo_h29_mixed88_resume20/global_step_80` | 100 |

- 结果在 `s3://arcwm-code-us-west-2/axiom/eval_results/verl-<EXP>-s<STEP>-t<总步数>-easy-h29/summary.json`
- 新增启动脚本 `examples/eval_backbones/launch_verl_grpo_checkpoint.sh`，用法：`EXP=... STEP=... TOTAL_STEP=... bash submit_eval_job.sh launch_verl_grpo_checkpoint.sh`
- verl 的 HF 存档和 SFT 存档对比过：`config.json` 只差 `bos_token_id: null` / `use_cache` 两个无关字段，`tokenizer_config.json` 完全相同，评测 harness 可以直接加载
- **顺手修了 `submit_eval_job.sh`**：koala 2.4 移除了 `--s3-log`，默认集群也改成了 H20，这个脚本之前一提交就会失败。现在去掉了 `--s3-log`，并显式指定 `--cluster aws`

#### ② 下一轮训练的改动（已提交代码，默认值见下）

| 改动 | 位置 | 默认值 | 作用 |
|---|---|---|---|
| prefix caching 改为可配置 | `vllm_rollout_spmd.py`，脚本变量 `ENABLE_PREFIX_CACHING` | `True`（不变） | 设为 `False` 做卡死对照。参数从 `engine_kwargs` 里取出后再传给 `LLM()`，避免"同一个参数传了两次"的报错 |
| 单次 generate 超时 | `single_controller/ray/base.py`，环境变量 `VERL_GENERATE_TIMEOUT_S` | 训练脚本设为 900s | driver 等 `generate_sequences` 超时后打印 `[GENERATE_TIMEOUT]`，再等 `VERL_GENERATE_TIMEOUT_GRACE_S`（900s）才退出，留时间让看门狗趁卡住的 worker 还活着抓栈 |
| 看门狗响应更快 | `run_grpo_minecraft_train.sh` | 每 30s 检查一次（上传仍是每 10 分钟） | 看到 `[GENERATE_TIMEOUT]` 立即抓现场并重启；训练进程已经因超时退出时同样重启。**卡死后的空转时间从约 3h 降到约 15 分钟** |
| 跳过已结束的环境 | `VERL_ROLLOUT_SKIP_DONE` | 训练脚本改为默认 `1` | 已在方案 B 实验中验证：rollout −20%，生成分布不变 |

- **多模态缓存（`MM_PROCESSOR_CACHE_GB=4`）暂不默认打开**：验证阶段的生成路径还没测过，而首轮的 `mm_hash` 报错正是出在训练前验证的第一批生成上。需要开着验证单独跑一次确认
- **正在跑的 `resume20` 不受影响**：koala 只在容器启动时从 S3 拉一次代码，看门狗在容器内重启时用的仍是旧代码

**下一轮训练（等 `resume20` 跑完、评测出结果后再决定）**：
- 配置：`ENABLE_PREFIX_CACHING=False`、`SKIP_DONE=1`、generate 超时 + 看门狗
- 如果评测显示 RL 有提升，从 `resume20/global_step_80` 继续训练；否则先回头调整训练设置
- 判断卡死是否消失：原来约每 20 步卡一次，跑满 50 步以上都没有出现 `[GENERATE_TIMEOUT]`，才算基本排除

---

## 下一步（从"能跑"到"能用于实验"）

1. **补齐成绩曲线**：t70 已确认显著提升（45.5% vs 29.0%）。等 t20/t40 出结果，再加上 t100，看成绩随步数怎么变化，判断继续训练是否还有收益。之后最好用另一个随机种子重训一次，确认结果可复现
2. **`resume20` 跑完后**：提交 `global_step_80` 的评测（`EXP=grpo_h29_mixed88_resume20 STEP=80 TOTAL_STEP=100`），凑齐 20/40/70/100 四个点的成绩曲线
3. **下一轮训练**（根据评测结果决定是否继续训练）：配置 `ENABLE_PREFIX_CACHING=False` + `SKIP_DONE=1` + generate 超时，验证卡死是否消失。另外要开着验证跑一次，确认多模态缓存在验证阶段也不出错，之后才能把它设为默认（rollout 可再降约 14%）
4. **继续给 rollout 提速**：`prep`（拼 prompt）是现在最大的 CPU 瓶颈，每次 rollout 约 415s。可以跳过已结束环境的 prompt 处理、增量渲染对话
5. **吞吐对比自研线**：在相同 episode 长度和任务难度下，对比自研 HTTP eval-harness 的 rollout 吞吐，这是切换主线的硬指标（目标 ≥3×）。要用修复同步问题之后的数字来比
4. ~~单图 vs h29 多图历史~~ ✅ 已解决（见上）
5. ~~reward 稀疏~~ ✅ 已解决（真实任务池替代单任务冒烟）
