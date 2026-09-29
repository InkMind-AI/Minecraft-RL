# verl 迁移调试进展汇总（截至 09-30）

> 本文档是"把 verl 调通"这条主线的执行记录，按时间顺序梳理每一层验收状态、
> 踩过的坑和修复方式。更偏顶层的迁移设计/规划见 `verl_migration.md`；本文档
> 侧重"实际调试过程中发生了什么、为什么坏、怎么修的"，供后续排错和写作复盘用。

## 总览：四层验收状态

| 层 | 内容 | 状态 |
|---|---|---|
| **1. 适配器** | verl-agent + qwen3_5 模型后端，真实 9B 权重前向 | ✅ 通过（`verl-c4`，09-27） |
| **2. 环境接入** | Minecraft/Malmo 环境包 + `MinecraftEnvironmentManager` | ✅ 通过（`verl-env-d5`，09-28） |
| **2.5 训练/推理共存可行性** | HF 训练 + vLLM 推理能否同进程、权重能否同步 | ✅ 可行（`colocate-p2`，09-28） |
| **3. GRPO 训练启动** | `verl.trainer.main_ppo` 端到端跑通 rollout→advantage→update | ⏳ 调试中（`grpo-smoke1`→`smoke9`，09-28~09-29，7 个根因已修，`smoke9` 排队中） |
| **4. 端到端训练 + 吞吐** | ≥2 个 update step 完整跑通；吞吐/GPU 利用率对比自研线 | ❌ 尚未开始 |

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

## 第 3 层：GRPO 训练启动（09-28~09-29，进行中，已修复 5 个独立 bug）

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
| `smoke9` | ⏳ 已提交，`Queuing #11`（集群排队，尚未开始执行） | — | — |

前 8 轮修复均已提交并同步到 S3；每轮都是"改代码 → 语法检查 → 提交 → 用 S3
日志目录核实真的入队 → 拉日志定位下一个问题"的循环，累计暴露 7 个独立
根因（6 个环境/框架兼容性问题 + openjdk 的 `set -u` 坑复现两次）。**`smoke6`
是重要里程碑**：环境组装（fla 内核、minestudio、真实起一次 Malmo）第一次
完整走完不再报错，问题域已从"环境能不能装起来"收窄到"训练配置/收尾细节"。
`smoke7`/`smoke8` 是同一个 openjdk 坑的两次不同触发路径，已改为从根上
（不恢复 `set -u`）解决，理论上不会再犯第三次。

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

## 当前状态（09-30，被集群基础设施问题阻塞）

`smoke9` 因 CLI 升级后 `--s3-log` 被移除而失败提交（见下），改用 `smoke10`
（1 GPU，去掉 `--s3-log`）重新提交，**已排队 2.5h+，队列位次 #21→#19→#15
后停滞不动**——这是集群本身资源紧张，不是代码或配额问题（配额 32 卡仅用 1-2，
远未打满）。

**已排除的绕行方案**：

1. **迁移到 H20 集群**（完全空闲，32 卡可用、队列为空）——探针任务提交后
   卡在 `ContainerCreating` 超过 5 分钟未起（诊断直接提示"节点故障"），大概率
   是阿里云节点拉 AWS ECR 镜像跨云失败/极慢。**代码和模型权重都在 AWS S3，
   短期内不具备迁移 H20 的条件**
2. **改用 debug 模式 + `koala exec`/`koala ssh` 交互式跑**——debug 配额是独立
   并发池，提交后立即 `Running`（不受 normal 队列排队影响），但 `koala ssh
   --connect` / `koala exec` 反复重试 40 次后必然报
   `Connection aborted: RemoteDisconnected`。排查过程：
   - `ssh cvm`（跳板机第一跳）直连测试**成功**——网络本身通
   - `~/.ssh/config` 里 `Host koala` 只有 `ProxyJump cvm`，缺 `HostName`——
     说明"cvm → pod"这一跳的隧道建立步骤从未成功写回配置
   - `koala get` 确认 pod 本身 `Running`（k8s/MPIJob 状态正常，命令是占位的
     `sleep 8h`）——**pod 健康，问题在 cvm 到 pod 的隧道/端口转发这一层**，
     是基础设施问题，非本地配置或代码可修
   - 已删除该 debug pod 释放配额（`koala exec` 交互式调试这条路当前**不可用**）

一旦 `smoke10` 真正开始跑，理论上应该：沿用 `smoke6` 已验证过的 setup 路径
（fla/minestudio/Malmo 全部装好）→ 用 `smoke6` 定位到的
`enable_chunked_prefill=True` 修复过 vLLM engine 初始化 → 用 `smoke7`/`smoke8`
修复过的 openjdk `set -u` 问题 → 第一次真正进入 GRPO 训练循环
（rollout → advantage → update）。**修复本身没有新问题，纯粹卡在排队。**

⚠️ **koala CLI 已升级 2.3.6 → 2.4.0，带来两处不兼容**：
1. 默认集群改为 H20，AWS 集群需显式加 `--cluster aws`（否则 `koala ls`
   看不到任务，会误判"队列已空"）
2. **`--s3-log` 参数已被移除**（"仅支持平台保留的 Pod 日志"），此前全程依赖的
   "`--s3-log` 提交 + `aws s3 cp .koala-logs/...` 拉日志"这套流程**不再可用**，
   改用 `koala logs <job> --cluster aws --all`（任务结束后日志是否仍可查还
   需要验证——`.koala-logs` 这条路径已知失效，后续必须切到 `koala logs`）

H20 集群的路径/存储约定也变了（个人持久目录 `/cpfs/<企业ID>`，代码走东京
OSS）——只有在真正决定迁移 H20 时才需要处理，目前不适用。

## 下一步

1. 继续等 `smoke10` 排队（唯一在推进的路径），到号后立即用 `koala logs
   --cluster aws --all` 拉取结果（新日志方式，不再是 S3 路径）
2. 若排队持续无进展，可考虑：拆分更小的资源请求（已试 1 GPU，效果有限）、
   找管理员/工单排查集群拥堵、或排查 `koala ssh`/`exec` 隧道故障是否是
   全局性问题（值得报给平台方，这样后续能用 debug 模式绕开 normal 排队）
3. 跑通冒烟后，扩大到真正的小规模训练（≥2 个 update step，非冒烟规模）
4. 测吞吐/GPU 利用率，对比自研线（HTTP eval-harness rollout）——这是切换
   决策的核心指标，目前仍是空白
5. 解决"每样本单图 vs h29 多图历史"的结构性差距，否则 verl 线产出的策略
   与评测条件不一致，无法做公平的成绩对比
