# NexusAlign — Config Framework Design

> 本文档定义 NexusAlign 的配置框架。它继承 `../NexusAlign` 的 Hydra 骨架,
> 但系统性修掉了几个会随「模态 × 算法」增多而塌方的耦合点。
>
> 设计三原则(与 `REPO_STRUCTURE.md` 一致):
> 1. **算法 ⟂ 模型正交** —— 算法配置里不出现模型,模型配置里不出现算法。
> 2. **组件 scope 隔离** —— 每个组件只拿到自己那段配置,绝不灌整份全局 dict。
> 3. **一切皆可组合的自由轴** —— model / algorithm / data / optimizer / scheduler /
>    runtime / reward_model 各自独立,recipe 只做几行 override。

---
 
 
## 0. 决策摘要(已拍板)

| 维度 | 选择 | 理由 |
|---|---|---|
| configs 位置 | **包内** `src/nexus_align/configs/` | 随 `pip install` 发布,复现性绑包版本 |
| 组合引擎 | **Hydra** (`defaults` 列表) | 白送 group 选择、CLI override、multirun sweep、结构化输出目录 |
| 采样/rollout 参数归属 | **algorithm 持有**,recipe 绑定 | 彻底解开「算法 × 模型」轴,杜绝 `flux.yaml` 里寄生 grpo/dpo 块 |

---

## 1. 与 `../NexusAlign` 的差异(改了什么、为什么)

| 痛点(旧) | 证据 | 新方案 |
|---|---|---|
| 整份配置灌给每个组件 `kwargs=cfg_dict` | `cli/train.py` 每个组件都收 `cfg_dict` | **组件 scope 化**:`build()` 只收 `cfg.<component>` + 轻量 `Context` |
| 算法参数寄生在模型配置 | `model/flux.yaml` 内有 `grpo:` / `dpo:` 块 | 采样参数移到 `algorithm.rollout`,模型只描述架构 |
| optimizer/scheduler 焊死在算法里、且重复 | `grpo.yaml`/`dpo.yaml` 的 optimizer 块逐字相同 | 拆成独立 `optim/` `sched/` config group,自由组合 |
| systems 层(fsdp/dtype)复制进每个组件 | `model` 和 `reward_model` 都各带 `fsdp:` | 抽到共享 `runtime/` group,组件只声明「是否参与分片」 |
| 扁平命名,无模态层 | `algorithm/grpo.yaml` 单层 | 加 `<algorithm>/<modality>/` recipe 层,支持矩阵可浏览 |
| 实例化在 entrypoint 手写 | `cli/train.py` 逐行 wire | 通用 `build_from_cfg(type=...)` + 显式 trainer 装配,二者并存 |

保留并继承的优点:Hydra `defaults` 组合、`registry` 按名取类、配置与 import 路径解耦、最小化顶层校验、`train.yaml`/`evaluation.yaml` 分入口。

---

## 2. 顶层 schema(权责边界)

每个 recipe 解析后是一棵固定顶层 key 的树。**新算法/新模型只是往这些 slot 填新内容,绝不发明新的顶层结构** —— 这是可扩展性的根。

```yaml
common:        # 实验元信息:seed / task / 调试开关
model:         # 模型轴:架构 + 架构内禀的推理默认(不含任何算法参数)
algorithm:     # 算法轴:loss/train/rollout/advantage(不含任何模型名)
data:          # 数据集 + dataloader(总是配对)
optimizer:     # 自由轴
lr_scheduler:  # 自由轴
runtime:       # systems 层:parallelism / precision / checkpoint / logging
reward_model:  # 仅 RLHF/评测需要;其余算法可缺省
trainer:       # 循环装配:max_epochs / grad_accum / 验证评测间隔
```

### 权责边界表(谁拥有什么 —— 正交性的硬约束)

| 字段 | 归属 | **绝不**出现在 |
|---|---|---|
| 架构超参(layers/dim)、权重路径、native 推理默认 | `model` | algorithm |
| loss 超参、rollout/sampling、advantage、KL/clip | `algorithm` | model |
| 优化器类型/lr/wd/betas | `optimizer` | algorithm |
| warmup/schedule 类型/总步数 | `lr_scheduler` | algorithm、optimizer |
| FSDP/TP/PP、amp_dtype、activation ckpt | `runtime` | model、reward_model |
| 数据集路径、清洗、num_workers | `data` | model、algorithm |

> **黄金规则**:看一个字段时问「它换模型会变,还是换算法会变?」
> 换模型变 → 归 `model`;换算法变 → 归 `algorithm`;两者都变 → 它属于
> **二者交汇处,即叶子 recipe**,在那里 override,而不是塞进任一组件的默认里。

---

## 3. 目录结构(包内)

```
src/nexus_align/configs/
├── train.yaml                 # 训练入口(@hydra.main config_name)
├── evaluation.yaml            # 评测入口
│
├── _base_/                    # 只被继承、从不直接跑的「零件」
│   ├── common.yaml            # seed / task / debug 默认
│   ├── runtime/
│   │   ├── single_gpu.yaml
│   │   ├── ddp_bf16.yaml
│   │   └── fsdp_bf16.yaml     # full_shard + activation ckpt + amp
│   ├── optim/
│   │   ├── adamw.yaml
│   │   └── adamw8bit.yaml
│   └── sched/
│       ├── constant_warmup.yaml
│       └── cosine.yaml
│
├── model/                     # 模型轴 —— 纯架构,一个文件一个模型
│   ├── llm/qwen2_0.5b.yaml
│   ├── vlm/llava_1.5_7b.yaml
│   └── image_gen/flux.yaml
│
├── data/                      # 数据集 + dataloader
│   ├── llm/alpaca.yaml
│   └── image_gen/hpd_v2.yaml
│
├── reward_model/              # 仅 RLHF/评测
│   └── image_gen/hps_v2.yaml
│
└── recipe/                    # ★ 真正能跑的实验,按「算法/模态」排
    ├── sft/
    │   ├── llm/qwen2_0.5b_alpaca.yaml
    │   └── vlm/llava_pretrain.yaml
    ├── pretrain/llm/...
    ├── rlhf/
    │   ├── dpo/image_gen/flux_hpd.yaml
    │   ├── ppo/llm/...
    │   └── grpo/image_gen/flux_hpd.yaml
    ├── distillation/...
    ├── pruning/...
    └── quantization/...
```

要点:
- `model/` `data/` `reward_model/` 内**按模态分子目录**,避免 `sft` 撞名,也让支持矩阵可浏览。
- `_base_/` 只放**合理默认零件**;真正的实验在 `recipe/`。两类文件物理分开。
- **继承链最多 1~2 层**(零件之间不深度互继承)—— 这是 MMEngine 系配置最容易踩的「简便性」坑。

---

## 4. 组合模型(Hydra `defaults`)

入口 `train.yaml` 只声明骨架与默认 group:

```yaml
# src/nexus_align/configs/train.yaml
defaults:
  - _self_
  - common: default          # _base_/common.yaml
  - runtime: fsdp_bf16        # _base_/runtime/fsdp_bf16.yaml
  - optimizer: adamw          # _base_/optim/adamw.yaml
  - lr_scheduler: cosine      # _base_/sched/cosine.yaml
  - model: null               # 由 recipe 指定
  - data: null
  - algorithm: null
  - reward_model: null        # 非 RLHF 可保持 null
  - recipe: null              # ★ recipe 是最后一层,可覆盖以上任意 group

trainer:
  max_epochs: 1
  grad_accum_steps: 1
  max_grad_norm: 1.0
  validate_interval: 0
  evaluate_interval: 0

ckpt_manager:
  save_ckpt_root: checkpoints/
  save_ckpt_every_n_steps: 50
  save_ckpt_keep_last_n_steps: 3
```

跑实验 = 选一个 recipe:

```bash
python -m nexus_align.cli.train recipe=rlhf/grpo/image_gen/flux_hpd
```

recipe 文件用 Hydra `override` 语法把各 group 一次性绑好,再写几行差异:

```yaml
# recipe/rlhf/grpo/image_gen/flux_hpd.yaml
# @package _global_
defaults:
  - override /model: image_gen/flux
  - override /data: image_gen/hpd_v2
  - override /reward_model: image_gen/hps_v2
  - override /runtime: fsdp_bf16
  - override /optimizer: adamw
  - override /lr_scheduler: constant_warmup
  - /algorithm: grpo            # 算法默认从 algorithm group 引入

common:
  task: image_gen

optimizer:
  lr: 1.0e-5                    # 只覆盖跟默认不同的

# ★ flux × grpo 的交汇参数,在 recipe 里绑定(不污染 model/algorithm)
algorithm:
  rollout:
    sample_height: 720
    sample_width: 720
    sample_steps: 20
    sample_cfg: 3.5

trainer:
  max_train_total_steps: 1000
```

> 算法配置 `algorithm/grpo.yaml` 提供**模型无关的 rollout 默认**(group_size、clip_range、kl_coeff…);
> 分辨率这种**模型相关**的值,由 recipe 覆盖。模型文件里**永远没有** grpo/dpo 块。

---

## 5. 组件 scope 化与实例化(核心修正)

### 5.1 不再灌全局 dict —— `Context` + scoped cfg

定义一个轻量、只读的运行时上下文,承载所有组件都需要的少量共享物:

```python
# nexus_align/core/context.py
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class Context:
    device: torch.device
    rank: int
    world_size: int
    runtime: dict      # 解析后的 cfg.runtime(fsdp/precision/...)
    common: dict       # seed/task/debug
    # 注意:这里没有 model/algorithm/data —— 组件拿不到别人的配置
```

每个组件的构造**只接收自己那段 cfg + context**:

```python
class BaseModel(ABC):
    @classmethod
    @abstractmethod
    def build(cls, cfg: dict, ctx: Context) -> "BaseModel":
        """cfg 是 cfg.model 这一段,不是全局。"""
        ...
```

这样从签名就能看出一个组件需要什么;模型再也无法读到 `cfg.algorithm.xxx`,
耦合在编译期就被掐断。`flux.yaml` 里那种 `grpo:` 寄生块**在新框架里根本无处安放**。

### 5.2 registry 按 `type` 取类,通用构造

沿用现有 `Registry`(component_type → name → class),但配置里用 `type` 当 key,
并提供一个通用构造器,供「自由轴」组件复用:

```python
# nexus_align/core/build.py
def build_from_cfg(component_type: str, cfg: dict, ctx: Context):
    cls = registry.get(component_type, cfg["type"])   # cfg["type"] 是注册名
    return cls.build(cfg, ctx)
```

```yaml
# model/image_gen/flux.yaml
type: flux                       # registry key(取代旧的 name 兼作选择器)
path: black-forest-labs/FLUX.1-dev
# 架构内禀的推理默认(eval 用),不是算法 rollout:
generation: { height: 720, width: 720, num_infer_steps: 50, cfg: 3.5 }
# 是否参与分片由 runtime 决定,这里只声明意图:
shardable: true
```

### 5.3 entrypoint:通用构造 + 显式 trainer 装配

`cli/train.py` 不再逐组件手抄全局 dict;而是 scoped 构造,最后显式装配 trainer
(trainer 装配保持显式,便于 debug —— 这是有意保留的旧优点):

```python
@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
@with_env_setup(validator=validate_train_config)
def main(cfg, env):
    ctx = Context(device=env.device, rank=env.rank, world_size=env.world_size,
                  runtime=cfg.runtime, common=cfg.common)

    data       = build_from_cfg("dataset",      cfg.data,        ctx)
    model      = build_from_cfg("model",        cfg.model,       ctx)
    reward     = build_from_cfg("reward_model", cfg.reward_model, ctx) if cfg.get("reward_model") else None
    optimizer  = build_from_cfg("optimizer",    cfg.optimizer,   ctx)   # 自由轴
    scheduler  = build_from_cfg("lr_scheduler", cfg.lr_scheduler, ctx)  # 自由轴
    algorithm  = build_from_cfg("algorithm",    cfg.algorithm,   ctx)   # 只拿 cfg.algorithm

    # 显式装配(对应 BaseTrainer.__init__ 的分离式签名)
    trainer = Trainer(train_dataloader=data, model=model, algorithm=algorithm,
                      reward_model=reward, optimizer=optimizer,
                      lr_scheduler=scheduler, cfg=cfg.trainer, ctx=ctx)
    trainer.run()
```

> 注意 optimizer / lr_scheduler 现在是**走 registry 的自由轴**,不再是 algorithm 里的
> 字典查表,也不再被复制进每个算法 yaml。

---

## 6. 各轴 config 范例

### algorithm(模型无关)
```yaml
# algorithm/grpo.yaml
type: grpo
train: { max_grad_norm: 1.0 }
rollout:                      # 模型无关默认;分辨率交给 recipe 覆盖
  group_size: 8
  adv_clip_max: 5.0
  clip_range: 1.0e-2
  kl_coeff: 0.0
  sample_steps: 20            # 通用默认
```

### optimizer / lr_scheduler(自由轴,可任意组合)
```yaml
# _base_/optim/adamw.yaml
type: adamw
lr: 1.0e-5
weight_decay: 0.0
betas: [0.9, 0.999]
```
```yaml
# _base_/sched/cosine.yaml
type: cosine
warmup_steps: 0
min_lr_ratio: 0.0
```

### runtime(systems 层,组件不再各自带 fsdp)
```yaml
# _base_/runtime/fsdp_bf16.yaml
parallel:
  strategy: fsdp
  fsdp_strategy: full_shard
  cpu_offload: false
  activation_ckpt: true
precision:
  model_dtype: fp32
  amp_dtype: bf16
checkpoint:
  sharded: true
logging:
  wandb: { entity: my_entity, project: my_project, offline: true }
```

---

## 7. 两个完整 recipe 范例

### 7.1 LLM SFT(最简)
```yaml
# recipe/sft/llm/qwen2_0.5b_alpaca.yaml
# @package _global_
defaults:
  - override /model: llm/qwen2_0.5b
  - override /data: llm/alpaca
  - override /runtime: fsdp_bf16
  - override /optimizer: adamw
  - override /lr_scheduler: cosine
  - /algorithm: sft

common: { task: llm }
optimizer: { lr: 2.0e-5 }
trainer: { max_epochs: 3 }
```
```bash
python -m nexus_align.cli.train recipe=sft/llm/qwen2_0.5b_alpaca
```

### 7.2 Flux GRPO(RLHF,见 §4)—— 同一套骨架,只是多了 reward_model 与 rollout override。

---

## 8. CLI override 与 sweep(Hydra 白送)

```bash
# 单值覆盖
python -m nexus_align.cli.train recipe=sft/llm/qwen2_0.5b_alpaca optimizer.lr=1e-5

# 换组件(group 级)
python -m nexus_align.cli.train recipe=... model=llm/qwen2_7b runtime=ddp_bf16

# 多组网格 sweep(multirun)
python -m nexus_align.cli.train -m recipe=... optimizer.lr=1e-5,2e-5 algorithm.rollout.group_size=4,8
```

---

## 9. 校验与 schema

沿用并加强现有 `validate_train_config` 的最小校验,fail-fast:

```python
TRAIN_REQUIRED_TOP_LEVEL = (
    "common", "model", "algorithm", "data",
    "optimizer", "lr_scheduler", "runtime", "trainer",
)
# reward_model 仅在 algorithm 声明 needs_reward=True 时必需
```

进阶(可选):用 Hydra structured config(dataclass schema)做**类型级**校验,
在组合期就拦下拼错的 key —— 但起步阶段 top-level 校验已够用,别过度工程。

---

## 10. 约定 checklist

- [ ] 配置里用 `type`(registry 名),**绝不写 import 路径**。
- [ ] 组件 `build(cfg_scoped, ctx)`,**绝不接收全局 cfg_dict**。
- [ ] 模型配置里**没有任何算法名/算法参数**;算法配置里**没有任何模型名**。
- [ ] optimizer / scheduler / runtime 是**独立 group**,不嵌进 algorithm。
- [ ] 模型 × 算法的交汇参数,写在**叶子 recipe**,不写进任一组件默认。
- [ ] `_base_` 继承链 ≤ 2 层。
- [ ] `model/` `data/` `reward_model/` 按**模态**分子目录;`recipe/` 按**算法/模态**排。
- [ ] 一个 recipe = 一条命令 = 一次可复现实验。

---

## 11. 从 `../NexusAlign` 迁移的动作清单

1. `model/flux.yaml`:删掉 `grpo:`/`dpo:` 块 → 通用部分进 `algorithm/*.yaml` 的 `rollout`,
   分辨率等进各 recipe;`fsdp:` 块 → 进 `runtime/`。
2. `algorithm/*.yaml`:抽出 `optimizer:` 块 → `_base_/optim/`;`run.lr_schedule` → `_base_/sched/`。
3. `cli/train.py`:`kwargs=cfg_dict` → scoped `build_from_cfg` + `Context`。
4. 把扁平 `data/ model/ algorithm/` 内的文件按模态归子目录;新增 `recipe/<algo>/<modality>/`。
5. `core/config.py`:更新 `TRAIN_REQUIRED_TOP_LEVEL`(加 optimizer/lr_scheduler/runtime/trainer)。
```
