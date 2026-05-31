# NexusAlign — Repository Structure

> NexusAlign is a unified research framework that aligns with state-of-the-art
> foundation model algorithms — pretraining, SFT, RLHF, distillation, pruning,
> and quantization — across LLMs, VLMs, visual generation, and unified models,
> in the simplest way possible.

This document describes the recommended file/directory layout. The design follows
two principles drawn from professional foundation-model repos (Megatron-LM, veRL,
OpenRLHF, LLaMA-Factory, transformers):

1. **Algorithm × Modality orthogonality** — algorithms (how you train) and models
   (what you train) are independent axes and must not be tangled. A new algorithm
   should work across models; a new model should work across algorithms.
2. **Config-driven, recipe-first** — every reproducible result is one YAML config +
   one launch command. Code stays generic; configs hold the specifics.

---

## Top-level layout

```
NexusAlign/
├── README.md                  # one-line intro + quickstart + model/algo matrix
├── LICENSE                    # Apache-2.0 recommended for model frameworks
├── CITATION.cff               # how to cite the repo
├── CONTRIBUTING.md            # dev setup, PR checklist, code style
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
├── pyproject.toml             # single source of truth: deps, build, tool config
├── requirements.txt           # optional pinned runtime deps (or extras in pyproject)
├── Makefile                   # make install / lint / test / format
├── .gitignore
├── .gitattributes             # Git LFS rules for large assets
├── .pre-commit-config.yaml    # ruff / black / isort / mypy hooks
│
├── .github/                   # CI, issue/PR templates, social preview
│   ├── workflows/
│   │   ├── ci.yml             # lint + unit tests on PR
│   │   └── release.yml        # build + publish to PyPI
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
│
├── assets/                    # images used by README/docs (relative-path refs)
│   ├── logo.png
│   ├── architecture.png
│   └── benchmarks/
│
├── docs/                      # full documentation (mkdocs / sphinx)
│   ├── index.md
│   ├── getting_started.md
│   ├── algorithms/            # one page per algorithm w/ math + references
│   ├── models/                # one page per model family
│   ├── tutorials/
│   └── faq.md
│
├── configs/                   # ALL experiment recipes live here (the heart)
│   ├── _base_/                # composable base fragments (inherited, not run)
│   │   ├── models/            # model arch defaults
│   │   ├── data/              # dataset/dataloader defaults
│   │   ├── optim/             # optimizer / scheduler defaults
│   │   └── runtime/           # parallelism, precision, logging defaults
│   ├── pretrain/
│   │   ├── llm/
│   │   ├── vlm/
│   │   └── visual_gen/
│   ├── sft/
│   ├── rlhf/                  # ppo / dpo / grpo / reward-model recipes
│   ├── distillation/
│   ├── pruning/
│   └── quantization/          # gptq / awq / smoothquant recipes
│
├── nexusalign/                # the installable Python package (src layout)
│   ├── __init__.py
│   │
│   ├── models/                # MODEL axis — architecture definitions only
│   │   ├── base.py            # common interfaces / mixins
│   │   ├── llm/
│   │   ├── vlm/
│   │   ├── visual_gen/        # diffusion / autoregressive image models
│   │   ├── unified/           # any-to-any / multimodal unified models
│   │   └── registry.py        # name -> model-class registry
│   │
│   ├── algorithms/            # ALGORITHM axis — training logic, model-agnostic
│   │   ├── base.py            # Algorithm ABC: build_loss / step / on_*
│   │   ├── pretrain/
│   │   ├── sft/
│   │   ├── rlhf/
│   │   │   ├── ppo.py
│   │   │   ├── dpo.py
│   │   │   ├── grpo.py
│   │   │   └── reward_model.py
│   │   ├── distillation/
│   │   ├── pruning/
│   │   └── quantization/
│   │
│   ├── data/                  # datasets, tokenization, preprocessing, collators
│   │   ├── datasets/
│   │   ├── tokenizers/
│   │   ├── processors/        # image/video/multimodal preprocessing
│   │   └── collators.py
│   │
│   ├── trainers/              # glue: drives algorithm over data on engine
│   │   ├── base_trainer.py
│   │   └── rl_trainer.py      # rollout + learner loop for RLHF
│   │
│   ├── engine/                # systems layer: parallelism, precision, ckpt
│   │   ├── parallel/          # ddp / fsdp / tensor / pipeline wrappers
│   │   ├── precision.py       # amp / bf16 / fp8
│   │   ├── checkpoint.py      # save/load/resume, sharded ckpt
│   │   └── distributed.py
│   │
│   ├── inference/             # generation / serving helpers for eval & rollout
│   │
│   ├── evaluation/            # metrics, benchmark harnesses
│   │   ├── metrics/
│   │   └── benchmarks/
│   │
│   ├── config/                # config schema, parsing, composition/override
│   │   ├── schema.py
│   │   └── builder.py
│   │
│   └── utils/                 # logging, seeding, registry, io, env
│       ├── logging.py
│       ├── registry.py
│       └── seed.py
│
├── scripts/                   # thin entrypoints — parse config, call trainer
│   ├── train.py               # python scripts/train.py --config configs/sft/...
│   ├── eval.py
│   ├── generate.py
│   └── convert_checkpoint.py  # to/from HF format
│
├── examples/                  # runnable, documented end-to-end walkthroughs
│   ├── llm_sft/
│   │   ├── README.md
│   │   └── run.sh
│   ├── llm_dpo/
│   ├── vlm_pretrain/
│   └── quantize_llm/
│
├── tools/                     # standalone utilities (data prep, weight conversion)
│
├── tests/                     # pytest: unit + integration + tiny smoke configs
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
└── benchmarks/                # reproducibility: scripts + expected numbers/curves
    └── results/
```

---

## Why this layout

### The two-axis core: `models/` ⟂ `algorithms/`
This is the single most important decision. Keeping **architecture** (`models/`)
and **training logic** (`algorithms/`) in separate trees is what lets you say
"SFT works on LLM, VLM, and unified models" without duplicating code. A trainer
composes `(model, algorithm, data, engine)` at runtime via the registry — no
algorithm hard-codes a model, and no model hard-codes a loss.

### `configs/` mirrors the algorithm tree
Experiments are recipes, not code. The `configs/<algorithm>/<modality>/` hierarchy
makes the supported matrix browsable at a glance, and `_base_/` fragments keep each
recipe a few lines of overrides instead of a wall of hyperparameters — this is the
concrete form of "simplest way".

### `scripts/` are thin; `nexusalign/` holds the logic
Entrypoints only parse a config and hand off to a trainer. All reusable logic lives
in the importable package, so the same code path runs in tests, examples, and prod.

### `engine/` isolates the systems layer
Parallelism, mixed precision, and checkpointing are orthogonal to *what* algorithm
runs. Isolating them means a researcher editing a loss function never touches FSDP
wiring, and a systems change applies to every algorithm at once.

### `examples/` vs `benchmarks/`
- `examples/` — minimal, fast, "here's how to use it" (often tiny models/data).
- `benchmarks/` — full reproductions with **expected numbers committed**, so anyone
  can verify a SOTA result matches the paper.

---

## README structure (recommended sections)

A professional foundation-model README typically follows this order:

1. **Logo + one-line intro** (the tagline)
2. **Badges** — CI, PyPI, license, docs, paper
3. **News / Updates** — reverse-chronological highlights
4. **Highlights / Features** — what makes it "simple" + the algo/model coverage
5. **Supported Matrix** — a table of Algorithm × Modality (✅ / 🚧 / planned)
6. **Installation**
7. **Quickstart** — one copy-paste command that trains something small
8. **Documentation** — link to `docs/`
9. **Benchmarks / Reproductions** — table linking to `benchmarks/`
10. **Roadmap**
11. **Contributing**
12. **Citation**
13. **License**
14. **Acknowledgements**

### Suggested support matrix (put this near the top of README)

| Algorithm     | LLM | VLM | Visual Gen | Unified |
|---------------|:---:|:---:|:----------:|:-------:|
| Pretraining   | ✅  | ✅  | 🚧         | 🚧      |
| SFT           | ✅  | ✅  | —          | 🚧      |
| RLHF          | ✅  | 🚧  | 🚧         | —       |
| Distillation  | ✅  | 🚧  | 🚧         | —       |
| Pruning       | ✅  | 🚧  | —          | —       |
| Quantization  | ✅  | 🚧  | —          | —       |

Legend: ✅ supported · 🚧 in progress · — planned/N/A

---

## Conventions

- **Package layout**: `src`-style is optional; a top-level `nexusalign/` package is
  fine and simpler. Pick one and keep imports absolute (`from nexusalign.algorithms ...`).
- **Configs**: YAML with inheritance (`_base_`) + CLI overrides
  (`--optim.lr 1e-4`). One config = one reproducible run.
- **Registries**: every model / algorithm / dataset registers under a string name so
  configs stay declarative and decoupled from import paths.
- **Checkpoints**: support round-trip conversion to/from Hugging Face format
  (`scripts/convert_checkpoint.py`) — essential for an "align with SOTA" framework.
- **Large files**: never commit weights; use Git LFS for unavoidable large assets and
  document download scripts in `tools/`.
- **Images**: README images go in `assets/`, referenced by relative path so they
  render on both GitHub and Hugging Face.
```
