"""CLI entry: Hydra-configured entry point with environment setup."""

import os

import hydra
import torch.distributed as dist

import nexus_align.models  # noqa: F401  # registers model factories on import
import nexus_align.datasets  # noqa: F401  # registers dataset factories on import
import nexus_align.algorithms  # noqa: F401  # registers algorithm factories on import
import nexus_align.trainers  # noqa: F401  # registers trainer factories on import
from nexus_align.registry import registry
from nexus_align.engine.setup import with_env_setup
from nexus_align.datasets.dist_dataloader import build_dataloader


@hydra.main(
    config_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs")),
    config_name="main",
    version_base="1.3",
)
@with_env_setup
def main(cfg, device):
    # 1. Prepare dataset
    train_dataset = registry.get("dataset", cfg.data.name)(cfg)
    train_dataloader = build_dataloader(cfg, train_dataset, mode="train")
    print("✅ Prepared training dataset")

    # 2. Prepare model (FSDP-wrapped with an EMA copy; see BaseModel)
    model = registry.get("model", cfg.model.name)(cfg, device)
    print("✅ Prepared model")

    # 3. Prepare algorithm
    algorithm = registry.get("algorithm", cfg.algorithm.name)(cfg)
    print("✅ Prepared algorithm")

    # 4. Prepare trainer
    trainer = registry.get("trainer", cfg.algorithm.trainer)(cfg, train_dataloader, model, algorithm)
    print("✅ Prepared trainer")

    # 5. Run training (evaluation is decoupled; run eval.sh on saved checkpoints)
    trainer.run()

    dist.barrier()
    print("✅ Training completed")


if __name__ == "__main__":
    main()
