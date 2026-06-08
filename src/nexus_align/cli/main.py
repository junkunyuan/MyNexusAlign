"""CLI entry: Hydra-configured entry point with environment setup."""

import os

import hydra

from nexus_align.registry import registry
from nexus_align.engine.setup import with_env_setup


@hydra.main(
    config_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs")),
    config_name="main",
    version_base="1.3",
)
@with_env_setup
def main(cfg):
    # 1. Prepare dataset
    train_dataset = registry.get("dataset", cfg.data.name)(cfg.data)
    print("✅ Prepared training dataset")

    # 2. Prepare model
    model = registry.get("model", cfg.model.name)(cfg)
    print("✅ Prepared model")

    # 3. Prepare algorithm
    algorithm = registry.get("algorithm", cfg.algorithm.name)(cfg)
    print("✅ Prepared algorithm")

    # 4. Prepare trainer
    trainer = registry.get("trainer", cfg.algorithm.trainer)(cfg, train_dataset, model, algorithm)
    print("✅ Prepared trainer")

    # 5. Run training
    trainer.run()
    print("✅ Training completed")


if __name__ == "__main__":
    main()
