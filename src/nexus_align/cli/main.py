"""CLI entry: Hydra-configured entry point with environment setup."""

import os

import hydra

import nexus_align.models  # noqa: F401  # registers model factories on import
from nexus_align.registry import registry
from nexus_align.engine.setup import with_env_setup


@hydra.main(
    config_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs")),
    config_name="main",
    version_base="1.3",
)
@with_env_setup
def main(cfg, env):
    world_size, rank, device = env.world_size, env.rank, env.device
    cfg_dict = env.cfg_dict

    # --------------------------------------------------------------------------------
    # 1. Prepare dataset
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # 2. Prepare models
    # --------------------------------------------------------------------------------
    model = registry.get("model", cfg.model.name)(
        input_size=cfg.model.resolution // 8,
        num_classes=cfg.model.num_classes,
        use_cfg=cfg.model.cfg_prob > 0,
    ).to(device)
    # --------------------------------------------------------------------------------
    # 3. Prepare algorithms
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # 4. Prepare running
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # 5. Run
    # --------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
