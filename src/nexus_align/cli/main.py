"""CLI entry: Hydra-configured entry point with environment setup."""

import hydra
from nexus_align.engine.setup import with_env_setup


@hydra.main(
    config_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs")),
    config_name="main",
    version_base="1.3",
)
@with_env_setup
def main(cfg, env):