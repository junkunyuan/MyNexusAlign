"""CLI entry: Hydra-configured entry point with environment setup."""

import hydra

@hydra.main(
    config_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs")),
    config_name="main",
    version_base="1.3",
)
@with_env_setup(validator=validate_train_config)
def main(cfg, env):