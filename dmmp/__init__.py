"""Self-contained DMMPv3 implementation."""

from .utils.config import AttackConfig, DefenseConfig

__all__ = ["AttackConfig", "DefenseConfig", "run_defense_pipeline", "run_v4_pipeline"]


def __getattr__(name: str):
    if name == "run_defense_pipeline":
        from .diffusion.pipeline import run_defense_pipeline

        return run_defense_pipeline
    if name == "run_v4_pipeline":
        from .diffusion.profile_pipeline import run_v4_pipeline

        return run_v4_pipeline
    raise AttributeError(name)
