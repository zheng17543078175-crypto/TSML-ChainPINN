"""Paper implementation; not compatible with legacy tsml3 checkpoints."""
__version__ = "0.3.0"

__all__ = ["ChainPINN", "__version__"]


def __getattr__(name):
    # Reporting/data tools should not initialize a GPU/torch/OpenMP runtime.
    if name == "ChainPINN":
        from .model import ChainPINN
        return ChainPINN
    raise AttributeError(name)
