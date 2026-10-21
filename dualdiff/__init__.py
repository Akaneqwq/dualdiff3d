"""DualDiff: dual structure-appearance diffusion priors for novel-view refinement.

The model is imported lazily so that reconstruction-only workflows do not pay
the cost of loading diffusers and its dependencies.
"""

__all__ = ["DualDiff"]


def __getattr__(name):
    if name == "DualDiff":
        from dualdiff.model import DualDiff as _DualDiff

        return _DualDiff
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
