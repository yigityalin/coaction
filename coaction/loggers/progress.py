"""Implementation of Progress logger"""

from coaction.experiments.callbacks import Callback


class ProgressLogger(Callback):
    """Currently a dummy class for logging progress."""

    def __init__(self, *args, **kwargs): ...

    def clone(self) -> "ProgressLogger":
        """Return a clone of the progress logger."""
        return ProgressLogger()
