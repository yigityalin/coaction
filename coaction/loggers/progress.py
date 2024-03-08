"""Implementation of Progress logger"""

from copy import deepcopy
from io import TextIOWrapper

from coaction.loggers.logger import Logger


class ProgressLogger(Logger):
    """Currently a dummy class for logging progress."""

    def __init__(self, *args, **kwargs): ...

    def clone(self) -> "ProgressLogger":
        """Return a clone of the progress logger."""
        return ProgressLogger()
