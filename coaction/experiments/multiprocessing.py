"""Multiprocessing utilities."""


class DummySemaphore:
    """Dummy semaphore class for cases the number of processes are not limited."""

    def __init__(self):
        """Initialize the dummy semaphore."""

    def __repr__(self) -> str:
        """Return a string representation of the dummy semaphore."""
        return f"{self.__class__.__name__}()"

    def acquire(self):
        """Dummy acquire method."""

    def release(self):
        """Dummy release method."""
