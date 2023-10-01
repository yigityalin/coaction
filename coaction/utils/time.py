"""Time utils."""

import time


def get_strftime():
    """Return a string representation of the current time."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
