"""Progress bar for logging"""

from io import TextIOWrapper
from typing import Final
import datetime
import time

from coaction.utils.time import get_strftime


_PB_PROGRESS_SYMBOL: Final = "█"
_PB_REMAINING_SYMBOL: Final = "."


def time_to_str(time_delta: float) -> str:
    """Return a string representation of a time delta."""
    return str(datetime.timedelta(seconds=time_delta))


class Progress:
    """A progress bar for logging"""

    def __init__(
        self,
        initial: int,
        total: int,
        prefix: str,
        text_only: bool = False,
        show_bar: bool = True,
        size: int = 60,
    ):
        self._initial = initial
        self._total = total
        self._prefix = prefix
        self._text_only = text_only
        self._show_bar = show_bar
        self._size = size

        self.current: int
        self._start_time: float
        self._current_time: float
        self._file: TextIOWrapper
        self._desc: str = ""
        self._text: str = ""

    @property
    def desc(self):
        """Return the description."""
        return self._desc

    @desc.setter
    def desc(self, value):
        self._desc = value

    @property
    def text(self):
        """Return the text."""
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    def start(self):
        """Start the progress bar."""
        self.current = self._initial
        self._text = ""
        self._start_time = time.perf_counter()
        self._current_time = self._start_time

    def update(self, n: int):  # pylint: disable=invalid-name
        """Update the progress bar.

        Args:
            n (int): The number of steps to update.
        """
        self.current = min(self.current + n, self._total)
        self._current_time = time.perf_counter()

    def add_text(self, text: str):
        """Add text to the progress bar."""
        self._text += f"[{get_strftime()}] - {text}\n"

    def display(self):
        """Return the progress bar as a string."""
        if self._text_only:
            return f"{self._text}\n"
        percentage = (self.current * 100) // self._total
        fill = (self.current * self._size) // self._total
        remaining = self._size - fill
        timedelta = self._current_time - self._start_time
        time_per_step = timedelta / self.current if self.current else float("NaN")
        time_remaining = time_per_step * (self._total - self.current)
        timedelta_str = time_to_str(timedelta)
        time_remaining_str = (
            time_to_str(time_remaining) if self.current else float("NaN")
        )
        prefix = f"{self._prefix}: {percentage}%"
        postfix = f"{self.current}/{self._total} [{timedelta_str}<{time_remaining_str}]"
        if not self._show_bar:
            return f"{prefix} ({postfix}) {self.desc}\n\n{self._text}"
        pb = f"|{_PB_PROGRESS_SYMBOL * fill}{_PB_REMAINING_SYMBOL * remaining}|"  # pylint: disable=invalid-name
        return f"{prefix} {pb} {postfix} {self.desc}\n\n{self._text}"
