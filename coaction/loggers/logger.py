"""Implementation of the base logger class."""

from typing import Protocol


class Logger(Protocol):
    """Base class for all loggers."""

    def on_experiment_begin(self, *args, **kwargs):
        """Called when an experiment begins."""

    def on_experiment_end(self, *args, **kwargs):
        """Called when an experiment ends."""

    def on_episode_begin(self, episode: int, *args, **kwargs):
        """Called when an episode begins."""

    def on_episode_end(self, episode: int, *args, **kwargs):
        """Called when an episode ends."""

    def on_stage_begin(self, stage: int, *args, **kwargs):
        """Called when a stage begins."""

    def on_stage_end(self, stage: int, *args, **kwargs):
        """Called when a stage ends."""
