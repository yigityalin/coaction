"""
Implements the Callback protocol.
Callbacks in coaction are the primary way to interact with the experiment
loop. A callback is a class that implements the Callback protocol and
provides methods that are called at different points in the experiment loop.

The Callback protocol is an abstract class that defines the methods that
a callback must implement. The methods are called at different points in
the experiment loop, such as when an experiment begins, when an episode
begins, or when a stage ends.
"""

import typing


class Callback(typing.Protocol):
    """An abstract class representing a callback.

    The callbacks are called at different points in the experiment loop.
    These points include when an experiment begins, when an episode begins,
    when a stage begins, when an experiment ends, when an episode ends,
    and when a stage ends.
    """

    def on_experiment_begin(self, *args, **kwargs):
        """Called when an experiment begins."""

    def on_experiment_end(self, *args, **kwargs):
        """Called when an experiment ends."""

    def on_episode_begin(self, *args, **kwargs):
        """Called when an episode begins."""

    def on_episode_end(self, *args, **kwargs):
        """Called when an episode ends."""

    def on_stage_begin(self, *args, **kwargs):
        """Called when a stage begins."""

    def on_stage_end(self, *args, **kwargs):
        """Called when a stage ends."""
