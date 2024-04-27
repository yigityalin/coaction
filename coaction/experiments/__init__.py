"""Main module for experiments.

This module implements the main functionality for running experiments. An
experiment is a collection of episodes, where each episode is a single game
between agents. The experiment loop consists of running multiple episodes
and collecting the results.

The main classes in this module are the `Experiment` class and the `Episode`
class. They implement the parallel execution of experiments and episodes.
"""

from coaction.experiments import callbacks
from coaction.experiments import config
from coaction.experiments import episode
from coaction.experiments import experiment
from coaction.experiments import project

__all__ = ["callbacks", "config", "episode", "experiment", "project"]
