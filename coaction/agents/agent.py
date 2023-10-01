"""Implements the base class for agents."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from typing import Any
import pickle

import numpy as np

from coaction.games.game import StateType, ActionType, RewardType


ParameterType = Any


class Agent(ABC):
    """Base class for all agents."""

    def __init__(self, name: str, seed: int, **kwargs):
        _ = kwargs  # ignore unused kwargs
        self._name: str = name
        self._seed: int = seed
        self._rng = None
        self._buffer = {}
        self._logged_params = {}

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def __setattr__(self, key: str, value: Any):
        """Set an attribute.

        If the attribute name starts with an underscore, it is set as a normal attribute.
        Otherwise, it is registered as a parameter.

        In order to register a parameter invisible to loggers, use `register_buffer` instead.
        """
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self.register_parameter(key, value)

    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return self._name

    @property
    def seed(self) -> int:
        """Return the seed."""
        return self._seed

    @property
    def rng(self):
        """Return the random number generator."""
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)
        return self._rng

    @property
    def params(self) -> dict[str, Any]:
        """Return the agent parameters."""
        return {**self._logged_params, **self._buffer}

    @property
    def logged_params(self) -> dict[str, Any]:
        """Return the agent parameters that can be logged."""
        return self._logged_params

    @property
    def buffer(self) -> dict[str, ParameterType]:
        """Return the agent parameters that cannot be logged."""
        return self._buffer

    @abstractmethod
    def reset(self):
        """Reset the agent's parameters."""
        self.reset_rng()

    @abstractmethod
    def act(self, state: StateType) -> ActionType:
        """Return the action to take given the current state.

        Args:
            state (StateType): The current state of the game.
        """

    @abstractmethod
    def update(
        self,
        state: StateType,
        actions: Sequence[ActionType],
        reward: RewardType,
        next_state: StateType,
        **kwargs,
    ):
        """Update the agent's parameters.

        Args:
            state (StateType): The current state of the game.
            actions (Sequence[ActionType]): The actions taken by the agents.
            reward (RewardType): The reward received by the agents.
            next_state (StateType): The next state of the game.
            **kwargs: Keyword arguments.
        """

    def reset_rng(self):
        """Reset the random number generator."""
        self._rng = self.rng.spawn(1)[0]

    def clone(self):
        """Return a deep copy of the agent."""
        return deepcopy(self)

    def save(self, path: str):
        """Save the agent to a file.

        Args:
            path (str): The path to the file.
        """
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str):
        """Load an agent from a file.

        Args:
            path (str): The path to the file.
        """
        with open(path, "rb") as file:
            return pickle.load(file)

    def register_buffer(self, name: str, value: ParameterType):
        """Register a parameter which will not be seen by loggers but will be saved.

        This method is useful for registering parameters that are not tracked by loggers.

        Args:
            name (str): The name of the parameter.
            value (Any): The value of the parameter.
        """
        if name in self._logged_params:
            raise ValueError(
                f"Parameter {name} is already registered as a logged parameter"
            )
        self._buffer[name] = value
        super().__setattr__(name, value)

    def register_parameter(self, name: str, value: ParameterType):
        """Register a parameter visible to loggers.

        This method is useful for registering parameters that are tracked by loggers.

        Args:
            name (str): The name of the parameter.
            value (Any): The value of the parameter.
        """
        if name in self._buffer:
            self.register_buffer(name, value)
        else:
            self._logged_params[name] = value
            super().__setattr__(name, value)


class TwoPlayerAgent(Agent):
    """Base class for agents in two-player games.

    This class is a convenience class for agents in two-player games.
    It does not add any functionality over the base `Agent` class.
    """


class MultiPlayerAgent(Agent):
    """Base class for agents in multi-player games.

    This class is a convenience class for agents in multi-player games.
    It does not add any functionality over the base `Agent` class.
    """
