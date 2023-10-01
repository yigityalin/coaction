"""Core classes for games.

This module contains the base classes for games. Each game is implemented as a
class that inherits from the abstract base class :class:`Game`. The
:class:`Game` class defines the interface that all games must implement.

For the sake of simplicity, we mainly focus on matrix/Markov games. These games
are implemented as subclasses of :class:`MatrixGame` and :class:`MarkovGame`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import numpy.typing as npt


StateType = int
ActionType = int
RewardType = np.float_


class Game(ABC):
    """Base class for games.

    Parameters
    ----------
    name : str
        The name of the game.
    seed : int
        The seed for the random number generator.
    metadata : dict
        Metadata for the game.
    """

    def __init__(self, name: str, seed: int, **metadata):
        """Initialize the game.

        Args:
            name (str): The name of the game.
            seed (int): The seed for the random number generator.
            metadata (dict): Metadata for the game.
        """
        self.name: str = name
        self.seed: int = seed
        self.metadata: dict = metadata
        self._rng: np.random.Generator | None = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    @property
    def rng(self) -> np.random.Generator:
        """Return the random number generator."""
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)
        return self._rng

    @property
    @abstractmethod
    def n_agents(self) -> int:
        """Return the number of agents."""

    def clone(self):
        """Return a copy of the game."""
        return deepcopy(self)

    @abstractmethod
    def reset(self):
        """Reset the game."""

    @abstractmethod
    def step(
        self, actions: Sequence[ActionType]
    ) -> npt.NDArray[RewardType] | tuple[StateType, npt.NDArray[RewardType]]:
        """Execute a step in the game."""

    @abstractmethod
    def view(
        self, agent_id: int, only_viewer: bool = True
    ) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:
        """Return a view of the game for the given agent."""

    @abstractmethod
    def to_json(self) -> dict:
        """Return a JSON representation of the game."""

    @classmethod
    @abstractmethod
    def from_json(cls, path: Path):
        """Load the game from a JSON representation."""

    @abstractmethod
    def as_markov_game(self) -> MarkovGame:
        """Return a Markov game with the given transition matrix."""


class MatrixGame(Game):
    """Base class for matrix games."""

    def __init__(self, name: str, reward_matrix: npt.NDArray, seed: int, **metadata):
        super().__init__(name, seed, **metadata)
        self._reward_matrix: npt.NDArray[np.float_] = reward_matrix

    @property
    def n_agents(self) -> int:
        """Return the number of agents."""
        return self.reward_matrix.shape[0]

    @property
    def reward_matrix(self) -> npt.NDArray[RewardType]:
        """Return the reward matrix."""
        return self._reward_matrix

    def reset(self):
        """Reset the game."""
        self._rng = None

    def step(self, actions: Sequence[ActionType]) -> npt.NDArray[RewardType]:
        """Execute a step in the game."""
        return self.reward_matrix[:, *actions]

    def view(
        self,
        agent_id: int,
        only_viewer: bool = True,
    ) -> npt.NDArray:
        """Return a view of the game for the given agent."""
        axes = np.asarray(
            list(range(agent_id, self.n_agents)) + list(range(0, agent_id)), dtype=int
        )
        reward_matrix = self.reward_matrix[axes].transpose(0, *(axes + 1))
        if only_viewer:
            return reward_matrix[0]
        return reward_matrix

    def as_markov_game(self) -> MarkovGame:
        """Return a Markov game with the given transition matrix."""
        reward_matrix = np.expand_dims(self.reward_matrix, axis=1)
        transition_matrix = np.ones((1, *self.reward_matrix.shape[1:], 1))
        return MarkovGame(
            self.name,
            reward_matrix,
            transition_matrix,
            self.seed,
            **self.metadata,
        )

    def to_json(self) -> dict:
        """Return a JSON representation of the game."""
        return {
            "name": self.name,
            "seed": self.seed,
            "metadata": self.metadata,
            "reward_matrix": self.reward_matrix.tolist(),
        }

    @classmethod
    def from_json(cls, path: Path):
        """Load the game from a JSON representation."""
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return cls(
            data["name"],
            np.asarray(data["reward_matrix"]),
            data["seed"],
            **data["metadata"],
        )


class MarkovGame(MatrixGame):
    """Base class for Markov games."""

    def __init__(
        self,
        name: str,
        reward_matrix: npt.NDArray,
        transition_matrix: npt.NDArray[np.float_],
        seed: int,
        **metadata,
    ):
        if reward_matrix.shape[1:] != transition_matrix.shape[:-1]:
            raise ValueError("The reward and transition matrices are not compatible.")
        super().__init__(name, reward_matrix, seed, **metadata)
        self._transition_matrix: npt.NDArray[np.float_] = transition_matrix

    @property
    def state(self) -> StateType:
        """Return the current state."""
        return self._state

    @state.setter
    def state(self, state: StateType):
        """Set the current state."""
        self._state = state

    @property
    def n_states(self) -> int:
        """Return the number of states."""
        return self.transition_matrix.shape[0]

    @property
    def transition_matrix(self) -> npt.NDArray:
        """Return the transition matrix."""
        return self._transition_matrix

    def reset(self):
        """Reset the game."""
        super().reset()
        self.state = self.rng.integers(self.n_states)
        return self.state

    def step(
        self, actions: Sequence[ActionType]
    ) -> npt.NDArray[RewardType] | tuple[StateType, npt.NDArray[RewardType]]:
        """Execute a step in the game."""
        reward = self.reward_matrix[:, self.state, *actions]
        self.state = self.rng.choice(
            self.n_states, p=self.transition_matrix[self.state, *actions]
        )
        return self.state, reward

    def view(
        self, agent_id: int, only_viewer: bool = True
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Return a view of the game for the given agent."""
        axes = np.asarray(
            list(range(agent_id, self.n_agents)) + list(range(0, agent_id)), dtype=int
        )
        reward_matrix = self.reward_matrix[axes].transpose(0, 1, *(axes + 2))
        transition_matrix = self.transition_matrix.transpose(0, *(axes + 1), -1)
        if only_viewer:
            return reward_matrix[0], transition_matrix
        return reward_matrix, transition_matrix

    def to_json(self) -> dict:
        """Return a JSON representation of the game."""
        return {
            "name": self.name,
            "seed": self.seed,
            "metadata": self.metadata,
            "reward_matrix": self.reward_matrix.tolist(),
            "transition_matrix": self.transition_matrix.tolist(),
        }

    @classmethod
    def from_json(cls, path: Path):
        """Load the game from a JSON representation."""
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return cls(
            data["name"],
            np.asarray(data["reward_matrix"]),
            np.asarray(data["transition_matrix"]),
            data["seed"],
            **data["metadata"],
        )

    def as_markov_game(self) -> MarkovGame:
        """Return a Markov game with the given transition matrix."""
        return self.clone()
