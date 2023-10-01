"""Implementation of individual Q-learning agent."""

from collections.abc import Callable
from typing import Sequence

import numpy as np
import numpy.typing as npt

from coaction.agents.agent import TwoPlayerAgent
from coaction.agents.ind_q import utils as ind_q_utils
from coaction.games.game import ActionType, RewardType, StateType
from coaction.utils.math import softmax


class IndividualQLearning(TwoPlayerAgent):
    """Implementation of individual Q-learning agent.

    Parameters
    ----------
    name : str
        The name of the agent.
    seed : int
        The seed for the random number generator.
    alpha : Callable[[int], float]
        Step size for q update.
    beta : Callable[[int], float]
        Step size for v update.
    gamma : float
        Discount factor
    tau : float
        Temperature parameter
    max_step_size : float
        Maximum step size for q update
    """

    def __init__(
        self,
        name: str,
        seed: int,
        reward_matrix: npt.NDArray[RewardType],
        alpha: Callable[[int], float],
        beta: Callable[[int], float],
        gamma: float,
        tau: float,
        max_step_size: float,
        initial_q: None | int | float | np.ndarray = None,
        **kwargs,
    ):
        """Initialize the agent.

        Args:
            name (str): The name of the agent.
            seed (int): The seed for the random number generator.
            reward_matrix (npt.NDArray[RewardType]): The reward matrix.
            alpha (Callable[[int], float]): Step size for q update.
            beta (Callable[[int], float]): Step size for v update.
            gamma (float): Discount factor
            tau (float): Temperature parameter
            max_step_size (float): Maximum step size for q update
            initial_q (None | int | float | np.ndarray): The initial Q matrix.
        """
        super().__init__(name, seed, **kwargs)
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._tau = tau
        self._max_step_size = max_step_size
        self._initial_q = initial_q

        self._n_states = reward_matrix.shape[0]
        self._n_actions = reward_matrix.shape[1]

        self._counts = np.zeros(self._n_states)
        self._q = ind_q_utils.get_initial_q(
            self._initial_q, self._n_states, self._n_actions
        )
        self.v = np.zeros(self._n_states)  # pylint: disable=invalid-name

        self._mu = None

    @property
    def loggable_params(self) -> set[str]:
        return ind_q_utils.LOGGABLE_PARAMS

    def reset(self):
        super().reset()
        self._counts = np.zeros_like(self._counts)
        self._q = ind_q_utils.get_initial_q(
            self._initial_q, self._n_states, self._n_actions
        )
        self.v = np.zeros_like(self.v)

    def act(self, state: StateType) -> ActionType:
        self._mu = softmax(self._q[state], self._tau)
        return self.rng.choice(self._n_actions, p=self._mu)

    def update(
        self,
        state: StateType,
        actions: Sequence[ActionType],
        reward: RewardType,
        next_state: StateType,
        **kwargs,
    ):
        action = actions[0]
        self._q[state, action] += min(
            self._alpha(self._counts[state]) / self._mu[action], self._max_step_size
        ) * (reward + self._gamma * self.v[next_state] - self._q[state, action])
        self.v[state] += self._beta(self._counts[state]) * (
            np.dot(self._q[state], self._mu) - self.v[state]
        )
        self._counts[state] += 1
