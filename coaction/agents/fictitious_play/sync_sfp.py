"""Implementation of the Fictitious Play algorithm."""

from collections.abc import Callable
from typing import Final, Sequence

import numpy as np
import numpy.typing as npt

from coaction.agents.agent import TwoPlayerAgent
from coaction.agents.fictitious_play import utils as fp_utils
from coaction.games.game import ActionType, RewardType, StateType


class SynchronousSmoothedFictitiousPlay(TwoPlayerAgent):
    """Implementation of the Fictitious Play algorithm.

    Parameters
    ----------
    name : str
        The name of the agent.
    seed : int
        The seed for the random number generator.
    reward_matrix : npt.NDArray[RewardType]
        The reward matrix.
    transition_matrix : npt.NDArray[np.float_]
        The transition matrix.
    alpha : Callable[[int], float]
        Step size for beliefs.
    beta : Callable[[int], float]
        Step size for Q function.
    gamma : float
        Discount factor
    tau : float
        Temperature parameter
    """

    def __init__(
        self,
        name: str,
        seed: int,
        reward_matrix: npt.NDArray[RewardType],
        transition_matrix: npt.NDArray[np.float_],
        alpha: Callable[[int], float],
        beta: Callable[[int], float],
        gamma: float,
        tau: float,
        **kwargs,
    ):
        """Initialize the agent.

        Args:
            name (str): The name of the agent.
            seed (int): The seed for the random number generator.
            reward_matrix (npt.NDArray[RewardType]): The reward matrix.
            transition_matrix (npt.NDArray[np.float_]): The transition matrix.
            alpha (Callable[[int], float]): Step size for beliefs.
            beta (Callable[[int], float]): Step size for Q function.
            gamma (float): Discount factor
            tau (float): Temperature parameter
        """
        super().__init__(name, seed, **kwargs)
        self._R = reward_matrix  # pylint: disable=invalid-name
        self._T = transition_matrix  # pylint: disable=invalid-name

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._tau = tau

        self._n_states = reward_matrix.shape[0]
        self._n_actions = reward_matrix.shape[1]
        self._n_opponent_actions = reward_matrix.shape[2]

        self.counts = np.zeros(self._n_states, dtype=np.int_)
        self.Q = reward_matrix.copy()  # pylint: disable=invalid-name
        self.pi = self.rng.dirichlet(  # pylint: disable=invalid-name
            np.ones(self._n_opponent_actions), size=self._n_states
        )
        self.v = np.zeros(  # pylint: disable=invalid-name
            self._n_states, dtype=np.float_
        )

        self._utils = fp_utils.SmoothedFictitiousPlayUtils(
            self.Q, self.pi, self._tau, self.rng
        )

        self._t = 0
        self._eye: Final = np.eye(reward_matrix.shape[-1])

    def reset(self):
        super().reset()
        self.counts = np.zeros(self._n_states, dtype=np.int_)
        self.Q = self._R.copy()  # pylint: disable=invalid-name
        self.pi = self.rng.dirichlet(  # pylint: disable=invalid-name
            np.ones(self._n_opponent_actions), size=self._n_states
        )
        self.v = np.zeros(  # pylint: disable=invalid-name
            self._n_states, dtype=np.float_
        )
        self._t = 0
        self._utils.reset(self.Q, self.pi)

    def act(self, state: StateType) -> ActionType:
        return self._utils.smoothed_best_response(state)

    def update(
        self,
        state: StateType,
        actions: Sequence[ActionType],
        reward: RewardType,
        next_state: StateType,
        **kwargs,
    ):
        self.pi[state] += fp_utils.belief_update(
            self.pi[state],
            self._alpha(self.counts[state]),
            actions[1],  # type: ignore
        )
        self.Q += fp_utils.model_based_sync_q_update(
            self.Q,
            self.v,
            self._beta(self._t),
            self._gamma,
            self._R,
            self._T,
        )
        self.v = self._utils.smoothed_value_function(self.Q, self.pi)
        self.counts[state] += 1
        self._t += 1
