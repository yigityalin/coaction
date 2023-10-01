"""Implementation of the Fictitious Play algorithm."""

from collections.abc import Callable
from typing import Final, Sequence

import numpy as np
import numpy.typing as npt

from coaction.agents.agent import TwoPlayerAgent
from coaction.agents.fictitious_play import utils as fp_utils
from coaction.games.game import ActionType, RewardType, StateType


class ModelFreeFictitiousPlay(TwoPlayerAgent):
    """Implementation of the Fictitious Play algorithm.

    Parameters
    ----------
    name : str
        The name of the agent.
    seed : int
        The seed for the random number generator.
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
        alpha: Callable[[int], float],
        beta: Callable[[int], float],
        gamma: float,
        initial_Q: None | int | float | np.ndarray = None,
        initial_pi: None | int | float | np.ndarray = None,
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
            initial_Q (None | int | float | np.ndarray): The initial Q matrix.
            initial_pi (None | int | float | np.ndarray): The initial pi matrix.
        """
        super().__init__(name, seed, **kwargs)

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._initial_Q = initial_Q  # pylint: disable=invalid-name
        self._initial_pi = initial_pi

        self._n_states = reward_matrix.shape[0]
        self._n_actions = reward_matrix.shape[1]
        self._n_opponent_actions = reward_matrix.shape[2]

        self.counts = np.zeros_like(reward_matrix)
        self.Q = fp_utils.get_initial_Q(  # pylint: disable=invalid-name
            self._initial_Q, self._n_states, self._n_actions, self._n_opponent_actions
        )
        self.pi = fp_utils.get_initial_pi(  # pylint: disable=invalid-name
            self._initial_pi, self._n_states, self._n_opponent_actions, self.rng
        )
        self.v = np.zeros(  # pylint: disable=invalid-name
            self._n_states, dtype=np.float_
        )
        self._utils = fp_utils.FictitiousPlayUtils(self.Q, self.pi, self.rng)

        self._eye: Final = np.eye(reward_matrix.shape[-1])

    def reset(self):
        super().reset()
        self.counts = np.zeros_like(self.counts)
        self.Q = fp_utils.get_initial_Q(
            self._initial_Q, self._n_states, self._n_actions, self._n_opponent_actions
        )
        self.pi = fp_utils.get_initial_pi(
            self._initial_pi, self._n_states, self._n_opponent_actions, self.rng
        )
        self.v = np.zeros(  # pylint: disable=invalid-name
            self._n_states, dtype=np.float_
        )
        self._utils.reset(self.Q, self.pi)

    def act(self, state: StateType) -> ActionType:
        return self._utils.best_response(state)

    def update(
        self,
        state: StateType,
        actions: Sequence[ActionType],
        reward: RewardType,
        next_state: StateType,
        **kwargs,
    ):
        self.pi[state] += fp_utils.belief_update(
            self.pi[state], self._alpha(self.counts[state].sum()), actions[1]  # type: ignore
        )
        self.Q[state, *actions] += fp_utils.model_free_q_update(
            self.Q[state, *actions],
            self.v[state],
            self._beta(self.counts[state, *actions]),
            self._gamma,
            reward,
        )
        self.v = self._utils.value_function(self.Q, self.pi)
        self.counts[state, *actions] += 1
