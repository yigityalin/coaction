"""Implementation of the Fictitious Play algorithm."""

from collections.abc import Callable, Collection
from typing import Final, Sequence

import numpy as np
import numpy.typing as npt

from coaction.agents.agent import Agent
from coaction.agents.fictitious_play import utils as fp_utils
from coaction.games.game import ActionType, RewardType, StateType


class ModelFreeFictitiousPlay(Agent):
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
        transition_matrix: npt.NDArray[np.float_],
        reward_matrix: npt.NDArray[RewardType],
        alpha: Callable[[int], float],
        beta: Callable[[int], float],
        gamma: float,
        initial_Q: None | int | float | np.ndarray = None,
        initial_pi: None | int | float | np.ndarray = None,
        logged_params: Collection[str] = None,
        **kwargs,
    ):
        """Initialize the agent.

        Args:
            name (str): The name of the agent.
            seed (int): The seed for the random number generator.
            transition_matrix (npt.NDArray[np.float_]): The transition matrix.
            reward_matrix (npt.NDArray[RewardType]): The reward matrix.
            transition_matrix (npt.NDArray[np.float_]): The transition matrix.
            alpha (Callable[[int], float]): Step size for beliefs.
            beta (Callable[[int], float]): Step size for Q function.
            gamma (float): Discount factor
            initial_Q (None | int | float | np.ndarray): The initial Q matrix.
            initial_pi (None | int | float | np.ndarray): The initial pi matrix.
            logged_params (Collection[str]): The parameters to log.
        """
        super().__init__(
            name, seed, transition_matrix, reward_matrix, logged_params, **kwargs
        )

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._initial_Q = initial_Q  # pylint: disable=invalid-name
        self._initial_pi = initial_pi

        self._n_states = reward_matrix.shape[0]
        self._n_actions = reward_matrix.shape[1]
        self._n_opponent_actions = reward_matrix.shape[2]

        self._counts = np.zeros_like(reward_matrix)
        self._Q = fp_utils.get_initial_Q(  # pylint: disable=invalid-name
            self._initial_Q, self._n_states, self._n_actions, self._n_opponent_actions
        )
        self._pi = fp_utils.get_initial_pi(  # pylint: disable=invalid-name
            self._initial_pi, self._n_states, self._n_opponent_actions, self.rng
        )
        self.v = np.zeros(  # pylint: disable=invalid-name
            self._n_states, dtype=np.float_
        )
        self._utils = fp_utils.FictitiousPlayUtils(self._Q, self._pi, self.rng)

        self._eye: Final = np.eye(reward_matrix.shape[-1])

    @property
    def loggable_params(self) -> set[str]:
        return fp_utils.LOGGABLE_PARAMS

    def reset(self):
        super().reset()
        self._counts = np.zeros_like(self._counts)
        self._Q = fp_utils.get_initial_Q(
            self._initial_Q, self._n_states, self._n_actions, self._n_opponent_actions
        )
        self._pi = fp_utils.get_initial_pi(
            self._initial_pi, self._n_states, self._n_opponent_actions, self.rng
        )
        self.v = np.zeros(  # pylint: disable=invalid-name
            self._n_states, dtype=np.float_
        )
        self._utils.reset(self._Q, self._pi)

    def act(self, state: StateType) -> ActionType:
        return self._utils.best_response(state)

    def update(
        self,
        state: StateType,
        actions: Sequence[ActionType],
        rewards: Sequence[RewardType],
        next_state: StateType,
        **kwargs,
    ):
        reward = rewards[0]
        self._pi[state] += fp_utils.belief_update(
            self._pi[state], self._alpha(self._counts[state].sum()), actions[1]  # type: ignore
        )
        self._Q[state, *actions] += fp_utils.model_free_q_update(
            self._Q[state, *actions],
            self.v[state],
            self._beta(self._counts[state, *actions]),
            self._gamma,
            reward,
        )
        self.v = self._utils.value_function(self._Q, self._pi)
        self._counts[state, *actions] += 1
