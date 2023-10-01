"""Utilities for fictitious play agents."""

from typing import Final

import numpy as np
import numpy.typing as npt

from coaction.games.game import ActionType, RewardType, StateType
from coaction.utils.math import softmax


LOGGABLE_PARAMS: Final[set] = {"Q", "pi", "counts", "v"}


def get_initial_Q(  # pylint: disable=invalid-name
    _initial_Q: None | int | float | np.ndarray,  # pylint: disable=invalid-name
    _n_states: int,
    _n_actions: int,
    _n_opponent_actions: int,
) -> npt.NDArray[RewardType]:
    """Return the initial Q matrix.

    Args:
        _initial_q (None | int | float | np.ndarray): The initial Q matrix.
        _n_states (int): The number of states.
        _n_actions (int): The number of actions.
        _n_opponent_actions (int): The number of opponent actions.
    """
    if _initial_Q is None:
        initial_Q = np.zeros(  # pylint: disable=invalid-name
            (_n_states, _n_actions, _n_opponent_actions)
        )
    elif isinstance(_initial_Q, (int, float)):
        initial_Q = (  # pylint: disable=invalid-name
            np.ones((_n_states, _n_actions, _n_opponent_actions)) * _initial_Q
        )
    elif isinstance(_initial_Q, np.ndarray):
        if _initial_Q.shape != (
            _n_states,
            _n_actions,
            _n_opponent_actions,
        ):
            raise ValueError(
                f"initial_q must have shape {(_n_states, _n_actions, _n_opponent_actions)}, "
                f"but has shape {_initial_Q.shape}."
            )
        initial_Q = _initial_Q.copy()  # pylint: disable=invalid-name
    else:
        raise TypeError(
            f"initial_q must be None, int, float, or np.ndarray, "
            f"but is {type(_initial_Q)}."
        )
    return initial_Q


def get_initial_pi(
    _initial_pi: None | np.ndarray,
    _n_states: int,
    _n_opponent_actions: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float_]:
    """Return the initial belief matrix.

    Args:
        _initial_pi (None | np.ndarray): The initial belief matrix.
        _n_states (int): The number of states.
        _n_opponent_actions (int): The number of opponent actions.
        rng (np.random.Generator): The random number generator.
    """
    if _initial_pi is None:
        initial_pi = rng.dirichlet(np.ones(_n_opponent_actions), size=_n_states)
    elif isinstance(_initial_pi, np.ndarray):
        if _initial_pi.shape != (_n_states, _n_opponent_actions):
            raise ValueError(
                f"initial_pi must have shape {(_n_states, _n_opponent_actions)}, "
                f"but has shape {_initial_pi.shape}."
            )
        initial_pi = _initial_pi.copy()
    else:
        raise TypeError(
            f"initial_pi must be None or np.ndarray, but is {type(_initial_pi)}."
        )
    return initial_pi


def q_function(
    Q: npt.NDArray[RewardType], pi: npt.NDArray[np.float_]
):  # pylint: disable=invalid-name
    """Return the Q matrix.

    Args:
        Q (npt.NDArray[RewardType]): The Q matrix.
        pi (npt.NDArray[np.float_]): The belief matrix.
    """
    return np.squeeze(Q @ pi[..., np.newaxis], axis=-1)


def belief_update(
    pi: npt.NDArray[np.float_], step_size: float, action: ActionType
):  # pylint: disable=invalid-name
    """Return the belief matrix update.

    Args:
        pi (npt.NDArray[np.float_]): The belief vector for a state.
        step_size (float): The step size.
        action (int): The action taken.
    """
    return step_size * (np.eye(pi.shape[-1])[action] - pi)


def model_free_q_update(
    Q: RewardType,  # pylint: disable=invalid-name
    v: RewardType,  # pylint: disable=invalid-name
    step_size: float,
    gamma: float,
    reward: RewardType,
):
    """Return the Q matrix update.

    Args:
        Q (RewardType): The Q matrix.
        v (RewardType): The value function.
        step_size (float): The step size.
        gamma (float): The discount factor.
        reward (RewardType): The reward received.
    """
    return step_size * (reward + gamma * v - Q)


def model_based_async_q_update(
    Q: npt.NDArray[RewardType],  # pylint: disable=invalid-name
    v: npt.NDArray[RewardType],  # pylint: disable=invalid-name
    step_size: float,
    gamma: float,
    state: StateType,
    reward_matrix: npt.NDArray[RewardType],
    transition_matrix: npt.NDArray[np.float_],
):
    """Return the Q matrix update.

    Args:
        Q (npt.NDArray[RewardType]): The Q matrix.
        v (npt.NDArray[np.float_]): The value function.
        step_size (float): The step size.
        gamma (float): The discount factor.
        state (StateType): The state.
        reward_matrix (npt.NDArray[RewardType]): The reward matrix.
        transition_matrix (npt.NDArray[np.float_]): The transition matrix.
    """
    return step_size * (
        reward_matrix[state] + gamma * transition_matrix[state] @ v - Q[state]
    )


def model_based_sync_q_update(
    Q: npt.NDArray[RewardType],  # pylint: disable=invalid-name
    v: npt.NDArray[RewardType],  # pylint: disable=invalid-name
    step_size: float,
    gamma: float,
    reward_matrix: npt.NDArray[RewardType],
    transition_matrix: npt.NDArray[np.float_],
):
    """Return the Q matrix update.

    Args:
        Q (npt.NDArray[RewardType]): The Q matrix.
        v (npt.NDArray[np.float_]): The value function.
        step_size (float): The step size.
        gamma (float): The discount factor.
        reward_matrix (npt.NDArray[RewardType]): The reward matrix.
        transition_matrix (npt.NDArray[np.float_]): The transition matrix.
    """
    return step_size * (reward_matrix + gamma * transition_matrix @ v - Q)


class FictitiousPlayUtils:
    """Utilities for fictitious play agents."""

    def __init__(
        self,
        Q: npt.NDArray,  # pylint: disable=invalid-name
        pi: npt.NDArray,  # pylint: disable=invalid-name
        rng: np.random.Generator,
    ):
        """Initialize the utilities.

        Args:
            Q (npt.NDArray[RewardType]): The Q matrix.
            pi (npt.NDArray[np.float_]): The belief matrix.
            tau (float): The temperature parameter.
            rng (np.random.Generator): The random number generator.
        """
        self._rng = rng

        self.q = q_function(Q, pi)  # pylint: disable=invalid-name

    def reset(self, Q: npt.NDArray, pi: npt.NDArray):  # pylint: disable=invalid-name
        """Reset the utilities.

        Args:
            Q (npt.NDArray[RewardType]): The Q matrix.
            pi (npt.NDArray[np.float_]): The belief matrix.
        """
        self.q = q_function(Q, pi)

    def value_function(
        self, Q: npt.NDArray[RewardType], pi: npt.NDArray[np.float_]
    ):  # pylint: disable=invalid-name
        """Return the value function.

        Args:
            state (StateType): The state.
            Q (npt.NDArray[RewardType]): The Q matrix.
            pi (npt.NDArray[np.float_]): The belief matrix.
        """
        self.q = q_function(Q, pi)  # pylint: disable=invalid-name
        return self.q.max(axis=-1)

    def best_response(
        self,
        state: StateType,
        *,
        Q: npt.NDArray[RewardType] | None = None,  # pylint: disable=invalid-name
        pi: npt.NDArray[np.float_] | None = None,  # pylint: disable=invalid-name
    ):
        """Return the best response.

        Args:
            Q (npt.NDArray[RewardType]): The Q matrix.
            pi (npt.NDArray[np.float_]): The belief matrix.
        """
        if Q is not None and pi is not None:
            self.q = q_function(Q, pi)
        q = self.q[state]  # pylint: disable=invalid-name
        if np.isclose(q.min(), q.max()):
            return self._rng.choice(len(q))
        return self.q[state].argmax()


class SmoothedFictitiousPlayUtils:
    """Utilities for smoothed fictitious play agents."""

    def __init__(
        self,
        Q: npt.NDArray,  # pylint: disable=invalid-name
        pi: npt.NDArray,  # pylint: disable=invalid-name
        tau: float,
        rng: np.random.Generator,
    ):
        """Initialize the utilities.

        Args:
            Q (npt.NDArray[RewardType]): The Q matrix.
            pi (npt.NDArray[np.float_]): The belief matrix.
            tau (float): The temperature parameter.
            rng (np.random.Generator): The random number generator.
        """
        self._tau = tau
        self._rng = rng

        self.q = q_function(Q, pi)  # pylint: disable=invalid-name
        self.mu = softmax(self.q, self._tau)  # pylint: disable=invalid-name

    def reset(self, Q: npt.NDArray, pi: npt.NDArray):  # pylint: disable=invalid-name
        """Reset the utilities.

        Args:
            Q (npt.NDArray[RewardType]): The Q matrix.
            pi (npt.NDArray[np.float_]): The belief matrix.
        """
        self.q = q_function(Q, pi)
        self.mu = softmax(self.q, self._tau)

    def smoothed_value_function(
        self,
        Q: npt.NDArray[RewardType],  # pylint: disable=invalid-name
        pi: npt.NDArray[np.float_],  # pylint: disable=invalid-name
    ) -> npt.NDArray:
        """Return the smoothed value function.

        Args:
            Q (npt.NDArray[RewardType]): The Q matrix.
            pi (npt.NDArray[np.float_]): The belief matrix.
        """
        self.q = q_function(Q, pi)
        self.mu = softmax(self.q, self._tau)
        return np.sum(self.mu * self.q, axis=-1)

    def smoothed_best_response(
        self,
        state: StateType,
        *,
        Q: npt.NDArray[RewardType] | None = None,  # pylint: disable=invalid-name
        pi: npt.NDArray[np.float_] | None = None,  # pylint: disable=invalid-name
    ) -> ActionType:
        """Return the smoothed best response.

        Args:
            state (StateType): The state.
            Q (npt.NDArray[RewardType]): The Q matrix.
            pi (npt.NDArray[np.float_]): The belief matrix.
        """
        if Q is not None and pi is not None:
            self.q = q_function(Q, pi)
            self.mu = softmax(self.q[state], self._tau)
        mu = self.mu[state]  # pylint: disable=invalid-name
        return self._rng.choice(len(mu), p=mu)  # type: ignore
