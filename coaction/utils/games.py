"""Utility functions for games."""

from typing import Sequence

import numpy as np
import numpy.typing as npt
import scipy.optimize as opt

from coaction.games.game import RewardType


def continuation_payoff(
    value_function: npt.NDArray[RewardType],
    reward_matrix: npt.NDArray[RewardType],
    transition_matrix: npt.NDArray[RewardType],
    gamma: float,
) -> np.ndarray:
    """Return the continuation payoff.

    Parameters
    ----------
    value_function : np.ndarray
        The value function.
    reward_matrix : np.ndarray
        The reward matrix.
    transition_matrix : np.ndarray
        The transition matrix.
    gamma : float
        The discount factor.

    Returns
    -------
    np.ndarray
        The continuation payoff.
    """
    return reward_matrix + gamma * transition_matrix @ value_function


def solve_minimax(arr: npt.NDArray[RewardType]):
    """Solve the linear program for the minimax theorem.

    Parameters
    ----------
    arr : npt.NDArray[RewardType]
        The payoff matrix.
    """
    n_rows, n_cols = arr.shape

    # coefficient vector
    c = np.zeros(1 + n_rows)  # pylint: disable=invalid-name
    c[-1] = -1

    # upper bound matrix
    A_ub = np.hstack((-arr.T, np.ones((n_cols, 1))))  # pylint: disable=invalid-name

    # upper bound vector
    b_ub = np.zeros((n_cols, 1))  # pylint: disable=invalid-name

    # equality constraint matrix
    A_eq = np.ones((1, n_rows + 1))  # pylint: disable=invalid-name
    A_eq[0, -1] = 0

    # equality constraint vector
    b_eq = 1

    # bounds
    bounds = [(0, None)] * n_rows + [(None, None)]

    # solve the linear program
    result = opt.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
    return result.x[-1]


def shapley_value(
    T: npt.NDArray[RewardType],  # pylint: disable=invalid-name
    R: npt.NDArray[RewardType],  # pylint: disable=invalid-name
    gamma: float,
    n_iterations=100,
):
    """Return the Shapley value of a Markov game.

    Parameters
    ----------
    T : npt.NDArray[RewardType]
        The transition matrix.
    R : npt.NDArray[RewardType]
        The reward matrix.
    gamma : float
        The discount factor.
    n_iterations : int, optional
        The number of iterations to run, by default 100
    """
    n_states = R.shape[0]
    Q = np.zeros_like(R)  # pylint: disable=invalid-name
    v = np.zeros(n_states)  # pylint: disable=invalid-name

    for _ in range(n_iterations):
        for state in range(n_states):
            v[state] = solve_minimax(Q[state])
        Q = continuation_payoff(v, R, T, gamma)  # pylint: disable=invalid-name
    return v


def generate_zsmg(
    n_states, n_actions: Sequence[int], min_reward=0.0, max_reward=1.0, seed=None
) -> tuple[npt.NDArray[RewardType], npt.NDArray[RewardType]]:
    """Generate a two-player zero-sum Markov game.

    Parameters
    ----------
    n_states : int
        The number of states.
    n_actions : Sequence[int]
        The number of actions for each player.
    min_reward : float, optional
        The minimum reward for agent 0, by default 0.0
    max_reward : float, optional
        The maximum reward for agent 0, by default 1.0
    seed : int, optional
        The seed for the random number generator, by default None

    Returns
    -------
    tuple[npt.NDArray[RewardType], npt.NDArray[RewardType]]
        The reward matrix and the transition matrix.

    """
    rng = np.random.default_rng(seed=seed)
    R = rng.uniform(  # pylint: disable=invalid-name
        min_reward, max_reward, size=(n_states, *n_actions)
    )
    T = rng.dirichlet(  # pylint: disable=invalid-name
        np.ones(n_states), size=(n_states, *n_actions)
    )
    return np.stack((R, -R)), T
