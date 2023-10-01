"""Utility functions for independent Q-learning agents."""

from typing import Final

import numpy as np
import numpy.typing as npt

from coaction.games.game import RewardType


LOGGABLE_PARAMS: Final[set] = {"q", "v", "counts"}


def get_initial_q(
    _initial_q: None | int | float | np.ndarray,
    _n_states: int,
    _n_actions: int,
) -> npt.NDArray[RewardType]:
    """Return the initial Q matrix.

    Args:
        _initial_q (None | int | float | np.ndarray): The initial Q matrix.
        _n_states (int): The number of states.
        _n_actions (int): The number of actions.
        _n_opponent_actions (int): The number of opponent actions.
    """
    if _initial_q is None:
        initial_q = np.zeros((_n_states, _n_actions))
    elif isinstance(_initial_q, (int, float)):
        initial_q = np.ones((_n_states, _n_actions)) * _initial_q
    elif isinstance(_initial_q, np.ndarray):
        if _initial_q.shape != (_n_states, _n_actions):
            raise ValueError(
                f"initial_q must have shape {(_n_states, _n_actions)}, "
                f"but has shape {_initial_q.shape}."
            )
        initial_q = _initial_q
    else:
        raise TypeError(
            f"initial_q must be None, int, float, or np.ndarray, "
            f"but is {type(_initial_q)}."
        )
    return initial_q
