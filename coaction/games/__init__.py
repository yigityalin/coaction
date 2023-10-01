"""Implementations of games.

This module contains implementations of games. Each game is implemented as a
class that inherits from the abstract base class :class:`Game`. The
:class:`Game` class defines the interface that all games must implement.

For the sake of simplicity, we mainly focus on two-player matrix/Markov games. These games
are implemented as subclasses of :class:`TwoPlayerMatrixGame` and :class:`TwoPlayerMarkovGame`.
"""

from coaction.games.game import (
    Game,
    MarkovGame,
    MatrixGame,
    StateType,
    ActionType,
    RewardType,
)


__all__ = [
    "Game",
    "MarkovGame",
    "MatrixGame",
    "StateType",
    "ActionType",
    "RewardType",
]
