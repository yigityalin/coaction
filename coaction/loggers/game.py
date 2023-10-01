"""Implementations of loggers for the game module."""

from copy import deepcopy
from typing import Sequence

import numpy.typing as npt

from coaction.games.game import Game, StateType, ActionType, RewardType
from coaction.utils.io import save_object
from coaction.utils.paths import ProjectPaths


class GameLogger:
    """Base class for all game loggers."""

    def __init__(
        self,
        agent_names: list[str],
        paths: ProjectPaths,
        log_each: int = 1,
        save_state_history: bool = False,
        save_action_history: bool = False,
        save_reward_history: bool = True,
    ):
        """Initialize the game logger.

        Args:
            game (mal.Game): The game to log.
            agent_names (list[str]): The names of the agents in the game.
            paths (ProjectPaths): The paths to the project directories.
            log_each (int, optional): The number of stages between each log. Defaults to 1.
            save_state_history (bool, optional): Whether to save the state history.
                Defaults to False.
            save_action_history (bool, optional): Whether to save the action history.
                Defaults to False.
            save_reward_history (bool, optional): Whether to save the reward history.
                Defaults to True.
        """
        if (
            not save_state_history
            and not save_action_history
            and not save_reward_history
        ):
            raise ValueError("At least one history must be saved.")
        self.paths = paths
        self._agent_names = agent_names
        self._log_each = log_each
        self._save_reward_history = save_reward_history
        self._save_state_history = save_state_history
        self._save_action_history = save_action_history
        self._state_history: list[StateType]
        self._action_history: dict[str, list[ActionType]]
        self._reward_history: dict[str, list[RewardType]]

    def on_experiment_begin(self, game: Game):
        """Called when an experiment begins."""
        save_object(
            game.to_json(),
            self.paths.get_game_log_path("game"),
        )

    def on_experiment_end(self, game: Game):
        """Called when an experiment ends."""

    def on_episode_begin(
        self, episode: int, game: Game
    ):  # pylint: disable=unused-argument
        """Called when an episode begins."""
        self._reset()

    def on_episode_end(
        self, episode: int, game: Game
    ):  # pylint: disable=unused-argument
        """Called when an episode ends."""
        self._save_histories(episode)

    def on_stage_begin(self, stage: int, game: Game):
        """Called when a stage begins."""

    def on_stage_end(
        self,
        stage: int,
        game: Game,
        state: StateType,
        actions: Sequence[ActionType],
        rewards: npt.NDArray[RewardType],
    ):  # pylint: disable=unused-argument
        """Called when a stage ends."""
        if stage % self._log_each == 0:
            if self._save_state_history:
                self._state_history.append(state)
            for agent_name, action, reward in zip(self._agent_names, actions, rewards):
                if self._save_action_history:
                    self._action_history[agent_name].append(action)
                if self._save_reward_history:
                    self._reward_history[agent_name].append(reward)

    def _reset(self):
        """Reset the logger."""
        if self._save_state_history:
            self._state_history = []
        if self._save_action_history:
            self._action_history = {name: [] for name in self._agent_names}
        if self._save_reward_history:
            self._reward_history = {name: [] for name in self._agent_names}

    def _save_histories(self, episode: int):
        """Save the histories."""
        if self._save_state_history:
            save_object(
                self._state_history,
                self.paths.get_game_episode_log_path(episode, "states"),
                allow_file_exists=True,
            )
        if self._save_action_history:
            self._save_history(self._action_history, episode, "actions")
        if self._save_reward_history:
            self._save_history(self._reward_history, episode, "rewards")

    def _save_history(self, history: dict[str, list], episode: int, log_name: str):
        """Save a history."""
        for agent_name, hist in history.items():
            save_object(
                hist,
                self.paths.get_agent_episode_log_path(episode, agent_name, log_name),
                allow_file_exists=True,
            )

    def clone(self):
        """Clone the logger."""
        return deepcopy(self)
