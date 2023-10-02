"""Implementation of the default agent logger."""

from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np

from coaction.agents.agent import Agent
from coaction.utils.io import save_object
from coaction.utils.paths import ProjectPaths


# TODO: Add np.memmap support
class AgentLogger:
    """Default agent logger.

    Logs the agent's logged parameters at the end of steps and writes the logs to a file.
    """

    def __init__(
        self,
        paths: ProjectPaths,
        log_each: int = 1,
        save_in_chunks: int | None = None,
    ):
        """Initialize the agent logger.

        Args:
            project_dir (Path | str): The path to the project directory.
            project_name (str): The name of the project.
            experiment_name (str): The name of the experiment.
            log_each (int, optional): The number of stages between each log. Defaults to 1.
            save_in_chunks (int | None, optional): The number of episodes to save in each
                chunk. Defaults to None. If None, all episodes are saved in a single file.
                If not None, the episodes are saved in chunks of size `save_in_chunks`.
        """
        self.paths = paths
        self._log_each: int = log_each
        self._logged_params: list[defaultdict[str, list[Any]]]
        self._save_in_chunks: int | None = save_in_chunks
        if self._save_in_chunks is not None:
            if self._save_in_chunks < 1:
                raise ValueError(
                    f"save_in_chunks must be greater than or equal to 1, got {self._save_in_chunks}"
                )
            self._current_chunk: int = 0
        else:
            self._current_chunk: int | None = None
        self._episode = 0

    def on_experiment_begin(self, agents: list[Agent]):
        """Called when an experiment begins."""
        for agent in agents:
            save_object(
                agent.buffer,
                self.paths.get_agent_log_path(agent.name, "buffer"),
            )

    def on_experiment_end(self, agents: list[Agent]):
        """Called when an experiment ends."""

    def on_episode_begin(
        self, episode: int, agents: list[Agent]
    ):  # pylint: disable=unused-argument
        """Called when an episode begins."""
        self._reset(agents)
        self._episode = episode

    def on_episode_end(
        self, episode: int, agents: list[Agent]
    ):  # pylint: disable=unused-argument
        """Called when an episode ends."""
        self._save_chunk(agents)

    def on_stage_begin(self, stage: int, agents: list[Agent]):
        """Called when a stage begins."""

    def on_stage_end(self, stage: int, agents: list[Agent]):
        """Called when a stage ends."""
        if stage % self._log_each == 0:
            for agent, params in zip(agents, self._logged_params):
                for key, value in agent.logged_params.items():
                    params[key].append(deepcopy(value))
        if self._save_in_chunks is not None:
            h_length = int(np.ceil(stage / self._log_each))
            if h_length % self._save_in_chunks == 0:
                self._save_chunk(agents)

    def _save_chunk(self, agents: list[Agent]):
        self._save_agent_params(self._episode, agents)
        if self._save_in_chunks is not None:
            self._current_chunk += 1
        for params in self._logged_params:
            for key in params.keys():
                params[key].clear()

    def _reset(self, agents: list[Agent]):
        """Reset the logger."""
        self._logged_params = [defaultdict(list) for _ in agents]
        if self._current_chunk is not None:
            self._current_chunk = 0

    def _save_agent_params(self, episode: int, agents: list[Agent]):
        """Save the agent's parameters."""
        for agent, params in zip(agents, self._logged_params):
            for key, value in params.items():
                if not len(value) == 0:
                    save_object(
                        value,
                        self.paths.get_agent_episode_log_path(
                            episode, agent.name, key, self._current_chunk
                        ),
                        allow_file_exists=True,
                    )

    def clone(self):
        """Clone the logger."""
        return deepcopy(self)
