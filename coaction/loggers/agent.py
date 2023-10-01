"""Implementation of the default agent logger."""

from collections import defaultdict
from copy import deepcopy
from typing import Any

from coaction.agents.agent import Agent
from coaction.utils.io import save_object
from coaction.utils.paths import ProjectPaths


class AgentLogger:
    """Default agent logger.

    Logs the agent's logged parameters at the end of steps and writes the logs to a file.
    """

    def __init__(
        self,
        paths: ProjectPaths,
        log_each: int = 1,
    ):
        """Initialize the agent logger.

        Args:
            project_dir (Path | str): The path to the project directory.
            project_name (str): The name of the project.
            experiment_name (str): The name of the experiment.
            log_each (int, optional): The number of stages between each log. Defaults to 1.
        """
        self.paths = paths
        self._log_each: int = log_each
        self._logged_params: list[defaultdict[str, list[Any]]]

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

    def on_episode_end(self, episode: int, agents: list[Agent]):
        """Called when an episode ends."""
        self._save_agent_params(episode, agents)

    def on_stage_begin(self, stage: int, agents: list[Agent]):
        """Called when a stage begins."""

    def on_stage_end(self, stage: int, agents: list[Agent]):
        """Called when a stage ends."""
        if stage % self._log_each == 0:
            for agent, params in zip(agents, self._logged_params):
                for key, value in agent.logged_params.items():
                    params[key].append(deepcopy(value))

    def _reset(self, agents: list[Agent]):
        """Reset the logger."""
        self._logged_params = [defaultdict(list) for _ in agents]

    def _save_agent_params(self, episode: int, agents: list[Agent]):
        """Save the agent's parameters."""
        for agent, params in zip(agents, self._logged_params):
            for key, value in params.items():
                save_object(
                    value,
                    self.paths.get_agent_episode_log_path(episode, agent.name, key),
                    allow_file_exists=True,
                )

    def clone(self):
        """Clone the logger."""
        return deepcopy(self)
