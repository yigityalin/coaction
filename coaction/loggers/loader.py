"""Implements the LogLoader class, which loads logs of a project run."""

from pathlib import Path

import numpy as np

from coaction.experiments.config import ExperimentConfig
from coaction.utils.io import find_and_load_object
from coaction.utils.modules import load_module
from coaction.utils.paths import ProjectPaths


# TODO: Add np.memmap support
class LogLoader:
    """Load the logs from a project."""

    def __init__(self, project_path: Path | str) -> None:
        """Initialize the log loader.

        Args:
            project_path (Path | str): The path to the project.
        """
        self.project_path = Path(project_path).resolve()
        self.paths = ProjectPaths(self.project_path)
        self._loader = find_and_load_object

    def get_project_paths(self, run: int, experiment_name: str):
        """Get the paths to the project.

        Args:
            run (int): The run number.
            experiment_name (str): The name of the experiment.
        """
        paths = self.paths
        if run is not None:
            paths = paths.with_run(run)
        if experiment_name is not None:
            paths = paths.with_experiment_name(experiment_name)
        return paths

    def load_agent_log(
        self, run: int, experiment_name: str, agent_name: str, log_name: str
    ):
        """Load an agent's log.

        Args:
            experiment_name (str): The name of the experiment.
            agent_name (str): The name of the agent.
            log_name (str): The name of the log.
        """
        return self._loader(
            self.paths.with_run(run)
            .with_experiment_name(experiment_name)
            .get_agent_log_path(agent_name, log_name)
        )

    def load_game_log(self, run: int, experiment_name: str, log_name: str):
        """Load a game's log.

        Args:
            experiment_name (str): The name of the experiment.
            log_name (str): The name of the log.
        """
        return self._loader(
            self.paths.with_run(run)
            .with_experiment_name(experiment_name)
            .get_game_log_path(log_name)
        )

    def load_episode_log(
        self,
        run: int,
        experiment_name: str,
        episode: int,
        agent_name: str,
        log_name: str,
        chunk: int | None = None,
    ):
        """Load an episode's log.

        Args:
            experiment_name (str): The name of the experiment.
            episode (int): The episode number.
            agent_name (str): The name of the agent.
            log_name (str): The name of the log.
            chunk (int | None, optional): The chunk number. Defaults to None. If None,
                the entire log is loaded. If not None, the chunk is loaded.
        """
        if chunk is not None:
            return self._loader(
                self.paths.with_run(run)
                .with_experiment_name(experiment_name)
                .get_agent_episode_log_path(episode, agent_name, log_name, chunk)
            )
        log_files = (
            self.paths.with_run(run)
            .with_experiment_name(experiment_name)
            .get_agent_episode_log_paths(episode, agent_name, log_name)
        )
        logs = [self._loader(log_file) for log_file in log_files]
        if len(logs) == 0:
            raise ValueError(
                f"Could not find log for episode {episode} of agent {agent_name} in experiment {experiment_name}"
            )
        if len(logs) == 1:
            return logs[0]
        return np.concatenate(logs)

    def load_experiment_logs(
        self, run: int, experiment_name: str, agent_name: str, log_name: str
    ):
        """Load an experiment's log.

        Args:
            experiment_name (str): The name of the experiment.
            log_name (str): The name of the log.
        """
        config = self.load_experiment_config(run, experiment_name)
        logs = [
            self.load_episode_log(run, experiment_name, episode, agent_name, log_name)
            for episode in range(config.total_episodes)
        ]
        return np.array(logs)

    def load_experiment_config(
        self, run: int, experiment_name: str
    ) -> ExperimentConfig:
        """Load the experiment config.

        Args:
            run (int): The run number.
            experiment_name (str): The name of the experiment.
        """
        config_path = (
            self.paths.with_run(run)
            .with_experiment_name(experiment_name)
            .get_project_run_config_path()
        )
        return load_module(config_path)
