from pathlib import Path

import numpy as np

from coaction.utils.io import load_object
from coaction.utils.paths import ProjectPaths


# TODO: Add np.memmap support
class LogLoader:
    """Load the logs from a project."""

    def __init__(self, project_path: Path | str, **memmap_kwargs) -> None:
        """Initialize the log loader.

        Args:
            project_path (Path | str): The path to the project.
        """
        self.project_path = Path(project_path).resolve()
        self.paths = ProjectPaths(self.project_path)
        self._loader = load_object

    @property
    def experiment_names(self):
        """Return the names of the experiments."""
        return [
            path.name
            for path in self.paths.get_project_config_dir().iterdir()
            if path.is_dir()
        ]

    def load_agent_log(self, experiment_name: str, agent_name: str, log_name: str):
        """Load an agent's log.

        Args:
            experiment_name (str): The name of the experiment.
            agent_name (str): The name of the agent.
            log_name (str): The name of the log.
        """
        return self._loader(
            self.paths.with_experiment_name(experiment_name).get_agent_log_path(
                agent_name, log_name
            )
        )

    def load_game_log(self, experiment_name: str, log_name: str):
        """Load a game's log.

        Args:
            experiment_name (str): The name of the experiment.
            log_name (str): The name of the log.
        """
        return self._loader(
            self.paths.with_experiment_name(experiment_name).get_game_log_path(log_name)
        )

    def load_episode_log(
        self,
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
        chunk = 0
        logs = []
        while (
            log_path := self.paths.with_experiment_name(
                experiment_name
            ).get_agent_episode_log_path(episode, agent_name, log_name, chunk)
        ).exists():
            logs.append(self._loader(log_path))
            chunk += 1
        if len(logs) == 0:
            raise ValueError(
                f"Could not find log for episode {episode} of agent {agent_name} in experiment {experiment_name}"
            )
        if len(logs) == 1:
            return logs[0]
        return np.concatenate(logs)
