from functools import partial
from pathlib import Path

import numpy as np

from coaction.utils.io import load_object
from coaction.utils.paths import ProjectPaths


_memmap_default_kwargs = dict(dtype=np.float32, mode="r")


class LogLoader:
    """Load the logs from a project."""

    def __init__(
        self, project_path: Path | str, use_memmap=False, **memmap_kwargs
    ) -> None:
        """Initialize the log loader.

        Args:
            project_path (Path | str): The path to the project.
            use_memmap (bool, optional): Whether to use memmap. Defaults to False. If
                True, the logs will be loaded as memmaps. Note that this will only work
                if the log format is compatible with memmap.
            **memmap_kwargs: Keyword arguments for np.memmap. Ignored if use_memmap is
                False.
        """
        self.project_path = Path(project_path).resolve()
        self.paths = ProjectPaths(self.project_path)

        if use_memmap:
            memmap_kwargs = {**_memmap_default_kwargs, **memmap_kwargs}
            _memmap = partial(np.memmap, **memmap_kwargs)
            self._loader = _memmap
        else:
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
        self, experiment_name: str, episode: int, agent_name: str, log_name: str
    ):
        """Load an episode's log.

        Args:
            experiment_name (str): The name of the experiment.
            episode (int): The episode number.
            agent_name (str): The name of the agent.
            log_name (str): The name of the log.
        """
        return self._loader(
            self.paths.with_experiment_name(experiment_name).get_agent_episode_log_path(
                episode, agent_name, log_name
            )
        )
