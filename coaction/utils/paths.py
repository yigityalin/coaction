"""Path utilities.

This module contains utilities for paths. The :class:`ProjectPaths` class contains
methods for getting paths to various directories and files.
"""

from __future__ import annotations

from pathlib import Path
from typing import final


class ProjectPaths:
    """Path utilities for the coaction project."""

    def __init__(
        self,
        project_dir: Path | str,
        experiment_name: str | None = None,
        increment_run: bool = False,
    ) -> None:
        """Initialize the logger paths.

        Args:
            project_dir (Path | str): The path to the project directory.
            project_name (str): The name of the project.
            experiment_name (str | None): The name of the experiment.
        """
        self.project_dir: Path = Path(project_dir).resolve()
        self.project_name: str = self.project_dir.stem
        self.experiment_name: str | None = experiment_name
        self.run: int = self._get_run()
        if increment_run:
            self._get_project_run_log_dir(self.run).mkdir(parents=True, exist_ok=True)
        elif self.run != 0:
            self.run -= 1

    def __repr__(self) -> str:
        """Return the string representation of the logger paths."""
        return (
            f"{self.__class__.__name__}(project_name={self.project_name}, "
            f"experiment_name={self.experiment_name})"
        )

    def _get_run(self) -> int:
        run = 0
        while self._get_project_run_log_dir(run).exists():
            run += 1
        return run

    def _get_project_run_log_dir(self, run: int) -> Path:
        """Return the path to a project run log directory.

        Args:
            run (int): The run number.
        """
        return self.get_project_logs_dir() / f"run_{run}"

    def cleanup(self) -> None:
        """Remove the project run log directory."""
        self.get_project_run_log_dir().rmdir()
        project_logs_dir = self.get_project_logs_dir()
        if project_logs_dir.exists() and not list(project_logs_dir.iterdir()):
            project_logs_dir.rmdir()
        project_dir = self.get_project_dir()
        if project_dir.exists() and not list(project_dir.iterdir()):
            project_dir.rmdir()

    @classmethod
    def from_parent(
        cls,
        parent_dir: Path | str,
        project_name: str,
        experiment_name: str | None = None,
    ) -> "ProjectPaths":
        """Return the logger paths from a parent directory.

        Args:
            parent_dir (Path | str): The path to the parent directory.
            project_name (str): The name of the project.
            experiment_name (str | None): The name of the experiment.
        """
        project_dir = Path(parent_dir).resolve() / project_name
        return cls(project_dir, experiment_name)

    @final
    @staticmethod
    def get_templates_dir() -> Path:
        """Return the path to the templates directory.

        The templates directory is the directory where all the templates are stored.
        """
        return Path(__file__).parent.parent / "_templates"

    @final
    @staticmethod
    def get_agent_template_path() -> Path:
        """Return the path to the agent template file.

        The agent template file is the file that contains the template for an agent.
        """
        return ProjectPaths.get_templates_dir() / "_agent.txt"

    @final
    @staticmethod
    def get_project_config_template_path() -> Path:
        """Return the path to the project config template file.

        The project config template file is the file containing the template for a project config.
        """
        return ProjectPaths.get_templates_dir() / "_project_config.txt"

    @final
    def get_project_dir(self) -> Path:
        """Return the path to a project directory.

        The project directory is the directory where all the data for a single project are stored.
        """
        return self.project_dir

    @final
    def get_project_config_dir(self) -> Path:
        """Return the path to a project config directory.

        The project config directory is the directory
        where all the configs for a single project are stored.
        """
        return self.get_project_dir() / "configs"

    @final
    def get_project_config_path(self) -> Path:
        """Return the path to a project config file.

        The project config file is the file that contains the configuration for a project.
        """
        return self.get_project_config_dir() / f"{self.project_name}.py"

    @final
    def get_experiment_config_path(self) -> Path:
        """Return the path to an experiment config file.

        The experiment config file is the file that contains the configuration for an experiment.
        """
        return self.get_project_config_dir() / f"{self.experiment_name}.py"

    @final
    def get_experiment_config_paths(self) -> list[Path]:
        """Return the paths to all the experiment config files.

        The experiment config file is the file that contains the configuration for an experiment.
        """
        return [
            path
            for path in self.get_project_config_dir().iterdir()
            if path.is_file()
            and not path.stem.startswith("_")
            and path.suffix == ".py"
            and path.stem != self.project_name
        ]

    @final
    def get_experiment_names(self) -> list[str]:
        """Return the names of all the experiments."""
        return [path.stem for path in self.get_experiment_config_paths()]

    @final
    def get_project_logs_dir(self) -> Path:
        """Return the path to a project logs directory.

        The project log directory is the directory
        where all the logs for all the project runs are stored.
        """
        return self.get_project_dir() / "logs"

    @final
    def get_project_run_log_dir(self) -> Path:
        """Return the path to a project log directory.

        The project log directory is the directory
        where all the logs for a single project run is stored.
        """
        return self._get_project_run_log_dir(self.run)

    @final
    def get_project_run_progress_log_path(self) -> Path:
        """Return the path to a project's progress log file."""
        return self.get_project_run_log_dir() / "progress.log"

    @final
    def get_project_run_config_dir(self) -> Path:
        """Return the path to a project run config directory."""
        return self.get_project_run_log_dir() / "configs"

    @final
    def get_experiment_log_dirs(self) -> list[Path]:
        """Return the paths to all the experiment log directories.

        The experiment log directory is the directory where all the logs for a single experiment.
        """
        return sorted(self.get_project_run_log_dir().iterdir())

    @final
    def get_experiment_log_dir(self) -> Path:
        """Return the path to an experiment directory.

        The experiment directory is the directory where all the logs for a single experiment.
        """
        if self.experiment_name is None:
            raise ValueError("Experiment name must be set.")
        return self.get_project_run_log_dir() / self.experiment_name

    @final
    def get_experiment_progress_log_path(self) -> Path:
        """Return the path to an experiment's progress log file.

        The progress log file is the file that contains the progress of an experiment.
        """
        return self.get_experiment_log_dir() / "progress.log"

    @final
    def get_agents_log_dir(self) -> Path:
        """Return the path to an episode's agents directory.

        The agents directory is the directory where all the logs for agents are stored.
        """
        return self.get_experiment_log_dir() / "agents"

    @final
    def get_agent_log_dir(self, agent_name: str) -> Path:
        """Return the path to an agent directory.

        The agent directory is the directory where all the logs for a single agent are stored.

        Args:
            agent_name (str): The name of the agent.
        """
        return self.get_agents_log_dir() / agent_name

    @final
    def get_agent_log_path(self, agent_name: str, log_name: str) -> Path:
        """Return the path to an agent's log file.

        Args:
            agent_name (str): The name of the agent.
            log_name (str): The name of the log file.
        """
        return self.get_agent_log_dir(agent_name) / log_name

    @final
    def get_game_log_dir(self) -> Path:
        """Return the path to an episode log directory.

        The game directory is the directory where all the data for games are stored.
        """
        return self.get_experiment_log_dir() / "games"

    @final
    def get_game_log_path(self, log_name: str) -> Path:
        """Return the path to an episode's game's log file.

        Args:
            log_name (str): The name of the log file.
        """
        return self.get_game_log_dir() / log_name

    @final
    def get_episodes_log_dir(self) -> Path:
        """Return the path to an episode log directory.

        The episode directory is the directory where all the data for episodes are stored.
        """
        return self.get_experiment_log_dir() / "episodes"

    @final
    def get_episode_log_dir(self, episode: int) -> Path:
        """Return the path to an episode directory.

        The episode directory is the directory where all the logs for a single episode are stored.

        Args:
            episode (int): The episode number.
        """
        return self.get_episodes_log_dir() / str(episode)

    @final
    def get_episode_progress_log_path(self, episode: int) -> Path:
        """Return the path to an experiment's config log file.

        The config log file is the file that contains the configuration for an experiment.
        """
        return self.get_episode_log_dir(episode) / "progress.log"

    @final
    def get_agents_episode_log_dir(self, episode: int) -> Path:
        """Return the path to an episode's agents directory.

        The agents directory is the directory where all the logs for agents are stored.

        Args:
            episode (int): The episode number.
        """
        return self.get_episode_log_dir(episode) / "agents"

    @final
    def get_agent_episode_log_dir(self, episode: int, agent_name: str) -> Path:
        """Return the path to an episode's agent's directory.

        The agent directory is the directory where all the logs for a single agent are stored.

        Args:
            episode (int): The episode number.
            agent_name (str): The name of the agent.
        """
        return self.get_agents_episode_log_dir(episode) / agent_name

    @final
    def get_agent_episode_log_path(
        self, episode: int, agent_name: str, log_name: str, chunk: int | None = None
    ) -> Path:
        """Return the path to an episode's agent's log file.

        Args:
            episode (int): The episode number.
            agent_name (str): The name of the agent.
            log_name (str): The name of the log file.
            chunk (int | None, optional): The chunk number. Defaults to None.
        """
        stem = f"{log_name}_chunk_{chunk or 0}"
        return self.get_agent_episode_log_dir(episode, agent_name) / stem

    @final
    def get_game_episode_log_dir(self, episode: int) -> Path:
        """Return the path to an episode's games directory.

        The games directory is the directory where all the logs for games are stored.

        Args:
            episode (int): The episode number.
        """
        return self.get_episode_log_dir(episode) / "game"

    @final
    def get_game_episode_log_path(
        self, episode: int, log_name: str, chunk: int | None = None
    ) -> Path:
        """Return the path to an episode's game's log file.

        Args:
            episode (int): The episode number.
            log_name (str): The name of the log file.
            chunk (int | None, optional): The chunk number. Defaults to None.
        """
        stem = f"{log_name}_chunk_{chunk or 0}"
        return self.get_game_episode_log_dir(episode) / stem

    @final
    def with_experiment_name(self, experiment_name: str) -> ProjectPaths:
        """Return a new ProjectPaths object with a new experiment name.

        Args:
            experiment_name (str): The name of the experiment.
        """
        return ProjectPaths(self.project_dir, experiment_name)
