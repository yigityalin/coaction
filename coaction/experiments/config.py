"""Configuration for experiments."""

from pathlib import Path
from typing import Any
import dataclasses
import shutil

from coaction.experiments.episode import Episode, DefaultEpisode
from coaction.utils.modules import load_module
from coaction.utils.paths import ProjectPaths


@dataclasses.dataclass
class ExperimentConfig:
    """Configuration for an experiment.

    An experiment contains multiple agents and a game.

    Attributes:
        name (str): The name of the experiment.
        agent_types (list[type]): The types of the agents.
        game_type (type): The type of the game.
        game_kwargs (dict[str, Any]): The keyword arguments for the game.
        agent_kwargs (list[dict[str, Any]]): The keyword arguments for the agents.
        agent_logger_kwargs (dict[str, Any]): The keyword arguments for the
            agent loggers.
        game_logger_kwargs (dict[str, Any]): The keyword arguments for the game
            logger.
        progress_logger_kwargs (dict[str, Any]): The keyword arguments for the
            progress logger.
        total_episodes (int): The total number of episodes.
        total_stages (int): The total number of stages.
        num_parallel_episodes (int): The number of episodes to run in parallel.
    """

    name: str
    agent_types: list[type]
    game_type: type
    game_kwargs: dict[str, Any]
    agent_kwargs: list[dict[str, Any]]
    agent_logger_kwargs: dict[str, Any]
    game_logger_kwargs: dict[str, Any]
    progress_logger_kwargs: dict[str, Any]
    total_episodes: int
    total_stages: int
    num_parallel_episodes: int
    episode_class: type[Episode]

    @classmethod
    def from_py_file(cls, path: Path | str):
        """Return an experiment config from a Python file."""
        module = load_module(path)
        fields = [field for field in dataclasses.fields(cls) if field.name != "name"]
        kwargs = {field.name: getattr(module, field.name, None) for field in fields}
        kwargs["name"] = getattr(module, "name", module.__name__)
        kwargs["episode_class"] = getattr(module, "episode_class", DefaultEpisode)
        config = cls(**kwargs)  # type: ignore
        config._validate()
        return config

    def _validate(self):
        agent_names = [kwargs["name"] for kwargs in self.agent_kwargs]  # type: ignore
        if len(agent_names) != len(set(agent_names)):
            raise ValueError("Agent names must be unique.")

    def to_dict(self):
        """Return a dictionary representation of the config."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class GlobalConfig:
    """Global configuration for a project.

    Attributes:
        num_parallel_experiments (int): The number of experiments to run in
            parallel.
        order (list[str]): The order in which to run the experiments.
    """

    run_name: str
    num_parallel_experiments: int
    num_parallel_episodes: int
    order: list[str]

    @classmethod
    def from_py_file(cls, path: Path | str):
        """Return a global config from a Python file."""
        module = load_module(path)
        fields = [field for field in dataclasses.fields(cls)]
        kwargs = {field.name: getattr(module, field.name) for field in fields}
        return cls(**kwargs)

    def to_dict(self):
        """Return a dictionary representation of the config."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ProjectConfig:
    """Configuration for a project.

    A project contains multiple experiments.

    Attributes:
        project_dir (Path | str): The path to the project directory.
        name (str): The name of the project.
        experiments (list[ExperimentConfig]): The experiments in the project.
    """

    project_dir: Path | str
    name: str
    global_config: GlobalConfig
    experiments: list[ExperimentConfig]
    paths: ProjectPaths

    @classmethod
    def from_dir(cls, path: Path | str):
        """Return a project config from a directory."""
        paths = ProjectPaths(path)
        project_dir = paths.get_project_dir()
        global_config = GlobalConfig.from_py_file(paths.get_project_config_path())
        paths.run_name = global_config.run_name
        if not project_dir.is_dir():
            paths.cleanup()
            raise ValueError(f"Path {path} is not a directory.")
        if not paths.get_project_config_dir().is_dir():
            paths.cleanup()
            raise ValueError(f"Could not find project config directory in {path}.")
        if paths.get_project_run_log_dir().exists():
            raise ValueError(
                f"'{global_config.run_name}' already exists in {paths.get_project_logs_dir()}. "
                "Please choose a different name."
            )
        experiments = sorted(
            [
                ExperimentConfig.from_py_file(experiment_path)
                for experiment_path in paths.get_experiment_config_paths()
            ],
            key=ProjectConfig._get_sort_key(global_config),  # type: ignore
        )
        config = cls(
            project_dir=project_dir,
            name=project_dir.stem,
            global_config=global_config,
            experiments=experiments,
            paths=paths,
        )
        config.copy_config_files()
        return config

    @staticmethod
    def _get_sort_key(config: GlobalConfig):
        """Return the sort key for an experiment."""
        if config.order is None:
            return lambda cfg: cfg.name
        return lambda cfg: config.order.index(cfg.name)

    def copy_config_files(self):
        """Copy the config to the project directory."""
        shutil.copytree(
            self.paths.get_project_config_dir(),
            self.paths.get_project_run_config_dir(),
            ignore=shutil.ignore_patterns("__pycache__"),
        )

    def to_dict(self):
        """Return a dictionary representation of the config."""
        return dataclasses.asdict(self)
