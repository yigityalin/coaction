"""Create a new project config template.

An example usage is given in create_project.sh.
"""

from collections.abc import Collection
from typing import Final
import argparse
import inspect

from coaction import agents, loggers, games
from coaction.loggers import AgentLogger, GameLogger, ProgressLogger
from coaction.utils.paths import ProjectPaths


_EXCLUDED_PARAMETERS: Final[frozenset[str]] = frozenset(
    [
        "self",
        "paths",
        "config",
        "queue",
        "project_dir",
        "project_name",
        "experiment_name",
        "agent",
        "agent_names",
        "game",
        "reward_matrix",
        "transition_matrix",
    ]
)


class ClassNotFoundError(Exception):
    """Raised when a class is not found."""


def _parse_args():
    """Parse command line arguments.

    An example usage is given in create_project.sh.
    """
    parser = argparse.ArgumentParser(
        description="Create a new project config template."
    )
    parser.add_argument(
        "--parent_dir",
        help="Path to the directory in which project config will be initialized.",
    )
    parser.add_argument("--project_name", help="Name of the project.")
    parser.add_argument(
        "--experiment_names", nargs="+", help="Names of the experiments."
    )
    parser.add_argument(
        "--agent_types", nargs="+", action="append", help="Types of the agents."
    )
    parser.add_argument(
        "--game_types",
        nargs="+",
        help="Types of the games.",
    )
    return parser.parse_args()


def _check_args(args: argparse.Namespace):
    """Validate the command line arguments.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    path = ProjectPaths.from_parent(
        args.parent_dir, args.project_name
    ).get_project_dir()
    if path.exists():
        raise ValueError(f"Project {args.project_name} already exists.")
    if len(args.experiment_names) != len(args.agent_types):
        raise ValueError(
            f"Number of experiments ({len(args.experiment_names)}) "
            f"does not match number of agent types ({len(args.agent_types)})."
        )
    if len(args.experiment_names) != len(args.game_types):
        raise ValueError(
            f"Number of experiments ({len(args.experiment_names)}) "
            f"does not match number of game types ({len(args.game_types)})."
        )


def _count_endline_characters(template: str) -> int:
    """Count the number of endline characters at the end of a string.

    Args:
        template (str): The string.
    """
    return len(template) - len(template.rstrip("\n"))


def _get_class_from_string(cls_name: str) -> type:
    """Return the class from a string.

    Args:
        cls_name (str): The name of the class.
    """
    if cls_name in agents.__all__:
        return getattr(agents, cls_name)
    if cls_name in loggers.__all__:
        return getattr(loggers, cls_name)
    if cls_name in games.__all__:
        return getattr(games, cls_name)
    raise ClassNotFoundError(f"Class {cls_name} not found.")


def _get_signature(cls: type, exclude: Collection[str] = _EXCLUDED_PARAMETERS):
    """Return the signature of a class.

    Args:
        cls (type): The class.
    """
    signature = inspect.signature(cls)
    return {k: v for k, v in signature.parameters.items() if k not in exclude}


def _get_param_template(
    param_name: str, cls: type, exclude: Collection[str] = _EXCLUDED_PARAMETERS
):
    """Return the template for a parameter.

    Args:
        param_name (str): The name of the parameter.
        cls_name (str): The name of the class.
    """
    signature = _get_signature(cls, exclude)
    template = param_name + " = {   # type: " + cls.__name__ + "\n"
    for k, val in signature.items():
        template += f"    '{k}': {val.default if val.default is not val.empty else ''},"
        type_ = (
            val.annotation
            if val.annotation is not val.empty
            else "typing.Any (additional keyword parameters)"
        )
        template += f"  # type: {type_} \n"
    return template + "}\n\n"


def _create_dir(args: argparse.Namespace):
    """Create the project directory.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    path = ProjectPaths.from_parent(
        args.parent_dir, args.project_name
    ).get_project_dir()
    path.mkdir(parents=True)


def _rm_dir(args: argparse.Namespace):
    """Remove the project directory.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    path = ProjectPaths.from_parent(
        args.parent_dir, args.project_name
    ).get_project_dir()
    path.rmdir()


def _create_project_config(args: argparse.Namespace):
    """Create the project config file.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    template_path = ProjectPaths.get_project_config_template_path()
    with open(template_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    config_path = ProjectPaths.from_parent(
        args.parent_dir, args.project_name
    ).get_project_config_path()
    with open(config_path, "w", encoding="utf-8") as file:
        file.write(file_content)


def _create_files(args: argparse.Namespace):
    """Create the project config files.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    ProjectPaths.from_parent(
        args.parent_dir, args.project_name
    ).get_project_config_dir().mkdir(parents=True)
    for experiment_name, agent_types, game_type in zip(
        args.experiment_names,
        args.agent_types,
        args.game_types,
    ):
        file_content = _create_file_content(agent_types, game_type)
        path = ProjectPaths.from_parent(
            args.parent_dir, args.project_name, experiment_name
        ).get_experiment_config_path()
        with open(path, "w", encoding="utf-8") as file:
            file.write(file_content)


def _create_file_content(agent_types: list[str], game_type: str):
    """Create the content of an experiment config file.

    Args:
        agent_types (list[type]): The types of the agents.
        game_type (type): The type of the game.
    """
    imports_template = _get_imports_template()
    agents_template = _get_agents_template(agent_types)
    game_template = _get_game_template(game_type)
    loggers_template = _get_loggers_template()
    episodes_and_stages_template = _get_episodes_and_stages_template()
    multiprocessing_template = _get_multiprocessing_template()
    template = (
        imports_template
        + multiprocessing_template
        + episodes_and_stages_template
        + agents_template
        + game_template
        + loggers_template
    )
    # Count the number of endline characters at the end of the template
    # and ensure that there is only one.
    endline_characters = _count_endline_characters(template)
    if endline_characters > 1:
        template = template[: -endline_characters + 1]
    elif endline_characters == 0:
        template += "\n"
    return template


def _load_agents_template(
    agent_index: int, agent_name: str, load_file: bool = True
) -> str:
    """Return the template for the agents.

    Args:
        agent_types (list[str]): The types of the agents.
    """
    if load_file:
        path = ProjectPaths.get_agent_template_path()
        with open(path, "r", encoding="utf-8") as file:
            template = file.read()
        template = template.format_map({"agent_name": agent_name})
    else:
        template = ""
    template += (
        "agent_{agent_index}_kwargs = ".format_map({"agent_index": agent_index}) + "{\n"
    )
    template += (
        "     # TODO: specify any keyword arguments for agent {agent_index} here\n"
    )
    template += "}\n\n"
    return template


def _get_imports_template() -> str:
    template = "import typing\n\n"
    template += "from coaction import agents, loggers, games, utils\n"
    return template + "\n\n"


def _get_agents_template(agent_types: list[str]) -> str:
    """Return the template for the agents.

    Args:
        agent_types (list[str]): The types of the agents.
    """
    agent_type_names = []
    agents_kwargs_template = "\n# Agent kwargs\n"
    for agent_index, agent_type in enumerate(agent_types):
        try:
            agent_cls = _get_class_from_string(agent_type)
            if agent_cls is agents.Agent:
                raise ValueError(
                    "Agent cannot be used as an agent type. "
                    "Please specify a subclass of Agent or a name for your custom type."
                )
            agents_kwargs_template += _get_param_template(
                f"agent_{agent_index}_kwargs", agent_cls
            )
            agent_type_names.append(f"agents.{agent_cls.__name__}")
        except ClassNotFoundError:
            load_file = agent_type not in agent_type_names
            agents_kwargs_template += _load_agents_template(
                agent_index, agent_type, load_file
            )
            agent_type_names.append(agent_type)

    agent_types_template = f'agent_types = [{", ".join(agent_type_names)}] \n'

    agent_kwargs_names = [
        f"agent_{agent_index}_kwargs" for agent_index in range(len(agent_types))
    ]
    agents_kwargs_list_template = (
        f'agent_kwargs = [{", ".join(agent_kwargs_names)}] \n\n'
    )
    return agents_kwargs_template + agent_types_template + agents_kwargs_list_template


def _get_game_template(game_type: str) -> str:
    """Return the template for the game.

    Args:
        game_type (type): The type of the game.
    """
    game_cls = _get_class_from_string(game_type)
    game_template = "# Game kwargs\n"
    game_template += f"game_type = games.{game_cls.__name__}\n"
    game_template += _get_param_template("game_kwargs", game_cls, exclude={"self"})
    return game_template


def _get_loggers_template() -> str:
    """Return the template for the logger.

    Args:
        logger_type (type): The type of the logger.
    """
    logger_template = "# Logger kwargs\n"
    logger_template += _get_param_template("agent_logger_kwargs", AgentLogger)
    logger_template += _get_param_template("game_logger_kwargs", GameLogger)
    logger_template += _get_param_template("progress_logger_kwargs", ProgressLogger)
    return logger_template


def _get_episodes_and_stages_template() -> str:
    """Return the template for the total episodes and stages.

    Args:
        total_episodes (int): The total number of episodes.
        total_stages (int): The total number of stages.
    """
    return "total_episodes = \ntotal_stages = \n\n"


def _get_multiprocessing_template() -> str:
    """Return the template for multiprocessing."""
    return "num_parallel_episodes = None  # 'None' means use all available cores\n\n"


def main(args: argparse.Namespace):
    """Run the coaction command line tool.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    _check_args(args)
    _create_dir(args)
    try:
        _create_files(args)
        _create_project_config(args)
    except ClassNotFoundError as exc:
        _rm_dir(args)
        raise exc


if __name__ == "__main__":
    main(_parse_args())
