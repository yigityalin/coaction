"""Create a new project config template.

An example usage is given in create_project.sh.
"""

from collections.abc import Collection
from typing import Final
import argparse
import inspect
import shutil

from coaction import agents, games
from coaction.utils.paths import ProjectPaths


_EXCLUDED_PARAMETERS: Final[frozenset[str]] = frozenset(
    [
        "self",
        "kwargs",
        "paths",
        "config",
        "queue",
        "project_dir",
        "project_name",
        "experiment_name",
        "agent",
        "agent_names",
        "game",
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
    shutil.rmtree(path)


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
    paths = ProjectPaths.from_parent(args.parent_dir, args.project_name)
    cfg_dir = paths.get_project_config_dir()
    cfg_dir.mkdir()

    for experiment_name, agent_types, game_type in zip(
        args.experiment_names,
        args.agent_types,
        args.game_types,
    ):
        file_content = _create_file_content(agent_types, game_type)
        path = paths.with_experiment_name(experiment_name).get_experiment_config_path()
        with open(path, "w", encoding="utf-8") as file:
            file.write(file_content)


def _ensure_one_endline_character(template: str) -> str:
    """Ensure that there is only one endline character at the end of a string.

    Args:
        template (str): The string.
    """
    endline_characters = _count_endline_characters(template)
    if endline_characters > 1:
        return template[: -endline_characters + 1]
    if endline_characters == 0:
        return template + "\n"
    return template


def _create_file_content(agent_types: list[str], game_type: str):
    """Create the content of an experiment config file.

    Args:
        agent_types (list[type]): The types of the agents.
        game_type (type): The type of the game.
    """
    imports_template = _get_imports_template(agent_types)
    agents_template = _get_agents_template(agent_types)
    game_template = _get_game_template(game_type)
    callbacks_template = _get_callbacks_template()
    episodes_and_stages_template = _get_episodes_and_stages_template()
    multiprocessing_template = _get_multiprocessing_template()
    template = (
        imports_template
        + multiprocessing_template
        + episodes_and_stages_template
        + agents_template
        + game_template
        + callbacks_template
    )
    return _ensure_one_endline_character(template)


def _create_custom_agents_file(args: argparse.Namespace):
    """Create the custom agents file.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    imports_template = "import typing\n\n"
    imports_template += "from coaction import agents, games\n\n"
    agent_types = [
        agent_type for agent_types in args.agent_types for agent_type in agent_types
    ]
    custom_agents = _get_custom_agents(agent_types)
    if not custom_agents:
        return
    agents_template = _create_custom_agents_template(custom_agents)
    template = imports_template + agents_template
    template = _ensure_one_endline_character(template)
    path = ProjectPaths.from_parent(
        args.parent_dir, args.project_name
    ).get_project_custom_agents_path()
    with open(path, "w", encoding="utf-8") as file:
        file.write(template)


def _create_custom_agents_template(custom_agents: list[str]) -> str:
    """Return the template for the custom agents.

    Args:
        agent_types (list[str]): The types of the agents.
    """

    template = ""
    for agent_name in custom_agents:
        template += _load_agents_template(agent_name)
    return template


def _load_agents_template(agent_name: str) -> str:
    """Return the template for the agents.

    Args:
        agent_types (list[str]): The types of the agents.
    """
    path = ProjectPaths.get_agent_template_path()
    with open(path, "r", encoding="utf-8") as file:
        template = file.read()
    return template.format_map({"agent_name": agent_name})


def _get_imports_template(agent_types: list[str]) -> str:
    """Return the template for the imports."""
    custom_agents = _get_custom_agents(agent_types)
    template = "from coaction import agents, experiments, games, utils\n\n"
    if custom_agents:
        template += "import _agents\n\n"
    return template + "\n"


def _get_custom_agents(agent_types: list[str]) -> list[str]:
    """Return the set of the types of custom agents.

    Args:
        agent_types (list[str]): The types of the agents.
    """
    agents_ = set()
    for agent_type in agent_types:
        try:
            _get_class_from_string(agent_type)
        except ClassNotFoundError:
            agents_.add(agent_type)
    return sorted(agents_)


def _get_agents_template(agent_types: list[str]) -> str:
    """Return the template for the agents.

    Args:
        agent_types (list[str]): The types of the agents.
    """
    agents_kwargs_template = "\n# Agent configs\n"
    agent_configs_template = "agent_configs = [\n"
    for agent_index, agent_type in enumerate(agent_types):
        agents_kwargs_template += f"# Agent {agent_index}\n"
        try:
            agent_cls = _get_class_from_string(agent_type)
            agent_type_name = f"agents.{agent_cls.__name__}"
            if agent_cls is agents.Agent:
                raise ValueError(
                    "Agent cannot be used as an agent type. "
                    "Please specify a subclass of Agent or a name for your custom type."
                )
        except ClassNotFoundError:
            agent_cls = None
            agent_type_name = f"_agents.{agent_type}"

        agent_type_config_var = f"agent_{agent_index}_type"
        agent_kwargs_config_var = f"agent_{agent_index}_kwargs"

        if agent_cls:
            agents_kwargs_template += f"{agent_type_config_var} = {agent_type_name}\n"
            agents_kwargs_template += _get_param_template(
                agent_kwargs_config_var, agent_cls
            )
        else:
            agents_kwargs_template += f"{agent_type_config_var} = {agent_type_name}  # TODO: implement this class in _agents.py \n"
            agents_kwargs_template += f"{agent_kwargs_config_var} = " + "{\n"
            agents_kwargs_template += (
                "    # TODO: specify all the arguments in your agent implementation\n"
            )
            agents_kwargs_template += "}\n\n"

        agent_configs_template += f"    experiments.config.AgentConfig({agent_type_config_var}, {agent_kwargs_config_var}),\n"
    agent_configs_template += "]\n\n"
    return agents_kwargs_template + agent_configs_template


def _get_game_template(game_type: str) -> str:
    """Return the template for the game.

    Args:
        game_type (type): The type of the game.
    """
    game_cls = _get_class_from_string(game_type)
    game_template = "# Game kwargs\n"
    game_template += f"game_type = games.{game_cls.__name__}\n"
    game_template += _get_param_template("game_kwargs", game_cls, exclude={"self"})
    game_template += (
        "game_config = experiments.config.GameConfig(game_type, game_kwargs)\n\n"
    )
    return game_template


def _get_callbacks_template() -> str:
    """Return the template for the callbacks."""
    cb_template = "# Callbacks\n"
    cb_template += "callback_configs = [\n"
    cb_template += (
        "    # TODO: specify the callbacks using experiments.config.CallbackConfig \n"
    )
    cb_template += "]\n\n"
    return cb_template


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
    try:
        _create_dir(args)
        _create_files(args)
        _create_custom_agents_file(args)
        _create_project_config(args)
    except ClassNotFoundError as exc:
        _rm_dir(args)
        raise exc


if __name__ == "__main__":
    main(_parse_args())
