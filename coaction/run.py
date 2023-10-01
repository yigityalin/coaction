"""Main entry point for the coaction command line tool.

This module is the main entry point for the coaction command line tool.
It parses the command line arguments and runs the project."""

import argparse

from coaction.experiments.config import ProjectConfig
from coaction.experiments.project import Project


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="Path to the project config directory.")
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Run the mal command line tool.

    Args:
        args (argparse.Namespace): The command line arguments.
    """

    config = ProjectConfig.from_dir(args.project)
    project = Project(config)
    project.run()


if __name__ == "__main__":
    main(_parse_args())
