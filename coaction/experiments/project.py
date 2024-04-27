"""A project that contains multiple experiments."""

import multiprocessing as mp
import os

from coaction.experiments.config import ProjectConfig
from coaction.experiments.experiment import Experiment


class Project:
    """A project that contains multiple experiments."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        config.paths.get_project_run_log_dir().mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name})"

    def run(self):
        """Run the project."""
        max_processes = os.cpu_count()
        semaphore = mp.Semaphore(
            self.config.global_config.num_parallel_experiments or max_processes
        )
        global_semaphore = mp.Semaphore(
            self.config.global_config.num_parallel_episodes or max_processes
        )

        experiments: list[Experiment] = []
        for experiment_config in self.config.experiments:
            experiment = Experiment(
                self.config, experiment_config, semaphore, global_semaphore
            )
            experiments.append(experiment)
            experiment.start()

        for experiment in experiments:
            experiment.join()
