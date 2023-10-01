"""A project that contains multiple experiments."""

import multiprocessing as mp
import time

from coaction.experiments.config import ProjectConfig
from coaction.experiments.experiment import Experiment
from coaction.experiments.multiprocessing import DummySemaphore
from coaction.utils.time import get_strftime


class Project:
    """A project that contains multiple experiments."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        config.paths.get_project_run_log_dir().mkdir(parents=True, exist_ok=True)
        self._log_file = config.paths.get_project_run_progress_log_path().open(
            "w+", encoding="utf-8"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name})"

    def run(self):
        """Run the project."""
        self.log_text(f"Project {self.config.name} started.")

        start = time.perf_counter()
        if self.config.global_config.num_parallel_experiments is not None:
            semaphore = mp.Semaphore(self.config.global_config.num_parallel_experiments)
        else:
            semaphore = DummySemaphore()

        if self.config.global_config.num_parallel_episodes is not None:
            global_semaphore = mp.Semaphore(
                self.config.global_config.num_parallel_episodes
            )
        else:
            global_semaphore = DummySemaphore()

        experiments: list[Experiment] = []
        for experiment_config in self.config.experiments:
            experiment = Experiment(
                self.config, experiment_config, semaphore, global_semaphore
            )
            experiments.append(experiment)
            experiment.start()

        for experiment in experiments:
            experiment.join()
        end = time.perf_counter()
        self.log_text(
            f"Project {self.config.name} finished in {end - start:0.4f} seconds."
        )
        self.cleanup()

    def cleanup(self):
        """Clean up the project."""
        self._log_file.close()

    def log_text(self, text):
        """Log text to the log file."""
        self._log_file.write(f"[{get_strftime()}] - {text}\n")
        self._log_file.flush()
