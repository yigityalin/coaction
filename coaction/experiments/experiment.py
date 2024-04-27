"""Experiment class."""

from multiprocessing.synchronize import Semaphore
import multiprocessing as mp
import os

from coaction.experiments.config import ExperimentConfig, ProjectConfig
from coaction.experiments.episode import Episode


class Experiment(mp.Process):
    """A single experiment."""

    def __init__(
        self,
        project_config: ProjectConfig,
        config: ExperimentConfig,
        semaphore: Semaphore,
        global_semaphore: Semaphore,
    ) -> None:
        """Initialize the experiment."""
        super().__init__()
        self.project_config: ProjectConfig = project_config
        self.config: ExperimentConfig = config
        self.paths = project_config.paths.with_experiment_name(config.name)
        self.semaphore = semaphore
        self.global_semaphore = global_semaphore

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name})"

    def run(self):
        """Run the experiment."""
        self.semaphore.acquire()

        experiment_semaphore = mp.Semaphore(
            self.config.num_parallel_episodes or os.cpu_count()
        )

        episodes: list[Episode] = []
        for episode_idx in range(self.config.total_episodes):
            episode = self.config.episode_class(
                config=self.config,
                episode=episode_idx,
                semaphore=experiment_semaphore,
                global_semaphore=self.global_semaphore,
            )
            episodes.append(episode)
            episode.start()

        for episode in episodes:
            episode.join()

        self.semaphore.release()
