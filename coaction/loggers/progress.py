"""Implementation of Progress logger"""

from copy import deepcopy
from io import TextIOWrapper

from coaction.experiments.config import ExperimentConfig
from coaction.loggers.logger import Logger
from coaction.utils.paths import ProjectPaths
from coaction.utils.progress import Progress


# TODO: implement logging to a queue
class ProgressLogger(Logger):
    """Logs progress of an experiment."""

    def __init__(
        self,
        config: ExperimentConfig,
        paths: ProjectPaths,
        log_each: int,
    ):
        """Initialize the progress logger.

        Args:
            config (ExperimentConfig): The configuration for the experiment.
            paths (ProjectPaths): The paths to the project directories.
            log_each (int): The number of episodes to log.
            queue (mp.Queue): The queue to log to.
        """
        super().__init__()
        self._config = config
        self._paths = paths
        self._log_each = log_each

        self._exp_pbar = Progress(
            initial=0, total=config.total_episodes, prefix=f"Experiment {config.name}"
        )
        self._exp_file: TextIOWrapper

        self._episode_pbars: list[Progress] = [  # type: ignore
            None for _ in range(config.total_episodes)
        ]
        self._episode_files: list[TextIOWrapper] = [  # type: ignore
            None for _ in range(config.total_episodes)
        ]

    def on_experiment_begin(self, *args, **kwargs):
        """Called when an experiment begins."""
        self._paths.get_experiment_log_dir().mkdir(parents=True, exist_ok=True)
        self._exp_file = self._paths.get_experiment_progress_log_path().open(
            "w+", encoding="utf-8"
        )
        self._exp_pbar.start()
        self._log_exp_text(f"Experiment {self._config.name} started.")
        self._display_exp(flush=True)

    def on_experiment_end(self, *args, **kwargs):
        """Called when an experiment ends."""
        self._log_exp_text(f"Experiment {self._config.name} ended.")
        self._exp_pbar.update(self._config.total_episodes - self._exp_pbar.current)
        self._display_exp(flush=True)
        self._exp_file.close()

    def on_episode_begin(self, episode: int, *args, **kwargs):
        """Called when an episode begins."""
        self._paths.get_episode_log_dir(episode).mkdir(parents=True, exist_ok=True)
        self._episode_files[episode] = self._paths.get_episode_progress_log_path(
            episode
        ).open("w", encoding="utf-8")
        self._episode_pbars[episode] = Progress(
            initial=0,
            total=self._config.total_stages,
            prefix=f"Episode {episode}",
        )
        self._episode_pbars[episode].start()
        self._log_episode_text(episode, f"Episode {episode} started.")
        self._display_episode(episode, flush=True)

    def on_episode_end(self, episode: int, *args, **kwargs):
        """Called when an episode ends."""
        self._log_episode_text(episode, f"Episode {episode} ended.")
        self._episode_pbars[episode].update(
            self._config.total_stages - self._episode_pbars[episode].current
        )
        self._display_episode(episode, flush=True)
        self._episode_files[episode].close()

    def on_stage_end(self, stage: int, *args, **kwargs):
        """Called when a stage ends."""
        if stage % self._log_each == 0:
            episode = kwargs["episode"]
            self._episode_pbars[episode].update(self._log_each)
            self._display_episode(episode, flush=True)

    def _log_exp_text(self, text: str):
        """Log text to experiment file."""
        self._exp_pbar.add_text(text)

    def _log_episode_text(self, episode: int, text: str):
        """Log text to episode file."""
        self._episode_pbars[episode].add_text(text)

    def _display_exp(self, flush: bool = True):
        display = self._exp_pbar.display()
        self._exp_file.seek(0)
        self._exp_file.write(display)
        if flush:
            self._exp_file.flush()

    def _display_episode(self, episode: int, flush: bool = True):
        display = self._episode_pbars[episode].display()
        self._episode_files[episode].seek(0)
        self._episode_files[episode].write(display)
        if flush:
            self._episode_files[episode].flush()

    def __deepcopy__(self, memo):
        """Return a deepcopy of the logger."""
        logger = ProgressLogger(
            config=self._config,
            paths=self._paths,
            log_each=self._log_each,
        )
        logger._exp_pbar = self._exp_pbar
        logger._exp_file = self._exp_file
        return logger

    def clone(self):
        """Return a clone of the logger."""
        return deepcopy(self)
