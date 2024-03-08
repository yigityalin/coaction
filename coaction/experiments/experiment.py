"""Experiment class."""

from copy import deepcopy
from multiprocessing.synchronize import Semaphore
from typing import NamedTuple
import inspect
import multiprocessing as mp

import numpy as np

from coaction.agents.agent import Agent
from coaction.experiments.config import ExperimentConfig, ProjectConfig
from coaction.experiments.multiprocessing import DummySemaphore
from coaction.games.game import MarkovGame
from coaction.loggers.agent import AgentLogger
from coaction.loggers.game import GameLogger
from coaction.loggers.progress import ProgressLogger


class _AgentRequirements(NamedTuple):
    """Requirements for an agent."""

    reward_matrix: bool
    transition_matrix: bool


def _inspect_agent(agent_type: type) -> _AgentRequirements:
    """Return the requirements for an agent."""
    signature = inspect.signature(agent_type.__init__)
    reward_matrix = "reward_matrix" in signature.parameters
    transition_matrix = "transition_matrix" in signature.parameters
    return _AgentRequirements(reward_matrix, transition_matrix)


class Experiment(mp.Process):
    """A single experiment."""

    def __init__(
        self,
        project_config: ProjectConfig,
        config: ExperimentConfig,
        semaphore: DummySemaphore | Semaphore,
        global_semaphore: DummySemaphore | Semaphore,
    ) -> None:
        """Initialize the experiment."""
        super().__init__()
        self.project_config: ProjectConfig = project_config
        self.config: ExperimentConfig = config
        self.paths = project_config.paths.with_experiment_name(config.name)
        self.semaphore = semaphore
        self.global_semaphore = global_semaphore
        self._agent_seed_sequences: dict[int, np.random.SeedSequence] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name})"

    def _construct_experiment(
        self,
    ) -> tuple[list[Agent], MarkovGame, AgentLogger, GameLogger, ProgressLogger]:
        """Construct the experiment."""
        game = self._construct_game()
        agents = self._construct_agents(game)
        agent_logger = self._construct_agent_logger()
        game_logger = self._construct_game_logger(agents)
        progress_logger = self._construct_progress_logger()
        return agents, game, agent_logger, game_logger, progress_logger

    def _construct_game(self) -> MarkovGame:
        """Construct the game."""
        return self.config.game_type(**self.config.game_kwargs).as_markov_game()

    def _construct_agents(self, game: MarkovGame) -> list[Agent]:
        """Construct the agents."""
        requirements = [
            _inspect_agent(agent_type) for agent_type in self.config.agent_types
        ]
        agent_kwargs = deepcopy(self.config.agent_kwargs)
        for agent_idx, requirement in enumerate(requirements):
            reward_matrix, transition_matrix = game.view(agent_idx)
            if requirement.reward_matrix:
                agent_kwargs[agent_idx]["reward_matrix"] = reward_matrix
            if requirement.transition_matrix:
                agent_kwargs[agent_idx]["transition_matrix"] = transition_matrix
        agents = [
            agent_type(**agent_kwargs)
            for agent_type, agent_kwargs in zip(self.config.agent_types, agent_kwargs)
        ]
        return agents

    def _construct_agent_logger(self) -> AgentLogger:
        """Construct the agent loggers."""
        agent_logger = AgentLogger(self.paths, **self.config.agent_logger_kwargs)
        return agent_logger

    def _construct_game_logger(self, agents: list[Agent]) -> GameLogger:
        """Construct the game logger."""
        agent_names = [agent.name for agent in agents]
        game_logger = GameLogger(
            agent_names,
            self.paths,
            **self.config.game_logger_kwargs,  # type: ignore
        )
        return game_logger

    def _construct_progress_logger(self) -> ProgressLogger:
        """Construct the progress logger."""
        progress_logger = ProgressLogger(
            self.config, self.paths, **self.config.progress_logger_kwargs
        )
        return progress_logger

    def _update_and_get_seed(self, agent_idx: int) -> int:
        if agent_idx not in self._agent_seed_sequences:
            self._agent_seed_sequences[agent_idx] = np.random.SeedSequence(
                self.config.agent_kwargs[agent_idx]["seed"]
            )
        else:
            self._agent_seed_sequences[agent_idx] = self._agent_seed_sequences[
                agent_idx
            ].spawn(1)[0]
        return self._agent_seed_sequences[agent_idx].generate_state(1)[0]

    def run(self):
        """Run the experiment."""
        # Acquire the semaphore to limit the number of parallel experiments.
        self.semaphore.acquire()

        (
            agents,
            game,
            agent_logger,
            game_logger,
            progress_logger,
        ) = self._construct_experiment()
        agent_logger.on_experiment_begin(agents)
        game_logger.on_experiment_begin(game)
        progress_logger.on_experiment_begin()

        if self.config.num_parallel_episodes is not None:
            semaphore = mp.Semaphore(self.config.num_parallel_episodes)
        else:
            semaphore = DummySemaphore()

        episodes: list[self.config.episode_class] = []
        for episode in range(self.config.total_episodes):
            episode = self.config.episode_class(
                game=game.clone(),
                agents=[
                    agent.clone(seed=self._update_and_get_seed(agent_idx))
                    for agent_idx, agent in enumerate(agents)
                ],
                agent_logger=agent_logger.clone(),
                game_logger=game_logger.clone(),
                progress_logger=progress_logger.clone(),
                episode=episode,
                total_stages=self.config.total_stages,
                semaphore=semaphore,
                global_semaphore=self.global_semaphore,
            )
            episodes.append(episode)
            episode.start()

        for episode in episodes:
            episode.join()

        agent_logger.on_experiment_end(agents)
        game_logger.on_experiment_end(game)
        progress_logger.on_experiment_end()

        # Release the semaphore to allow another experiment to run.
        self.semaphore.release()
