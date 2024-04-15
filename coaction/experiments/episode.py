"""Implementation of episodes representing a single game between agents."""

from multiprocessing.synchronize import Semaphore
import abc
import multiprocessing as mp

from coaction.agents.agent import Agent
from coaction.experiments.multiprocessing import DummySemaphore
from coaction.games.game import MarkovGame
from coaction.loggers.agent import AgentLogger
from coaction.loggers.game import GameLogger
from coaction.loggers.progress import ProgressLogger


class Episode(abc.ABC, mp.Process):
    """An abstract class representing a single game between agents.

    Every subclass must call the `__init__` method of the superclass in its
    `__init__` method.
    Every subclass must implement the `play` method.
    """

    def __init__(
        self,
        game: MarkovGame,
        agents: list[Agent],
        agent_logger: AgentLogger,
        game_logger: GameLogger,
        progress_logger: ProgressLogger,
        episode: int,
        total_stages: int,
        semaphore: DummySemaphore | Semaphore,
        global_semaphore: DummySemaphore | Semaphore,
    ) -> None:
        """Initialize the episode."""
        mp.Process.__init__(self)

        self.game: MarkovGame = game
        self.agents: list[Agent] = agents
        self.agent_logger: AgentLogger = agent_logger
        self.game_logger: GameLogger = game_logger
        self.progress_logger: ProgressLogger = progress_logger
        self.episode: int = episode
        self.total_stages: int = total_stages
        self.semaphore = semaphore
        self.global_semaphore = global_semaphore

    def run(self) -> None:
        """Run the episode."""
        self.global_semaphore.acquire()
        self.semaphore.acquire()

        self.play(
            game=self.game,
            agents=self.agents,
            agent_logger=self.agent_logger,
            game_logger=self.game_logger,
            progress_logger=self.progress_logger,
            episode=self.episode,
            total_stages=self.total_stages,
        )

        self.semaphore.release()
        self.global_semaphore.release()

    @abc.abstractmethod
    def play(
        self,
        game: MarkovGame,
        agents: list[Agent],
        agent_logger: AgentLogger,
        game_logger: GameLogger,
        progress_logger: ProgressLogger,
        episode: int,
        total_stages: int,
    ) -> None:
        """
        Implements the logic for each episode.
        This method should be overridden by subclasses.

        It is discouraged to use instance variables in this method.
        It is, however, allowed to use instance variables that are set in the
        `__init__` method.

        Args:
            game (MarkovGame): The game to play.
            agents (list[Agent]): The agents in the game.
            agent_logger (AgentLogger): The agent logger.
            game_logger (GameLogger): The game logger.
            progress_logger (ProgressLogger): The progress logger.
            episode (int): The episode number.
            total_stages (int): The total number of stages.
        """


class DefaultEpisode(Episode):
    """A single game between agents."""

    def play(
        self,
        game: MarkovGame,
        agents: list[Agent],
        agent_logger: AgentLogger,
        game_logger: GameLogger,
        progress_logger: ProgressLogger,
        episode: int,
        total_stages: int,
    ) -> None:
        agent_logger.on_episode_begin(episode, agents)
        game_logger.on_episode_begin(episode, game)
        progress_logger.on_episode_begin(episode, agents)

        state = game.reset()
        for agent in self.agents:
            agent.reset()

        for stage in range(1, 1 + total_stages):
            agent_logger.on_stage_begin(stage, agents)
            game_logger.on_stage_begin(stage, game)
            progress_logger.on_stage_begin(stage, agents, episode)

            actions = [agent.act(game.state) for agent in agents]
            next_state, rewards = game.step(actions)

            for i, agent in enumerate(agents):
                actions_view = actions[i:] + actions[:i]
                agent.update(
                    state=state,
                    actions=actions_view,
                    reward=rewards[i],
                    next_state=next_state,
                )

            agent_logger.on_stage_end(stage, agents)
            game_logger.on_stage_end(stage, game, state, actions, rewards)
            progress_logger.on_stage_end(stage, agents=agents, episode=episode)

            state = next_state

        agent_logger.on_episode_end(episode, agents)
        game_logger.on_episode_end(episode, game)
        progress_logger.on_episode_end(episode)
