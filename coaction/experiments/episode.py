"""Implementation of episodes representing a single game between agents."""

from typing import Any
import multiprocessing as mp

from coaction.agents.agent import Agent
from coaction.experiments.multiprocessing import DummySemaphore
from coaction.games.game import MarkovGame
from coaction.loggers.agent import AgentLogger
from coaction.loggers.game import GameLogger
from coaction.loggers.progress import ProgressLogger


class Episode(mp.Process):
    """A single game between agents."""

    def __init__(
        self,
        game: MarkovGame,
        agents: list[Agent],
        agent_logger: AgentLogger,
        game_logger: GameLogger,
        progress_logger: ProgressLogger,
        episode: int,
        total_stages: int,
        semaphore: DummySemaphore
        | Any,  # TODO: Use multiprocessing.Semaphore instead of Any.
        global_semaphore: DummySemaphore
        | Any,  # TODO: Use multiprocessing.Semaphore instead of Any.
    ) -> None:
        """Initialize the episode."""
        super().__init__()
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
        # Acquire the semaphore to limit the number of parallel episodes.
        self.global_semaphore.acquire()
        self.semaphore.acquire()

        game = self.game
        agents = self.agents
        agent_logger = self.agent_logger
        game_logger = self.game_logger
        progress_logger = self.progress_logger

        agent_logger.on_episode_begin(self.episode, agents)
        game_logger.on_episode_begin(self.episode, game)
        progress_logger.on_episode_begin(self.episode)

        state = game.reset()
        for agent in self.agents:
            agent.reset()

        for stage in range(1, 1 + self.total_stages):
            agent_logger.on_stage_begin(stage, agents)
            game_logger.on_stage_begin(stage, game)
            progress_logger.on_stage_begin(stage, agents=agents, episode=self.episode)

            actions = [agent.act(game.state) for agent in self.agents]
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
            progress_logger.on_stage_end(stage, agents=agents, episode=self.episode)

            state = next_state

        agent_logger.on_episode_end(self.episode, agents)
        game_logger.on_episode_end(self.episode, game)
        progress_logger.on_episode_end(self.episode)

        # Release the semaphore to allow another episode to run.
        self.semaphore.release()
        self.global_semaphore.release()
