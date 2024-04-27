"""Implementation of episodes representing a single game between agents."""

from multiprocessing.synchronize import Semaphore
import abc
import copy
import multiprocessing as mp

import numpy as np

from coaction.agents.agent import Agent
from coaction.experiments.callbacks import Callback
from coaction.experiments.config import ExperimentConfig
from coaction.games.game import Game, MarkovGame


class Episode(abc.ABC, mp.Process):
    """An abstract class representing a single game between agents.

    Every subclass must call the `__init__` method of the superclass in its
    `__init__` method.
    Every subclass must implement the `play` method.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        episode: int,
        semaphore: Semaphore,
        global_semaphore: Semaphore,
    ) -> None:
        """Initialize the episode."""
        mp.Process.__init__(self)

        self.config: ExperimentConfig = config
        self.episode: int = episode
        self.semaphore = semaphore
        self.global_semaphore = global_semaphore
        self.game: MarkovGame = None
        self.agents: list[Agent] = None
        self.callbacks: list[Callback] = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(episode={self.episode})"

    def _construct_agents(self, game: MarkovGame) -> list[Agent]:
        """Construct the agents.

        Args:
            game (MarkovGame): The game to play.

        Returns:
            list[Agent]: The agents in the game.
        """
        agent_configs = copy.deepcopy(self.config.agent_configs)
        for agent_idx, (_, agent_kwargs) in enumerate(agent_configs):
            R, T = game.view(agent_idx)  # pylint: disable=invalid-name
            agent_configs[agent_idx].agent_kwargs["seed"] = (
                agent_kwargs["seed"] + self.episode
            )
            agent_configs[agent_idx].agent_kwargs["transition_matrix"] = T
            agent_configs[agent_idx].agent_kwargs["reward_matrix"] = R
        agents = [
            agent_type(**agent_kwargs) for agent_type, agent_kwargs in agent_configs
        ]
        return agents

    def _construct_callbacks(self) -> list[Callback]:
        """Construct the callbacks from configs.

        Returns:
            list[Callback]: The callbacks.
        """
        return [
            callback_type(**callback_kwargs)
            for callback_type, callback_kwargs in self.config.callback_configs
        ]

    def _construct_episode(self) -> tuple[MarkovGame, list[Agent], list[Callback]]:
        """Construct the game and the agents from configs."""
        game: Game = self.config.game_config.game_type(
            **self.config.game_config.game_kwargs
        )
        game: MarkovGame = game.as_markov_game()

        agents: list[Agent] = self._construct_agents(game)
        callbacks: list[Callback] = self._construct_callbacks()
        return game, agents, callbacks

    def run(self) -> None:
        """Run the episode.

        This method acquires the semaphore to run the episode and releases it
        when the episode is done.

        This method should not be overridden. Calling this method will
        call the `play` method. If you want to implement a custom episode,
        you should override the `play` method instead.
        """
        self.global_semaphore.acquire()
        self.semaphore.acquire()

        try:
            self.game, self.agents, self.callbacks = self._construct_episode()

            self.play(
                game=self.game,
                agents=self.agents,
                callbacks=self.callbacks,
                episode=self.episode,
                total_stages=self.config.total_stages,
            )
        finally:
            self.semaphore.release()
            self.global_semaphore.release()

    def play(
        self,
        game: MarkovGame,
        agents: list[Agent],
        callbacks: list[Callback],
        episode: int,
        total_stages: int,
    ) -> None:
        """
        Implements the logic for each episode.
        The default implementation plays the game for the given number of stages
        and calls the callbacks at different points in the episode.

        If you are implementing a custom episode, you should override this method.
        If you only need to implement the logic for each stage, you should override
        the `stage` method instead.

        It is discouraged to use instance variables in this method.
        It is, however, allowed to use instance variables that are set in the
        `__init__` method.

        Args:
            game (MarkovGame): The game to play.
            agents (list[Agent]): The agents in the game.
            callbacks (list[Callback]): The callbacks to call.
            episode (int): The episode number.
            total_stages (int): The total number of stages.
        """
        game.reset()
        for agent in self.agents:
            agent.reset()

        for callback in callbacks:
            callback.on_episode_begin(
                episode=episode,
                game=game,
                agents=agents,
            )

        for stage in range(total_stages):
            for callback in callbacks:
                callback.on_stage_begin(
                    episode=episode,
                    stage=stage,
                    game=game,
                    agents=agents,
                )

            done = self.stage(game=game, agents=agents, stage=stage)

            for callback in callbacks:
                callback.on_stage_end(
                    episode=episode,
                    stage=stage,
                    game=game,
                    agents=agents,
                )

            if done:
                break

        for callback in callbacks:
            callback.on_episode_end(
                episode=episode,
                game=game,
                agents=agents,
            )

    @abc.abstractmethod
    def stage(self, game: MarkovGame, agents: list[Agent], stage: int) -> bool:
        """
        Implements the logic for each stage.
        This method should be overridden by subclasses.

        It is discouraged to use instance variables in this method.
        It is, however, allowed to use instance variables that are set in the
        `__init__` method.

        Important: This method should not call the callbacks unless you are
        implementing a custom episode that does not override the `play` method.

        Things to consider:
        - The game state is accessible through `game.state`.
        - The agents are accessible through `agents`.
        - The stage number is accessible through `stage`.
        - The agents in coaction are stateful, so you should call the `act`
            method of each agent to get their actions, and the `update` method
            to update their state.
        - The game is stateful, so you should call the `step` method to get the
            next state and rewards.
        - The agents in coaction assume that they are the first agents in the
            list of agents. This means that when you call the `update` method,
            you might want to reorder the actions for each agent. Check the
            `DefaultEpisode` class for an example of how to achieve this.

        Args:
            game (MarkovGame): The game to play.
            agents (list[Agent]): The agents in the game.
            stage (int): The stage number.

        Returns:
            bool: Whether the episode is done. If `True`, the episode ends.
        """


class DefaultEpisode(Episode):
    """The default implementation of an episode."""

    def stage(self, game: MarkovGame, agents: list[Agent], stage: int) -> bool:
        """Implements the default logic for each stage.

        The default implementation calls the `act` method of each agent to get
        their actions, calls the `step` method of the game to get the next state
        and rewards, and calls the `update` method of each agent to update their
        state.
        """
        state = game.state
        actions = [agent.act(game.state) for agent in agents]
        next_state, rewards = game.step(actions)

        for i, agent in enumerate(agents):
            actions_view = actions[i:] + actions[:i]
            rewards_view = np.concatenate([rewards[i:], rewards[:i]])
            agent.update(
                state=state,
                actions=actions_view,
                rewards=rewards_view,
                next_state=next_state,
            )

        return False
