"""This module contains the base and ready-to-use classes for agents."""

from coaction.agents.agent import Agent
from coaction.agents import fictitious_play
from coaction.agents.fictitious_play.async_fp import AsynchronousFictitiousPlay
from coaction.agents.fictitious_play.async_sfp import AsynchronousSmoothedFictitiousPlay
from coaction.agents.fictitious_play.sync_fp import SynchronousFictitiousPlay
from coaction.agents.fictitious_play.sync_sfp import SynchronousSmoothedFictitiousPlay
from coaction.agents.fictitious_play.model_free_fp import ModelFreeFictitiousPlay
from coaction.agents.fictitious_play.model_free_sfp import (
    ModelFreeSmoothedFictitiousPlay,
)
from coaction.agents.ind_q.ind_q import IndividualQLearning


__all__ = [
    "Agent",
    "fictitious_play",
    "AsynchronousFictitiousPlay",
    "AsynchronousSmoothedFictitiousPlay",
    "SynchronousFictitiousPlay",
    "SynchronousSmoothedFictitiousPlay",
    "ModelFreeFictitiousPlay",
    "ModelFreeSmoothedFictitiousPlay",
    "IndividualQLearning",
]
