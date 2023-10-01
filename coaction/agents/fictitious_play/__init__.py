"""Fictitious play agents."""

from coaction.agents.fictitious_play.async_fp import AsynchronousFictitiousPlay
from coaction.agents.fictitious_play.async_sfp import AsynchronousSmoothedFictitiousPlay
from coaction.agents.fictitious_play.sync_fp import SynchronousFictitiousPlay
from coaction.agents.fictitious_play.sync_sfp import SynchronousSmoothedFictitiousPlay
from coaction.agents.fictitious_play.model_free_fp import ModelFreeFictitiousPlay
from coaction.agents.fictitious_play.model_free_sfp import (
    ModelFreeSmoothedFictitiousPlay,
)


__all__ = [
    "AsynchronousFictitiousPlay",
    "AsynchronousSmoothedFictitiousPlay",
    "SynchronousFictitiousPlay",
    "SynchronousSmoothedFictitiousPlay",
    "ModelFreeFictitiousPlay",
    "ModelFreeSmoothedFictitiousPlay",
]
