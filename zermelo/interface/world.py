"""A problem, whole"""

from dataclasses import dataclass
from typing import Any

import jax
from jaxtyping import PRNGKeyArray

from zermelo.interface.domain import ProductDomain
from zermelo.interface.observation import Observation, Readout
from zermelo.interface.prior import Prior
from zermelo.interface.transition import Transition


@dataclass(frozen=True)
class World:
    """A state domain, a prior, a transition and a readout; never subclassed"""

    state_domain: ProductDomain
    """Named parts, one of which is the latent field the agent cannot see"""

    prior: Prior
    """The law an episode's initial state is drawn by, covering the same parts as `state_domain`"""

    transition: Transition[Any]
    """How a state advances"""

    readout: Readout
    """What an agent sees of a state"""

    def reset(self, key: PRNGKeyArray) -> tuple[dict[str, Any], Observation]:
        """A fresh state drawn from the prior, and the first observation of it"""
        k_state, k_read = jax.random.split(key)
        state = self.prior.sample(self.state_domain, k_state)
        return state, Observation(self.readout.reset(k_read, state), self.transition.legal_actions(state))

    def step(self, key: PRNGKeyArray, state: dict[str, Any], action: Any) -> tuple[dict[str, Any], Observation]:
        """One step, and the observation of the state it lands in"""
        k_next, k_read = jax.random.split(key)
        next_state = self.transition(k_next, state, action)
        reading = self.readout.step(k_read, state, action, next_state)
        return next_state, Observation(reading, self.transition.legal_actions(next_state))
