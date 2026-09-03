"""The control arm: acts at random and learns nothing"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray

from zermelo.interface import Agent, Decision, Enumerable, Function, Observation


@dataclass(frozen=True)
class RandomAgent(Agent[Int[Array, ""]]):
    """Draws a legal act uniformly every step, and asserts the same thing about the field at every one of them"""

    claim: Function
    """What it asserts about the field, unchanged all episode"""

    def reset(self, key: PRNGKeyArray, obs: Observation) -> Int[Array, ""]:
        """A step count, starting at zero"""
        return jnp.zeros((), jnp.int32)

    def decide(self, key: PRNGKeyArray, agent_state: Int[Array, ""], obs: Observation) -> tuple[Int[Array, ""], Decision]:
        if not isinstance(obs.legal_actions, Enumerable):
            raise TypeError(f"acting at random needs an index over the acts, and a {type(obs.legal_actions).__name__} has none")
        return agent_state + 1, Decision(jax.random.choice(key, jnp.asarray(obs.legal_actions.elements())), self.claim)
