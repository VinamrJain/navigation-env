"""What is scored, and when the episode ends"""

from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from zermelo.interface.decision import Decision
from zermelo.interface.domain import Domain


class Objective[ObjectiveState](ABC):
    """True states and a claim in, a reward out, with `ObjectiveState` threaded between steps"""

    @property
    @abstractmethod
    def claim_domain(self) -> Domain:
        """The domain of the claim this objective scores; `ProductDomain({})` where it scores only acting"""

    @abstractmethod
    def reset(self, state: dict[str, Any]) -> ObjectiveState:
        """The initial `ObjectiveState`, from the initial true state"""

    @abstractmethod
    def score(
        self, objective_state: ObjectiveState, state: dict[str, Any], decision: Decision, next_state: dict[str, Any]
    ) -> tuple[ObjectiveState, Float[Array, ""]]:
        """One scalar reward for the step, and the `ObjectiveState` carried forward"""

    def terminated(self, objective_state: ObjectiveState, state: dict[str, Any]) -> Bool[Array, ""]:
        """Whether the episode is over, as a traced boolean; never, unless overridden"""
        return jnp.bool_(False)
