"""The law an element is drawn by, kept separate from the set it lies in"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from zermelo.interface.domain import BoxDomain, Domain, Enumerable, ProductDomain, Subset


class Prior[X](ABC):
    """A law over the elements of a domain supplied per draw"""

    @abstractmethod
    def sample(self, domain: Domain[X], key: PRNGKeyArray) -> X:
        """One element of `domain`"""


class Uniform(Prior[Any]):
    """Uniform over a finite domain (over the live elements, where it is narrowed) or over a bounded box"""

    def sample(self, domain: Domain[Any], key: PRNGKeyArray) -> Any:
        if isinstance(domain, Subset):  # -1e30 rather than -inf: an all-dead subset must still return an element
            return domain.from_index(jax.random.categorical(key, jnp.where(domain.live, 0.0, -1e30)))
        if isinstance(domain, Enumerable):
            return domain.from_index(jax.random.randint(key, (), 0, domain.size()))
        if isinstance(domain, BoxDomain):
            if not (math.isfinite(domain.low) and math.isfinite(domain.high)):
                raise ValueError(f"an unbounded box has no uniform law: [{domain.low}, {domain.high}]")
            return jax.random.uniform(key, domain.shape, minval=domain.low, maxval=domain.high)
        raise TypeError(f"a {type(domain).__name__} is neither finite nor a bounded box, so Uniform cannot draw from it")


@dataclass(frozen=True)
class ProductPrior(Prior[dict[str, Any]]):
    """One prior per named part, drawn independently"""

    parts: dict[str, Prior]

    def sample(self, domain: Domain[dict[str, Any]], key: PRNGKeyArray) -> dict[str, Any]:
        if not isinstance(domain, ProductDomain) or domain.parts.keys() != self.parts.keys():
            raise ValueError(f"this prior covers parts {sorted(self.parts)}, which do not match the domain it was asked to draw from")
        keys = jax.random.split(key, len(self.parts))
        return {name: prior.sample(domain.parts[name], k) for (name, prior), k in zip(self.parts.items(), keys, strict=True)}
