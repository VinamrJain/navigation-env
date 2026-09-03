"""What an agent emits each step"""

from dataclasses import dataclass
from typing import Any

from jax.tree_util import register_dataclass


@register_dataclass
@dataclass(frozen=True)
class Decision:
    """The pair `(action, claim)`"""

    action: Any
    """Read by the world and only by the world"""

    claim: Any
    """Read by the objective and only by the objective"""
