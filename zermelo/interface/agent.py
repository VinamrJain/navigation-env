"""What a method implements"""

from abc import ABC, abstractmethod

from jaxtyping import PRNGKeyArray

from zermelo.interface.decision import Decision
from zermelo.interface.observation import Observation


class Agent[AgentState](ABC):
    """A method: observations in, decisions out, with `AgentState` threaded between steps"""

    @abstractmethod
    def reset(self, key: PRNGKeyArray, obs: Observation) -> AgentState:
        """The initial `AgentState`, from the first observation"""

    @abstractmethod
    def decide(self, key: PRNGKeyArray, agent_state: AgentState, obs: Observation) -> tuple[AgentState, Decision]:
        """The next `AgentState`, and a decision whose action lies in `obs.legal_actions`"""
