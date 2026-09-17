"""Agent State Schema for TripMate LangGraph."""

from typing import Any, Optional, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class ReasoningStep(TypedDict):
    """Structured record of an agent decision or tool execution step."""
    step: int
    tool: str
    args: dict[str, Any]
    reason: str
    output: Any


class AgentState(TypedDict):
    """The central state maintained throughout the LangGraph workflow."""
    messages: Sequence[BaseMessage]
    query: str
    reasoning_trace: list[ReasoningStep]
    final_answer: Optional[str]
    is_out_of_scope: bool
