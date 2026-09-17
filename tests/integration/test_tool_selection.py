"""Integration tests for dynamic tool selection in TripMate.

Validates that the LangGraph StateGraph invokes the correct tool(s) based on query type:
- RAG-only (e.g., visa requirements)
- Weather-only (e.g., seasonal temperature)
- Multi-tool (e.g., destination packing advice requiring weather context)
- Out-of-scope / Refusal (e.g., booking a flight)
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage

from agent.graph import create_tripmate_graph
from main import run_tripmate_query


class MockRoutingLLM:
    """Mock LLM to simulate dynamic tool-calling decisions for testing graph routing."""

    def __init__(self, ai_message: AIMessage, synthesis_message: AIMessage = None):
        self.ai_message = ai_message
        self.synthesis_message = synthesis_message or AIMessage(
            content="Synthesized travel advisory based on tool outputs."
        )

    def bind_tools(self, tools):
        mock_bound = MagicMock()
        mock_bound.invoke.return_value = self.ai_message
        return mock_bound

    def invoke(self, messages):
        return self.synthesis_message


def test_tool_selection_rag_only():
    """Verify that a visa inquiry calls only search_destination_guide."""
    decision_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_destination_guide",
                "args": {"query": "Bangkok visa and entry requirements"},
                "id": "call_rag_1",
            }
        ],
    )
    mock_llm = MockRoutingLLM(decision_message)
    graph = create_tripmate_graph(llm=mock_llm)

    result = run_tripmate_query(
        "What are the visa requirements for Bangkok?",
        verbose=False,
        graph=graph,
    )

    tools_called = [step["tool"] for step in result["reasoning_trace"]]
    assert tools_called == ["search_destination_guide"]
    assert len(result["reasoning_trace"]) == 1
    assert result["is_out_of_scope"] is False
    # Check that tool output was populated
    assert result["reasoning_trace"][0]["output"] is not None


def test_tool_selection_weather_only():
    """Verify that a weather inquiry calls only get_weather_forecast."""
    decision_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_weather_forecast",
                "args": {"city": "Barcelona", "date_or_month": "May"},
                "id": "call_weather_1",
            }
        ],
    )
    mock_llm = MockRoutingLLM(decision_message)
    graph = create_tripmate_graph(llm=mock_llm)

    result = run_tripmate_query(
        "What is the weather like in Barcelona in May?",
        verbose=False,
        graph=graph,
    )

    tools_called = [step["tool"] for step in result["reasoning_trace"]]
    assert tools_called == ["get_weather_forecast"]
    assert len(result["reasoning_trace"]) == 1
    assert result["is_out_of_scope"] is False
    assert result["reasoning_trace"][0]["output"]["city"] == "Barcelona"


def test_tool_selection_multi_tool():
    """Verify that a packing query dynamically triggers BOTH tools in sequence."""
    decision_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_destination_guide",
                "args": {"query": "Tokyo packing tips December"},
                "id": "call_rag_packing",
            },
            {
                "name": "get_weather_forecast",
                "args": {"city": "Tokyo", "date_or_month": "December"},
                "id": "call_weather_packing",
            },
        ],
    )
    mock_llm = MockRoutingLLM(decision_message)
    graph = create_tripmate_graph(llm=mock_llm)

    result = run_tripmate_query(
        "What should I pack for Tokyo in December?",
        verbose=False,
        graph=graph,
    )

    tools_called = [step["tool"] for step in result["reasoning_trace"]]
    assert "search_destination_guide" in tools_called
    assert "get_weather_forecast" in tools_called
    assert len(result["reasoning_trace"]) == 2
    assert result["is_out_of_scope"] is False


def test_tool_selection_out_of_scope():
    """Verify that a transactional booking query triggers refusal and ZERO tool calls."""
    refusal_text = (
        "I cannot book flights or hotel reservations. TripMate is an informational travel assistant."
    )
    decision_message = AIMessage(
        content=refusal_text,
        tool_calls=[],
    )
    mock_llm = MockRoutingLLM(decision_message)
    graph = create_tripmate_graph(llm=mock_llm)

    result = run_tripmate_query(
        "Please book a flight from London to Reykjavik for next Tuesday.",
        verbose=False,
        graph=graph,
    )

    tools_called = [step["tool"] for step in result["reasoning_trace"]]
    assert len(tools_called) == 0
    assert result["is_out_of_scope"] is True
    assert "cannot book" in result["answer"].lower()
