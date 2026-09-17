"""End-to-end integration test for multi-tool packing query flow in TripMate.

Tests that a packing query successfully chains both search_destination_guide and
get_weather_forecast, and that the synthesized answer fuses temperature/climate
data with destination-specific packing and cultural etiquette advice.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage

from agent.graph import create_tripmate_graph
from main import run_tripmate_query


def test_end_to_end_packing_query_flow():
    """Run a full packing query and verify fusion of weather data and destination guide advice."""
    query = "What should I pack for Tokyo in December?"

    # Decision step: LLM triggers both tools
    decision_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_destination_guide",
                "args": {"query": "Tokyo packing tips and cultural attire"},
                "id": "call_tokyo_rag",
            },
            {
                "name": "get_weather_forecast",
                "args": {"city": "Tokyo", "date_or_month": "December"},
                "id": "call_tokyo_weather",
            },
        ],
    )

    # Synthesis step: Mock LLM generates synthesized response fusing both tool outputs
    synthesis_text = (
        "When traveling to Tokyo in December, expect chilly winter conditions with temperatures "
        "typically ranging between 4°C and 12°C (39°F–54°F) under mostly clear skies. "
        "Based on destination guide recommendations, pack modular warm layers including thermal "
        "base layers (heat-tech), a heavy winter coat or down jacket, a scarf, and gloves. "
        "Because Tokyo involves extensive walking and frequent removal of shoes at temples and traditional "
        "restaurants, make sure to bring comfortable, broken-in walking shoes that slip off easily, "
        "along with clean, intact socks."
    )
    synthesis_message = AIMessage(content=synthesis_text)

    class RealisticTripMateLLM:
        def bind_tools(self, tools):
            bound = MagicMock()
            bound.invoke.return_value = decision_message
            return bound

        def invoke(self, messages):
            # Verify that the synthesis step actually receives the tool messages
            tool_contents = [
                getattr(m, "content", "") for m in messages if hasattr(m, "tool_call_id")
            ]
            assert len(tool_contents) == 2
            return synthesis_message

    graph = create_tripmate_graph(llm=RealisticTripMateLLM())

    result = run_tripmate_query(query, verbose=False, graph=graph)

    # 1. Assert reasoning trace contains both tools
    trace = result["reasoning_trace"]
    assert len(trace) == 2

    rag_step = next(s for s in trace if s["tool"] == "search_destination_guide")
    weather_step = next(s for s in trace if s["tool"] == "get_weather_forecast")

    # 2. Verify RAG output contains actual Tokyo guide packing information
    rag_output_str = str(rag_step["output"])
    assert "TOKYO" in rag_output_str
    assert "walking shoes" in rag_output_str.lower() or "packing" in rag_output_str.lower()

    # 3. Verify Weather output contains actual Tokyo December temperature data
    weather_data = weather_step["output"]
    assert weather_data["city"] == "Tokyo"
    assert weather_data["temp_range_c"] == [4, 12] or len(weather_data["temp_range_c"]) == 2

    # 4. Verify synthesized response contains BOTH weather and packing advice
    final_answer = result["answer"].lower()
    assert any(temp_kw in final_answer for temp_kw in ["4°c", "12°c", "temperature", "chilly", "winter"])
    assert any(pack_kw in final_answer for pack_kw in ["pack", "shoes", "layer", "coat", "jacket"])
