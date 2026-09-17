"""TripMate - AI Travel Assistant CLI and Orchestrator.

Main entry point for running TripMate interactively or via single query.
Displays visible reasoning traces, structured JSON logging, and synthesized answers.
"""

import argparse
import json
import sys
from typing import Any, Optional

# Ensure UTF-8 output encoding on Windows terminals to prevent charmap errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from langchain_core.messages import HumanMessage

from config.settings import settings
from config.logger import setup_logger, JSONLinesFormatter
from agent.graph import create_tripmate_graph
from vectorstore.store import get_vector_store

logger = setup_logger()


def format_trace_box(reasoning_trace: list[dict[str, Any]]) -> str:
    """Format the agent's dynamic reasoning trace into an elegant CLI visual box."""
    if not reasoning_trace:
        return "  [No external tools were invoked for this query]"

    border = "+------------------------------------------------------------------------------+"
    lines = [
        border,
        "| [TRIPMATE AGENT REASONING TRACE]                                             |",
        border,
    ]
    for idx, step in enumerate(reasoning_trace, start=1):
        tool = step.get("tool", "unknown_tool")
        args_str = json.dumps(step.get("args", {}), ensure_ascii=False)
        reason = step.get("reason", "")
        raw_output = step.get("output", "")

        # Format output preview
        if isinstance(raw_output, (dict, list)):
            out_preview = json.dumps(raw_output, ensure_ascii=False)
        else:
            out_preview = str(raw_output)

        if len(out_preview) > 140:
            out_preview = out_preview[:137] + "..."

        lines.append(f"| Step {idx}: Tool [{tool}]")
        lines.append(f"|   Args:      {args_str}")
        lines.append(f"|   Rationale: {reason}")
        lines.append(f"|   Result:    {out_preview}")
        if idx < len(reasoning_trace):
            lines.append("+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +")

    lines.append(border)
    return "\n".join(lines)


def run_tripmate_query(
    query: str,
    verbose: bool = True,
    graph: Optional[Any] = None,
) -> dict[str, Any]:
    """Execute a travel query through the TripMate LangGraph orchestrator.

    Args:
        query: The traveler's natural language question.
        verbose: If True, prints formatted reasoning trace to stdout.
        graph: Optional compiled LangGraph (useful for mock testing).

    Returns:
        A dictionary containing:
        - query: original user query
        - answer: final synthesized response
        - reasoning_trace: list of tool decision/execution steps
        - is_out_of_scope: boolean flag indicating if query was refused
    """
    # 1. Input Validation
    if not query or not query.strip():
        err_msg = (
            "Input query cannot be empty. "
            "Please ask a travel question (e.g., 'What should I pack for Tokyo in December?')."
        )
        return {
            "query": query,
            "answer": err_msg,
            "reasoning_trace": [],
            "is_out_of_scope": False,
        }

    clean_query = query.strip()
    active_graph = graph or create_tripmate_graph()

    initial_state = {
        "query": clean_query,
        "messages": [HumanMessage(content=clean_query)],
        "reasoning_trace": [],
        "final_answer": None,
        "is_out_of_scope": False,
    }

    logger.info("Starting TripMate query execution: '%s'", clean_query)

    try:
        final_state = active_graph.invoke(initial_state)
    except Exception as exc:
        logger.error("Execution failed in TripMate graph: %s", exc, exc_info=True)
        return {
            "query": clean_query,
            "answer": f"TripMate encountered an error while processing your request: {str(exc)}",
            "reasoning_trace": [],
            "is_out_of_scope": False,
        }

    final_answer = final_state.get("final_answer") or "No answer synthesized."
    reasoning_trace = final_state.get("reasoning_trace", [])
    is_out_of_scope = final_state.get("is_out_of_scope", False)

    # Print trace in verbose mode
    if verbose:
        print("\n" + format_trace_box(reasoning_trace))
        print("\n[TripMate Answer]:\n" + final_answer + "\n")

    return {
        "query": clean_query,
        "answer": final_answer,
        "reasoning_trace": reasoning_trace,
        "is_out_of_scope": is_out_of_scope,
    }


def main():
    """CLI REPL entrypoint."""
    parser = argparse.ArgumentParser(
        description="TripMate - Agentic Travel Assistant Core"
    )
    parser.add_argument(
        "-q", "--query", type=str, help="Single query to execute non-interactively"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=settings.TRIPMATE_VERBOSE,
        help="Display dynamic agent reasoning trace",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Reindex destination guide vector store before starting",
    )
    args = parser.parse_args()

    if args.reindex:
        print("Reindexing destination guide data pack...")
        store = get_vector_store(force_reload=True)
        print(f"Indexed {len(store._chunks)} destination guide chunks.")

    # Initialize graph and vector store
    print("===================================================================")
    print("                    TripMate Travel Assistant                      ")
    print("  Supported Destinations: Bangkok, Barcelona, Reykjavik, Tokyo     ")
    print("===================================================================")

    if args.query:
        run_tripmate_query(args.query, verbose=args.verbose)
        return

    print("Type your travel question below (or 'exit' / 'quit' to end):")
    while True:
        try:
            user_input = input("\nTraveler > ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                print("Thank you for using TripMate. Safe travels!")
                break
            if not user_input:
                continue

            run_tripmate_query(user_input, verbose=args.verbose)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting TripMate. Goodbye!")
            break


if __name__ == "__main__":
    main()
