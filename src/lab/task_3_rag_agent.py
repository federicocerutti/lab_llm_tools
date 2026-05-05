"""
Task 4: RAG Agent with Tool Calling
====================================
Goal: wrap the retrieval strategies from Task 2 as LLM tools and let the
model decide which one (or which combination) to use for each question.

The model will:
  1. Receive a question.
  2. Choose a retrieval tool (or multiple).
  3. Use the tool output as context.
  4. Produce a grounded final answer.

You can refer to example_2_function_calls.py for a template on how to implement
tool calling with litellm.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from textwrap import dedent

import litellm
from rdflib import Graph

from lab.example_3_wikidata import main as export_graph
from lab.settings import Settings
from lab.task_1_sparql import answer
from lab.task_2_hybrid_retrieval import retrieve_embedding, retrieve_sparql, retrieve_text
settings = Settings()
if not settings.litellm_api_key:
    raise SystemExit("Set LITELLM_API_KEY to your LiteLLM key.")

litellm.api_base = settings.litellm_base_url
litellm.api_key = settings.litellm_api_key

GRAPH_PATH = Path("f1_drivers.ttl")

SYSTEM_PROMPT = dedent(
    """
    You are a Formula 1 assistant with access to retrieval tools.
    You must call at least one tool before answering.
    Never answer from general knowledge or guess.
    For direct fact questions about a driver's nationality, team, or where they drove, call sparql_retrieve first.
    If the first tool result is incomplete, you may call text_retrieve or embedding_retrieve next.
    Use only tool output as evidence for the final answer.
    If the tools do not return enough evidence, say so briefly.
    """
).strip()


def _completion_kwargs() -> dict:
    return {
        "model": settings.litellm_model,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "top_k": settings.top_k,
        "presence_penalty": settings.presence_penalty,
        "repetition_penalty": settings.repetition_penalty,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False}
        },
    }


def _build_tool_functions(graph: Graph) -> dict[str, Callable[[str], str]]:
    def sparql_retrieve(question: str) -> str:
        """Retrieve a concise F1 context from the local graph with SPARQL."""

        return retrieve_sparql(graph, question)

    def text_retrieve(question: str) -> str:
        """Retrieve a loose keyword-based F1 context from the local graph."""

        return retrieve_text(graph, question)

    def embedding_retrieve(question: str) -> str:
        """Retrieve the closest F1 driver summaries using embeddings."""

        return retrieve_embedding(graph, question)

    return {
        "sparql_retrieve": sparql_retrieve,
        "text_retrieve": text_retrieve,
        "embedding_retrieve": embedding_retrieve,
    }


def _fallback_tool_name(question: str) -> str:
    lowered = question.lower()
    if any(keyword in lowered for keyword in ("nationality", "team", "drive")):
        return "sparql_retrieve"
    return "text_retrieve"


def run_agent(question: str, graph: Graph) -> str:
    """Run a tool-using RAG agent over the local Formula 1 graph."""

    functions = _build_tool_functions(graph)
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": (
                    fn.__doc__
                    or "Retrieve grounded Formula 1 evidence. Use this before answering."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "A Formula 1 question about a driver, nationality, or team.",
                        }
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
        }
        for name, fn in functions.items()
    ]

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    used_tools = False
    context_chunks: list[str] = []
    while True:
        response = litellm.completion(
            **_completion_kwargs(),
            messages=messages,
            tools=tools,
            stream=False,
        )

        if not isinstance(response, litellm.ModelResponse):
            raise ValueError("Expected a non-streaming model response.")

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls or []
        if not tool_calls:
            if not used_tools:
                fallback_name = _fallback_tool_name(question)
                function_response = str(functions[fallback_name](question))
                print(f"{fallback_name}({{'question': {question!r}}}) -> {function_response}")
                context_chunks.append(function_response)
                messages.append(
                    {
                        "role": "tool",
                        "name": fallback_name,
                        "content": function_response,
                        "tool_call_id": "fallback",
                    }
                )
                used_tools = True
                continue
            break

        used_tools = True
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            raw_args = tool_call.function.arguments or "{}"
            function_args = json.loads(raw_args)
            if not function_name or function_name not in functions:
                continue

            try:
                function_response = str(functions[function_name](**function_args))
            except Exception as exc:
                function_response = str(exc)

            print(f"{function_name}({function_args}) -> {function_response}")
            context_chunks.append(function_response)
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

    combined_context = "\n".join(context_chunks).strip()
    if not combined_context:
        return ""
    return answer(question, combined_context)


def _load_graph() -> Graph:
    if not GRAPH_PATH.exists():
        export_graph()

    graph = Graph()
    graph.parse(GRAPH_PATH, format="turtle")
    return graph


if __name__ == "__main__":
    graph = _load_graph()
    question = "What nationality is Lewis Hamilton?"
    print(f"Question: {question}")
    run_agent(question, graph)
