"""
Task 2: Hybrid Retrieval over RDF store
=======================================
Goal: implement different retrieval strategies over the local rdflib graph,
then compare their outputs side by side for the same question.

Retrieval strategies
---
A) SPARQL retrieval: precise, structured, uses a query tailored to the local graph
B) Text retrieval: keyword-based search over labels and literal values
C) Embedding retrieval: semantic similarity between the question and driver summaries

Instructions
---
1. Load the RDF graph you built from the Wikidata export.
2. Complete retrieve_sparql(), retrieve_text(), retrieve_embedding().
3. Run compare() for at least two questions and note differences.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

from rdflib import Graph, Namespace
from rdflib.namespace import RDFS

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

from lab.example_3_wikidata import main as export_graph
from lab.task_1_sparql import answer

LAB = Namespace("http://lab.unibs.it/ontology/")
GRAPH_PATH = Path("f1_drivers.ttl")

STOPWORDS = {
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "of",
    "for",
    "to",
    "in",
    "on",
    "and",
    "or",
    "did",
    "do",
    "does",
    "drive",
    "driven",
    "drive",
    "driver",
    "drivers",
    "team",
    "teams",
    "nationality",
    "nationalities",
    "from",
    "with",
    "by",
}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""

    if not a or not b:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@lru_cache(maxsize=1)
def _embedding_model() -> SentenceTransformer:
    """Load and cache the local sentence-transformer model."""

    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed")
    return SentenceTransformer("all-MiniLM-L6-v2")


def _get_embedding(text: str) -> list[float]:
    """Get an embedding vector for a text string.

    Falls back to an empty vector on failure.
    """

    try:
        embeddings = _embedding_model().encode([text], show_progress_bar=False)
    except Exception:
        return []
    return embeddings[0].tolist()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _question_terms(question: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", question)
    terms = [word.lower() for word in words if word.lower() not in STOPWORDS and len(word) > 2]
    phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", question)
    ordered: list[str] = []
    for term in phrases + terms:
        normalized = _normalize(term)
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    return ordered


def _best_label_match(graph: Graph, question: str) -> str | None:
    labels = sorted(
        {str(label) for label in graph.objects(None, RDFS.label)},
        key=len,
        reverse=True,
    )
    normalized_question = _normalize(question)
    for label in labels:
        if _normalize(label) in normalized_question:
            return label
    return None


def _driver_records(graph: Graph) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen_subjects: set[str] = set()

    for subject in sorted({s for s, _, _ in graph.triples((None, RDFS.label, None))}, key=str):
        subject_id = str(subject)
        if subject_id in seen_subjects:
            continue
        seen_subjects.add(subject_id)

        labels = [str(label) for label in graph.objects(subject, RDFS.label)]
        if not labels:
            continue

        nationalities = sorted({str(value) for value in graph.objects(subject, LAB.nationality)})
        teams = sorted({str(value) for value in graph.objects(subject, LAB.team)})

        records.append(
            {
                "subject": subject_id,
                "driver": labels[0],
                "nationalities": nationalities,
                "teams": teams,
            }
        )

    return records


def _record_to_sentence(record: dict[str, object]) -> str:
    driver = str(record.get("driver", "")).strip()
    nationalities = [str(value) for value in record.get("nationalities", []) if str(value).strip()]
    teams = [str(value) for value in record.get("teams", []) if str(value).strip()]

    parts = [driver] if driver else []
    if nationalities:
        parts.append(f"nationality: {', '.join(nationalities)}")
    if teams:
        parts.append(f"team: {', '.join(teams)}")

    return " | ".join(parts).strip()


def _format_rows(rows: list[dict]) -> str:
    if not rows:
        return ""

    sentences = []
    for row in rows:
        driver = row.get("driverLabel") or row.get("driver")
        nationality = row.get("nationalityLabel") or row.get("nationality")
        team = row.get("teamLabel") or row.get("team")
        if driver and nationality and team:
            sentences.append(f"{driver} has nationality {nationality} and drove for {team}.")
        elif driver and nationality:
            sentences.append(f"{driver} has nationality {nationality}.")
        elif driver and team:
            sentences.append(f"{driver} drove for {team}.")
        else:
            pieces = [f"{key}={value}" for key, value in sorted(row.items()) if value]
            if pieces:
                sentences.append("; ".join(pieces))

    return "\n".join(sentences)


def _execute_query(graph: Graph, sparql: str) -> list[dict]:
    """Run a SPARQL query over the local graph and return dictionaries."""

    results = graph.query(sparql)
    rows: list[dict] = []
    for row in results:
        if hasattr(row, "asdict"):
            row_dict = {key: str(value) for key, value in row.asdict().items() if value is not None}
        else:
            row_dict = {}
        if row_dict:
            rows.append(row_dict)
    return rows


def retrieve_sparql(graph: Graph, question: str) -> str:
    """Retrieve context using a structured SPARQL query tailored to the local graph."""

    driver_label = _best_label_match(graph, question)
    terms = _question_terms(question)

    filters: list[str] = []
    if driver_label:
        escaped = driver_label.replace('"', '\\"').lower()
        filters.append(f'CONTAINS(LCASE(STR(?driverLabel)), "{escaped}")')
    else:
        for term in terms[:3]:
            escaped = term.replace('"', '\\"')
            filters.append(
                "("
                f'CONTAINS(LCASE(STR(?driverLabel)), "{escaped}") || '
                f'CONTAINS(LCASE(STR(?nationalityLabel)), "{escaped}") || '
                f'CONTAINS(LCASE(STR(?teamLabel)), "{escaped}")'
                ")"
            )

    filter_clause = ""
    if filters:
        filter_clause = "FILTER(\n    " + " ||\n    ".join(filters) + "\n  )\n"

    sparql = f"""
PREFIX rdfs: <{RDFS}>
PREFIX lab: <{LAB}>

SELECT DISTINCT ?driverLabel ?nationalityLabel ?teamLabel WHERE {{
  ?driver rdfs:label ?driverLabel .
  OPTIONAL {{ ?driver lab:nationality ?nationalityLabel . }}
  OPTIONAL {{ ?driver lab:team ?teamLabel . }}
  {filter_clause}}}
ORDER BY LCASE(STR(?driverLabel))
LIMIT 10
""".strip()

    try:
        rows = _execute_query(graph, sparql)
    except Exception as exc:
        return f"[SPARQL query error] {exc}"

    return _format_rows(rows)


def retrieve_text(graph: Graph, question: str) -> str:
    """Retrieve context using keyword/regex matching over the local graph."""

    terms = _question_terms(question)
    if not terms:
        terms = [_normalize(question)]
    scored: list[tuple[int, str]] = []
    for record in _driver_records(graph):
        sentence = _record_to_sentence(record)
        normalized_sentence = _normalize(sentence)
        matches = sum(1 for term in terms if term in normalized_sentence)
        if matches:
            scored.append((matches, sentence))

    if not scored:
        return ""

    scored.sort(key=lambda item: (-item[0], item[1]))
    return "\n".join(sentence for _, sentence in scored[:5])


def retrieve_embedding(graph: Graph, question: str) -> str:
    """Retrieve the most semantically similar driver summaries to the question."""

    question_embedding = _get_embedding(question)
    if not question_embedding:
        return retrieve_text(graph, question)

    scored: list[tuple[float, str]] = []
    for record in _driver_records(graph):
        summary = _record_to_sentence(record)
        summary_embedding = _get_embedding(summary)
        if not summary_embedding:
            continue
        score = _cosine_similarity(question_embedding, summary_embedding)
        scored.append((score, summary))

    if not scored:
        return retrieve_text(graph, question)

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:3]
    return "\n".join(f"{summary} [score={score:.3f}]" for score, summary in top)


def compare(graph: Graph, question: str) -> None:
    """Run all retrieval strategies on the same question and print results."""

    print(f"\n{'=' * 60}")
    print(f"QUESTION: {question}")
    print("=" * 60)

    for label, fn in [
        ("A) SPARQL retrieval", retrieve_sparql),
        ("B) Text retrieval", retrieve_text),
        ("C) Embedding retrieval", retrieve_embedding),
    ]:
        print(f"\n--- {label} ---")
        try:
            context = fn(graph, question)
        except NotImplementedError:
            print("  [not implemented yet]")
            continue
        print(context or "  (no results)")

        print("\n[ANSWER]")
    print(answer(question, context))


def _load_graph() -> Graph:
    if not GRAPH_PATH.exists():
        export_graph()

    graph = Graph()
    graph.parse(GRAPH_PATH, format="turtle")
    return graph


if __name__ == "__main__":
    graph = _load_graph()
    for sample_question in [
        "What nationality is Lewis Hamilton?",
        "Which team did Lewis Hamilton drive for?",
        "What nationality is Max Verstappen?",
    ]:
        compare(graph, sample_question)
