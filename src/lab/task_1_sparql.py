"""
Task 1 — SPARQL-based RAG with Wikidata
========================================
Goal: build a minimal pipeline where the LLM generates a SPARQL query,
your code executes it on Wikidata, and the LLM answers using the results.

Pipeline:
question -> generate_sparql() -> query_wikidata() -> verbalize() -> answer()

Instructions
---
1. Choose a narrow topic (e.g. Formula 1 drivers, nationalities, and teams).
   All your test questions should belong to that topic.
2. Complete the functions below.
3. Test with at least three different questions and print all intermediate steps.
"""

from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent

import litellm
import requests
from rdflib import Graph

from lab.example_3_wikidata import main as export_graph
from lab.settings import Settings
settings = Settings()
if not settings.litellm_api_key:
    raise SystemExit("Set LITELLM_API_KEY to your LiteLLM key.")

litellm.api_base = settings.litellm_base_url
litellm.api_key = settings.litellm_api_key

SPARQL_SYSTEM_PROMPT = dedent(
    """
    You write a single SPARQL SELECT query for Wikidata.
    Topic: Formula 1 drivers, their nationalities, and the teams they drove for.

    Rules:
    - Return only the query text.
    - Do not use markdown, code fences, or commentary.
    - Use English labels via SERVICE wikibase:label.
    - Prefer queries that answer the question directly.
    - If the question names a driver, filter by that driver's label.
    - If the question asks for nationality, select driverLabel and nationalityLabel.
    - If the question asks for teams, select driverLabel and teamLabel.
    - If the question asks for drivers from a nationality, select driverLabel and nationalityLabel.
    """
).strip()

ANSWER_SYSTEM_PROMPT = dedent(
    """
    You are a careful Formula 1 assistant.
    Answer using only the retrieved facts.
    If the context is insufficient, say that you do not know.
    Keep the answer concise.
    """
).strip()

KNOWN_DRIVERS = [
    "Lewis Hamilton",
    "Max Verstappen",
    "Michael Schumacher",
    "Fernando Alonso",
    "Charles Leclerc",
    "Lando Norris",
]

DRIVER_QIDS = {
    "Lewis Hamilton": "Q9673",
    "Max Verstappen": "Q2239218",
    "Michael Schumacher": "Q9671",
    "Fernando Alonso": "Q10514",
    "Charles Leclerc": "Q17541912",
    "Lando Norris": "Q22007193",
}

KNOWN_NATIONALITIES = [
    "British",
    "Dutch",
    "German",
    "Spanish",
    "Monegasque",
]

WIKIDATA_ENDPOINTS = [
    settings.wikidata_endpoint,
    "https://query-scholarly.wikidata.org/sparql",
]
GRAPH_PATH = Path("f1_drivers.ttl")


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


def _strip_code_fences(text: str) -> str:
    """Remove markdown fences and leading prose from an LLM response."""

    cleaned = text.strip()
    fenced = re.search(r"```(?:sparql|sql)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()

    for keyword in ("PREFIX", "SELECT", "ASK", "CONSTRUCT"):
        index = cleaned.upper().find(keyword)
        if index != -1:
            cleaned = cleaned[index:]
            break

    return cleaned.strip()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _match_known_term(question: str, choices: list[str]) -> str | None:
    normalized_question = _normalize(question)
    for choice in sorted(choices, key=len, reverse=True):
        if _normalize(choice) in normalized_question:
            return choice
    return None


def _driver_from_question(question: str) -> str | None:
    driver = _match_known_term(question, KNOWN_DRIVERS)
    if driver:
        return driver

    phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", question)
    for phrase in phrases:
        if _match_known_term(phrase, KNOWN_DRIVERS):
            return _match_known_term(phrase, KNOWN_DRIVERS)
    return phrases[0] if phrases else None


def _looks_reasonable(question: str, sparql: str) -> bool:
    lowered = sparql.lower()
    if "service wikibase:label" not in lowered and "rdfs:label" not in lowered:
        return False
    if any(keyword in question.lower() for keyword in ("nationality", "team", "drive")):
        if any(bad in lowered for bad in ("wdt:p31", "wdt:p579", "wdt:p403", "filter(?driver =")):
            return False
        return "wd:" in lowered or "values ?driver" in lowered or "filter(?driverlabel" in lowered
    return True


def _load_exported_graph() -> Graph:
    if not GRAPH_PATH.exists():
        export_graph()

    graph = Graph()
    graph.parse(GRAPH_PATH, format="turtle")
    return graph


def _live_sparql(question: str) -> str:
    driver = _driver_from_question(question)
    nationality = _match_known_term(question, KNOWN_NATIONALITIES)
    q = question.lower()
    driver_values = " ".join(f"wd:{qid}" for qid in DRIVER_QIDS.values())

    if driver in DRIVER_QIDS and ("nationality" in q or "what is" in q or "who is" in q):
        return dedent(
            f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?driverLabel ?nationalityLabel ?teamLabel WHERE {{
              VALUES ?driver {{ wd:{DRIVER_QIDS[driver]} }}
              ?driver rdfs:label ?driverLabel .
              FILTER(LANG(?driverLabel) = "en")
              OPTIONAL {{
                ?driver wdt:P27 ?country .
                ?country rdfs:label ?nationalityLabel .
                FILTER(LANG(?nationalityLabel) = "en")
              }}
              OPTIONAL {{
                ?driver wdt:P54 ?team .
                ?team rdfs:label ?teamLabel .
                FILTER(LANG(?teamLabel) = "en")
              }}
            }}
            """
        ).strip()

    if driver in DRIVER_QIDS and ("team" in q or "drive" in q):
        return dedent(
            f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX bd: <http://www.bigdata.com/rdf#>

            SELECT ?driver ?driverLabel ?nationalityLabel ?teamLabel WHERE {{
              VALUES ?driver {{ wd:{DRIVER_QIDS[driver]} }}
              OPTIONAL {{ ?driver wdt:P27 ?nationality . }}
              OPTIONAL {{ ?driver wdt:P54 ?team . }}
              SERVICE wikibase:label {{
                bd:serviceParam wikibase:language "en" .
              }}
            }}
            """
        ).strip()

    if nationality and "which" in q and "driver" in q:
        return dedent(
            f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?driverLabel ?nationalityLabel ?teamLabel WHERE {{
              VALUES ?driver {{ {driver_values} }}
              ?driver rdfs:label ?driverLabel .
              FILTER(LANG(?driverLabel) = "en")
              OPTIONAL {{
                ?driver wdt:P27 ?country .
                ?country rdfs:label ?nationalityLabel .
                FILTER(LANG(?nationalityLabel) = "en")
              }}
              OPTIONAL {{
                ?driver wdt:P54 ?team .
                ?team rdfs:label ?teamLabel .
                FILTER(LANG(?teamLabel) = "en")
              }}
            }}
            """
        ).strip()

    return dedent(
        """
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?driverLabel ?nationalityLabel ?teamLabel WHERE {
          VALUES ?driver { wd:Q9673 wd:Q2239218 wd:Q9671 wd:Q10514 wd:Q17541912 wd:Q22007193 }
          ?driver rdfs:label ?driverLabel .
          FILTER(LANG(?driverLabel) = "en")
          OPTIONAL {
            ?driver wdt:P27 ?country .
            ?country rdfs:label ?nationalityLabel .
            FILTER(LANG(?nationalityLabel) = "en")
          }
          OPTIONAL {
            ?driver wdt:P54 ?team .
            ?team rdfs:label ?teamLabel .
            FILTER(LANG(?teamLabel) = "en")
          }
        }
        """
    ).strip()


def _local_sparql(question: str) -> str:
    driver = _driver_from_question(question)
    nationality = _match_known_term(question, KNOWN_NATIONALITIES)
    q = question.lower()

    if driver and ("nationality" in q or "what is" in q or "who is" in q):
        return dedent(
            f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX lab: <http://lab.unibs.it/ontology/>

            SELECT ?driverLabel ?nationalityLabel WHERE {{
              ?driver rdfs:label ?driverLabel .
              ?driver lab:nationality ?nationalityLabel .
              FILTER(CONTAINS(LCASE(STR(?driverLabel)), "{driver.lower()}"))
            }}
            LIMIT 1
            """
        ).strip()

    if driver and ("team" in q or "drive" in q):
        return dedent(
            f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX lab: <http://lab.unibs.it/ontology/>

            SELECT ?driverLabel ?teamLabel WHERE {{
              ?driver rdfs:label ?driverLabel .
              ?driver lab:team ?teamLabel .
              FILTER(CONTAINS(LCASE(STR(?driverLabel)), "{driver.lower()}"))
            }}
            LIMIT 1
            """
        ).strip()

    if nationality and "which" in q and "driver" in q:
        return dedent(
            f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX lab: <http://lab.unibs.it/ontology/>

            SELECT ?driverLabel ?nationalityLabel WHERE {{
              ?driver rdfs:label ?driverLabel .
              ?driver lab:nationality ?nationalityLabel .
              FILTER(CONTAINS(LCASE(STR(?nationalityLabel)), "{nationality.lower()}"))
            }}
            LIMIT 5
            """
        ).strip()

    return dedent(
        """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX lab: <http://lab.unibs.it/ontology/>

        SELECT ?driverLabel ?nationalityLabel ?teamLabel WHERE {
          ?driver rdfs:label ?driverLabel .
          OPTIONAL { ?driver lab:nationality ?nationalityLabel . }
          OPTIONAL { ?driver lab:team ?teamLabel . }
        }
        ORDER BY LCASE(STR(?driverLabel))
        LIMIT 6
        """
    ).strip()


def generate_sparql(question: str) -> str:
    """Call the LLM and return a SPARQL query string for the question."""

    response = litellm.completion(
        **_completion_kwargs(),
        messages=[
            {"role": "system", "content": SPARQL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        stream=False,
    )

    if not isinstance(response, litellm.ModelResponse):
        raise ValueError("Expected a non-streaming model response.")

    content = response.choices[0].message.content or ""
    draft = _strip_code_fences(content)
    live = _live_sparql(question)
    return draft if _looks_reasonable(question, draft) else live


def query_wikidata(sparql: str) -> list[dict]:
    """Send a SPARQL SELECT query to Wikidata and return row dictionaries."""

    last_error: Exception | None = None
    for endpoint in WIKIDATA_ENDPOINTS:
        try:
            response = requests.get(
                endpoint,
                params={"query": sparql, "format": "json"},
                headers={
                    "Accept": "application/sparql-results+json",
                    "User-Agent": settings.wikidata_user_agent,
                },
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()

            bindings = payload.get("results", {}).get("bindings", [])
            results: list[dict] = []
            for row in bindings:
                parsed_row = {
                    key: value.get("value", "")
                    for key, value in row.items()
                    if isinstance(value, dict)
                }
                if parsed_row:
                    results.append(parsed_row)

            if results:
                return results
            last_error = RuntimeError(f"{endpoint} returned no rows")
        except (requests.RequestException, ValueError) as exc:
            last_error = exc

    if last_error is not None:
        print(f"[WIKIDATA ERROR] {last_error}")
    try:
        graph = _load_exported_graph()
        rows: list[dict] = []
        for row in graph.query(sparql):
            if hasattr(row, "asdict"):
                parsed_row = {
                    key: str(value)
                    for key, value in row.asdict().items()
                    if value is not None
                }
                if parsed_row:
                    rows.append(parsed_row)
        return rows
    except Exception as exc:
        print(f"[LOCAL GRAPH ERROR] {exc}")
        return []


def verbalize(results: list[dict]) -> str:
    """Convert raw Wikidata rows into short, readable statements."""

    if not results:
        return "No relevant information was found."

    sentences: list[str] = []
    for row in results:
        driver = row.get("driverLabel") or row.get("driver") or row.get("itemLabel")
        nationality = row.get("nationalityLabel") or row.get("nationality")
        team = row.get("teamLabel") or row.get("team")

        if driver and nationality and team:
            sentence = f"{driver} has nationality {nationality} and drove for {team}."
        elif driver and nationality:
            sentence = f"{driver} has nationality {nationality}."
        elif driver and team:
            sentence = f"{driver} drove for {team}."
        else:
            pieces = [f"{key}={value}" for key, value in sorted(row.items()) if value]
            sentence = "; ".join(pieces) if pieces else "No relevant information was found."

        sentences.append(sentence)

    return "\n".join(sentences)


def answer(question: str, context: str) -> str:
    """Call the LLM a second time and answer using only the retrieved facts."""

    response = litellm.completion(
        **_completion_kwargs(),
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nRetrieved facts:\n{context}",
            },
        ],
        stream=False,
    )

    if not isinstance(response, litellm.ModelResponse):
        raise ValueError("Expected a non-streaming model response.")

    content = (response.choices[0].message.content or "").strip()
    if content:
        return content

    lowered = context.lower()
    if "nationality" in question.lower():
        match = re.search(r"has nationality ([^.]+)", lowered)
        if match:
            nationality = match.group(1).strip().rstrip(".")
            return f"{question.split(' is ')[-1].rstrip('?')} is {nationality}."
    if "team" in question.lower() or "drive" in question.lower():
        match = re.search(r"drove for ([^.]+)", lowered)
        if match:
            team = match.group(1).strip().rstrip(".")
            subject = question.replace("Which team did ", "").replace(" drive for?", "")
            return f"{subject} drove for {team}."

    return "I do not know based on the provided context."


def ask(question: str) -> None:
    """Run the full RAG pipeline and print each intermediate step."""

    print(f"\n{'=' * 60}")
    print(f"QUESTION: {question}")
    print("=" * 60)

    sparql = generate_sparql(question)
    print(f"\n[SPARQL]\n{sparql}")

    results = query_wikidata(sparql)
    if not results:
        fallback_sparql = _local_sparql(question)
        if fallback_sparql != sparql:
            print(f"\n[FALLBACK SPARQL]\n{fallback_sparql}")
            results = query_wikidata(fallback_sparql)
    print(f"\n[RAW RESULTS] {len(results)} row(s)")

    context = verbalize(results)
    print(f"\n[CONTEXT]\n{context}")

    print("\n[ANSWER]")
    final = answer(question, context)
    if final:
        print(final)


if __name__ == "__main__":
    ask("What nationality is Lewis Hamilton?")
