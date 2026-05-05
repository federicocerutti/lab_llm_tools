import requests
from rdflib import Graph, Literal, Namespace
from rdflib.namespace import RDFS

from lab.settings import Settings

settings = Settings()
LAB = Namespace("http://lab.unibs.it/ontology/")
ENTITY = Namespace("http://lab.unibs.it/entity/")

SPARQL_QUERY = """\
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>

SELECT ?driver ?driverLabel ?countryLabel ?teamLabel ?start ?end WHERE {
  VALUES ?driver { wd:Q9673 wd:Q2239218 wd:Q9671 wd:Q10514 wd:Q17541912 wd:Q22007193 }
  ?driver rdfs:label ?driverLabel .
  FILTER(LANG(?driverLabel) = "en")
  OPTIONAL {
    ?driver wdt:P27 ?country .
    ?country rdfs:label ?countryLabel .
    FILTER(LANG(?countryLabel) = "en")
  }
  OPTIONAL {
    ?driver p:P54 ?teamStmt .
    ?teamStmt ps:P54 ?team .
    OPTIONAL { ?teamStmt pq:P580 ?start . }
    OPTIONAL { ?teamStmt pq:P582 ?end . }
    ?team rdfs:label ?teamLabel .
    FILTER(LANG(?teamLabel) = "en")
  }
}
"""

OUTPUT_FILE = "f1_drivers.ttl"
QUERY_ENDPOINTS = [
    settings.wikidata_endpoint,
    "https://query-scholarly.wikidata.org/sparql",
]

QID_TO_SLUG = {
    "Q9673": "lewis-hamilton",
    "Q2239218": "max-verstappen",
    "Q9671": "michael-schumacher",
    "Q10514": "fernando-alonso",
    "Q17541912": "charles-leclerc",
    "Q22007193": "lando-norris",
}

QID_TO_DRIVER = {
    "Q9673": "Lewis Hamilton",
    "Q2239218": "Max Verstappen",
    "Q9671": "Michael Schumacher",
    "Q10514": "Fernando Alonso",
    "Q17541912": "Charles Leclerc",
    "Q22007193": "Lando Norris",
}

COUNTRY_TO_NATIONALITY = {
    "United Kingdom": "British",
    "Great Britain": "British",
    "England": "English",
    "Scotland": "Scottish",
    "Wales": "Welsh",
    "Netherlands": "Dutch",
    "Kingdom of the Netherlands": "Dutch",
    "Germany": "German",
    "Spain": "Spanish",
    "Monaco": "Monegasque",
    "Belgium": "Belgian",
    "Italy": "Italian",
    "France": "French",
    "Finland": "Finnish",
    "Brazil": "Brazilian",
}

PRIMARY_NATIONALITY_ORDER = [
    "British",
    "Dutch",
    "German",
    "Spanish",
    "Monegasque",
    "French",
    "Italian",
    "Finnish",
    "Brazilian",
    "Belgian",
]


def _normalize_nationality(label: str | None) -> str | None:
    if not label:
        return None
    return COUNTRY_TO_NATIONALITY.get(label, label)


def _choose_nationality(labels: list[str]) -> str | None:
    normalized = []
    seen = set()
    for label in labels:
        nationality = _normalize_nationality(label)
        if nationality and nationality not in seen:
            seen.add(nationality)
            normalized.append(nationality)

    for preferred in PRIMARY_NATIONALITY_ORDER:
        if preferred in normalized:
            return preferred
    return normalized[0] if normalized else None


def _choose_team(rows: list[dict[str, str]]) -> str | None:
    if not rows:
        return None

    def sort_key(row: dict[str, str]) -> tuple[int, str, str]:
        current = 1 if not row.get("end") else 0
        return (current, row.get("start") or "", row.get("team") or "")

    return max(rows, key=sort_key).get("team")


def fetch_rdf(sparql: str) -> Graph:
    """Send a SELECT query to Wikidata and convert rows into a compact RDF graph."""

    last_error: Exception | None = None
    for endpoint in QUERY_ENDPOINTS:
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
            if not bindings:
                raise RuntimeError("Wikidata returned no rows")

            records: dict[str, dict[str, object]] = {}
            for row in bindings:
                driver_uri = row.get("driver", {}).get("value", "")
                qid = driver_uri.rsplit("/", 1)[-1] if driver_uri else ""
                slug = QID_TO_SLUG.get(qid)
                driver_label = row.get("driverLabel", {}).get("value")
                if not slug:
                    continue
                if not driver_label or driver_label == qid or (
                    driver_label.startswith("Q") and driver_label[1:].isdigit()
                ):
                    driver_label = QID_TO_DRIVER.get(qid, driver_label or qid)

                record = records.setdefault(
                    qid,
                    {"slug": slug, "driver": driver_label, "countries": [], "teams": []},
                )

                country_label = row.get("countryLabel", {}).get("value")
                if country_label:
                    record["countries"].append(country_label)  # type: ignore[attr-defined]

                team_label = row.get("teamLabel", {}).get("value")
                if team_label:
                    record["teams"].append(
                        {
                            "team": team_label,
                            "start": row.get("start", {}).get("value", ""),
                            "end": row.get("end", {}).get("value", ""),
                        }
                    )  # type: ignore[attr-defined]

            graph = Graph()
            for record in records.values():
                subject = ENTITY[str(record["slug"])]
                graph.add((subject, RDFS.label, Literal(str(record["driver"]))))

                nationality = _choose_nationality(list(record["countries"]))
                if nationality:
                    graph.add((subject, LAB.nationality, Literal(nationality)))

                team = _choose_team(list(record["teams"]))
                if team:
                    graph.add((subject, LAB.team, Literal(team)))

            if len(graph):
                return graph
            raise RuntimeError("Wikidata returned rows but no usable triples")
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("No Wikidata endpoint configured")


def save_graph(g: Graph, path: str) -> None:
    """Serialise the graph to a Turtle file."""

    g.serialize(path, format="turtle")
    print(f"Saved {len(g)} triples to {path}")


def main():
    print("Querying Wikidata...")
    g = fetch_rdf(SPARQL_QUERY)
    print(f"Retrieved {len(g)} triples")
    print(g.serialize(format="turtle"))
    save_graph(g, OUTPUT_FILE)


if __name__ == "__main__":
    main()
