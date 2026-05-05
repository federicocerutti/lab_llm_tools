#!/usr/bin/env python3
"""Benchmark token throughput for the three classroom LiteLLM models.

This script only targets the models listed in the operator manual:
`qwen35-4b`, `ministral3-3b`, and `phi4-mini`.

It measures:
- one isolated request per model
- stepped concurrency from 1 to 10 concurrent requests per model
- average and standard deviation of token speed by model

It also saves a graph and a markdown report under `artifacts/`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, stdev
import csv
import math
import os
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import litellm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ARTIFACTS = ROOT / "artifacts"
GRAPH_PATH = ARTIFACTS / "cluster_token_speed.png"
CSV_PATH = ARTIFACTS / "cluster_token_speed.csv"
REPORT_PATH = ARTIFACTS / "cluster_token_speed.md"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lab.settings import Settings


MODELS = ("qwen35-4b", "ministral3-3b", "phi4-mini")
CONCURRENCY_LEVELS = tuple(range(1, 11))  # 1..10 requests per model; 3..30 total requests
SYSTEM_PROMPT = "You are a concise assistant."
USER_PROMPT = "In exactly three short bullet points, explain what a language model does."


def get_api_base(settings: Settings) -> str:
    return (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("LITELLM_BASE_URL")
        or settings.litellm_base_url
    )


def get_api_key(settings: Settings) -> str | None:
    return os.getenv("OPENAI_API_KEY") or os.getenv("LITELLM_API_KEY") or settings.litellm_api_key


def completion_kwargs(settings: Settings) -> dict[str, Any]:
    return {
        "model": None,  # filled in per call
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "max_tokens": 64,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "top_k": settings.top_k,
        "presence_penalty": settings.presence_penalty,
        "repetition_penalty": settings.repetition_penalty,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "stream": False,
    }


def get_completion_tokens(response: Any) -> int | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        tokens = usage.get("completion_tokens")
    else:
        tokens = getattr(usage, "completion_tokens", None)
    return int(tokens) if tokens is not None else None


def call_model(settings: Settings, model: str) -> dict[str, Any]:
    response = litellm.completion(
        **{
            **completion_kwargs(settings),
            "model": f"openai/{model}",
        }
    )

    content = response.choices[0].message.content
    finish_reason = getattr(response.choices[0], "finish_reason", None)
    completion_tokens = get_completion_tokens(response)

    if not content or not content.strip():
        raise RuntimeError(f"empty response (finish_reason={finish_reason!r})")
    if completion_tokens is None:
        raise RuntimeError("missing completion token count")

    return {
        "content": content.strip(),
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
    }


def run_request(settings: Settings, model: str, batch_label: str, request_id: int) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        result = call_model(settings, model)
        elapsed = time.perf_counter() - start
        completion_tokens = int(result["completion_tokens"])
        return {
            "ok": True,
            "batch_label": batch_label,
            "model": model,
            "request_id": request_id,
            "elapsed": elapsed,
            "completion_tokens": completion_tokens,
            "tokens_per_second": completion_tokens / elapsed if elapsed > 0 else math.inf,
            "content": result["content"],
            "finish_reason": result["finish_reason"],
        }
    except Exception as exc:  # pragma: no cover - practical CLI failure reporting
        elapsed = time.perf_counter() - start
        return {
            "ok": False,
            "batch_label": batch_label,
            "model": model,
            "request_id": request_id,
            "elapsed": elapsed,
            "error": str(exc),
        }


def run_batch(settings: Settings, requests_per_model: int, batch_label: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=requests_per_model * len(MODELS)) as executor:
        future_map = {
            executor.submit(run_request, settings, model, batch_label, request_id): (model, request_id)
            for model in MODELS
            for request_id in range(1, requests_per_model + 1)
        }

        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            model = result["model"]
            request_id = result["request_id"]
            elapsed = result["elapsed"]
            if result["ok"]:
                content = str(result["content"]).replace("\n", " ")
                tokens = int(result["completion_tokens"])
                tps = float(result["tokens_per_second"])
                print(f"[ok] {batch_label} {model} #{request_id:02d} ({elapsed:.2f}s, {tokens} tok, {tps:.2f} tok/s) {content[:110]}")
            else:
                print(f"[fail] {batch_label} {model} #{request_id:02d} ({elapsed:.2f}s): {result['error']}")

    wall_time = time.perf_counter() - total_start
    print(f"Batch {batch_label}: {len(results)} requests in {wall_time:.2f}s")
    print()
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    summary: dict[str, dict[int, dict[str, Any]]] = {}
    for model in MODELS:
        summary[model] = {}
        model_results = [result for result in results if result["model"] == model]
        for level in CONCURRENCY_LEVELS:
            level_results = [
                result
                for result in model_results
                if result["batch_label"] == f"load-{level}" and result["ok"]
            ]
            speeds = [float(result["tokens_per_second"]) for result in level_results]
            summary[model][level] = {
                "successes": len(level_results),
                "failures": len(
                    [
                        result
                        for result in model_results
                        if result["batch_label"] == f"load-{level}" and not result["ok"]
                    ]
                ),
                "mean": mean(speeds) if speeds else math.nan,
                "stdev": stdev(speeds) if len(speeds) > 1 else 0.0 if len(speeds) == 1 else math.nan,
                "speeds": speeds,
            }
    return summary


def write_csv(results: list[dict[str, Any]]) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_label",
        "model",
        "request_id",
        "ok",
        "elapsed",
        "completion_tokens",
        "tokens_per_second",
        "finish_reason",
        "error",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({name: result.get(name, "") for name in fieldnames})


def write_report(
    baseline_results: list[dict[str, Any]],
    summary: dict[str, dict[int, dict[str, Any]]],
) -> None:
    lines: list[str] = []
    lines.append("# Bia token-speed benchmark")
    lines.append("")
    lines.append("## Baseline")
    lines.append("")
    lines.append("| Model | Completion tokens | Seconds | Tokens/sec |")
    lines.append("| --- | ---: | ---: | ---: |")
    for result in baseline_results:
        if result["ok"]:
            lines.append(
                f"| {result['model']} | {int(result['completion_tokens'])} | {float(result['elapsed']):.2f} | {float(result['tokens_per_second']):.2f} |"
            )
        else:
            lines.append(f"| {result['model']} | failed | {float(result['elapsed']):.2f} | failed |")

    lines.append("")
    lines.append("## Load test")
    lines.append("")
    lines.append("| Total concurrent | Model | Mean tokens/sec | Std dev | Success | Failures |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: |")
    for level in CONCURRENCY_LEVELS:
        total_concurrent = level * len(MODELS)
        for model in MODELS:
            entry = summary[model][level]
            mean_value = entry["mean"]
            stdev_value = entry["stdev"]
            mean_text = f"{mean_value:.2f}" if not math.isnan(mean_value) else "n/a"
            stdev_text = f"{stdev_value:.2f}" if not math.isnan(stdev_value) else "n/a"
            lines.append(
                f"| {total_concurrent} | {model} | {mean_text} | {stdev_text} | {entry['successes']} | {entry['failures']} |"
            )

    lines.append("")
    lines.append(f"Graph: `{GRAPH_PATH.relative_to(ROOT)}`")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_graph(summary: dict[str, dict[int, dict[str, Any]]]) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for model in MODELS:
        x = [level * len(MODELS) for level in CONCURRENCY_LEVELS]
        y = [summary[model][level]["mean"] for level in CONCURRENCY_LEVELS]
        yerr = [summary[model][level]["stdev"] for level in CONCURRENCY_LEVELS]
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, linewidth=2, label=model)

    plt.title("Token throughput vs concurrent requests")
    plt.xlabel("Total concurrent requests")
    plt.ylabel("Tokens per second")
    plt.xticks([level * len(MODELS) for level in CONCURRENCY_LEVELS])
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_PATH, dpi=180)


def print_baseline_summary(results: list[dict[str, Any]]) -> None:
    print("Baseline: one isolated request per model")
    for result in results:
        if result["ok"]:
            print(
                f"{result['model']}: {float(result['tokens_per_second']):.2f} tok/s "
                f"({int(result['completion_tokens'])} tokens in {float(result['elapsed']):.2f}s)"
            )
        else:
            print(f"{result['model']}: failed ({result['error']})")
    print()


def print_load_summary(summary: dict[str, dict[int, dict[str, Any]]]) -> None:
    print("Load test: stepped concurrency up to 30 total requests")
    for level in CONCURRENCY_LEVELS:
        total_concurrent = level * len(MODELS)
        print(f"Total concurrent requests: {total_concurrent}")
        for model in MODELS:
            entry = summary[model][level]
            mean_value = entry["mean"]
            stdev_value = entry["stdev"]
            if math.isnan(mean_value):
                stats_text = "n/a"
            else:
                stats_text = f"mean={mean_value:.2f} tok/s std={stdev_value:.2f} tok/s"
            print(
                f"  {model}: {stats_text} "
                f"(success={entry['successes']}, failures={entry['failures']})"
            )
    print()


def main() -> int:
    settings = Settings()
    api_key = get_api_key(settings)
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY or LITELLM_API_KEY before running this script.")

    litellm.api_base = get_api_base(settings)
    litellm.api_key = api_key

    baseline_results: list[dict[str, Any]] = []
    for model in MODELS:
        baseline_results.append(run_request(settings, model, "baseline", 1))

    print_baseline_summary(baseline_results)

    load_results: list[dict[str, Any]] = []
    for level in CONCURRENCY_LEVELS:
        load_results.extend(run_batch(settings, level, f"load-{level}"))

    summary = summarize(load_results)
    print_load_summary(summary)

    write_csv(baseline_results + load_results)
    plot_graph(summary)
    write_report(baseline_results, summary)

    print(f"Saved graph to {GRAPH_PATH}")
    print(f"Saved CSV to {CSV_PATH}")
    print(f"Saved report to {REPORT_PATH}")

    failures = [result for result in baseline_results + load_results if not result["ok"]]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
