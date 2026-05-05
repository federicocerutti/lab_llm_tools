#!/usr/bin/env python3
"""Fine-grained token-speed benchmark for the three classroom LiteLLM models.

This script only targets the models listed in the operator manual:
`qwen35-4b`, `ministral3-3b`, and `phi4-mini`.

For each model it:
- runs concurrency levels from 1 to 10 requests
- repeats each concurrency level 10 times
- records token/sec per individual request
- saves a violin plot per model

The output goes under `artifacts/`.
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
PLOT_PATH = ARTIFACTS / "cluster_token_speed_violin.png"
CSV_PATH = ARTIFACTS / "cluster_token_speed_violin.csv"
REPORT_PATH = ARTIFACTS / "cluster_token_speed_violin.md"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lab.settings import Settings


MODELS = ("qwen35-4b", "ministral3-3b", "phi4-mini")
CONCURRENCY_LEVELS = tuple(range(1, 11))
REPETITIONS = 10
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


def completion_kwargs(settings: Settings, model: str) -> dict[str, Any]:
    return {
        "model": f"openai/{model}",
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
    response = litellm.completion(**completion_kwargs(settings, model))
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


def run_request(
    settings: Settings,
    model: str,
    concurrency: int,
    repetition: int,
    request_id: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        result = call_model(settings, model)
        elapsed = time.perf_counter() - start
        completion_tokens = int(result["completion_tokens"])
        return {
            "ok": True,
            "model": model,
            "concurrency": concurrency,
            "repetition": repetition,
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
            "model": model,
            "concurrency": concurrency,
            "repetition": repetition,
            "request_id": request_id,
            "elapsed": elapsed,
            "error": str(exc),
        }


def run_condition(settings: Settings, model: str, concurrency: int, repetition: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {
            executor.submit(run_request, settings, model, concurrency, repetition, request_id): request_id
            for request_id in range(1, concurrency + 1)
        }
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            request_id = result["request_id"]
            elapsed = result["elapsed"]
            if result["ok"]:
                tokens = int(result["completion_tokens"])
                tps = float(result["tokens_per_second"])
                print(
                    f"[ok] {model} c={concurrency:02d} r={repetition:02d} "
                    f"#{request_id:02d} ({elapsed:.2f}s, {tokens} tok, {tps:.2f} tok/s)"
                )
            else:
                print(
                    f"[fail] {model} c={concurrency:02d} r={repetition:02d} "
                    f"#{request_id:02d} ({elapsed:.2f}s): {result['error']}"
                )
    return results


def collect_results(settings: Settings) -> list[dict[str, Any]]:
    all_results: list[dict[str, Any]] = []
    for model in MODELS:
        print(f"\n=== {model} ===")
        for concurrency in CONCURRENCY_LEVELS:
            for repetition in range(1, REPETITIONS + 1):
                batch_results = run_condition(settings, model, concurrency, repetition)
                all_results.extend(batch_results)
    return all_results


def summarize(results: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    summary: dict[str, dict[int, dict[str, Any]]] = {}
    for model in MODELS:
        summary[model] = {}
        model_results = [result for result in results if result["model"] == model]
        for concurrency in CONCURRENCY_LEVELS:
            level_results = [
                result
                for result in model_results
                if result["concurrency"] == concurrency and result["ok"]
            ]
            speeds = [float(result["tokens_per_second"]) for result in level_results]
            summary[model][concurrency] = {
                "successes": len(level_results),
                "failures": len(
                    [
                        result
                        for result in model_results
                        if result["concurrency"] == concurrency and not result["ok"]
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
        "model",
        "concurrency",
        "repetition",
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


def plot_graph(summary: dict[str, dict[int, dict[str, Any]]]) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(11, 14), sharex=True, sharey=True)
    if len(MODELS) == 1:
        axes = [axes]

    for ax, model in zip(axes, MODELS, strict=True):
        data = [summary[model][concurrency]["speeds"] for concurrency in CONCURRENCY_LEVELS]
        parts = ax.violinplot(
            data,
            positions=list(CONCURRENCY_LEVELS),
            widths=0.8,
            showmeans=True,
            showmedians=True,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_alpha(0.7)
        if "cmeans" in parts:
            parts["cmeans"].set_color("black")
        if "cmedians" in parts:
            parts["cmedians"].set_color("white")
            parts["cmedians"].set_linewidth(1.6)
        ax.set_title(model)
        ax.set_ylabel("Tokens/sec")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xticks(list(CONCURRENCY_LEVELS))

    axes[-1].set_xlabel("Concurrent requests per model")
    fig.suptitle("Token rate distributions by concurrency level", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(PLOT_PATH, dpi=180)


def write_report(summary: dict[str, dict[int, dict[str, Any]]], results: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Fine-grained token-speed benchmark")
    lines.append("")
    lines.append(f"- Repetitions per concurrency level: {REPETITIONS}")
    lines.append(f"- Concurrency levels per model: 1..{max(CONCURRENCY_LEVELS)}")
    lines.append("")
    for model in MODELS:
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| Concurrent requests | Mean tokens/sec | Std dev | Successes | Failures |")
        lines.append("| ---: | ---: | ---: | ---: | ---: |")
        for concurrency in CONCURRENCY_LEVELS:
            entry = summary[model][concurrency]
            mean_value = entry["mean"]
            stdev_value = entry["stdev"]
            mean_text = f"{mean_value:.2f}" if not math.isnan(mean_value) else "n/a"
            stdev_text = f"{stdev_value:.2f}" if not math.isnan(stdev_value) else "n/a"
            lines.append(
                f"| {concurrency} | {mean_text} | {stdev_text} | {entry['successes']} | {entry['failures']} |"
            )
        lines.append("")

    failures = [result for result in results if not result["ok"]]
    lines.append(f"Total requests: {len(results)}")
    lines.append(f"Failed requests: {len(failures)}")
    lines.append(f"Plot: `{PLOT_PATH.relative_to(ROOT)}`")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(summary: dict[str, dict[int, dict[str, Any]]]) -> None:
    for model in MODELS:
        print(f"\n=== Summary: {model} ===")
        for concurrency in CONCURRENCY_LEVELS:
            entry = summary[model][concurrency]
            mean_value = entry["mean"]
            stdev_value = entry["stdev"]
            if math.isnan(mean_value):
                stats = "n/a"
            else:
                stats = f"mean={mean_value:.2f} tok/s std={stdev_value:.2f} tok/s"
            print(
                f"{concurrency} concurrent requests: {stats} "
                f"(success={entry['successes']}, failures={entry['failures']})"
            )


def main() -> int:
    settings = Settings()
    api_key = get_api_key(settings)
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY or LITELLM_API_KEY before running this script.")

    litellm.api_base = get_api_base(settings)
    litellm.api_key = api_key

    start = time.perf_counter()
    results = collect_results(settings)
    summary = summarize(results)

    write_csv(results)
    plot_graph(summary)
    write_report(summary, results)

    print_summary(summary)
    print(f"\nSaved plot to {PLOT_PATH}")
    print(f"Saved CSV to {CSV_PATH}")
    print(f"Saved report to {REPORT_PATH}")
    print(f"Total wall time: {time.perf_counter() - start:.2f}s")

    return 1 if any(not result["ok"] for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
