#!/usr/bin/env python3
"""Performance and accuracy benchmarks for models on Amazon Bedrock Runtime.

Mirrors scripts/mantle_gemma_benchmark.py but uses the Bedrock Runtime
Converse / ConverseStream APIs. Authentication is SigV4 through the standard
AWS credential chain (environment variables, shared config/credentials files,
or an instance/execution role); no bearer token is used.

Workload shapes, synthetic prompts, accuracy probes, scoring, summaries, and
CSV writing are imported from mantle_gemma_benchmark so both runners stay
comparable.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mantle_gemma_benchmark import (  # noqa: E402
    BUILTIN_ACCURACY_ITEMS,
    AccuracyItem,
    Workload,
    accuracy_prompt,
    estimate_tokens,
    extract_choice,
    extract_number,
    generated_png_data_uri,
    load_accuracy_jsonl,
    make_synthetic_prompt,
    now_run_id,
    override_workloads,
    print_table,
    resolution_size,
    score_answer,
    summarize_accuracy,
    summarize_performance,
    workloads_for_profile,
    write_csv,
)

DEFAULT_MODELS = ["openai.gpt-oss-120b-1:0"]
DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"


def empty_usage() -> Dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
    }


def parse_converse_usage(usage: Any) -> Dict[str, int]:
    usage = usage or {}
    prompt_tokens = int(usage.get("inputTokens", 0) or 0)
    completion_tokens = int(usage.get("outputTokens", 0) or 0)
    total_tokens = int(usage.get("totalTokens", 0) or 0) or prompt_tokens + completion_tokens
    cached_tokens = int(usage.get("cacheReadInputTokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
    }


def load_bedrock_runtime_client(
    region: str,
    timeout: float,
    max_retries: int,
    max_pool_connections: int,
):
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install boto3") from exc

    config = Config(
        region_name=region,
        connect_timeout=min(timeout, 60.0),
        read_timeout=timeout,
        retries={"max_attempts": max_retries + 1, "mode": "standard"},
        max_pool_connections=max(10, max_pool_connections),
    )
    return boto3.client("bedrock-runtime", config=config)


def list_bedrock_models(region: str) -> List[str]:
    import boto3

    bedrock = boto3.client("bedrock", region_name=region)
    model_ids = {
        summary["modelId"]
        for summary in bedrock.list_foundation_models().get("modelSummaries", [])
    }
    next_token: Optional[str] = None
    while True:
        kwargs = {"nextToken": next_token} if next_token else {}
        page = bedrock.list_inference_profiles(**kwargs)
        model_ids.update(
            summary["inferenceProfileId"]
            for summary in page.get("inferenceProfileSummaries", [])
        )
        next_token = page.get("nextToken")
        if not next_token:
            break
    return sorted(model_ids)


def generated_png_bytes(width: int, height: int, image_index: int) -> bytes:
    data_uri = generated_png_data_uri(width, height, image_index)
    return base64.b64decode(data_uri.split(",", 1)[1])


def build_converse_messages(
    workload: Workload, prompt: str, request_index: int
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"text": prompt}]
    if workload.modality == "vision":
        width, height = resolution_size(workload.image_resolution)
        for image_index in range(workload.image_count):
            content.append(
                {
                    "image": {
                        "format": "png",
                        "source": {
                            "bytes": generated_png_bytes(width, height, image_index + request_index)
                        },
                    }
                }
            )
    return [{"role": "user", "content": content}]


def prompt_text_from_converse_messages(messages: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for message in messages:
        for block in message.get("content", []) or []:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
    return "\n".join(parts)


def finalize_usage(usage: Dict[str, int], text: str, prompt_text: str) -> str:
    token_source = "usage"
    if not usage["completion_tokens"] and text:
        usage["completion_tokens"] = estimate_tokens(text)
        token_source = "estimated"
    if not usage["prompt_tokens"]:
        usage["prompt_tokens"] = estimate_tokens(prompt_text)
        token_source = "estimated"
    if not usage["total_tokens"]:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return token_source


def converse_completion(
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> Dict[str, Any]:
    prompt_text = prompt_text_from_converse_messages(messages)
    inference_config = {"maxTokens": max_tokens, "temperature": temperature}

    if stream:
        t0 = time.perf_counter()
        try:
            response = client.converse_stream(
                modelId=model, messages=messages, inferenceConfig=inference_config
            )
            output_parts: List[str] = []
            ttft_s: Optional[float] = None
            usage = empty_usage()
            for event in response["stream"]:
                delta = (event.get("contentBlockDelta") or {}).get("delta") or {}
                text_delta = delta.get("text")
                # Reasoning models stream reasoning deltas before any content;
                # count them for time-to-first-token so the generation window
                # (latency - TTFT) stays meaningful.
                reasoning_delta = delta.get("reasoningContent")
                if (text_delta or reasoning_delta) and ttft_s is None:
                    ttft_s = time.perf_counter() - t0
                if text_delta:
                    output_parts.append(text_delta)
                metadata = event.get("metadata")
                if metadata and metadata.get("usage"):
                    usage = parse_converse_usage(metadata["usage"])
            latency_s = time.perf_counter() - t0
            text = "".join(output_parts)
            token_source = finalize_usage(usage, text, prompt_text)
            return {
                "ok": True,
                "streaming": True,
                "text": text,
                "ttft_s": ttft_s if ttft_s is not None else latency_s,
                "latency_s": latency_s,
                "usage": usage,
                "token_source": token_source,
                "error": "",
            }
        except Exception as exc:
            # If streaming is rejected for this model, keep the run usable by
            # falling back to Converse. TTFT is unavailable in this mode.
            stream_error = str(exc)
    else:
        stream_error = ""

    t0 = time.perf_counter()
    try:
        response = client.converse(
            modelId=model, messages=messages, inferenceConfig=inference_config
        )
        latency_s = time.perf_counter() - t0
        blocks = (response.get("output", {}).get("message", {}) or {}).get("content", []) or []
        text = "".join(block.get("text", "") for block in blocks if isinstance(block, dict))
        usage = parse_converse_usage(response.get("usage"))
        token_source = finalize_usage(usage, text, prompt_text)
        return {
            "ok": True,
            "streaming": False,
            "text": text,
            "ttft_s": None,
            "latency_s": latency_s,
            "usage": usage,
            "token_source": token_source,
            "error": stream_error,
        }
    except Exception as exc:
        latency_s = time.perf_counter() - t0
        error = str(exc)
        if stream_error:
            error = f"{stream_error}; fallback failed: {error}"
        return {
            "ok": False,
            "streaming": False,
            "text": "",
            "ttft_s": None,
            "latency_s": latency_s,
            "usage": empty_usage(),
            "token_source": "usage",
            "error": error,
        }


def run_one_perf_request(
    client: Any,
    model: str,
    workload: Workload,
    request_index: int,
    seed: int,
    temperature: float,
) -> Dict[str, Any]:
    prompt = make_synthetic_prompt(workload.random_input_len, seed, request_index)
    messages = build_converse_messages(workload, prompt, request_index)
    started_epoch = time.time()
    result = converse_completion(
        client=client,
        model=model,
        messages=messages,
        max_tokens=workload.max_output_tokens,
        temperature=temperature,
        stream=True,
    )
    ended_epoch = time.time()
    usage = result["usage"]
    ttft_ms = (result["ttft_s"] * 1000.0) if result["ttft_s"] is not None else 0.0
    latency_ms = result["latency_s"] * 1000.0
    generation_s = max(result["latency_s"] - (result["ttft_s"] or 0.0), 1e-9)
    output_tps = usage["completion_tokens"] / generation_s if result["ok"] else 0.0
    return {
        "model": model,
        "workload": workload.name,
        "modality": workload.modality,
        "request_index": request_index,
        "num_prompts": workload.num_prompts,
        "max_concurrency": workload.max_concurrency,
        "random_input_len": workload.random_input_len,
        "max_output_tokens": workload.max_output_tokens,
        "image_count": workload.image_count,
        "image_resolution": workload.image_resolution,
        "ok": result["ok"],
        "streaming": result["streaming"],
        "started_epoch": round(started_epoch, 6),
        "ended_epoch": round(ended_epoch, 6),
        "ttft_ms": round(ttft_ms, 3),
        "latency_ms": round(latency_ms, 3),
        "generation_ms": round(max(latency_ms - ttft_ms, 0.0), 3),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "cached_tokens": usage["cached_tokens"],
        "token_source": result["token_source"],
        "output_tps": round(output_tps, 3),
        "output_chars": len(result["text"]),
        "output_preview": result["text"][:160].replace("\n", "\\n"),
        "error": result["error"][:500],
    }


def run_workload(
    client: Any,
    model: str,
    workload: Workload,
    seed: int,
    temperature: float,
) -> List[Dict[str, Any]]:
    print(
        f"\n[{model} | {workload.name}] prompts={workload.num_prompts} "
        f"concurrency={workload.max_concurrency} max_tokens={workload.max_output_tokens}",
        flush=True,
    )
    started = time.perf_counter()
    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workload.max_concurrency) as executor:
        futures = [
            executor.submit(
                run_one_perf_request,
                client,
                model,
                workload,
                request_index,
                seed,
                temperature,
            )
            for request_index in range(workload.num_prompts)
        ]
        for done_count, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            if record["ok"]:
                print(
                    f"  {done_count:4d}/{workload.num_prompts}: "
                    f"TTFT={record['ttft_ms']:8.1f} ms  "
                    f"lat={record['latency_ms']:9.1f} ms  "
                    f"out={record['completion_tokens']:5d}  "
                    f"rate={record['output_tps']:7.2f} tok/s",
                    flush=True,
                )
            else:
                print(
                    f"  {done_count:4d}/{workload.num_prompts}: FAILED {record['error'][:120]}",
                    flush=True,
                )
    batch_wall_s = time.perf_counter() - started
    for record in records:
        record["batch_wall_s"] = round(batch_wall_s, 3)
    return sorted(records, key=lambda item: item["request_index"])


def run_one_accuracy_item(
    client: Any,
    model: str,
    item: AccuracyItem,
    max_tokens: int,
) -> Dict[str, Any]:
    prompt = accuracy_prompt(item)
    started_epoch = time.time()
    result = converse_completion(
        client=client,
        model=model,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=False,
    )
    ended_epoch = time.time()
    usage = result["usage"]
    prediction, correct = score_answer(item, result["text"]) if result["ok"] else ("", False)
    return {
        "model": model,
        "suite": item.suite,
        "item_id": item.item_id,
        "category": item.category,
        "answer_type": item.answer_type,
        "expected": item.answer,
        "prediction": prediction,
        "correct": bool(correct),
        "ok": result["ok"],
        "started_epoch": round(started_epoch, 6),
        "ended_epoch": round(ended_epoch, 6),
        "latency_ms": round(result["latency_s"] * 1000.0, 3),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "token_source": result["token_source"],
        "output_text": result["text"].replace("\n", "\\n")[:500],
        "error": result["error"][:500],
    }


def run_accuracy(
    client: Any,
    model: str,
    items: Sequence[AccuracyItem],
    max_tokens: int,
    max_concurrency: int,
) -> List[Dict[str, Any]]:
    print(f"\n[{model} | accuracy] items={len(items)} concurrency={max_concurrency}", flush=True)
    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = [
            executor.submit(run_one_accuracy_item, client, model, item, max_tokens)
            for item in items
        ]
        for done_count, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            status = "ok" if record["correct"] else "wrong"
            if not record["ok"]:
                status = "failed"
            print(
                f"  {done_count:4d}/{len(items)}: {record['suite']}:{record['item_id']} "
                f"{status} pred={record['prediction']} expected={record['expected']}",
                flush=True,
            )
    return sorted(records, key=lambda item: (item["suite"], item["item_id"]))


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    workloads = workloads_for_profile(args.profile, args.include_vision)
    workloads = override_workloads(
        workloads,
        num_prompts=args.num_prompts,
        max_concurrency=args.max_concurrency,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        image_resolution=args.image_resolution,
    )

    max_pool = max(
        [workload.max_concurrency for workload in workloads] + [args.accuracy_concurrency]
    )
    client = load_bedrock_runtime_client(
        region=args.region,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_pool_connections=max_pool,
    )

    if args.list_models:
        try:
            available = list_bedrock_models(args.region)
            print(f"\nAvailable Bedrock models and inference profiles in {args.region}:")
            for model in available:
                print(model)
            missing = sorted(set(args.models) - set(available))
            if missing:
                print("\nRequested models not present in model list:")
                for model in missing:
                    print(model)
        except Exception as exc:
            print(f"\nModel listing failed; continuing with requested models. Error: {exc}")

    run_id = args.run_id or now_run_id()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    accuracy_items = list(BUILTIN_ACCURACY_ITEMS)
    if args.accuracy_jsonl:
        accuracy_items.extend(load_accuracy_jsonl(Path(args.accuracy_jsonl).expanduser()))

    perf_rows: List[Dict[str, Any]] = []
    accuracy_rows: List[Dict[str, Any]] = []

    if not args.skip_performance:
        for model in args.models:
            for workload in workloads:
                perf_rows.extend(
                    run_workload(
                        client=client,
                        model=model,
                        workload=workload,
                        seed=args.seed,
                        temperature=args.temperature,
                    )
                )

    if not args.skip_accuracy:
        for model in args.models:
            accuracy_rows.extend(
                run_accuracy(
                    client=client,
                    model=model,
                    items=accuracy_items,
                    max_tokens=args.accuracy_max_tokens,
                    max_concurrency=args.accuracy_concurrency,
                )
            )

    perf_summary = summarize_performance(perf_rows)
    accuracy_summary = summarize_accuracy(accuracy_rows)

    paths = {
        "performance_csv": str(out_dir / f"bedrock_runtime_performance_{run_id}.csv"),
        "performance_summary_csv": str(out_dir / f"bedrock_runtime_performance_summary_{run_id}.csv"),
        "accuracy_csv": str(out_dir / f"bedrock_runtime_accuracy_{run_id}.csv"),
        "accuracy_summary_csv": str(out_dir / f"bedrock_runtime_accuracy_summary_{run_id}.csv"),
        "manifest_json": str(out_dir / f"bedrock_runtime_manifest_{run_id}.json"),
        "latest_manifest_json": str(out_dir / "latest_bedrock_runtime_manifest.json"),
    }

    write_csv(Path(paths["performance_csv"]), perf_rows)
    write_csv(Path(paths["performance_summary_csv"]), perf_summary)
    write_csv(Path(paths["accuracy_csv"]), accuracy_rows)
    write_csv(Path(paths["accuracy_summary_csv"]), accuracy_summary)

    manifest = {
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "api": "bedrock-runtime converse",
        "auth": "sigv4",
        "region": args.region,
        "models": args.models,
        "profile": args.profile,
        "include_vision": args.include_vision,
        "workloads": [asdict(workload) for workload in workloads],
        "skip_performance": args.skip_performance,
        "skip_accuracy": args.skip_accuracy,
        "paths": paths,
    }
    Path(paths["manifest_json"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    Path(paths["latest_manifest_json"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if perf_summary:
        print("\nPerformance summary")
        print_table(
            perf_summary,
            [
                "model",
                "workload",
                "successful_requests",
                "errors",
                "request_rps",
                "aggregate_output_tps",
                "ttft_p50_ms",
                "latency_p50_ms",
                "latency_p90_ms",
            ],
        )

    if accuracy_summary:
        print("\nAccuracy summary")
        print_table(
            [row for row in accuracy_summary if "category" not in row],
            ["model", "suite", "items", "correct", "accuracy", "latency_mean_ms"],
        )

    print("\nWrote outputs:")
    for label, path in paths.items():
        print(f"  {label}: {path}")

    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark models on Amazon Bedrock Runtime via the Converse API (SigV4 auth).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--out-dir", default="results/bedrock_runtime")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--profile",
        choices=["smoke", "sglang-text", "sglang-full"],
        default="smoke",
        help="smoke is cheap; sglang-text mirrors text commands; sglang-full adds 2x720p vision.",
    )
    parser.add_argument("--include-vision", action="store_true")
    parser.add_argument("--skip-performance", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=None, help="Override prompts per workload.")
    parser.add_argument("--max-concurrency", type=int, default=None, help="Override workload concurrency.")
    parser.add_argument("--input-tokens", type=int, default=None, help="Override synthetic input length.")
    parser.add_argument("--output-tokens", type=int, default=None, help="Override max output tokens.")
    parser.add_argument("--image-resolution", default=None, help="Override vision image resolution.")
    parser.add_argument("--accuracy-jsonl", default="", help="Optional extra accuracy items in JSONL.")
    parser.add_argument(
        "--accuracy-max-tokens",
        type=int,
        default=256,
        help="Reasoning models spend tokens on hidden reasoning; keep this >= 256.",
    )
    parser.add_argument("--accuracy-concurrency", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run_benchmark(args)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
