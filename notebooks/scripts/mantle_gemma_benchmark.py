#!/usr/bin/env python3
"""Performance and accuracy benchmarks for Gemma 4 models on AWS Mantle.

The benchmark mirrors the structure of /codes/utils/test_sglang_perf.sh:

* latency workload: small request count, concurrency 1
* throughput workload: high request count, high concurrency
* optional vision workload: 2 generated 720p images, OpenAI vision payloads
* accuracy workload: GSM8K-style numeric probes and MMLU-style MCQ probes

Mantle credentials are read from BEDROCK_MANTLE_TOKEN by default. Do not store
tokens in notebooks or source files.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import random
import re
import statistics
import struct
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_MODELS = ["google.gemma-4-31b", "google.gemma-4-26b-a4b"]
DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-2"

@dataclass(frozen=True)
class Workload:
    name: str
    modality: str
    num_prompts: int
    max_concurrency: int
    random_input_len: int
    max_output_tokens: int
    image_count: int = 0
    image_resolution: str = "720p"


@dataclass(frozen=True)
class AccuracyItem:
    suite: str
    item_id: str
    category: str
    question: str
    answer: str
    answer_type: str
    choices: Optional[Dict[str, str]] = None


WORD_BANK = [
    "adapter",
    "attention",
    "batch",
    "cache",
    "compiler",
    "context",
    "decoder",
    "embedding",
    "endpoint",
    "evaluation",
    "executor",
    "graph",
    "inference",
    "kernel",
    "latency",
    "memory",
    "metric",
    "model",
    "pipeline",
    "prefill",
    "prompt",
    "quantization",
    "request",
    "scheduler",
    "sequence",
    "server",
    "stream",
    "tensor",
    "throughput",
    "token",
    "tracing",
    "worker",
]


BUILTIN_ACCURACY_ITEMS = [
    AccuracyItem(
        suite="mini_gsm8k",
        item_id="gsm8k_001",
        category="arithmetic",
        question="Lena has 12 marbles. She gives away 5 marbles and then buys 9 more. How many marbles does she have?",
        answer="16",
        answer_type="numeric",
    ),
    AccuracyItem(
        suite="mini_gsm8k",
        item_id="gsm8k_002",
        category="arithmetic",
        question="A train travels 45 miles per hour for 3 hours. How many miles does it travel?",
        answer="135",
        answer_type="numeric",
    ),
    AccuracyItem(
        suite="mini_gsm8k",
        item_id="gsm8k_003",
        category="arithmetic",
        question="If 6 notebooks cost 24 dollars in total, what is the cost of one notebook in dollars?",
        answer="4",
        answer_type="numeric",
    ),
    AccuracyItem(
        suite="mini_gsm8k",
        item_id="gsm8k_004",
        category="arithmetic",
        question="There are 7 boxes with 8 pencils in each box. If 13 pencils are used, how many pencils remain?",
        answer="43",
        answer_type="numeric",
    ),
    AccuracyItem(
        suite="mini_gsm8k",
        item_id="gsm8k_005",
        category="arithmetic",
        question="A number multiplied by 4 equals 52. What is the number?",
        answer="13",
        answer_type="numeric",
    ),
    AccuracyItem(
        suite="mini_gsm8k",
        item_id="gsm8k_006",
        category="arithmetic",
        question="Mia saves 15 dollars each week for 6 weeks, then spends 20 dollars. How many dollars does she have left?",
        answer="70",
        answer_type="numeric",
    ),
    AccuracyItem(
        suite="mini_gsm8k",
        item_id="gsm8k_007",
        category="geometry",
        question="A rectangle has length 9 and width 5. What is its perimeter?",
        answer="28",
        answer_type="numeric",
    ),
    AccuracyItem(
        suite="mini_gsm8k",
        item_id="gsm8k_008",
        category="arithmetic",
        question="72 cookies are split equally among 9 students. How many cookies does each student get?",
        answer="8",
        answer_type="numeric",
    ),
    AccuracyItem(
        suite="mini_mmlu",
        item_id="mmlu_001",
        category="biology",
        question="Which gas do plants primarily absorb from the atmosphere during photosynthesis?",
        choices={"A": "Oxygen", "B": "Nitrogen", "C": "Carbon dioxide", "D": "Helium"},
        answer="C",
        answer_type="choice",
    ),
    AccuracyItem(
        suite="mini_mmlu",
        item_id="mmlu_002",
        category="mathematics",
        question="What is the derivative of x squared with respect to x?",
        choices={"A": "x", "B": "2x", "C": "x cubed", "D": "2"},
        answer="B",
        answer_type="choice",
    ),
    AccuracyItem(
        suite="mini_mmlu",
        item_id="mmlu_003",
        category="geography",
        question="What is the capital city of Canada?",
        choices={"A": "Toronto", "B": "Vancouver", "C": "Ottawa", "D": "Montreal"},
        answer="C",
        answer_type="choice",
    ),
    AccuracyItem(
        suite="mini_mmlu",
        item_id="mmlu_004",
        category="computer_science",
        question="In computing, what does CPU stand for?",
        choices={
            "A": "Central Processing Unit",
            "B": "Computer Protocol Utility",
            "C": "Control Program Unit",
            "D": "Core Packet Uplink",
        },
        answer="A",
        answer_type="choice",
    ),
    AccuracyItem(
        suite="mini_mmlu",
        item_id="mmlu_005",
        category="physics",
        question="Who proposed the theory of general relativity?",
        choices={
            "A": "Isaac Newton",
            "B": "Albert Einstein",
            "C": "Marie Curie",
            "D": "Niels Bohr",
        },
        answer="B",
        answer_type="choice",
    ),
    AccuracyItem(
        suite="mini_mmlu",
        item_id="mmlu_006",
        category="anatomy",
        question="Which organ pumps blood through the human body?",
        choices={"A": "Liver", "B": "Kidney", "C": "Lung", "D": "Heart"},
        answer="D",
        answer_type="choice",
    ),
    AccuracyItem(
        suite="mini_mmlu",
        item_id="mmlu_007",
        category="computer_science",
        question="Which protocol is commonly used for secure web browsing?",
        choices={"A": "FTP", "B": "SMTP", "C": "HTTPS", "D": "Telnet"},
        answer="C",
        answer_type="choice",
    ),
    AccuracyItem(
        suite="mini_mmlu",
        item_id="mmlu_008",
        category="chemistry",
        question="What common substance has the chemical formula H2O?",
        choices={"A": "Salt", "B": "Water", "C": "Oxygen", "D": "Hydrogen peroxide"},
        answer="B",
        answer_type="choice",
    ),
    AccuracyItem(
        suite="mini_mmlu",
        item_id="mmlu_009",
        category="astronomy",
        question="Which planet is commonly called the red planet?",
        choices={"A": "Venus", "B": "Mars", "C": "Jupiter", "D": "Mercury"},
        answer="B",
        answer_type="choice",
    ),
    AccuracyItem(
        suite="mini_mmlu",
        item_id="mmlu_010",
        category="statistics",
        question="What is the median of the numbers 2, 8, 4, 10, and 6?",
        choices={"A": "4", "B": "5", "C": "6", "D": "8"},
        answer="C",
        answer_type="choice",
    ),
]


def get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def percentile(values: Sequence[float], q: float) -> float:
    clean = sorted(v for v in values if v is not None and not math.isnan(v))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    return clean[lo] + (clean[hi] - clean[lo]) * (pos - lo)


def now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def mantle_root_url(value: str) -> str:
    trimmed = value.rstrip("/")
    # Accept full chat endpoints (e.g. BEDROCK_MANTLE_URL_CHAT) as well as
    # /v1 or /openai/v1 base URLs; reduce them all to the endpoint root.
    if trimmed.endswith("/chat/completions"):
        trimmed = trimmed[: -len("/chat/completions")]
    for suffix in ("/openai/v1", "/v1"):
        if trimmed.endswith(suffix):
            return trimmed[: -len(suffix)]
    return trimmed


def resolve_mantle_root_url(base_url: str, region: str) -> str:
    value = (
        base_url
        or os.environ.get("MANTLE_GEMMA_BASE_URL")
        or os.environ.get("BEDROCK_MANTLE_URL")
        or f"https://bedrock-mantle.{region}.api.aws"
    )
    return mantle_root_url(value)


def mantle_generation_base_url(root_url: str, model: str) -> str:
    # Gemma 4 preview models are only served on the /openai/v1 route; all
    # other Mantle models (and the /v1/models listing) use the /v1 route.
    if model.startswith("google.gemma-4"):
        return f"{root_url}/openai/v1"
    return f"{root_url}/v1"


def load_openai_client(api_key: str, base_url: str, timeout: float, max_retries: int):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install openai") from exc

    try:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
    except TypeError:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def list_models(client: Any) -> List[str]:
    models = client.models.list()
    return sorted(get_attr(model, "id", "") for model in models.data if get_attr(model, "id", ""))


def make_synthetic_prompt(target_tokens: int, seed: int, request_index: int) -> str:
    rng = random.Random(seed + request_index * 7919)
    words = [rng.choice(WORD_BANK) for _ in range(max(1, target_tokens))]
    body = " ".join(words)
    return (
        "Synthetic inference benchmark prompt. Use the context to answer in clear prose.\n\n"
        f"{body}\n\n"
        "Task: summarize the operational tradeoffs in two concise paragraphs."
    )


def resolution_size(name: str) -> Tuple[int, int]:
    normalized = name.lower().strip()
    sizes = {
        "224p": (224, 224),
        "360p": (640, 360),
        "480p": (854, 480),
        "720p": (1280, 720),
    }
    if normalized in sizes:
        return sizes[normalized]
    match = re.fullmatch(r"(\d+)x(\d+)", normalized)
    if not match:
        raise ValueError(f"Unsupported image resolution: {name}")
    return int(match.group(1)), int(match.group(2))


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc)
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc & 0xFFFFFFFF)


@lru_cache(maxsize=16)
def generated_png_data_uri(width: int, height: int, image_index: int) -> str:
    rows = []
    for y in range(height):
        row = bytearray()
        row.append(0)
        for x in range(width):
            r = (x * 255) // max(1, width - 1)
            g = (y * 255) // max(1, height - 1)
            b = (image_index * 67 + x // 16 + y // 16) % 256
            row.extend((r, g, b))
        rows.append(bytes(row))
    raw = b"".join(rows)
    png = b"\x89PNG\r\n\x1a\n"
    png += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += png_chunk(b"IDAT", zlib.compress(raw, level=6))
    png += png_chunk(b"IEND", b"")
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_messages(workload: Workload, prompt: str, request_index: int) -> List[Dict[str, Any]]:
    if workload.modality == "text":
        return [{"role": "user", "content": prompt}]

    width, height = resolution_size(workload.image_resolution)
    content = [{"type": "text", "text": prompt}]
    for image_index in range(workload.image_count):
        uri = generated_png_data_uri(width, height, image_index + request_index)
        content.append({"type": "image_url", "image_url": {"url": uri}})
    return [{"role": "user", "content": content}]


def prompt_text_from_messages(messages: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
    return "\n".join(parts)


def parse_usage(usage: Any) -> Dict[str, int]:
    if not usage:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }

    prompt_tokens = get_attr(usage, "prompt_tokens", 0) or get_attr(usage, "input_tokens", 0) or 0
    completion_tokens = (
        get_attr(usage, "completion_tokens", 0) or get_attr(usage, "output_tokens", 0) or 0
    )
    total_tokens = get_attr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)

    cached_tokens = 0
    details = get_attr(usage, "prompt_tokens_details", None) or get_attr(
        usage, "input_tokens_details", None
    )
    if details:
        cached_tokens = get_attr(details, "cached_tokens", 0) or 0

    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
        "cached_tokens": int(cached_tokens or 0),
    }


def should_retry_without_stream_options(exc: Exception) -> bool:
    text = str(exc).lower()
    return "stream_options" in text or "stream options" in text or "extra inputs are not permitted" in text


def chat_completion(
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> Dict[str, Any]:
    prompt_text = prompt_text_from_messages(messages)
    output_parts: List[str] = []
    ttft_s: Optional[float] = None
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
    }
    token_source = "usage"

    if stream:
        last_exc: Optional[Exception] = None
        for include_usage in (True, False):
            output_parts = []
            ttft_s = None
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
            }
            t0 = time.perf_counter()
            try:
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                }
                if include_usage:
                    kwargs["stream_options"] = {"include_usage": True}
                response_stream = client.chat.completions.create(**kwargs)
                for chunk in response_stream:
                    chunk_usage = get_attr(chunk, "usage", None)
                    if chunk_usage:
                        usage = parse_usage(chunk_usage)
                    choices = get_attr(chunk, "choices", []) or []
                    for choice in choices:
                        delta = get_attr(choice, "delta", None)
                        content = get_attr(delta, "content", None)
                        # Reasoning models stream reasoning deltas before any
                        # content; count them for time-to-first-token so the
                        # generation window (latency - TTFT) stays meaningful.
                        reasoning = get_attr(delta, "reasoning", None) or get_attr(
                            delta, "reasoning_content", None
                        )
                        if (content or reasoning) and ttft_s is None:
                            ttft_s = time.perf_counter() - t0
                        if content:
                            output_parts.append(content)
                latency_s = time.perf_counter() - t0
                text = "".join(output_parts)
                if not usage["completion_tokens"] and text:
                    usage["completion_tokens"] = estimate_tokens(text)
                    token_source = "estimated"
                if not usage["prompt_tokens"]:
                    usage["prompt_tokens"] = estimate_tokens(prompt_text)
                    token_source = "estimated"
                if not usage["total_tokens"]:
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
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
                last_exc = exc
                if include_usage and should_retry_without_stream_options(exc):
                    continue
                break

        # If streaming is rejected by the endpoint, keep the run usable by
        # falling back to a normal completion. TTFT is unavailable in this mode.
        stream_error = str(last_exc) if last_exc else "streaming failed"
    else:
        stream_error = ""

    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        latency_s = time.perf_counter() - t0
        choices = get_attr(response, "choices", []) or []
        text = ""
        if choices:
            text = get_attr(get_attr(choices[0], "message", None), "content", "") or ""
        usage = parse_usage(get_attr(response, "usage", None))
        token_source = "usage"
        if not usage["completion_tokens"] and text:
            usage["completion_tokens"] = estimate_tokens(text)
            token_source = "estimated"
        if not usage["prompt_tokens"]:
            usage["prompt_tokens"] = estimate_tokens(prompt_text)
            token_source = "estimated"
        if not usage["total_tokens"]:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
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
            "usage": usage,
            "token_source": token_source,
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
    messages = build_messages(workload, prompt, request_index)
    started_epoch = time.time()
    result = chat_completion(
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


def accuracy_prompt(item: AccuracyItem) -> str:
    if item.answer_type == "choice":
        assert item.choices is not None
        choice_keys = sorted(str(key).upper() for key in item.choices)
        allowed = ", ".join(choice_keys)
        choices = "\n".join(f"{key}. {value}" for key, value in sorted(item.choices.items()))
        return (
            f"Answer the multiple-choice question. Return only the letter {allowed}.\n\n"
            f"Question: {item.question}\n{choices}"
        )
    if item.answer_type == "numeric":
        return (
            "Solve the problem. Return only the final numeric answer, with no explanation.\n\n"
            f"Question: {item.question}"
        )
    return f"Answer the question concisely.\n\nQuestion: {item.question}"


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def extract_choice(value: str, valid_choices: Optional[Iterable[str]] = None) -> str:
    valid = [str(choice).upper() for choice in (valid_choices or ["A", "B", "C", "D"])]
    escaped = "|".join(re.escape(choice) for choice in sorted(valid, key=len, reverse=True))
    match = re.search(rf"\b({escaped})\b", value.upper()) if escaped else None
    return match.group(1) if match else ""


def extract_number(value: str) -> str:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return numbers[-1] if numbers else ""


def score_answer(item: AccuracyItem, output: str) -> Tuple[str, bool]:
    if item.answer_type == "choice":
        prediction = extract_choice(output, item.choices.keys() if item.choices else None)
        return prediction, prediction == item.answer.upper()
    if item.answer_type == "numeric":
        prediction = extract_number(output)
        if not prediction:
            return prediction, False
        try:
            return prediction, math.isclose(float(prediction), float(item.answer), rel_tol=0, abs_tol=1e-6)
        except ValueError:
            return prediction, False
    prediction = normalize_text(output)
    return prediction, prediction == normalize_text(item.answer)


def load_accuracy_jsonl(path: Path) -> List[AccuracyItem]:
    items: List[AccuracyItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            choices = row.get("choices")
            if isinstance(choices, list):
                choices = {chr(ord("A") + idx): str(value) for idx, value in enumerate(choices)}
            items.append(
                AccuracyItem(
                    suite=str(row.get("suite", "custom")),
                    item_id=str(row.get("id", row.get("item_id", f"line_{line_no}"))),
                    category=str(row.get("category", "general")),
                    question=str(row["question"]),
                    choices=choices,
                    answer=str(row["answer"]),
                    answer_type=str(row.get("answer_type", "choice" if choices else "text")),
                )
            )
    return items


def run_one_accuracy_item(
    client: Any,
    model: str,
    item: AccuracyItem,
    max_tokens: int,
) -> Dict[str, Any]:
    prompt = accuracy_prompt(item)
    started_epoch = time.time()
    result = chat_completion(
        client=client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
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


def workloads_for_profile(profile: str, include_vision: bool) -> List[Workload]:
    if profile == "smoke":
        workloads = [
            Workload("text_latency_smoke", "text", 2, 1, 128, 128),
            Workload("text_throughput_smoke", "text", 8, 4, 128, 128),
        ]
        if include_vision:
            workloads.extend(
                [
                    Workload("vision_latency_smoke", "vision", 1, 1, 128, 128, 2, "720p"),
                    Workload("vision_throughput_smoke", "vision", 2, 2, 128, 128, 2, "720p"),
                ]
            )
        return workloads

    if profile == "sglang-text":
        workloads = [
            Workload("text_latency", "text", 10, 1, 1024, 1024),
            Workload("text_throughput", "text", 1000, 100, 1024, 1024),
        ]
        if include_vision:
            workloads.extend(
                [
                    Workload("vision_latency", "vision", 10, 1, 128, 1024, 2, "720p"),
                    Workload("vision_throughput", "vision", 1000, 100, 128, 1024, 2, "720p"),
                ]
            )
        return workloads

    if profile == "sglang-full":
        return [
            Workload("text_latency", "text", 10, 1, 1024, 1024),
            Workload("text_throughput", "text", 1000, 100, 1024, 1024),
            Workload("vision_latency", "vision", 10, 1, 128, 1024, 2, "720p"),
            Workload("vision_throughput", "vision", 1000, 100, 128, 1024, 2, "720p"),
        ]

    raise ValueError(f"Unknown profile: {profile}")


def override_workloads(
    workloads: Iterable[Workload],
    num_prompts: Optional[int],
    max_concurrency: Optional[int],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    image_resolution: Optional[str],
) -> List[Workload]:
    updated = []
    for workload in workloads:
        next_workload = workload
        if num_prompts is not None:
            next_workload = replace(next_workload, num_prompts=num_prompts)
        if max_concurrency is not None:
            next_workload = replace(next_workload, max_concurrency=max_concurrency)
        if input_tokens is not None:
            next_workload = replace(next_workload, random_input_len=input_tokens)
        if output_tokens is not None:
            next_workload = replace(next_workload, max_output_tokens=output_tokens)
        if image_resolution is not None and next_workload.modality == "vision":
            next_workload = replace(next_workload, image_resolution=image_resolution)
        updated.append(next_workload)
    return updated


def group_by(rows: Iterable[Dict[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        groups.setdefault(key, []).append(row)
    return groups


def summarize_performance(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for (model, workload), group in group_by(rows, ["model", "workload"]).items():
        ok = [row for row in group if row.get("ok")]
        errors = len(group) - len(ok)
        batch_wall_s = max((float(row.get("batch_wall_s", 0.0)) for row in group), default=0.0)
        completion_tokens = [float(row.get("completion_tokens", 0.0)) for row in ok]
        prompt_tokens = [float(row.get("prompt_tokens", 0.0)) for row in ok]
        ttft = [float(row.get("ttft_ms", 0.0)) for row in ok]
        latency = [float(row.get("latency_ms", 0.0)) for row in ok]
        output_tps = [float(row.get("output_tps", 0.0)) for row in ok]
        total_completion_tokens = sum(completion_tokens)
        summaries.append(
            {
                "model": model,
                "workload": workload,
                "modality": group[0].get("modality", ""),
                "requests": len(group),
                "successful_requests": len(ok),
                "errors": errors,
                "max_concurrency": group[0].get("max_concurrency", ""),
                "batch_wall_s": round(batch_wall_s, 3),
                "request_rps": round((len(ok) / batch_wall_s) if batch_wall_s else 0.0, 3),
                "aggregate_output_tps": round(
                    (total_completion_tokens / batch_wall_s) if batch_wall_s else 0.0, 3
                ),
                "ttft_mean_ms": round(statistics.mean(ttft), 3) if ttft else 0.0,
                "ttft_p50_ms": round(percentile(ttft, 0.50), 3),
                "latency_mean_ms": round(statistics.mean(latency), 3) if latency else 0.0,
                "latency_p50_ms": round(percentile(latency, 0.50), 3),
                "latency_p90_ms": round(percentile(latency, 0.90), 3),
                "latency_p99_ms": round(percentile(latency, 0.99), 3),
                "per_request_output_tps_mean": round(statistics.mean(output_tps), 3)
                if output_tps
                else 0.0,
                "prompt_tokens_mean": round(statistics.mean(prompt_tokens), 3)
                if prompt_tokens
                else 0.0,
                "completion_tokens_mean": round(statistics.mean(completion_tokens), 3)
                if completion_tokens
                else 0.0,
            }
        )
    return sorted(summaries, key=lambda row: (row["model"], row["workload"]))


def summarize_accuracy(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for key in (["model", "suite"], ["model", "suite", "category"]):
        for group_key, group in group_by(rows, key).items():
            ok = [row for row in group if row.get("ok")]
            correct = [row for row in ok if row.get("correct")]
            summary = {field: value for field, value in zip(key, group_key)}
            summary.update(
                {
                    "items": len(group),
                    "successful_items": len(ok),
                    "correct": len(correct),
                    "accuracy": round((len(correct) / len(ok)) if ok else 0.0, 4),
                    "latency_mean_ms": round(
                        statistics.mean(float(row.get("latency_ms", 0.0)) for row in ok), 3
                    )
                    if ok
                    else 0.0,
                }
            )
            summaries.append(summary)
    return sorted(summaries, key=lambda row: tuple(str(row.get(k, "")) for k in ("model", "suite", "category")))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    if not rows:
        print("(no rows)")
        return
    widths = {
        field: max(len(field), *(len(str(row.get(field, ""))) for row in rows))
        for field in fields
    }
    header = "  ".join(field.ljust(widths[field]) for field in fields)
    print(header)
    print("  ".join("-" * widths[field] for field in fields))
    for row in rows:
        print("  ".join(str(row.get(field, "")).ljust(widths[field]) for field in fields))


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} before running the benchmark")

    root_url = resolve_mantle_root_url(args.base_url, args.region)
    clients: Dict[str, Any] = {}

    def client_for(model: str) -> Any:
        url = mantle_generation_base_url(root_url, model)
        if url not in clients:
            clients[url] = load_openai_client(
                api_key=api_key,
                base_url=url,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
        return clients[url]

    if args.list_models:
        try:
            model_list_client = load_openai_client(
                api_key=api_key,
                base_url=f"{root_url}/v1",
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            available = list_models(model_list_client)
            print("\nAvailable Mantle models:")
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

    workloads = workloads_for_profile(args.profile, args.include_vision)
    workloads = override_workloads(
        workloads,
        num_prompts=args.num_prompts,
        max_concurrency=args.max_concurrency,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        image_resolution=args.image_resolution,
    )

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
                        client=client_for(model),
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
                    client=client_for(model),
                    model=model,
                    items=accuracy_items,
                    max_tokens=args.accuracy_max_tokens,
                    max_concurrency=args.accuracy_concurrency,
                )
            )

    perf_summary = summarize_performance(perf_rows)
    accuracy_summary = summarize_accuracy(accuracy_rows)

    paths = {
        "performance_csv": str(out_dir / f"mantle_gemma_performance_{run_id}.csv"),
        "performance_summary_csv": str(out_dir / f"mantle_gemma_performance_summary_{run_id}.csv"),
        "accuracy_csv": str(out_dir / f"mantle_gemma_accuracy_{run_id}.csv"),
        "accuracy_summary_csv": str(out_dir / f"mantle_gemma_accuracy_summary_{run_id}.csv"),
        "manifest_json": str(out_dir / f"mantle_gemma_manifest_{run_id}.json"),
        "latest_manifest_json": str(out_dir / "latest_mantle_gemma_manifest.json"),
    }

    write_csv(Path(paths["performance_csv"]), perf_rows)
    write_csv(Path(paths["performance_summary_csv"]), perf_summary)
    write_csv(Path(paths["accuracy_csv"]), accuracy_rows)
    write_csv(Path(paths["accuracy_summary_csv"]), accuracy_summary)

    manifest = {
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": root_url,
        "model_base_urls": {model: mantle_generation_base_url(root_url, model) for model in args.models},
        "model_list_base_url": f"{root_url}/v1",
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
        description="Benchmark Gemma 4 31B and 26B-a4b on AWS Mantle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument(
        "--base-url",
        default="",
        help="Mantle OpenAI-compatible generation base URL. Empty derives /openai/v1 from --region.",
    )
    parser.add_argument("--api-key-env", default="BEDROCK_MANTLE_TOKEN")
    parser.add_argument("--out-dir", default="results/mantle_gemma")
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
    parser.add_argument("--accuracy-max-tokens", type=int, default=32)
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
