"""
LLM-as-a-Judge Quality Evaluation Script.

Evaluates whether watermarking degrades text quality using an LLM judge.
Supports two evaluation modes:

  1. Pointwise — Score each response on a 1-5 rubric per quality dimension.
     Good for parameter sweeps and absolute quality assessment.
  2. Pairwise — Compare watermarked vs. non-watermarked responses side-by-side.
     Better for detecting subtle quality degradation. Uses position swapping
     to eliminate judge position bias.

Judge backends:
  - openai  — OpenAI-compatible API (OpenAI, OpenRouter, Together, etc.)
  - local   — Local HuggingFace model via AutoModelForCausalLM

Usage examples:
  python scripts/llm_judge.py \
      --mode pointwise \
      --watermarked-text output/wm_experiment/ \
      --judge-backend openai --judge-model gpt-4o-mini \
      --api-base-url https://openrouter.ai/api/v1 \
      --dimensions coherence relevance fluency overall \
      --n-texts 100 --verbose

  python scripts/llm_judge.py \
      --mode pairwise \
      --no-watermark-text output/no_wm_experiment/ \
      --watermarked-text output/wm_experiment/ \
      --judge-backend openai --judge-model gpt-4o-mini \
      --n-texts 100

  python scripts/llm_judge.py \
      --mode pointwise \
      --watermarked-text output/wm_experiment/ \
      --judge-backend local --judge-model Qwen/Qwen2.5-0.5B-Instruct \
      --n-texts 50
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.utils import load_experiment, parse_text_source, source_description


# ---------------------------------------------------------------------------
# Quality dimensions & prompt templates
# ---------------------------------------------------------------------------

DIMENSIONS = {
    "coherence": {
        "description": "internal logical consistency and flow of the response",
        "rubric": {
            1: "Incoherent, with contradictions and disjointed sentences",
            2: "Mostly incoherent, hard to follow the main point",
            3: "Somewhat coherent with notable lapses in logic or flow",
            4: "Mostly coherent with minor issues",
            5: "Fully coherent, well-structured, and easy to follow",
        },
    },
    "relevance": {
        "description": "how well the response addresses the given prompt",
        "rubric": {
            1: "Completely off-topic, unrelated to the prompt",
            2: "Mostly off-topic with tangential relevance",
            3: "Partially relevant but misses key aspects",
            4: "Mostly relevant with minor deviations",
            5: "Directly and fully relevant to the prompt",
        },
    },
    "accuracy": {
        "description": "factual correctness of claims in the response",
        "rubric": {
            1: "Multiple clear factual errors",
            2: "Several factual errors present",
            3: "Some errors but mostly plausible claims",
            4: "Minor errors or unverified claims only",
            5: "No apparent factual errors",
        },
    },
    "instruction_following": {
        "description": "how well the response follows instructions or constraints in the prompt",
        "rubric": {
            1: "Ignores the instructions entirely",
            2: "Mostly ignores instructions, does something else",
            3: "Partially follows instructions with notable omissions",
            4: "Mostly follows instructions with minor deviations",
            5: "Perfectly follows all instructions and constraints",
        },
    },
    "fluency": {
        "description": "grammatical correctness and natural language quality",
        "rubric": {
            1: "Very poor grammar, essentially unreadable",
            2: "Many grammatical errors, awkward phrasing",
            3: "Some grammatical errors but generally readable",
            4: "Minor issues only, reads naturally",
            5: "Perfectly fluent with no grammatical issues",
        },
    },
    "overall": {
        "description": "overall quality considering all aspects",
        "rubric": {
            1: "Very poor quality",
            2: "Poor quality with significant issues",
            3: "Acceptable quality with some problems",
            4: "Good quality with minor issues",
            5: "Excellent quality across all dimensions",
        },
    },
    "generic_authenticity": {
        "description": "how believable the text is as genuine human-written text in general",
        "rubric": {
            1: "Clearly synthetic or AI-generated, reads nothing like a real person wrote it",
            2: "Mostly artificial, has obvious giveaways that it is machine-generated",
            3: "Somewhat believable but with notable signs of being generated or generic",
            4: "Mostly authentic, reads like a real person with minor oddities",
            5: "Indistinguishable from genuine human-written text",
        },
    },
    "social_authenticity": {
        "description": "how believable the text is as a genuine human-written product review or social media post",
        "rubric": {
            1: "Clearly synthetic or AI-generated, sounds nothing like a real user review or post",
            2: "Mostly artificial, has obvious giveaways that it is not from a real person",
            3: "Somewhat believable but with notable signs of being generated or generic",
            4: "Mostly authentic, reads like a real person with minor oddities",
            5: "Indistinguishable from a genuine human product review or social media post",
        },
    },
}

_RATING_RE = re.compile(r"\[\[(\d+)\]\]")
_CHOICE_RE = re.compile(r"\[\[([ABC])\]\]")

_POINTWISE_SYSTEM = (
    "You are an impartial judge evaluating AI-generated text quality. "
    "Provide objective, consistent ratings based on the rubric."
)

_POINTWISE_USER = """\
Evaluate the following response on the dimension of {dimension}: {description}

Rubric:
1 — {r1}
2 — {r2}
3 — {r3}
4 — {r4}
5 — {r5}

First provide brief reasoning (1-2 sentences), then your rating as [[N]] where N is 1-5.

Prompt:
{prompt}

Response:
{response}"""

_PAIRWISE_SYSTEM = (
    "You are an impartial judge comparing two AI-generated responses. "
    "Determine which is better on the specified dimension."
)

_PAIRWISE_USER = """\
Which response is better on the dimension of {dimension}: {description}?

Prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Respond with [[A]], [[B]], or [[C]] (tie / both equally good). First provide brief reasoning (1-2 sentences)."""


def build_pointwise_messages(prompt: str, response: str, dimension: str) -> list[dict]:
    dim = DIMENSIONS[dimension]
    rubric = dim["rubric"]
    user_content = _POINTWISE_USER.format(
        dimension=dimension,
        description=dim["description"],
        r1=rubric[1], r2=rubric[2], r3=rubric[3], r4=rubric[4], r5=rubric[5],
        prompt=prompt,
        response=response,
    )
    return [
        {"role": "system", "content": _POINTWISE_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def build_pairwise_messages(
    prompt: str, response_a: str, response_b: str, dimension: str,
) -> list[dict]:
    dim = DIMENSIONS[dimension]
    user_content = _PAIRWISE_USER.format(
        dimension=dimension,
        description=dim["description"],
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
    )
    return [
        {"role": "system", "content": _PAIRWISE_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def parse_rating(raw: str) -> int | None:
    m = _RATING_RE.search(raw)
    if m:
        val = int(m.group(1))
        return val if 1 <= val <= 5 else None
    return None


def parse_choice(raw: str) -> str | None:
    m = _CHOICE_RE.search(raw)
    if m:
        return m.group(1)
    return None


def _map_choice(choice: str | None) -> str | None:
    if choice is None:
        return None
    return {"A": "a", "B": "b", "C": "tie"}.get(choice, None)


def resolve_with_swap(choice_orig: str | None, choice_swap: str | None) -> str:
    """Resolve pairwise choices from original and swapped positions.

    In the swapped run, A and B are physically swapped, so we flip the
    swapped choice back. If both agree, that's the winner; otherwise it's a tie
    (position bias detected).

    Returns: "a", "b", or "tie"
    """
    orig = _map_choice(choice_orig)
    swap = _map_choice(choice_swap)

    if orig is None or swap is None:
        return "tie"

    swap_flipped = {"a": "b", "b": "a", "tie": "tie"}[swap]

    if orig == swap_flipped:
        return orig
    return "tie"


# ---------------------------------------------------------------------------
# Judge backends
# ---------------------------------------------------------------------------

class JudgeBackend(ABC):
    @abstractmethod
    def judge(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024) -> tuple[str, int, int]:
        """Send messages to the judge. Returns (response_text, input_tokens, output_tokens)."""


class OpenAICompatibleBackend(JudgeBackend):
    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        from openai import OpenAI
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def judge(self, messages, temperature=0.0, max_tokens=1024):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        inp = getattr(response.usage, "prompt_tokens", 0) or 0
        out = getattr(response.usage, "completion_tokens", 0) or 0
        return text, inp, out


class LocalHFBackend(JudgeBackend):
    def __init__(self, model: str, load_in_4bit: bool = False, load_in_8bit: bool = False):
        import torch
        from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        architectures = getattr(config, "architectures", []) or []
        is_multimodal = any("ConditionalGeneration" in arch for arch in architectures)

        kwargs = {"torch_dtype": "auto"}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = "auto"
        elif load_in_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = "auto"

        if is_multimodal:
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(model, trust_remote_code=True, **kwargs)
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, **kwargs)
        self.device = next(self.model.parameters()).device

    def judge(self, messages, temperature=0.0, max_tokens=1024):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        inp_tokens = inputs.input_ids.shape[1]

        gen_kwargs = {"max_new_tokens": max_tokens}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        outputs = self.model.generate(**inputs, **gen_kwargs)
        response_text = self.tokenizer.decode(outputs[0][inp_tokens:], skip_special_tokens=True)
        out_tokens = outputs.shape[1] - inp_tokens
        return response_text, inp_tokens, out_tokens


class CostTracker:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.api_calls = 0
        self.errors = 0
        self._lock = threading.Lock()

    def record(self, input_tokens: int, output_tokens: int):
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.api_calls += 1

    def inc_errors(self):
        with self._lock:
            self.errors += 1

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "api_calls": self.api_calls,
            "errors": self.errors,
        }


class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.interval = 60.0 / requests_per_minute
        self.last_call = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            elapsed = time.time() - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()


def judge_with_retry(
    backend: JudgeBackend,
    messages: list[dict],
    max_retries: int = 3,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> tuple[str, int, int, bool]:
    """Call judge with exponential backoff retry. Returns (text, inp, out, success)."""
    for attempt in range(max_retries):
        try:
            text, inp, out = backend.judge(messages, temperature, max_tokens)
            return text, inp, out, True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}", 0, 0, False


# ---------------------------------------------------------------------------
# Loading prompt-response pairs
# ---------------------------------------------------------------------------

def _load_pairs(source: dict, n_texts: int | None) -> list[dict]:
    """Load prompt-response pairs from an experiment directory.

    Returns list of {"prompt": str, "response": str, "index": int}.
    Only experiment directory sources are supported (not HF datasets),
    since judge evaluation requires both prompt and response text.
    """
    if source["type"] != "directory":
        raise ValueError(
            f"LLM-as-a-Judge requires experiment directory sources (got {source['type']}). "
            f"Provide a path to an experiment output directory with generations.csv."
        )

    experiment = load_experiment(source["path"])
    df = experiment["generations"]
    mask = df["generated_text"].notna() & (df["generated_text"].astype(str) != "")

    if "prompt_text" not in df.columns:
        raise ValueError(
            f"Experiment at {source['path']} lacks 'prompt_text' column in generations.csv. "
            f"Re-run the experiment to generate it."
        )

    pairs = []
    for i, (_, row) in enumerate(df.loc[mask].iterrows()):
        if n_texts is not None and i >= n_texts:
            break
        pairs.append({
            "prompt": str(row["prompt_text"]),
            "response": str(row["generated_text"]),
            "index": i,
        })
    return pairs


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def evaluate_pointwise(
    pairs: list[dict],
    condition_name: str,
    backend: JudgeBackend,
    dimensions: list[str],
    rate_limiter: RateLimiter,
    cost_tracker: CostTracker,
    max_concurrent: int = 4,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    verbose: bool = False,
) -> list[dict]:
    results = []

    def _evaluate_one(idx_dim: tuple[int, str]):
        idx, dim = idx_dim
        pair = pairs[idx]
        messages = build_pointwise_messages(pair["prompt"], pair["response"], dim)
        rate_limiter.wait()
        raw, inp, out, success = judge_with_retry(backend, messages, temperature=temperature, max_tokens=max_tokens)
        cost_tracker.record(inp, out)
        if not success:
            cost_tracker.inc_errors()
        rating = parse_rating(raw) if success else None
        return {
            "index": pair["index"],
            "condition": condition_name,
            "dimension": dim,
            "rating": rating,
            "raw_response": raw,
        }

    tasks = [(i, dim) for i in range(len(pairs)) for dim in dimensions]

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {pool.submit(_evaluate_one, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Pointwise evaluation", disable=verbose):
            result = future.result()
            results.append(result)
            if verbose:
                rating_str = str(result["rating"]) if result["rating"] is not None else "PARSE_FAIL"
                print(f"  [{result['condition']}] idx={result['index']} dim={result['dimension']} rating={rating_str}")

    results.sort(key=lambda r: (r["index"], r["dimension"]))
    return results


def evaluate_pairwise(
    pairs_a: list[dict],
    pairs_b: list[dict],
    name_a: str,
    name_b: str,
    backend: JudgeBackend,
    dimensions: list[str],
    rate_limiter: RateLimiter,
    cost_tracker: CostTracker,
    swap: bool = True,
    max_concurrent: int = 4,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    verbose: bool = False,
) -> list[dict]:
    assert len(pairs_a) == len(pairs_b), f"Pair counts must match: {len(pairs_a)} vs {len(pairs_b)}"
    results = []

    def _evaluate_one(idx_dim: tuple[int, str]):
        idx, dim = idx_dim
        pa, pb = pairs_a[idx], pairs_b[idx]

        # Original order: A vs B
        messages_orig = build_pairwise_messages(pa["prompt"], pa["response"], pb["response"], dim)
        rate_limiter.wait()
        raw_orig, inp1, out1, ok1 = judge_with_retry(backend, messages_orig, temperature=temperature, max_tokens=max_tokens)
        cost_tracker.record(inp1, out1)
        if not ok1:
            cost_tracker.inc_errors()
        choice_orig = parse_choice(raw_orig) if ok1 else None

        # Swapped order: B vs A
        raw_swap = None
        choice_swap = None
        if swap:
            messages_swap = build_pairwise_messages(pa["prompt"], pb["response"], pa["response"], dim)
            rate_limiter.wait()
            raw_swap, inp2, out2, ok2 = judge_with_retry(backend, messages_swap, temperature=temperature, max_tokens=max_tokens)
            cost_tracker.record(inp2, out2)
            if not ok2:
                cost_tracker.inc_errors()
            choice_swap = parse_choice(raw_swap) if ok2 else None

        winner = resolve_with_swap(choice_orig, choice_swap) if swap else (_map_choice(choice_orig) or "tie")

        return {
            "index": pa["index"],
            "condition_a": name_a,
            "condition_b": name_b,
            "dimension": dim,
            "winner": winner,
            "choice_orig": choice_orig,
            "choice_swap": choice_swap,
            "raw_orig": raw_orig,
            "raw_swap": raw_swap,
        }

    tasks = [(i, dim) for i in range(len(pairs_a)) for dim in dimensions]

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {pool.submit(_evaluate_one, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Pairwise evaluation", disable=verbose):
            result = future.result()
            results.append(result)
            if verbose:
                print(
                    f"  idx={result['index']} dim={result['dimension']} "
                    f"winner={result['winner']} orig={result['choice_orig']} swap={result['choice_swap']}"
                )

    results.sort(key=lambda r: (r["index"], r["dimension"]))
    return results


# ---------------------------------------------------------------------------
# Output & visualization
# ---------------------------------------------------------------------------

_CONDITION_ORDER = ["human", "no_watermark", "watermarked"]
_CONDITION_LABELS = {
    "human": "Human",
    "no_watermark": "No Watermark",
    "watermarked": "Watermarked",
}
_COLORS = ["#4DAF4A", "#377EB8", "#E41A1C"]


def _fmt(v) -> str:
    fv = float(v)
    return f"{fv:>10.3f}" if not math.isnan(fv) else f"{'nan':>10}"


def _plot_pointwise(
    results: list[dict], run_dir: str, dimensions: list[str],
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return None

    df = pd.DataFrame(results)
    conditions_present = [c for c in _CONDITION_ORDER if c in df["condition"].unique()]
    labels = [_CONDITION_LABELS[c] for c in conditions_present]
    colors = [_COLORS[_CONDITION_ORDER.index(c)] for c in conditions_present]

    n_dims = len(dimensions)
    fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 5))
    if n_dims == 1:
        axes = [axes]

    for ax, dim in zip(axes, dimensions):
        dim_df = df[df["dimension"] == dim]
        values = []
        for cond in conditions_present:
            vals = dim_df.loc[dim_df["condition"] == cond, "rating"].dropna().values.astype(float)
            values.append(vals)

        bp = ax.boxplot(
            values, tick_labels=labels, showfliers=False,
            patch_artist=True, widths=0.5,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for i, vals in enumerate(values):
            if len(vals) > 0:
                ax.annotate(
                    f"{np.median(vals):.2f}", xy=(i + 1.3, float(np.median(vals))),
                    fontsize=8, ha="left", va="center",
                )

        ax.set_title(dim.capitalize(), fontsize=10)
        ax.set_ylim(0.5, 5.5)
        ax.set_ylabel("Rating" if dim == dimensions[0] else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("LLM-as-a-Judge: Pointwise Evaluation", fontsize=13)
    fig.tight_layout()

    output_path = os.path.join(run_dir, "llm_judge.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_pairwise(
    results: list[dict], run_dir: str, dimensions: list[str],
    name_a: str, name_b: str,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return None

    df = pd.DataFrame(results)
    bar_colors = ["#377EB8", "#E41A1C", "#BDBDBD"]
    bar_labels = [name_a, name_b, "Tie"]

    n_dims = len(dimensions)
    fig, ax = plt.subplots(figsize=(2.5 * n_dims + 2, 5))

    x = np.arange(n_dims)
    width = 0.25

    for i, (bar_label, color) in enumerate(zip(bar_labels, bar_colors)):
        rates = []
        for dim in dimensions:
            dim_df = df[df["dimension"] == dim]
            total = len(dim_df)
            if total == 0:
                rates.append(0)
            elif i == 0:
                rates.append((dim_df["winner"] == "a").sum() / total * 100)
            elif i == 1:
                rates.append((dim_df["winner"] == "b").sum() / total * 100)
            else:
                rates.append((dim_df["winner"] == "tie").sum() / total * 100)

        bars = ax.bar(x + i * width, rates, width, label=bar_label, color=color, alpha=0.8)
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{rate:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([d.capitalize() for d in dimensions])
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("LLM-as-a-Judge: Pairwise Evaluation", fontsize=13)
    fig.tight_layout()

    output_path = os.path.join(run_dir, "llm_judge.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_config(run_dir: str, args: argparse.Namespace, source_descriptions: dict[str, str]):
    config = {
        "mode": args.mode,
        "judge": {
            "backend": args.judge_backend,
            "model": args.judge_model,
            "temperature": args.judge_temperature,
            "max_tokens": args.judge_max_tokens,
        },
        "evaluation": {
            "dimensions": args.dimensions,
            "n_texts": args.n_texts,
            "seed": args.seed,
        },
        "sources": source_descriptions,
    }
    if args.mode == "pairwise":
        config["evaluation"]["position_swap"] = not args.no_swap
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def _output_results_pointwise(
    results: list[dict],
    source_descriptions: dict[str, str],
    args: argparse.Namespace,
    cost_tracker: CostTracker,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_clean = args.judge_model.replace("/", "_")
    run_dir = os.path.join(args.save_dir, f"llm_judge_pointwise_{model_clean}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    _save_config(run_dir, args, source_descriptions)

    df = pd.DataFrame(results)

    # Per-condition CSVs
    for condition in df["condition"].unique():
        cond_df = df[df["condition"] == condition]
        cond_df.to_csv(os.path.join(run_dir, f"{condition}.csv"), index=False)

    # Summary
    summary_records = []
    for condition in df["condition"].unique():
        for dim in args.dimensions:
            mask = (df["condition"] == condition) & (df["dimension"] == dim) & df["rating"].notna()
            vals = df.loc[mask, "rating"].values.astype(float)
            summary_records.append({
                "condition": condition,
                "dimension": dim,
                "count": int(((df["condition"] == condition) & (df["dimension"] == dim)).sum()),
                "valid": int(mask.sum()),
                "mean": float(np.mean(vals)) if len(vals) > 0 else float("nan"),
                "median": float(np.median(vals)) if len(vals) > 0 else float("nan"),
                "std": float(np.std(vals)) if len(vals) > 0 else float("nan"),
                "min": float(np.min(vals)) if len(vals) > 0 else float("nan"),
                "max": float(np.max(vals)) if len(vals) > 0 else float("nan"),
            })

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    # Text report
    lines = [
        "=" * 70,
        "LLM-as-a-Judge: Pointwise Evaluation Results",
        "=" * 70,
        "",
        f"Judge backend:  {args.judge_backend}",
        f"Judge model:    {args.judge_model}",
        f"Dimensions:     {', '.join(args.dimensions)}",
        f"N texts:        {args.n_texts}",
        f"Seed:           {args.seed}",
        "",
    ]
    for cond, desc in source_descriptions.items():
        lines.append(f"  {cond}: {desc}")
    lines.append("")

    for dim in args.dimensions:
        dim_rows = summary_df[summary_df["dimension"] == dim]
        lines.append(f"--- {dim.capitalize()} ---")
        lines.append("")
        lines.append(
            f"{'Condition':<16} {'Valid':>6} {'Mean':>10} "
            f"{'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10}"
        )
        lines.append("-" * 70)
        for _, row in dim_rows.iterrows():
            lines.append(
                f"{row['condition']:<16} {int(row['valid']):>6} "
                f"{_fmt(row['mean'])} {_fmt(row['median'])} {_fmt(row['std'])} "
                f"{_fmt(row['min'])} {_fmt(row['max'])}"
            )
        lines.append("")

    cost = cost_tracker.to_dict()
    lines.append("=" * 70)
    lines.append("Cost Summary")
    lines.append("=" * 70)
    lines.append(f"  API calls:       {cost['api_calls']}")
    lines.append(f"  Input tokens:    {cost['input_tokens']}")
    lines.append(f"  Output tokens:   {cost['output_tokens']}")
    lines.append(f"  Errors:          {cost['errors']}")
    lines.append("=" * 70)

    text_output = "\n".join(lines)
    print(text_output)

    with open(os.path.join(run_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(text_output)

    with open(os.path.join(run_dir, "cost.json"), "w") as f:
        json.dump(cost, f, indent=2)

    plot_path = _plot_pointwise(results, run_dir, args.dimensions)
    if plot_path:
        print(f"Plot saved to:    {plot_path}")
    print(f"Results saved to: {run_dir}")
    return run_dir


def _output_results_pairwise(
    results: list[dict],
    source_descriptions: dict[str, str],
    args: argparse.Namespace,
    cost_tracker: CostTracker,
    name_a: str,
    name_b: str,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_clean = args.judge_model.replace("/", "_")
    run_dir = os.path.join(args.save_dir, f"llm_judge_pairwise_{model_clean}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    _save_config(run_dir, args, source_descriptions)

    df = pd.DataFrame(results)
    csv_name = f"pairwise_{name_a}_vs_{name_b}.csv"
    df.to_csv(os.path.join(run_dir, csv_name), index=False)

    # Summary
    summary_records = []
    for dim in args.dimensions:
        dim_df = df[df["dimension"] == dim]
        total = len(dim_df)
        win_a = (dim_df["winner"] == "a").sum()
        win_b = (dim_df["winner"] == "b").sum()
        ties = (dim_df["winner"] == "tie").sum()
        position_bias = 0
        if "choice_orig" in dim_df.columns and "choice_swap" in dim_df.columns:
            valid = dim_df[dim_df["choice_orig"].notna() & dim_df["choice_swap"].notna()]
            for _, row in valid.iterrows():
                orig = _map_choice(row["choice_orig"])
                swap = _map_choice(row["choice_swap"])
                if orig is not None and swap is not None:
                    swap_flipped = {"a": "b", "b": "a", "tie": "tie"}[swap]
                    if orig != swap_flipped:
                        position_bias += 1

        summary_records.append({
            "dimension": dim,
            "win_rate_a": win_a / total * 100 if total > 0 else float("nan"),
            "win_rate_b": win_b / total * 100 if total > 0 else float("nan"),
            "tie_rate": ties / total * 100 if total > 0 else float("nan"),
            "n_pairs": total,
            "n_position_bias": position_bias,
        })

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    # Text report
    lines = [
        "=" * 70,
        "LLM-as-a-Judge: Pairwise Evaluation Results",
        "=" * 70,
        "",
        f"Judge backend:  {args.judge_backend}",
        f"Judge model:    {args.judge_model}",
        f"Dimensions:     {', '.join(args.dimensions)}",
        f"N texts:        {args.n_texts}",
        f"Position swap:  {'disabled' if args.no_swap else 'enabled'}",
        "",
    ]
    for cond, desc in source_descriptions.items():
        lines.append(f"  {cond}: {desc}")
    lines.append("")

    lines.append(f"{'Dimension':<20} {name_a + ' wins':>12} {name_b + ' wins':>12} {'Tie':>8} {'Pos. bias':>10}")
    lines.append("-" * 70)
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['dimension'].capitalize():<20} "
            f"{row['win_rate_a']:>11.1f}% {row['win_rate_b']:>11.1f}% "
            f"{row['tie_rate']:>7.1f}% {int(row['n_position_bias']):>10}"
        )

    cost = cost_tracker.to_dict()
    lines.append("")
    lines.append("=" * 70)
    lines.append("Cost Summary")
    lines.append("=" * 70)
    lines.append(f"  API calls:       {cost['api_calls']}")
    lines.append(f"  Input tokens:    {cost['input_tokens']}")
    lines.append(f"  Output tokens:   {cost['output_tokens']}")
    lines.append(f"  Errors:          {cost['errors']}")
    lines.append("=" * 70)

    text_output = "\n".join(lines)
    print(text_output)

    with open(os.path.join(run_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(text_output)

    with open(os.path.join(run_dir, "cost.json"), "w") as f:
        json.dump(cost, f, indent=2)

    plot_path = _plot_pairwise(results, run_dir, args.dimensions, name_a, name_b)
    if plot_path:
        print(f"Plot saved to:    {plot_path}")
    print(f"Results saved to: {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge quality evaluation for watermarked text",
    )

    # Text sources
    parser.add_argument(
        "--human-text", type=str, nargs="+", default=None,
        help="Human text source (experiment dir or 4-arg HF dataset spec)",
    )
    parser.add_argument(
        "--no-watermark-text", type=str, nargs="+", default=None,
        help="No-watermark text source (experiment dir or 4-arg HF dataset spec)",
    )
    parser.add_argument(
        "--watermarked-text", type=str, nargs="+", default=None,
        help="Watermarked text source (experiment dir or 4-arg HF dataset spec)",
    )

    # Mode
    parser.add_argument(
        "--mode", type=str, required=True, choices=["pointwise", "pairwise"],
        help="Evaluation mode: pointwise (1-5 scoring) or pairwise (A/B comparison)",
    )

    # Judge configuration
    parser.add_argument(
        "--judge-backend", type=str, required=True, choices=["openai", "local"],
        help="Judge backend: openai (API) or local (HF model)",
    )
    parser.add_argument(
        "--judge-model", type=str, required=True,
        help="Judge model ID (e.g., gpt-4o-mini, Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key (falls back to OPENROUTER_API_KEY or OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--api-base-url", type=str, default=None,
        help="OpenAI-compatible API base URL (e.g., https://openrouter.ai/api/v1)",
    )
    parser.add_argument(
        "--judge-temperature", type=float, default=0.0,
        help="Judge temperature (default: 0.0 for deterministic judging)",
    )
    parser.add_argument(
        "--judge-max-tokens", type=int, default=1024,
        help="Max tokens for judge response (default: 1024)",
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load local model in 4-bit quantization",
    )
    parser.add_argument(
        "--load-in-8bit", action="store_true",
        help="Load local model in 8-bit quantization",
    )

    # Evaluation parameters
    parser.add_argument(
        "--dimensions", type=str, nargs="+",
        default=["coherence", "relevance", "fluency", "overall"],
        choices=list(DIMENSIONS.keys()),
        help="Quality dimensions to evaluate (default: coherence relevance fluency overall)",
    )
    parser.add_argument(
        "--n-texts", type=str, default="all",
        help="Number of texts per condition, or 'all' (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Pairwise-specific
    parser.add_argument(
        "--no-swap", action="store_true",
        help="Disable position swapping in pairwise mode (not recommended)",
    )

    # Rate limiting
    parser.add_argument(
        "--rate-limit", type=int, default=60,
        help="API requests per minute (default: 60)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=4,
        help="Max concurrent API calls (default: 4)",
    )

    # Output
    parser.add_argument(
        "--save-dir", type=str, default="output",
        help="Base directory to save outputs (default: output)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed per-evaluation output")

    args = parser.parse_args()

    if not any([args.human_text, args.no_watermark_text, args.watermarked_text]):
        parser.error("At least one of --human-text, --no-watermark-text, --watermarked-text is required")

    if args.mode == "pairwise":
        n_sources = sum(1 for s in [args.human_text, args.no_watermark_text, args.watermarked_text] if s is not None)
        if n_sources < 2:
            parser.error("Pairwise mode requires at least two text sources")

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    n_texts = None
    if args.n_texts.lower() != "all":
        n_texts = int(args.n_texts)

    # --- Load prompt-response pairs ---
    conditions: dict[str, list[dict]] = {}
    source_descs: dict[str, str] = {}

    for attr_name, condition_name in [
        ("human_text", "human"),
        ("no_watermark_text", "no_watermark"),
        ("watermarked_text", "watermarked"),
    ]:
        flag_value = getattr(args, attr_name)
        if flag_value is None:
            continue

        source = parse_text_source(flag_value)
        pairs = _load_pairs(source, n_texts=n_texts)

        if not pairs:
            print(f"Warning: no texts loaded for --{attr_name.replace('_', '-')}")
            continue

        conditions[condition_name] = pairs
        source_descs[condition_name] = source_description(source)
        print(f"Loaded {len(pairs)} prompt-response pairs for '{condition_name}' from {source_descs[condition_name]}")

    if not conditions:
        print("Error: no texts loaded from any source")
        sys.exit(1)

    # --- Create judge backend ---
    cost_tracker = CostTracker()
    rate_limiter = RateLimiter(requests_per_minute=args.rate_limit)

    if args.judge_backend == "openai":
        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: --api-key or OPENROUTER_API_KEY/OPENAI_API_KEY env var required for openai backend")
            sys.exit(1)
        backend = OpenAICompatibleBackend(args.judge_model, api_key, args.api_base_url)
    elif args.judge_backend == "local":
        backend = LocalHFBackend(args.judge_model, args.load_in_4bit, args.load_in_8bit)
    else:
        print(f"Error: unknown judge backend '{args.judge_backend}'")
        sys.exit(1)

    # --- Run evaluation ---
    if args.mode == "pointwise":
        all_results = []
        for condition_name, pairs in conditions.items():
            print(f"\nEvaluating '{condition_name}' ({len(pairs)} texts, {len(args.dimensions)} dimensions)...")
            results = evaluate_pointwise(
                pairs, condition_name, backend, args.dimensions,
                rate_limiter, cost_tracker,
                max_concurrent=args.max_concurrent,
                temperature=args.judge_temperature,
                max_tokens=args.judge_max_tokens,
                verbose=args.verbose,
            )
            all_results.extend(results)

        _output_results_pointwise(all_results, source_descs, args, cost_tracker)

    elif args.mode == "pairwise":
        cond_names = [c for c in _CONDITION_ORDER if c in conditions]
        if len(cond_names) < 2:
            print("Error: pairwise mode requires at least two conditions")
            sys.exit(1)

        name_a, name_b = cond_names[0], cond_names[1]
        pairs_a = conditions[name_a]
        pairs_b = conditions[name_b]

        min_len = min(len(pairs_a), len(pairs_b))
        if min_len < len(pairs_a) or min_len < len(pairs_b):
            print(f"Warning: trimming pairs to {min_len} (from {len(pairs_a)} and {len(pairs_b)})")
            pairs_a = pairs_a[:min_len]
            pairs_b = pairs_b[:min_len]

        swap = not args.no_swap
        print(f"\nPairwise: '{name_a}' vs '{name_b}' ({min_len} pairs, {len(args.dimensions)} dims, swap={swap})...")

        results = evaluate_pairwise(
            pairs_a, pairs_b, name_a, name_b,
            backend, args.dimensions,
            rate_limiter, cost_tracker,
            swap=swap,
            max_concurrent=args.max_concurrent,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
            verbose=args.verbose,
        )

        _output_results_pairwise(results, source_descs, args, cost_tracker, name_a, name_b)


if __name__ == "__main__":
    main()