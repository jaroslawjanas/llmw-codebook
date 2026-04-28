"""
LLM-as-a-Judge Pairwise Quality Evaluation.

Compares watermarked vs. non-watermarked responses side-by-side using an LLM judge.
Each pair is evaluated once with randomized A/B position assignment to eliminate
judge position bias in aggregate.

Judge backends:
  - openai  — OpenAI-compatible API (OpenAI, OpenRouter, Together, etc.)
  - local   — Local HuggingFace model via AutoModelForCausalLM

Usage:
  python scripts/llm_judge.py \
      --no-watermark-text output/no_wm_experiment/ \
      --watermarked-text output/wm_experiment/ \
      --judge-backend openai --judge-model gpt-4o-mini \
      --api-base-url https://openrouter.ai/api/v1 \
      --n-texts 100

  python scripts/llm_judge.py \
      --no-watermark-text output/no_wm_experiment/ \
      --watermarked-text output/wm_experiment/ \
      --judge-backend local --judge-model Qwen/Qwen2.5-0.5B-Instruct \
      --n-texts 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
# Quality dimension (social authenticity only)
# ---------------------------------------------------------------------------

DIMENSION = "social_authenticity"
DIMENSION_DESCRIPTION = (
    "how believable the text is as a genuine human-written product review or social media post"
)
DIMENSION_RUBRIC = {
    1: "Clearly synthetic or AI-generated, sounds nothing like a real user review or post",
    2: "Mostly artificial, has obvious giveaways that it is not from a real person",
    3: "Somewhat believable but with notable signs of being generated or generic",
    4: "Mostly authentic, reads like a real person with minor oddities",
    5: "Indistinguishable from a genuine human product review or social media post",
}


# ---------------------------------------------------------------------------
# Prompt template & parsing
# ---------------------------------------------------------------------------

_CHOICE_RE = re.compile(r"\[\[([ABC])\]\]") if False else None  # imported below

import re

_CHOICE_RE = re.compile(r"\[\[([ABC])\]\]")

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


def build_pairwise_messages(
    prompt: str, response_a: str, response_b: str,
) -> list[dict]:
    user_content = _PAIRWISE_USER.format(
        dimension=DIMENSION,
        description=DIMENSION_DESCRIPTION,
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
    )
    return [
        {"role": "system", "content": _PAIRWISE_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def parse_choice(raw: str) -> str | None:
    m = _CHOICE_RE.search(raw)
    if m:
        return m.group(1)
    return None


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
            "prompt": str(row["prompt_text"]).strip(),
            "response": str(row["generated_text"]),
            "index": i,
        })
    return pairs


def align_pairs_by_prompt(pairs_a: list[dict], pairs_b: list[dict]) -> tuple[list[dict], list[dict]]:
    """Align two pair lists by matching prompts.

    Experiments may have different orderings or different subsets of prompts.
    This function matches pairs by prompt content, keeping only prompts present
    in both lists. Returns aligned (pairs_a, pairs_b) in matching order.
    """
    index_b = {p["prompt"]: p for p in pairs_b}
    aligned_a, aligned_b = [], []
    for pa in pairs_a:
        pb = index_b.get(pa["prompt"])
        if pb is not None:
            aligned_a.append(pa)
            aligned_b.append(pb)
    return aligned_a, aligned_b


# ---------------------------------------------------------------------------
# Pairwise evaluation (randomized position, 1 vote per pair)
# ---------------------------------------------------------------------------

_CONDITION_ORDER = ["human", "no_watermark", "watermarked"]
_CONDITION_LABELS = {
    "human": "Human",
    "no_watermark": "No Watermark",
    "watermarked": "Watermarked",
}
_COLORS = ["#4DAF4A", "#377EB8", "#E41A1C"]


def evaluate_pairwise(
    pairs_a: list[dict],
    pairs_b: list[dict],
    name_a: str,
    name_b: str,
    backend: JudgeBackend,
    rate_limiter: RateLimiter,
    cost_tracker: CostTracker,
    seed: int = 42,
    max_concurrent: int = 4,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    verbose: bool = False,
) -> list[dict]:
    assert len(pairs_a) == len(pairs_b), f"Pair counts must match: {len(pairs_a)} vs {len(pairs_b)}"
    results = []

    # Pre-compute random position assignments
    rng = random.Random(seed)
    pos_assignments: dict[tuple[int], bool] = {}
    for i in range(len(pairs_a)):
        pos_assignments[(i,)] = rng.random() < 0.5  # True = natural (condition_a in position A)

    def _evaluate_one(idx: int):
        pa, pb = pairs_a[idx], pairs_b[idx]
        natural = pos_assignments[(idx,)]

        if natural:
            text_a, text_b = pa["response"], pb["response"]
        else:
            text_a, text_b = pb["response"], pa["response"]

        messages = build_pairwise_messages(pa["prompt"], text_a, text_b)
        rate_limiter.wait()
        raw, inp, out, ok = judge_with_retry(backend, messages, temperature=temperature, max_tokens=max_tokens)
        cost_tracker.record(inp, out)
        if not ok:
            cost_tracker.inc_errors()

        choice = parse_choice(raw) if ok else None

        # name_a=no_watermark, name_b=watermarked (per _CONDITION_ORDER)
        # natural: A=no_watermark, B=watermarked → watermarked is B
        # swapped: A=watermarked, B=no_watermark → watermarked is A
        wm_pos = "B" if natural else "A"
        nwm_pos = "A" if natural else "B"

        return {
            "index": pa["index"],
            "generation_a": text_a,
            "generation_b": text_b,
            "watermarked": wm_pos,
            "not_watermarked": nwm_pos,
            "choice": choice,
            "reasoning": raw,
        }

    tasks = list(range(len(pairs_a)))

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {pool.submit(_evaluate_one, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Pairwise evaluation", disable=verbose):
            result = future.result()
            results.append(result)
            if verbose:
                print(
                    f"  idx={result['index']} choice={result['choice']} "
                    f"watermarked={result['watermarked']}"
                )

    results.sort(key=lambda r: r["index"])
    return results


# ---------------------------------------------------------------------------
# Output & visualization
# ---------------------------------------------------------------------------

def _plot_pairwise(
    results: list[dict], run_dir: str, name_a: str, name_b: str,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return None

    df = pd.DataFrame(results)
    bar_colors = ["#377EB8", "#E41A1C", "#BDBDBD"]
    bar_labels = ["Not Watermarked", "Watermarked", "Tie"]

    fig, ax = plt.subplots(figsize=(5, 5))

    valid = df[df["choice"].notna() & (df["choice"] != "C")]
    wm_wins = int(((valid["choice"] == "A") & (valid["watermarked"] == "A") |
                    (valid["choice"] == "B") & (valid["watermarked"] == "B")).sum())
    nwm_wins = int(((valid["choice"] == "A") & (valid["not_watermarked"] == "A") |
                     (valid["choice"] == "B") & (valid["not_watermarked"] == "B")).sum())
    ties = int((df["choice"] == "C").sum() + df["choice"].isna().sum())
    total = len(df)
    rates = [nwm_wins / total * 100, wm_wins / total * 100, ties / total * 100]

    bars = ax.bar(bar_labels, rates, color=bar_colors, alpha=0.8)
    for bar, rate in zip(bars, rates):
        if rate > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{rate:.0f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Social Authenticity: Pairwise Comparison", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    output_path = os.path.join(run_dir, "llm_judge.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_config(run_dir: str, args: argparse.Namespace, source_descriptions: dict[str, str]):
    config = {
        "mode": "pairwise",
        "judge": {
            "backend": args.judge_backend,
            "model": args.judge_model,
            "temperature": args.judge_temperature,
            "max_tokens": args.judge_max_tokens,
        },
        "evaluation": {
            "dimension": DIMENSION,
            "n_texts": args.n_texts,
            "seed": args.seed,
            "position_strategy": "randomize",
        },
        "sources": source_descriptions,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


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

    # Summary — derive winners from choice + watermarked position
    total = len(df)
    valid = df[df["choice"].notna() & (df["choice"] != "C")]
    wm_wins = int(((valid["choice"] == "A") & (valid["watermarked"] == "A") |
                    (valid["choice"] == "B") & (valid["watermarked"] == "B")).sum())
    nwm_wins = int(((valid["choice"] == "A") & (valid["not_watermarked"] == "A") |
                     (valid["choice"] == "B") & (valid["not_watermarked"] == "B")).sum())
    ties = int((df["choice"] == "C").sum() + (df["choice"].isna()).sum())

    valid = df[df["choice"].notna()]
    pos_a_total = len(valid)
    pos_a_wins = int((valid["choice"] == "A").sum())
    position_a_win_rate = pos_a_wins / pos_a_total * 100 if pos_a_total > 0 else float("nan")

    summary_records = [{
        "dimension": DIMENSION,
        "win_rate_watermarked": wm_wins / total * 100 if total > 0 else float("nan"),
        "win_rate_not_watermarked": nwm_wins / total * 100 if total > 0 else float("nan"),
        "tie_rate": ties / total * 100 if total > 0 else float("nan"),
        "n_pairs": total,
        "position_a_win_rate": position_a_win_rate,
    }]
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
        f"Dimension:      {DIMENSION}",
        f"N texts:        {args.n_texts}",
        f"Seed:           {args.seed}",
        f"Position:       randomized (per-pair)",
        "",
    ]
    for cond, desc in source_descriptions.items():
        lines.append(f"  {cond}: {desc}")
    lines.append("")

    lines.append(f"{'Dimension':<20} {'Not Watermarked':>16} {'Watermarked':>12} {'Tie':>8} {'Pos. A Win%':>12}")
    lines.append("-" * 72)
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['dimension'].capitalize():<20} "
            f"{row['win_rate_not_watermarked']:>15.1f}% {row['win_rate_watermarked']:>11.1f}% "
            f"{row['tie_rate']:>7.1f}% {row['position_a_win_rate']:>9.1f}%"
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

    plot_path = _plot_pairwise(results, run_dir, name_a, name_b)
    if plot_path:
        print(f"Plot saved to:    {plot_path}")
    print(f"Results saved to: {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge pairwise quality evaluation for watermarked text",
    )

    # Text sources
    parser.add_argument(
        "--human-text", type=str, nargs="+", default=None,
        help="Human text source (experiment dir)",
    )
    parser.add_argument(
        "--no-watermark-text", type=str, nargs="+", default=None,
        help="No-watermark text source (experiment dir)",
    )
    parser.add_argument(
        "--watermarked-text", type=str, nargs="+", default=None,
        help="Watermarked text source (experiment dir)",
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
        "--n-texts", type=str, default="all",
        help="Number of texts per condition, or 'all' (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

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

    n_sources = sum(1 for s in [args.human_text, args.no_watermark_text, args.watermarked_text] if s is not None)
    if n_sources < 2:
        parser.error("Pairwise comparison requires at least two text sources")

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

    # --- Select pair of conditions ---
    cond_names = [c for c in _CONDITION_ORDER if c in conditions]
    if len(cond_names) < 2:
        print("Error: at least two conditions required for pairwise comparison")
        sys.exit(1)

    name_a, name_b = cond_names[0], cond_names[1]
    pairs_a = conditions[name_a]
    pairs_b = conditions[name_b]

    pairs_a, pairs_b = align_pairs_by_prompt(pairs_a, pairs_b)
    n_pairs = len(pairs_a)
    if n_pairs == 0:
        print("Error: no matching prompts between conditions")
        sys.exit(1)

    print(f"\nPairwise: '{name_a}' vs '{name_b}' ({n_pairs} pairs, dimension={DIMENSION}, randomized-position)...")

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
    results = evaluate_pairwise(
        pairs_a, pairs_b, name_a, name_b,
        backend, rate_limiter, cost_tracker,
        seed=args.seed,
        max_concurrent=args.max_concurrent,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
        verbose=args.verbose,
    )

    _output_results_pairwise(results, source_descs, args, cost_tracker, name_a, name_b)


if __name__ == "__main__":
    main()