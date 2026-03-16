#!/usr/bin/env python3
"""
Word-Level Attack Robustness Analysis Script.

Tests how watermark detection degrades under word-level attacks applied to
generated text, reporting match rate (recovered watermark == original).

Four attacks: word deletion, word insertion, synonym replacement, paraphrasing.

Usage examples:
  python scripts/attack_robustness.py --input-dir output/some_experiment --attack-budget 0.1 --verbose
  python scripts/attack_robustness.py --input-dir output/some_experiment --attack-budget 0.2 --attacks deletion insertion synonym
  python scripts/attack_robustness.py --input-dir output/some_experiment --attack-budget 0.1 --attacks paraphrase \
      --api-base-url https://openrouter.ai/api/v1 --api-key $OPENROUTER_API_KEY --paraphrase-model openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import os
import sys
import time as time_mod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import src.paths as paths
from scripts.utils import (
    create_triton_codebooks,
    load_experiment,
    multi_gpu_triton_batch_search,
    resolve_triton_gpu_count,
)
from src.codebooks import (
    BCHCodewordGenerator,
    CodebookBase,
    CodebookHamming,
    CodebookLevenshtein,
    LinearCodeGenerator,
    RNGCodewordGenerator,
)
from src.llm_watermark import LLMWatermarkDecoder


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Word-level attack robustness analysis for watermarked text"
    )
    # Required
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Path to experiment output directory (must contain config.json and generations.csv)",
    )
    parser.add_argument(
        "--attack-budget", type=float, required=True,
        help="Fraction of tokens to affect (e.g., 0.1 = 10%%)",
    )

    # Attack selection
    parser.add_argument(
        "--attacks", type=str, nargs="+",
        default=["deletion", "insertion", "synonym", "paraphrase"],
        choices=["deletion", "insertion", "synonym", "paraphrase"],
        help="Attack types to run (default: all four)",
    )

    # Paraphrasing API
    parser.add_argument("--api-base-url", type=str, default=None,
        help="OpenAI-compatible API base URL (e.g., https://openrouter.ai/api/v1)")
    parser.add_argument("--api-key", type=str, default=None,
        help="API key for paraphrasing (falls back to OPENROUTER_API_KEY env var)")
    parser.add_argument("--paraphrase-model", type=str, default=None,
        help="Model name for paraphrasing (e.g., openai/gpt-4o-mini)")
    parser.add_argument("--paraphrase-timeout", type=float, default=30.0,
        help="Timeout in seconds for paraphrase API calls (default: 30)")
    parser.add_argument("--paraphrase-workers", type=int, default=0,
        help="Number of parallel API workers for paraphrasing (0 = auto/cpu_count, default: 0)")

    # Search configuration
    parser.add_argument(
        "--distance", type=str, default=None,
        choices=["hamming", "levenshtein", "levenshtein-triton"],
        help="Override distance metric (defaults to value in config.json)",
    )
    parser.add_argument("--num-gpus", type=int, default=0,
        help="Number of GPUs for Triton search (0 = all available, default: 0)")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Output
    parser.add_argument("--save-dir", type=str, default="output",
        help="Directory to save analysis outputs (default: output)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")

    args = parser.parse_args()

    # Validate paraphrase requirements
    if "paraphrase" in args.attacks:
        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not all([args.api_base_url, api_key, args.paraphrase_model]):
            parser.error(
                "--api-base-url, --api-key (or OPENROUTER_API_KEY env var), and "
                "--paraphrase-model are required for paraphrase attack"
            )
        # Resolve API key from env if not provided via CLI
        if args.api_key is None:
            args.api_key = api_key

    return args


# ---------------------------------------------------------------------------
# NLTK setup
# ---------------------------------------------------------------------------

def _ensure_nltk_data(resources: list[str]) -> None:
    """Download NLTK data to cache/nltk/ if not already present."""
    import nltk

    nltk_dir = os.path.abspath(paths.NLTK_CACHE_DIR)
    os.makedirs(nltk_dir, exist_ok=True)

    # Prepend our cache dir so nltk finds data there
    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_dir)

    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1], download_dir=nltk_dir, quiet=True)


# ---------------------------------------------------------------------------
# Attack functions
# ---------------------------------------------------------------------------

def apply_word_deletion(
    text: str,
    attack_budget: float,
    tokenizer,
    rng: np.random.Generator,
) -> tuple[str, int]:
    """
    Delete random words until token budget is met.

    Returns:
        (attacked_text, tokens_affected)
    """
    words = text.split()
    if len(words) <= 1:
        return text, 0

    total_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    target_tokens = int(attack_budget * total_tokens)
    if target_tokens <= 0:
        return text, 0

    available_indices = list(range(len(words)))
    rng.shuffle(available_indices)

    tokens_affected = 0
    indices_to_delete: set[int] = set()

    for idx in available_indices:
        if tokens_affected >= target_tokens:
            break
        word_tokens = len(tokenizer.encode(words[idx], add_special_tokens=False))
        indices_to_delete.add(idx)
        tokens_affected += word_tokens

    remaining = [w for i, w in enumerate(words) if i not in indices_to_delete]
    # Keep at least one word
    if not remaining:
        kept_idx = available_indices[-1]  # Last one added, remove it from deletion
        remaining = [words[kept_idx]]
        tokens_affected -= len(tokenizer.encode(words[kept_idx], add_special_tokens=False))

    return " ".join(remaining), tokens_affected


def apply_word_insertion(
    text: str,
    attack_budget: float,
    tokenizer,
    rng: np.random.Generator,
    word_pool: list[str],
) -> tuple[str, int]:
    """
    Insert random words from the pool until token budget is met.

    Returns:
        (attacked_text, tokens_affected)
    """
    words = text.split()
    total_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    target_tokens = int(attack_budget * total_tokens)
    if target_tokens <= 0:
        return text, 0

    result = list(words)
    tokens_affected = 0
    pool_size = len(word_pool)

    while tokens_affected < target_tokens:
        insert_word = word_pool[rng.integers(pool_size)]
        insert_pos = rng.integers(len(result) + 1)
        word_tokens = len(tokenizer.encode(insert_word, add_special_tokens=False))
        result.insert(insert_pos, insert_word)
        tokens_affected += word_tokens

    return " ".join(result), tokens_affected


def _pos_to_wordnet(tag: str):
    """Map Penn Treebank POS tag to WordNet POS constant."""
    from nltk.corpus import wordnet

    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    return None


def _get_synonyms(word: str, pos) -> list[str]:
    """Get single-word synonyms for a word with optional POS filter."""
    from nltk.corpus import wordnet

    synonyms: set[str] = set()
    synsets = wordnet.synsets(word, pos=pos) if pos else wordnet.synsets(word)
    for syn in synsets:
        for lemma in syn.lemmas():
            name = lemma.name()
            if name.lower() != word.lower() and "_" not in name:
                synonyms.add(name)
    return list(synonyms)


def apply_synonym_replace(
    text: str,
    attack_budget: float,
    tokenizer,
    rng: np.random.Generator,
) -> tuple[str, int]:
    """
    Replace words with WordNet synonyms until token budget is met.

    Uses text.split() instead of nltk.word_tokenize() to avoid contraction
    splitting issues ("don't" -> "do n't" -> corrupted on rejoin).

    Returns:
        (attacked_text, tokens_affected)
    """
    import nltk

    words = text.split()
    total_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    target_tokens = int(attack_budget * total_tokens)
    if target_tokens <= 0:
        return text, 0

    # POS-tag for better synonym quality
    tagged = nltk.pos_tag(words)

    # Build candidate list: (word_index, original_word, synonyms)
    candidates = []
    for i, (word, tag) in enumerate(tagged):
        wn_pos = _pos_to_wordnet(tag)
        syns = _get_synonyms(word, wn_pos)
        if syns:
            candidates.append((i, word, syns))

    if not candidates:
        return text, 0

    # Shuffle candidates
    candidate_order = list(range(len(candidates)))
    rng.shuffle(candidate_order)

    tokens_affected = 0
    result_words = list(words)

    for ci in candidate_order:
        if tokens_affected >= target_tokens:
            break
        idx, orig_word, syns = candidates[ci]
        synonym = syns[rng.integers(len(syns))]
        result_words[idx] = synonym
        orig_tokens = len(tokenizer.encode(orig_word, add_special_tokens=False))
        tokens_affected += orig_tokens

    return " ".join(result_words), tokens_affected


def _call_paraphrase_api(client, model_name: str, text: str, timeout: float, max_retries: int = 3) -> str:
    """Call OpenAI-compatible API with retries and exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a text rewriter. Rewrite the following text while "
                            "preserving its meaning. Output only the rewritten text, nothing else."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                timeout=timeout,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time_mod.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Paraphrase API failed after {max_retries} retries: {e}")


def _select_paraphrase_region(
    text: str,
    attack_budget: float,
    tokenizer,
    rng: np.random.Generator,
) -> tuple[list[str], int, int, int] | None:
    """
    Select a contiguous region of words to paraphrase.

    Returns:
        (words, left, right, region_tokens) or None if text is too short.
    """
    words = text.split()
    if len(words) <= 2:
        return None

    total_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    target_tokens = int(attack_budget * total_tokens)
    if target_tokens <= 0:
        return None

    # Start from a random center, expand alternating left/right
    center = int(rng.integers(len(words)))
    left = center
    right = center  # inclusive

    region_text = words[center]
    region_tokens = len(tokenizer.encode(region_text, add_special_tokens=False))

    expand_left = True
    while region_tokens < target_tokens:
        expanded = False
        if expand_left and left > 0:
            left -= 1
            expanded = True
        elif not expand_left and right < len(words) - 1:
            right += 1
            expanded = True
        elif expand_left and right < len(words) - 1:
            # Can't go left, try right instead
            right += 1
            expanded = True
        elif not expand_left and left > 0:
            # Can't go right, try left instead
            left -= 1
            expanded = True

        if not expanded:
            break  # Boundaries reached on both sides

        region_text = " ".join(words[left : right + 1])
        region_tokens = len(tokenizer.encode(region_text, add_special_tokens=False))
        expand_left = not expand_left

    return words, left, right, region_tokens


def _reconstruct_paraphrased(
    words: list[str], left: int, right: int, paraphrased: str,
) -> str:
    """Reconstruct text: prefix + paraphrased region + suffix."""
    prefix_words = words[:left]
    suffix_words = words[right + 1 :]
    parts = []
    if prefix_words:
        parts.append(" ".join(prefix_words))
    parts.append(paraphrased)
    if suffix_words:
        parts.append(" ".join(suffix_words))
    return " ".join(parts).strip()


def apply_paraphrase(
    text: str,
    attack_budget: float,
    tokenizer,
    rng: np.random.Generator,
    client,
    model_name: str,
    timeout: float,
) -> tuple[str, int]:
    """
    Select a contiguous region of words, expand until token budget is met,
    then paraphrase via API.

    Returns:
        (attacked_text, tokens_affected)
    """
    result = _select_paraphrase_region(text, attack_budget, tokenizer, rng)
    if result is None:
        return text, 0

    words, left, right, region_tokens = result
    selected_text = " ".join(words[left : right + 1])
    paraphrased = _call_paraphrase_api(client, model_name, selected_text, timeout)
    attacked_text = _reconstruct_paraphrased(words, left, right, paraphrased)

    return attacked_text, region_tokens


# ---------------------------------------------------------------------------
# Codebook search helpers
# ---------------------------------------------------------------------------

def _adapt_for_hamming(query: np.ndarray, u: int) -> np.ndarray:
    """Pad or truncate a bit array to exactly length u for Hamming search."""
    if len(query) == u:
        return query
    elif len(query) > u:
        return query[:u]
    else:
        padded = np.zeros(u, dtype=np.uint8)
        padded[: len(query)] = query
        return padded


def _batch_search(
    queries: list[np.ndarray],
    codebook,
    codebooks: list | None,
    base_codebook: CodebookBase,
    u: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run batch codebook search across all decoded queries.

    Returns:
        (retrieved_watermarks, distances) — both 1D arrays of length len(queries)
    """
    if not queries:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int32)

    if isinstance(codebook, CodebookHamming):
        adapted = np.stack([_adapt_for_hamming(q, u) for q in queries])
        packed = CodebookBase._pack_bits_2d(adapted, u)
        dist_arr, idx_arr = codebook.index.search(packed, 1)
        distances = dist_arr[:, 0].astype(np.int32)
        retrieved_watermarks = base_codebook.watermark_identities[idx_arr[:, 0]]
    elif codebooks and len(codebooks) > 1:
        _, retrieved_watermarks, distances = multi_gpu_triton_batch_search(codebooks, queries)
    else:
        # Use individual queries to avoid Numba type-unification issue in batch mode
        n_queries = len(queries)
        retrieved_watermarks = np.empty(n_queries, dtype=np.int64)
        distances = np.empty(n_queries, dtype=np.int32)
        for i, q in enumerate(queries):
            _, wm, dist = codebook.find_closest_match(q)
            retrieved_watermarks[i] = wm
            distances[i] = dist

    return np.asarray(retrieved_watermarks, dtype=np.int64), np.asarray(distances, dtype=np.int32)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def compute_baseline(
    generations_df: pd.DataFrame,
    decoder: LLMWatermarkDecoder,
    codebook,
    codebooks: list | None,
    base_codebook: CodebookBase,
    u: int,
    verbose: bool = False,
) -> dict:
    """
    Decode original (unattacked) texts and search codebook. Returns baseline stats.
    """
    queries = []
    watermarks = []

    for _, row in tqdm(
        generations_df.iterrows(), total=len(generations_df),
        desc="  baseline decode", disable=not verbose,
    ):
        text = str(row["generated_text"])
        bit_seq, _, _ = decoder.decode(text)
        queries.append(np.array(bit_seq, dtype=np.uint8))
        watermarks.append(int(row["watermark"]))

    watermarks_arr = np.array(watermarks, dtype=np.int64)
    retrieved_wm, distances = _batch_search(queries, codebook, codebooks, base_codebook, u)

    matches = retrieved_wm == watermarks_arr
    return {
        "total": len(watermarks),
        "match_rate": float(matches.mean()) if len(matches) > 0 else 0.0,
        "avg_distance": float(distances.mean()) if len(distances) > 0 else 0.0,
        "matches": int(matches.sum()),
    }


def _run_paraphrase_attack(
    generations_df: pd.DataFrame,
    decoder: LLMWatermarkDecoder,
    codebook,
    codebooks: list | None,
    base_codebook: CodebookBase,
    u: int,
    attack_budget: float,
    seed: int,
    attack_idx: int,
    client,
    paraphrase_model: str,
    paraphrase_timeout: float,
    paraphrase_workers: int,
    verbose: bool,
) -> pd.DataFrame:
    """Run paraphrase attack with parallel API calls."""
    rng = np.random.default_rng(seed + attack_idx)

    # Phase 1: Pre-select regions sequentially (preserves RNG determinism)
    tasks = []  # (row_idx, original_text, original_watermark, total_tokens, region_info)
    for row_idx, (_, row) in enumerate(generations_df.iterrows()):
        original_text = str(row["generated_text"])
        original_watermark = int(row["watermark"])
        total_tokens = len(decoder.tokenizer.encode(original_text, add_special_tokens=False))

        words = original_text.split()
        if len(words) < 3:
            tasks.append((row_idx, original_text, original_watermark, total_tokens, None))
            continue

        region = _select_paraphrase_region(original_text, attack_budget, decoder.tokenizer, rng)
        tasks.append((row_idx, original_text, original_watermark, total_tokens, region))

    # Phase 2: Parallel API calls
    # Build work items for samples that have a valid region
    api_work = []  # (task_index, selected_text, words, left, right, region_tokens)
    for i, (row_idx, original_text, original_watermark, total_tokens, region) in enumerate(tasks):
        if region is not None:
            words, left, right, region_tokens = region
            selected_text = " ".join(words[left : right + 1])
            api_work.append((i, selected_text, words, left, right, region_tokens))

    paraphrased_results = {}  # task_index -> (attacked_text, region_tokens) or error str

    def _api_call(work_item):
        idx, selected_text, words, left, right, region_tokens = work_item
        paraphrased = _call_paraphrase_api(client, paraphrase_model, selected_text, paraphrase_timeout)
        attacked_text = _reconstruct_paraphrased(words, left, right, paraphrased)
        return idx, attacked_text, region_tokens

    if paraphrase_workers <= 0:
        paraphrase_workers = os.cpu_count() or 4

    with ThreadPoolExecutor(max_workers=paraphrase_workers) as pool:
        futures = {pool.submit(_api_call, w): w[0] for w in api_work}
        with tqdm(total=len(futures), desc="  paraphrase", disable=not verbose) as pbar:
            for future in as_completed(futures):
                task_idx = futures[future]
                try:
                    idx, attacked_text, region_tokens = future.result()
                    paraphrased_results[idx] = (attacked_text, region_tokens)
                except Exception as e:
                    paraphrased_results[task_idx] = str(e)
                pbar.update(1)

    # Phase 3: Assemble records and decode
    records = []
    queries = []
    query_indices = []

    for i, (row_idx, original_text, original_watermark, total_tokens, region) in enumerate(tasks):
        record = {
            "sample_index": row_idx,
            "attack_type": "paraphrase",
            "original_watermark": original_watermark,
            "total_tokens": total_tokens,
        }

        if region is None:
            record.update({
                "recovered_watermark": -1, "distance": -1, "success": False,
                "tokens_affected": 0, "original_text": original_text,
                "attacked_text": original_text, "skipped": True,
                "skip_reason": "text too short",
            })
            records.append(record)
            continue

        result = paraphrased_results.get(i)
        if isinstance(result, str):
            # API error
            record.update({
                "recovered_watermark": -1, "distance": -1, "success": False,
                "tokens_affected": 0, "original_text": original_text,
                "attacked_text": "", "skipped": True,
                "skip_reason": result,
            })
            records.append(record)
            continue

        attacked_text, tokens_affected = result
        record.update({
            "tokens_affected": tokens_affected,
            "original_text": original_text,
            "attacked_text": attacked_text,
            "skipped": False,
            "skip_reason": "",
        })

        bit_seq, _, _ = decoder.decode(attacked_text, truncate=False)
        queries.append(np.array(bit_seq, dtype=np.uint8))
        query_indices.append(len(records))
        records.append(record)

    if queries:
        retrieved_wm, distances = _batch_search(queries, codebook, codebooks, base_codebook, u)
        for j, rec_idx in enumerate(query_indices):
            rec = records[rec_idx]
            rec["recovered_watermark"] = int(retrieved_wm[j])
            rec["distance"] = int(distances[j])
            rec["success"] = int(retrieved_wm[j]) == rec["original_watermark"]

    return pd.DataFrame(records)


def evaluate_attacks(
    generations_df: pd.DataFrame,
    decoder: LLMWatermarkDecoder,
    codebook,
    codebooks: list | None,
    base_codebook: CodebookBase,
    u: int,
    attack_budget: float,
    attacks: list[str],
    seed: int,
    word_pool: list[str] | None = None,
    client=None,
    paraphrase_model: str = "",
    paraphrase_timeout: float = 30.0,
    paraphrase_workers: int = 0,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Run requested attacks on all samples.

    Returns:
        dict mapping attack_name -> DataFrame of per-sample results.
    """
    results: dict[str, pd.DataFrame] = {}

    for attack_idx, attack_name in enumerate(attacks):
        # Paraphrase has its own parallel path
        if attack_name == "paraphrase":
            results[attack_name] = _run_paraphrase_attack(
                generations_df, decoder, codebook, codebooks, base_codebook,
                u, attack_budget, seed, attack_idx, client, paraphrase_model,
                paraphrase_timeout, paraphrase_workers, verbose,
            )
            continue

        rng = np.random.default_rng(seed + attack_idx)

        records = []
        queries = []
        query_indices = []  # Track which records have valid queries

        for row_idx, (_, row) in enumerate(tqdm(
            generations_df.iterrows(), total=len(generations_df),
            desc=f"  {attack_name}", disable=not verbose,
        )):
            original_text = str(row["generated_text"])
            original_watermark = int(row["watermark"])
            total_tokens = len(decoder.tokenizer.encode(original_text, add_special_tokens=False))

            record = {
                "sample_index": row_idx,
                "attack_type": attack_name,
                "original_watermark": original_watermark,
                "total_tokens": total_tokens,
            }

            # Skip very short texts
            words = original_text.split()
            if len(words) < 3:
                record.update({
                    "recovered_watermark": -1, "distance": -1, "success": False,
                    "tokens_affected": 0, "original_text": original_text,
                    "attacked_text": original_text, "skipped": True,
                    "skip_reason": "text too short",
                })
                records.append(record)
                continue

            # Apply attack
            try:
                if attack_name == "deletion":
                    attacked_text, tokens_affected = apply_word_deletion(
                        original_text, attack_budget, decoder.tokenizer, rng)
                elif attack_name == "insertion":
                    attacked_text, tokens_affected = apply_word_insertion(
                        original_text, attack_budget, decoder.tokenizer, rng, word_pool)
                elif attack_name == "synonym":
                    attacked_text, tokens_affected = apply_synonym_replace(
                        original_text, attack_budget, decoder.tokenizer, rng)
                else:
                    raise ValueError(f"Unknown attack: {attack_name}")
            except Exception as e:
                record.update({
                    "recovered_watermark": -1, "distance": -1, "success": False,
                    "tokens_affected": 0, "original_text": original_text,
                    "attacked_text": "", "skipped": True,
                    "skip_reason": str(e),
                })
                records.append(record)
                continue

            record.update({
                "tokens_affected": tokens_affected,
                "original_text": original_text,
                "attacked_text": attacked_text,
                "skipped": False,
                "skip_reason": "",
            })

            # Decode attacked text (no truncation — let search handle length differences)
            bit_seq, _, _ = decoder.decode(attacked_text, truncate=False)
            queries.append(np.array(bit_seq, dtype=np.uint8))
            query_indices.append(len(records))
            records.append(record)

        # Batch codebook search for all non-skipped samples
        if queries:
            retrieved_wm, distances = _batch_search(
                queries, codebook, codebooks, base_codebook, u)

            for j, rec_idx in enumerate(query_indices):
                rec = records[rec_idx]
                rec["recovered_watermark"] = int(retrieved_wm[j])
                rec["distance"] = int(distances[j])
                rec["success"] = int(retrieved_wm[j]) == rec["original_watermark"]

        results[attack_name] = pd.DataFrame(records)

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _shorten_model_name(name: str) -> str:
    """Shorten HuggingFace model name for display."""
    return name.split("/")[-1] if "/" in name else name


def write_detail_csvs(
    attack_results: dict[str, pd.DataFrame],
    output_dir: str,
) -> None:
    """Write per-attack detail CSVs."""
    for attack_name, df in attack_results.items():
        csv_path = os.path.join(output_dir, f"detail_{attack_name}.csv")
        df.to_csv(csv_path, index=False)


def write_results_txt(
    args: argparse.Namespace,
    config: dict,
    baseline: dict,
    attack_results: dict[str, pd.DataFrame],
    output_dir: str,
) -> None:
    """Write summary results text file."""
    wm = config["watermark"]
    model_name = _shorten_model_name(config["model"]["name"])
    distance = args.distance or wm.get("distance", "hamming")

    lines = [
        "=" * 70,
        "Word-Level Attack Robustness Analysis",
        "=" * 70,
        "",
        f"Experiment:  {args.input_dir}",
        f"Model:       {model_name}",
        f"Parameters:  n={wm['n']}, u={wm['u']}, distance={distance}",
        f"Budget:      {args.attack_budget} ({args.attack_budget * 100:.0f}% of tokens)",
        f"Seed:        {args.seed}",
        *([f"Paraphrase:  {args.paraphrase_model}"] if args.paraphrase_model else []),
        "",
        "-" * 70,
        f"{'Attack':<16} {'Samples':>8} {'Skipped':>8} {'Match Rate':>12} {'Avg Dist':>10}",
        "-" * 70,
    ]

    # Baseline row
    lines.append(
        f"{'No Attack':<16} {baseline['total']:>8} {'0':>8} "
        f"{baseline['match_rate'] * 100:>11.1f}% {baseline['avg_distance']:>10.1f}"
    )

    # Attack rows
    for attack_name, df in attack_results.items():
        total = len(df)
        skipped = int(df["skipped"].sum()) if "skipped" in df.columns else 0
        non_skipped = df[~df["skipped"]] if "skipped" in df.columns else df
        matches = int(non_skipped["success"].sum()) if len(non_skipped) > 0 else 0
        match_rate = matches / total if total > 0 else 0.0
        avg_dist = float(non_skipped["distance"].mean()) if len(non_skipped) > 0 else 0.0
        avg_tokens_affected = float(non_skipped["tokens_affected"].mean()) if len(non_skipped) > 0 else 0.0

        lines.append(
            f"{attack_name:<16} {total:>8} {skipped:>8} "
            f"{match_rate * 100:>11.1f}% {avg_dist:>10.1f}"
        )

    lines.extend([
        "-" * 70,
        "",
        "Token budget details (non-skipped samples):",
    ])

    for attack_name, df in attack_results.items():
        non_skipped = df[~df["skipped"]] if "skipped" in df.columns else df
        if len(non_skipped) > 0:
            avg_affected = float(non_skipped["tokens_affected"].mean())
            avg_total = float(non_skipped["total_tokens"].mean())
            actual_pct = (avg_affected / avg_total * 100) if avg_total > 0 else 0.0
            lines.append(
                f"  {attack_name}: avg {avg_affected:.1f} / {avg_total:.1f} tokens affected "
                f"({actual_pct:.1f}% actual vs {args.attack_budget * 100:.0f}% target)"
            )

    lines.append("")

    txt_path = os.path.join(output_dir, "results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also print to console
    print("\n".join(lines))


def create_bar_chart(
    baseline: dict,
    attack_results: dict[str, pd.DataFrame],
    args: argparse.Namespace,
    config: dict,
    output_dir: str,
) -> None:
    """Create bar chart showing match rate per attack type."""
    labels = ["No Attack"]
    match_rates = [baseline["match_rate"] * 100]

    for attack_name, df in attack_results.items():
        total = len(df)
        non_skipped = df[~df["skipped"]] if "skipped" in df.columns else df
        matches = int(non_skipped["success"].sum()) if len(non_skipped) > 0 else 0
        rate = matches / total * 100 if total > 0 else 0.0
        labels.append(attack_name.capitalize())
        match_rates.append(rate)

    wm = config["watermark"]
    model_name = _shorten_model_name(config["model"]["name"])

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2ecc71"] + ["#e74c3c"] * len(attack_results)
    bars = ax.bar(labels, match_rates, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, rate in zip(bars, match_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{rate:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Match Rate (%)")
    ax.set_title(
        f"Watermark Robustness — {model_name}\n"
        f"n={wm['n']}, u={wm['u']}, budget={args.attack_budget * 100:.0f}%"
    )
    ax.set_ylim(0, min(max(match_rates) + 15, 105))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    png_path = os.path.join(output_dir, "robustness.png")
    fig.savefig(png_path, dpi=150)
    print(f"Plot saved: {png_path}")

    if args.show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    paths.ensure_directories()

    # ------------------------------------------------------------------
    # 1. Load experiment
    # ------------------------------------------------------------------
    experiment = load_experiment(args.input_dir)
    config = experiment["config"]
    generations_df = experiment["generations"]

    wm = config["watermark"]
    model_name = config["model"]["name"]
    n, u = wm["n"], wm["u"]
    secret_key = wm["secret_key"]
    green_fraction = wm.get("green_fraction", 0.5)
    codeword_generation = wm.get("codeword_generation", "rng")
    bch_m = wm.get("bch_m", None)
    distance = args.distance or wm.get("distance", "hamming")

    # Device must match the original experiment so torch.randperm produces
    # identical green/red token splits (CPU vs CUDA RNG differ).
    experiment_device = config.get("experiment", {}).get("device", "cpu")
    if experiment_device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: experiment used CUDA but no GPU available — decoded bits will differ!")
        experiment_device = "cpu"

    # Filter to rows with valid generated_text
    mask = generations_df["generated_text"].notna() & (generations_df["generated_text"].astype(str) != "")
    working_df = generations_df[mask].reset_index(drop=True)

    if working_df.empty:
        print("Error: No valid generated_text found in generations.csv")
        sys.exit(1)

    if args.verbose:
        print(f"Experiment: {args.input_dir}")
        print(f"  Model: {model_name}")
        print(f"  n={n}, u={u}, distance={distance}, codeword_generation={codeword_generation}")
        print(f"  Samples: {len(working_df)}")
        print(f"  Attack budget: {args.attack_budget} ({args.attack_budget * 100:.0f}%)")
        print(f"  Attacks: {', '.join(args.attacks)}")

    # ------------------------------------------------------------------
    # 2. Create decoder (tokenizer-only)
    # ------------------------------------------------------------------
    if args.verbose:
        print("\nLoading decoder...")
    decoder = LLMWatermarkDecoder(
        model_name=model_name,
        secret_key=secret_key,
        n=n,
        u=u,
        green_list_fraction=green_fraction,
        device=experiment_device,
    )

    # ------------------------------------------------------------------
    # 3. Build codebook
    # ------------------------------------------------------------------
    if args.verbose:
        print("Building codebook...")

    if codeword_generation == "rng":
        generator_cls = RNGCodewordGenerator
        generator_kwargs: dict = {}
    elif codeword_generation == "linear":
        generator_cls = LinearCodeGenerator
        generator_kwargs = {}
    else:
        generator_cls = BCHCodewordGenerator
        generator_kwargs = {"m": bch_m}

    base_codebook = CodebookBase(
        n=n,
        u=u,
        secret_key=secret_key,
        generator_cls=generator_cls,
        generator_kwargs=generator_kwargs,
    )

    codebooks = None
    if distance == "hamming":
        codebook = CodebookHamming(base_codebook)
    elif distance == "levenshtein-triton":
        num_gpus = resolve_triton_gpu_count(args.num_gpus)
        codebooks = create_triton_codebooks(base_codebook, num_gpus)
        codebook = codebooks[0]
        if args.verbose:
            print(f"Triton search across {num_gpus} GPU{'s' if num_gpus > 1 else ''}")
    else:
        codebook = CodebookLevenshtein(base_codebook)

    # ------------------------------------------------------------------
    # 4. Compute baseline
    # ------------------------------------------------------------------
    if args.verbose:
        print("\nComputing baseline (no attack)...")
    baseline = compute_baseline(
        working_df, decoder, codebook, codebooks, base_codebook, u, verbose=args.verbose)

    if args.verbose:
        print(f"  Baseline match rate: {baseline['match_rate'] * 100:.1f}%")
        print(f"  Baseline avg distance: {baseline['avg_distance']:.1f}")

    # ------------------------------------------------------------------
    # 5. Prepare attack resources
    # ------------------------------------------------------------------
    nltk_needed = []
    if "synonym" in args.attacks:
        nltk_needed.extend(["taggers/averaged_perceptron_tagger_eng", "corpora/wordnet"])
    if "insertion" in args.attacks:
        nltk_needed.append("corpora/words")

    if nltk_needed:
        if args.verbose:
            print("\nDownloading NLTK data...")
        _ensure_nltk_data(nltk_needed)

    # Build word pool for insertion
    word_pool = None
    if "insertion" in args.attacks:
        from nltk.corpus import words
        word_pool = words.words()
        if args.verbose:
            print(f"  Word pool: {len(word_pool)} words from NLTK corpus")

    # Create OpenAI client for paraphrase
    client = None
    if "paraphrase" in args.attacks:
        from openai import OpenAI
        client = OpenAI(base_url=args.api_base_url, api_key=args.api_key)

    # ------------------------------------------------------------------
    # 6. Run attacks
    # ------------------------------------------------------------------
    if args.verbose:
        print("\nRunning attacks...")

    attack_results = evaluate_attacks(
        generations_df=working_df,
        decoder=decoder,
        codebook=codebook,
        codebooks=codebooks,
        base_codebook=base_codebook,
        u=u,
        attack_budget=args.attack_budget,
        attacks=args.attacks,
        seed=args.seed,
        word_pool=word_pool,
        client=client,
        paraphrase_model=args.paraphrase_model or "",
        paraphrase_timeout=args.paraphrase_timeout,
        paraphrase_workers=args.paraphrase_workers,
        verbose=args.verbose,
    )

    # ------------------------------------------------------------------
    # 7. Create output directory and write results
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    budget_str = str(args.attack_budget).replace(".", "p")
    run_dir_name = f"word_attack_n{n}_u{u}_budget-{budget_str}_{timestamp}"
    run_save_dir = os.path.join(args.save_dir, run_dir_name)
    os.makedirs(run_save_dir, exist_ok=True)

    if args.verbose:
        print(f"\nSaving results to: {run_save_dir}")

    write_detail_csvs(attack_results, run_save_dir)
    write_results_txt(args, config, baseline, attack_results, run_save_dir)
    create_bar_chart(baseline, attack_results, args, config, run_save_dir)

    print(f"\nDone. Results saved to: {run_save_dir}")


if __name__ == "__main__":
    main()
