"""Data loading utilities for experiment results."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


def _get_default_output_dir() -> Path:
    """Get the default output directory relative to project root."""
    return Path(__file__).parent.parent.parent / "output"


def discover_experiments(output_dir: Path | str = None) -> pd.DataFrame:
    """
    Scan output directory and return metadata about all experiments.

    Reads config.json from each experiment to get authoritative metadata.

    Args:
        output_dir: Path to output directory. Defaults to project output/.

    Returns:
        DataFrame with columns:
          - model: str (e.g., 'meta-llama/Llama-3.2-3B')
          - dataset: str (e.g., 'Hello-SimpleAI/HC3')
          - n: int (watermark bits)
          - u: int (codeword length)
          - distance: str ('hamming' or 'levenshtein')
          - codeword_generation: str ('rng', 'bch', or 'linear')
          - timestamp: str (e.g., '20260201_002241')
          - path: Path (full path to experiment directory)
    """
    if output_dir is None:
        output_dir = _get_default_output_dir()
    else:
        output_dir = Path(output_dir)

    experiments = []

    for exp_dir in output_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        config_path = exp_dir / "config.json"
        generations_path = exp_dir / "generations.csv"

        # Skip directories missing required files
        if not config_path.exists() or not generations_path.exists():
            continue

        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            experiments.append({
                "model": config["model"]["name"],
                "dataset": config["dataset"]["name"],
                "n": config["watermark"]["n"],
                "u": config["watermark"]["u"],
                "distance": config["watermark"].get("distance", "hamming").replace("-triton", ""),
                "codeword_generation": config["watermark"].get("codeword_generation", "rng"),
                "timestamp": config["experiment"]["timestamp"],
                "path": exp_dir,
            })
        except (json.JSONDecodeError, KeyError):
            # Skip malformed config files
            continue

    df = pd.DataFrame(experiments)

    # Sort by model, dataset, timestamp for consistent ordering
    if not df.empty:
        df = df.sort_values(["model", "dataset", "timestamp"]).reset_index(drop=True)

    return df


def load_experiment(experiment_dir: Path | str) -> dict:
    """
    Load a single experiment's data.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        dict with:
          - 'config': dict (from config.json)
          - 'generations': pd.DataFrame (from generations.csv)
          - 'path': Path (experiment directory)

    Raises:
        FileNotFoundError: If config.json or generations.csv not found
        ValueError: If config.json is malformed
    """
    experiment_dir = Path(experiment_dir)

    config_path = experiment_dir / "config.json"
    generations_path = experiment_dir / "generations.csv"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {experiment_dir}")
    if not generations_path.exists():
        raise FileNotFoundError(f"generations.csv not found in {experiment_dir}")

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed config.json in {experiment_dir}: {e}")

    generations = pd.read_csv(generations_path)

    return {
        "config": config,
        "generations": generations,
        "path": experiment_dir,
    }


def load_and_merge(experiment_paths: list[Path]) -> pd.DataFrame:
    """
    Load multiple experiments by path and merge into one DataFrame.

    Args:
        experiment_paths: List of paths to experiment directories

    Returns:
        DataFrame with columns from generations.csv plus:
          - dataset: str (short dataset name, e.g., 'HC3')

    Raises:
        ValueError: If experiment_paths is empty
    """
    if not experiment_paths:
        raise ValueError("No experiment paths provided")

    dfs = []
    for path in experiment_paths:
        exp = load_experiment(path)
        df = exp["generations"].copy()

        # Extract dataset name (last part after /)
        dataset_name = exp["config"]["dataset"]["name"].split("/")[-1]
        df["dataset"] = dataset_name

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def parse_text_source(args_list: list[str]) -> dict:
    """Parse a text source argument list into a structured descriptor.

    Returns dict with either:
      {"type": "directory", "path": str}
      {"type": "dataset", "name": str, "subset": str, "split": str,
       "column": str, "index": int | None}
    """
    if len(args_list) == 1 and os.path.isdir(args_list[0]):
        return {"type": "directory", "path": args_list[0]}

    if len(args_list) == 4:
        name, subset, split, column = args_list
        col_index = None
        if "[" in column and column.endswith("]"):
            bracket_pos = column.index("[")
            col_index = int(column[bracket_pos + 1 : -1])
            column = column[:bracket_pos]
        return {
            "type": "dataset",
            "name": name,
            "subset": subset,
            "split": split,
            "column": column,
            "index": col_index,
        }

    if len(args_list) == 1:
        raise ValueError(
            f"'{args_list[0]}' is not an existing directory, and a HuggingFace "
            f"dataset requires exactly 4 arguments: name subset split column"
        )
    raise ValueError(
        f"Expected 1 arg (directory) or 4 args (dataset name subset split column), "
        f"got {len(args_list)}: {args_list}"
    )


def source_description(source: dict) -> str:
    """Human-readable description of a text source."""
    if source["type"] == "directory":
        return f"dir: {source['path']}"
    col_desc = source["column"]
    if source.get("index") is not None:
        col_desc += f"[{source['index']}]"
    return f"dataset: {source['name']}/{source['subset']} ({source['split']} split, {col_desc} column)"


def load_texts(source: dict, seed: int, n_texts: int | None = None) -> list[str]:
    """Load texts from either an experiment directory or HuggingFace dataset."""
    from src.utils import get_shuffled_essays

    if source["type"] == "directory":
        experiment = load_experiment(source["path"])
        df = experiment["generations"]
        mask = df["generated_text"].notna() & (df["generated_text"].astype(str) != "")
        texts = df.loc[mask, "generated_text"].astype(str).tolist()
        if n_texts is not None:
            texts = texts[:n_texts]
        return texts

    raw_items = get_shuffled_essays(
        dataset_name=source["name"],
        dataset_subset=source["subset"],
        dataset_split=source["split"],
        dataset_column=source["column"],
        seed=seed,
        n_prompts=n_texts,
    )

    if source.get("index") is not None:
        idx = source["index"]
        texts = []
        for item in raw_items:
            if isinstance(item, (list, tuple)):
                if idx < len(item):
                    texts.append(str(item[idx]))
                else:
                    raise ValueError(
                        f"Column '{source['column']}' item has {len(item)} elements, "
                        f"requested index {idx}"
                    )
            else:
                texts.append(str(item))
        return texts

    return [str(item) for item in raw_items]
