"""Data loading utilities for experiment results."""

from __future__ import annotations

import json
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


def load_by_model(
    model: str,
    output_dir: Path | str = None
) -> pd.DataFrame:
    """
    Load and merge all experiments for a specific model.

    Args:
        model: Model name (e.g., 'meta-llama/Llama-3.2-3B')
        output_dir: Output directory (default: project output/)

    Returns:
        DataFrame with columns:
          - watermark: int64
          - codeword: str
          - decoded_codeword: str
          - recovered_watermark: int64
          - bit_errors: int64
          - decode_success: bool
          - dataset: str (added - source dataset name)

    Raises:
        ValueError: If no experiments found for the specified model
    """
    experiments = discover_experiments(output_dir)

    # Filter to matching model
    model_experiments = experiments[experiments["model"] == model]

    if model_experiments.empty:
        available_models = experiments["model"].unique().tolist()
        raise ValueError(
            f"No experiments found for model '{model}'. "
            f"Available models: {available_models}"
        )

    dfs = []
    for _, row in model_experiments.iterrows():
        exp = load_experiment(row["path"])
        df = exp["generations"].copy()

        # Extract dataset name (last part after /)
        dataset_name = exp["config"]["dataset"]["name"].split("/")[-1]
        df["dataset"] = dataset_name

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
