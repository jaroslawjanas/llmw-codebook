import json
import os
import sys
import random
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
import src.paths as paths


def load_hf_token():
    """Load HuggingFace token from hf_token file."""
    token_path = "hf_token"
    token = None
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            token = f.read().strip()
    return token

def get_shuffled_essays(
    dataset_name: str,
    dataset_subset: str,
    dataset_split: str,
    dataset_column: str,
    seed: int,
    n_prompts: Optional[int] = None
) -> List[str]:
    """
    Get a shuffled list of prompts from a specified Hugging Face dataset, subset, split, and column.
    
    Args:
        dataset_name: The name of the Hugging Face dataset (e.g., "ChristophSchuhmann/essays-with-instructions").
        dataset_subset: The name of the dataset subset (e.g., "default").
        dataset_split: The name of the dataset split (e.g., "train", "validation", "test").
        dataset_column: The name of the column in the dataset to extract prompts from (e.g., "text" or "instructions").
        seed: Random seed for reproducibility.
        n_prompts: Number of prompts to return.
        
    Returns:
        List of prompt texts (shuffled deterministically based on seed).
    """
    # Set random seed for reproducible shuffling
    random.seed(seed)
    
    # Get HuggingFace token if available
    token = load_hf_token()
    
    # Configure dataset loading options
    dataset_kwargs = {
        "cache_dir": paths.DATASETS_CACHE_DIR  # Use datasets subdirectory
    }
    if token:
        dataset_kwargs["token"] = token
    
    # Load the dataset
    try:
        dataset = load_dataset(dataset_name, dataset_subset, **dataset_kwargs)
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{dataset_name}' with subset '{dataset_subset}': {e}")
    
    if dataset_split not in dataset:
        raise ValueError(f"Dataset '{dataset_name}' with subset '{dataset_subset}' does not have a '{dataset_split}' split. "
                         f"Available splits: {list(dataset.keys())}")
        
    total_prompts_in_dataset = len(dataset[dataset_split])

    all_texts = []
    for idx in range(total_prompts_in_dataset):
        prompt_data = dataset[dataset_split][idx]
        try:
            prompt_text = prompt_data[dataset_column]
        except KeyError:
            raise ValueError(f"Column '{dataset_column}' not found in dataset entry at index {idx} of split '{dataset_split}'. "
                             f"Available columns: {list(prompt_data.keys())}")
        if not prompt_text:
            raise ValueError(f"No text found in dataset entry at index {idx} for column '{dataset_column}' in split '{dataset_split}'.")
        all_texts.append(prompt_text)

    return get_shuffled_texts(all_texts, seed, n_prompts)


def get_shuffled_texts(
    texts: List[str],
    seed: int,
    n_texts: Optional[int] = None
) -> List[str]:
    """
    Deterministically shuffle a list of texts and return a subset.

    Args:
        texts: Pre-loaded list of text strings.
        seed: Random seed for reproducible shuffling.
        n_texts: Number of texts to return. If None, returns all.

    Returns:
        Shuffled subset of texts.

    Raises:
        ValueError: If n_texts exceeds len(texts).
    """
    random.seed(seed)

    total = len(texts)
    if n_texts is None:
        n_texts = total
    elif n_texts > total:
        raise ValueError(f"Requested {n_texts} texts but only {total} are available.")

    indices = list(range(total))
    random.shuffle(indices)
    selected = indices[:n_texts]

    return [texts[i] for i in selected]


def load_local_texts(path: str) -> List[str]:
    """
    Load prompts from a local JSON or TXT file.

    Args:
        path: File path. Must end in .json or .txt.

    Returns:
        List of prompt strings.

    Raises:
        ValueError: If file extension is unsupported, JSON is not a list, or file is empty.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                f"JSON file '{path}' must contain a top-level array (list of strings). "
                f"Got {type(data).__name__}."
            )

        texts = [str(item) for item in data]

        if not texts:
            raise ValueError(f"JSON file '{path}' is an empty array.")

        return texts

    elif ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        texts = []
        for line in lines:
            stripped = line.rstrip("\n")
            if stripped.strip():
                texts.append(stripped)

        if not texts:
            raise ValueError(f"TXT file '{path}' contains no non-empty lines.")

        return texts

    else:
        raise ValueError(
            f"Unsupported file extension '{ext}' for '{path}'. "
            f"Only .json and .txt are supported."
        )