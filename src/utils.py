import os
import sys
import random
from typing import Dict, List, Tuple
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
    n_prompts: int
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
    
    # Check if we have enough prompts
    if n_prompts > total_prompts_in_dataset:
        raise ValueError(f"Requested {n_prompts} prompts but dataset '{dataset_name}' "
                         f"({dataset_subset} subset, {dataset_split} split) only contains {total_prompts_in_dataset} entries.")
    
    # Create a list of indices and shuffle them
    prompt_indices = list(range(total_prompts_in_dataset))
    random.shuffle(prompt_indices)
    
    # Get the first n_prompts indices
    selected_indices = prompt_indices[:n_prompts]
    
    # Extract the prompts
    prompts = []
    for idx in selected_indices:
        prompt_data = dataset[dataset_split][idx]
        try:
            prompt_text = prompt_data[dataset_column]
        except KeyError:
            raise ValueError(f"Column '{dataset_column}' not found in dataset entry at index {idx} of split '{dataset_split}'. "
                             f"Available columns: {list(prompt_data.keys())}")
        if not prompt_text:
            raise ValueError(f"No text found in dataset entry at index {idx} for column '{dataset_column}' in split '{dataset_split}'.")
        prompts.append(prompt_text)
    
    return prompts