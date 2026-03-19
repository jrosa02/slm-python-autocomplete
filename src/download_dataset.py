"""Download high-quality Python code dataset from Hugging Face for SLM LoRA fine-tuning."""

import json
from pathlib import Path

from datasets import load_dataset, Dataset


def download_python_code_dataset(
    cache_dir: str | None = None,
    max_samples: int | None = None,
    min_length: int = 50,
    max_length: int = 2048,
) -> Dataset:
    """
    Download high-quality source code dataset for autocomplete fine-tuning.

    Uses the 'shibing624/source_code' dataset which contains:
    - Multi-language source code (including Python)
    - High-quality code snippets
    - Suitable for code completion and generation tasks
    - Curated for LoRA fine-tuning

    Args:
        cache_dir: Local directory to cache dataset. Defaults to ~/.cache/huggingface/datasets
        max_samples: Maximum number of samples to load. None = all samples
        min_length: Minimum code snippet length in characters
        max_length: Maximum code snippet length in characters

    Returns:
        Loaded and filtered Hugging Face Dataset ready for LoRA fine-tuning
    """
    print("🔄 Downloading source code dataset from Hugging Face...")

    # Load the dataset - shibing624/source_code contains high-quality source code
    dataset = load_dataset(
        "shibing624/source_code",
        "python",
        split="train",
        cache_dir=cache_dir,
    )

    print(f"✓ Loaded {len(dataset)} total samples")

    # Filter by length for quality control
    def is_valid_length(example):
        # Determine the code field name - could be "code", "text", or "content"
        code = example.get("code") or example.get("text") or example.get("content", "")
        code_len = len(code)
        return min_length <= code_len <= max_length

    dataset = dataset.filter(is_valid_length)
    print(f"✓ Filtered to {len(dataset)} samples (length: {min_length}-{max_length} chars)")

    # Limit samples if requested
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"✓ Limited to {len(dataset)} samples")

    return dataset


def download_code_search_net_dataset(
    cache_dir: str | None = None,
    max_samples: int | None = 1000,
) -> Dataset:
    """
    Alternative: Download CodeSearchNet dataset with multiple programming languages.

    Uses high-quality code from open-source repositories.

    Args:
        cache_dir: Local directory to cache dataset
        max_samples: Maximum number of samples

    Returns:
        Loaded Dataset with source code examples
    """
    print("🔄 Downloading CodeSearchNet dataset as fallback...")

    dataset = load_dataset(
        "code_search_net",
        "python",
        split="train",
        cache_dir=cache_dir,
    )

    print(f"✓ Loaded {len(dataset)} total samples")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"✓ Limited to {len(dataset)} samples")

    return dataset


def download_code_x_glue_completion_dataset(
    cache_dir: str | None = None,
    max_samples: int | None = None,
    min_length: int = 20,
    max_length: int = 512,
) -> Dataset:
    """
    Download Code-X-GLUE code completion dataset.

    This dataset is specifically designed for line-level code completion:
    - Curated for code completion at line-level
    - High-quality programming examples
    - Optimal for autocomplete fine-tuning tasks
    - Smaller context windows suitable for SLM

    Args:
        cache_dir: Local directory to cache dataset
        max_samples: Maximum number of samples to load
        min_length: Minimum code line length in characters
        max_length: Maximum code line length in characters

    Returns:
        Loaded and filtered Dataset ready for LoRA fine-tuning
    """
    print("🔄 Downloading Code-X-GLUE code completion dataset...")

    dataset_dict = load_dataset(
        "google/code_x_glue_cc_code_completion_line",
        "python",
        cache_dir=cache_dir,
    )

    # Extract train split - dataset returns DatasetDict
    dataset = dataset_dict["train"] if isinstance(dataset_dict, dict) else dataset_dict

    print(f"✓ Loaded {len(dataset)} total samples")

    # Filter by length for quality control
    def is_valid_length(example):
        # Code-X-GLUE uses 'input' field for code context
        code = example.get("input") or ""
        code_len = len(code)
        return min_length <= code_len <= max_length

    dataset = dataset.filter(is_valid_length)
    print(f"✓ Filtered to {len(dataset)} samples (length: {min_length}-{max_length} chars)")

    # Limit samples if requested
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"✓ Limited to {len(dataset)} samples")

    return dataset


def prepare_for_lora_finetuning(
    dataset: Dataset,
    output_dir: str | Path = "data/formatted",
    format_type: str = "completion",
) -> Path:
    """
    Format dataset for LoRA fine-tuning.

    Args:
        dataset: Input dataset from Hugging Face
        output_dir: Output directory for formatted data
        format_type: "completion" (next tokens) or "instruction" (instruction-following)

    Returns:
        Path to output directory with formatted data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📝 Preparing {len(dataset)} samples for LoRA fine-tuning...")

    formatted_data = []

    if format_type == "completion":
        # For code completion: use raw code as training text
        for example in dataset:
            # Handle multiple possible field names from various datasets
            code = (example.get("code") or
                   example.get("input") or
                   example.get("gt_code") or
                   example.get("text") or
                   example.get("content") or
                   example.get("source_code") or
                   "")
            if code and len(code) > 10:
                formatted_data.append({
                    "text": code,
                    "type": "source_code",
                })

    elif format_type == "instruction":
        # For instruction-based fine-tuning
        for example in dataset:
            code = (example.get("code") or
                   example.get("input") or
                   example.get("gt_code") or
                   example.get("text") or
                   example.get("content") or
                   example.get("source_code") or
                   "")
            if code and len(code) > 10:
                formatted_data.append({
                    "instruction": "Complete this code:",
                    "input": code[:100],  # First 100 chars as context
                    "output": code,
                })

    # Save as JSONL (one record per line - efficient for large datasets)
    output_file = output_dir / f"{format_type}_data.jsonl"
    with open(output_file, "w") as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")

    print(f"✓ Saved {len(formatted_data)} formatted examples to {output_file}")

    # Save dataset info
    info_file = output_dir / "dataset_info.json"
    with open(info_file, "w") as f:
        json.dump({
            "total_samples": len(formatted_data),
            "format_type": format_type,
            "min_chars": min(len(item["text"]) for item in formatted_data) if formatted_data else 0,
            "max_chars": max(len(item["text"]) for item in formatted_data) if formatted_data else 0,
            "avg_chars": sum(len(item["text"]) for item in formatted_data) // len(formatted_data)
            if formatted_data
            else 0,
        }, f, indent=2)

    return output_file


def main(dataset_name: str = "shibing624", max_samples: int = 5000):
    """
    Download and prepare dataset for fine-tuning.

    Args:
        dataset_name: Which dataset to download
            - "shibing624": shibing624/source_code (primary, 5k samples)
            - "code_x_glue": code_x_glue_cc_code_completion_line (specialized for completion)
            - "codesearch": CodeSearchNet (alternative)
        max_samples: Maximum samples to download
    """
    import sys

    dataset = None

    # Try primary dataset based on selection
    if dataset_name == "shibing624":
        try:
            print("📦 Downloading shibing624/source_code dataset...")
            dataset = download_python_code_dataset(
                max_samples=max_samples,
                min_length=50,
                max_length=2048,
            )
        except Exception as e:
            print(f"⚠️  shibing624 dataset unavailable: {e}")

    elif dataset_name == "code_x_glue":
        try:
            print("📦 Downloading code_x_glue_cc_code_completion_line dataset...")
            dataset = download_code_x_glue_completion_dataset(
                max_samples=max_samples,
                min_length=20,
                max_length=4096,
            )
        except Exception as e:
            print(f"⚠️  Code-X-GLUE dataset unavailable: {e}")

    elif dataset_name == "codesearch":
        try:
            print("📦 Downloading CodeSearchNet dataset...")
            dataset = download_code_search_net_dataset(max_samples=max_samples)
        except Exception as e:
            print(f"⚠️  CodeSearchNet dataset unavailable: {e}")

    # Fallback chain if primary fails
    if dataset is None:
        print("\n⚠️  Attempting fallback datasets...")
        fallbacks = [
            ("code_x_glue", download_code_x_glue_completion_dataset),
            ("shibing624", download_python_code_dataset),
            ("codesearch", download_code_search_net_dataset),
        ]

        for fallback_name, fallback_func in fallbacks:
            if fallback_name == dataset_name:
                continue  # Skip if it's the one that already failed
            try:
                print(f"  Trying {fallback_name}...")
                if fallback_name == "code_x_glue":
                    dataset = fallback_func(max_samples=max_samples, min_length=20, max_length=512)
                else:
                    dataset = fallback_func(max_samples=max_samples)
                break
            except Exception as e:
                print(f"    Failed: {e}")
                continue

    if dataset is None:
        print("❌ All datasets failed to download")
        sys.exit(1)

    # Prepare for fine-tuning
    output_path = prepare_for_lora_finetuning(dataset, format_type="completion")
    print(f"\n✅ Dataset ready at: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download code datasets for SLM LoRA fine-tuning")
    parser.add_argument(
        "--dataset",
        choices=["shibing624", "code_x_glue", "codesearch"],
        default="shibing624",
        help="Which dataset to download (default: shibing624)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Maximum number of samples to download (default: 5000)",
    )

    args = parser.parse_args()
    main(dataset_name=args.dataset, max_samples=args.samples)
