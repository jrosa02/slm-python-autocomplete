"""Download and cache LLM models from Hugging Face for fine-tuning."""

from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-3B"
CACHE_DIR = Path("models")


def download_model(
    model_name: str = DEFAULT_MODEL,
    cache_dir: Optional[Path] = None,
    trust_remote_code: bool = True,
    device_map: str = "auto",
) -> tuple:
    """
    Download and cache model and tokenizer from Hugging Face.

    Args:
        model_name: Hugging Face model identifier (e.g., "Qwen/Qwen2.5-Coder-3B")
        cache_dir: Local directory to cache models. Defaults to ./models
        trust_remote_code: Allow execution of custom code from remote
        device_map: How to distribute model across devices ("auto", "cpu", "cuda", etc.)

    Returns:
        Tuple of (model, tokenizer)

    Examples:
        >>> # Download Qwen2.5-Coder-3B (default)
        >>> model, tokenizer = download_model()

        >>> # Download a different model
        >>> model, tokenizer = download_model("meta-llama/Llama-2-7b")

        >>> # Use custom cache directory
        >>> model, tokenizer = download_model(cache_dir=Path("/custom/path"))
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading {model_name}...")
    print(f"   Cache directory: {cache_dir.absolute()}")

    # Download tokenizer
    print(f"\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        trust_remote_code=trust_remote_code,
    )
    print(f"   ✓ Tokenizer loaded ({tokenizer.__class__.__name__})")

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   ℹ️  Pad token set to eos_token")

    # Download model
    print(f"\n🤖 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        torch_dtype="auto",
    )
    print(f"   ✓ Model loaded ({model.__class__.__name__})")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Print model config
    if hasattr(model, "config"):
        config = model.config
        print(f"\n⚙️  Model Config:")
        print(f"   Hidden size: {config.hidden_size if hasattr(config, 'hidden_size') else 'N/A'}")
        print(f"   Num layers: {config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 'N/A'}")
        print(f"   Vocab size: {config.vocab_size if hasattr(config, 'vocab_size') else 'N/A'}")

    print(f"\n✅ Model and tokenizer ready for fine-tuning!")

    return model, tokenizer


def download_tokenizer_only(
    model_name: str = DEFAULT_MODEL,
    cache_dir: Optional[Path] = None,
    trust_remote_code: bool = True,
):
    """
    Download and cache only the tokenizer from Hugging Face.

    Useful when you only need the tokenizer for preprocessing.

    Args:
        model_name: Hugging Face model identifier
        cache_dir: Local directory to cache tokenizer
        trust_remote_code: Allow execution of custom code from remote

    Returns:
        Tokenizer instance
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔤 Downloading tokenizer for {model_name}...")
    print(f"   Cache directory: {cache_dir.absolute()}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✅ Tokenizer ready!")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print(f"   Type: {tokenizer.__class__.__name__}")

    return tokenizer


def list_cache_dir(cache_dir: Optional[Path] = None) -> dict:
    """
    List all cached models in the cache directory.

    Args:
        cache_dir: Cache directory to list. Defaults to ./models

    Returns:
        Dictionary with model names and their sizes
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR

    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        return {}

    models = {}
    for model_dir in cache_dir.glob("**/models--*"):
        model_name = model_dir.name.replace("models--", "").replace("--", "/")
        size_gb = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / (1024**3)
        models[model_name] = size_gb

    if models:
        print(f"\n📦 Cached models in {cache_dir.absolute()}:")
        for model_name, size_gb in sorted(models.items()):
            print(f"   {model_name}: {size_gb:.2f} GB")
    else:
        print(f"No cached models found in {cache_dir.absolute()}")

    return models


def main():
    """CLI for downloading models."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Download LLM models for fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Qwen2.5-Coder-3B (default)
  python -m src.model_downloader

  # Download a different model
  python -m src.model_downloader --model meta-llama/Llama-2-7b

  # Download tokenizer only
  python -m src.model_downloader --tokenizer-only

  # List cached models
  python -m src.model_downloader --list
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to download (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help=f"Cache directory (default: {CACHE_DIR})",
    )
    parser.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="Download tokenizer only (skip model)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List cached models and exit",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to load model on (default: auto)",
    )
    parser.add_argument(
        "--no-remote-code",
        action="store_true",
        help="Disable execution of custom remote code",
    )

    args = parser.parse_args()

    # List cached models
    if args.list:
        list_cache_dir(args.cache_dir)
        return

    # Download tokenizer only
    if args.tokenizer_only:
        download_tokenizer_only(
            model_name=args.model,
            cache_dir=args.cache_dir,
            trust_remote_code=not args.no_remote_code,
        )
        return

    # Download full model
    try:
        download_model(
            model_name=args.model,
            cache_dir=args.cache_dir,
            trust_remote_code=not args.no_remote_code,
            device_map=args.device,
        )
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()