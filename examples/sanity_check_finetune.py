"""Ultra-short sanity check for fine-tuning on weak machines."""

import sys
import tempfile
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.finetune import LoRACodeModel, train
from src.data_module import CodeLLMDataModule, CodeCompletionDataset


def _is_model_cached(model_name: str = "Qwen/Qwen2.5-Coder-3B") -> bool:
    """Check if model is already cached locally in project models/ or HF cache."""
    # Check local project cache first
    local_cache = Path("models") / f"models--{model_name.replace('/', '--')}"
    if local_cache.exists():
        return True

    # Fall back to global HuggingFace cache
    import os

    hf_cache = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
    model_cache_dir = hf_cache / f"models--{model_name.replace('/', '--')}"
    return model_cache_dir.exists()


def create_dummy_dataset(num_samples: int = 2, save_path: Path | None = None) -> Path:
    """Create a tiny dummy dataset for testing."""
    import json

    if save_path is None:
        save_path = Path(tempfile.mkdtemp()) / "test_data.jsonl"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Create minimal code examples
    code_samples = [
        "def add(a, b):\n    return a + b",
        "def greet(name):\n    print(f'Hello {name}')",
    ]

    with open(save_path, "w") as f:
        for i in range(num_samples):
            example = {"text": code_samples[i % len(code_samples)]}
            f.write(json.dumps(example) + "\n")

    print(f"✓ Created dummy dataset with {num_samples} samples at {save_path}")
    return save_path


def sanity_check_dataset():
    """Check if dataset loading works."""
    print("\n" + "=" * 60)
    print("1️⃣ Testing Dataset Loading")
    print("=" * 60)

    # Create dummy dataset
    data_file = create_dummy_dataset(num_samples=2)

    # Try loading with tokenizer
    from transformers import AutoTokenizer

    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-3B",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Tokenizer loaded")

    # Load dataset
    print("\n📊 Loading dataset...")
    dataset = CodeCompletionDataset(
        data_file=data_file,
        tokenizer=tokenizer,
        max_length=256,
    )

    print(f"✓ Dataset loaded ({len(dataset)} samples)")

    # Check a batch
    print("\n🔍 Inspecting batch...")
    sample = dataset[0]
    print(f"  Keys: {sample.keys()}")
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Attention mask shape: {sample['attention_mask'].shape}")
    print(f"  Labels shape: {sample['labels'].shape}")

    return True


def sanity_check_model_and_forward_pass():
    """Check if model loading, LoRA, and forward pass work (loads model once)."""
    print("\n" + "=" * 60)
    print("2️⃣ Testing Model with LoRA")
    print("=" * 60)

    print("\n🤖 Initializing LoRA model...")
    try:
        model = LoRACodeModel(
            model_name="Qwen/Qwen2.5-Coder-3B",
            lora_rank=4,  # Very small for testing
            learning_rate=5e-4,
            cache_dir="models",
        )
        print("✓ Model initialized with LoRA")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return False

    print("\n" + "=" * 60)
    print("3️⃣ Testing Forward Pass")
    print("=" * 60)

    print("\n🔤 Loading tokenizer...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-3B",
        trust_remote_code=True,
        cache_dir="models",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dummy input
    print("\n📝 Creating dummy input...")
    text = "def hello():\n    print('hello')"
    inputs = tokenizer(
        text,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    print(f"  Input shape: {inputs['input_ids'].shape}")

    # Forward pass
    print("\n🚀 Running forward pass...")
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["input_ids"],
            )
        print(f"✓ Forward pass successful")
        print(f"  Loss: {outputs.loss.item():.4f}")
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False


def sanity_check_mini_training():
    """Ultra-mini training using finetune.py with 2 epochs on dummy data."""
    print("\n" + "=" * 60)
    print("4️⃣ Testing Mini Training Loop (finetune.py)")
    print("=" * 60)

    print("\n⚙️  Setting up training with dummy data...")

    # Create temporary dummy dataset directory
    temp_dir = Path(tempfile.mkdtemp())
    data_file = temp_dir / "completion_data.jsonl"

    # Create dummy data with 2 samples
    import json

    with open(data_file, "w") as f:
        samples = [
            "def add(a, b):\n    return a + b",
            "def greet(name):\n    print(f'Hello {name}')",
        ]
        for text in samples:
            f.write(json.dumps({"text": text}) + "\n")

    print(f"✓ Created dummy dataset at {data_file}")

    # Run training using finetune.train()
    print("\n🔥 Running mini training with finetune.train()...")
    try:
        model, trainer = train(
            model_name="Qwen/Qwen2.5-Coder-3B",
            data_dir=str(temp_dir),
            output_dir=str(temp_dir / "checkpoints"),
            batch_size=1,
            max_epochs=1,  # Just 1 epoch for sanity check
            lora_rank=4,
            learning_rate=5e-4,
            precision="32",  # Use fp32 for weak machines
            num_workers=0,
            cache_dir="models",
            enable_early_stopping=False,  # Disabled for mini dataset
        )
        print("✓ Training loop completed successfully")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all sanity checks."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  🧪 FINE-TUNING SANITY CHECK (Weak Machine)".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")

    # Check if model is already cached
    model_name = "Qwen/Qwen2.5-Coder-3B"
    if _is_model_cached(model_name):
        print(f"\n✅ Model cache found: {model_name}")
        print("   Skipping download, using cached version")
    else:
        print(f"\n📥 Model not cached: {model_name}")
        print("   Will download on first load (tests 2-4)")

    results = {}

    # Test 1: Dataset
    try:
        results["Dataset Loading"] = sanity_check_dataset()
    except Exception as e:
        print(f"\n❌ Dataset check failed: {e}")
        results["Dataset Loading"] = False

    # Test 2-3: Model initialization and forward pass (loads model once)
    try:
        results["Model + Forward Pass"] = sanity_check_model_and_forward_pass()
    except Exception as e:
        print(f"\n❌ Model/forward pass check failed: {e}")
        results["Model + Forward Pass"] = False

    # Test 4: Mini training
    try:
        results["Mini Training"] = sanity_check_mini_training()
    except Exception as e:
        print(f"\n❌ Training check failed: {e}")
        results["Mini Training"] = False

    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8} | {test_name}")

    all_passed = all(results.values())

    print("=" * 60)

    if all_passed:
        print("\n✅ All sanity checks passed!")
        print("\nYou're ready to:")
        print("  1. Run full fine-tuning: python -m src.finetune")
        print("  2. Send setup to server")
        print("\n")
        return 0
    else:
        print("\n❌ Some checks failed. Debug issues before proceeding.")
        print("\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)