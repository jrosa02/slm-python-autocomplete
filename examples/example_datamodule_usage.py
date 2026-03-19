"""Example: Using the CodeLLMDataModule with PyTorch Lightning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_module import CodeLLMDataModule


def main():
    """Demonstrate DataModule usage."""
    print("🚀 PyTorch Lightning DataModule Example\n")

    # Initialize DataModule
    print("1️⃣ Initializing DataModule...")
    dm = CodeLLMDataModule(
        data_dir="data/formatted",
        model_name="Qwen/Qwen2.5-Coder-3B",
        batch_size=8,
        max_seq_length=512,
        train_split_ratio=0.8,
        val_split_ratio=0.1,
        num_workers=0,
    )

    # Setup datasets
    print("\n2️⃣ Setting up datasets...")
    dm.setup()

    # Get data loaders
    print("\n3️⃣ Creating data loaders...")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Inspect a single batch
    print("\n4️⃣ Inspecting a training batch...")
    batch = next(iter(train_loader))
    print(f"   Batch keys: {batch.keys()}")
    print(f"   Input shape: {batch['input_ids'].shape}")
    print(f"   Attention mask shape: {batch['attention_mask'].shape}")
    print(f"   Labels shape: {batch['labels'].shape}")

    # Show tokenizer config
    print("\n5️⃣ Tokenizer configuration:")
    config = dm.tokenizer_config
    for key, value in config.items():
        print(f"   {key}: {value}")

    print("\n✅ DataModule ready for model training with LoRA!")
    print("\nNext steps:")
    print("  1. Initialize Qwen2.5-Coder-3B model")
    print("  2. Apply LoRA adapters via peft")
    print("  3. Create Lightning Trainer with desired callbacks")
    print("  4. Call trainer.fit(model, dm)")


if __name__ == "__main__":
    main()
