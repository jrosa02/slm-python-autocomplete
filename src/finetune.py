"""Fine-tune Qwen models with LoRA using PyTorch Lightning."""

from pathlib import Path
from typing import Optional

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_module import CodeLLMDataModule


class LoRACodeModel(L.LightningModule):
    """
    PyTorch Lightning module for LoRA fine-tuning of code LLMs.

    Wraps a causal language model with LoRA adapters and provides
    training/validation/test loops.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-3B",
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        learning_rate: float = 1e-4,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        cache_dir: str | None = None,
    ):
        """
        Initialize LoRA fine-tuning module.

        Args:
            model_name: Hugging Face model identifier
            lora_rank: LoRA rank (r parameter)
            lora_alpha: LoRA scaling factor (alpha parameter)
            lora_dropout: LoRA dropout rate
            learning_rate: Learning rate for optimizer
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for AdamW optimizer
            cache_dir: Local directory to cache models (e.g., "models")
        """
        super().__init__()
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.cache_dir = cache_dir

        # Save hyperparameters for checkpointing
        self.save_hyperparameters()

        # Load model
        self.model = self._load_model_with_lora()

    def _load_model_with_lora(self):
        """Load model and apply LoRA adapters."""
        print(f"📥 Loading base model: {self.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            cache_dir=self.cache_dir,
        )

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            target_modules=["q_proj", "v_proj"],  # Common for Qwen/LLaMA models
            modules_to_save=None,
        )

        print(f"🎯 Applying LoRA adapters (rank={self.lora_rank}, alpha={self.lora_alpha})")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def training_step(self, batch, batch_idx):
        """Training step."""
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Optional: add warmup scheduler
        from transformers import get_linear_schedule_with_warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def train(
    model_name: str = "Qwen/Qwen2.5-Coder-3B",
    data_dir: str = "data/formatted",
    output_dir: str = "outputs/checkpoints",
    batch_size: int = 4,
    max_epochs: int = 3,
    lora_rank: int = 8,
    learning_rate: float = 5e-4,
    accumulate_grad_batches: int = 1,
    precision: str = "16-mixed",
    num_workers: int = 0,
    cache_dir: str | None = None,
    enable_early_stopping: bool = True,
):
    """
    Train model with LoRA fine-tuning.

    Args:
        model_name: Hugging Face model identifier
        data_dir: Path to formatted dataset directory
        output_dir: Output directory for checkpoints
        batch_size: Batch size for training
        max_epochs: Number of training epochs
        lora_rank: LoRA rank parameter
        learning_rate: Learning rate
        accumulate_grad_batches: Gradient accumulation steps
        precision: Training precision ("32", "16-mixed", "bf16-mixed")
        num_workers: Number of workers for data loading
        cache_dir: Local directory to cache models/tokenizers (e.g., "models")
    """
    print("\n" + "=" * 60)
    print("🚀 Starting LoRA Fine-tuning")
    print("=" * 60)

    # Initialize DataModule
    print("\n📦 Setting up DataModule...")
    dm = CodeLLMDataModule(
        data_dir=data_dir,
        model_name=model_name,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
    )

    # Initialize model
    print("\n🤖 Initializing LoRA model...")
    model = LoRACodeModel(
        model_name=model_name,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        cache_dir=cache_dir,
    )

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    print("\n⚙️  Configuring trainer...")
    callbacks_list = [
        ModelCheckpoint(
            dirpath=output_path,
            filename="model-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        ),
    ]
    if enable_early_stopping:
        callbacks_list.append(
            EarlyStopping(
                monitor="val_loss",
                patience=2,
                mode="min",
            )
        )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        default_root_dir=str(output_path),
        callbacks=callbacks_list,
    )

    # Train
    print("\n🔥 Starting training...")
    print("-" * 60)
    trainer.fit(model, datamodule=dm)

    # Test
    print("\n📊 Evaluating on test set...")
    print("-" * 60)
    trainer.test(model, datamodule=dm)

    print("\n" + "=" * 60)
    print("✅ Fine-tuning complete!")
    print(f"📁 Checkpoints saved to: {output_path.absolute()}")
    print("=" * 60 + "\n")

    return model, trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-3B",
        help="Model to fine-tune",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/formatted",
        help="Path to formatted dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Training precision",
    )

    args = parser.parse_args()

    train(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        precision=args.precision,
    )