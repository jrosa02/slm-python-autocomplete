"""PyTorch Lightning DataModule for code completion fine-tuning."""

import json
from pathlib import Path
from typing import Optional

import pytorch_lightning as L
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class CodeCompletionDataset(Dataset):
    """Dataset for code completion training."""

    def __init__(
        self,
        data_file: Path,
        tokenizer,
        max_length: int = 512,
        truncation: bool = True,
        padding: str = "max_length",
    ):
        """
        Initialize code completion dataset.

        Args:
            data_file: Path to JSONL file with code examples
            tokenizer: Hugging Face tokenizer (e.g., Qwen tokenizer)
            max_length: Maximum token sequence length
            truncation: Whether to truncate sequences longer than max_length
            padding: Padding strategy ("max_length", "longest", or False)
        """
        self.data_file = Path(data_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        self.examples = []

        self._load_data()

    def _load_data(self):
        """Load examples from JSONL file."""
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        with open(self.data_file) as f:
            for line in f:
                if line.strip():
                    example = json.loads(line)
                    # Extract code text from various field names
                    text = example.get("text") or example.get("code") or ""
                    if text:
                        self.examples.append(text)

        print(f"✓ Loaded {len(self.examples)} examples from {self.data_file.name}")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single training example.

        Returns tokenized code with attention masks and labels for causal language modeling.
        """
        text = self.examples[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors="pt",
        )

        # For causal language modeling, labels = input_ids shifted by 1
        # This allows the model to predict the next token
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels are the input_ids (for causal language modeling)
        labels = input_ids.clone()

        # Mask out padding tokens in the loss calculation
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class CodeLLMDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for code completion fine-tuning.

    Designed for fine-tuning Qwen2.5Coder3B using LoRA on Python code completion.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/formatted",
        model_name: str = "Qwen/Qwen2.5-Coder-3B",
        batch_size: int = 8,
        max_seq_length: int = 512,
        num_workers: int = 0,
        train_split_ratio: float = 0.8,
        val_split_ratio: float = 0.1,
        seed: int = 42,
        cache_dir: str | None = None,
    ):
        """
        Initialize DataModule.

        Args:
            data_dir: Directory containing formatted dataset (JSONL files)
            model_name: Hugging Face model name for tokenizer
            batch_size: Batch size for training/validation
            max_seq_length: Maximum sequence length for tokenization
            num_workers: Number of workers for DataLoader
            train_split_ratio: Proportion of data for training (0.8 = 80%)
            val_split_ratio: Proportion of data for validation (0.1 = 10%)
            seed: Random seed for reproducibility
            cache_dir: Local directory to cache models/tokenizers
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio
        self.seed = seed
        self.cache_dir = cache_dir

        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        Prepare datasets for training/validation/testing.

        Loads tokenizer and creates Dataset instances with train/val/test splits.
        """
        if self.tokenizer is None:
            print(f"📥 Loading tokenizer from {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            # Set pad token if not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Find and load the completion data file
        data_file = self.data_dir / "completion_data.jsonl"
        if not data_file.exists():
            raise FileNotFoundError(
                f"Dataset not found at {data_file}. "
                "Run 'python -m src.download_dataset' first."
            )

        # Load full dataset
        print(f"📊 Loading dataset from {data_file.name}...")
        full_dataset = CodeCompletionDataset(
            data_file=data_file,
            tokenizer=self.tokenizer,
            max_length=self.max_seq_length,
        )

        # Split into train/val/test
        total_size = len(full_dataset)
        train_size = int(total_size * self.train_split_ratio)
        val_size = int(total_size * self.val_split_ratio)
        test_size = total_size - train_size - val_size

        # Create indices for splitting
        import random

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        indices = list(range(total_size))
        random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        # Create subset datasets
        self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        self.test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

        print(
            f"✓ Split dataset: train={len(self.train_dataset)}, "
            f"val={len(self.val_dataset)}, test={len(self.test_dataset)}"
        )

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        """Return prediction DataLoader (uses test set)."""
        return self.test_dataloader()

    @property
    def tokenizer_config(self) -> dict:
        """Return tokenizer configuration for model training."""
        return {
            "model_name": self.model_name,
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None,
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer else None,
            "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer else None,
        }