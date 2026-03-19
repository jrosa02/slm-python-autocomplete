"""
Main orchestration script for SLM LoRA fine-tuning pipeline.

Loads configuration from config.yaml and runs all processes sequentially:
1. Download dataset
2. Download model
3. Fine-tune with LoRA

Saves all training metrics to a JSON file for analysis.
"""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.download_dataset import download_python_code_dataset, prepare_for_lora_finetuning
from src.finetune import train
from src.model_downloader import download_model


class MetricsCollector:
    """Collect and save training metrics."""

    def __init__(self, metrics_dir: str | Path):
        """Initialize metrics collector."""
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_stages": {},
        }

    def record_stage(self, stage_name: str, status: str, details: dict | None = None):
        """Record completion of a pipeline stage."""
        self.metrics["pipeline_stages"][stage_name] = {
            "status": status,  # "success", "failed", "skipped"
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
        }

    def save(self, filename: str = "training_metrics.json"):
        """Save metrics to JSON file."""
        output_path = self.metrics_dir / filename
        with open(output_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"\n📊 Metrics saved to: {output_path.absolute()}")
        return output_path


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"✓ Loaded config from {config_path.absolute()}")
    return config


def download_dataset_stage(config: dict, metrics: MetricsCollector):
    """Download and prepare dataset."""
    print("\n" + "=" * 60)
    print("1️⃣ DATASET DOWNLOAD")
    print("=" * 60)

    try:
        dataset_cfg = config["dataset"]
        print(f"\n📦 Downloading dataset: {dataset_cfg['dataset_name']}")
        print(f"   Max samples: {dataset_cfg['max_samples']}")

        # Download dataset
        dataset = download_python_code_dataset(
            max_samples=dataset_cfg["max_samples"],
            min_length=dataset_cfg.get("min_length", 50),
            max_length=dataset_cfg.get("max_length", 2048),
        )

        # Prepare for training
        output_file = prepare_for_lora_finetuning(
            dataset,
            output_dir=dataset_cfg["output_dir"],
            format_type=dataset_cfg.get("format_type", "completion"),
        )

        metrics.record_stage(
            "dataset_download",
            "success",
            {
                "dataset_name": dataset_cfg["dataset_name"],
                "total_samples": len(dataset),
                "output_file": str(output_file),
            },
        )

        print(f"\n✅ Dataset downloaded and prepared successfully")
        return True

    except Exception as e:
        print(f"\n❌ Dataset download failed: {e}")
        traceback.print_exc()
        metrics.record_stage("dataset_download", "failed", {"error": str(e)})
        return False


def download_model_stage(config: dict, metrics: MetricsCollector):
    """Download model and tokenizer."""
    print("\n" + "=" * 60)
    print("2️⃣ MODEL DOWNLOAD")
    print("=" * 60)

    try:
        model_cfg = config["model"]
        print(f"\n🤖 Downloading model: {model_cfg['model_name']}")

        model, tokenizer = download_model(
            model_name=model_cfg["model_name"],
            cache_dir=Path(model_cfg["cache_dir"]),
            device_map=model_cfg.get("device", "auto"),
        )

        metrics.record_stage(
            "model_download",
            "success",
            {
                "model_name": model_cfg["model_name"],
                "cache_dir": model_cfg["cache_dir"],
                "device": model_cfg.get("device", "auto"),
            },
        )

        print(f"\n✅ Model and tokenizer downloaded successfully")
        return True

    except Exception as e:
        print(f"\n❌ Model download failed: {e}")
        traceback.print_exc()
        metrics.record_stage("model_download", "failed", {"error": str(e)})
        return False


def training_stage(config: dict, metrics: MetricsCollector):
    """Run fine-tuning."""
    print("\n" + "=" * 60)
    print("3️⃣ FINE-TUNING WITH LoRA")
    print("=" * 60)

    try:
        train_cfg = config["training"]
        model_cfg = config["model"]

        print(f"\n🔥 Starting fine-tuning...")
        print(f"   Model: {model_cfg['model_name']}")
        print(f"   Data dir: {train_cfg['data_dir']}")
        print(f"   Batch size: {train_cfg['batch_size']}")
        print(f"   Epochs: {train_cfg['max_epochs']}")
        print(f"   LoRA rank: {train_cfg['lora_rank']}")

        model, trainer = train(
            model_name=model_cfg["model_name"],
            data_dir=train_cfg["data_dir"],
            output_dir=train_cfg["output_dir"],
            batch_size=train_cfg["batch_size"],
            max_epochs=train_cfg["max_epochs"],
            lora_rank=train_cfg["lora_rank"],
            learning_rate=train_cfg["learning_rate"],
            accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 1),
            precision=train_cfg.get("precision", "16-mixed"),
            num_workers=train_cfg.get("num_workers", 0),
            cache_dir=model_cfg["cache_dir"],
            enable_early_stopping=train_cfg.get("enable_early_stopping", True),
        )

        # Extract metrics from trainer
        training_metrics = {
            "model_name": model_cfg["model_name"],
            "batch_size": train_cfg["batch_size"],
            "epochs": train_cfg["max_epochs"],
            "lora_rank": train_cfg["lora_rank"],
            "learning_rate": train_cfg["learning_rate"],
            "output_dir": train_cfg["output_dir"],
        }

        # Try to extract metrics from trainer if available
        if hasattr(trainer, "logged_metrics") and trainer.logged_metrics:
            training_metrics["final_metrics"] = trainer.logged_metrics

        metrics.record_stage("training", "success", training_metrics)

        print(f"\n✅ Fine-tuning completed successfully")
        return True

    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        traceback.print_exc()
        metrics.record_stage("training", "failed", {"error": str(e)})
        return False


def main(config_path: str | Path = "config.yaml"):
    """Run the complete fine-tuning pipeline."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  🚀 SLM LoRA FINE-TUNING PIPELINE".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")

    # Load configuration
    config = load_config(config_path)
    print("\n" + "─" * 60)
    print("Configuration Summary:")
    print("─" * 60)
    print(f"Dataset: {config['dataset']['dataset_name']} ({config['dataset']['max_samples']} samples)")
    print(f"Model: {config['model']['model_name']}")
    print(f"Training epochs: {config['training']['max_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"LoRA rank: {config['training']['lora_rank']}")
    print("─" * 60)

    # Initialize metrics collector
    metrics = MetricsCollector(config["logging"]["metrics_dir"])

    # Track overall success
    all_success = True

    # 1. Download Dataset
    if config["pipeline"]["download_dataset"]:
        if not download_dataset_stage(config, metrics):
            all_success = False
    else:
        print("\n⏭️  Skipping dataset download (disabled in config)")
        metrics.record_stage("dataset_download", "skipped")

    # 2. Download Model
    if config["pipeline"]["download_model"]:
        if not download_model_stage(config, metrics):
            all_success = False
    else:
        print("\n⏭️  Skipping model download (disabled in config)")
        metrics.record_stage("model_download", "skipped")

    # 3. Fine-tune
    if config["pipeline"]["train"]:
        if not training_stage(config, metrics):
            all_success = False
    else:
        print("\n⏭️  Skipping training (disabled in config)")
        metrics.record_stage("training", "skipped")

    # Save metrics
    metrics.save(config["logging"]["metrics_file"])

    # Summary
    print("\n" + "=" * 60)
    print("📋 PIPELINE SUMMARY")
    print("=" * 60)

    for stage_name, stage_info in metrics.metrics["pipeline_stages"].items():
        status = stage_info["status"]
        status_symbol = {
            "success": "✅",
            "failed": "❌",
            "skipped": "⏭️ ",
        }.get(status, "❓")
        print(f"{status_symbol} {stage_name.upper()}: {status}")

    print("=" * 60)

    if all_success:
        print("\n✅ Pipeline completed successfully!")
        print(f"📁 Checkpoints: {config['training']['output_dir']}")
        print(f"📊 Metrics: {metrics.metrics_dir / config['logging']['metrics_file']}")
        return 0
    else:
        print("\n❌ Pipeline failed. Check logs above for details.")
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SLM LoRA Fine-tuning Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )

    args = parser.parse_args()
    exit_code = main(args.config)
    sys.exit(exit_code)
