import os
import argparse
import torchaudio
import torch
from tqdm import tqdm
import json
import sys
from sklearn.metrics import classification_report
from meld_dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer
from install_ffmpeg import install_ffmpeg
import matplotlib

import seaborn as sns

matplotlib.use("Agg")  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt

# Multimodal Sentiment Training Script for AWS SageMaker
#
# - This script is the entry point for training a multimodal sentiment analysis model using PyTorch on AWS SageMaker.
# - It supports training, validation, and test evaluation for a model that fuses text, video, and audio modalities.
# - The script is compatible with SageMaker environment variables for data and model directories, allowing seamless integration with SageMaker’s managed infrastructure.
# - Hyperparameters and data paths can be set via command-line arguments or SageMaker environment variables.
# - The script handles FFmpeg installation for video/audio processing, prepares data loaders, tracks GPU memory, logs metrics in SageMaker-compatible JSON format, and saves the best model checkpoint.
# - After training, it evaluates the model on the test set and prints final metrics.
#
# Usage:
#   - As a SageMaker entry point script (automatically called by SageMaker training jobs).
#   - As a standalone script for local debugging and development.
#
# Key SageMaker integration points:
#   - Uses `SM_MODEL_DIR`, `SM_CHANNEL_TRAINING`, `SM_CHANNEL_VALIDATION`, and `SM_CHANNEL_TEST` for model and data paths.
#   - Logs metrics in JSON for SageMaker to track progress and results.
#   - Writes model artifacts to the directory expected by SageMaker for deployment.


# AWS SageMaker environment variables for model and data directories

SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", ".")
SM_CHANNEL_TRAINING = os.environ.get(
    "SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"
)
SM_CHANNEL_VALIDATION = os.environ.get(
    "SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"
)
SM_CHANNEL_TEST = os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test")

# Configure PyTorch CUDA memory for better GPU usage (especially on SageMaker)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Parse command-line arguments and set defaults using SageMaker environment variables.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)

    # Data directories (can be overridden by SageMaker environment variables)

    parser.add_argument("--train-dir", type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument("--val-dir", type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument("--test-dir", type=str, default=SM_CHANNEL_TEST)
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)

    return parser.parse_args()


def main():
    # Ensure FFmpeg is installed for audio/video processing
    if not install_ffmpeg():
        print("Error: FFmpeg installation failed. Cannot continue training.")
        sys.exit(1)

    # Print available torchaudio backends for debugging
    print("Available audio backends:")
    print(str(torchaudio.list_audio_backends()))

    args = parse_args()  # Parse command-line or SageMaker arguments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print initial GPU memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Initial GPU memory used: {memory_used:.2f} GB")

    # Prepare DataLoaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, "train_sent_emo.csv"),
        train_video_dir=os.path.join(args.train_dir, "train_splits"),
        dev_csv=os.path.join(args.val_dir, "dev_sent_emo.csv"),
        dev_video_dir=os.path.join(args.val_dir, "dev_splits_complete"),
        test_csv=os.path.join(args.test_dir, "test_sent_emo.csv"),
        test_video_dir=os.path.join(args.test_dir, "output_repeated_splits_test"),
        batch_size=args.batch_size,
    )

    print(f"Training CSV path: {os.path.join(args.train_dir, 'train_sent_emo.csv')}")
    print(f"Training video directory: {os.path.join(args.train_dir, 'train_splits')}")

    # Initialize model and trainer
    model = MultimodalSentimentModel().to(device)
    trainer = MultimodalTrainer(model, train_loader, val_loader)
    best_val_loss = float("inf")

    # Initialize metrics storage
    metrics_data = {
        "train_losses": [],
        "val_losses": [],
        "epochs": [],
        "test_metrics": {},
    }

    # Training loop
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        # Training phase
        train_loss = trainer.train_epoch()

        # Validation phase
        val_loss, val_metrics, emotion_metrics, sentiment_metrics = trainer.evaluate(
            val_loader
        )

        # Store metrics
        metrics_data["train_losses"].append(train_loss["total"])
        metrics_data["val_losses"].append(val_loss["total"])
        metrics_data["epochs"].append(epoch)

        # SageMaker-compatible metric logging
        metric_payload = {
            "metrics": [
                {"Name": "train:loss", "Value": train_loss["total"]},
                {"Name": "validation:loss", "Value": val_loss["total"]},
                {
                    "Name": "validation:emotion_accuracy",
                    "Value": val_metrics["emotion_accuracy"],
                },
                {
                    "Name": "validation:emotion_precision",
                    "Value": val_metrics["emotion_precision"],
                },
                {
                    "Name": "validation:emotion_recall",
                    "Value": val_metrics["emotion_recall"],
                },
                {"Name": "validation:emotion_f1", "Value": val_metrics["emotion_f1"]},
                {
                    "Name": "validation:emotion_roc_auc",
                    "Value": val_metrics.get("emotion_roc_auc", 0),
                },
                {
                    "Name": "validation:sentiment_accuracy",
                    "Value": val_metrics["sentiment_accuracy"],
                },
                {
                    "Name": "validation:sentiment_precision",
                    "Value": val_metrics["sentiment_precision"],
                },
                {
                    "Name": "validation:sentiment_recall",
                    "Value": val_metrics["sentiment_recall"],
                },
                {
                    "Name": "validation:sentiment_f1",
                    "Value": val_metrics["sentiment_f1"],
                },
                {
                    "Name": "validation:sentiment_roc_auc",
                    "Value": val_metrics.get("sentiment_roc_auc", 0),
                },
            ]
        }
        print(json.dumps(metric_payload))

        # GPU memory monitoring
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory used: {memory_used:.2f} GB")

        # Save best model
        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

        # Detailed reports every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("\n=== Emotion Classification Report ===")
            print(
                classification_report(
                    emotion_metrics["y_true"],
                    emotion_metrics["y_pred"],
                    target_names=[
                        "anger",
                        "disgust",
                        "fear",
                        "joy",
                        "neutral",
                        "sadness",
                        "surprise",
                    ],
                    digits=4,
                )
            )

            print("\n=== Sentiment Classification Report ===")
            print(
                classification_report(
                    sentiment_metrics["y_true"],
                    sentiment_metrics["y_pred"],
                    target_names=["negative", "neutral", "positive"],
                    digits=4,
                )
            )

    # Final test evaluation
    print("\nEvaluating on test set...")
    test_loss, test_metrics, test_emotion, test_sentiment = trainer.evaluate(
        test_loader, phase="test"
    )
    metrics_data["test_metrics"] = {
        "emotion": test_emotion,
        "sentiment": test_sentiment,
        "loss": test_loss["total"],
    }

    # Save final artifacts
    trainer.save_metrics_report(path=args.model_dir)
    trainer.visualize_performance().savefig(
        os.path.join(args.model_dir, "training_history.png")
    )

    # Log test metrics
    print(
        json.dumps(
            {
                "metrics": [
                    {"Name": "test:loss", "Value": test_loss["total"]},
                    {
                        "Name": "test:emotion_accuracy",
                        "Value": test_metrics["emotion_accuracy"],
                    },
                    {"Name": "test:emotion_f1", "Value": test_metrics["emotion_f1"]},
                    {
                        "Name": "test:emotion_roc_auc",
                        "Value": test_metrics.get("emotion_roc_auc", 0),
                    },
                    {
                        "Name": "test:sentiment_accuracy",
                        "Value": test_metrics["sentiment_accuracy"],
                    },
                    {
                        "Name": "test:sentiment_f1",
                        "Value": test_metrics["sentiment_f1"],
                    },
                    {
                        "Name": "test:sentiment_roc_auc",
                        "Value": test_metrics.get("sentiment_roc_auc", 0),
                    },
                ]
            }
        )
    )

    # Save confusion matrices
    for task, metrics in [("emotion", test_emotion), ("sentiment", test_sentiment)]:
        plt.figure(figsize=(10, 8))
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues")
        plt.title(f"Test {task.capitalize()} Confusion Matrix")
        plt.savefig(os.path.join(args.model_dir, f"{task}_confusion_matrix.png"))
        plt.close()

    print(f"\nAll artifacts saved to {args.model_dir}")


if __name__ == "__main__":
    main()
