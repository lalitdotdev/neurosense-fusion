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
