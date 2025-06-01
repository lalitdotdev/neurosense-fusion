import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
)

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

from meld_dataset import MELDDataset


# TextEncoder: Encodes input text into fixed-size embeddings using a frozen BERT model.
#
# - Loads a pre-trained BERT model (bert-base-uncased) as a feature extractor.
#   # Reason: Pre-trained BERT has learned rich semantic and syntactic representations from large text corpora, enabling it to generate high-quality embeddings for a wide range of NLP tasks[1][4][7].
#
# - Freezes all BERT parameters to prevent updates during training (no fine-tuning).
#   # Reason: Freezing BERT reduces computational cost and memory usage, and leverages BERT's general language understanding without risk of overfitting or catastrophic forgetting on small datasets[2][3][5].
#
# - Projects the 768-dimensional [CLS] token embedding from BERT down to 128 dimensions.
#   # Reason: The [CLS] token output is a fixed-size, aggregate representation of the input sequence, suitable for classification or retrieval tasks. Reducing its dimensionality with a projection layer makes embeddings more compact and efficient for downstream models[4][6][8].
#
# - Intended for use in downstream tasks where compact, meaningful text representations are needed.
#   # Reason: Compact embeddings facilitate efficient storage, retrieval, and computation for applications such as similarity search, classification, or multi-modal learning[7].


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")  # Load BERT model

        # Freeze BERT parameters to prevent training
        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # Extract BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooler_output = outputs.pooler_output

        return self.projection(pooler_output)


# VideoEncoder: Encodes input video clips into fixed-size embeddings using a frozen 3D ResNet-18 model.
#
# Loads a pre-trained 3D ResNet-18 (r3d_18) as a feature extractor.
#   # Reason: Pre-trained 3D ResNet-18 has learned to capture both spatial and temporal features from large-scale video datasets, making it effective for extracting meaningful representations from video sequences[5].
#
# - Freezes all backbone parameters to prevent updates during training (no fine-tuning).
#   # Reason: Freezing the backbone reduces computational cost and memory usage, and leverages the general spatiotemporal knowledge captured during pre-training, which is especially useful when data or compute is limited.
#
# - Replaces the final fully connected (fc) layer with a small head: Linear(…, 128) + ReLU + Dropout.
#   # Reason: The new head projects the high-dimensional video features to a compact 128-dimensional embedding, adds non-linearity for expressiveness, and uses dropout to help prevent overfitting in downstream tasks.
#
# - Intended for use in downstream tasks where compact, meaningful video representations are needed.
#   # Reason: Compact embeddings enable efficient storage, retrieval, and computation for applications such as video classification, retrieval, or multi-modal learning.


class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_fts = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128), nn.ReLU(), nn.Dropout(0.2)
        )

    def forward(self, x):
        # [batch_size, frames, channels, height, width]->[batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)
        return self.backbone(x)


# AudioEncoder: Encodes input audio into compact, fixed-size embeddings using a frozen convolutional feature extractor.
#
# - Uses a stack of 1D convolutional layers to extract hierarchical features from raw audio input.
#   # Reason: Convolutional layers are effective at capturing local and global temporal patterns in audio signals.
#
# - Freezes all convolutional layer parameters to prevent updates during training.
#   # Reason: Using the conv layers as a fixed feature extractor reduces overfitting and computational cost, especially when pre-trained or when data is limited.
#
# - Projects the high-dimensional output of the conv layers to a 128-dimensional embedding using a small projection head.
#   # Reason: The projection head adds non-linearity and regularization, and produces compact, robust embeddings suitable for downstream tasks.
#
# - Intended for use in downstream tasks such as audio classification, retrieval, or multimodal learning.
#   # Reason: Compact embeddings enable efficient storage, computation, and integration with other modalities.


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extractor: stack of 1D conv layers with normalization, activation, and pooling
        self.conv_layers = nn.Sequential(
            # Lower-level feature extraction ----------------------------------->
            # Capture local temporal patterns .A 1D convolutional layer is designed to process sequential or temporal data, such as audio waveforms or time series. It applies a set of learnable filters (in this case, 64 filters) across the input sequence to extract features
            # The kernel (or filter) has a width of 3, meaning it looks at three consecutive time steps at a time as it slides across the sequence. This enables the network to learn and detect local patterns or features that span three time steps, such as short motifs, transitions, or local changes in the signal
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),  # Normalize for stable training
            nn.ReLU(),  # Non-linear activation
            nn.MaxPool1d(2),  # Downsample and focus on salient features
            # Higher level features
            nn.Conv1d(64, 128, kernel_size=3),  # Capture more abstract patterns
            nn.BatchNorm1d(128),  # Normalize
            nn.ReLU(),  # Non-linearity
            nn.AdaptiveAvgPool1d(1),  # Aggregate features to fixed length
        )

        # Freeze all convolutional layer parameters to use as a fixed feature extractor
        for param in self.conv_layers.parameters():
            param.requires_grad = False

        # Projection head: reduces dimensionality and adds regularization

        # Project to 128-dim embedding , # Non-linearity , # Prevent overfitting
        self.projection = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2))

    def forward(self, x):
        x = x.squeeze(1)  # Remove channel dimension if present: [B, 1, T] -> [B, T]

        features = self.conv_layers(x)  # Extract features: [B, 128, 1]
        # Features output: [batch_size, 128, 1]

        return self.projection(
            features.squeeze(
                -1
            )  # Project to final embedding: [B, 128]  ,Remove last dimension: [B, 128, 1] -> [B, 128]
        )


# MultimodalSentimentModel: Deep neural network for multimodal sentiment and emotion analysis from text, video, and audio.
#
# - Integrates three specialized unimodal encoders (TextEncoder, VideoEncoder, AudioEncoder) to extract meaningful features from each modality.
#   # Reason: Each modality (text, video, audio) captures unique and complementary aspects of human sentiment and emotion, and unimodal encoders are effective at extracting modality-specific features[2][3][5].
#
# - Fuses the three modality embeddings by concatenation and passing through a fusion layer.
#   # Reason: Fusion combines information from all modalities, enabling the model to leverage complementary cues and correlations for more accurate sentiment/emotion prediction.
#
# - Uses a shared fused representation for two classification heads: emotion and sentiment.
#   # Reason: Multi-task learning allows the model to jointly predict both emotion categories and sentiment polarity, improving generalization and robustness by sharing representations[2].
#
# - Designed for robust multimodal sentiment analysis, where each modality may provide partial, noisy, or even conflicting information.
#   # Reason: Multimodal approaches are more resilient to missing or ambiguous signals in any single modality and can achieve higher accuracy than unimodal models.
#
# - Example application: Human-computer interaction, social media monitoring, affective computing, and customer feedback analysis.


class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Unimodal feature extractors for text, video, and audio

        self.text_encoder = TextEncoder()  # Extracts 128-dim text features
        self.video_encoder = VideoEncoder()  # Extracts 128-dim video features
        self.audio_encoder = AudioEncoder()  # Extracts 128-dim audio features

        # Fusion layer: combines all modality features into a shared representation

        # Concatenate and project to 256-dim -> # Normalize for stability -->  # Non-linearity ---> # Prevent overfitting
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )

        # Emotion classification head (multi-class, e.g., 7 emotions)

        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7),  # Output: 7 emotion categories
        )

        # Sentiment classification head (multi-class, e.g., negative/neutral/positive)
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),  # Output: 3 sentiment classes (negative/neutral/positive)
        )

    def forward(self, text_inputs, video_frames, audio_features):
        # Extract features from each modality

        text_features = self.text_encoder(
            text_inputs["input_ids"],
            text_inputs["attention_mask"],
        )
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # Concatenate multimodal features along the feature dimension

        combined_features = torch.cat(
            [text_features, video_features, audio_features], dim=1
        )  # [batch_size, 128 * 3]

        # Fuse features from all modalities

        fused_features = self.fusion_layer(combined_features)

        # Predict emotion and sentiment using shared fused representation

        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)
        # Return both predictions as a dictionary
        return {"emotions": emotion_output, "sentiments": sentiment_output}


def compute_class_weights(dataset):
    emotion_counts = torch.zeros(7)
    sentiment_counts = torch.zeros(3)
    skipped = 0
    total = len(dataset)
    print("\Counting class distributions...")

    for i in range(total):
        sample = dataset[i]

        if sample is None:
            skipped += 1
            continue
        emotion_label = sample["emotion_label"]
        sentiment_label = sample["sentiment_label"]

        emotion_counts[emotion_label] += 1
        sentiment_counts[sentiment_label] += 1

    valid = total - skipped
    print(f"Skipped samples: {skipped}/{total}")

    print("\nClass distribution")
    print("Emotions:")
    emotion_map = {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "joy",
        4: "neutral",
        5: "sadness",
        6: "surprise",
    }
    for i, count in enumerate(emotion_counts):
        print(f"{emotion_map[i]}: {count/valid:.2f}")

    print("\nSentiments:")
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    for i, count in enumerate(sentiment_counts):
        print(f"{sentiment_map[i]}: {count/valid:.2f}")

    # Calculate class weights
    emotion_weights = 1.0 / emotion_counts
    sentiment_weights = 1.0 / sentiment_counts

    # Normalize weights
    emotion_weights = emotion_weights / emotion_weights.sum()
    sentiment_weights = sentiment_weights / sentiment_weights.sum()

    return emotion_weights, sentiment_weights


# MultimodalTrainer: Utility class for training and validating a multimodal sentiment model.
#
# - Handles model training, validation, logging, and optimization for a model that predicts both emotions and sentiments from text, video, and audio.
# - Manages optimizer and learning rate scheduling for different model components, allowing fine-grained control over learning rates.
# - Computes and applies class weights to handle class imbalance in both emotion and sentiment tasks.
# - Logs metrics and losses to TensorBoard for easy experiment tracking and visualization.
# - Provides methods for running one epoch of training (`train_epoch`) and for evaluating on a validation/test set (`evaluate`).
# - Supports multi-task learning by computing and logging losses and metrics for both emotion and sentiment outputs.


class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Initialize metrics containers
        self.metrics_history = {
            "train": {"loss": [], "emotion_f1": [], "sentiment_f1": []},
            "val": {"loss": [], "emotion_f1": [], "sentiment_f1": []},
        }
        # Log dataset sizes for transparency
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        print("\nDataset sizes:")
        print(f"Training samples: {train_size:,}")
        print(f"Validation samples: {val_size:,}")
        print(f"Batches per epoch: {len(train_loader):,}")

        # Set up TensorBoard logging directory with timestamp for experiment tracking
        timestamp = datetime.now().strftime("%b%d_%H-%M-%S")  # Dec17_14-22-35
        base_dir = (
            "/opt/ml/output/tensorboard" if "SM_MODEL_DIR" in os.environ else "runs"
        )
        log_dir = f"{base_dir}/run_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

        # Set up optimizer with different learning rates for each submodule
        # Lower learning rates for pre-trained encoders, higher for new layers
        # Very high: 1, high: 0.1-0.01, medium: 1e-1, low: 1e-4, very low: 1e-5

        # Optimizers are algorithms or methods used to update the parameters (weights and biases) of neural networks in order to minimize the loss function during training. Their main goals are:

        # Minimize Loss: Find the set of parameters that makes the model’s predictions as close as possible to the true values.

        # Efficient Learning: Adjust parameters efficiently and effectively, even for very large and complex models.

        # Handle Complex Landscapes: Navigate the complicated, high-dimensional surface of the loss function, which may have many local minima, saddle points, and flat regions.

        # Adam (Adaptive Moment Estimation) is a popular optimizer because:
        # It adapts the learning rate for each parameter individually.
        # It combines the advantages of two other methods: AdaGrad (good for sparse gradients) and RMSProp (good for non-stationary objectives).
        # It’s robust and works well for most deep learning tasks.

        self.optimizer = torch.optim.Adam(
            [
                {"params": model.text_encoder.parameters(), "lr": 8e-6},
                {"params": model.video_encoder.parameters(), "lr": 8e-5},
                {"params": model.audio_encoder.parameters(), "lr": 8e-5},
                {"params": model.fusion_layer.parameters(), "lr": 5e-4},
                {"params": model.emotion_classifier.parameters(), "lr": 5e-4},
                {"params": model.sentiment_classifier.parameters(), "lr": 5e-4},
            ],
            weight_decay=1e-5,
        )
        # Learning rate scheduler: Reduce LR on plateau of validation loss

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=2
        )

        self.current_train_losses = None

        # Compute class weights to handle class imbalance in both tasks

        print("\nCalculating class weights...")
        emotion_weights, sentiment_weights = compute_class_weights(train_loader.dataset)

        device = next(model.parameters()).device  # Get model device (CPU/GPU)

        self.emotion_weights = emotion_weights.to(device)
        self.sentiment_weights = sentiment_weights.to(device)

        print(f"Emotion weights on device: {self.emotion_weights.device}")
        print(f"Sentiments weights on device: {self.sentiment_weights.device}")

        # Define loss functions with label smoothing and class weights for robustness
        self.emotion_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05, weight=self.emotion_weights
        )

        self.sentiment_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05, weight=self.sentiment_weights
        )

    def compute_metrics(self, y_true, y_pred, y_probs=None):
        """Enhanced metric calculation with error handling"""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1": f1_score(y_true, y_pred, average="weighted"),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }

        if y_probs is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_probs, multi_class="ovr", average="weighted"
                )
            except Exception as e:
                print(f"ROC AUC calculation skipped: {str(e)}")
                metrics["roc_auc"] = None

        metrics["classification_report"] = classification_report(
            y_true, y_pred, digits=4, output_dict=True
        )
        return metrics

    #! Log training and validation losses/metrics to TensorBoard.

    def log_metrics(self, losses, metrics=None, phase="train"):
        """Enhanced TensorBoard logging"""
        if phase == "train":
            self.current_train_losses = losses
        else:
            # Log comparison metrics
            for metric in ["total", "emotion", "sentiment"]:
                self.writer.add_scalars(
                    f"loss/{metric}",
                    {"train": self.current_train_losses[metric], "val": losses[metric]},
                    self.global_step,
                )

            # Log detailed metrics
            if metrics:
                for task in ["emotion", "sentiment"]:
                    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                        key = f"{task}_{metric}"
                        if key in metrics:
                            self.writer.add_scalar(
                                f"{phase}/{key}", metrics[key], self.global_step
                            )

    #  Run a single epoch of training over the training DataLoader.
    #  Returns average losses for the epoch.
    def train_epoch(self):
        self.model.train()
        running_loss = {"total": 0, "emotion": 0, "sentiment": 0}

        for batch in self.train_loader:
            # Move all batch data to the correct device
            device = next(self.model.parameters()).device
            text_inputs = {
                "input_ids": batch["text_inputs"]["input_ids"].to(device),
                "attention_mask": batch["text_inputs"]["attention_mask"].to(device),
            }
            video_frames = batch["video_frames"].to(device)
            audio_features = batch["audio_features"].to(device)
            emotion_labels = batch["emotion_label"].to(device)
            sentiment_labels = batch["sentiment_label"].to(device)

            # Zero gradients before backward pass
            self.optimizer.zero_grad()

            # Forward pass: get model predictions
            outputs = self.model(text_inputs, video_frames, audio_features)

            # Compute losses for both emotion and sentiment outputs

            emotion_loss = self.emotion_criterion(outputs["emotions"], emotion_labels)
            sentiment_loss = self.sentiment_criterion(
                outputs["sentiments"], sentiment_labels
            )
            total_loss = emotion_loss + sentiment_loss

            # Backward pass. Calculate gradients
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update model parameters
            self.optimizer.step()

            # Accumulate running losses
            running_loss["total"] += total_loss.item()
            running_loss["emotion"] += emotion_loss.item()
            running_loss["sentiment"] += sentiment_loss.item()

            self.log_metrics(
                {
                    "total": total_loss.item(),
                    "emotion": emotion_loss.item(),
                    "sentiment": sentiment_loss.item(),
                }
            )

            self.global_step += 1

        # Return average losses for the epoch
        return {k: v / len(self.train_loader) for k, v in running_loss.items()}

    # ^  ---------------------------> Evaluate the model on a validation/test set. ------------------------->

    # Evaluate the multimodal sentiment model on a validation or test dataset.
    #
    # - Switches the model to evaluation mode to disable dropout and batch norm updates.
    # - Iterates over the data loader, moving each batch to the correct device (CPU/GPU).
    # - For each batch, extracts text, video, and audio inputs, and their corresponding emotion and sentiment labels.
    # - Runs the model to get predictions for both emotion and sentiment.
    # - Calculates the loss for both tasks and accumulates total, emotion, and sentiment losses.
    # - Collects all predictions and true labels for both emotion and sentiment for metric computation.
    # - After all batches, computes the average loss for each task.
    # - Calculates key evaluation metrics: precision and accuracy for both emotion and sentiment predictions.
    # - Logs losses and metrics for monitoring and visualization.
    # - If in validation phase, updates the learning rate scheduler based on validation loss.
    # - Returns average losses and evaluation metrics for further analysis or reporting.

    def evaluate(self, data_loader, phase="val"):
        """Enhanced evaluation with full metrics"""
        self.model.eval()
        losses = {"total": 0, "emotion": 0, "sentiment": 0}
        emotion_preds, emotion_labels, emotion_probs = [], [], []
        sentiment_preds, sentiment_labels, sentiment_probs = [], [], []

        with torch.inference_mode():
            for batch in data_loader:
                # Batch processing (original code)
                text_inputs = {
                    "input_ids": batch["text_inputs"]["input_ids"].to(self.device),
                    "attention_mask": batch["text_inputs"]["attention_mask"].to(
                        self.device
                    ),
                }
                video_frames = batch["video_frames"].to(self.device)
                audio_features = batch["audio_features"].to(self.device)
                emotion_labels_batch = batch["emotion_label"].to(self.device)
                sentiment_labels_batch = batch["sentiment_label"].to(self.device)

                outputs = self.model(text_inputs, video_frames, audio_features)

                # Loss calculation
                emotion_loss = self.emotion_criterion(
                    outputs["emotions"], emotion_labels_batch
                )
                sentiment_loss = self.sentiment_criterion(
                    outputs["sentiments"], sentiment_labels_batch
                )
                total_loss = emotion_loss + sentiment_loss

                # Store predictions and probabilities
                emotion_probs.extend(
                    torch.softmax(outputs["emotions"], dim=1).cpu().numpy()
                )
                emotion_preds.extend(
                    torch.argmax(outputs["emotions"], dim=1).cpu().numpy()
                )
                emotion_labels.extend(emotion_labels_batch.cpu().numpy())

                sentiment_probs.extend(
                    torch.softmax(outputs["sentiments"], dim=1).cpu().numpy()
                )
                sentiment_preds.extend(
                    torch.argmax(outputs["sentiments"], dim=1).cpu().numpy()
                )
                sentiment_labels.extend(sentiment_labels_batch.cpu().numpy())

                # Update losses
                losses["total"] += total_loss.item()
                losses["emotion"] += emotion_loss.item()
                losses["sentiment"] += sentiment_loss.item()

        # Calculate metrics
        avg_loss = {k: v / len(data_loader) for k, v in losses.items()}
        emotion_metrics = self.compute_metrics(
            emotion_labels, emotion_preds, np.array(emotion_probs)
        )
        sentiment_metrics = self.compute_metrics(
            sentiment_labels, sentiment_preds, np.array(sentiment_probs)
        )

        # Format metrics for logging
        log_metrics = {
            "emotion_accuracy": emotion_metrics["accuracy"],
            "emotion_precision": emotion_metrics["precision"],
            "emotion_recall": emotion_metrics["recall"],
            "emotion_f1": emotion_metrics["f1"],
            "emotion_roc_auc": emotion_metrics.get("roc_auc", None),
            "sentiment_accuracy": sentiment_metrics["accuracy"],
            "sentiment_precision": sentiment_metrics["precision"],
            "sentiment_recall": sentiment_metrics["recall"],
            "sentiment_f1": sentiment_metrics["f1"],
            "sentiment_roc_auc": sentiment_metrics.get("roc_auc", None),
        }

        # Update metrics history
        self.metrics_history[phase]["loss"].append(avg_loss["total"])
        self.metrics_history[phase]["emotion_f1"].append(log_metrics["emotion_f1"])
        self.metrics_history[phase]["sentiment_f1"].append(log_metrics["sentiment_f1"])

        # Learning rate scheduling
        if phase == "val":
            self.scheduler.step(avg_loss["total"])
            self.log_metrics(avg_loss, log_metrics, phase="val")

        return avg_loss, log_metrics, emotion_metrics, sentiment_metrics

    def visualize_performance(self):
        """Generate training history plots"""
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))

        # Loss plot
        ax[0].plot(self.metrics_history["train"]["loss"], label="Train")
        ax[0].plot(self.metrics_history["val"]["loss"], label="Validation")
        ax[0].set_title("Loss Evolution")
        ax[0].legend()

        # Emotion F1 plot
        ax[1].plot(self.metrics_history["train"]["emotion_f1"], label="Train")
        ax[1].plot(self.metrics_history["val"]["emotion_f1"], label="Validation")
        ax[1].set_title("Emotion F1 Score")
        ax[1].legend()

        # Sentiment F1 plot
        ax[2].plot(self.metrics_history["train"]["sentiment_f1"], label="Train")
        ax[2].plot(self.metrics_history["val"]["sentiment_f1"], label="Validation")
        ax[2].set_title("Sentiment F1 Score")
        ax[2].legend()

        plt.tight_layout()
        return fig
