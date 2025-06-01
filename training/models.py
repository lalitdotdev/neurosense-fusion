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
