from torch.utils.data import Dataset, DataLoader
import pandas as pd  # Data manipulation and analysis
import torch  # PyTorch for tensor operations and deep learning
from transformers import AutoTokenizer  # Hugging Face Transformers for BERT tokenizer
import os  # Operating system interface for file handling
import cv2  # OpenCV for video processing
import numpy as np  # Numerical operations
import subprocess  # For running shell commands
import torchaudio  # PyTorch for audio processing


# Disable tokenizer parallelism warning ie "Tokenizers parallelism is enabled, which may cause issues in multi-threaded environments."
# This warning is common when using Hugging Face Transformers in a multi-threaded context.
# Setting this to "false" can help avoid potential issues with parallel processing.
# This is particularly useful when running in environments where multiple threads are used, such as in some web servers or multi-GPU setups.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Custom PyTorch dataset class for the MELD dataset
class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        """
        Initialize dataset by loading CSV, setting up tokenizer, and mapping labels.
        Args:
            csv_path: Path to CSV file containing utterance-level labels and metadata.
            video_dir: Path to the directory containing corresponding video files.
        """
        # Load CSV file
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir

        # Sanity checks for file and directory
        if not os.path.exists(video_dir):
            raise ValueError(f"Video directory not found: {video_dir}")
        if not os.path.exists(csv_path):
            raise ValueError(f"CSV file not found: {csv_path}")
        if self.data.empty:
            raise ValueError(f"CSV file is empty: {csv_path}")

        # Load BERT tokenizer Its tokenizer breaks text into tokens (words or subwords) and converts them into token IDs (numbers).

        # These tokens preserve semantic meaning — BERT understands things like grammar, word context, etc.
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Emotion and sentiment label mappings Emotion → 7-class classification

        self.emotion_map = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6,
        }

        # Sentiment → 3-class classification

        self.sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}

    def _load_video_frames(self, video_path):
        """
        Extracts up to 30 frames (one per ~few se   conds) and preprocess 30 video frames from a given video file.
        Resizes frames to 224x224, normalizes pixel values, and pads (Pads with black frames if fewer than 30 frames) if needed.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")

            # Validate video by reading the first frame
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found: {video_path}")

            # Reset back to first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Read up to 30 frames
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize and normalize frame
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0  #
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video error: {str(e)}")
        finally:
            cap.release()

        # If no frames extracted, raise error
        if len(frames) == 0:
            raise ValueError("No frames could be extracted")

        # Pad with black frames or truncate to exactly 30
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        #  Converts to PyTorch tensor: Shape [30, 3, 224, 224] (frames, channels, height, width).

        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
