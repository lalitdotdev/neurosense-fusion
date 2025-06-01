import torch
from models import MultimodalSentimentModel
import os
import cv2
import numpy as np
import subprocess
import torchaudio
import whisper
from transformers import AutoTokenizer
import sys
import json
import boto3
import tempfile

EMOTION_MAP = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "neutral",
    5: "sadness",
    6: "surprise",
}
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def install_ffmpeg():
    print("Starting Ffmpeg installation...")

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"]
    )

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
        print("Installed ffmpeg-python successfully")
    except subprocess.CalledProcessError as e:
        print("Failed to install ffmpeg-python via pip")

    try:
        subprocess.check_call(
            [
                "wget",
                "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
                "-O",
                "/tmp/ffmpeg.tar.xz",
            ]
        )

        subprocess.check_call(["tar", "-xf", "/tmp/ffmpeg.tar.xz", "-C", "/tmp/"])

        result = subprocess.run(
            ["find", "/tmp", "-name", "ffmpeg", "-type", "f"],
            capture_output=True,
            text=True,
        )
        ffmpeg_path = result.stdout.strip()

        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])

        subprocess.check_call(["chmod", "+x", "/usr/local/bin/ffmpeg"])

        print("Installed static FFmpeg binary successfully")
    except Exception as e:
        print(f"Failed to install static FFmpeg: {e}")

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True
        )
        print("FFmpeg version:")
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg installation verification failed")
        return False


class VideoProcessor:
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")

            # Try and read first frame to validate video
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found: {video_path}")

            # Reset index to not skip first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video error: {str(e)}")
        finally:
            cap.release()

        if len(frames) == 0:
            raise ValueError("No frames could be extracted")

        # Pad or truncate frames
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        # Before permute: [frames, height, width, channels]
        # After permute: [frames, channels, height, width]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
