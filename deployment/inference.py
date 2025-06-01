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


class AudioProcessor:
    def extract_features(self, video_path, max_length=300):
        audio_path = video_path.replace(".mp4", ".wav")

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    video_path,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    audio_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512
            )

            mel_spec = mel_spectrogram(waveform)

            # Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Audio error: {str(e)}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)


class VideoUtteranceProcessor:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()

    def extract_segment(self, video_path, start_time, end_time, temp_dir="/tmp"):
        os.makedirs(temp_dir, exist_ok=True)
        segment_path = os.path.join(temp_dir, f"segment_{start_time}_{end_time}.mp4")

        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-ss",
                str(start_time),
                "-to",
                str(end_time),
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-y",
                segment_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not os.path.exists(segment_path) or os.path.getsize(segment_path) == 0:
            raise ValueError("Segment extraction failed: " + segment_path)

        return segment_path


def download_from_s3(s3_uri):
    s3_client = boto3.client("s3")
    bucket = s3_uri.split("/")[2]
    key = "/".join(s3_uri.split("/")[3:])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        s3_client.download_file(bucket, key, temp_file.name)
        return temp_file.name


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        s3_uri = input_data["video_path"]
        local_path = download_from_s3(s3_uri)
        return {"video_path": local_path}
    raise ValueError(f"Unsupported content type: {request_content_type}")


def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {response_content_type}")


def model_fn(model_dir):
    # Load the model for inference
    if not install_ffmpeg():
        raise RuntimeError("FFmpeg installation failed - required for inference")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalSentimentModel().to(device)

    model_path = os.path.join(model_dir, "model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "model", "model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found in path " + model_path)

    print("Loading model from path: " + model_path)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    return {
        "model": model,
        "tokenizer": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "transcriber": whisper.load_model(
            "base",
            device="cpu" if device.type == "cpu" else device,
        ),
        "device": device,
    }


def predict_fn(input_data, model_dict):
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model_dict["device"]
    video_path = input_data["video_path"]

    result = model_dict["transcriber"].transcribe(video_path, word_timestamps=True)

    utterance_processor = VideoUtteranceProcessor()
    predictions = []

    for segment in result["segments"]:
        try:
            segment_path = utterance_processor.extract_segment(
                video_path, segment["start"], segment["end"]
            )

            video_frames = utterance_processor.video_processor.process_video(
                segment_path
            )
            audio_features = utterance_processor.audio_processor.extract_features(
                segment_path
            )
            text_inputs = tokenizer(
                segment["text"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            # Move to device
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            video_frames = video_frames.unsqueeze(0).to(device)
            audio_features = audio_features.unsqueeze(0).to(device)

            # Get predictions
            with torch.inference_mode():
                outputs = model(text_inputs, video_frames, audio_features)
                emotion_probs = torch.softmax(outputs["emotions"], dim=1)[0]
                sentiment_probs = torch.softmax(outputs["sentiments"], dim=1)[0]

                emotion_values, emotion_indices = torch.topk(emotion_probs, 3)
                sentiment_values, sentiment_indices = torch.topk(sentiment_probs, 3)

            predictions.append(
                {
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "text": segment["text"],
                    "emotions": [
                        {"label": EMOTION_MAP[idx.item()], "confidence": conf.item()}
                        for idx, conf in zip(emotion_indices, emotion_values)
                    ],
                    "sentiments": [
                        {"label": SENTIMENT_MAP[idx.item()], "confidence": conf.item()}
                        for idx, conf in zip(sentiment_indices, sentiment_values)
                    ],
                }
            )

        except Exception as e:
            print("Segment failed inference: " + str(e))

        finally:
            # Cleanup
            if os.path.exists(segment_path):
                os.remove(segment_path)
    return {"utterances": predictions}


# def process_local_video(video_path, model_dir="model_normalized"):
#     model_dict = model_fn(model_dir)

#     input_data = {'video_path': video_path}

#     predictions = predict_fn(input_data, model_dict)

#     for utterance in predictions["utterances"]:
#         print("\nUtterance:")
#         print(f"""Start: {utterance['start_time']}s, End: {
#               utterance['end_time']}s""")
#         print(f"Text: {utterance['text']}")
#         print("\n Top Emotions:")
#         for emotion in utterance['emotions']:
#             print(f"{emotion['label']}: {emotion['confidence']:.2f}")
#         print("\n Top Sentiments:")
#         for sentiment in utterance['sentiments']:
#             print(f"{sentiment['label']}: {sentiment['confidence']:.2f}")
#         print("-"*50)


# if __name__ == "__main__":
#     process_local_video("./dia2_utt3.mp4")
