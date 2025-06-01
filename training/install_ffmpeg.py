# install_ffmpeg.py
#
# Utility function to ensure FFmpeg is installed and available in the environment.
#
# - Installs ffmpeg-python (Python bindings for FFmpeg) using pip.
# - Downloads and installs a static FFmpeg binary if not already present.
# - Verifies the installation by running `ffmpeg -version`.
# - Returns True if FFmpeg is available, otherwise False.
#
# This script is especially useful in cloud environments (e.g., AWS SageMaker) where FFmpeg may not be pre-installed,
# and is required for audio/video processing tasks.

import subprocess
import sys


def install_ffmpeg():
    print("Starting Ffmpeg installation...")

    # Upgrade pip to the latest version for reliable package installation
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Upgrade setuptools to avoid compatibility issues with some packages
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"]
    )

    # Attempt to install ffmpeg-python (Python bindings for FFmpeg)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
        print("Installed ffmpeg-python successfully")
    except subprocess.CalledProcessError as e:
        print("Failed to install ffmpeg-python via pip")

    # Download and install the static FFmpeg binary
    try:
        # Download the latest static FFmpeg binary (for AMD64 architecture)
        subprocess.check_call(
            [
                "wget",
                "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
                "-O",
                "/tmp/ffmpeg.tar.xz",
            ]
        )

        # Extract the downloaded archive to /tmp/
        subprocess.check_call(["tar", "-xf", "/tmp/ffmpeg.tar.xz", "-C", "/tmp/"])

        # Find the ffmpeg executable within the extracted files
        result = subprocess.run(
            ["find", "/tmp", "-name", "ffmpeg", "-type", "f"],
            capture_output=True,
            text=True,
        )
        ffmpeg_path = result.stdout.strip()

        # Copy the ffmpeg binary to /usr/local/bin for system-wide access
        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])

        # Make the ffmpeg binary executable
        subprocess.check_call(["chmod", "+x", "/usr/local/bin/ffmpeg"])

        print("🟢 Installed static FFmpeg binary successfully")
    except Exception as e:
        print(f"Failed to install static FFmpeg: {e}")

    # Verify that FFmpeg is installed and accessible
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True
        )
        print("FFmpeg version:")
        print(result.stdout)
        return True  # Installation successful
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg installation verification failed")
        return False  # Installation failed
