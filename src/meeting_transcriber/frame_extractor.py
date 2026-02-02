"""Video frame extraction module."""

import base64
from pathlib import Path

import cv2


def extract_frame(video_path: str, timestamp_seconds: float) -> str:
    """
    Extract a frame from video at specified timestamp.

    Args:
        video_path: Path to video file
        timestamp_seconds: Time in seconds to extract frame

    Returns:
        Base64 encoded PNG image
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    try:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if timestamp_seconds < 0 or timestamp_seconds > duration:
            raise ValueError(f"Timestamp {timestamp_seconds}s is out of range (0-{duration:.1f}s)")

        # Seek to frame
        frame_number = int(timestamp_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame at {timestamp_seconds}s")

        # Encode as PNG
        _, buffer = cv2.imencode('.png', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        return base64_image

    finally:
        cap.release()


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return total_frames / fps if fps > 0 else 0
    finally:
        cap.release()
