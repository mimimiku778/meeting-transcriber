"""Video frame extraction module."""

import base64
from pathlib import Path

import cv2


def extract_frame(
    video_path: str,
    timestamp_seconds: float,
    max_width: int = 1280,
    jpeg_quality: int = 80,
) -> str:
    """
    Extract a frame from video at specified timestamp.

    Args:
        video_path: Path to video file
        timestamp_seconds: Time in seconds to extract frame
        max_width: Maximum width for resizing (default: 1280)
        jpeg_quality: JPEG compression quality 0-100 (default: 80)

    Returns:
        Base64 encoded JPEG image
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

        # Resize if larger than max_width
        height, width = frame.shape[:2]
        if width > max_width:
            scale = max_width / width
            new_width = max_width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Encode as JPEG with compression
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
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
