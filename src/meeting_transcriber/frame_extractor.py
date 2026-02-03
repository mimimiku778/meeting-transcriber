"""Video frame extraction module."""

from pathlib import Path

import cv2
import Vision
import Quartz


def ocr_image(image_path: str) -> list[str]:
    """
    Extract text from image using macOS Vision framework.

    Args:
        image_path: Path to image file

    Returns:
        List of recognized text strings
    """
    # Load image
    image_url = Quartz.CFURLCreateWithFileSystemPath(
        None, image_path, Quartz.kCFURLPOSIXPathStyle, False
    )
    image_source = Quartz.CGImageSourceCreateWithURL(image_url, None)
    if not image_source:
        return []

    cg_image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)
    if not cg_image:
        return []

    # Create text recognition request
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setRecognitionLanguages_(["ja", "en"])
    request.setUsesLanguageCorrection_(True)

    # Process image
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
    success = handler.performRequests_error_([request], None)

    if not success:
        return []

    # Extract results
    texts = []
    results = request.results()
    if results:
        for observation in results:
            text = observation.topCandidates_(1)[0].string()
            texts.append(text)

    return texts


def extract_frame(
    video_path: str,
    timestamp_seconds: float,
    output_path: str | None = None,
) -> tuple[str, list[str]]:
    """
    Extract a frame from video at specified timestamp and save as JPEG.
    Also performs OCR on the frame.

    Args:
        video_path: Path to video file
        timestamp_seconds: Time in seconds to extract frame
        output_path: Output path for the image (default: same dir as video)

    Returns:
        Tuple of (path to saved JPEG image, list of OCR texts)
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

        # Determine output path (use /tmp by default)
        if output_path is None:
            output_path = Path("/tmp") / f"{video_path.stem}_frame_{int(timestamp_seconds)}s.jpg"
        else:
            output_path = Path(output_path)

        # Save as high-quality JPEG (no resize)
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Perform OCR
        texts = ocr_image(str(output_path))

        return str(output_path), texts

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
