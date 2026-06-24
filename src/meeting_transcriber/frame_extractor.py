"""Video frame extraction module."""

from concurrent.futures import ThreadPoolExecutor
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
    output_dir: str | None = None,
) -> tuple[str, list[str], str | None]:
    """
    Extract a frame from video at specified timestamp and save as JPEG.
    Also performs OCR on the frame.

    Args:
        video_path: Path to video file
        timestamp_seconds: Time in seconds to extract frame
        output_dir: Output directory for organized files (creates frames/ subdir)
                   If None, saves to /tmp (temporary)

    Returns:
        Tuple of (path to saved JPEG image, list of OCR texts, path to OCR text file or None)
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

        # Determine output path
        timestamp_str = f"{int(timestamp_seconds):05d}"
        if output_dir is not None:
            # Organized output: create frames/ subdirectory
            frames_dir = Path(output_dir) / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            output_path = frames_dir / f"frame_{timestamp_str}s.jpg"
            ocr_path = frames_dir / f"frame_{timestamp_str}s_ocr.txt"
        else:
            # Temporary output
            output_path = Path("/tmp") / f"{video_path.stem}_frame_{timestamp_str}s.jpg"
            ocr_path = None

        # Save as high-quality JPEG (no resize)
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Perform OCR
        texts = ocr_image(str(output_path))

        # Save OCR text if organized output
        if ocr_path is not None and texts:
            ocr_path.write_text("\n".join(texts), encoding="utf-8")

        return str(output_path), texts, str(ocr_path) if ocr_path else None

    finally:
        cap.release()


def extract_frames(
    video_path: str,
    timestamps: list[float],
    output_dir: str | None = None,
    max_ocr_workers: int = 6,
) -> list[dict]:
    """複数タイムスタンプのフレームを動画1回オープンで抽出し、OCRを並列実行する。

    1枚ずつ extract_frame を呼ぶと毎回 VideoCapture を開き直し＋OCRを逐次実行するため、
    枚数に比例して遅い。ここでは動画を1回だけ開いて全フレームを連続デコードし、重い
    Vision OCR をスレッドプールで並列化する（OCRはGILを解放するブロッキング呼び出しなので
    並列が効く）。フレーム抽出が「文字起こし後」工程の主要コストなので効果が大きい。

    戻り値: [{timestamp, image_path, ocr_path, ocr_texts}], timestamp昇順。
    """
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video}")

    if output_dir is not None:
        frames_dir = Path(output_dir) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        frames_dir = Path("/tmp")

    saved: list[dict] = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # 前方シークは速いので昇順に処理する
        for ts in sorted(set(float(t) for t in timestamps)):
            if ts < 0 or (duration and ts > duration):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
            ret, frame = cap.read()
            if not ret:
                continue
            ts_str = f"{int(ts):05d}"
            if output_dir is not None:
                img_path = frames_dir / f"frame_{ts_str}s.jpg"
                ocr_path = frames_dir / f"frame_{ts_str}s_ocr.txt"
            else:
                img_path = frames_dir / f"{video.stem}_frame_{ts_str}s.jpg"
                ocr_path = None
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append({"timestamp": ts, "image_path": str(img_path), "ocr_path": ocr_path})
    finally:
        cap.release()

    # OCR を並列実行（Vision はGILを解放するので ThreadPool が効く）
    def _ocr(entry: dict) -> dict:
        texts = ocr_image(entry["image_path"])
        if entry["ocr_path"] is not None and texts:
            Path(entry["ocr_path"]).write_text("\n".join(texts), encoding="utf-8")
        return {
            "timestamp": entry["timestamp"],
            "image_path": entry["image_path"],
            "ocr_path": str(entry["ocr_path"]) if entry["ocr_path"] else None,
            "ocr_texts": texts,
        }

    if not saved:
        return []
    workers = max(1, min(max_ocr_workers, len(saved)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_ocr, saved))
    results.sort(key=lambda r: r["timestamp"])
    return results


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
