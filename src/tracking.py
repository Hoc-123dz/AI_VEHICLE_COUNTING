
import argparse
import csv
from datetime import datetime
import hashlib
from pathlib import Path
from collections import defaultdict

import cv2
from ultralytics import YOLO

try:
    from .reporting import PROJECT_DIR, RESULTS_DIR, load_records, upsert_record, write_report
except ImportError:
    from reporting import PROJECT_DIR, RESULTS_DIR, load_records, upsert_record, write_report

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
VEHICLE_CLASSES = [1, 2, 3, 5, 7]
OUTPUT_DIR = PROJECT_DIR / "output" / "vehicle_tracking"
VIDEOS_DIR = OUTPUT_DIR / "videos"
SUMMARY_PATH = OUTPUT_DIR / "video_counts.csv"
REPORT_PATH = RESULTS_DIR / "index.html"
CLASS_LABELS = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def load_summary() -> dict[str, dict[str, str]]:
    if not SUMMARY_PATH.is_file():
        return {}
    with SUMMARY_PATH.open("r", newline="", encoding="utf-8-sig") as summary_file:
        return {
            row["source_video"]: row
            for row in csv.DictReader(summary_file)
            if row.get("source_video")
        }


def output_video_path(video_path: Path, source: Path, summaries: dict[str, dict[str, str]]) -> Path:
    source_key = str(video_path.resolve())
    previous = summaries.get(source_key)
    if previous and previous.get("result_video"):
        previous_path = PROJECT_DIR / previous["result_video"]
        if previous_path.parent == VIDEOS_DIR:
            return previous_path

    if source.is_dir():
        name_parts = video_path.relative_to(source).with_suffix("").parts
    else:
        try:
            name_parts = video_path.relative_to(PROJECT_DIR).with_suffix("").parts
        except ValueError:
            name_parts = (video_path.parent.name, video_path.stem)
    output_stem = "__".join(name_parts) or video_path.stem
    candidate = VIDEOS_DIR / f"{output_stem}_tracking.mp4"

    existing_sources = {
        key
        for key, row in summaries.items()
        if row.get("result_video") == str(candidate.relative_to(PROJECT_DIR))
    }
    if candidate.exists() and source_key not in existing_sources:
        suffix = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:8]
        candidate = VIDEOS_DIR / f"{output_stem}_{suffix}_tracking.mp4"
    return candidate


def process_video(model: YOLO, video_path: Path, output_path: Path) -> dict[str, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if fps <= 0:
        fps = 30.0

    writer = None
    tracked_frames = 0
    frames_with_ids = 0
    seen_track_ids = defaultdict(set)

    try:
        results = model.track(
            source=str(video_path),
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            conf=0.25,
            stream=True,
            persist=True,
            save=False,
            show=False,
            verbose=False,
        )

        for result in results:
            frame = result.orig_img.copy()
            boxes = result.boxes

            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create output video: {output_path}")

            if boxes is not None and boxes.id is not None:
                frames_with_ids += 1
                coordinates = boxes.xyxy.int().cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist()

                for (x1, y1, x2, y2), class_id, track_id in zip(
                    coordinates, class_ids, track_ids
                ):
                    seen_track_ids[class_id].add(track_id)
                    label = f"{CLASS_LABELS[class_id]} ID:{track_id}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    text_y = max(y1 - 8, 24)
                    cv2.putText(
                        frame,
                        label,
                        (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 220, 0),
                        2,
                        cv2.LINE_AA,
                    )

            summary = "  ".join(
                f"{CLASS_LABELS[class_id]}: {len(seen_track_ids[class_id])}"
                for class_id in VEHICLE_CLASSES
            )
            cv2.rectangle(frame, (8, 8), (900, 45), (0, 0, 0), -1)
            cv2.putText(
                frame,
                summary,
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)
            tracked_frames += 1
    finally:
        if writer is not None:
            writer.release()

    if tracked_frames == 0:
        raise RuntimeError(f"No frames were processed from: {video_path}")

    print(f"Input video: {video_path}")
    print(f"Processed frames: {tracked_frames}; frames with IDs: {frames_with_ids}")
    print("Counts: " + ", ".join(
        f"{CLASS_LABELS[class_id]}={len(seen_track_ids[class_id])}"
        for class_id in VEHICLE_CLASSES
    ))
    print(f"Video result: {output_path}")

    row = {
        "source_video": str(video_path.resolve()),
        "processed_frames": str(tracked_frames),
        "result_video": str(output_path.relative_to(PROJECT_DIR)),
    }
    row.update(
        {
            CLASS_LABELS[class_id]: str(len(seen_track_ids[class_id]))
            for class_id in VEHICLE_CLASSES
        }
    )
    upsert_record(
        {
            "source_id": row["source_video"],
            "source_type": "Video",
            "source_name": video_path.name,
            **{
                CLASS_LABELS[class_id]: row[CLASS_LABELS[class_id]]
                for class_id in VEHICLE_CLASSES
            },
            "processed_frames": row["processed_frames"],
            "count_method": "ID theo dõi duy nhất",
            "result_path": row["result_video"],
        }
    )
    return row


def write_summary(summaries: dict[str, dict[str, str]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_video",
        *[CLASS_LABELS[class_id] for class_id in VEHICLE_CLASSES],
        "processed_frames",
        "result_video",
    ]
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        for source_video in sorted(summaries):
            writer.writerow(summaries[source_video])


def write_html_report(summaries: dict[str, dict[str, str]]) -> None:
    write_report(load_records())


def process_camera(model: YOLO, camera_index: int) -> None:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Cannot open camera {camera_index}. Check the camera index and permissions."
        )

    session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"camera_{camera_index}_{session_time}"
    output_path = VIDEOS_DIR / f"{session_name}_tracking.mp4"
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    writer = None
    tracked_frames = 0
    seen_track_ids = defaultdict(set)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.track(
                source=frame,
                tracker="bytetrack.yaml",
                classes=VEHICLE_CLASSES,
                conf=0.25,
                persist=True,
                verbose=False,
            )[0]
            boxes = result.boxes
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create camera recording: {output_path}")

            if boxes is not None and boxes.id is not None:
                coordinates = boxes.xyxy.int().cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist()
                for (x1, y1, x2, y2), class_id, track_id in zip(
                    coordinates, class_ids, track_ids
                ):
                    seen_track_ids[class_id].add(track_id)
                    label = f"{CLASS_LABELS[class_id]} ID:{track_id}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 8, 24)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 220, 0),
                        2,
                        cv2.LINE_AA,
                    )

            counts_text = "  ".join(
                f"{CLASS_LABELS[class_id]}: {len(seen_track_ids[class_id])}"
                for class_id in VEHICLE_CLASSES
            )
            cv2.rectangle(frame, (8, 8), (900, 45), (0, 0, 0), -1)
            cv2.putText(
                frame,
                counts_text,
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            cv2.imshow("Vehicle tracking - press q to stop", frame)
            tracked_frames += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if tracked_frames == 0:
        raise RuntimeError("Camera session ended before any frames were captured.")

    relative_output = str(output_path.relative_to(PROJECT_DIR))
    source_id = f"camera://{camera_index}/{session_time}"
    upsert_record(
        {
            "source_id": source_id,
            "source_type": "Camera",
            "source_name": f"Camera {camera_index} - {session_time}",
            **{
                CLASS_LABELS[class_id]: str(len(seen_track_ids[class_id]))
                for class_id in VEHICLE_CLASSES
            },
            "processed_frames": str(tracked_frames),
            "count_method": "ID theo dõi duy nhất trong phiên",
            "result_path": relative_output,
        }
    )
    print(f"Camera session finished: {tracked_frames} frames")
    print("Counts: " + ", ".join(
        f"{CLASS_LABELS[class_id]}={len(seen_track_ids[class_id])}"
        for class_id in VEHICLE_CLASSES
    ))
    print(f"Recording: {output_path}")
    print(f"Dashboard: {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track and count vehicles separately in each video."
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=PROJECT_DIR / "dataset" / "videos",
        help="Video file or directory (defaults to dataset/videos/).",
    )
    parser.add_argument(
        "--camera",
        nargs="?",
        const=0,
        type=int,
        help="Track a live camera by device index (default index: 0). Press q to end the session.",
    )
    args = parser.parse_args()
    model = YOLO(str(PROJECT_DIR / "yolo11n.pt"))
    if args.camera is not None:
        process_camera(model, args.camera)
        return

    source = args.source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input not found: {source}")
    if source.is_file():
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video type: {source.suffix}")
        video_paths = [source]
    else:
        video_paths = sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
    if not video_paths:
        raise FileNotFoundError(f"No supported videos found in: {source}")

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = load_summary()
    for video_path in video_paths:
        output_path = output_video_path(video_path, source, summaries)
        summaries[str(video_path.resolve())] = process_video(
            model, video_path, output_path
        )
        write_summary(summaries)

    print(f"Master count report: {SUMMARY_PATH}")
    write_html_report(summaries)
    print(f"Readable table: {REPORT_PATH}")


if __name__ == "__main__":
    main()
