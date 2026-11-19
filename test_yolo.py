
from pathlib import Path

import cv2
from ultralytics import YOLO

project_dir = Path(__file__).resolve().parent
video_path = project_dir / "dataset" / "videos" / "traffic.mp4"
output_dir = project_dir / "output" / "yolo_detection"
output_path = output_dir / "traffic_detection.mp4"
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading YOLO11 model...")
model = YOLO("yolo11s.pt")

video_capture = cv2.VideoCapture(str(video_path))
if not video_capture.isOpened():
    raise FileNotFoundError(f"Cannot open input video: {video_path}")
fps = video_capture.get(cv2.CAP_PROP_FPS)
video_capture.release()
if fps <= 0:
    fps = 30.0

print("Detecting vehicles...")
writer = None
try:
    for result in model.predict(
        source=str(video_path),
        classes=[2, 3, 5, 7],
        conf=0.35,
        iou=0.55,
        imgsz=1280,
        device="cpu",
        stream=True,
        save=False,
    ):
        frame = result.plot()
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
        writer.write(frame)
finally:
    if writer is not None:
        writer.release()

if writer is None:
    raise RuntimeError("No video frames were processed.")

print("Detection complete.")
print(f"Open the result video: {output_path}")
