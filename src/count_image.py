import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    from .reporting import PROJECT_DIR, upsert_record
except ImportError:
    from reporting import PROJECT_DIR, upsert_record


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VEHICLE_CLASSES = [1, 2, 3, 5, 7]
CLASS_LABELS = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def process_image(model: YOLO, image_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_output = output_dir / f"{image_path.stem}_counted.jpg"
    csv_output = output_dir / f"{image_path.stem}_counts.csv"

    result = model.predict(
        source=str(image_path),
        classes=VEHICLE_CLASSES,
        conf=0.25,
        verbose=False,
    )[0]
    image = result.orig_img.copy()
    counts = {class_id: 0 for class_id in VEHICLE_CLASSES}
    boxes = result.boxes

    if boxes is not None and len(boxes):
        coordinates = boxes.xyxy.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        for detection_id, (coords, class_id, confidence) in enumerate(
            zip(coordinates, class_ids, confidences), start=1
        ):
            counts[class_id] += 1
            x1, y1, x2, y2 = coords
            label = f"{CLASS_LABELS[class_id]} #{detection_id} {confidence:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
            text_y = max(y1 - 8, 22)
            cv2.putText(
                image,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    summary = "  ".join(
        f"{CLASS_LABELS[class_id]}: {counts[class_id]}"
        for class_id in VEHICLE_CLASSES
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        image,
        summary,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if not cv2.imwrite(str(image_output), image):
        raise RuntimeError(f"Could not write result image: {image_output}")
    with csv_output.open("w", newline="", encoding="utf-8") as counts_file:
        writer = csv.writer(counts_file)
        writer.writerow(["vehicle_type", "detected_count"])
        for class_id in VEHICLE_CLASSES:
            writer.writerow([CLASS_LABELS[class_id], counts[class_id]])

    upsert_record(
        {
            "source_id": str(image_path.resolve()),
            "source_type": "Ảnh",
            "source_name": image_path.name,
            **{
                CLASS_LABELS[class_id]: str(counts[class_id])
                for class_id in VEHICLE_CLASSES
            },
            "processed_frames": "1",
            "count_method": "Phát hiện trong ảnh",
            "result_path": str(image_output.relative_to(PROJECT_DIR)),
        }
    )

    print(f"Input: {image_path}")
    print(f"Image result: {image_output}")
    print(f"Count report: {csv_output}")
    print("Dashboard: output/vehicle_tracking/index.html")
    print("Counts: " + ", ".join(
        f"{CLASS_LABELS[class_id]}={counts[class_id]}"
        for class_id in VEHICLE_CLASSES
    ))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and count vehicles separately in each image."
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=PROJECT_DIR / "dataset" / "images",
        help="Image file or directory (defaults to dataset/images/).",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input not found: {source}")
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {source.suffix}")
        image_paths = [source]
        output_root = PROJECT_DIR / "output" / "vehicle_counts"
    else:
        image_paths = sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        output_root = PROJECT_DIR / "output" / "vehicle_counts" / source.name
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {source}")

    model = YOLO(str(PROJECT_DIR / "yolo11n.pt"))
    for image_path in image_paths:
        if source.is_dir():
            relative_parent = image_path.relative_to(source).parent
            output_dir = output_root / relative_parent
        else:
            output_dir = output_root
        process_image(model, image_path, output_dir)


if __name__ == "__main__":
    main()