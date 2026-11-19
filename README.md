# Vehicle Counting

Vehicle detection and counting for images, videos, and live cameras using YOLO11 and ByteTrack.

## Requirements

- Python 3.11 or newer
- A webcam for live camera mode

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The YOLO11s checkpoint (`yolo11s.pt`) is downloaded automatically by Ultralytics on first run.
For faster processing on CPU, you can use `--model yolo11n.pt`; for more accuracy, use `yolo11m.pt`.
The tracker only counts objects that remain valid for at least 5 frames and removes unusually small detections.

## Add Inputs

- Put images in `dataset/images/`.
- Put videos in `dataset/videos/`.

These folders are excluded from Git so input media is not uploaded accidentally.

## Run

Process every image in `dataset/images/`:

```powershell
python src/count_image.py
```

Process every video in `dataset/videos/` with tracking IDs:

```powershell
python src/tracking.py
```

Common accuracy options:

```powershell
python src/tracking.py --model yolo11s.pt --conf 0.35 --imgsz 640 --min-track-frames 5
python src/count_image.py --model yolo11s.pt --conf 0.35 --imgsz 1280
```

Run live tracking with the default camera (device index 0). Press `q` to end the session and save its result:

```powershell
python src/tracking.py --camera 0
```

You can pass one image or video path instead of a dataset directory:

```powershell
python src/count_image.py dataset/images/example.jpg
python src/tracking.py dataset/videos/example.mp4
```

## Results

- `output/vehicle_counts/`: annotated images and per-image CSV files.
- `output/vehicle_tracking/videos/`: annotated video and camera-session recordings.
- `output/vehicle_tracking/all_counts.csv`: one statistics row per input image, video, or camera session.
- `output/vehicle_tracking/index.html`: searchable dashboard with totals and links to results.

Generated outputs are excluded from Git by default.

Counts are model predictions. Image counts are detections in a single image; video and camera counts are unique tracker IDs within each video or camera session. The total adds results across sources, so the same physical vehicle appearing in multiple sources is counted once per source.