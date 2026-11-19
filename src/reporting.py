import csv
import html
import os
from pathlib import Path
from urllib.parse import quote


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
RESULTS_DIR = OUTPUT_DIR / "vehicle_tracking"
RESULTS_PATH = RESULTS_DIR / "all_counts.csv"
REPORT_PATH = RESULTS_DIR / "index.html"
VEHICLE_LABELS = {
    "car": "Ô tô",
    "motorcycle": "Xe máy",
    "bus": "Xe buýt",
    "truck": "Xe tải",
}
FIELDNAMES = [
    "source_id",
    "source_type",
    "source_name",
    *VEHICLE_LABELS,
    "processed_frames",
    "count_method",
    "result_path",
]


def _legacy_records() -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    legacy_video_csv = RESULTS_DIR / "video_counts.csv"
    if legacy_video_csv.is_file():
        with legacy_video_csv.open("r", newline="", encoding="utf-8-sig") as file:
            for old in csv.DictReader(file):
                source_id = old.get("source_video", "")
                if not source_id:
                    continue
                records[source_id] = {
                    "source_id": source_id,
                    "source_type": "Video",
                    "source_name": Path(source_id).name,
                    **{label: old.get(label, "0") for label in VEHICLE_LABELS},
                    "processed_frames": old.get("processed_frames", "0"),
                    "count_method": "ID theo dõi duy nhất",
                    "result_path": old.get("result_video", ""),
                }

    dataset_images = list((PROJECT_DIR / "dataset" / "images").rglob("*"))
    for image_csv in (OUTPUT_DIR / "vehicle_counts").rglob("*_counts.csv"):
        image_stem = image_csv.name.removesuffix("_counts.csv")
        matches = [
            image for image in dataset_images
            if image.is_file() and image.stem == image_stem
        ]
        source_path = matches[0] if len(matches) == 1 else image_csv
        source_id = str(source_path.resolve())
        image_counts = {label: "0" for label in VEHICLE_LABELS}
        with image_csv.open("r", newline="", encoding="utf-8-sig") as file:
            for count_row in csv.DictReader(file):
                label = count_row.get("vehicle_type", "")
                if label in image_counts:
                    image_counts[label] = count_row.get("detected_count", "0")
        try:
            image_output = next(
                image_csv.parent.glob(f"{image_stem}_counted.*")
            ).relative_to(PROJECT_DIR).as_posix()
        except (StopIteration, ValueError):
            image_output = ""
        records[source_id] = {
            "source_id": source_id,
            "source_type": "Ảnh",
            "source_name": source_path.name,
            **image_counts,
            "processed_frames": "1",
            "count_method": "Phát hiện trong ảnh",
            "result_path": image_output,
        }
    return records


def load_records() -> dict[str, dict[str, str]]:
    if RESULTS_PATH.is_file():
        with RESULTS_PATH.open("r", newline="", encoding="utf-8-sig") as file:
            return {
                row["source_id"]: row
                for row in csv.DictReader(file)
                if row.get("source_id")
            }
    return _legacy_records()


def save_records(records: dict[str, dict[str, str]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for source_id in sorted(records):
            writer.writerow(
                {field: records[source_id].get(field, "") for field in FIELDNAMES}
            )
    write_report(records)


def upsert_record(record: dict[str, str]) -> None:
    records = load_records()
    normalized = {field: str(record.get(field, "")) for field in FIELDNAMES}
    for label in VEHICLE_LABELS:
        normalized[label] = normalized[label] or "0"
    records[normalized["source_id"]] = normalized
    save_records(records)


def write_report(records: dict[str, dict[str, str]]) -> None:
    totals = {
        label: sum(int(record.get(label, "0") or 0) for record in records.values())
        for label in VEHICLE_LABELS
    }
    rows = []
    for source_id in sorted(records):
        record = records[source_id]
        result_path = PROJECT_DIR / record.get("result_path", "")
        result_link = ""
        if record.get("result_path") and result_path.is_file():
            relative = os.path.relpath(result_path, RESULTS_DIR).replace("\\", "/")
            result_link = f'<a class="open-result" href="{quote(relative)}">Mở kết quả</a>'
        count_cells = "".join(
            f'<td class="number">{html.escape(record.get(label, "0"))}</td>'
            for label in VEHICLE_LABELS
        )
        rows.append(
            "<tr>"
            f'<td><span class="file-name">{html.escape(record.get("source_type", ""))}</span></td>'
            f'<td title="{html.escape(source_id, quote=True)}">'
            f'<span class="file-name">{html.escape(record.get("source_name", ""))}</span>'
            f'<span class="file-path">{html.escape(source_id)}</span></td>'
            f"{count_cells}"
            f'<td class="number">{html.escape(record.get("processed_frames", "0"))}</td>'
            f'<td>{html.escape(record.get("count_method", ""))}</td>'
            f'<td>{result_link}</td></tr>'
        )

    column_count = 2 + len(VEHICLE_LABELS) + 3
    table_rows = "\n".join(rows) if rows else (
        f'<tr><td class="empty" colspan="{column_count}">Chưa có ảnh, video hoặc phiên camera nào.</td></tr>'
    )
    cards = "".join(
        f'<div class="total-card"><span>{VEHICLE_LABELS[label]}</span><strong>{totals[label]}</strong></div>'
        for label in VEHICLE_LABELS
    )
    total_cells = "".join(
        f'<td class="number">{totals[label]}</td>' for label in VEHICLE_LABELS
    )
    count_headers = "".join(
        f'<th class="number">{html.escape(label_name)}</th>'
        for label_name in VEHICLE_LABELS.values()
    )
    report = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Thống kê phương tiện</title>
  <style>
    :root {{ color-scheme: light; --ink: #17252b; --muted: #65747a; --line: #dce5e4; --paper: #f3f7f5; --accent: #087f72; --accent-soft: #d9f1e9; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.5 "Segoe UI", sans-serif; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 32px 26px 52px; }}
    h1 {{ margin: 4px 0; font-size: 28px; }}
    .subtitle {{ margin: 0; color: var(--muted); }}
    .totals {{ display: grid; grid-template-columns: repeat({len(VEHICLE_LABELS)}, minmax(110px, 1fr)); gap: 10px; margin: 22px 0; }}
    .total-card {{ padding: 13px 15px; border: 1px solid var(--line); border-radius: 6px; background: white; }}
    .total-card span {{ display: block; color: var(--muted); font-size: 13px; }}
    .total-card strong {{ display: block; color: var(--accent); font-size: 25px; font-variant-numeric: tabular-nums; }}
    .toolbar {{ display: flex; justify-content: space-between; gap: 14px; align-items: center; margin-bottom: 12px; }}
    input {{ width: min(100%, 380px); padding: 10px 12px; border: 1px solid var(--line); border-radius: 6px; background: white; font: inherit; }}
    .muted {{ color: var(--muted); }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 7px; background: white; }}
    table {{ width: 100%; min-width: 1120px; border-collapse: collapse; }}
    th, td {{ padding: 11px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    thead th {{ background: #eaf2ef; color: #43565b; font-size: 12px; }}
    .number {{ text-align: center; font-variant-numeric: tabular-nums; }}
    .file-name {{ display: block; font-weight: 650; }}
    .file-path {{ display: block; max-width: 310px; overflow: hidden; color: var(--muted); font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }}
    .total-row th, .total-row td {{ border-top: 2px solid var(--accent); background: var(--accent-soft); font-weight: 750; }}
    .open-result {{ display: inline-block; padding: 6px 9px; border-radius: 5px; background: var(--accent-soft); color: #075e55; font-weight: 650; text-decoration: none; white-space: nowrap; }}
    .empty {{ padding: 28px; color: var(--muted); text-align: center; }}
    @media (max-width: 640px) {{ main {{ padding: 22px 13px; }} h1 {{ font-size: 23px; }} .totals {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .toolbar {{ align-items: stretch; flex-direction: column; }} input {{ width: 100%; }} }}
  </style>
</head>
<body><main>
  <h1>Thống kê phương tiện</h1>
  <p class="subtitle">Tổng hợp ảnh, video và phiên camera. Mỗi dòng là một nguồn riêng; ảnh đếm phát hiện, video/camera đếm ID duy nhất trong phiên.</p>
  <section class="totals" aria-label="Tổng số trên các nguồn đã xử lý">{cards}</section>
  <p class="muted">Tổng là tổng lượt đếm của các nguồn đã xử lý; cùng một xe xuất hiện ở nhiều nguồn sẽ được tính trong từng nguồn.</p>
  <div class="toolbar"><input id="search" type="search" placeholder="Tìm theo loại nguồn hoặc tên file..." aria-label="Tìm nguồn"><span class="muted">Nguồn: <strong id="source-count">{len(records)}</strong></span></div>
  <div class="table-wrap"><table>
    <thead><tr><th>Loại</th><th>Nguồn</th>{count_headers}<th class="number">Frames</th><th>Cách đếm</th><th>Kết quả</th></tr></thead>
    <tbody id="rows">{table_rows}</tbody>
    <tfoot><tr class="total-row"><th colspan="2">TỔNG TẤT CẢ NGUỒN ({len(records)})</th>{total_cells}<th>-</th><th>-</th><th>-</th></tr></tfoot>
  </table></div>
  <script>
    const search = document.getElementById('search');
    const rows = [...document.querySelectorAll('#rows tr')];
    search.addEventListener('input', () => {{
      const query = search.value.trim().toLocaleLowerCase();
      let visible = 0;
      for (const row of rows) {{
        const matches = row.textContent.toLocaleLowerCase().includes(query);
        row.hidden = !matches;
        if (matches) visible += 1;
      }}
      document.getElementById('source-count').textContent = String(visible);
    }});
  </script>
</main></body>
</html>
"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")