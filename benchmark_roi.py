from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_ROI = "916,0,1915,1080"


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    label: str
    roi: str


CASES = [
    BenchmarkCase(name="full_frame", label="Full frame", roi="none"),
    BenchmarkCase(name="blue_roi", label="Blue ROI", roi=DEFAULT_ROI),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark full-frame tracking against ROI-gated tracking.")
    parser.add_argument("--video", default="entrance.mov")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--tracker", default="tracker_configs/bytetrack_long_occlusion.yaml")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--roi", default=DEFAULT_ROI)
    parser.add_argument("--output-dir", default="experiments/roi_benchmark")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full video.")
    parser.add_argument(
        "--keep-videos",
        action="store_true",
        help="Write annotated videos for both benchmark cases.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse case summary.json files when they already exist.",
    )
    return parser.parse_args()


def command_for_case(case: BenchmarkCase, output_dir: Path, args: argparse.Namespace) -> list[str]:
    roi = args.roi if case.name == "blue_roi" else case.roi
    command = [
        sys.executable,
        "longest_stationary.py",
        "--video",
        args.video,
        "--model",
        args.model,
        "--tracker",
        args.tracker,
        "--conf",
        str(args.conf),
        "--roi",
        roi,
        "--output-dir",
        str(output_dir),
    ]
    if args.max_frames:
        command.extend(["--max-frames", str(args.max_frames)])
    if args.keep_videos:
        command.extend(["--output-video", f"{case.name}.mp4"])
    else:
        command.append("--no-video-output")
        command.append("--no-frame-strip")
    return command


def read_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def row_from_summary(case: BenchmarkCase, output_dir: Path, summary: dict) -> dict[str, object]:
    identity = summary.get("identity_relinking", {})
    result = summary.get("result") or {}
    raw_track_ids = result.get("raw_track_ids") or []
    roi = summary.get("roi")

    return {
        "case": case.name,
        "label": case.label,
        "roi_enabled": "yes" if roi else "no",
        "roi": "" if not roi else "{x1},{y1},{x2},{y2}".format(**roi),
        "detected_tracking_id_count": int(identity.get("detected_raw_tracker_id_count", 0)),
        "tracking_id_count": int(identity.get("raw_tracker_id_count", 0)),
        "ignored_outside_roi_tracking_id_count": int(
            identity.get("ignored_outside_roi_raw_track_count", 0)
        ),
        "person_id_count": int(identity.get("stable_person_id_count", 0)),
        "relinked_tracking_id_count": int(identity.get("relinked_raw_track_count", 0)),
        "merged_person_id_count": int(identity.get("merged_person_id_count", 0)),
        "longest_person_id": result.get("person_id", ""),
        "longest_person_raw_track_ids": "|".join(str(value) for value in raw_track_ids),
        "longest_start_sec": result.get("start_sec", ""),
        "longest_end_sec": result.get("end_sec", ""),
        "longest_duration_sec": float(result.get("duration_sec") or 0.0),
        "longest_duration_human": result.get("duration_human", ""),
        "output_dir": str(output_dir),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value: float, metric_key: str) -> str:
    if metric_key == "longest_duration_sec":
        return f"{value:.1f}s"
    return str(int(round(value)))


def draw_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.6,
    color: tuple[int, int, int] = (35, 35, 35),
    thickness: int = 1,
) -> None:
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_benchmark_chart(rows: list[dict[str, object]], path: Path) -> None:
    metrics = [
        ("tracking_id_count", "Analyzed raw tracking IDs"),
        ("person_id_count", "Stable person IDs"),
        ("relinked_tracking_id_count", "Relinked raw IDs"),
        ("merged_person_id_count", "Merged stable IDs"),
        ("longest_duration_sec", "Longest stationary duration"),
    ]
    colors = {
        "full_frame": (80, 90, 220),
        "blue_roi": (255, 180, 40),
    }

    canvas = np.full((720, 1280, 3), 255, dtype=np.uint8)
    draw_text(canvas, "ROI Tracking Benchmark", (40, 54), scale=1.05, thickness=2)
    draw_text(
        canvas,
        "Each metric is independently scaled; lower ID counts usually mean less tracking noise.",
        (40, 88),
        scale=0.58,
        color=(80, 80, 80),
    )

    x_label = 40
    x_bar = 430
    max_bar_width = 620
    y_start = 145
    metric_gap = 102
    bar_height = 24

    for metric_index, (metric_key, metric_label) in enumerate(metrics):
        y = y_start + metric_index * metric_gap
        draw_text(canvas, metric_label, (x_label, y + 24), scale=0.62, thickness=2)
        max_value = max(float(row[metric_key]) for row in rows) or 1.0

        for row_index, row in enumerate(rows):
            value = float(row[metric_key])
            bar_y = y + row_index * 34
            bar_width = int(round(max_bar_width * value / max_value))
            color = colors.get(str(row["case"]), (120, 120, 120))
            cv2.rectangle(canvas, (x_bar, bar_y), (x_bar + bar_width, bar_y + bar_height), color, -1)
            cv2.rectangle(
                canvas,
                (x_bar, bar_y),
                (x_bar + max_bar_width, bar_y + bar_height),
                (225, 225, 225),
                1,
            )
            label = f"{row['label']}: {format_metric(value, metric_key)}"
            draw_text(canvas, label, (x_bar + max_bar_width + 24, bar_y + 19), scale=0.55)

    legend_y = 675
    for index, row in enumerate(rows):
        x = 40 + index * 190
        color = colors.get(str(row["case"]), (120, 120, 120))
        cv2.rectangle(canvas, (x, legend_y - 16), (x + 28, legend_y + 8), color, -1)
        draw_text(canvas, str(row["label"]), (x + 40, legend_y + 3), scale=0.55)

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Could not write benchmark chart: {path}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    cases = [
        CASES[0],
        BenchmarkCase(name="blue_roi", label="Blue ROI", roi=args.roi),
    ]

    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", ".cache/ultralytics")
    env.setdefault("MPLCONFIGDIR", ".cache/matplotlib")
    env.setdefault("XDG_CACHE_HOME", ".cache")

    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        case_dir = output_root / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        summary_path = case_dir / "summary.json"

        if args.reuse_existing and summary_path.exists():
            print(f"[{index}/{len(cases)}] Reusing {case.label}: {summary_path}", flush=True)
        else:
            command = command_for_case(case, case_dir, args)
            print(f"[{index}/{len(cases)}] Running {case.label}", flush=True)
            print(" ".join(command), flush=True)
            completed = subprocess.run(command, check=False, env=env)
            if completed.returncode != 0:
                raise SystemExit(f"Benchmark case failed: {case.name}")

        row = row_from_summary(case, case_dir, read_summary(summary_path))
        rows.append(row)
        print(
            f"  analyzed IDs={row['tracking_id_count']} "
            f"stable IDs={row['person_id_count']} "
            f"longest=P{row['longest_person_id']} "
            f"{float(row['longest_duration_sec']):.3f}s",
            flush=True,
        )

    results_path = output_root / "benchmark_results.csv"
    summary_path = output_root / "benchmark_summary.json"
    chart_path = output_root / "benchmark_roi.png"

    write_csv(results_path, rows)
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    draw_benchmark_chart(rows, chart_path)

    print()
    print(f"Wrote benchmark results: {results_path}")
    print(f"Wrote benchmark summary: {summary_path}")
    print(f"Wrote benchmark chart: {chart_path}")


if __name__ == "__main__":
    main()
