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
    # One benchmark branch: full-frame baseline or the blue monitored ROI.
    name: str
    label: str
    roi: str


CASES = [
    # Full frame shows tracking noise from the whole scene; blue ROI keeps only
    # the entrance area used for the final result.
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
    parser.add_argument(
        "--comparison-frame",
        type=int,
        default=2103,
        help="Frame index used for the full-frame vs blue-ROI comparison PNG.",
    )
    parser.add_argument(
        "--no-comparison-frame",
        action="store_true",
        help="Skip the full-frame vs blue-ROI comparison PNG.",
    )
    return parser.parse_args()


def command_for_case(case: BenchmarkCase, output_dir: Path, args: argparse.Namespace) -> list[str]:
    roi = args.roi if case.name == "blue_roi" else case.roi
    # Reuse longest_stationary.py for both branches so the only intended
    # difference is the ROI gate.
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
        # The benchmark chart only needs summary metrics; videos are optional
        # because they are slower and much larger.
        command.append("--no-video-output")
        command.append("--no-frame-strip")
        command.append("--no-duration-chart")
        command.append("--no-identity-map")
    return command


def read_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def row_from_summary(case: BenchmarkCase, output_dir: Path, summary: dict) -> dict[str, object]:
    # Normalize the nested summary into comparable ROI-vs-full-frame metrics.
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
    # Metrics have different scales, so each row is normalized independently
    # while the numeric labels keep the exact values visible.
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


def read_video_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    if not video_path.exists():
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count:
            # Clamp the requested frame so the comparison still works on short
            # debug clips produced with --max-frames.
            frame_idx = max(0, min(frame_count - 1, frame_idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            return None
        return frame
    finally:
        cap.release()


def draw_metric_card(
    canvas: np.ndarray,
    origin: tuple[int, int],
    title: str,
    row: dict[str, object],
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    card_width = 560
    card_height = 116
    cv2.rectangle(canvas, (x, y), (x + card_width, y + card_height), (248, 248, 248), -1)
    cv2.rectangle(canvas, (x, y), (x + card_width, y + card_height), (225, 225, 225), 1)
    cv2.rectangle(canvas, (x, y), (x + 10, y + card_height), color, -1)
    draw_text(canvas, title, (x + 24, y + 34), scale=0.72, thickness=2)
    draw_text(
        canvas,
        f"Analyzed raw tracking IDs: {row['tracking_id_count']}",
        (x + 24, y + 66),
        scale=0.52,
    )
    draw_text(
        canvas,
        f"Stable person IDs: {row['person_id_count']}",
        (x + 300, y + 66),
        scale=0.52,
    )
    draw_text(
        canvas,
        f"Longest stationary: P{row['longest_person_id']} / {row['longest_duration_human']}",
        (x + 24, y + 96),
        scale=0.52,
        thickness=2,
    )


def draw_roi_comparison_frame(
    rows: list[dict[str, object]],
    output_root: Path,
    path: Path,
    *,
    frame_idx: int,
) -> Path | None:
    # This side-by-side figure is available only when --keep-videos created the
    # annotated videos for both benchmark branches.
    row_by_case = {str(row["case"]): row for row in rows}
    full_row = row_by_case.get("full_frame")
    roi_row = row_by_case.get("blue_roi")
    if full_row is None or roi_row is None:
        return None

    full_frame = read_video_frame(output_root / "full_frame" / "full_frame.mp4", frame_idx)
    blue_roi_frame = read_video_frame(output_root / "blue_roi" / "blue_roi.mp4", frame_idx)
    if full_frame is None or blue_roi_frame is None:
        return None

    tile_width = 600
    tile_height = 338
    full_tile = cv2.resize(full_frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
    roi_tile = cv2.resize(blue_roi_frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)

    margin = 40
    gap = 24
    header_height = 100
    card_height = 136
    canvas_width = margin * 2 + tile_width * 2 + gap
    canvas_height = header_height + tile_height + card_height + margin
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    draw_text(canvas, "Full-frame vs Blue ROI", (margin, 52), scale=1.05, thickness=2)
    draw_text(
        canvas,
        "Blue ROI keeps the same longest-stay answer while reducing tracker noise outside the entrance area.",
        (margin, 84),
        scale=0.55,
        color=(80, 80, 80),
    )

    y = header_height
    full_x = margin
    roi_x = margin + tile_width + gap
    canvas[y : y + tile_height, full_x : full_x + tile_width] = full_tile
    canvas[y : y + tile_height, roi_x : roi_x + tile_width] = roi_tile
    cv2.rectangle(canvas, (full_x, y), (full_x + tile_width, y + tile_height), (210, 210, 210), 1)
    cv2.rectangle(canvas, (roi_x, y), (roi_x + tile_width, y + tile_height), (210, 210, 210), 1)

    card_y = y + tile_height + 20
    draw_metric_card(canvas, (full_x, card_y), "Full frame", full_row, (80, 90, 220))
    draw_metric_card(canvas, (roi_x, card_y), "Blue ROI", roi_row, (255, 180, 40))

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Could not write ROI comparison frame: {path}")
    return path


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    cases = [
        CASES[0],
        # Allow callers to override the default blue ROI without changing the
        # full-frame baseline branch.
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
            # Reuse lets README/asset generation refresh charts without
            # rerunning the detector when summaries already exist.
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
    comparison_path = output_root / "benchmark_roi_comparison.png"

    write_csv(results_path, rows)
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    draw_benchmark_chart(rows, chart_path)
    comparison_result = None
    if not args.no_comparison_frame:
        comparison_result = draw_roi_comparison_frame(
            rows,
            output_root,
            comparison_path,
            frame_idx=args.comparison_frame,
        )

    print()
    print(f"Wrote benchmark results: {results_path}")
    print(f"Wrote benchmark summary: {summary_path}")
    print(f"Wrote benchmark chart: {chart_path}")
    if comparison_result is not None:
        print(f"Wrote ROI comparison frame: {comparison_result}")
    elif args.no_comparison_frame:
        print("ROI comparison frame: skipped")
    else:
        print("ROI comparison frame: skipped because benchmark videos are missing")


if __name__ == "__main__":
    main()
