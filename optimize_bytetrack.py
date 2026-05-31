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


@dataclass(frozen=True)
class Experiment:
    name: str
    tracker: str
    conf: float
    track_high_thresh: float
    track_low_thresh: float
    new_track_thresh: float
    track_buffer: int
    match_thresh: float
    fuse_score: bool = True


EXPERIMENTS = [
    Experiment(
        name="default_conf030",
        tracker="tracker_configs/bytetrack_default.yaml",
        conf=0.30,
        track_high_thresh=0.25,
        track_low_thresh=0.10,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.80,
    ),
    Experiment(
        name="default_conf012",
        tracker="tracker_configs/bytetrack_default.yaml",
        conf=0.12,
        track_high_thresh=0.25,
        track_low_thresh=0.10,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.80,
    ),
    Experiment(
        name="balanced_buffer60",
        tracker="tracker_configs/bytetrack_balanced.yaml",
        conf=0.12,
        track_high_thresh=0.20,
        track_low_thresh=0.05,
        new_track_thresh=0.35,
        track_buffer=60,
        match_thresh=0.85,
    ),
    Experiment(
        name="occlusion_buffer90",
        tracker="tracker_configs/bytetrack_occlusion.yaml",
        conf=0.12,
        track_high_thresh=0.20,
        track_low_thresh=0.05,
        new_track_thresh=0.35,
        track_buffer=90,
        match_thresh=0.85,
    ),
    Experiment(
        name="long_occlusion_buffer120",
        tracker="tracker_configs/bytetrack_long_occlusion.yaml",
        conf=0.10,
        track_high_thresh=0.15,
        track_low_thresh=0.03,
        new_track_thresh=0.40,
        track_buffer=120,
        match_thresh=0.90,
    ),
    Experiment(
        name="conservative_buffer90",
        tracker="tracker_configs/bytetrack_conservative.yaml",
        conf=0.20,
        track_high_thresh=0.30,
        track_low_thresh=0.10,
        new_track_thresh=0.45,
        track_buffer=90,
        match_thresh=0.80,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep ByteTrack hyperparameters.")
    parser.add_argument("--video", default="entrance.mov")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--experiments-dir", default="experiments/bytetrack_optimization")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full video.")
    parser.add_argument(
        "--keep-videos",
        action="store_true",
        help="Write annotated video for each experiment. Slow and large.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional subset of experiment names to run.",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help=(
            "Optional ROI passed to longest_stationary.py as x1,y1,x2,y2. "
            "Use 'none' for full-frame analysis. When omitted, longest_stationary.py "
            "uses its default blue-box ROI."
        ),
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip ByteTrack sweep chart generation.",
    )
    return parser.parse_args()


def command_for_experiment(
    experiment: Experiment,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "longest_stationary.py",
        "--video",
        args.video,
        "--model",
        args.model,
        "--tracker",
        experiment.tracker,
        "--conf",
        str(experiment.conf),
        "--output-dir",
        str(output_dir),
    ]
    if args.max_frames:
        command.extend(["--max-frames", str(args.max_frames)])
    if not args.keep_videos:
        command.append("--no-video-output")
        command.append("--no-frame-strip")
        command.append("--no-duration-chart")
        command.append("--no-identity-map")
    else:
        command.extend(["--output-video", f"{experiment.name}.mp4"])
    if args.roi is not None:
        command.extend(["--roi", args.roi])
    return command


def read_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def row_from_summary(experiment: Experiment, output_dir: Path, summary: dict) -> dict[str, object]:
    identity = summary.get("identity_relinking", {})
    result = summary.get("result") or {}
    raw_track_ids = result.get("raw_track_ids") or []
    duration = result.get("duration_sec") or 0.0

    return {
        "experiment": experiment.name,
        "roi_enabled": "yes" if summary.get("roi") else "no",
        "roi": (
            ""
            if not summary.get("roi")
            else "{x1},{y1},{x2},{y2}".format(**summary["roi"])
        ),
        "tracker_config": experiment.tracker,
        "conf": experiment.conf,
        "track_high_thresh": experiment.track_high_thresh,
        "track_low_thresh": experiment.track_low_thresh,
        "new_track_thresh": experiment.new_track_thresh,
        "track_buffer": experiment.track_buffer,
        "match_thresh": experiment.match_thresh,
        "fuse_score": experiment.fuse_score,
        "all_person_id_count": identity.get("stable_person_id_count", 0),
        "tracking_id_count": identity.get("raw_tracker_id_count", 0),
        "relinked_tracking_id_count": identity.get("relinked_raw_track_count", 0),
        "merged_person_id_count": identity.get("merged_person_id_count", 0),
        "longest_person_id": result.get("person_id", ""),
        "longest_person_raw_track_ids": "|".join(str(value) for value in raw_track_ids),
        "longest_start_sec": result.get("start_sec", ""),
        "longest_end_sec": result.get("end_sec", ""),
        "longest_duration_sec": duration,
        "longest_duration_human": result.get("duration_human", ""),
        "output_dir": str(output_dir),
    }


def choose_best(rows: list[dict[str, object]]) -> dict[str, object] | None:
    if not rows:
        return None

    max_duration = max(float(row["longest_duration_sec"]) for row in rows)
    min_acceptable_duration = max_duration * 0.95
    eligible = [
        row for row in rows if float(row["longest_duration_sec"]) >= min_acceptable_duration
    ]

    # Without ground truth, use a conservative proxy:
    # preserve the longest-stay answer, then minimize tracker fragmentation.
    return min(
        eligible,
        key=lambda row: (
            int(row["tracking_id_count"]),
            int(row["all_person_id_count"]),
            -float(row["longest_duration_sec"]),
        ),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.55,
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


def compact_experiment_name(name: object) -> str:
    return str(name).replace("_buffer", " b").replace("_conf", " c")


def draw_bytetrack_sweep_chart(
    rows: list[dict[str, object]],
    best: dict[str, object] | None,
    path: Path,
) -> None:
    if not rows:
        return

    sorted_rows = sorted(
        rows,
        key=lambda row: (int(row["track_buffer"]), float(row["conf"]), str(row["experiment"])),
    )
    max_tracking_ids = max(int(row["tracking_id_count"]) for row in sorted_rows) or 1
    max_person_ids = max(int(row["all_person_id_count"]) for row in sorted_rows) or 1
    max_duration = max(float(row["longest_duration_sec"]) for row in sorted_rows) or 1.0
    best_name = None if best is None else str(best.get("experiment"))

    width = 1280
    height = 760
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    draw_text(canvas, "ByteTrack config sweep", (40, 54), scale=1.05, thickness=2)
    draw_text(
        canvas,
        "Lower ID counts mean less fragmentation; confidence gates shown as detector conf | high/low/new.",
        (40, 88),
        scale=0.56,
        color=(80, 80, 80),
    )

    y_start = 162
    row_gap = 92
    label_x = 40
    buffer_x = 300
    raw_x = 430
    stable_x = 430
    duration_x = 1015
    bar_max_width = 460
    bar_h = 20

    draw_text(canvas, "config", (label_x, 122), scale=0.5, color=(90, 90, 90))
    draw_text(canvas, "track_buffer", (buffer_x, 122), scale=0.5, color=(90, 90, 90))
    draw_text(canvas, "raw / stable IDs", (raw_x, 122), scale=0.5, color=(90, 90, 90))
    draw_text(canvas, "longest", (duration_x, 122), scale=0.5, color=(90, 90, 90))

    for index, row in enumerate(sorted_rows):
        y = y_start + index * row_gap
        is_best = str(row["experiment"]) == best_name
        if is_best:
            cv2.rectangle(canvas, (28, y - 42), (1248, y + 44), (245, 250, 255), -1)
            cv2.rectangle(canvas, (28, y - 42), (1248, y + 44), (210, 230, 245), 1)

        name = compact_experiment_name(row["experiment"])
        draw_text(canvas, name, (label_x, y), scale=0.52, thickness=2 if is_best else 1)
        confidence_label = (
            f"conf {float(row['conf']):.2f} | "
            f"{float(row['track_high_thresh']):.2f}/"
            f"{float(row['track_low_thresh']):.2f}/"
            f"{float(row['new_track_thresh']):.2f}"
        )
        draw_text(canvas, confidence_label, (label_x, y + 25), scale=0.38, color=(80, 80, 80))
        if is_best:
            draw_text(canvas, "selected", (label_x, y + 45), scale=0.4, color=(35, 120, 35), thickness=2)

        buffer_value = int(row["track_buffer"])
        cv2.rectangle(canvas, (buffer_x, y - 22), (buffer_x + 82, y + 12), (235, 235, 235), -1)
        cv2.rectangle(canvas, (buffer_x, y - 22), (buffer_x + 82, y + 12), (210, 210, 210), 1)
        draw_text(canvas, str(buffer_value), (buffer_x + 22, y + 1), scale=0.58, thickness=2)

        tracking_ids = int(row["tracking_id_count"])
        person_ids = int(row["all_person_id_count"])
        raw_width = int(round(bar_max_width * tracking_ids / max_tracking_ids))
        stable_width = int(round(bar_max_width * person_ids / max_person_ids))
        cv2.rectangle(canvas, (raw_x, y - 28), (raw_x + bar_max_width, y - 8), (232, 232, 232), 1)
        cv2.rectangle(canvas, (stable_x, y + 2), (stable_x + bar_max_width, y + 22), (232, 232, 232), 1)
        cv2.rectangle(canvas, (raw_x, y - 28), (raw_x + raw_width, y - 8), (80, 90, 220), -1)
        cv2.rectangle(canvas, (stable_x, y + 2), (stable_x + stable_width, y + 22), (255, 180, 40), -1)
        draw_text(canvas, f"raw {tracking_ids}", (raw_x + bar_max_width + 16, y - 11), scale=0.45)
        draw_text(canvas, f"stable {person_ids}", (stable_x + bar_max_width + 16, y + 19), scale=0.45)

        duration_sec = float(row["longest_duration_sec"])
        duration_width = int(round(145 * duration_sec / max_duration))
        cv2.rectangle(canvas, (duration_x, y - 18), (duration_x + 145, y + 8), (232, 232, 232), 1)
        cv2.rectangle(canvas, (duration_x, y - 18), (duration_x + duration_width, y + 8), (44, 160, 44), -1)
        draw_text(canvas, str(row["longest_duration_human"]), (duration_x + 160, y + 3), scale=0.45)

    legend_y = 720
    cv2.rectangle(canvas, (40, legend_y - 16), (68, legend_y + 8), (80, 90, 220), -1)
    draw_text(canvas, "raw tracker IDs", (80, legend_y + 4), scale=0.48)
    cv2.rectangle(canvas, (245, legend_y - 16), (273, legend_y + 8), (255, 180, 40), -1)
    draw_text(canvas, "stable person IDs", (285, legend_y + 4), scale=0.48)
    cv2.rectangle(canvas, (490, legend_y - 16), (518, legend_y + 8), (44, 160, 44), -1)
    draw_text(canvas, "longest stationary duration", (530, legend_y + 4), scale=0.48)

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Could not write ByteTrack sweep chart: {path}")


def main() -> None:
    args = parse_args()
    experiments_dir = Path(args.experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    selected_names = set(args.names) if args.names else None
    experiments = [
        experiment
        for experiment in EXPERIMENTS
        if selected_names is None or experiment.name in selected_names
    ]
    if not experiments:
        raise SystemExit("No experiments selected.")

    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", ".cache/ultralytics")
    env.setdefault("MPLCONFIGDIR", ".cache/matplotlib")
    env.setdefault("XDG_CACHE_HOME", ".cache")

    rows: list[dict[str, object]] = []
    for index, experiment in enumerate(experiments, start=1):
        output_dir = experiments_dir / experiment.name
        output_dir.mkdir(parents=True, exist_ok=True)
        command = command_for_experiment(experiment, output_dir, args)

        print(f"[{index}/{len(experiments)}] Running {experiment.name}")
        print(" ".join(command))
        completed = subprocess.run(command, check=False, env=env)
        if completed.returncode != 0:
            raise SystemExit(f"Experiment failed: {experiment.name}")

        summary_path = output_dir / "summary.json"
        row = row_from_summary(experiment, output_dir, read_summary(summary_path))
        rows.append(row)
        print(
            f"  person IDs={row['all_person_id_count']} "
            f"tracking IDs={row['tracking_id_count']} "
            f"longest=P{row['longest_person_id']} "
            f"{float(row['longest_duration_sec']):.3f}s"
        )

    results_path = experiments_dir / "experiment_results.csv"
    write_csv(results_path, rows)

    best = choose_best(rows)
    best_path = experiments_dir / "best_experiment.json"
    best_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    chart_path = experiments_dir / "bytetrack_sweep.png"
    if not args.no_chart:
        draw_bytetrack_sweep_chart(rows, best, chart_path)

    print()
    print(f"Wrote experiment log: {results_path}")
    print(f"Wrote best experiment: {best_path}")
    if not args.no_chart:
        print(f"Wrote ByteTrack sweep chart: {chart_path}")
    if best:
        print(
            "Best by proxy objective: "
            f"{best['experiment']} "
            f"(tracking IDs={best['tracking_id_count']}, "
            f"person IDs={best['all_person_id_count']}, "
            f"longest=P{best['longest_person_id']} "
            f"{float(best['longest_duration_sec']):.3f}s)"
        )


if __name__ == "__main__":
    main()
