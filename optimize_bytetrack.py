from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


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

    print()
    print(f"Wrote experiment log: {results_path}")
    print(f"Wrote best experiment: {best_path}")
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
