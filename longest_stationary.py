from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Iterable

import cv2
import numpy as np

os.environ.setdefault("YOLO_CONFIG_DIR", ".cache/ultralytics")
os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", ".cache")

from ultralytics import YOLO


Point = tuple[float, float]
Rect = tuple[int, int, int, int]
DEFAULT_ROI = "916,0,1915,1080"


@dataclass
class Segment:
    person_id: int
    start_frame: int
    end_frame: int

    @property
    def duration_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame + 1)


@dataclass
class TrackState:
    person_id: int
    raw_track_ids: set[int] = field(default_factory=set)
    last_raw_track_id: int | None = None
    first_seen_frame: int = -1
    smooth_point: Point | None = None
    positions: Deque[Point] = field(default_factory=deque)
    bbox_height: float = 0.0
    last_bbox: tuple[int, int, int, int] | None = None
    last_seen_frame: int = -1
    appearance_hist: np.ndarray | None = None
    last_status: str = "observing"
    in_stationary: bool = False
    stationary_start_frame: int | None = None
    unstable_frames: int = 0
    best_segment: Segment | None = None


@dataclass
class BestUpdate:
    frame: int
    time_sec: float
    person_id: int
    start_sec: float
    end_sec: float
    duration_sec: float


@dataclass
class IdentityMatch:
    person_id: int
    score: float
    position_score: float
    color_score: float
    size_score: float
    time_score: float
    gap_frames: int
    distance_px: float


@dataclass
class ReIdEvent:
    event: str
    frame: int
    time_sec: float
    raw_track_id: int
    person_id: int
    previous_raw_track_id: int | None = None
    score: float | None = None
    position_score: float | None = None
    color_score: float | None = None
    size_score: float | None = None
    time_score: float | None = None
    gap_frames: int | None = None
    distance_px: float | None = None


def parse_roi(value: str | None) -> Rect | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"", "none", "off", "full", "full-frame", "full_frame"}:
        return None

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x1,y1,x2,y2 or 'none'.")

    try:
        x1, y1, x2, y2 = [int(round(float(part))) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI values must be numeric.") from exc

    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("ROI requires x2 > x1 and y2 > y1.")
    return x1, y1, x2, y2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the person who stayed stationary the longest in a video."
    )
    parser.add_argument("--video", default="entrance.mov", help="Input video path.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO model.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for output files.")
    parser.add_argument("--output-video", default="annotated_entrance.mp4", help="Annotated result video filename.")
    parser.add_argument(
        "--frame-strip-output",
        default="longest_stay_strip.png",
        help="PNG filename for sampled frames across the longest stationary interval.",
    )
    parser.add_argument(
        "--frame-strip-count",
        type=int,
        default=5,
        help="Number of frames to sample in the longest-stay frame strip.",
    )
    parser.add_argument(
        "--no-frame-strip",
        action="store_true",
        help="Skip longest-stay frame strip generation.",
    )
    parser.add_argument(
        "--duration-chart-output",
        default="top_stationary_durations.png",
        help="PNG filename for the top stationary durations bar chart.",
    )
    parser.add_argument(
        "--duration-chart-count",
        type=int,
        default=10,
        help="Number of people to show in the top stationary durations bar chart.",
    )
    parser.add_argument(
        "--no-duration-chart",
        action="store_true",
        help="Skip top stationary durations chart generation.",
    )
    parser.add_argument(
        "--identity-map-output",
        default="identity_map.png",
        help="PNG filename for the stable-person to raw-tracker ID mapping chart.",
    )
    parser.add_argument(
        "--identity-map-count",
        type=int,
        default=8,
        help="Number of merged stable identities to show in the identity mapping chart.",
    )
    parser.add_argument(
        "--no-identity-map",
        action="store_true",
        help="Skip stable-person to raw-tracker ID mapping chart generation.",
    )
    parser.add_argument(
        "--no-video-output",
        action="store_true",
        help="Skip annotated video generation. Useful for hyperparameter sweeps.",
    )
    parser.add_argument("--conf", type=float, default=0.30, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.50, help="YOLO IoU threshold.")
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.30,
        help="EMA smoothing factor for bottom-center positions. Higher reacts faster.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=0.75,
        help="Recent time window used to decide whether a track is stationary.",
    )
    parser.add_argument(
        "--stationary-ratio",
        type=float,
        default=0.04,
        help="Stationary threshold as a ratio of current bbox height.",
    )
    parser.add_argument(
        "--min-stationary-px",
        type=float,
        default=8.0,
        help="Minimum stationary movement threshold in pixels.",
    )
    parser.add_argument(
        "--grace-frames",
        type=int,
        default=5,
        help="Number of unstable frames allowed before closing a stationary segment.",
    )
    parser.add_argument(
        "--max-lost-frames",
        type=int,
        default=30,
        help="Finalize an active stationary segment after this many missed frames.",
    )
    parser.add_argument(
        "--disable-reid",
        action="store_true",
        help="Disable custom appearance-based identity relinking.",
    )
    parser.add_argument(
        "--reid-max-gap-seconds",
        type=float,
        default=2.0,
        help="Only relink a new raw track to identities last seen within this many seconds.",
    )
    parser.add_argument(
        "--reid-score-threshold",
        type=float,
        default=0.70,
        help="Minimum weighted score required to relink a new raw track.",
    )
    parser.add_argument(
        "--reid-min-color-score",
        type=float,
        default=0.45,
        help="Minimum trouser color histogram similarity required for relinking.",
    )
    parser.add_argument(
        "--reid-min-position-score",
        type=float,
        default=0.10,
        help="Minimum position-continuity score required for relinking.",
    )
    parser.add_argument(
        "--reid-position-ratio",
        type=float,
        default=0.75,
        help="Position relinking gate as a ratio of bbox height.",
    )
    parser.add_argument(
        "--reid-min-position-px",
        type=float,
        default=80.0,
        help="Minimum position relinking gate in pixels.",
    )
    parser.add_argument(
        "--reid-motion-px-per-sec",
        type=float,
        default=80.0,
        help="Extra allowed relinking distance per second of missed detections.",
    )
    parser.add_argument(
        "--appearance-alpha",
        type=float,
        default=0.20,
        help="EMA update factor for each stable person's trouser color histogram.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional debug limit. 0 means process the full video.",
    )
    parser.add_argument(
        "--roi",
        type=parse_roi,
        default=DEFAULT_ROI,
        help=(
            "Analysis ROI as x1,y1,x2,y2. Only tracks whose bottom-center point is "
            "inside this rectangle are counted. Use 'none' for full-frame analysis. "
            f"Default is the blue ROI from ROI.png: {DEFAULT_ROI}."
        ),
    )
    return parser.parse_args()


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bottom_center(xyxy: Iterable[float]) -> tuple[Point, float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return ((x1 + x2) / 2.0, y2), max(1.0, y2 - y1)


def clamp_bbox(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return x1, y1, x2, y2


def clamp_roi(roi: Rect, frame_size: tuple[int, int]) -> Rect:
    width, height = frame_size
    x1, y1, x2, y2 = roi
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"ROI is outside the video frame: {roi}")
    return x1, y1, x2, y2


def point_in_roi(point: Point, roi: Rect | None) -> bool:
    if roi is None:
        return True
    x1, y1, x2, y2 = roi
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2


def extract_trouser_hist(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    clamped = clamp_bbox(bbox, frame.shape)
    if clamped is None:
        return None

    x1, y1, x2, y2 = clamped
    width = x2 - x1
    height = y2 - y1

    # Center/lower crop reduces background, shoes, and nearby people inside the bbox.
    rx1 = int(round(x1 + 0.20 * width))
    rx2 = int(round(x1 + 0.80 * width))
    ry1 = int(round(y1 + 0.55 * height))
    ry2 = int(round(y1 + 0.90 * height))
    clamped_roi = clamp_bbox((rx1, ry1, rx2, ry2), frame.shape)
    if clamped_roi is None:
        return None

    rx1, ry1, rx2, ry2 = clamped_roi
    crop = frame[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.flatten().astype("float32")


def update_appearance_hist(state: TrackState, new_hist: np.ndarray | None, alpha: float) -> None:
    if new_hist is None:
        return
    if state.appearance_hist is None:
        state.appearance_hist = new_hist
        return
    updated = (1.0 - alpha) * state.appearance_hist + alpha * new_hist
    total = float(updated.sum())
    if total > 0:
        updated = updated / total
    state.appearance_hist = updated.astype("float32")


def color_similarity(old_hist: np.ndarray | None, new_hist: np.ndarray | None) -> float:
    if old_hist is None or new_hist is None:
        return 0.0
    distance_value = float(cv2.compareHist(old_hist, new_hist, cv2.HISTCMP_BHATTACHARYYA))
    if math.isnan(distance_value):
        return 0.0
    return max(0.0, min(1.0, 1.0 - distance_value))


def bbox_size_similarity(old_height: float, new_height: float) -> float:
    old_height = max(1.0, float(old_height))
    new_height = max(1.0, float(new_height))
    return min(old_height, new_height) / max(old_height, new_height)


def smooth_position(previous: Point | None, current: Point, alpha: float) -> Point:
    if previous is None:
        return current
    return (
        alpha * current[0] + (1.0 - alpha) * previous[0],
        alpha * current[1] + (1.0 - alpha) * previous[1],
    )


def stationary_threshold(bbox_height: float, ratio: float, min_px: float) -> float:
    return max(min_px, ratio * bbox_height)


def window_radius(points: Deque[Point]) -> float:
    if not points:
        return 0.0
    mean_x = sum(p[0] for p in points) / len(points)
    mean_y = sum(p[1] for p in points) / len(points)
    return max(distance(p, (mean_x, mean_y)) for p in points)


def find_identity_match(
    person_states: dict[int, TrackState],
    assigned_person_ids: set[int],
    frame_idx: int,
    point: Point,
    bbox_height: float,
    appearance_hist: np.ndarray | None,
    *,
    fps: float,
    args: argparse.Namespace,
) -> IdentityMatch | None:
    if args.disable_reid:
        return None

    max_gap_frames = max(1, int(round(args.reid_max_gap_seconds * fps)))
    best: IdentityMatch | None = None

    for person_id, state in person_states.items():
        if person_id in assigned_person_ids:
            continue
        if state.smooth_point is None or state.last_seen_frame < 0:
            continue

        gap_frames = frame_idx - state.last_seen_frame
        if gap_frames < 1 or gap_frames > max_gap_frames:
            continue

        gap_seconds = gap_frames / fps if fps > 0 else 0.0
        scale = max(float(state.bbox_height), float(bbox_height), 1.0)
        position_limit = (
            max(args.reid_min_position_px, args.reid_position_ratio * scale)
            + args.reid_motion_px_per_sec * gap_seconds
        )
        distance_px = distance(state.smooth_point, point)
        position_score = max(0.0, min(1.0, 1.0 - distance_px / position_limit))
        if position_score < args.reid_min_position_score:
            continue

        color_score = color_similarity(state.appearance_hist, appearance_hist)
        if color_score < args.reid_min_color_score:
            continue

        size_score = bbox_size_similarity(state.bbox_height, bbox_height)
        time_score = max(0.0, min(1.0, 1.0 - gap_frames / max_gap_frames))
        score = (
            0.45 * position_score
            + 0.35 * color_score
            + 0.10 * size_score
            + 0.10 * time_score
        )

        if score < args.reid_score_threshold:
            continue

        candidate = IdentityMatch(
            person_id=person_id,
            score=score,
            position_score=position_score,
            color_score=color_score,
            size_score=size_score,
            time_score=time_score,
            gap_frames=gap_frames,
            distance_px=distance_px,
        )
        if best is None or candidate.score > best.score:
            best = candidate

    return best


def resolve_person_id(
    raw_track_id: int,
    raw_to_person: dict[int, int],
    person_states: dict[int, TrackState],
    assigned_person_ids: set[int],
    frame_idx: int,
    point: Point,
    bbox_height: float,
    appearance_hist: np.ndarray | None,
    *,
    fps: float,
    args: argparse.Namespace,
    next_person_id: int,
) -> tuple[int, int, ReIdEvent | None]:
    if raw_track_id in raw_to_person:
        return raw_to_person[raw_track_id], next_person_id, None

    match = find_identity_match(
        person_states,
        assigned_person_ids,
        frame_idx,
        point,
        bbox_height,
        appearance_hist,
        fps=fps,
        args=args,
    )
    if match is not None:
        raw_to_person[raw_track_id] = match.person_id
        previous_raw_track_id = person_states[match.person_id].last_raw_track_id
        return (
            match.person_id,
            next_person_id,
            ReIdEvent(
                event="relinked",
                frame=frame_idx,
                time_sec=seconds(frame_idx, fps),
                raw_track_id=raw_track_id,
                person_id=match.person_id,
                previous_raw_track_id=previous_raw_track_id,
                score=match.score,
                position_score=match.position_score,
                color_score=match.color_score,
                size_score=match.size_score,
                time_score=match.time_score,
                gap_frames=match.gap_frames,
                distance_px=match.distance_px,
            ),
        )

    person_id = next_person_id
    raw_to_person[raw_track_id] = person_id
    return (
        person_id,
        next_person_id + 1,
        ReIdEvent(
            event="created",
            frame=frame_idx,
            time_sec=seconds(frame_idx, fps),
            raw_track_id=raw_track_id,
            person_id=person_id,
        ),
    )


def update_best_segment(state: TrackState, candidate: Segment) -> None:
    if state.best_segment is None:
        state.best_segment = candidate
        return
    if candidate.duration_frames > state.best_segment.duration_frames:
        state.best_segment = candidate


def finalize_stationary_segment(state: TrackState, end_frame: int) -> Segment | None:
    if not state.in_stationary or state.stationary_start_frame is None:
        return None
    if end_frame < state.stationary_start_frame:
        state.in_stationary = False
        state.stationary_start_frame = None
        state.unstable_frames = 0
        return None

    segment = Segment(
        person_id=state.person_id,
        start_frame=state.stationary_start_frame,
        end_frame=end_frame,
    )
    update_best_segment(state, segment)
    state.in_stationary = False
    state.stationary_start_frame = None
    state.unstable_frames = 0
    return segment


def update_track_state(
    state: TrackState,
    frame_idx: int,
    raw_track_id: int,
    raw_point: Point,
    bbox_height: float,
    bbox: tuple[int, int, int, int],
    appearance_hist: np.ndarray | None,
    *,
    alpha: float,
    appearance_alpha: float,
    window_frames: int,
    threshold_ratio: float,
    min_threshold_px: float,
    grace_frames: int,
) -> Segment | None:
    state.raw_track_ids.add(raw_track_id)
    state.last_raw_track_id = raw_track_id
    if state.first_seen_frame < 0:
        state.first_seen_frame = frame_idx
    update_appearance_hist(state, appearance_hist, appearance_alpha)

    state.smooth_point = smooth_position(state.smooth_point, raw_point, alpha)
    state.positions.append(state.smooth_point)
    state.bbox_height = bbox_height
    state.last_bbox = bbox
    state.last_seen_frame = frame_idx

    while len(state.positions) > window_frames:
        state.positions.popleft()

    threshold = stationary_threshold(bbox_height, threshold_ratio, min_threshold_px)
    if len(state.positions) < window_frames:
        state.last_status = "observing"
        return None

    is_stationary = window_radius(state.positions) <= threshold

    if is_stationary:
        state.unstable_frames = 0
        if not state.in_stationary:
            state.in_stationary = True
            state.stationary_start_frame = frame_idx - len(state.positions) + 1
        state.last_status = "stationary"
        current = Segment(
            person_id=state.person_id,
            start_frame=state.stationary_start_frame,
            end_frame=frame_idx,
        )
        update_best_segment(state, current)
        return current

    if state.in_stationary:
        state.unstable_frames += 1
        state.last_status = "unstable"
        if state.unstable_frames > grace_frames:
            return finalize_stationary_segment(state, frame_idx - state.unstable_frames)
    else:
        state.last_status = "moving"

    return None


def seconds(frame_idx: int, fps: float) -> float:
    return frame_idx / fps if fps > 0 else 0.0


def format_seconds(value: float) -> str:
    minutes = int(value // 60)
    sec = value - minutes * 60
    if minutes:
        return f"{minutes:d}m {sec:04.1f}s"
    return f"{sec:.1f}s"


def draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x, y - th - baseline - 6), (x + tw + 6, y + 4), color, -1)
    cv2.putText(frame, text, (x + 3, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    person_states: dict[int, TrackState],
    seen_person_ids: set[int],
    fps: float,
    best_update: BestUpdate | None,
    frame_idx: int,
    roi: Rect | None,
) -> None:
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 40), 3)
        draw_label(frame, "ROI", (x1 + 8, max(28, y1 + 28)), (255, 180, 40))

    for person_id in seen_person_ids:
        state = person_states[person_id]
        if state.last_bbox is None:
            continue

        x1, y1, x2, y2 = state.last_bbox
        if state.last_status == "stationary":
            color = (44, 160, 44)
        elif state.last_status == "unstable":
            color = (0, 165, 255)
        else:
            color = (60, 60, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        current_duration = 0.0
        if state.in_stationary and state.stationary_start_frame is not None:
            current_duration = seconds(frame_idx - state.stationary_start_frame + 1, fps)

        raw_label = f"T{state.last_raw_track_id}" if state.last_raw_track_id is not None else "T?"
        label = f"P{person_id}/{raw_label} {state.last_status}"
        if current_duration > 0:
            label += f" {format_seconds(current_duration)}"
        draw_label(frame, label, (x1, max(24, y1 - 6)), color)

        if state.smooth_point is not None:
            cx, cy = int(state.smooth_point[0]), int(state.smooth_point[1])
            cv2.circle(frame, (cx, cy), 4, color, -1)

    panel_lines = [
        f"Frame {frame_idx}",
        "Green=stationary Orange=grace Red=moving",
    ]
    if best_update is not None:
        panel_lines.append(
            f"Best: P{best_update.person_id} {format_seconds(best_update.duration_sec)}"
        )
    else:
        panel_lines.append("Best: waiting for stable track")

    x, y = 16, 28
    cv2.rectangle(frame, (8, 8), (520, 96), (0, 0, 0), -1)
    cv2.addWeighted(frame, 1.0, frame, 0.0, 0.0, frame)
    for i, line in enumerate(panel_lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def create_writer(output_path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    if writer.isOpened():
        return writer

    fallback = output_path.with_suffix(".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(fallback), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video writer for {output_path}")
    print(f"MP4 writer unavailable; wrote AVI instead: {fallback}")
    return writer


def draw_plain_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.65,
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


def find_written_video_path(output_video_path: Path) -> Path | None:
    if output_video_path.exists():
        return output_video_path

    fallback = output_video_path.with_suffix(".avi")
    if fallback.exists():
        return fallback

    return None


def frame_strip_label(index: int, total: int) -> str:
    if total <= 1:
        return "sample"
    if index == 0:
        return "start"
    if index == total - 1:
        return "end"
    if total == 3 and index == 1:
        return "middle"
    percent = round(100 * index / (total - 1))
    return f"{percent}%"


def write_longest_stay_frame_strip(
    source_video_path: Path,
    output_path: Path,
    best_update: BestUpdate,
    *,
    fps: float,
    frame_count: int,
    sample_count: int,
) -> Path:
    sample_count = max(1, sample_count)
    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for frame strip: {source_video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps or 30.0)
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or frame_count or 0)
    max_frame_index = max(0, source_frame_count - 1) if source_frame_count else None

    sample_times = np.linspace(best_update.start_sec, best_update.end_sec, sample_count)
    frames: list[tuple[np.ndarray, int, float, str]] = []

    try:
        for index, time_sec in enumerate(sample_times):
            frame_idx = int(round(float(time_sec) * source_fps))
            if max_frame_index is not None:
                frame_idx = max(0, min(max_frame_index, frame_idx))
            else:
                frame_idx = max(0, frame_idx)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append((frame, frame_idx, float(time_sec), frame_strip_label(index, sample_count)))
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"Could not read any sampled frames from {source_video_path}")

    tile_width = 360
    first_height, first_width = frames[0][0].shape[:2]
    tile_height = max(1, int(round(tile_width * first_height / first_width)))
    header_height = 78
    caption_height = 44
    margin = 24
    gap = 12

    canvas_width = margin * 2 + tile_width * len(frames) + gap * (len(frames) - 1)
    canvas_height = header_height + tile_height + caption_height + margin
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    title = (
        f"Longest-stay frame strip: P{best_update.person_id} "
        f"{format_seconds(best_update.duration_sec)}"
    )
    subtitle = (
        f"Sampled from {best_update.start_sec:.1f}s to {best_update.end_sec:.1f}s "
        f"({source_video_path.name})"
    )
    draw_plain_text(canvas, title, (margin, 36), scale=0.9, thickness=2)
    draw_plain_text(canvas, subtitle, (margin, 64), scale=0.55, color=(80, 80, 80))

    for index, (frame, frame_idx, time_sec, label) in enumerate(frames):
        x = margin + index * (tile_width + gap)
        y = header_height
        resized = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        canvas[y : y + tile_height, x : x + tile_width] = resized
        cv2.rectangle(canvas, (x, y), (x + tile_width, y + tile_height), (210, 210, 210), 1)
        caption = f"{label}: {time_sec:.1f}s | frame {frame_idx}"
        draw_plain_text(
            canvas,
            caption,
            (x, y + tile_height + 28),
            scale=0.48,
            color=(55, 55, 55),
            thickness=1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write longest-stay frame strip: {output_path}")
    return output_path


def best_duration_seconds(state: TrackState, fps: float) -> float:
    if state.best_segment is None:
        return 0.0
    return seconds(state.best_segment.duration_frames, fps)


def write_top_stationary_durations_chart(
    output_path: Path,
    person_states: dict[int, TrackState],
    *,
    fps: float,
    limit: int,
    roi: Rect | None,
) -> Path | None:
    rows = [
        (
            state.person_id,
            best_duration_seconds(state, fps),
            "|".join(str(value) for value in sorted(state.raw_track_ids)),
        )
        for state in person_states.values()
    ]
    rows = [row for row in rows if row[1] > 0.0]
    rows.sort(key=lambda row: (-row[1], row[0]))
    rows = rows[: max(1, limit)]
    if not rows:
        return None

    width = 1280
    height = 720
    margin_x = 56
    title_y = 56
    chart_top = 124
    chart_bottom = 662
    label_width = 220
    value_width = 120
    bar_x = margin_x + label_width
    bar_max_width = width - bar_x - value_width - margin_x
    row_gap = max(44, min(68, (chart_bottom - chart_top) // len(rows)))
    bar_height = min(34, max(18, row_gap - 18))
    max_duration = max(row[1] for row in rows)

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    draw_plain_text(canvas, "Top stationary durations", (margin_x, title_y), scale=1.05, thickness=2)
    if roi is None:
        subtitle = "Full-frame analysis"
    else:
        subtitle = f"Blue ROI analysis: x1={roi[0]}, y1={roi[1]}, x2={roi[2]}, y2={roi[3]}"
    draw_plain_text(canvas, subtitle, (margin_x, 88), scale=0.58, color=(80, 80, 80))

    for index, (person_id, duration_sec, raw_track_ids) in enumerate(rows):
        y = chart_top + index * row_gap
        bar_width = int(round(bar_max_width * duration_sec / max_duration))
        is_winner = index == 0
        bar_color = (255, 180, 40) if is_winner and roi is not None else (44, 160, 44)
        if not is_winner:
            bar_color = (165, 165, 165)

        label = f"P{person_id}"
        if raw_track_ids:
            label += f" / T{raw_track_ids}"
        draw_plain_text(canvas, label, (margin_x, y + 26), scale=0.58, thickness=2)

        cv2.rectangle(
            canvas,
            (bar_x, y),
            (bar_x + bar_max_width, y + bar_height),
            (232, 232, 232),
            1,
        )
        cv2.rectangle(canvas, (bar_x, y), (bar_x + bar_width, y + bar_height), bar_color, -1)
        draw_plain_text(
            canvas,
            format_seconds(duration_sec),
            (bar_x + bar_max_width + 20, y + 25),
            scale=0.58,
            thickness=2 if is_winner else 1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write top stationary durations chart: {output_path}")
    return output_path


def write_identity_mapping_chart(
    output_path: Path,
    person_states: dict[int, TrackState],
    reid_events: list[ReIdEvent],
    *,
    fps: float,
    limit: int,
    roi: Rect | None,
    highlight_person_id: int | None,
) -> Path | None:
    rows = [state for state in person_states.values() if len(state.raw_track_ids) > 1]
    if not rows:
        return None

    rows.sort(
        key=lambda state: (
            0 if highlight_person_id is not None and state.person_id == highlight_person_id else 1,
            -best_duration_seconds(state, fps),
            state.person_id,
        )
    )
    rows = rows[: max(1, limit)]
    relinked_raw_ids = {event.raw_track_id for event in reid_events if event.event == "relinked"}

    width = 1280
    row_gap = 78
    margin_x = 56
    header_height = 126
    footer_height = 72
    height = max(420, header_height + row_gap * len(rows) + footer_height)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    draw_plain_text(canvas, "Stable identity map", (margin_x, 56), scale=1.05, thickness=2)
    if roi is None:
        subtitle = "Full-frame analysis"
    else:
        subtitle = f"Blue ROI analysis: x1={roi[0]}, y1={roi[1]}, x2={roi[2]}, y2={roi[3]}"
    draw_plain_text(canvas, subtitle, (margin_x, 88), scale=0.58, color=(80, 80, 80))

    p_box_w = 94
    p_box_h = 42
    raw_box_w = 92
    raw_box_h = 42
    raw_gap = 36
    p_x = margin_x
    raw_start_x = 230
    duration_x = 1040
    y_start = header_height

    for row_index, state in enumerate(rows):
        y = y_start + row_index * row_gap
        is_highlight = highlight_person_id is not None and state.person_id == highlight_person_id
        p_color = (44, 160, 44)

        cv2.rectangle(canvas, (p_x, y), (p_x + p_box_w, y + p_box_h), p_color, -1)
        draw_plain_text(
            canvas,
            f"P{state.person_id}",
            (p_x + 16, y + 28),
            scale=0.7,
            color=(255, 255, 255),
            thickness=2,
        )

        raw_ids = sorted(state.raw_track_ids)
        first_raw_x = raw_start_x
        cv2.arrowedLine(
            canvas,
            (p_x + p_box_w + 12, y + p_box_h // 2),
            (first_raw_x - 14, y + p_box_h // 2),
            (130, 130, 130),
            2,
            tipLength=0.18,
        )

        for raw_index, raw_id in enumerate(raw_ids):
            x = raw_start_x + raw_index * (raw_box_w + raw_gap)
            raw_color = (255, 180, 40) if raw_id in relinked_raw_ids else (165, 165, 165)
            cv2.rectangle(canvas, (x, y), (x + raw_box_w, y + raw_box_h), raw_color, -1)
            draw_plain_text(
                canvas,
                f"T{raw_id}",
                (x + 18, y + 28),
                scale=0.7,
                color=(255, 255, 255),
                thickness=2,
            )
            if raw_index < len(raw_ids) - 1:
                cv2.arrowedLine(
                    canvas,
                    (x + raw_box_w + 6, y + raw_box_h // 2),
                    (x + raw_box_w + raw_gap - 8, y + raw_box_h // 2),
                    (130, 130, 130),
                    2,
                    tipLength=0.25,
                )

        duration = best_duration_seconds(state, fps)
        draw_plain_text(
            canvas,
            f"best stationary: {format_seconds(duration)}",
            (duration_x, y + 28),
            scale=0.55,
            color=(45, 45, 45),
            thickness=2 if is_highlight else 1,
        )

    legend_y = height - 36
    cv2.rectangle(canvas, (margin_x, legend_y - 22), (margin_x + 28, legend_y + 2), (44, 160, 44), -1)
    draw_plain_text(canvas, "stable P identity", (margin_x + 40, legend_y), scale=0.48)
    cv2.rectangle(canvas, (margin_x + 230, legend_y - 22), (margin_x + 258, legend_y + 2), (165, 165, 165), -1)
    draw_plain_text(canvas, "new raw T track", (margin_x + 270, legend_y), scale=0.48)
    cv2.rectangle(canvas, (margin_x + 440, legend_y - 22), (margin_x + 468, legend_y + 2), (255, 180, 40), -1)
    draw_plain_text(canvas, "relinked raw T track", (margin_x + 480, legend_y), scale=0.48)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write identity mapping chart: {output_path}")
    return output_path


def write_updates_csv(path: Path, updates: list[BestUpdate]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["frame", "time_sec", "person_id", "start_sec", "end_sec", "duration_sec"],
        )
        writer.writeheader()
        for update in updates:
            writer.writerow(
                {
                    "frame": int(update.frame),
                    "time_sec": f"{update.time_sec:.3f}",
                    "person_id": int(update.person_id),
                    "start_sec": f"{update.start_sec:.3f}",
                    "end_sec": f"{update.end_sec:.3f}",
                    "duration_sec": f"{update.duration_sec:.3f}",
                }
            )


def write_reid_events_csv(path: Path, events: list[ReIdEvent]) -> None:
    with path.open("w", newline="") as file:
        fieldnames = [
            "event",
            "frame",
            "time_sec",
            "raw_track_id",
            "person_id",
            "previous_raw_track_id",
            "score",
            "position_score",
            "color_score",
            "size_score",
            "time_score",
            "gap_frames",
            "distance_px",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "event": event.event,
                    "frame": int(event.frame),
                    "time_sec": f"{event.time_sec:.3f}",
                    "raw_track_id": int(event.raw_track_id),
                    "person_id": int(event.person_id),
                    "previous_raw_track_id": (
                        "" if event.previous_raw_track_id is None else int(event.previous_raw_track_id)
                    ),
                    "score": "" if event.score is None else f"{event.score:.3f}",
                    "position_score": (
                        "" if event.position_score is None else f"{event.position_score:.3f}"
                    ),
                    "color_score": "" if event.color_score is None else f"{event.color_score:.3f}",
                    "size_score": "" if event.size_score is None else f"{event.size_score:.3f}",
                    "time_score": "" if event.time_score is None else f"{event.time_score:.3f}",
                    "gap_frames": "" if event.gap_frames is None else int(event.gap_frames),
                    "distance_px": "" if event.distance_px is None else f"{event.distance_px:.1f}",
                }
            )


def write_identity_tracks_csv(path: Path, person_states: dict[int, TrackState], fps: float) -> None:
    with path.open("w", newline="") as file:
        fieldnames = [
            "person_id",
            "raw_track_ids",
            "raw_track_count",
            "first_seen_sec",
            "last_seen_sec",
            "best_stationary_duration_sec",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for person_id in sorted(person_states):
            state = person_states[person_id]
            best_duration = (
                seconds(state.best_segment.duration_frames, fps)
                if state.best_segment is not None
                else 0.0
            )
            writer.writerow(
                {
                    "person_id": person_id,
                    "raw_track_ids": "|".join(str(v) for v in sorted(state.raw_track_ids)),
                    "raw_track_count": len(state.raw_track_ids),
                    "first_seen_sec": f"{seconds(state.first_seen_frame, fps):.3f}",
                    "last_seen_sec": f"{seconds(state.last_seen_frame, fps):.3f}",
                    "best_stationary_duration_sec": f"{best_duration:.3f}",
                }
            )


def write_summary_json(
    path: Path,
    *,
    video_path: Path,
    fps: float,
    frame_count: int,
    best_update: BestUpdate | None,
    person_states: dict[int, TrackState],
    raw_track_count: int,
    reid_events: list[ReIdEvent],
    args: argparse.Namespace,
    roi: Rect | None,
    detected_raw_track_count: int,
    ignored_outside_roi_raw_track_count: int,
    frame_strip_path: Path | None,
    duration_chart_path: Path | None,
    identity_map_path: Path | None,
) -> None:
    if best_update is None:
        result = None
    else:
        best_state = person_states.get(best_update.person_id)
        result = {
            "person_id": int(best_update.person_id),
            "raw_track_ids": sorted(best_state.raw_track_ids) if best_state else [],
            "start_sec": round(float(best_update.start_sec), 3),
            "end_sec": round(float(best_update.end_sec), 3),
            "duration_sec": round(float(best_update.duration_sec), 3),
            "duration_human": format_seconds(best_update.duration_sec),
        }

    merged_people = [
        {
            "person_id": int(state.person_id),
            "raw_track_ids": sorted(int(v) for v in state.raw_track_ids),
        }
        for state in sorted(person_states.values(), key=lambda item: item.person_id)
        if len(state.raw_track_ids) > 1
    ]
    relinked_count = sum(1 for event in reid_events if event.event == "relinked")

    payload = {
        "video": str(video_path),
        "fps": float(fps),
        "frame_count": int(frame_count),
        "result": result,
        "visualizations": {
            "longest_stay_frame_strip": None if frame_strip_path is None else str(frame_strip_path),
            "top_stationary_durations_chart": (
                None if duration_chart_path is None else str(duration_chart_path)
            ),
            "identity_mapping_chart": None if identity_map_path is None else str(identity_map_path),
        },
        "identity_relinking": {
            "enabled": not args.disable_reid,
            "raw_tracker_id_count": int(raw_track_count),
            "detected_raw_tracker_id_count": int(detected_raw_track_count),
            "ignored_outside_roi_raw_track_count": int(ignored_outside_roi_raw_track_count),
            "stable_person_id_count": int(len(person_states)),
            "relinked_raw_track_count": int(relinked_count),
            "merged_person_id_count": int(len(merged_people)),
            "merged_people": merged_people,
        },
        "roi": (
            None
            if roi is None
            else {
                "x1": int(roi[0]),
                "y1": int(roi[1]),
                "x2": int(roi[2]),
                "y2": int(roi[3]),
                "selection_rule": "bottom-center point must be inside the ROI",
            }
        ),
        "method": {
            "detector_tracker": f"Ultralytics YOLO ({args.model}) with {args.tracker}",
            "conf": args.conf,
            "iou": args.iou,
            "annotated_video_enabled": not args.no_video_output,
            "roi": (
                "disabled; full-frame analysis"
                if roi is None
                else (
                    "Only detections whose bottom-center ground point is inside "
                    f"ROI x1={roi[0]}, y1={roi[1]}, x2={roi[2]}, y2={roi[3]} "
                    "are used for identity relinking and stationary-duration scoring."
                )
            ),
            "identity": (
                "YOLO tracker IDs are mapped to stable person IDs. New raw tracks can be "
                "relinked to recently lost person IDs using bottom-center continuity, "
                "lower-body HSV color histogram similarity, bbox-size ratio, and time gap."
            ),
            "position": "Smoothed bottom-center of each stable person bounding box",
            "stationary_rule": (
                "A track is stationary when the smoothed bottom-center points within the "
                "recent time window stay inside max(min_stationary_px, stationary_ratio * bbox_height)."
            ),
            "window_seconds": args.window_seconds,
            "smooth_alpha": args.smooth_alpha,
            "stationary_ratio": args.stationary_ratio,
            "min_stationary_px": args.min_stationary_px,
            "grace_frames": args.grace_frames,
            "reid_max_gap_seconds": args.reid_max_gap_seconds,
            "reid_score_threshold": args.reid_score_threshold,
            "reid_min_color_score": args.reid_min_color_score,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run() -> None:
    args = parse_args()
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_video_path = output_dir / args.output_video
    frame_strip_path = output_dir / args.frame_strip_output
    duration_chart_path = output_dir / args.duration_chart_output
    identity_map_path = output_dir / args.identity_map_output
    summary_path = output_dir / "summary.json"
    updates_path = output_dir / "longest_updates.csv"
    reid_events_path = output_dir / "reid_events.csv"
    identity_tracks_path = output_dir / "identity_tracks.csv"

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window_frames = max(2, int(round(args.window_seconds * fps)))
    roi = clamp_roi(args.roi, (width, height)) if args.roi is not None else None

    writer = None if args.no_video_output else create_writer(output_video_path, fps, (width, height))
    model = YOLO(args.model)

    person_states: dict[int, TrackState] = {}
    raw_to_person: dict[int, int] = {}
    all_raw_track_ids_seen: set[int] = set()
    raw_track_ids_seen: set[int] = set()
    reid_events: list[ReIdEvent] = []
    next_person_id = 1
    best_update: BestUpdate | None = None
    updates: list[BestUpdate] = []
    frame_idx = -1

    print(f"Processing {video_path} ({width}x{height}, {fps:.2f} FPS, {frame_count} frames)")
    print(f"Stationary window: {window_frames} frames ({args.window_seconds:.2f}s)")
    if roi is None:
        print("ROI: disabled (full-frame analysis)")
    else:
        print(f"ROI: x1={roi[0]}, y1={roi[1]}, x2={roi[2]}, y2={roi[3]} (bottom-center gate)")
    print(
        "Identity relinking: "
        f"{'enabled' if not args.disable_reid else 'disabled'} "
        f"(max gap {args.reid_max_gap_seconds:.1f}s, threshold {args.reid_score_threshold:.2f})"
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break

            results = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                classes=[0],
                conf=args.conf,
                iou=args.iou,
                verbose=False,
            )

            seen_person_ids: set[int] = set()
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes
                xyxy_values = boxes.xyxy.cpu().numpy()
                id_values = boxes.id.cpu().numpy().astype(int)

                for xyxy, raw_track_id in zip(xyxy_values, id_values):
                    raw_track_id = int(raw_track_id)
                    all_raw_track_ids_seen.add(raw_track_id)
                    raw_point, bbox_height = bottom_center(xyxy)
                    if not point_in_roi(raw_point, roi):
                        continue

                    raw_track_ids_seen.add(raw_track_id)
                    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                    bbox = (x1, y1, x2, y2)
                    appearance_hist = extract_trouser_hist(frame, bbox)

                    person_id, next_person_id, reid_event = resolve_person_id(
                        raw_track_id,
                        raw_to_person,
                        person_states,
                        seen_person_ids,
                        frame_idx,
                        raw_point,
                        bbox_height,
                        appearance_hist,
                        fps=fps,
                        args=args,
                        next_person_id=next_person_id,
                    )
                    if reid_event is not None:
                        reid_events.append(reid_event)

                    state = person_states.setdefault(person_id, TrackState(person_id=person_id))
                    seen_person_ids.add(person_id)

                    if state.last_seen_frame >= 0 and frame_idx - state.last_seen_frame > args.max_lost_frames:
                        closed_segment = finalize_stationary_segment(state, state.last_seen_frame)
                        state.positions.clear()
                        state.smooth_point = None
                        if closed_segment is not None:
                            duration_sec = seconds(closed_segment.duration_frames, fps)
                            if best_update is None or duration_sec > best_update.duration_sec:
                                best_update = BestUpdate(
                                    frame=frame_idx,
                                    time_sec=seconds(frame_idx, fps),
                                    person_id=person_id,
                                    start_sec=seconds(closed_segment.start_frame, fps),
                                    end_sec=seconds(closed_segment.end_frame, fps),
                                    duration_sec=duration_sec,
                                )
                                updates.append(best_update)

                    segment = update_track_state(
                        state,
                        frame_idx,
                        raw_track_id,
                        raw_point,
                        bbox_height,
                        bbox,
                        appearance_hist,
                        alpha=args.smooth_alpha,
                        appearance_alpha=args.appearance_alpha,
                        window_frames=window_frames,
                        threshold_ratio=args.stationary_ratio,
                        min_threshold_px=args.min_stationary_px,
                        grace_frames=args.grace_frames,
                    )

                    if segment is not None:
                        duration_sec = seconds(segment.duration_frames, fps)
                        if best_update is None or duration_sec > best_update.duration_sec:
                            best_update = BestUpdate(
                                frame=frame_idx,
                                time_sec=seconds(frame_idx, fps),
                                person_id=person_id,
                                start_sec=seconds(segment.start_frame, fps),
                                end_sec=seconds(segment.end_frame, fps),
                                duration_sec=duration_sec,
                            )
                            updates.append(best_update)

            for state in person_states.values():
                if state.person_id in seen_person_ids:
                    continue
                if state.in_stationary and frame_idx - state.last_seen_frame > args.max_lost_frames:
                    segment = finalize_stationary_segment(state, state.last_seen_frame)
                    if segment is not None:
                        duration_sec = seconds(segment.duration_frames, fps)
                        if best_update is None or duration_sec > best_update.duration_sec:
                            best_update = BestUpdate(
                                frame=frame_idx,
                                time_sec=seconds(frame_idx, fps),
                                person_id=state.person_id,
                                start_sec=seconds(segment.start_frame, fps),
                                end_sec=seconds(segment.end_frame, fps),
                                duration_sec=duration_sec,
                            )
                            updates.append(best_update)

            if writer is not None:
                draw_overlay(frame, person_states, seen_person_ids, fps, best_update, frame_idx, roi)
                writer.write(frame)

            if frame_idx % max(1, int(fps * 10)) == 0:
                current_best = (
                    f"P{best_update.person_id} {format_seconds(best_update.duration_sec)}"
                    if best_update
                    else "none yet"
                )
                print(f"Frame {frame_idx}/{frame_count}: best={current_best}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    for state in person_states.values():
        if state.in_stationary:
            segment = finalize_stationary_segment(state, state.last_seen_frame)
            if segment is not None:
                duration_sec = seconds(segment.duration_frames, fps)
                if best_update is None or duration_sec > best_update.duration_sec:
                    best_update = BestUpdate(
                        frame=max(frame_idx, state.last_seen_frame),
                        time_sec=seconds(max(frame_idx, state.last_seen_frame), fps),
                        person_id=state.person_id,
                        start_sec=seconds(segment.start_frame, fps),
                        end_sec=seconds(segment.end_frame, fps),
                        duration_sec=duration_sec,
                    )
                    updates.append(best_update)

    output_dir.mkdir(parents=True, exist_ok=True)
    written_video_path = find_written_video_path(output_video_path) if writer is not None else None
    written_frame_strip_path: Path | None = None
    if not args.no_frame_strip and best_update is not None:
        source_video_path = written_video_path
        if source_video_path is None:
            source_video_path = video_path
        written_frame_strip_path = write_longest_stay_frame_strip(
            source_video_path,
            frame_strip_path,
            best_update,
            fps=fps,
            frame_count=frame_count,
            sample_count=args.frame_strip_count,
        )

    written_duration_chart_path: Path | None = None
    if not args.no_duration_chart:
        written_duration_chart_path = write_top_stationary_durations_chart(
            duration_chart_path,
            person_states,
            fps=fps,
            limit=args.duration_chart_count,
            roi=roi,
        )

    written_identity_map_path: Path | None = None
    if not args.no_identity_map:
        written_identity_map_path = write_identity_mapping_chart(
            identity_map_path,
            person_states,
            reid_events,
            fps=fps,
            limit=args.identity_map_count,
            roi=roi,
            highlight_person_id=None if best_update is None else best_update.person_id,
        )

    write_updates_csv(updates_path, updates)
    write_reid_events_csv(reid_events_path, reid_events)
    write_identity_tracks_csv(identity_tracks_path, person_states, fps)
    write_summary_json(
        summary_path,
        video_path=video_path,
        fps=fps,
        frame_count=frame_count,
        best_update=best_update,
        person_states=person_states,
        raw_track_count=len(raw_track_ids_seen),
        reid_events=reid_events,
        args=args,
        roi=roi,
        detected_raw_track_count=len(all_raw_track_ids_seen),
        ignored_outside_roi_raw_track_count=len(all_raw_track_ids_seen - raw_track_ids_seen),
        frame_strip_path=written_frame_strip_path,
        duration_chart_path=written_duration_chart_path,
        identity_map_path=written_identity_map_path,
    )

    print()
    if best_update is None:
        print("No stationary person found.")
    else:
        print("Longest stationary person:")
        best_state = person_states.get(best_update.person_id)
        raw_ids = sorted(best_state.raw_track_ids) if best_state else []
        print(f"  Person ID: {best_update.person_id}")
        print(f"  Raw tracker IDs: {raw_ids}")
        print(f"  Duration: {best_update.duration_sec:.2f}s ({format_seconds(best_update.duration_sec)})")
        print(f"  From: {best_update.start_sec:.2f}s")
        print(f"  To: {best_update.end_sec:.2f}s")
    print(f"Raw tracker IDs analyzed: {len(raw_track_ids_seen)}")
    if roi is not None:
        print(f"Raw tracker IDs detected full-frame: {len(all_raw_track_ids_seen)}")
        print(f"Raw tracker IDs ignored outside ROI: {len(all_raw_track_ids_seen - raw_track_ids_seen)}")
    print(f"Stable person IDs: {len(person_states)}")
    print(f"Relinked raw tracks: {sum(1 for event in reid_events if event.event == 'relinked')}")
    if writer is not None:
        print(f"Annotated video: {written_video_path or output_video_path}")
    else:
        print("Annotated video: skipped")
    if written_frame_strip_path is not None:
        print(f"Longest-stay frame strip: {written_frame_strip_path}")
    elif args.no_frame_strip:
        print("Longest-stay frame strip: skipped")
    else:
        print("Longest-stay frame strip: not generated")
    if written_duration_chart_path is not None:
        print(f"Top stationary durations chart: {written_duration_chart_path}")
    elif args.no_duration_chart:
        print("Top stationary durations chart: skipped")
    else:
        print("Top stationary durations chart: not generated")
    if written_identity_map_path is not None:
        print(f"Stable identity map: {written_identity_map_path}")
    elif args.no_identity_map:
        print("Stable identity map: skipped")
    else:
        print("Stable identity map: not generated")
    print(f"Summary JSON: {summary_path}")
    print(f"Best-update log: {updates_path}")
    print(f"ReID event log: {reid_events_path}")
    print(f"Identity track summary: {identity_tracks_path}")


if __name__ == "__main__":
    run()
