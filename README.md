# Longest Stay Detection

This project finds the tracked person who stayed stationary for the longest
duration in `entrance.mov`.

## 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run and Main Parameters

Run the default blue-ROI analysis:

```bash
python longest_stationary.py --video entrance.mov
```

Run with the optimized ByteTrack config from the experiment sweep:

```bash
python longest_stationary.py \
  --video entrance.mov \
  --tracker tracker_configs/bytetrack_long_occlusion.yaml \
  --conf 0.10 \
  --output-dir outputs_best_bytetrack \
  --output-video annotated_entrance_best_bytetrack.mp4
```

By default, analysis is limited to the blue ROI:

```text
x1=916, y1=0, x2=1915, y2=1080
```

To disable the ROI:

```bash
python longest_stationary.py --video entrance.mov --roi none
```

Main parameters:

- `--video`: input video path
- `--model`: YOLO model path, default `yolov8n.pt`
- `--tracker`: ByteTrack YAML config path
- `--conf`: detector confidence threshold before detections are passed to ByteTrack
- `--roi`: analysis ROI as `x1,y1,x2,y2`; use `none` for full-frame analysis
- `--window-seconds`: recent time window used to decide stationary status
- `--smooth-alpha`: position smoothing factor
- `--stationary-ratio`: movement threshold relative to bbox height
- `--min-stationary-px`: minimum movement threshold in pixels
- `--grace-frames`: short tolerance before ending a stationary segment
- `--disable-reid`: turn off appearance-based relinking
- `--reid-score-threshold`: minimum weighted score for merging a new raw tracker ID into an existing stable person ID
- `--reid-min-color-score`: minimum trouser-color histogram similarity for relinking
- `--reid-max-gap-seconds`: maximum time gap allowed for relinking
- `--frame-strip-count`: number of frames in the longest-stay frame strip
- `--duration-chart-count`: number of people shown in the top-duration chart
- `--identity-map-count`: number of merged stable identities shown in the identity map

The script writes:

- `outputs/annotated_entrance.mp4`: video with stable person IDs, raw tracker IDs, and stationary status
- `outputs/summary.json`: final answer and method details
- `outputs/longest_updates.csv`: log whenever a person creates a new longest stationary duration
- `outputs/reid_events.csv`: log of created and relinked identities
- `outputs/identity_tracks.csv`: stable person IDs and raw tracker IDs merged into each one
- `outputs/longest_stay_strip.png`: sampled frame strip from the longest stationary interval
- `outputs/top_stationary_durations.png`: bar chart of the top stationary durations
- `outputs/identity_map.png`: chart showing which raw tracker IDs were merged into each stable person ID

## 3. Methodology

1. Detect and track only `person` objects using Ultralytics YOLO with ByteTrack.
2. Map raw ByteTrack IDs to stable person IDs. When a new raw tracker ID appears,
   compare it with recently lost stable IDs using:
   - bottom-center position continuity
   - lower-body/trouser HSV color histogram similarity
   - bbox-height ratio
   - time gap
3. Keep only tracks whose bottom-center point falls inside the configured ROI.
4. Use the bottom-center point of each bounding box as the approximate ground
   position of the person.
5. Smooth each point using an exponential moving average to reduce detector
   jitter.
6. Over a short recent time window, measure how far the smoothed points spread.
7. Mark a person as stationary when that spread is below:

```text
max(min_stationary_px, stationary_ratio * bbox_height)
```

The bbox-height normalization helps account for perspective: people closer to
the camera appear larger, so the same real-world movement creates larger pixel
movement.

## 4. Strategy to Make Sure the Longest Stationary Stay Is Reliable

The task is not only to detect people, but to keep the same real person linked
through short occlusions, crowding, and irrelevant motion outside the entrance.
The three main reliability strategies are customized ReID, ByteTrack
hyperparameter tuning, and the blue ROI.

### 4.1. Customized ReID

Necessity:

ByteTrack provides raw tracker IDs, but those IDs can change when a person is
briefly lost, occluded, or re-detected. If the same person receives a new raw ID,
their stationary duration may be split into separate fragments.

ID concepts:

- `T<raw_tracker_id>` is the raw ID assigned by ByteTrack to one continuous
  track segment.
- `P<stable_person_id>` is this project's stable person identity after custom
  ReID.
- One stable `P` can contain one or more raw `T` IDs.

Mechanism:

When a new raw `T` appears, the script either creates a new stable `P` or links
that `T` back to a recently lost `P`. The relinking score combines:

- bottom-center position continuity
- lower-body HSV color similarity
- bbox-height ratio
- time gap

Before vs after:

| Stage | Outcome |
| --- | --- |
| Before customized ReID | Same real person can be split into multiple raw `T` IDs after occlusion. |
| After customized ReID | Related raw IDs are merged into one stable `P`, so stationary duration continues under the same person identity. |

Observed outcome in the blue ROI run: `6` raw tracks were relinked and `6`
stable identities contained more than one raw `T` ID. The longest-stay person is
`P9`, which merges raw IDs `T13` and `T19`. When the overlay shows `P9/T19`,
`T19` is the current ByteTrack ID, but the stable person and stationary-duration
score are still `P9`.

| Stable identity map, blue ROI |
| --- |
| <img src="assets/identity_map.png" alt="Stable person IDs mapped to raw tracker IDs in the blue ROI run" width="900"> |

| ReID check frame 548 | ReID check frame 565 | ReID check frame 566 |
| --- | --- | --- |
| <img src="assets/reid_frame_548.jpg" alt="Annotated ReID check frame 548" width="280"> | <img src="assets/reid_frame_565.jpg" alt="Annotated ReID check frame 565" width="280"> | <img src="assets/reid_frame_566.jpg" alt="Annotated ReID check frame 566" width="280"> |

### 4.2. Experimental Tracking of ByteTrack Hyperparameters and Detection Confidence

Run the tracker-parameter sweep:

```bash
python optimize_bytetrack.py --video entrance.mov
```

The sweep writes:

- `experiments/bytetrack_optimization/experiment_results.csv`
- `experiments/bytetrack_optimization/best_experiment.json`
- `experiments/bytetrack_optimization/bytetrack_sweep.png`

Necessity:

The doorway scene has frequent occlusion and crowded motion. The default
ByteTrack settings tend to drop tracks quickly and restart people as new raw
`T` IDs. This increases identity fragmentation and makes the custom ReID layer
work harder.

The sweep image shows each config's detector confidence and ByteTrack confidence
gates:

```text
conf <detector confidence> | <track_high_thresh>/<track_low_thresh>/<new_track_thresh>
```

For example, `conf 0.10 | 0.15/0.03/0.40` means:

- YOLO detections are passed to tracking at detector confidence `0.10`
- detections above `0.15` are strong ByteTrack matches
- detections down to `0.03` can help recover an existing track
- a detection needs `0.40` confidence before it can start a new raw track

| ByteTrack config sweep |
| --- |
| <img src="assets/bytetrack_sweep.png" alt="ByteTrack config sweep comparing confidence thresholds, track buffer, raw tracking IDs, stable person IDs, and longest duration" width="1000"> |

Selected config:

```yaml
# tracker_configs/bytetrack_long_occlusion.yaml
track_high_thresh: 0.15
track_low_thresh: 0.03
new_track_thresh: 0.40
track_buffer: 120
match_thresh: 0.90
```

Before vs after:

| Config | Detector conf | `track_buffer` | Raw tracker IDs | Stable person IDs | Longest duration |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default baseline | `0.30` | `30` | `183` | `150` | `42.4s` |
| Selected long-occlusion config | `0.10` | `120` | `102` | `97` | `42.2s` |

The selected config was chosen by a proxy objective: keep the longest-stay answer
within 95% of the best duration, then minimize tracking fragmentation.

Why `track_buffer` is necessary:

`track_buffer` is the number of frames ByteTrack keeps a lost track alive before
deleting it. The video is about `29.88 FPS`, so:

| `track_buffer` | Approx. lost-track memory |
| --- | --- |
| `30` | `1.0s` |
| `60` | `2.0s` |
| `90` | `3.0s` |
| `120` | `4.0s` |

A short buffer drops the track quickly, so a briefly hidden person can return as
a new raw `T` ID. A longer buffer gives ByteTrack more time to reconnect the same
person after occlusion.

Why confidence scores are necessary:

High detector confidence, such as `--conf 0.30`, can miss weak person detections
during blur or occlusion. Lower detector confidence, such as `--conf 0.10`,
combined with `track_low_thresh: 0.03`, gives ByteTrack weak detections that can
bridge short gaps. To prevent too many weak detections from starting false new
tracks, the selected config keeps `new_track_thresh: 0.40`.

Tradeoff:

An overly long buffer or overly loose confidence gate can reconnect the wrong
people in a crowd. The selected config balances this with `match_thresh: 0.90`
and the blue ROI, which reduces irrelevant candidates.

### 4.3. Blue ROI

Run the ROI benchmark:

```bash
python benchmark_roi.py --video entrance.mov --keep-videos
```

The benchmark writes:

- `experiments/roi_benchmark/benchmark_results.csv`
- `experiments/roi_benchmark/benchmark_summary.json`
- `experiments/roi_benchmark/benchmark_roi.png`
- `experiments/roi_benchmark/benchmark_roi_comparison.png`

Necessity:

The full frame contains people outside the entrance area who are not relevant to
the longest-stay question. Those extra detections increase tracker noise and
identity fragmentation. The tracker still runs on the full frame, but only
detections whose bottom-center ground point falls inside the blue ROI are used
for ReID and stationary-duration scoring.

Before vs after:

| Analysis area | Raw tracker IDs analyzed | Stable person IDs | Longest duration |
| --- | ---: | ---: | ---: |
| Full frame | `102` | `97` | `42.2s` |
| Blue ROI | `68` | `62` | `42.2s` |

Outcome:

The blue ROI keeps the same longest stationary duration while reducing analyzed
raw tracker IDs from `102` to `68` and stable person IDs from `97` to `62`.

| Full-frame vs Blue ROI |
| --- |
| <img src="assets/roi_comparison.png" alt="Side-by-side comparison of full-frame tracking and blue ROI tracking" width="1000"> |

| ROI benchmark |
| --- |
| <img src="assets/benchmark_roi.png" alt="ROI benchmark comparing full-frame and blue ROI tracking metrics" width="900"> |

## 5. Result: Who Stayed Stationary the Longest?

For the blue ROI run, the longest stationary stay is:

```text
Stable person ID: P9
Raw tracker IDs: T13, T19
Start: 32.728s
End: 74.893s
Duration: 42.199s (42.2s)
```

The strip below samples the beginning, middle, and end of that interval.

| Longest-stay frame strip, blue ROI |
| --- |
| <img src="assets/longest_stay_strip.png" alt="Sampled frames across the longest stationary interval in the blue ROI run" width="1000"> |

The top-duration chart shows that `P9` is clearly the longest stationary person.

| Top stationary durations, blue ROI |
| --- |
| <img src="assets/top_stationary_durations.png" alt="Top stationary durations bar chart for the blue ROI run" width="900"> |

| Blue ROI result frame |
| --- |
| <img src="assets/blue_roi_result.png" alt="Blue ROI annotated result frame with longest stationary person" width="800"> |

The checked-in images under `assets/` are copies of generated artifacts. To
refresh them, rerun the corresponding command above and copy the regenerated
image into `assets/`.

## 6. Limitations

- The custom ReID logic reduces tracker ID switches, but it can still fail when
  people have similar trousers, heavy occlusion, or strong lighting changes.
- Stationary movement is measured in image pixels, not real-world meters.
- The result depends on camera perspective, detection quality, threshold
  settings, and video resolution.
