# Sidhart Sack Counting Project

YOLO-based automated sack counting with custom constant-velocity Hungarian tracker and horizontal two-zone transition validation.

## Features
- **Two-Zone Horizontal Transition Verification**: Tracks must transition between `lower_zone` and `upper_zone` (with confirmed entry hits) before counting to avoid false triggers.
- **Tracklet Merging & De-duplication**: Rebuilds canonical tracks across duplicate detections or split tracklets.
- **Minimal Clean Output**: Suppresses verbose model and inference logs, outputting clean timestamps in Indian Standard Time (IST, UTC+5:30) whenever a sack is counted.
- **Configurable Artifact Generation**: Option to log counts to file or produce annotated video.

## Output Format
When running the pipeline, logs are emitted only when the count increases:
```text
[2026-11-02 16:35:10 IST] Count: 1 (lower_zone -> upper_zone)
[2026-11-02 16:35:14 IST] Count: 2 (lower_zone -> upper_zone)
```

## Running the Script
Ensure dependencies (`ultralytics`, `torch`, `opencv-python`, `scipy`) are installed.
The model checkpoint is located at `model/best.pt`.
```bash
python main.py
```

