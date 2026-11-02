# ================================================================
# WHITE_IMG_6360: DIRECT VIDEO -> YOLO -> TRACKING -> HORIZONTAL ZONE COUNTING
# Google Colab ready | Full-width upper/lower zones | Max 5 FPS
# ================================================================

import os
import math
import time
import logging
import warnings
from collections import deque
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# Suppress external library and YOLO logging
warnings.filterwarnings('ignore')
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Indian Standard Time (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

# Automatically mount Google Drive when running inside Colab.
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

# ================================================================
# 1. CONFIGURATION
# ================================================================

VIDEO_PATH = (
    '/content/drive/MyDrive/Amrendra_Guardex/sidhart_sack_counting/'
    'input_videos_training/White_IMG_6360.MP4'
)

MODEL_PATH = (
    '/content/drive/MyDrive/Amrendra_Guardex/sidhart_sack_counting/'
    'ultralytics dataset/yolo_dataset/dataset/runs/'
    'throwing_box_two_class/weights/best.pt'
)

OUTPUT_DIR = (
    '/content/drive/MyDrive/Amrendra_Guardex/sidhart_sack_counting/'
    'input_videos_training/White_IMG_6360_horizontal_zone_counting_output'
)

# Inference sampling: never process more than 5 FPS.
TARGET_FPS = 5.0
IMGSZ = 1280
YOLO_CONF = 0.20
YOLO_IOU = 0.50
MAX_DET = 300
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
USE_HALF = bool(torch.cuda.is_available())

# This JSON shows that ID 1 (old_class) is the dominant, high-confidence
# class in White_IMG_6360. Keep ID 0 excluded for this first experiment.
COUNT_CLASS_IDS = {0,1}

# ---------------------------------------------------------------
# HORIZONTAL FULL-WIDTH ZONES:
# Lower zone spans the full video width, from 52% to 100% of image height.
# Upper zone spans the full video width, from 0% to 46% of image height.
# The 6% vertical gap between them is a neutral transit band. A track must
# be confirmed in both zones before it can count. Coordinates are normalized
# so they scale to the actual input video dimensions.
# ---------------------------------------------------------------
ZONE_A_NAME = 'lower_zone'
ZONE_B_NAME = 'upper_zone'

ZONE_A_NORMALIZED = [
    (0.00, 0.52),
    (1.00, 0.52),
    (1.00, 1.00),
    (0.00, 1.00),
]

ZONE_B_NORMALIZED = [
    (0.00, 0.00),
    (1.00, 0.00),
    (1.00, 0.46),
    (0.00, 0.46),
]

# Require two consecutive sampled observations inside a zone before
# accepting a zone entry. This reduces brief centroid-jitter events.
ZONE_CONFIRM_HITS = 2

# Track quality gates before accepting a zone-to-zone journey.
TRACK_CONF_MIN = 0.20
NEW_TRACK_CONF_MIN = 0.35
NMS_IOU_THRESHOLD = 0.50
MIN_TRACK_HITS_TO_COUNT = 4
MIN_TRACK_AGE_SEC_TO_COUNT = 0.40
MIN_TRACK_MEAN_CONF_TO_COUNT = 0.35
COUNT_ONCE_PER_TRACK = True

# Custom constant-velocity + Hungarian assignment tracker.
MAX_TRACK_AGE_SEC = 1.60
BASE_MATCH_GATE_PX = 220.0
MATCH_GATE_GROWTH_PX_PER_SEC = 220.0
MAX_MATCH_GATE_PX = 850.0
MAX_ASSOCIATION_COST = 0.88

# Tracklet de-duplication / identity recovery.
# Two active IDs are considered duplicate fragments only when their boxes
# overlap substantially for several sampled frames, their relative centre
# offset remains stable, and their box dimensions are comparable.
ENABLE_DUPLICATE_TRACKLET_MERGING = True
DUPLICATE_PAIR_MIN_OBSERVATIONS = 3
DUPLICATE_PAIR_MAX_GAP_SEC = 0.45       # allows one missed 5-FPS sample
DUPLICATE_PAIR_MIN_INTERSECTION_SMALLER = 0.62
DUPLICATE_PAIR_MAX_AREA_RATIO = 1.35
DUPLICATE_PAIR_MAX_WIDTH_RATIO = 1.25
DUPLICATE_PAIR_MAX_HEIGHT_RATIO = 1.25
DUPLICATE_PAIR_MAX_DX_FRAC = 0.10
DUPLICATE_PAIR_MAX_DY_FRAC = 0.38
DUPLICATE_PAIR_MAX_OFFSET_STD_X = 45.0
DUPLICATE_PAIR_MAX_OFFSET_STD_Y = 45.0

# Output settings: Log count in IST only when count increases.
SAVE_COUNT_LOG_FILE = True

# ================================================================
# 2. GEOMETRY / DETECTION HELPERS
# ================================================================

def box_iou_xyxy(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_area(box):
    return max(1.0, (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])))


def box_intersection_over_smaller_area(a, b):
    """Overlap relative to the smaller box; useful when boxes are offset."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max(1.0, min(box_area(a), box_area(b)))


def make_zone_polygon(normalized_points, width, height):
    pts = []
    for x_frac, y_frac in normalized_points:
        x = int(round(np.clip(x_frac, 0.0, 1.0) * (width - 1)))
        y = int(round(np.clip(y_frac, 0.0, 1.0) * (height - 1)))
        pts.append([x, y])
    return np.asarray(pts, dtype=np.int32)


def point_in_polygon(point_xy, polygon):
    x, y = float(point_xy[0]), float(point_xy[1])
    return cv2.pointPolygonTest(
        polygon,
        (x, y),
        False
    ) >= 0


def get_zone_name(point_xy, zone_a, zone_b):
    in_a = point_in_polygon(point_xy, zone_a)
    in_b = point_in_polygon(point_xy, zone_b)
    if in_a and not in_b:
        return ZONE_A_NAME
    if in_b and not in_a:
        return ZONE_B_NAME
    # The configured polygons do not overlap; treat any ambiguous point
    # as outside both zones rather than creating an uncertain transition.
    return None


def nms_frame(detections):
    """Class-aware NMS to suppress repeated overlapping boxes."""
    candidates = []
    for d in detections:
        class_id = int(d['class_id'])
        conf = float(d['confidence'])
        if COUNT_CLASS_IDS is not None and class_id not in COUNT_CLASS_IDS:
            continue
        if conf < TRACK_CONF_MIN:
            continue
        bbox = np.asarray(d['bbox_xyxy'], dtype=np.float64)
        if bbox.shape != (4,) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        item = dict(d)
        item['bbox_xyxy'] = bbox.tolist()
        item['center_xy'] = [float((bbox[0] + bbox[2]) / 2.0), float((bbox[1] + bbox[3]) / 2.0)]
        item['width'] = float(bbox[2] - bbox[0])
        item['height'] = float(bbox[3] - bbox[1])
        candidates.append(item)

    candidates.sort(key=lambda z: z['confidence'], reverse=True)
    kept = []
    for det in candidates:
        if any(
            int(det['class_id']) == int(chosen['class_id'])
            and box_iou_xyxy(det['bbox_xyxy'], chosen['bbox_xyxy']) > NMS_IOU_THRESHOLD
            for chosen in kept
        ):
            continue
        kept.append(det)
    return kept

# ================================================================
# 3. TRACK STATE + ZONE-TRANSITION COUNTING
# ================================================================

class Track:
    def __init__(self, track_id, det, timestamp, zone_a, zone_b):
        self.track_id = int(track_id)
        self.class_id = int(det['class_id'])
        self.bbox = np.asarray(det['bbox_xyxy'], dtype=np.float64)
        self.center = np.asarray(det['center_xy'], dtype=np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.first_seen = float(timestamp)
        self.last_seen = float(timestamp)
        self.hits = 1
        self.misses = 0
        self.confidences = [float(det['confidence'])]
        self.path = deque(maxlen=100)
        self.path.append({
            'timestamp_sec': float(timestamp),
            'center_x': float(self.center[0]),
            'center_y': float(self.center[1]),
            'confidence': float(det['confidence']),
            'bbox_xyxy': [float(v) for v in self.bbox]
        })

        self.merged_into = None
        self.merged_track_ids = []

        # Zone state machine. A zone must be occupied for N consecutive
        # detections before it becomes the confirmed zone for this track.
        self.candidate_zone = None
        self.candidate_zone_hits = 0
        self.confirmed_zone = None
        self.confirmed_zone_entered_at = None
        self.confirmed_zone_point = None

        self.counted = False
        self.count_event = None
        self.last_event = self._update_zone_state(timestamp, zone_a, zone_b)

    @property
    def age_sec(self):
        return max(0.0, self.last_seen - self.first_seen)

    @property
    def mean_confidence(self):
        return float(np.mean(self.confidences)) if self.confidences else 0.0

    def predict_center(self, timestamp):
        dt = max(0.0, float(timestamp) - self.last_seen)
        return self.center + self.velocity * dt

    def update(self, det, timestamp, zone_a, zone_b):
        timestamp = float(timestamp)
        new_center = np.asarray(det['center_xy'], dtype=np.float64)
        dt = max(timestamp - self.last_seen, 1e-4)
        measured_velocity = (new_center - self.center) / dt
        self.velocity = 0.50 * self.velocity + 0.50 * measured_velocity
        self.center = new_center
        self.bbox = np.asarray(det['bbox_xyxy'], dtype=np.float64)
        self.last_seen = timestamp
        self.hits += 1
        self.misses = 0
        self.confidences.append(float(det['confidence']))
        self.path.append({
            'timestamp_sec': timestamp,
            'center_x': float(new_center[0]),
            'center_y': float(new_center[1]),
            'confidence': float(det['confidence']),
            'bbox_xyxy': [float(v) for v in self.bbox]
        })
        self.last_event = self._update_zone_state(timestamp, zone_a, zone_b)
        return self.last_event

    def _quality_passes(self):
        return (
            self.hits >= MIN_TRACK_HITS_TO_COUNT
            and self.age_sec >= MIN_TRACK_AGE_SEC_TO_COUNT
            and self.mean_confidence >= MIN_TRACK_MEAN_CONF_TO_COUNT
        )

    def _update_zone_state(self, timestamp, zone_a, zone_b):
        current_zone = get_zone_name(self.center, zone_a, zone_b)
        if current_zone is None:
            # Preserve confirmed_zone while the object travels between zones.
            self.candidate_zone = None
            self.candidate_zone_hits = 0
            return None

        if current_zone == self.candidate_zone:
            self.candidate_zone_hits += 1
        else:
            self.candidate_zone = current_zone
            self.candidate_zone_hits = 1

        if self.candidate_zone_hits < ZONE_CONFIRM_HITS:
            return None

        # First confirmed zone establishes the starting zone; it does not count.
        if self.confirmed_zone is None:
            self.confirmed_zone = current_zone
            self.confirmed_zone_entered_at = float(timestamp)
            self.confirmed_zone_point = self.center.copy()
            return None

        # Remaining in the same zone does not create a new event.
        if current_zone == self.confirmed_zone:
            return None

        from_zone = self.confirmed_zone
        to_zone = current_zone
        source_entered_at = self.confirmed_zone_entered_at
        source_point = self.confirmed_zone_point.copy() if self.confirmed_zone_point is not None else None

        # Advance zone state even if quality gates fail, so stale transitions
        # cannot fire later after the track has moved somewhere else.
        self.confirmed_zone = to_zone
        self.confirmed_zone_entered_at = float(timestamp)
        self.confirmed_zone_point = self.center.copy()

        if (COUNT_ONCE_PER_TRACK and self.counted) or not self._quality_passes():
            return None

        if from_zone == ZONE_A_NAME and to_zone == ZONE_B_NAME:
            direction = 'zone_a_to_zone_b'
        elif from_zone == ZONE_B_NAME and to_zone == ZONE_A_NAME:
            direction = 'zone_b_to_zone_a'
        else:
            return None

        self.counted = True
        event = {
            'track_id': self.track_id,
            'class_id': self.class_id,
            'class_name': 'old_class' if self.class_id == 1 else str(self.class_id),
            'direction': direction,
            'from_zone': from_zone,
            'to_zone': to_zone,
            'timestamp_sec': round(float(timestamp), 6),
            'zone_entry_point_xy': [round(float(self.center[0]), 2), round(float(self.center[1]), 2)],
            'previous_zone_entry_sec': round(float(source_entered_at), 6) if source_entered_at is not None else None,
            'previous_zone_entry_point_xy': [round(float(v), 2) for v in source_point] if source_point is not None else None,
            'track_hits': int(self.hits),
            'track_age_sec': round(self.age_sec, 3),
            'mean_confidence': round(self.mean_confidence, 4),
            'first_seen_sec': round(self.first_seen, 6),
            'last_seen_sec': round(self.last_seen, 6)
        }
        self.count_event = event
        return event

    def rebuild_from_merged_samples(self, samples, zone_a, zone_b):
        """Rebuild one canonical track from two duplicate track histories."""
        samples = sorted(samples, key=lambda z: z['timestamp_sec'])
        self.path = deque(maxlen=100)
        self.confidences = []
        self.first_seen = float(samples[0]['timestamp_sec'])
        self.last_seen = float(samples[0]['timestamp_sec'])
        self.hits = 0
        self.misses = 0
        self.counted = False
        self.count_event = None
        self.candidate_zone = None
        self.candidate_zone_hits = 0
        self.confirmed_zone = None
        self.confirmed_zone_entered_at = None
        self.confirmed_zone_point = None
        self.last_event = None

        latest_event = None
        for sample in samples:
            ts = float(sample['timestamp_sec'])
            bbox = np.asarray(sample['bbox_xyxy'], dtype=np.float64)
            center = np.asarray(sample['center_xy'], dtype=np.float64)
            conf = float(sample['confidence'])

            self.hits += 1
            self.last_seen = ts
            self.confidences.append(conf)
            self.bbox = bbox
            self.center = center
            self.path.append({
                'timestamp_sec': ts,
                'center_x': float(center[0]),
                'center_y': float(center[1]),
                'confidence': conf,
                'bbox_xyxy': [float(v) for v in bbox]
            })
            event = self._update_zone_state(ts, zone_a, zone_b)
            if event is not None:
                latest_event = event
                self.count_event = event

        if len(self.path) >= 2:
            p0, p1 = self.path[-2], self.path[-1]
            dt = max(float(p1['timestamp_sec']) - float(p0['timestamp_sec']), 1e-4)
            self.velocity = np.asarray([
                (float(p1['center_x']) - float(p0['center_x'])) / dt,
                (float(p1['center_y']) - float(p0['center_y'])) / dt
            ], dtype=np.float64)
        else:
            self.velocity = np.zeros(2, dtype=np.float64)
        return latest_event

# ================================================================
# 4. CUSTOM CONSTANT-VELOCITY + HUNGARIAN TRACKER
# ================================================================

class ZoneTracker:
    def __init__(self, zone_a, zone_b):
        self.zone_a = zone_a
        self.zone_b = zone_b
        self.tracks = {}
        self.next_track_id = 1
        self.events = []
        self.zone_a_to_b_count = 0
        self.zone_b_to_a_count = 0
        self.duplicate_pair_history = {}
        self.track_merge_events = []
        self.last_track_merges = []
        self.track_aliases = {}

    def resolve_track_id(self, track_id):
        if track_id is None:
            return None
        tid = int(track_id)
        seen = set()
        while tid in self.track_aliases and tid not in seen:
            seen.add(tid)
            tid = int(self.track_aliases[tid])
        return tid

    def _association_cost(self, track, det, timestamp):
        dt = max(0.0, float(timestamp) - track.last_seen)
        predicted = track.predict_center(timestamp)
        det_center = np.asarray(det['center_xy'], dtype=np.float64)
        distance = float(np.linalg.norm(det_center - predicted))

        det_diag = math.sqrt(max(float(det['width']) * float(det['height']), 1.0))
        gate = max(BASE_MATCH_GATE_PX, 0.65 * det_diag) + MATCH_GATE_GROWTH_PX_PER_SEC * dt
        gate = min(gate, MAX_MATCH_GATE_PX)
        if distance > gate:
            return 1e6

        overlap = box_iou_xyxy(track.bbox, det['bbox_xyxy'])
        old_area = box_area(track.bbox)
        new_area = box_area(det['bbox_xyxy'])
        scale_change = abs(math.log(max(new_area, 1.0) / max(old_area, 1.0)))
        cost = (
            0.55 * (distance / max(gate, 1.0))
            + 0.30 * (1.0 - overlap)
            + 0.15 * (min(scale_change, 2.0) / 2.0)
        )
        return float(cost)

    def _duplicate_pair_metrics(self, det_a, det_b):
        if int(det_a['class_id']) != int(det_b['class_id']):
            return None
        a = np.asarray(det_a['bbox_xyxy'], dtype=np.float64)
        b = np.asarray(det_b['bbox_xyxy'], dtype=np.float64)
        ios = box_intersection_over_smaller_area(a, b)
        area_ratio = max(box_area(a), box_area(b)) / min(box_area(a), box_area(b))
        width_a, width_b = a[2] - a[0], b[2] - b[0]
        height_a, height_b = a[3] - a[1], b[3] - b[1]
        width_ratio = max(width_a, width_b) / max(1.0, min(width_a, width_b))
        height_ratio = max(height_a, height_b) / max(1.0, min(height_a, height_b))
        ca = np.asarray(det_a['center_xy'], dtype=np.float64)
        cb = np.asarray(det_b['center_xy'], dtype=np.float64)
        dx = float(ca[0] - cb[0])
        dy = float(ca[1] - cb[1])
        max_width = max(width_a, width_b)
        max_height = max(height_a, height_b)

        qualifies = (
            ios >= DUPLICATE_PAIR_MIN_INTERSECTION_SMALLER
            and area_ratio <= DUPLICATE_PAIR_MAX_AREA_RATIO
            and width_ratio <= DUPLICATE_PAIR_MAX_WIDTH_RATIO
            and height_ratio <= DUPLICATE_PAIR_MAX_HEIGHT_RATIO
            and abs(dx) <= DUPLICATE_PAIR_MAX_DX_FRAC * max_width
            and abs(dy) <= DUPLICATE_PAIR_MAX_DY_FRAC * max_height
        )
        if not qualifies:
            return None
        return {
            'ios': float(ios),
            'area_ratio': float(area_ratio),
            'dx': dx,
            'dy': dy,
            'width_ratio': float(width_ratio),
            'height_ratio': float(height_ratio)
        }

    @staticmethod
    def _fuse_same_timestamp_samples(samples):
        """Confidence-weighted box fusion for duplicate observations at one time."""
        if len(samples) == 1:
            return dict(samples[0])
        weights = np.asarray([max(float(s['confidence']), 0.01) for s in samples], dtype=np.float64)
        boxes = np.asarray([s['bbox_xyxy'] for s in samples], dtype=np.float64)
        fused_box = np.average(boxes, axis=0, weights=weights)
        center = [(fused_box[0] + fused_box[2]) / 2.0, (fused_box[1] + fused_box[3]) / 2.0]
        return {
            'timestamp_sec': float(samples[0]['timestamp_sec']),
            'center_x': float(center[0]),
            'center_y': float(center[1]),
            'center_xy': [float(center[0]), float(center[1])],
            'confidence': float(max(float(s['confidence']) for s in samples)),
            'bbox_xyxy': [float(v) for v in fused_box]
        }

    def _make_merged_samples(self, primary, secondary):
        by_time = {}
        for tr in (primary, secondary):
            for sample in tr.path:
                if 'bbox_xyxy' not in sample:
                    continue
                key = round(float(sample['timestamp_sec']), 6)
                by_time.setdefault(key, []).append({
                    'timestamp_sec': float(sample['timestamp_sec']),
                    'center_x': float(sample['center_x']),
                    'center_y': float(sample['center_y']),
                    'center_xy': [float(sample['center_x']), float(sample['center_y'])],
                    'confidence': float(sample['confidence']),
                    'bbox_xyxy': [float(v) for v in sample['bbox_xyxy']]
                })
        merged = [self._fuse_same_timestamp_samples(v) for _, v in sorted(by_time.items())]
        return merged

    def _rebuild_event_counters(self):
        self.events.sort(key=lambda e: float(e['timestamp_sec']))
        self.zone_a_to_b_count = sum(1 for e in self.events if e['direction'] == 'zone_a_to_zone_b')
        self.zone_b_to_a_count = sum(1 for e in self.events if e['direction'] == 'zone_b_to_zone_a')

    def _merge_tracklets(self, track_id_a, track_id_b, observations, timestamp):
        ta = self.tracks[self.resolve_track_id(track_id_a)]
        tb = self.tracks[self.resolve_track_id(track_id_b)]
        if ta.track_id == tb.track_id or ta.merged_into is not None or tb.merged_into is not None:
            return None

        # Keep the earliest-created/first-seen ID as the canonical identity.
        primary, secondary = sorted((ta, tb), key=lambda tr: (tr.first_seen, tr.track_id))
        old_ids = {primary.track_id, secondary.track_id}
        merged_samples = self._make_merged_samples(primary, secondary)
        if len(merged_samples) < 2:
            return None

        # Remove any provisional events from either split fragment, then replay
        # the combined trajectory through the zone state machine exactly once.
        self.events = [e for e in self.events if int(e['track_id']) not in old_ids]
        event = primary.rebuild_from_merged_samples(merged_samples, self.zone_a, self.zone_b)
        primary.merged_track_ids.extend([secondary.track_id] + list(secondary.merged_track_ids))
        secondary.merged_into = primary.track_id
        secondary.counted = False
        secondary.count_event = None
        self.track_aliases[secondary.track_id] = primary.track_id
        for nested_id in secondary.merged_track_ids:
            self.track_aliases[int(nested_id)] = primary.track_id

        if event is not None:
            self.events.append(event)
        self._rebuild_event_counters()

        dx_values = [float(o['dx']) for o in observations]
        dy_values = [float(o['dy']) for o in observations]
        merge_record = {
            'timestamp_sec': round(float(timestamp), 6),
            'primary_track_id': int(primary.track_id),
            'merged_track_id': int(secondary.track_id),
            'merged_track_ids': list(primary.merged_track_ids),
            'duplicate_observations': len(observations),
            'mean_intersection_over_smaller_box': round(float(np.mean([o['ios'] for o in observations])), 4),
            'relative_center_offset_std_xy': [round(float(np.std(dx_values)), 2), round(float(np.std(dy_values)), 2)],
            'reason': 'repeated high-overlap detections with stable relative position and similar box dimensions'
        }
        self.track_merge_events.append(merge_record)
        self.last_track_merges.append(merge_record)
        return primary.track_id

    def _check_and_merge_duplicate_tracks(self, detections, detection_to_track, timestamp):
        if not ENABLE_DUPLICATE_TRACKLET_MERGING:
            return

        self.last_track_merges = []
        current = []
        for det in detections:
            det_index = int(det.get('detection_index', -1))
            tid = detection_to_track.get(det_index)
            if tid is None:
                continue
            tid = self.resolve_track_id(tid)
            if tid not in self.tracks or self.tracks[tid].merged_into is not None:
                continue
            current.append((tid, det))
        # Keep pair ordering deterministic so relative-offset stability is
        # measured in the same ID order even when detector confidence order flips.
        current.sort(key=lambda item: item[0])

        seen_pairs = set()
        for i in range(len(current)):
            tid_a, det_a = current[i]
            for j in range(i + 1, len(current)):
                tid_b, det_b = current[j]
                tid_a = self.resolve_track_id(tid_a)
                tid_b = self.resolve_track_id(tid_b)
                if tid_a == tid_b:
                    continue
                pair = tuple(sorted((int(tid_a), int(tid_b))))
                metrics = self._duplicate_pair_metrics(det_a, det_b)
                if metrics is None:
                    continue
                seen_pairs.add(pair)
                history = self.duplicate_pair_history.get(pair, [])
                if history and float(timestamp) - float(history[-1]['timestamp_sec']) > DUPLICATE_PAIR_MAX_GAP_SEC:
                    history = []
                history.append({'timestamp_sec': float(timestamp), **metrics})
                history = history[-DUPLICATE_PAIR_MIN_OBSERVATIONS:]
                self.duplicate_pair_history[pair] = history

                if len(history) < DUPLICATE_PAIR_MIN_OBSERVATIONS:
                    continue
                dx_std = float(np.std([o['dx'] for o in history]))
                dy_std = float(np.std([o['dy'] for o in history]))
                if dx_std <= DUPLICATE_PAIR_MAX_OFFSET_STD_X and dy_std <= DUPLICATE_PAIR_MAX_OFFSET_STD_Y:
                    canonical = self._merge_tracklets(pair[0], pair[1], history, timestamp)
                    if canonical is not None:
                        for det_index, mapped_tid in list(detection_to_track.items()):
                            detection_to_track[det_index] = self.resolve_track_id(mapped_tid)

        # Remove stale pair signatures so old IDs cannot be merged much later.
        stale_after = max(DUPLICATE_PAIR_MAX_GAP_SEC * 2.0, 0.9)
        for pair in list(self.duplicate_pair_history):
            if float(timestamp) - float(self.duplicate_pair_history[pair][-1]['timestamp_sec']) > stale_after:
                del self.duplicate_pair_history[pair]

    def update(self, detections, timestamp):
        timestamp = float(timestamp)
        self.last_track_merges = []
        active_tracks = [
            tr for tr in self.tracks.values()
            if tr.merged_into is None and 0.0 <= timestamp - tr.last_seen <= MAX_TRACK_AGE_SEC
        ]
        matched_cols = set()
        detection_to_track = {}
        new_events = []

        if active_tracks and detections:
            cost_matrix = np.full((len(active_tracks), len(detections)), 1e6, dtype=np.float64)
            for i, track in enumerate(active_tracks):
                for j, det in enumerate(detections):
                    if track.class_id != int(det['class_id']):
                        continue
                    cost_matrix[i, j] = self._association_cost(track, det, timestamp)

            rows, cols = linear_sum_assignment(cost_matrix)
            for row, col in zip(rows, cols):
                cost = float(cost_matrix[row, col])
                if cost > MAX_ASSOCIATION_COST:
                    continue
                track = active_tracks[row]
                det = detections[col]
                event = track.update(det, timestamp, self.zone_a, self.zone_b)
                detection_to_track[int(det.get('detection_index', col))] = track.track_id
                matched_cols.add(col)
                if event is not None:
                    self.events.append(event)
                    new_events.append(event)

        for j, det in enumerate(detections):
            if j in matched_cols:
                continue
            if float(det['confidence']) < NEW_TRACK_CONF_MIN:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            tr = Track(track_id, det, timestamp, self.zone_a, self.zone_b)
            self.tracks[track_id] = tr
            detection_to_track[int(det.get('detection_index', j))] = track_id

        self._rebuild_event_counters()
        self._check_and_merge_duplicate_tracks(detections, detection_to_track, timestamp)
        # Return only canonical IDs for detections in this frame.
        for det_index, tid in list(detection_to_track.items()):
            detection_to_track[det_index] = self.resolve_track_id(tid)
        new_events = [e for e in self.events if abs(float(e['timestamp_sec']) - timestamp) < 1e-5]
        return detection_to_track, new_events

# ================================================================
# 5. DIRECT VIDEO INFERENCE & IST COUNT LOGGING
# ================================================================

def main():
    if not os.path.isfile(VIDEO_PATH):
        raise FileNotFoundError(f'Input video not found: {VIDEO_PATH}')
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f'YOLO model not found: {MODEL_PATH}')

    video_stem = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    count_log_path = os.path.join(OUTPUT_DIR, f'{video_stem}_count_log_ist.txt')

    log_file = None
    if SAVE_COUNT_LOG_FILE:
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            log_file = open(count_log_path, 'w', encoding='utf-8')
        except Exception:
            log_file = None

    model = YOLO(MODEL_PATH)
    names = model.names

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {VIDEO_PATH}')

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if source_fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError('Could not read source FPS or video dimensions.')

    process_every_frame = source_fps <= TARGET_FPS

    zone_a = make_zone_polygon(ZONE_A_NORMALIZED, width, height)
    zone_b = make_zone_polygon(ZONE_B_NORMALIZED, width, height)
    tracker = ZoneTracker(zone_a, zone_b)

    decoded_index = 0
    sample_index = 0
    next_sample_index = 0
    previous_total_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if process_every_frame or decoded_index >= next_sample_index:
                timestamp = decoded_index / source_fps
                results = model.predict(
                    source=frame,
                    imgsz=IMGSZ,
                    conf=YOLO_CONF,
                    iou=YOLO_IOU,
                    max_det=MAX_DET,
                    classes=sorted(COUNT_CLASS_IDS) if COUNT_CLASS_IDS is not None else None,
                    device=DEVICE,
                    half=USE_HALF,
                    verbose=False,
                    save=False,
                    stream=False
                )
                result = results[0]
                raw_dets = []

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.detach().cpu().numpy()
                    confs = result.boxes.conf.detach().cpu().numpy()
                    clss = result.boxes.cls.detach().cpu().numpy()
                    for di, (box, conf, cls_value) in enumerate(zip(boxes, confs, clss)):
                        cid = int(cls_value)
                        if COUNT_CLASS_IDS is not None and cid not in COUNT_CLASS_IDS:
                            continue
                        x1, y1, x2, y2 = [float(v) for v in box]
                        if x2 <= x1 or y2 <= y1:
                            continue
                        class_name = names.get(cid, str(cid)) if isinstance(names, dict) else names[cid]
                        raw_dets.append({
                            'detection_index': di,
                            'class_id': cid,
                            'class_name': str(class_name),
                            'confidence': float(conf),
                            'bbox_xyxy': [x1, y1, x2, y2],
                            'center_xy': [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                            'width': x2 - x1,
                            'height': y2 - y1
                        })

                cleaned = nms_frame(raw_dets)
                tracker.update(cleaned, timestamp)

                # Keep log in IST only when count increases
                current_total_count = tracker.zone_a_to_b_count + tracker.zone_b_to_a_count
                if current_total_count > previous_total_count:
                    diff = current_total_count - previous_total_count
                    for i in range(diff):
                        cnt = previous_total_count + i + 1
                        now_ist = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')
                        log_line = f"[{now_ist}] Count: {cnt}"
                        print(log_line, flush=True)
                        if log_file is not None:
                            log_file.write(log_line + '\n')
                            log_file.flush()
                    previous_total_count = current_total_count

                sample_index += 1
                if not process_every_frame:
                    next_sample_index = int(round(sample_index * source_fps / TARGET_FPS))

            decoded_index += 1
    finally:
        cap.release()
        if log_file is not None:
            log_file.close()


if __name__ == '__main__':
    main()
