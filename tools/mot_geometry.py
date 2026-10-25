"""Conservative MOT geometry validation; never infer new trajectories."""
from __future__ import annotations

import math


def parse_gt(line):
    row = [float(value) for value in line.strip().split(",")]
    if len(row) != 9 or not all(math.isfinite(value) for value in row):
        raise ValueError("GT rows require nine finite values")
    frame, identity, _, _, _, _, mark, category, visibility = row
    if frame < 1 or identity < 1 or frame != int(frame) or identity != int(identity):
        raise ValueError("Invalid frame or identity")
    if mark not in (0, 1) or category != 1 or not 0 <= visibility <= 1:
        raise ValueError("Unexpected validity/class/visibility")
    return row


def clip_small_overflow(row, width, height, max_pixels=5.0, min_retained_area=0.90):
    _, _, x, y, w, h, *_ = row
    if w <= 0 or h <= 0:
        raise ValueError("Nonpositive boxes require a separate documented exclusion policy")
    overflow = max(0, -x, -y, x+w-width, y+h-height)
    if overflow > max_pixels:
        raise ValueError("Overflow exceeds conservative clipping limit")
    x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x+w), min(height, y+h)
    retained = (x2-x1)*(y2-y1)/(w*h)
    if x2 <= x1 or y2 <= y1 or retained < min_retained_area:
        raise ValueError("Clipping would remove too much target area")
    result = list(row)
    if overflow > 1e-6:
        result[2:6] = [x1, y1, x2-x1, y2-y1]
    return result, overflow > 1e-6


def validate_gt(lines, width, height, frames):
    seen, rows = set(), []
    for line in lines:
        if not line.strip():
            continue
        row = parse_gt(line)
        key = tuple(row[:2])
        if key in seen or row[0] > frames:
            raise ValueError("Duplicate frame-ID or frame outside video")
        seen.add(key)
        _, overflow = clip_small_overflow(row, width, height, max_pixels=0.000001)
        if overflow:
            raise ValueError("GT not inside image")
        rows.append(row)
    if not rows:
        raise ValueError("Empty GT")
    return rows
