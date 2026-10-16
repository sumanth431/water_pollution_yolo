"""Helper functions shared by the app and the CLI scripts (no torch import needed)."""
from __future__ import annotations

import numpy as np

CLASS_NAMES = ["plastic", "oil_spill", "micro_algae"]

# Alert severity per class: (level, message template)
ALERT_RULES = {
    "oil_spill": ("error", "Oil spill detected ({n} region(s)). Immediate inspection recommended."),
    "micro_algae": ("warning", "Algal bloom detected ({n} region(s)). Check water quality."),
    "plastic": ("warning", "Plastic waste detected ({n} item(s))."),
}


def counts_from_result(result, names) -> dict:
    """Number of detections per class name for one Ultralytics result."""
    counts: dict = {}
    if result.boxes is None or len(result.boxes) == 0:
        return counts
    for c in np.asarray(result.boxes.cls).astype(int).tolist():
        name = names[c]
        counts[name] = counts.get(name, 0) + 1
    return counts


def coverage_from_result(result, names) -> dict:
    """Approximate % of the frame covered per class.

    Uses segmentation masks when available (union per class); otherwise
    the sum of bounding-box areas, clipped to 100 %. Treat as an estimate.
    """
    h, w = result.orig_shape
    cov: dict = {}
    if result.boxes is None or len(result.boxes) == 0:
        return cov
    cls = np.asarray(result.boxes.cls).astype(int)
    if getattr(result, "masks", None) is not None:
        masks = np.asarray(result.masks.data) > 0.5  # (N, mh, mw)
        for c in np.unique(cls):
            union = masks[cls == c].any(axis=0)
            cov[names[c]] = float(union.mean() * 100)
    else:
        xyxy = np.asarray(result.boxes.xyxy)
        for c in np.unique(cls):
            b = xyxy[cls == c]
            area = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).sum()
            cov[names[c]] = float(min(area / (w * h) * 100, 100.0))
    return cov


def build_alerts(counts: dict) -> list:
    """Return a list of (level, message) tuples. Empty counts -> 'success' message."""
    if not counts:
        return [("success", "No pollutants detected in this input.")]
    alerts = []
    for name, n in counts.items():
        level, tmpl = ALERT_RULES.get(name, ("info", "{n} x " + name + " detected."))
        alerts.append((level, tmpl.format(n=n)))
    order = {"error": 0, "warning": 1, "info": 2, "success": 3}
    return sorted(alerts, key=lambda a: order.get(a[0], 9))
