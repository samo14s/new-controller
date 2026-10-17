"""Shared paths and loaders for the consistency audit.

Only reads from ../../src (model) and ../../digitized (vector data recovered
from the published PDF). Nothing outside this folder is written.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "src")
DIG = os.path.join(ROOT, "digitized")
PDF = os.environ.get(
    "OZSOY_PDF",
    "/tmp/claude-0/-home-user-new-controller/d42dfb76-3818-53d8-8bcc-e62ff0a55916/scratchpad/paper.pdf")

if SRC not in sys.path:
    sys.path.insert(0, SRC)


def load_json(name):
    with open(os.path.join(DIG, name)) as fh:
        return json.load(fh)


def curve(name, key):
    """Concatenate the segments of one digitised curve into (x, y) arrays."""
    segs = load_json(name)[key]
    pts = np.array([p for s in segs for p in s], float)
    return pts[:, 0], pts[:, 1]


def dedup(x, y):
    """Drop consecutive duplicate points (polyline joints)."""
    keep = np.r_[True, (np.abs(np.diff(x)) > 1e-9) | (np.abs(np.diff(y)) > 1e-12)]
    return x[keep], y[keep]


def fig8():
    return dedup(*curve("fig8_gains.json", "gain"))


def fig9_controlled():
    return dedup(*curve("fig9_sld.json", "blue"))


def fig9_uncontrolled():
    return dedup(*curve("fig9_sld.json", "blue_dashed"))


def save_json(name, obj):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(os.path.join(HERE, name), "w") as fh:
        json.dump(obj, fh, indent=1, default=conv)


def save_csv(name, header, rows):
    with open(os.path.join(HERE, name), "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(_fmt(v) for v in r) + "\n")


def _fmt(v):
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    try:
        if np.isnan(v):
            return "nan"
    except TypeError:
        pass
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.6g}"
