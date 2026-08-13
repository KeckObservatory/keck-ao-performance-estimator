#!/usr/bin/env python3
"""Measured-SR field map: rejected-star (×) markers no longer persist
across frame changes (2026-07-26, Eduardo bug report).

`_on_nirc2_frame_done` reset `_n2_field` when a new frame loaded ("the
map belongs to ONE frame") but never reset `_n2_field_dropped` or the
selection (`_n2_sel_star`/`_n2_sel_dropped`) -- the previous frame's ×
pool (and any selection ring) stayed on the map, redrawn at their old
pixel positions against the new image. Fix mirrors the existing
Clear-button reset (`_on_nirc2_field_clear`), which already empties
both pools + selection.

Checks: (1) a genuine new frame (`reduced is not self._n2_image`)
clears both pools and the selection, and the redrawn map carries no
"dropped" artist and no rejected count in the title; (2) reprocessing
the SAME frame object (`reduced is self._n2_image`, e.g. a re-measure
on unchanged data) must NOT clear the pools -- only the frame-boundary
case is a reset. Fully offline; run headless
(QT_QPA_PLATFORM=offscreen).
"""
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from qtcompat import QtWidgets

import keck_ao_estimator.gui as gui


def mk(x, y, sr, params):
    return SimpleNamespace(strehl=sr, fwhm_mas=60.0, x=x, y=y,
                           sr_err=0.004, edge=False, crowded=False,
                           params=params)


def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow()
    params = SimpleNamespace(plate_scale_mas=20.0, effwave_um=2.124,
                             lgs=True, utc=None)
    win._nirc2_display = lambda r: None      # results block not under test

    frame_a = np.zeros((400, 400))
    frame_b = np.ones((400, 400))            # a genuinely different frame

    # ---- frame A: measure a field, reject one star, select the other -
    win._n2_image = frame_a
    win._n2_imno = 1
    win._n2_params = params
    win._n2_field = [mk(120.0, 150.0, 0.20, params),
                     mk(200.0, 220.0, 0.28, params)]
    win._n2_field_dropped = [mk(330.0, 90.0, 0.05, params)]
    win._n2_sel_star = 0
    win._n2_sel_dropped = None
    win._nirc2_draw_map()
    ax = win.n2_map_fig.axes[0]
    assert "dropped" in {getattr(c, "_n2_pool", None)
                         for c in ax.collections}
    assert "+1 rejected" in ax.get_title(), ax.get_title()

    # ---- same frame reprocessed: pools must SURVIVE -------------------
    win._on_nirc2_frame_done(1, object(), params, frame_a, None,
                             header={})
    assert win._n2_field_dropped and len(win._n2_field) == 2, \
        "reprocessing the SAME frame object must not clear the pools"
    assert win._n2_sel_star == 0, \
        "reprocessing the SAME frame must not clear the selection"

    # ---- new frame loads: both pools + selection must clear -----------
    win._n2_field_dropped = [mk(330.0, 90.0, 0.05, params)]  # restore
    win._n2_sel_star = 0
    win._on_nirc2_frame_done(2, object(), params, frame_b, None,
                             header={})
    assert win._n2_field == [], "a new frame must clear the kept pool"
    assert win._n2_field_dropped == [], \
        "a new frame must clear the × (dropped) pool -- the bug"
    assert win._n2_sel_star is None and win._n2_sel_dropped is None, \
        "a new frame must clear the selection ring"

    ax = win.n2_map_fig.axes[0]
    pools = {getattr(c, "_n2_pool", None) for c in ax.collections}
    assert "dropped" not in pools, \
        f"stale × marker must not survive onto the new frame's map ({pools})"
    assert "rejected" not in ax.get_title(), ax.get_title()

    print("gui_phase32: all checks passed")


if __name__ == "__main__":
    main()
