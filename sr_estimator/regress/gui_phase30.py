#!/usr/bin/env python3
"""Gradient-aware field clip + reject/reinsert on the measured map
(2026-07-25, Eduardo's M92 finding: on strong-FA nights with the LGS
asterism off the field centre, the raw-median clip discarded the
well-corrected minority on the asterism side).

Checks, engine: (1) a strongly-tilted synthetic SR/FWHM field -- the
old median clip WOULD reject the bright end (asserted by construction:
the good tail sits > k*MAD + floor off the median), the new
gradient-aware clip keeps every on-plane star while still catching
planted photometry artifacts (off-plane SR dropout + broken-FWHM
blend); (2) collinear positions and small fields fall back to the
median clip (unchanged legacy behavior); (3) < 5 stars pass through.

Checks, GUI: rejected stars stay on the field map as pickable ×
markers (their own artist, tagged pool="dropped"); picking one turns
the button into 'Reinsert star' and clicking it moves the star back
into the fit; picking a kept star turns it back into 'Reject star'
and clicking moves the star to the × pool (the same mechanism, the
opposite direction); the map title counts the rejected stars; Clear
empties both pools. Fully offline; run headless
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

import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui


def star(x, y, sr, fwhm, err=0.005):
    return SimpleNamespace(strehl=sr, fwhm_mas=fwhm, x=x, y=y,
                           sr_err=err, edge=False, crowded=False)


def tilted_field():
    """The M92 0501 geometry: a bad MAJORITY far from the asterism and
    a well-corrected MINORITY on the asterism side, all riding one
    SR/FWHM plane -- the raw-median clip reads the good tail as
    outliers.  Plus 2 planted artifacts genuinely off the plane."""
    rng = np.random.default_rng(42)

    def plane_star(x):
        y = float(rng.uniform(120.0, 1880.0))       # non-collinear
        sr = 0.05 + (x - 100.0) / 1800.0 * 0.26     # 0.05 -> ~0.31
        fwhm = 85.0 - (x - 100.0) / 1800.0 * 27.0   # 85 -> ~58 mas
        return star(x, y, sr + float(rng.normal(0, 0.004)),
                    fwhm + float(rng.normal(0, 0.8)))

    stars = [plane_star(float(rng.uniform(100.0, 600.0)))
             for _ in range(15)]                    # bad majority
    stars += [plane_star(float(rng.uniform(1500.0, 1900.0)))
              for _ in range(5)]                    # asterism-side few
    blend = star(950.0, 1100.0, 0.17, 190.0)        # FWHM artifact
    dropout = star(1600.0, 1400.0, 0.02, 62.0)      # SR hole off-plane
    return stars, blend, dropout


def main():
    # ---- engine: gradient-aware keep, artifact catch -----------------
    stars, blend, dropout = tilted_field()
    srs = np.array([s.strehl for s in stars])
    med = float(np.median(srs))
    mad = 1.4826 * float(np.median(np.abs(srs - med)))
    top = srs.max()
    assert top - med > max(2.5 * mad, 0.05), \
        "field must be tilted enough that a median clip WOULD cut the top"
    kept, out = engine.field_consistent(stars + [blend, dropout])
    assert blend in out, "broken-FWHM blend must be rejected"
    assert dropout in out, "off-plane SR dropout must be rejected"
    assert all(s in kept for s in stars), \
        f"no on-plane star may be rejected (lost {len(out) - 2})"

    # ---- engine: collinear fallback = legacy median behavior ---------
    col = [star(100.0 + 40 * i, 500.0, 0.10, 70.0) for i in range(10)]
    col[7].strehl = 0.45                            # median outlier
    kept, out = engine.field_consistent(col)
    assert col[7] in out, "collinear field: median clip must still act"
    few = [star(10.0 * i, 10.0 * i, 0.1 + 0.2 * (i == 3), 70.0)
           for i in range(4)]
    kept, out = engine.field_consistent(few)
    assert len(kept) == 4 and not out, "< 5 stars pass through untouched"

    # ---- GUI: × markers, pick -> Reinsert, both directions -----------
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow()
    params = SimpleNamespace(plate_scale_mas=20.0, effwave_um=2.124,
                             lgs=True)
    mk = lambda x, y, sr: SimpleNamespace(                  # noqa: E731
        strehl=sr, fwhm_mas=60.0, x=x, y=y, sr_err=0.004,
        edge=False, crowded=False, params=params)
    win._n2_image = np.zeros((400, 400))
    win._n2_imno = 999
    win._n2_params = params
    win._n2_field = [mk(120.0 + 30 * i, 150.0 + 25 * i, 0.10 + 0.02 * i)
                     for i in range(6)]
    win._n2_field_dropped = [mk(330.0, 90.0, 0.31)]
    win._nirc2_display = lambda r: None      # results block not under test
    win._nirc2_draw_map()

    ax = win.n2_map_fig.axes[0]
    pools = {getattr(c, "_n2_pool", None) for c in ax.collections}
    assert "kept" in pools and "dropped" in pools, \
        f"map must carry kept + dropped artists (got {pools})"
    assert "+1 rejected ×" in ax.get_title(), ax.get_title()
    drop_artist = next(c for c in ax.collections
                       if getattr(c, "_n2_pool", None) == "dropped")
    kept_artist = next(c for c in ax.collections
                       if getattr(c, "_n2_pool", None) == "kept")

    win._on_nirc2_map_pick(SimpleNamespace(artist=drop_artist, ind=[0]))
    assert win.n2_reject_star.text() == "Reinsert star"
    assert win.n2_reject_star.isEnabled()
    win._on_nirc2_reject_star()              # reinsert the ×
    assert len(win._n2_field) == 7 and not win._n2_field_dropped, \
        "reinsert must move the star back into the fit"
    assert "reinserted into the fit by user" in win.n2_log.toPlainText()
    assert win.n2_reject_star.text() == "Reject star", \
        "button must rearm as Reject star after the move"

    win._nirc2_draw_map()
    kept_artist = next(c for c in win.n2_map_fig.axes[0].collections
                       if getattr(c, "_n2_pool", None) == "kept")
    win._on_nirc2_map_pick(SimpleNamespace(artist=kept_artist, ind=[2]))
    assert win.n2_reject_star.text() == "Reject star"
    win._on_nirc2_reject_star()              # reject a kept star -> ×
    assert len(win._n2_field) == 6 and len(win._n2_field_dropped) == 1, \
        "reject must move the star to the × pool, not delete it"
    assert "reinsertable" in win.n2_log.toPlainText()

    win._on_nirc2_field_clear()
    assert not win._n2_field and not win._n2_field_dropped, \
        "Clear must empty both pools"

    print("gui_phase30: all checks passed")


if __name__ == "__main__":
    main()
