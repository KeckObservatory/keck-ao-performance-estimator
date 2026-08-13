#!/usr/bin/env python3
"""EE aperture correction (2026-07-25, Eduardo): engine model +
field-map integration.

Engine: ee_correct/ee_expected_small round-trip; ee_calibrate_h
recovers a known h from synthetic two-aperture pairs and refuses
(<5 pairs) with a plain-language ValueError.

GUI: the checkbox is gated on Auto aperture (disabled + unchecked
when auto is off); _nirc2_apply_ee takes clean full-radius values
verbatim, growth-curve-corrects crowded ones with the field's own
fitted h, logs the summary, and is idempotent across backfill
passes. Offscreen (QT_QPA_PLATFORM=offscreen), fully synthetic.
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


def result(sr, photrad, crowding=0.0, saturated=False):
    return engine.Nirc2StrehlResult(
        strehl=sr, fwhm_mas=55.0, wfe_nm=300.0, x=100.0, y=100.0,
        peak=5000.0, flux=1e6, sky=1.0, saturated=saturated,
        params=SimpleNamespace(plate_scale_mas=20.0, effwave_um=2.124,
                               lgs=True),
        crowding=crowding, sr_err=0.01, photrad_used_arcsec=photrad)


def main():
    # ---- engine: model round-trip + h recovery -----------------------
    h0 = 0.40
    true = np.linspace(0.05, 0.6, 12)
    small = engine.ee_expected_small(true, h0)
    back = engine.ee_correct(small, h0)
    assert np.allclose(back, true, atol=1e-9), "round-trip must be exact"
    h_fit, rms = engine.ee_calibrate_h(list(zip(small, true)))
    assert abs(h_fit - h0) < 0.02 and rms < 1e-3, \
        f"h recovery failed: {h_fit} rms {rms}"
    try:
        engine.ee_calibrate_h([(0.3, 0.2)])
        raise AssertionError("must refuse < 5 pairs")
    except ValueError as e:
        assert "pairs" in str(e)

    # ---- GUI ----------------------------------------------------------
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv[:1])
    import keck_ao_estimator.gui as gui
    win = gui.MainWindow()

    assert not win.n2_ee_corr.isEnabled(), "gated on Auto aperture"
    win.n2_auto_rad.setChecked(True)
    assert win.n2_ee_corr.isEnabled()
    win.n2_ee_corr.setChecked(True)
    win.n2_auto_rad.setChecked(False)
    assert not win.n2_ee_corr.isEnabled() and not win.n2_ee_corr.isChecked(), \
        "auto-off must clear and disable the EE box"
    win.n2_auto_rad.setChecked(True)
    win.n2_ee_corr.setChecked(True)

    # synthetic field: 6 clean calibration stars (full companion
    # uncrowded) + 2 crowded ones (companion contaminated) at h0
    field, pairs = [], {}
    for sr_t in (0.10, 0.18, 0.26, 0.34, 0.42, 0.50):
        small_r = result(float(engine.ee_expected_small(sr_t, h0)), 0.3)
        full_r = result(sr_t, 1.0, crowding=0.02)
        field.append(small_r)
        pairs[id(small_r)] = full_r
    crowded_smalls = []
    for sr_t in (0.22, 0.38):
        small_r = result(float(engine.ee_expected_small(sr_t, h0)), 0.3)
        full_r = result(sr_t * 0.7, 1.0, crowding=0.60)  # contaminated
        field.append(small_r)
        pairs[id(small_r)] = full_r
        crowded_smalls.append((small_r, sr_t))
    win._n2_field = field
    win._n2_ee_pairs = pairs
    win._n2_ee_done = set()
    win._nirc2_apply_ee()

    for i, sr_t in enumerate((0.10, 0.18, 0.26, 0.34, 0.42, 0.50)):
        assert abs(win._n2_field[i].strehl - sr_t) < 1e-9, \
            "clean full-radius value must be adopted verbatim"
    for _small_r, sr_t in crowded_smalls:
        got = [r for r in win._n2_field
               if abs(r.strehl - sr_t) < 0.02 and r.photrad_used_arcsec == 0.3]
        assert got, f"crowded star must be corrected to ~{sr_t}"
    log = win.n2_log.toPlainText()
    assert "EE aperture correction" in log and "h=0.4" in log, log
    n_before = len(win._n2_field)
    win._nirc2_apply_ee()          # idempotent on a second pass
    assert len(win._n2_field) == n_before
    assert log.count("EE aperture correction") == \
        win.n2_log.toPlainText().count("EE aperture correction"), \
        "second pass must be a no-op"

    print("gui_phase31: all checks passed")


if __name__ == "__main__":
    main()
