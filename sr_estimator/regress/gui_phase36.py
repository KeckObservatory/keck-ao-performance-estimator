#!/usr/bin/env python3
"""TRICK tip-tilt sensor must survive a Run + recompute (2026-09-04).

Found via the Prediction tab: the same scenario gave a different LTAO field
map with a night loaded than without. The only engine input that changed was
the tip-tilt sensor -- collect_args() never set the derived
args._tt_sensor_base, prepare_night() resolved it in place on the Run's args,
and then recompute_and_draw() replaced args_cached with a fresh, unresolved
collection. Every consumer that reads getattr(args, "_tt_sensor_base",
"strap") -- compute_timeline, the field map, the terms tab -- silently fell
back to STRAP with TRICK selected. collect_args() now resolves the sensor
itself.

Checks: (1) after a TRICK (H) run the cached args carry the resolved sensor
through the recompute; (2) the GUI's timeline equals a compute_timeline call
on explicitly resolved args, and differs from a STRAP computation; (3) the
prediction-scenario field map gives the same on-axis Strehl with and without
the night loaded. Run headless."""
import copy
import os
import sys
import time

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

DATA = os.path.join(HERE, "data")


def _pump(app, ms=800):
    end = time.time() + ms / 1000.0
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def _pred_on_axis(win):
    """On-axis LTAO Strehl of the Prediction-tab scenario through the same
    args/prep/snapshot path _render_field_map uses."""
    args, prep, snap = win._fm_args(), win._fm_prep(), win._pred_snapshot()
    fc = win._sky_field_center()
    with engine.budget_overrides(**win.last_offsets):
        return engine.field_metric_at(
            args, prep, snap, "ltao", "strehl",
            win.ngs_offset.offset_xy(fc)[:2], win.tt_offset.offset_xy(fc)[:2],
            win._laser_xy(), (0.0, 0.0))


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1500, 950); win.show()
    app.processEvents()

    win.tel_k1.setChecked(True)
    win.tt_sensor.setCurrentText("TRICK (H)")
    win.tt_mag.setValue(12.0)
    win.tt_offset.setValue(15.0)
    win.fm_mode.setCurrentText("LTAO")
    win.pred_dimm.setValue(0.64)
    win.pred_mass.setValue(0.41)

    # --- (3a) prediction with NO night --------------------------------------
    win.pred_enable.setChecked(True)
    _pump(app)
    s_no_night = _pred_on_axis(win)
    a0 = win._fm_args()
    assert getattr(a0, "_tt_sensor_base", None) == "trick", vars(a0).get("_tt_sensor_base")

    # --- Run a night with TRICK selected -----------------------------------
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(f"{DATA}/20260525_dimm.dat")
    win.mass_edit.setText(f"{DATA}/20260525_mass.dat")
    win.masspro_edit.setText(f"{DATA}/20260525_masspro.dat")
    win._validate(); win.on_run()
    t0 = time.time()
    while win.res is None and time.time() - t0 < 120:
        _pump(app, 200)
    assert win.res is not None, "run did not complete"

    # --- (1) the cached args keep the resolved sensor ----------------------
    a = win.args_cached
    assert a.tt_sensor == "trick-h", a.tt_sensor
    assert getattr(a, "_tt_sensor_base", None) == "trick", \
        "args_cached lost _tt_sensor_base after the recompute"
    assert getattr(a, "_tt_wfs_band", None) == "H"
    assert a.band == "K", a.band          # dichroic: science in the other band
    print("  [ok] args_cached carries the resolved TRICK sensor after a run")

    # a compute-only edit re-collects args again -- must still be resolved
    win.tt_mag.setValue(12.5)
    _pump(app)
    assert getattr(win.args_cached, "_tt_sensor_base", None) == "trick", \
        "a recompute dropped the resolved sensor"
    print("  [ok] resolved sensor survives a compute-only recompute")

    # --- (2) the timeline was computed as TRICK, not STRAP -----------------
    ref = engine.compute_timeline(win.args_cached, win.prep)
    assert np.allclose(win.res.sr_ltao, ref.sr_ltao, equal_nan=True)
    strap = copy.copy(win.args_cached)
    strap.tt_sensor = "strap"
    engine.resolve_tt_sensor(strap)
    res_strap = engine.compute_timeline(strap, win.prep)
    m_gui = np.nanmedian(win.res.sr_ltao)
    m_strap = np.nanmedian(res_strap.sr_ltao)
    assert m_gui > m_strap + 0.01, (m_gui, m_strap)
    print(f"  [ok] GUI LTAO timeline is the TRICK budget "
          f"(median {m_gui:.3f} vs STRAP {m_strap:.3f})")

    # --- (3b) prediction with the night loaded matches the no-night map ----
    win.tt_mag.setValue(12.0)
    _pump(app)
    win.pred_enable.setChecked(True)
    _pump(app)
    s_night = _pred_on_axis(win)
    assert abs(s_night - s_no_night) < 1e-6, (s_no_night, s_night)
    print(f"  [ok] prediction scenario identical with/without a night "
          f"({s_no_night:.4f})")
    print("PASS gui_phase36")


if __name__ == "__main__":
    main()
