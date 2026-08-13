#!/usr/bin/env python3
"""Prediction mode with NO night loaded (2026-08-12): the field map and the
Error-terms tab must both work straight from the controls + the Prediction
tab's scenario, with no MKWC data and no Run.

Covers: (1) enabling the scenario before any Run renders a real field map
(PREDICTED SCENARIO title) instead of the "Run first" placeholder; (2) the
Error-terms tab shows the snapshot term breakdown whose totals match an
independent lgs_budget_terms + Marechal computation; (3) scenario slider
edits re-render the terms; (3b/3c/3d) CONTROL edits with no run loaded --
TT-star magnitude and WFE sliders -- must live-update the predicted terms
and field map through the _schedule debounce (2026-08-12 fix: _schedule
used to bail when prep is None, freezing both views until the scenario
was toggled); (4) toggling the scenario off with no run restores the
placeholder (no stale predicted figure); (5) the surrogate args/prep path
agrees with a direct engine call. Run headless."""
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


def _pump(app, ms=600):
    """Spin the event loop long enough for the 150 ms _schedule throttle
    (and the 220 ms field-map LOD settle timer) to fire."""
    end = time.time() + ms / 1000.0
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1500, 950); win.show()
    app.processEvents()
    assert win.prep is None and win.res is None, "test needs a fresh window"

    # --- (1) field map renders from the scenario with no run ---------------
    win.pred_enable.setChecked(True)         # jumps to the Field map tab
    app.processEvents()
    assert win.plot_tabs.currentIndex() == 1, "enable must land on Field map"
    fig = win._fm_holder["canvas"].figure
    axes = [a for a in fig.axes if a.get_title()]
    assert axes, "field map must render (no placeholder) with no run"
    title = axes[0].get_title()
    assert "PREDICTED SCENARIO" in title, title
    print("  [ok] no-run field map renders with the PREDICTED SCENARIO title")

    # --- (5) surrogate args/prep agree with a direct engine call -----------
    args, prep = win._fm_args(), win._fm_prep()
    snap = win._pred_snapshot()
    on_axis = engine.field_metric_at(args, prep, snap, "single", "strehl",
                                     (0.0, 0.0), (0.0, 0.0), win._laser_xy(),
                                     (0.0, 0.0))
    direct = engine.lgs_strehl(
        snap["eps_tot_los"], snap["eps_fa_los"], args.telescope, "single",
        prep.lam_nm, tt_mag=args.tt_mag, tt_offset=0.1,
        lgs_offset=args.lgs_offset, legacy=args.legacy_budget,
        v_ground=args.wind_ground, v_free=args.wind_free,
        aniso_scale=snap["aniso_scale"],
        tt_sensor=getattr(args, "_tt_sensor_base", "strap"),
        ltao_tt_theta0_gain=args.ltao_tt_theta0_gain)
    assert abs(on_axis - direct) < 0.02, (on_axis, direct)
    print(f"  [ok] surrogate path matches a direct engine call "
          f"({on_axis:.3f} vs {direct:.3f})")

    # --- (2) predicted error terms on the Error-terms tab ------------------
    win.plot_tabs.setCurrentIndex(2)
    app.processEvents()
    tfig = win._terms_holder["canvas"].figure
    ttitle = tfig.axes[0].get_title()
    assert "PREDICTED SCENARIO" in ttitle and "error-budget terms" in ttitle, \
        ttitle
    # totals in the title must match an independent computation
    t = engine.lgs_budget_terms(
        snap["eps_tot_los"], snap["eps_fa_los"], args.telescope, "single",
        None, tt_mag=args.tt_mag, tt_offset=args.tt_offset,
        lgs_offset=args.lgs_offset, legacy=args.legacy_budget,
        v_ground=args.wind_ground, v_free=args.wind_free,
        aniso_scale=snap["aniso_scale"],
        tt_sensor=getattr(args, "_tt_sensor_base", "strap"),
        ltao_tt_theta0_gain=args.ltao_tt_theta0_gain)
    static = np.sqrt(t["stat_tel"]**2 + t["stat_calib"]**2 + t["stat_dm"]**2
                     + t["stat_inst"]**2 + t["stat_reg"]**2)
    ho = np.sqrt(t["fit"]**2 + t["scint"]**2 + t["ang"]**2 + t["bw"]**2
                 + t["alt"]**2 + t["meas"]**2 + t["nafoc"]**2 + static**2
                 + t["margin"]**2)
    assert f"HO {ho:.0f} nm" in ttitle, (ho, ttitle)
    print(f"  [ok] predicted terms figure: single-beacon HO total "
          f"{ho:.0f} nm matches the title")

    # --- (3) a scenario edit re-renders the terms --------------------------
    old_title = ttitle
    win.pred_dimm.setValue(1.20)
    app.processEvents()
    new_title = win._terms_holder["canvas"].figure.axes[0].get_title()
    assert new_title != old_title and "1.20" in new_title, new_title
    print("  [ok] scenario edit re-renders the predicted terms live")

    # --- (3b) a CONTROL edit (TT-star mag) live-updates the terms ----------
    old_title = new_title
    win.tt_mag.setValue(win.tt_mag.value() + 3.0)   # fires _schedule()
    _pump(app)
    t3 = win._terms_holder["canvas"].figure.axes[0].get_title()
    assert t3 != old_title, \
        "TT-mag edit must re-render the predicted terms with no run"
    print("  [ok] TT-star edit live-updates the predicted terms (no run)")

    # --- (3c) a WFE slider edit live-updates + flags MODIFIED BUDGET -------
    name, row = next(iter(win.wfe_rows.items()))
    bump = min(row["spin"].maximum(), row["default"] + 50.0)
    assert bump != row["default"], f"cannot bump {name} within its range"
    row["spin"].setValue(bump)
    _pump(app)
    assert abs(win.last_offsets.get(name, 0.0) - bump) < 1e-9, \
        (name, win.last_offsets)
    tfig3 = win._terms_holder["canvas"].figure
    texts = ([t.get_text() for t in tfig3.texts]
             + [t.get_text() for a in tfig3.axes for t in a.texts])
    assert any("MODIFIED BUDGET" in t for t in texts), texts
    row["spin"].setValue(row["default"])
    _pump(app)
    print(f"  [ok] WFE edit ({name}) live-updates the terms + "
          "MODIFIED BUDGET flag")

    # --- (3d) with the Field-map tab showing, a control edit redraws it ----
    # NB the field map redraws IN PLACE (im.set_data) -- compare the grid
    # data, not the canvas identity.
    def fm_grid_sum():
        fig_fm = win._fm_holder["canvas"].figure
        for a in fig_fm.axes:
            for im in a.get_images():
                return float(np.nansum(np.asarray(im.get_array(), float)))
        raise AssertionError("no field-map heatmap found")
    win.plot_tabs.setCurrentIndex(1)
    _pump(app)
    grid_before = fm_grid_sum()
    win.tt_mag.setValue(win.tt_mag.value() - 3.0)   # back to the original
    _pump(app)
    assert abs(fm_grid_sum() - grid_before) > 1e-6, \
        "TT-mag edit must redraw the no-run field map"
    fm_title = [a.get_title() for a in win._fm_holder["canvas"].figure.axes
                if a.get_title()][0]
    assert "PREDICTED SCENARIO" in fm_title, fm_title
    win.plot_tabs.setCurrentIndex(2)
    app.processEvents()
    print("  [ok] control edit redraws the no-run field map live")

    # --- (4) toggling off with no run restores the placeholder -------------
    win.pred_enable.setChecked(False)
    app.processEvents()
    win._render_terms_if_visible()
    app.processEvents()
    tfig2 = win._terms_holder["canvas"].figure
    texts = [t.get_text() for a in tfig2.axes for t in a.texts]
    assert any("Prediction tab" in t or "after a run" in t for t in texts), \
        f"terms tab must fall back to the placeholder, got {texts}"
    print("  [ok] scenario off with no run: terms placeholder restored")

    win.close()
    print("gui_phase35: all checks passed")


if __name__ == "__main__":
    main()
