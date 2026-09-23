#!/usr/bin/env python3
"""LGS-offset default by instrument, and the optional LGS-flux model for the
measurement-error term (2026-09-22, Eduardo Marin).

(1) LGS offset. The laser is offset by default ONLY on K1 with the OSIRIS
    imager (4.97"); the OSIRIS spectrograph and K2/NIRC2 are on axis (0).
    Engine: default_lgs_offset(telescope, instrument); resolve_lgs_offset
    (args) honours --lgs-offset, else --instrument, else the telescope's
    default instrument (K1 imager, K2 NIRC2) -- so every pre-existing K1
    result is unchanged. GUI: the instrument comes from the Field map tab's
    OSIRIS imager/spectrograph selector on K1; switching it moves the
    laser on the field map and recomputes.
(2) LGS flux. lgs_flux.py models the sodium return vs (az, el): 1/airmass,
    extinction, geomagnetic efficiency. Zenith = 1; it reproduces the
    2026-09-21 LGS-WFS count ratio TYC 5858-779-1 / UCAC4 748-00066 = 0.75
    (model 0.74). With the option on, the measurement term is HOMEAS *
    F^(-1/2); off, and under legacy, the budget is byte-identical. GUI: an
    "LGS flux" SUB-tab of LGS (house rule: the dock never scrolls), with a
    checkbox that reaches collect_args and survives a config round-trip.
Fully offline, headless.
"""
import os, sys
from types import SimpleNamespace
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
from keck_ao_estimator.lgs_flux import lgs_return_rel, lgs_return_rel_azavg, meas_scale
np = engine.np


def settle(n=6):
    app = QtWidgets.QApplication.instance()
    for _ in range(n):
        app.processEvents(); QtCore.QThread.msleep(25)


def engine_contract():
    d = engine.default_lgs_offset
    assert d("K1") == 4.97 and d("K1", "osiris-imager") == 4.97
    assert d("K1", "osiris-spec") == 0.0 and d("K1", "nirc2") == 0.0
    assert d("K2") == 0.0 and d("K2", "osiris-imager") == 0.0
    ns = lambda **k: SimpleNamespace(**{"telescope": "K1", "lgs_offset": None,
                                        "instrument": None, **k})
    assert engine.resolve_lgs_offset(ns()) == 4.97
    assert engine.resolve_lgs_offset(ns(instrument="osiris-spec")) == 0.0
    assert engine.resolve_lgs_offset(ns(instrument="osiris-spec", lgs_offset=3.0)) == 3.0
    assert engine.resolve_lgs_offset(ns(telescope="K2")) == 0.0
    assert engine.resolve_lgs_offset(ns(telescope="K2", instrument="osiris-imager")) == 0.0
    print("  [ok] engine: LGS offset 4.97\" only for K1 + OSIRIS imager; "
          "spectrograph/K2 0; override wins")

    assert abs(lgs_return_rel(0.0, 90.0) - 1.0) < 1e-12
    r = lgs_return_rel(128.7, 31.4) / lgs_return_rel(8.7, 49.5)
    assert abs(r - 0.74) < 0.02, r              # measured 0.75 (AOAOAMED)
    # return falls with airmass on average; south of zenith beats north at
    # the same zenith distance (beam along the field lines)
    assert lgs_return_rel_azavg(30.0) < lgs_return_rel_azavg(60.0) < 1.0
    assert lgs_return_rel(189.5, 60.0) > lgs_return_rel(9.5, 60.0)
    assert abs(meas_scale(None, 90.0) - 1.0) < 1e-9
    base = engine.lgs_budget_terms(0.8, 0.4, "K2", "single")
    on = engine.lgs_budget_terms(0.8, 0.4, "K2", "single",
                                 lgs_flux_azel=(128.7, 31.4))
    zen = engine.lgs_budget_terms(0.8, 0.4, "K2", "single",
                                  lgs_flux_azel=(0.0, 90.0))
    leg = engine.lgs_budget_terms(0.8, 0.4, "K2", "single", legacy=True,
                                  lgs_flux_azel=(128.7, 31.4))
    assert base["meas"] == engine.budget.HOMEAS == leg["meas"]
    assert abs(zen["meas"] - base["meas"]) < 1e-9
    assert on["meas"] > 1.4 * base["meas"]
    assert all(on[k] == base[k] for k in base if k != "meas")
    s0 = engine.lgs_strehl(0.8, 0.4, "K2", "single", 2124)
    s1 = engine.lgs_strehl(0.8, 0.4, "K2", "single", 2124, lgs_flux_azel=(128.7, 31.4))
    assert s1 < s0
    print(f"  [ok] engine: flux model zenith=1, 09-21 ratio {r:.2f} (meas 0.75); "
          f"meas {base['meas']:.1f} -> {on['meas']:.1f} nm at EL 31; off/legacy untouched")


def gui_tab():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = gui.MainWindow(); win.resize(1550, 950); win.show(); app.processEvents()
    settle(4)

    # (1) LGS offset follows the OSIRIS imager/spectrograph selector on K1
    win.tel_k1.setChecked(True); settle(2)
    win.lgs_offset_enable.setChecked(False)
    win.fm_osiris_mode.setCurrentText("imager (20×20″)"); settle(2)
    a = win.collect_args(out_path="/tmp/x.png")
    assert engine.resolve_instrument(a) == "osiris-imager" and engine.resolve_lgs_offset(a) == 4.97
    assert abs(np.hypot(*win._laser_xy()) - 4.97) < 1e-9
    win.fm_osiris_mode.setCurrentText("spectrograph"); settle(2)
    a = win.collect_args(out_path="/tmp/x.png")
    assert engine.resolve_instrument(a) == "osiris-spec" and engine.resolve_lgs_offset(a) == 0.0
    assert np.hypot(*win._laser_xy()) == 0.0
    win.lgs_offset_enable.setChecked(True); win.lgs_offset.setValue(2.5); settle(2)
    assert engine.resolve_lgs_offset(win.collect_args(out_path="/tmp/x.png")) == 2.5
    win.lgs_offset_enable.setChecked(False)
    win.fm_osiris_mode.setCurrentText("imager (20×20″)")
    win.tel_k2.setChecked(True); settle(2)
    a = win.collect_args(out_path="/tmp/x.png")
    assert engine.resolve_instrument(a) == "nirc2" and engine.resolve_lgs_offset(a) == 0.0
    # the K1 OSIRIS choice survives a telescope swap (Summary tab copies args)
    a.telescope = "K1"
    assert engine.resolve_lgs_offset(a) == 4.97
    print("  [ok] GUI: K1 imager 4.97\" / spectrograph 0\" (laser moves); "
          "override wins; K2 NIRC2 0\"")

    # (2) LGS flux: a sub-tab of LGS, no scrollbars, checkbox -> args
    ti = [win.tabs.tabText(i) for i in range(win.tabs.count())].index("LGS")
    win.tabs.setCurrentIndex(ti); settle(3)
    assert win.lgs_subtabs.tabText(0) == "Budget"
    assert win.lgs_subtabs.tabText(1) == "LGS flux"
    scroll = win.tabs.widget(ti)
    for i in (0, 1):
        win.lgs_subtabs.setCurrentIndex(i); settle(3)
        assert not scroll.verticalScrollBar().isVisible(), \
            f"LGS sub-page {i} must not need a vertical scrollbar"
        assert not scroll.horizontalScrollBar().isVisible()
    assert not win.collect_args(out_path="/tmp/x.png").lgs_flux_model
    win.lgs_flux_cb.setChecked(True); settle(2)
    assert win.collect_args(out_path="/tmp/x.png").lgs_flux_model
    cfg = win._collect_config()
    assert cfg["lgs_flux_model"] is True
    win.lgs_flux_cb.setChecked(False); settle(2)
    win._apply_config(cfg); settle(2)
    assert win.lgs_flux_cb.isChecked()
    old = {k: v for k, v in cfg.items() if k != "lgs_flux_model"}
    win._apply_config(old); settle(2)
    assert not win.lgs_flux_cb.isChecked()
    print("  [ok] GUI: 'LGS flux' is an LGS sub-tab; no scrollbars; checkbox -> "
          "args; config round-trip; older config loads with it off")
    # Measured-SR compare path: pointing from the FRAME header, guarded
    win.lgs_flux_cb.setChecked(True); settle(1)
    win.args_cached = win.collect_args(out_path="/tmp/x.png")
    P = lambda az, el: SimpleNamespace(camname="osiris", az_deg=az, el_deg=el)
    assert win._frame_flux_azel(P(129.0, 31.0)) == (129.0, 31.0)
    assert win._frame_flux_azel(P(None, 31.0)) is None
    assert win._frame_flux_azel(P(129.0, None)) is None
    assert win._frame_flux_azel(P(129.0, -3.0)) is None
    win.lgs_flux_cb.setChecked(False); settle(1)
    win.args_cached = win.collect_args(out_path="/tmp/x.png")
    assert win._frame_flux_azel(P(129.0, 31.0)) is None
    print("  [ok] GUI: Measured-SR compare takes az/el from the frame header; "
          "missing AZ/EL -> no flux scaling")
    win.lgs_subtabs.setCurrentIndex(1); settle(2)
    win.grab().save(os.path.join(HERE, "gui_phase44.png"))


def main():
    engine_contract()
    gui_tab()
    print("  [ok] LGS offset by instrument + LGS-flux measurement-error option")


if __name__ == "__main__":
    main()
