#!/usr/bin/env python3
"""Headless smoke test + screenshot for the GUI (Phase 1).

Run with QT_QPA_PLATFORM=offscreen. Checks:
  1. window builds;
  2. collect_args() with untouched widgets is behaviorally equal to the parser
     defaults (§6 round-trip);
  3. a real Run on the local May files produces a figure, and we grab a PNG
     screenshot of the whole window for Eduardo.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
os.chdir(ROOT)

from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui

DATA = os.path.join(HERE, "data")
SHOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "gui_phase1.png")


def check_collect_args_roundtrip(win):
    defs = engine.build_parser().parse_args([])
    got = win.collect_args(out_path="/tmp/x.png")
    # wavelength: GUI seeds band='K', parser seeds band=None -- both resolve to K
    dw = engine.resolve_wavelength(defs)
    gw = engine.resolve_wavelength(got)
    assert dw == gw, f"wavelength drift: {dw} vs {gw}"
    # spot-check the fields the GUI maps 1:1
    for f in ["telescope", "ngs_bright", "ngs_faint", "ngs_offset",
              "assumed_theta0", "ngs_seeing_law", "tt_mag", "tt_offset",
              "lgs_offset", "ltao_bw_floor_frac", "legacy_budget",
              "show_target", "zenith_angle", "tomography"]:
        assert getattr(defs, f) == getattr(got, f), \
            f"{f}: default {getattr(defs,f)!r} != collected {getattr(got,f)!r}"
    print("  [ok] collect_args round-trip matches parser defaults")


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow()
    win.show()
    app.processEvents()
    print("  [ok] window built,", gui.BINDING)

    check_collect_args_roundtrip(win)

    # drive a real run on the local May files, K1
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(os.path.join(DATA, "20260525_dimm.dat"))
    win.mass_edit.setText(os.path.join(DATA, "20260525_mass.dat"))
    win.masspro_edit.setText(os.path.join(DATA, "20260525_masspro.dat"))
    win.tel_k1.setChecked(True)
    win._validate()
    assert win.run_btn.isEnabled(), "Run disabled with valid inputs"
    win.on_run()

    t0 = time.time()
    while win.worker is not None and time.time() - t0 < 120:
        app.processEvents()
        QtCore.QThread.msleep(30)
    assert win.res is not None, "run produced no result"
    print("  [ok] run finished:", win.status.text()[:90])

    app.processEvents()
    ok = win.grab().save(SHOT)
    print(f"  [ok] screenshot saved: {SHOT} ({ok})")


if __name__ == "__main__":
    main()
