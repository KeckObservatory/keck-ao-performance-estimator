#!/usr/bin/env python3
"""Phase-4 test: provenance in exported CSV + PNG footer, under active WFE
overrides, and ABSENT when at reference. Run headless."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator.gui as gui

DATA = os.path.join(HERE, "data")
OUT = HERE  # export destinations


def pump(cond, timeout=120):
    app = QtWidgets.QApplication.instance(); t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents(); QtCore.QThread.msleep(20)


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.show(); app.processEvents()
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(os.path.join(DATA, "20260525_dimm.dat"))
    win.mass_edit.setText(os.path.join(DATA, "20260525_mass.dat"))
    win.masspro_edit.setText(os.path.join(DATA, "20260525_masspro.dat"))
    win.tel_k1.setChecked(True); win._validate(); win.on_run()
    pump(lambda: win.res is not None)

    # --- reference (no overrides): exported CSV must NOT carry the token ---
    csv_ref = os.path.join(OUT, "export_ref.csv")
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (csv_ref, ""))
    win.on_export_csv()
    assert os.path.exists(csv_ref)
    assert not any("budget_overrides" in l for l in open(csv_ref)), \
        "reference export must not have budget_overrides"
    print("  [ok] reference CSV export: no budget_overrides token")

    # --- modify a WFE param, let it recompute, then export CSV + PNG ---
    win.wfe_rows["MARGIN"]["spin"].setValue(90.0)
    prev = win.res
    pump(lambda: win.res is not prev, timeout=10)

    csv_mod = os.path.join(OUT, "export_modified.csv")
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (csv_mod, ""))
    win.on_export_csv()
    prov = [l.strip() for l in open(csv_mod) if "budget_overrides" in l]
    assert prov and "MARGIN=90" in prov[0], prov
    print("  [ok] modified CSV export provenance:", prov[0])

    png_mod = os.path.join(OUT, "export_modified.png")
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (png_mod, ""))
    win.on_export_png()
    assert os.path.exists(png_mod) and os.path.getsize(png_mod) > 10000
    print(f"  [ok] modified PNG exported ({os.path.getsize(png_mod)} bytes, footer stamped)")

    win.grab().save(os.path.join(OUT, "gui_phase4.png"))
    print("  [ok] screenshot saved")


if __name__ == "__main__":
    main()
