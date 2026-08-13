#!/usr/bin/env python3
"""Phase-5 test: terms tab, window title, config save/load round-trip, and the
reference-budget preset. Run headless."""
import os, sys, time, json, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator.gui as gui

DATA = os.path.join(HERE, "data")


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

    # terms tab present + enabled (MASS data available)
    assert win.plot_tabs.count() == 4, "expected 4 plot tabs (Timeline/Field/Terms/NIRC2)"
    assert win.plot_tabs.isTabEnabled(2), "terms tab should be enabled"
    print("  [ok] terms tab present and enabled")

    # window title shows night + telescope
    assert "2026-05-24" in win.windowTitle() and "K1" in win.windowTitle(), \
        win.windowTitle()
    print("  [ok] window title:", win.windowTitle())

    # config round-trip: save, change a field, load, confirm restored
    cfg = os.path.join(tempfile.gettempdir(), "p5_cfg.json")
    with open(cfg, "w") as fh:
        json.dump(win._collect_config(), fh)
    win.ngs_bright.setValue(11.0)
    win.seeing_law.setCurrentText("gaussian")
    with open(cfg) as fh:
        win._apply_config(json.load(fh))
    assert abs(win.ngs_bright.value() - 8.0) < 1e-6, win.ngs_bright.value()
    assert win.seeing_law.currentText() == "kolmogorov", win.seeing_law.currentText()
    print("  [ok] config save/load round-trip restored ngs_bright + seeing_law")

    # reference-budget preset clears overrides
    win.wfe_rows["STATIC_CALIB"]["spin"].setValue(120.0)
    assert win.current_offsets(), "override not set"
    win._preset_reference_budget()
    assert win.current_offsets() == {}, "preset did not reset WFE"
    print("  [ok] reference-budget preset cleared overrides")

    # let any pending debounced recompute settle, then capture the terms tab
    time.sleep(0.3); app.processEvents()
    win.plot_tabs.setCurrentIndex(2)
    win._terms_holder["canvas"].draw()
    for _ in range(5):
        app.processEvents(); QtCore.QThread.msleep(20)
    # sanity: the terms figure has real content (axes with lines), not blank
    tfig = win._terms_holder["canvas"].figure
    assert len(tfig.axes) >= 4, f"terms figure has {len(tfig.axes)} axes"
    assert any(ax.lines or ax.collections for ax in tfig.axes), "terms figure empty"
    print(f"  [ok] terms figure has {len(tfig.axes)} axes with content")
    win.grab().save(os.path.join(HERE, "gui_phase5_terms.png"))
    win.plot_tabs.setCurrentIndex(0)
    win._main_holder["canvas"].draw()
    for _ in range(5):
        app.processEvents(); QtCore.QThread.msleep(20)
    win.grab().save(os.path.join(HERE, "gui_phase5.png"))
    print("  [ok] screenshots saved (timeline + terms tabs)")


if __name__ == "__main__":
    main()
