#!/usr/bin/env python3
"""Tests: 'LGS' tab rename, WFE reference-profile/scaling note, and the
layer-mismatch m readout. Run headless."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
DATA = os.path.join(HERE, "data")


def pump(cond, timeout=90):
    app = QtWidgets.QApplication.instance(); t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents(); QtCore.QThread.msleep(20)


def draw(win, holder):
    win._save = holder["canvas"].draw()
    for _ in range(5):
        QtWidgets.QApplication.instance().processEvents(); QtCore.QThread.msleep(20)


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.show(); app.processEvents()

    # 1) tab renamed Budget -> LGS
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert "LGS" in titles and "Budget" not in titles, titles
    print("  [ok] control tabs:", titles)

    # 2) WFE scaling map covers every registry entry
    assert set(gui.WFE_SCALING) == set(engine.ADJUSTABLE_BUDGET_PARAMS), \
        "WFE_SCALING must cover every adjustable param"
    print("  [ok] WFE scaling tags cover all", len(gui.WFE_SCALING), "params")

    win.mode_local.setChecked(True)
    win.dimm_edit.setText(os.path.join(DATA, "20260525_dimm.dat"))
    win.mass_edit.setText(os.path.join(DATA, "20260525_mass.dat"))
    win.masspro_edit.setText(os.path.join(DATA, "20260525_masspro.dat"))
    win.tel_k1.setChecked(True); win._validate(); win.on_run()
    pump(lambda: win.res is not None)

    # 3) m readout populated (K1 tomography on -> applied)
    mtext = win.m_label.text()
    assert "mean" in mtext and "applied to LTAO" in mtext, mtext
    print("  [ok] layer-mismatch m (K1):", mtext)

    # legacy -> n/a
    win.legacy_cb.setChecked(True)
    prev = win.prep; pump(lambda: win.prep is not prev, timeout=30)
    assert "legacy" in win.m_label.text(), win.m_label.text()
    print("  [ok] legacy budget -> m:", win.m_label.text())
    win.legacy_cb.setChecked(False)
    prev = win.prep; pump(lambda: win.prep is not prev, timeout=30)

    # screenshots of the LGS tab and the WFE sliders tab
    win.tabs.setCurrentIndex([win.tabs.tabText(i) for i in
                              range(win.tabs.count())].index("LGS"))
    app.processEvents(); win.grab().save(os.path.join(HERE, "gui_lgs_tab.png"))
    win.tabs.setCurrentIndex([win.tabs.tabText(i) for i in
                              range(win.tabs.count())].index("WFE sliders"))
    app.processEvents(); win.grab().save(os.path.join(HERE, "gui_wfe_tab.png"))
    print("  [ok] screenshots saved (LGS + WFE tabs)")


if __name__ == "__main__":
    main()
