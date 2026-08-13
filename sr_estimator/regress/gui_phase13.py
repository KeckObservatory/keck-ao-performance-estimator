#!/usr/bin/env python3
"""Toolbar zoom/pan must survive the figure reassignment done on every redraw.

matplotlib >= 3.6 keeps the event CallbackRegistry on the FIGURE
(canvas.callbacks -> figure._canvas_callbacks), so assigning canvas.figure = new
orphans the navigation toolbar's handlers unless the registry is carried across.
Run headless."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore, Qt, BINDING
import keck_ao_estimator.gui as gui
if BINDING == "PyQt6":
    from PyQt6.QtTest import QTest
else:
    from PyQt5.QtTest import QTest
DATA = os.path.join(HERE, "data")


def pump(cond, timeout=90):
    app = QtWidgets.QApplication.instance(); t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents(); QtCore.QThread.msleep(10)


def qt_zoom(holder):
    """Drive a real Qt left-drag through the canvas; return (before, after)."""
    app = QtWidgets.QApplication.instance()
    c, nav = holder["canvas"], holder["navbar"]
    fig = c.figure; ax = fig.axes[0]
    nav.zoom()
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    dA = ax.transData.transform((x0 + (x1 - x0) * 0.3, y0 + (y1 - y0) * 0.3))
    dB = ax.transData.transform((x0 + (x1 - x0) * 0.7, y0 + (y1 - y0) * 0.7))
    h = fig.bbox.height
    qA = QtCore.QPoint(int(dA[0]), int(h - dA[1]))
    qB = QtCore.QPoint(int(dB[0]), int(h - dB[1]))
    before = ax.get_xlim()
    QTest.mousePress(c, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, qA); app.processEvents()
    QTest.mouseMove(c, qB); app.processEvents()
    QTest.mouseRelease(c, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, qB); app.processEvents()
    nav.zoom()
    return before, ax.get_xlim(), ax, nav


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1500, 950); win.show(); app.processEvents()
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(os.path.join(DATA, "20260525_dimm.dat"))
    win.mass_edit.setText(os.path.join(DATA, "20260525_mass.dat"))
    win.masspro_edit.setText(os.path.join(DATA, "20260525_masspro.dat"))
    win.tel_k1.setChecked(True); win._validate(); win.on_run()
    pump(lambda: win.res is not None)

    # 1) zoom works on the freshly rendered timeline
    before, after, ax, nav = qt_zoom(win._main_holder)
    assert abs(after[0] - before[0]) > 1e-9, "zoom did not change xlim"
    print(f"  [ok] zoom works on timeline ({before[0]:.4f} -> {after[0]:.4f})")

    # 2) Home restores the pre-zoom view (nav stack valid for the new axes)
    nav.home(); app.processEvents()
    assert abs(ax.get_xlim()[0] - before[0]) < 1e-6, "Home did not restore"
    print("  [ok] toolbar Home restores the original view")

    # 3) still works after redraws (each swaps in a brand-new Figure)
    c = win._main_holder["canvas"]
    for v in (7.0, 9.0, 8.0):
        win.ngs_bright.setValue(v); win.recompute_and_draw(); app.processEvents()
    before, after, _, _ = qt_zoom(win._main_holder)
    assert abs(after[0] - before[0]) > 1e-9, "zoom broke after recompute"
    print("  [ok] zoom still works after 3 recomputes")

    # 4) handlers must not ACCUMULATE on the carried-over registry. (The count
    # may DECREASE: the carried registry holds weakref-to-figure callbacks for
    # the swapped-out figures, which matplotlib auto-drops as those orphaned
    # figures are garbage-collected. That is cleanup, not a leak; the toolbar's
    # zoom handler persists -- verified by (3) above and (5) below.)
    n = len(c.callbacks.callbacks.get("button_press_event", {}))
    for v in (6.0, 10.0):
        win.ngs_bright.setValue(v); win.recompute_and_draw(); app.processEvents()
    n2 = len(c.callbacks.callbacks.get("button_press_event", {}))
    assert n2 <= n, f"button_press subscribers grew {n} -> {n2} (leak)"
    print(f"  [ok] no handler leak across redraws ({n} -> {n2} subscribers)")

    # 5) the terms tab's toolbar works too
    win.plot_tabs.setCurrentIndex(2); app.processEvents()
    before, after, _, _ = qt_zoom(win._terms_holder)
    assert abs(after[0] - before[0]) > 1e-9, "zoom broken on terms tab"
    print("  [ok] zoom works on the error-terms tab")


if __name__ == "__main__":
    main()
