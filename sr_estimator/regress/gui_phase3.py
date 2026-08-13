#!/usr/bin/env python3
"""Phase-3 test: live WFE slider -> debounced recompute + MODIFIED BUDGET.
Run headless (QT_QPA_PLATFORM=offscreen)."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui

DATA = os.path.join(HERE, "data")
SHOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "gui_phase3.png")


def pump(win, cond, timeout=120):
    """Wait for `cond()`, actually pumping the native event loop (a nested
    QEventLoop) rather than alternating processEvents()/msleep(). On
    Windows a QTimer's callback is delivered via a posted message that a
    plain processEvents() call right after a blocking msleep() can miss for
    much longer than the timer's own interval (observed: a 150 ms debounce
    not landing within a manually-pumped 10 s budget) -- msleep() blocks the
    thread without servicing the message queue at all, so whether a timer
    "landed" depends on exactly when processEvents() happens to be called
    relative to it. A nested loop.exec() driven by re-arming QTimer.singleShot
    is the standard robust pattern instead."""
    app = QtWidgets.QApplication.instance()
    loop = QtCore.QEventLoop()
    t0 = time.time()

    done = False

    def check():
        # A stale re-arm MUST be inert. The last singleShot(10, check) is
        # armed BEFORE cond() goes true, so it survives loop.quit() and
        # fires after pump() has returned -- by which time `loop` may be
        # garbage-collected, and calling quit() on a deleted C++
        # QEventLoop segfaults. That is the SIGSEGV (-11) that failed
        # gui_phase24 on CI 2026-07-28: intermittent, load-dependent, and
        # not reproducible on an idle box.
        nonlocal done
        if done:
            return
        if cond() or time.time() - t0 > timeout:
            done = True
            loop.quit()
        else:
            QtCore.QTimer.singleShot(10, check)
    QtCore.QTimer.singleShot(0, check)
    loop.exec()
    done = True          # neutralise any callback still in flight


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.show(); app.processEvents()

    win.mode_local.setChecked(True)
    win.dimm_edit.setText(os.path.join(DATA, "20260525_dimm.dat"))
    win.mass_edit.setText(os.path.join(DATA, "20260525_mass.dat"))
    win.masspro_edit.setText(os.path.join(DATA, "20260525_masspro.dat"))
    win.tel_k1.setChecked(True); win._validate()
    win.on_run()
    pump(win, lambda: win.res is not None)
    assert win.res is not None, "initial run failed"
    ltao0 = float(engine.np.nanmean(win.res.sr_ltao))
    print(f"  [ok] initial run; LTAO mean = {ltao0:.3f}")

    # sanity: no overrides -> global unchanged
    assert win.current_offsets() == {}, "unexpected initial offsets"

    # push FA_REF (focal aniso) well off default via its spinbox
    default = win.wfe_rows["FA_REF"]["default"]
    newval = default + 80.0
    win.wfe_rows["FA_REF"]["spin"].setValue(newval)
    app.processEvents()
    assert "FA_REF" in win.current_offsets(), "offset not registered"

    # wait for the debounced recompute to land a new result
    prev_res = win.res
    pump(win, lambda: win.res is not prev_res, timeout=10)
    assert win.res is not prev_res, "debounced recompute did not fire"
    ltao1 = float(engine.np.nanmean(win.res.sr_ltao))
    print(f"  [ok] after FA_REF={newval:g}: LTAO mean = {ltao1:.3f} (was {ltao0:.3f})")
    assert abs(ltao1 - ltao0) > 1e-4, "budget override did not change the numbers"

    # MODIFIED BUDGET must be in the status line
    assert "MODIFIED BUDGET" in win.status.text(), win.status.text()
    print("  [ok] MODIFIED BUDGET shown in status")

    # budget_overrides must have RESTORED the global after the context exited
    assert engine.get_budget_param("FA_REF") == default, \
        f"global not restored: {engine.get_budget_param('FA_REF')} != {default}"
    print("  [ok] budget_overrides restored global FA_REF")

    app.processEvents()
    win.grab().save(SHOT)
    print(f"  [ok] screenshot saved: {SHOT}")


if __name__ == "__main__":
    main()
