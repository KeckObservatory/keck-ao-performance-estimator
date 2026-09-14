#!/usr/bin/env python3
"""parallel PR-P0-3 (Sonnet, 2026-09-13) / Phase 2 (2026-09-13): the
field-solve fix, updated for Phase 2's `FieldPrologueWorker` (which
generalizes fieldsolve P3-2's narrower `FieldSolveWorker` to the WHOLE
"Measure field" prologue -- find_stars, the catalogue, the ePSF, and the
solve, all off the GUI thread) -- and the Cancel button Phase 2 adds.

FS-OPEN-7 was never a hang (parallel/STATUS.md PR-P0-2): every earlier
"hang" was a diagnostic polling `n2_field_btn.isEnabled()`. Phase 2 makes
that button double as Cancel while busy, so it is deliberately never
disabled any more -- checks here use `win._n2_field_busy` instead, the
same fix generalized.
  (a) the prologue is a REAL running QThread; the flow is busy (not the
      button's enabled state, which now means "click to cancel"); a
      SECOND click while busy CANCELS (this is Phase 2's Cancel, not a
      no-op -- the old "second click is a no-op" behaviour is gone by
      design);
  (b) the solution is logged within 30 s, and a 50 ms QTimer fires at
      least once DURING the solve (the GUI thread is not blocked);
  (c) the field measurement completes, with the ePSF line before the
      solution line before the first star line (order, not just
      presence);
  (d) re-measuring the frame while a second prologue is still building
      makes its result stale; it is discarded with the exact log line,
      never used to clean the new frame.

Fully offline; run headless (QT_QPA_PLATFORM=offscreen).
"""
import os
import sys
import tempfile
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import warnings
warnings.filterwarnings("ignore")

from qtcompat import QtCore, QtWidgets

import gui_phase34 as p34
import keck_ao_estimator.gui as gui
from keck_ao_estimator.gui.workers import FieldPrologueWorker

FAILURES = []


def check(name, cond, detail=""):
    tag = "ok" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def pump(cond, timeout=60):
    app = QtWidgets.QApplication.instance()
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        QtCore.QThread.msleep(10)
    return cond()


def click_at(win, x, y):
    # same duplicate-frame-dialog stub as gui_phase34/37's click_at
    ev = SimpleNamespace(xdata=float(x), ydata=float(y),
                         inaxes=win.n2_fig.axes[0])
    orig_exec = QtWidgets.QMessageBox.exec
    QtWidgets.QMessageBox.exec = lambda self: 0
    try:
        win._on_nirc2_click(ev)
    finally:
        QtWidgets.QMessageBox.exec = orig_exec


def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow()

    tmp = tempfile.mkdtemp(prefix="gui38_field_fix_")
    p34.make_psf_clean_field(tmp, 1)
    win.n2_path.setText(tmp)
    win.n2_im1.setValue(1)
    win.n2_nim.setValue(1)
    win.n2_nbg.setValue(0)
    win.n2_autofind.setChecked(False)
    win.n2_add_star.setChecked(True)
    win._on_nirc2_go()
    pump(lambda: win.n2_go.isEnabled())
    assert win._n2_image is not None, win.n2_log.toPlainText()

    win.n2_psf_clean.setChecked(True)
    win.n2_psf_clean_engine.setCurrentText("field")
    win.n2_autofind.setChecked(True)
    win.n2_add_star.setChecked(False)
    win.n2_nstars.setValue(6)

    # a 50 ms timer that just counts ticks -- if it fires DURING the
    # solve below, the GUI thread was free to service it, i.e. not
    # blocked inside the solve the way the pre-fix synchronous .run()
    # call blocked it
    ticks = {"n": 0}
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: ticks.__setitem__("n", ticks["n"] + 1))
    timer.start(50)

    # ---- (a) real thread, flow held busy, second click cancels ----------
    win._on_nirc2_measure_field()
    w = win._n2_field_prologue
    check("a: the prologue runs on a real QThread from the live flow",
          isinstance(w, FieldPrologueWorker) and w.isRunning(),
          f"isRunning={w.isRunning()}")
    check("a: the flow is busy while it runs (Cancel, not disabled -- "
          "the button now doubles as Cancel)",
          win._n2_field_busy and win.n2_field_btn.isEnabled())
    win._on_nirc2_measure_field()   # a second click while busy: CANCEL
    check("a: a second click during the prologue CANCELS (Phase 2)",
          win._n2_field_cancel_requested)
    pump(lambda: not win._n2_field_busy, timeout=30)
    check("a: cancelling mid-prologue ends the flow (not busy any more)",
          not win._n2_field_busy)

    # ---- (b) solution within 30s, GUI thread free meanwhile, for real
    # this time (not cancelled) --------------------------------------------
    win.n2_log.clear()
    win._on_nirc2_measure_field()
    ticks["n"] = 0
    t0 = time.perf_counter()
    got = pump(lambda: "[psf-clean:field] field solution:"
              in win.n2_log.toPlainText(), timeout=30)
    dt = time.perf_counter() - t0
    ticks_during_solve = ticks["n"]
    check("b: solution logged within 30 s", got, f"{dt:.2f}s")
    check("b: GUI thread not blocked during the solve (50 ms timer fired)",
          ticks_during_solve >= 1, f"{ticks_during_solve} ticks in {dt:.2f}s")

    # ---- (c) completion, lines in the right order ------------------------
    done = pump(lambda: not win._n2_field_busy, timeout=120)
    log = win.n2_log.toPlainText()
    i_epsf = log.find("field ePSF:")
    i_sol = log.find("field solution:")
    i_star1 = log.find("star 1:")
    check("c: field measurement completes; ePSF < solution < first star line",
          done and 0 <= i_epsf < i_sol < i_star1,
          f"positions {i_epsf} {i_sol} {i_star1}")

    timer.stop()

    # ---- (d) frame replaced mid-prologue -> the late result is discarded --
    win._on_nirc2_field_clear()
    win._on_nirc2_measure_field()
    w2 = win._n2_field_prologue
    check("d: second field prologue started", w2 is not w and w2.isRunning())
    win._on_nirc2_go()  # re-measures the SAME frame path: a NEW reduced array
    pump(lambda: w2.isFinished() and win.n2_go.isEnabled(), timeout=60)
    pump(lambda: False, timeout=1.0)  # let any queued slot run
    log2 = win.n2_log.toPlainText()[len(log):]
    last = log2.strip().splitlines()[-1][:90] if log2.strip() else ""
    check("d: a result for a replaced frame is discarded, legibly",
          "discarded — the frame changed while it was running" in log2
          and win._n2_field_solution is None
          and not win._n2_field_busy
          and win.n2_field_btn.isEnabled(),
          last)

    # ---- wait for every worker to finish before the script exits (R7) ----
    for name in ("_n2_field_prologue", "_n2_field_measurer", "_n2_worker"):
        worker = getattr(win, name, None)
        if worker is not None:
            worker.wait(30000)

    win.n2_psf_clean.setChecked(False)
    win._on_nirc2_field_clear()
    win.close()
    app.processEvents()

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("gui_phase38: all checks passed")


if __name__ == "__main__":
    main()
