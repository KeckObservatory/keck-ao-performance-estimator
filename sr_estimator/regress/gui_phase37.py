#!/usr/bin/env python3
"""fieldsolve P3-2 (Sonnet, 2026-09-13): the field engine on the
Measured-SR tab -- the engine selector (native/field, FS-D17 default
field), its config round-trip, per-target threading of the engine
choice, `[psf-clean:field]` log lines (a real subtraction and the
null/refusal outcomes VERBATIM, including "field solution did not
converge"), and the once-per-frame field-solution build running on a
WORKER THREAD (`FieldPrologueWorker`, parallel Phase 2 -- supersedes
fieldsolve P3-2's narrower `FieldSolveWorker`), not blocking the GUI
thread the way `_nirc2_stage()`'s old `processEvents()` calls did.

Reuses gui_phase34.py's crowded synthetic field (same donor/target/
neighbour geometry, the validated psf_clean test fixture) rather than
building a new one -- this file is about the ENGINE SELECTOR and the
field-specific log/threading behaviour, not re-deriving the psf_clean
feature's own correctness, which gui_phase34 already covers for the
native engine.

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


def pump(cond, timeout=120):
    app = QtWidgets.QApplication.instance()
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        QtCore.QThread.msleep(10)


def click_at(win, x, y):
    # see gui_phase34.py's click_at for why the duplicate-frame dialog is
    # stubbed the same way here
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

    # ---- selector present, defaults to field (FS-D17), native selectable ----
    assert hasattr(win, "n2_psf_clean_engine")
    assert win.n2_psf_clean_engine.currentText() == "field", \
        "FS-D17: field is the default engine"
    assert [win.n2_psf_clean_engine.itemText(i)
            for i in range(win.n2_psf_clean_engine.count())] == ["field", "native"]
    assert not win.n2_psf_clean_engine.isEnabled(), \
        "disabled until PSF-fit cleaning itself is on"
    win.n2_psf_clean.setChecked(True)
    assert win.n2_psf_clean_engine.isEnabled()
    win.n2_psf_clean.setChecked(False)
    print("  [ok] engine selector present, defaults to field (FS-D17), "
          "native selectable, enabled only with cleaning on")

    # ---- config round-trip ---------------------------------------------------
    win.n2_psf_clean_engine.setCurrentText("native")
    c = win._collect_config()
    assert c["nirc2"]["psf_clean_engine"] == "native"
    win.n2_psf_clean_engine.setCurrentText("field")
    win._apply_config(c)
    assert win.n2_psf_clean_engine.currentText() == "native", \
        "psf_clean_engine round-trips"
    win.n2_psf_clean_engine.setCurrentText("field")
    print("  [ok] psf_clean_engine config round-trip")

    # ---- per-target state threading (fieldsolve P3-2) -------------------------
    win.tname_edit.setText("T-native")
    win.ra_edit.setText("10:00:00")
    win.dec_edit.setText("20:00:00")
    win.n2_psf_clean_engine.setCurrentText("native")
    win._save_current_target()
    win.tname_edit.setText("T-field")
    win.ra_edit.setText("11:00:00")
    win.dec_edit.setText("21:00:00")
    win.n2_psf_clean_engine.setCurrentText("field")
    win._save_current_target()
    names = [t["name"] for t in win._targets]
    i_nat, i_fld = names.index("T-native"), names.index("T-field")
    win._on_target_selected(i_nat)
    assert win.n2_psf_clean_engine.currentText() == "native"
    win._on_target_selected(i_fld)
    assert win.n2_psf_clean_engine.currentText() == "field"
    print("  [ok] per-target engine choice threaded through "
          "_add_target/_on_target_selected/_save_current_target")

    # ---- build and load the crowded field (gui_phase34's fixture) ------------
    tmp = tempfile.mkdtemp(prefix="gui37_field_engine_")
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

    # ---- ON, field engine, auto-find FIELD measure: the field solution is
    # built ONCE per frame, on a real WORKER THREAD, after the ePSF (which
    # stays a blocking GUI-thread stage, unchanged) -----------------------
    win.n2_psf_clean.setChecked(True)
    win.n2_psf_clean_engine.setCurrentText("field")
    win._on_nirc2_field_clear()
    win.n2_autofind.setChecked(True)
    win.n2_add_star.setChecked(False)
    win.n2_nstars.setValue(6)
    win._on_nirc2_measure_field()
    assert isinstance(win._n2_field_prologue, FieldPrologueWorker), \
        "the once-per-frame solve must go through FieldPrologueWorker"
    # wait for the whole flow (prologue -> per-target loop) to finish --
    # _n2_field_busy, not the button (which stays enabled throughout so
    # it can double as Cancel, parallel Phase 2) or a since-removed queue
    pump(lambda: not win._n2_field_busy)
    log = win.n2_log.toPlainText()
    assert "[psf-clean:field] field ePSF:" in log, log
    assert "[psf-clean:field] field solution:" in log, log
    i_epsf = log.find("[psf-clean:field] field ePSF:")
    i_sol = log.find("[psf-clean:field] field solution:")
    assert i_sol > i_epsf, \
        f"the field-solution log line must follow the ePSF one:\n{log}"
    sol_line = log[i_sol:log.find("\n", i_sol) if "\n" in log[i_sol:] else None]
    assert "converged=True" in sol_line, log
    print("  [ok] field engine: solution built ONCE per frame via "
          "FieldPrologueWorker, logged after the ePSF")

    win._on_nirc2_field_clear()
    win.n2_autofind.setChecked(False)
    win.n2_add_star.setChecked(True)

    # ---- [psf-clean:field] tag on a real subtraction, VERBATIM engine note --
    n_before = len(win.n2_log.toPlainText())
    click_at(win, *p34.BLEND_TARGET)
    added = win.n2_log.toPlainText()[n_before:]
    assert "[psf-clean:field]" in added, added
    assert "component(s) subtracted" in added, added
    assert "[psf-clean] " not in added.replace("[psf-clean:field] ", ""), \
        f"the native tag must not leak when the field engine is selected:\n{added}"
    print("  [ok] [psf-clean:field] tag on a real field-engine subtraction, "
          "engine note verbatim")

    # ---- null outcome VERBATIM, field-tagged ----------------------------------
    win._on_nirc2_field_clear()
    click_at(win, *p34.ISO_POS)
    log = win.n2_log.toPlainText()
    assert ("[psf-clean:field] 0 components to subtract for this target"
            in log and "effectively isolated" in log), log
    print("  [ok] null outcome printed verbatim, field-tagged")

    # ---- "field solution did not converge" -- exercised for real by
    # forcing solve_field's own non-convergence path (max_sweeps
    # exhausted before tol is met -- Opus's O1 field_solve_model.py slot
    # (d) uses the same max_sweeps=1/unreachable-tol technique), never
    # hand-authored as a string -------------------------------------------
    import keck_ao_estimator.field_solve as fs
    real_solve_field = fs.solve_field

    def _never_converges(*a, **kw):
        kw["max_sweeps"] = 1
        kw["tol"] = 1e-12
        return real_solve_field(*a, **kw)

    fs.solve_field = _never_converges
    try:
        win._on_nirc2_field_clear()
        win.n2_autofind.setChecked(True)
        win.n2_add_star.setChecked(False)
        win._on_nirc2_measure_field()
        pump(lambda: not win._n2_field_busy)
    finally:
        fs.solve_field = real_solve_field
    log = win.n2_log.toPlainText()
    assert "[psf-clean:field] field solution:" in log and \
        "converged=False" in log[log.find("field solution:"):], log
    assert "field solution did not converge" in log, log
    print("  [ok] non-convergence refuses legibly, verbatim "
          "'field solution did not converge', never a silent fallback")

    win.n2_psf_clean.setChecked(False)
    win._on_nirc2_field_clear()
    win.close()
    app.processEvents()
    print("gui_phase37: all checks passed")


if __name__ == "__main__":
    main()
