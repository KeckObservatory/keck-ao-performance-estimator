#!/usr/bin/env python3
"""parallel PR-CP3 acceptance, round 3 (Opus, 2026-09-13, blocker B3):
results that arrive after the per-target loop has already asked to
stop must not be counted, logged, or kept.

Opus's evidence (M15 real frame, offscreen, 6 stars requested,
psf_clean on): `workers=8` gave `tried=26` where `workers=1` gave the
correct `tried=21` -- 5 late candidates (`star 22`-`26: rejected`)
were processed and logged after the target was already reached.
Reproduced here on the self-contained gui_phase34 crowded fixture
(no dependency on the bundled M15/starfinder_integration data this
repo does not carry) with psf_clean OFF (native path, faster, same
race): before the fix, `workers=1` gave `tried=7, kept=6` and
`workers=8` gave `tried=9, kept=8` -- not just extra "rejected" log
lines, a PASSING late result was actually kept, so the map ended up
with MORE stars than requested.

Cause: `_nirc2_field_on_result` calls `FieldMeasureWorker.request_stop()`
in two places (auto quality-gate tripped twice; kept count reaches the
target) but does not stop processing results ITSELF -- with
`MeasureBatch`, the pool has already computed and queued the next
targets before `FieldMeasureWorker.run` next checks its stop flag, so
those queued signals still land in the handler. The core engine
(`measure_field`) does not have this: its decision loop breaks
in-thread before reading another result, already asserted identical
serial vs parallel by `parallel_model.py` (c) -- this was purely a
GUI-side race.

Fix: `_n2_field_late_result_guard`, set at both `request_stop()` call
sites, checked at the top of `_nirc2_field_on_result` (same idiom
already used for `_n2_field_cancel_requested`) so a late result is
dropped before it touches `tried`, the log, or the map. Reset once per
run in `_nirc2_field_start_measurer` (covers the field-consistency
backfill restart too).

Checks: on the SAME crowded fixture, `workers=1` and `workers=8` give
the identical kept count, identical `tried` count, and identical log
text (byte for byte -- this fixture's frame is small/fast enough that
no timing numbers appear in the "Measure field" log lines themselves,
so no masking is needed, unlike Opus's `opus_accept_log_diff_probe.py`
scratch probe this file is adapted from). Also asserts neither run
shows a "rejected" log line for a star index beyond `tried` (the
literal symptom Opus quoted -- "star 22"-"26: rejected" after the
6-of-6 summary line).

Fully offline; run headless (QT_QPA_PLATFORM=offscreen).
"""
import os
import sys
import tempfile
import time

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

FAILURES = []


def check(name, cond, detail=""):
    tag = "ok" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def pump(cond, timeout=90):
    app = QtWidgets.QApplication.instance()
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        QtCore.QThread.msleep(10)
    return cond()


def run_once(workers, n_stars=6):
    """The live flow, from scratch, on the crowded gui_phase34 fixture
    -- more candidates than requested (>= 8 real stars for 6 asked),
    the exact shape B3 needs: a stop request with candidates still
    in flight behind it."""
    tmp = tempfile.mkdtemp(prefix="gui41_b3_")
    p34.make_psf_clean_field(tmp, 1)
    win = gui.MainWindow()
    win.n2_path.setText(tmp)
    win.n2_im1.setValue(1)
    win.n2_nim.setValue(1)
    win.n2_nbg.setValue(0)
    win.n2_autofind.setChecked(False)
    win.n2_add_star.setChecked(True)
    win._on_nirc2_go()
    pump(lambda: win.n2_go.isEnabled())
    assert win._n2_image is not None, win.n2_log.toPlainText()

    win.n2_psf_clean.setChecked(False)
    win.n2_autofind.setChecked(True)
    win.n2_add_star.setChecked(False)
    win.n2_nstars.setValue(n_stars)
    win.n2_workers.setValue(workers)

    before = win.n2_log.toPlainText()
    win._on_nirc2_measure_field()
    done = pump(lambda: not win._n2_field_busy, timeout=90)
    text = win.n2_log.toPlainText()[len(before):]
    tried = win._n2_field_tried
    kept = [(round(r.x, 3), round(r.y, 3), repr(r.strehl), repr(r.sr_err))
           for r in win._n2_field]

    for name in ("_n2_field_prologue", "_n2_field_measurer", "_n2_worker"):
        worker = getattr(win, name, None)
        if worker is not None:
            worker.wait(30000)
    win.close()
    QtWidgets.QApplication.instance().processEvents()
    return {"done": done, "tried": tried, "kept": kept, "text": text}


def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv[:1])

    r1 = run_once(1)
    r8 = run_once(8)

    check("workers=1 run completes", r1["done"])
    check("workers=8 run completes", r8["done"])
    check("workers=1 vs workers=8: identical tried count "
          "(the literal B3 symptom -- Opus's evidence was 21 vs 26)",
          r1["tried"] == r8["tried"],
          f"workers=1 tried={r1['tried']}, workers=8 tried={r8['tried']}")
    check("workers=1 vs workers=8: identical kept stars (position, "
          "SR, SR error -- not just the same COUNT)",
          r1["kept"] == r8["kept"],
          f"workers=1 kept={len(r1['kept'])}, workers=8 kept={len(r8['kept'])}")
    check("workers=1 vs workers=8: identical log text, byte for byte "
          "(this fixture's log lines carry no timing numbers, so no "
          "masking is needed)",
          r1["text"] == r8["text"],
          "" if r1["text"] == r8["text"] else
          f"workers=1 ({len(r1['text'])} chars) != "
          f"workers=8 ({len(r8['text'])} chars)")

    # the literal symptom Opus quoted: a "rejected" line for a star
    # index beyond what was actually tried (late arrivals logged after
    # the stop request)
    for tag, r in (("workers=1", r1), ("workers=8", r8)):
        late_lines = [ln for ln in r["text"].splitlines()
                     if "rejected" in ln
                     and any(f"star {i}:" in ln
                             for i in range(r["tried"] + 1, r["tried"] + 10))]
        check(f"{tag}: no 'rejected' log line for a star index beyond "
              "tried (no late arrivals logged)",
              not late_lines, late_lines)

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("gui_phase41: all checks passed")


if __name__ == "__main__":
    main()
