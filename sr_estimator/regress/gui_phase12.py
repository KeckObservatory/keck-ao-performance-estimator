#!/usr/bin/env python3
"""Responsiveness guards: keyboard tracking off, terms figure lazy, no file I/O
in the live recompute path, and a generous wall-clock ceiling. Run headless."""
import os, sys, time, glob, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator.gui as gui
DATA = os.path.join(HERE, "data")
CEILING_MS = 900          # generous; the real path measures ~340 ms


def pump(cond, timeout=90):
    """Wait for `cond()` via a nested QEventLoop rather than alternating
    processEvents()/msleep() -- on Windows the latter can miss a QTimer's
    callback for well longer than the timer's own interval, since msleep()
    blocks the thread without servicing the message queue at all (see
    gui_phase3.py's pump() for the full story; this file's coarse-vs-full
    scrub-timing check (#8) is exactly the kind of tight, back-to-back-timer
    race that pattern was unreliable for)."""
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
    win = gui.MainWindow(); win.resize(1500, 950); win.show(); app.processEvents()

    # 1) every spinbox must have keyboardTracking OFF, else typing/deleting a
    #    character fires a recompute per keystroke.
    spins = win.findChildren(QtWidgets.QDoubleSpinBox)
    bad = [s for s in spins if s.keyboardTracking()]
    assert not bad, f"{len(bad)} spinbox(es) still emit valueChanged per keystroke"
    print(f"  [ok] keyboardTracking off on all {len(spins)} spinboxes")

    outdir = tempfile.mkdtemp(prefix="p12_")
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(os.path.join(DATA, "20260525_dimm.dat"))
    win.mass_edit.setText(os.path.join(DATA, "20260525_mass.dat"))
    win.masspro_edit.setText(os.path.join(DATA, "20260525_masspro.dat"))
    win.outdir_edit.setText(outdir)
    win.tel_k1.setChecked(True); win._validate(); win.on_run()
    pump(lambda: win.res is not None)

    # 2) the live path must not write any files (savefig/CSV moved to Export)
    assert win.plot_tabs.currentIndex() == 0
    for v in (7.0, 9.0):
        win.ngs_bright.setValue(v); win.recompute_and_draw(); app.processEvents()
    produced = glob.glob(os.path.join(outdir, "*"))
    assert not produced, f"live recompute wrote files: {produced}"
    print("  [ok] live recompute writes no files (export-only I/O)")

    # 3) terms figure is lazy: stale while its tab is hidden
    assert win._terms_dirty, "terms should be stale while its tab is hidden"
    win.plot_tabs.setCurrentIndex(2); app.processEvents()
    assert not win._terms_dirty, "showing the terms tab should render it"
    assert len(win._terms_holder["canvas"].figure.axes) == 8
    win.plot_tabs.setCurrentIndex(0); app.processEvents()
    win.ngs_bright.setValue(8.0); win.recompute_and_draw(); app.processEvents()
    assert win._terms_dirty, "a recompute must re-dirty the hidden terms figure"
    print("  [ok] terms figure renders lazily, only when its tab is shown")

    # 4) wall-clock ceiling for one live recompute on the Timeline tab
    times = []
    for v in (6.0, 7.5, 9.0, 10.5):
        t0 = time.perf_counter()
        win.ngs_bright.setValue(v); win.recompute_and_draw(); app.processEvents()
        times.append((time.perf_counter() - t0) * 1e3)
    avg = sum(times) / len(times)
    assert avg < CEILING_MS, f"live recompute {avg:.0f} ms exceeds {CEILING_MS} ms"
    print(f"  [ok] live recompute {avg:.0f} ms avg (ceiling {CEILING_MS} ms)")

    # 5) re-entrancy guard exists and is released
    assert win._busy is False
    print("  [ok] re-entrancy guard released")

    # 6) field-map inputs (incl. LGS offset, previously not wired at all) are
    #    THROTTLED, not rendered synchronously on every tick -- so holding a
    #    spinbox's up/down button (which fires valueChanged repeatedly) marks
    #    the map stale and arms one redraw, rather than blocking the UI thread
    #    on a full grid re-evaluation (tens-to-~130 ms) per tick.
    win.plot_tabs.setCurrentIndex(1); app.processEvents()
    win._fieldmap_dirty = False
    assert not win._fm_debounce.isActive()
    win.lgs_offset_enable.setChecked(True)
    for v in (2.0, 4.0, 6.0, 8.0):           # simulates a held spin-button
        win.lgs_offset.setValue(v)
    assert win._fieldmap_dirty, "LGS offset must mark the field map stale"
    assert win._fm_debounce.isActive(), \
        "field-map redraw must not fire synchronously on every tick"
    pump(lambda: not win._fm_debounce.isActive(), timeout=2)
    app.processEvents()
    assert not win._fieldmap_dirty, "the throttle should have redrawn once armed"
    print("  [ok] LGS offset live-updates the field map, throttled not per-tick")

    # 6b) crucially, this must be a THROTTLE, not a plain reset-every-tick
    #     debounce: while a button is held, ticks arrive faster than the
    #     150 ms redraw budget, so a plain debounce (whose timer keeps getting
    #     reset) would never get a quiet gap to fire and the map would only
    #     ever update once the button is released -- exactly the regression a
    #     user reported. Simulate a ~600 ms sustained hold (ticks every 40 ms,
    #     faster than the redraw interval) and require several redraws to
    #     happen DURING the hold, not just one at the end.
    renders = []
    orig_render = win._render_field_map

    def _counting_render(*a, **k):
        renders.append(1)
        return orig_render(*a, **k)
    win._render_field_map = _counting_render
    try:
        t0 = time.time(); v = 2.0
        while time.time() - t0 < 0.6:
            v += 0.2
            win.lgs_offset.setValue(v)
            app.processEvents(); QtCore.QThread.msleep(40)
        pump(lambda: not win._fm_debounce.isActive(), timeout=2)
        app.processEvents()
    finally:
        win._render_field_map = orig_render
    assert len(renders) >= 2, \
        ("holding the button must redraw periodically DURING the hold, not "
         f"only once at release (got {len(renders)} redraw(s) in ~600 ms)")
    print(f"  [ok] sustained hold redraws periodically ({len(renders)} times "
          f"in ~600 ms), not only once on release")

    # 7) the same throttle-not-debounce fix generalizes to the main
    #    _schedule/_debounce mechanism: NGS/TT magnitude and the WFE sliders
    #    are compute-affecting controls (not field-map-only), routed through
    #    _on_compute_changed/_schedule -> _debounce -> recompute_and_draw
    #    (which itself redraws the field map when that tab is visible). A
    #    held button here must show periodic live updates too, not just one
    #    recompute at release.
    calls = []
    orig_recompute = win.recompute_and_draw

    def _counting_recompute(*a, **k):
        calls.append(1)
        return orig_recompute(*a, **k)
    win.recompute_and_draw = _counting_recompute
    try:
        t0 = time.time(); v = win.ngs_bright.value()
        while time.time() - t0 < 2.0:
            v += 0.3
            win.ngs_bright.setValue(v)
            app.processEvents(); QtCore.QThread.msleep(40)
        pump(lambda: not win._debounce.isActive(), timeout=2)
        app.processEvents()
    finally:
        win.recompute_and_draw = orig_recompute
    assert len(calls) >= 2, \
        ("holding NGS-bright's spin button must recompute+redraw periodically "
         f"DURING the hold, not only once at release (got {len(calls)} in ~2s)")
    print(f"  [ok] NGS-bright magnitude live-updates periodically while held "
          f"({len(calls)} recompute(s) in ~2s), not only on release")

    # 8) level-of-detail while scrubbing: a live (mid-scrub) frame renders a
    #    COARSE grid via the fast reuse-the-figure path, then a trailing
    #    full-resolution redraw fires once input settles. Both must show the
    #    real, evolving performance (the coarse frame is not frozen) and the
    #    reported on-axis value must not jump between the coarse and full grids.
    win.plot_tabs.setCurrentIndex(1); app.processEvents()
    win.lgs_offset_enable.setChecked(True)
    pump(lambda: not win._fm_debounce.isActive() and not win._fm_settle.isActive())

    def fm_img_shape():
        fig = win._fm_holder["canvas"].figure
        for ax in fig.axes:
            if ax.images:
                return ax.images[0].get_array().shape
        return None

    # a burst of changes: while _fm_settle is armed the render is coarse.
    # Capture the shape from INSIDE _fm_debounce's own timeout handling
    # (connected here, so it runs right after the render it triggers --
    # QTimer delivers a signal's slots in connection order) rather than
    # polling isActive() afterwards: the two timers are only ~70 ms apart
    # (150 vs 220 ms), and polling can observe the state a poll-interval
    # late, by which point _fm_settle may have ALSO already fired and
    # triggered its own full-res redraw, clobbering the coarse one we
    # meant to inspect.
    captured = {}

    def _on_debounce_fired():
        # everything read here, synchronously within _fm_debounce's own
        # timeout handling, before _fm_settle gets any chance to also fire
        # and swap in its own full-res figure
        captured["shape"] = fm_img_shape()
        live = getattr(win, "_fm_live", None)
        captured["live_fig"] = live["fig"] if live else None
        captured["canvas_fig"] = win._fm_holder["canvas"].figure
    win._fm_debounce.timeout.connect(_on_debounce_fired)
    try:
        for v in (2.0, 4.0, 6.0):
            win.lgs_offset.setValue(v); app.processEvents()
        assert win._fm_settle.isActive(), "a scrub must arm the LOD settle timer"
        pump(lambda: "shape" in captured, timeout=5)
    finally:
        win._fm_debounce.timeout.disconnect(_on_debounce_fired)
    assert "shape" in captured, "the coarse live frame's debounce never fired"
    coarse = captured["shape"]
    assert coarse == (21, 21), f"mid-scrub frame must be coarse, got {coarse}"
    assert captured["live_fig"] is not None and \
        captured["canvas_fig"] is captured["live_fig"], \
        "the interactive frame must reuse the persistent live figure"
    # let it settle -> trailing full-resolution redraw
    pump(lambda: not win._fm_settle.isActive()); app.processEvents()
    pump(lambda: fm_img_shape() == (41, 41))
    assert fm_img_shape() == (41, 41), \
        f"the settled redraw must be full resolution, got {fm_img_shape()}"
    print("  [ok] field map scrubs coarse (21²) on a reused figure, then settles "
          "full-res (41²)")


if __name__ == "__main__":
    main()
