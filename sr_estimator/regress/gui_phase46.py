#!/usr/bin/env python3
"""K2 NGS fit per WFS mode (2026-09-28, Eduardo Marin): the K2 57x57 fit is
the HAKA N53 fit, and a PRELIMINARY 29x29-mode fit is selectable (CLI
--ngs-wfs, GUI "K2 NGS WFS" combo on the NGS tab).

Engine contract: NGS_PARAMS["K2"] is N53 and IS the 57x57 entry; the 29x29
fit ties A and w to 57x57, keeps its ceiling under the 0.965x fitting-error
bound, and reproduces the fit's own numbers (haka_29x29_fit.py: 29x29 =
57x57 at R 12.2, x1.5 at R 14 at 0.41" K seeing, gaussian form); K1 has no
29x29 mode (ValueError), and the default is 57x57.
CLI: --ngs-wfs 29x29 changes only the NGS columns and records itself in the
CSV provenance; on K1 it is refused.
GUI: the combo reseeds the four Gompertz fields, is disabled on K1 (where
the mode in force is 57x57), reaches args (the getattr landmine), changes
the NGS estimate, round-trips through the config, the other-telescope
summary stats survive it, and the NGS tab still needs no scrollbar.
Headless, offline.
"""
import csv
import glob
import io
import os, sys, tempfile, time
from contextlib import redirect_stdout
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
from keck_ao_estimator import cli
np = engine.np
DATA = os.path.join(HERE, "data")
N53 = dict(S0=0.747, A=0.661, m0=13.62, w=1.58)
F29 = dict(S0=0.614, A=0.661, m0=14.64, w=1.58)


def pump(cond, timeout=90):
    """Nested-QEventLoop wait (see gui_phase12.pump for why not msleep)."""
    loop = QtCore.QEventLoop()
    t0 = time.time()
    done = False

    def check():
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
    done = True


def settle(n=6):
    app = QtWidgets.QApplication.instance()
    for _ in range(n):
        app.processEvents(); QtCore.QThread.msleep(25)


def engine_contract():
    assert engine.NGS_PARAMS["K2"] == N53, engine.NGS_PARAMS["K2"]
    assert engine.NGS_PARAMS_K2_WFS["57x57"] is engine.NGS_PARAMS["K2"]
    assert engine.NGS_PARAMS_K2_WFS["29x29"] == F29
    assert engine.DEF_NGS_WFS == "57x57"
    assert engine.NGS_WFS_MODES == ("57x57", "29x29")
    p57, p29 = engine.ngs_fit_params("K2"), engine.ngs_fit_params("K2", "29x29")
    assert p29["A"] == p57["A"] and p29["w"] == p57["w"], "A, w must be tied"
    assert p29["S0"] <= 0.965 * p57["S0"], "ceiling above the fitting bound"
    assert engine.ngs_fit_params("K1") is engine.NGS_PARAMS["K1"]
    for bad in (("K1", "29x29"), ("K2", "30x30")):
        try:
            engine.ngs_fit_params(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"ngs_fit_params{bad} must raise")
    try:
        engine.ngs_strehl(0.5, 12.0, "K1", ngs_wfs="29x29")
    except ValueError:
        pass
    else:
        raise AssertionError("K1 29x29 must raise")
    # default == explicit 57x57; K1 unchanged by the new argument
    for R in (8.0, 12.0, 15.0):
        assert engine.ngs_strehl(0.5, R, "K2") == \
            engine.ngs_strehl(0.5, R, "K2", ngs_wfs="57x57")
        assert engine.ngs_strehl(0.5, R, "K1") == \
            engine.ngs_strehl(0.5, R, "K1", ngs_wfs="57x57")
    # the fit's own printed numbers (fit_output_29x29.txt, gaussian, 0.41")
    eps = 0.41 / engine.V2K
    g = lambda R, w: engine.ngs_strehl(eps, R, "K2", ngs_wfs=w,   # noqa: E731
                                       seeing_law="gaussian")
    for R, s29, s57 in ((8, 0.541, 0.650), (12, 0.456, 0.468),
                        (14, 0.282, 0.187), (15, 0.157, 0.061)):
        assert abs(g(R, "29x29") - s29) < 0.0015, (R, g(R, "29x29"), s29)
        assert abs(g(R, "57x57") - s57) < 0.0015, (R, g(R, "57x57"), s57)
    lo, hi = 10.0, 14.0                        # crossover by bisection
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if g(mid, "29x29") < g(mid, "57x57") else (lo, mid)
    assert abs(lo - 12.22) < 0.02, f"crossover R {lo:.3f}, fit says 12.22"
    # overrides still apply on top of the selected mode's fit
    assert engine.ngs_strehl(0.5, 14.0, "K2", ngs_wfs="29x29", ngs_m0=13.62) \
        < engine.ngs_strehl(0.5, 14.0, "K2", ngs_wfs="29x29")
    assert engine.NGS_PARAMS_K2_WFS["29x29"] == F29, "module fit mutated"
    print(f"  [ok] engine: K2 = N53, 29x29 tied A/w, ceiling "
          f"{p29['S0'] / p57['S0']:.3f}x, crossover R {lo:.2f}, K1 refuses")


def run_cli(extra):
    out = tempfile.mkdtemp(prefix="p46_")
    png = os.path.join(out, "run.png")
    a = cli.build_parser().parse_args(
        ["--dimm", os.path.join(DATA, "20260525_dimm.dat"),
         "--mass", os.path.join(DATA, "20260525_mass.dat"),
         "--masspro", os.path.join(DATA, "20260525_masspro.dat"),
         "--out", png, "--force", *extra])
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.main(a)
    (path,) = glob.glob(os.path.join(out, "*.csv"))
    text = open(path).read()
    rows = list(csv.DictReader(l for l in text.splitlines()
                               if not l.startswith("#")))
    return text, rows, buf.getvalue()


def cli_contract():
    t57, r57, _ = run_cli(["--telescope", "K2", "--ngs-faint", "14"])
    t29, r29, out29 = run_cli(["--telescope", "K2", "--ngs-faint", "14",
                               "--ngs-wfs", "29x29"])
    assert "ngs_wfs=29x29(PRELIMINARY_fit)" in t29
    assert "ngs_wfs" not in t57, "default run must not record the mode"
    assert "NGS WFS: 29x29 (PRELIMINARY fit)" in out29
    cols = list(r57[0].keys())
    changed = {c for a, b in zip(r57, r29) for c in cols if a[c] != b[c]}
    assert changed == {"ngs_R8_strehl", "ngs_R14_strehl"}, changed
    m = lambda rows, c: np.nanmean([float(r[c]) for r in rows  # noqa: E731
                                    if r[c] not in ("", "nan")])
    assert m(r29, "ngs_R8_strehl") < m(r57, "ngs_R8_strehl")    # bright: 57 wins
    assert m(r29, "ngs_R14_strehl") > m(r57, "ngs_R14_strehl")  # faint: 29 wins
    try:
        run_cli(["--telescope", "K1", "--ngs-wfs", "29x29"])
    except SystemExit as e:
        assert "K2" in str(e), e
    else:
        raise AssertionError("K1 --ngs-wfs 29x29 must be refused")
    print(f"  [ok] CLI: only NGS columns move (R8 {m(r57, 'ngs_R8_strehl'):.3f}"
          f"->{m(r29, 'ngs_R8_strehl'):.3f}, R14 {m(r57, 'ngs_R14_strehl'):.3f}"
          f"->{m(r29, 'ngs_R14_strehl'):.3f}); provenance recorded; K1 refused")


def preview(win):
    """(dashed-overlay labels, crossing, title-fits) of the NGS fit preview."""
    settle(); win.fit_canvas.draw(); settle()
    ax = win.fit_fig.axes[0]
    dashed = [ln.get_label() for ln in ax.get_lines()
              if ln.get_linestyle() == "--" and not ln.get_label().startswith("_")]
    bb = ax.title.get_window_extent(win.fit_canvas.get_renderer())
    fits = bb.x0 >= 0 and bb.x1 <= win.fit_fig.bbox.width
    return dashed, win._ngs_preview_crossing, fits


def fields(win):
    return dict(S0=win.ngs_s0.value(), A=win.ngs_a.value(),
                m0=win.ngs_m0.value(), w=win.ngs_w.value())


def gui_contract():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1500, 950); win.show(); settle()
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(os.path.join(DATA, "20260525_dimm.dat"))
    win.mass_edit.setText(os.path.join(DATA, "20260525_mass.dat"))
    win.masspro_edit.setText(os.path.join(DATA, "20260525_masspro.dat"))
    win.tel_k2.setChecked(True); settle()
    assert win.ngs_wfs.currentText() == "57x57" and win.ngs_wfs.isEnabled()
    assert fields(win) == N53, fields(win)
    win.ngs_faint.setValue(14.0)
    win._validate(); win.on_run()
    pump(lambda: win.res is not None)
    assert win.args_cached.ngs_wfs == "57x57"
    b57 = float(np.nanmean(win.res.ngs_bright))
    f57 = float(np.nanmean(win.res.ngs_faint))

    # switching the mode reseeds the fields and recomputes
    prev = win.res
    win.ngs_wfs.setCurrentText("29x29")
    pump(lambda: win.res is not prev, timeout=10)
    assert fields(win) == F29, fields(win)
    assert win.args_cached.ngs_wfs == "29x29"
    b29 = float(np.nanmean(win.res.ngs_bright))
    f29 = float(np.nanmean(win.res.ngs_faint))
    assert b29 < b57 and f29 > f57, (b57, b29, f57, f29)
    print(f"  [ok] GUI 29x29: fields reseeded, NGS R8 {b57:.3f}->{b29:.3f}, "
          f"R14 {f57:.3f}->{f29:.3f}")

    # the NGS tab still fits without a scrollbar (dock never scrolls)
    idx = [i for i in range(win.tabs.count()) if win.tabs.tabText(i) == "NGS"]
    assert idx, [win.tabs.tabText(i) for i in range(win.tabs.count())]
    win.tabs.setCurrentIndex(idx[0]); settle(3)
    scroll = win.tabs.widget(idx[0])
    assert not scroll.verticalScrollBar().isVisible(), "NGS tab scrolls"
    assert not scroll.horizontalScrollBar().isVisible()

    # fit preview (mock-up B): the OTHER mode dashed at 0.5", the crossover
    # (R 12.2, 29x29 better fainter) marked, the title not clipped
    dashed, xing, fits = preview(win)            # 29x29 active
    assert dashed == ['57x57 @ 0.5"'], dashed
    assert xing and abs(xing[0] - 12.22) < 0.05 and xing[1] == "29x29", xing
    assert fits, "preview title clipped (29x29)"
    win.ngs_wfs.setCurrentText("57x57"); settle()
    dashed, xing, fits = preview(win)            # 57x57 active
    assert dashed == ['29x29 prelim @ 0.5"'], dashed
    assert xing and abs(xing[0] - 12.22) < 0.05 and xing[1] == "29x29", xing
    assert fits, "preview title clipped (57x57)"
    prev = win.res
    win.ngs_wfs.setCurrentText("29x29")
    pump(lambda: win.res is not prev, timeout=10)
    print(f"  [ok] preview: other mode dashed at 0.5\", crossover R "
          f"{xing[0]:.2f} ({xing[1]} better fainter), title fits")

    # K1: combo disabled, K1 fit loaded, mode in force is 57x57
    prev = win.res
    win.tel_k1.setChecked(True)
    pump(lambda: win.res is not prev and win.prep is not None, timeout=30)
    assert not win.ngs_wfs.isEnabled()
    assert fields(win) == engine.NGS_PARAMS["K1"], fields(win)
    assert win.collect_args("").ngs_wfs == "57x57"
    assert win.args_cached.ngs_wfs == "57x57"
    dashed, xing, fits = preview(win)            # K1: no overlay
    assert dashed == [] and xing is None and fits, (dashed, xing, fits)
    # summary stats' other telescope: K2 takes the combo's mode (29x29) ...
    # (_other_telescope_res returns None on any engine error)
    assert win._other_telescope_res(win.collect_args(""), {}) is not None
    # back on K2 the combo's 29x29 fit returns ...
    prev = win.res
    win.tel_k2.setChecked(True)
    pump(lambda: win.res is not prev and win.prep is not None, timeout=30)
    assert win.ngs_wfs.isEnabled() and fields(win) == F29, fields(win)
    # ... and the other telescope (K1) must not inherit 29x29
    assert win._other_telescope_res(win.collect_args(""), {}) is not None, \
        "K1 summary stats failed from K2 29x29"

    # Reset fit reloads the active mode's fit after an edit
    win.ngs_m0.setValue(13.0); settle()
    win._sync_ngs_fit_fields(force=True)
    assert fields(win) == F29

    # config round trip
    cfg = win._collect_config()
    assert cfg["ngs_wfs"] == "29x29"
    win.ngs_wfs.setCurrentText("57x57"); settle()
    assert fields(win) == N53
    win._apply_config(cfg); settle()
    assert win.ngs_wfs.currentText() == "29x29" and fields(win) == F29
    old = {k: v for k, v in cfg.items() if k not in ("ngs_wfs", "ngs_fit_wfs")}
    win._apply_config(old); settle()        # a pre-mode config -> 57x57
    assert win.ngs_wfs.currentText() == "57x57"
    print("  [ok] GUI: K1 disables the mode, Reset fit, other-telescope stats, "
          "config round trip, NGS tab does not scroll")
    win.close()


def main():
    engine_contract()
    cli_contract()
    gui_contract()
    print("gui_phase46: PASS")


if __name__ == "__main__":
    main()
