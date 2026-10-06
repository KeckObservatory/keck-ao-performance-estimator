#!/usr/bin/env python3
"""Measured-SR "Series stats" (Eduardo 2026-10-06): select a series of
frames and log the mean and sample standard deviation of SR, FWHM and WFE.
(1) series_stats.summarize_series: numbers vs numpy (ddof=1), exclusions
(failed message, None, unphysical, saturated), a single frame's stdev is
nan and prints as a dash. (2) The frame list is multi-select. (3) End to
end, offline: three synthetic frames plus one broken file, all selected;
the run logs ONE summary line (n=3, the broken file excluded) whose
mean/stdev equal the per-frame log values, and opens a small pop-up table
with the same numbers; asking again for already-logged frames reuses the
log and measures nothing (Eduardo 2026-10-06). Run headless
(QT_QPA_PLATFORM=offscreen).
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

import numpy as np
from qtcompat import QtCore, QtWidgets

import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui


def pump(cond, timeout=180):
    app = QtWidgets.QApplication.instance()
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        QtCore.QThread.msleep(10)


def fake(sr, fwhm, wfe, ok=True, saturated=False):
    return SimpleNamespace(strehl=sr, fwhm_mas=fwhm, wfe_nm=wfe,
                           ok=ok, error="" if ok else "centroid failed",
                           saturated=saturated,
                           unphysical=ok and not 0.0 < sr <= 1.0)


def make_frame(dirpath, imno, core):
    """A NIRC2-headed synthetic frame (gui_phase29's recipe): a fraction
    `core` of the light in the DL PSF and the rest in a 10-px Gaussian
    halo, so the Strehl is about `core`; noise differs per frame."""
    from astropy.io import fits
    rng = np.random.default_rng(imno)
    psf = engine.nirc2_dl_psf("narrow", "largehex", 2.2705, 171.3,
                              npix=512, pos=(0.3, 0.2))
    yy, xx = np.mgrid[0:512, 0:512] - 256.0
    halo = np.exp(-(xx ** 2 + yy ** 2) / (2 * 10.0 ** 2))
    halo /= halo.sum()
    frame = rng.normal(0.0, 0.5, (1024, 1024))
    frame[256:768, 256:768] += 6e4 * (core * psf / psf.sum()
                                      + (1.0 - core) * halo)
    hdr = fits.Header()
    for k, v in (("CAMNAME", "narrow"), ("PMSNAME", "largehex"),
                 ("EFFWAVE", 2.2705), ("ROTPPOSN", -1.0), ("EL", 42.3),
                 ("COADDS", 1), ("DETGAIN", 8.0), ("AOHATCH", "open"),
                 ("PCUNAME", "telescope"), ("OBJECT", "synthetic"),
                 ("DATE-OBS", "2026-07-23"),
                 ("UTC", f"08:0{imno}:31.49"), ("LSPROP", "yes"),
                 ("RA", "17:17:40.00"), ("DEC", "-22:01:30.5")):
        hdr[k] = v
    flat, _ = engine.load_nirc2_calibration()
    fits.writeto(os.path.join(dirpath, f"n{imno:04d}.fits"),
                 (frame * flat).astype(np.float32), hdr, overwrite=True)


def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # ---- (1) the pure summary -------------------------------------------
    items = [("1", fake(0.40, 50.0, 300.0)), ("2", fake(0.44, 52.0, 290.0)),
             ("3", fake(0.42, 51.0, 295.0)), ("4", "centroid failed"),
             ("5", None), ("6", fake(1.30, 40.0, 0.0)),
             ("7", fake(0.50, 49.0, 280.0, saturated=True)),
             ("8", fake(0.0, 0.0, 0.0, ok=False))]
    st = engine.summarize_series(items)
    assert st.n == 3 and st.labels == ("1", "2", "3"), st
    sr = np.array([0.40, 0.44, 0.42])
    m = st.metrics["strehl"]
    assert abs(m.mean - sr.mean()) < 1e-12
    assert abs(m.std - sr.std(ddof=1)) < 1e-12, (m.std, sr.std(ddof=1))
    assert (m.min, m.max) == (0.40, 0.44)
    assert abs(st.metrics["fwhm_mas"].std - np.std([50, 52, 51], ddof=1)) < 1e-12
    reasons = dict(st.excluded)
    assert reasons == {"4": "centroid failed", "5": "not measured",
                       "6": "unphysical SR +1.300", "7": "saturated",
                       "8": "centroid failed"}, reasons
    lines = engine.format_series_stats(st)
    assert lines[0] == "Series stats: 3 frame(s), 5 excluded", lines
    assert any("SR" in l and "mean 0.420" in l and "stdev 0.020" in l
               for l in lines), lines
    one = engine.format_series_stats(
        engine.summarize_series([("9", fake(0.3, 50.0, 300.0))]))
    assert any("stdev —" in l for l in one), one
    none = engine.format_series_stats(engine.summarize_series([("x", None)]))
    assert "  no usable frames" in none, none
    print("  [ok] summarize_series / format_series_stats")

    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow()

    # ---- (2) multi-select list ------------------------------------------
    assert (win.n2_files.selectionMode()
            == QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
    print("  [ok] frame list is multi-select")

    # ---- (3) end to end -------------------------------------------------
    tmp = tempfile.mkdtemp(prefix="gui49_series_")
    for i, core in ((1, 0.50), (2, 0.42), (3, 0.35)):
        make_frame(tmp, i, core)
    with open(os.path.join(tmp, "n0004.fits"), "w") as fh:
        fh.write("not a FITS file")
    win.n2_instrument.setCurrentText("NIRC2")
    win.n2_path.setText(tmp)
    win._nirc2_refresh_files()
    win.n2_nbg.setValue(0)
    win.n2_autofind.setChecked(True)
    # offline: no guide-star prefetch, and no modal duplicate prompt
    win._nirc2_prefetch_guide_stars = lambda seq, cont: cont()
    QtWidgets.QMessageBox.exec = lambda self: 0
    win.n2_files.selectAll()
    assert len(win.n2_files.selectedItems()) == 4
    win.n2_log.clear()
    win._on_nirc2_series_stats()
    pump(lambda: win.n2_go.isEnabled() and win._n2_series is None)
    log = win.n2_log.toPlainText()
    line = next((l for l in log.splitlines() if l.startswith("Series ")), "")
    assert line.startswith("Series n0001..n0003: n=3 (1 excluded)"), log
    assert "from log" not in line, line
    assert len([l for l in log.splitlines() if l.startswith("Series ")]) == 1
    per_frame = [float(l.split("SR")[1].split()[0])
                 for l in log.splitlines()
                 if l.startswith("Image n000") and "  SR " in l]
    assert len(per_frame) == 3, log
    # the log rounds each frame to 3 decimals; the summary uses full
    # precision, so compare to within that rounding
    mean_txt, std_txt = line.split("SR ")[1].split()[0].split("±")
    assert abs(float(mean_txt) - np.mean(per_frame)) <= 0.0015, (line, per_frame)
    assert abs(float(std_txt) - np.std(per_frame, ddof=1)) <= 0.0015, line
    dlg, table = win._n2_series_dialog, win._n2_series_table
    assert dlg.isVisible() and dlg.width() <= 600, dlg.size()
    assert table.item(0, 0).text() == mean_txt, (table.item(0, 0).text(), mean_txt)
    assert table.item(0, 1).text() == std_txt
    print(f"  [ok] end to end: {line}")

    # same frames again: all taken from the log, nothing re-measured
    def no_run(files=None):
        raise AssertionError(f"re-measured {files}")
    win._nirc2_start = no_run
    win.n2_files.clearSelection()
    for i in range(3):
        win.n2_files.item(i).setSelected(True)
    win.n2_log.clear()
    win._on_nirc2_series_stats()
    line2 = win.n2_log.toPlainText().strip()
    assert line2.startswith("Series n0001..n0003: n=3 (3 from log)"), line2
    assert line2.split("SR ")[1].split()[0] == f"{mean_txt}±{std_txt}", line2
    print("  [ok] logged frames are reused, not re-measured")

    # refused start (no AUTOFIND) leaves no series pending
    win.n2_autofind.setChecked(False)
    win._on_nirc2_series_stats()
    assert getattr(win, "_n2_series", None) is None
    assert "needs AUTOFIND" in win.n2_log.toPlainText()
    print("gui_phase49: all checks passed")


if __name__ == "__main__":
    main()
