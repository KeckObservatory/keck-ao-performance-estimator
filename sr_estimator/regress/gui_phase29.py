#!/usr/bin/env python3
"""NIRC2 measured-Strehl tab (2026-07-23): the IDL strehl_widget remake on
the plot side, driven by the ported engine + Nirc2MeasureWorker.

Checks: tab structure and the width rules on the status label; the
strehl_widget radii-ordering guard; a full GO! run against a synthetic
frame (a shifted DL PSF written as a NIRC2-headed FITS -> measured S ~ 1,
IDL-style log line, readouts and all three canvases populated); the
AUTOFIND-OFF click re-measure path; a missing frame logging an error and
re-enabling GO!; and the nirc2 config round-trip. Fully offline -- the
synthetic frame is generated here; nothing proprietary is read. Run
headless (QT_QPA_PLATFORM=offscreen).
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


def pump(cond, timeout=120):
    app = QtWidgets.QApplication.instance()
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        QtCore.QThread.msleep(10)


def make_frame(dirpath, imno):
    """A NIRC2-headed synthetic frame: the DL PSF, sub-pixel shifted, at
    3e6 counts, on faint noise -- measured Strehl must come out ~1."""
    from astropy.io import fits
    rng = np.random.default_rng(imno)
    psf = engine.nirc2_dl_psf("narrow", "largehex", 2.2705, 171.3,
                              npix=512, pos=(0.3, 0.2))
    frame = rng.normal(0.0, 0.5, (1024, 1024))
    frame[256:768, 256:768] += 3e6 * psf
    hdr = fits.Header()
    hdr["CAMNAME"] = "narrow"
    hdr["PMSNAME"] = "largehex"
    hdr["EFFWAVE"] = 2.2705
    hdr["ROTPPOSN"] = -1.0
    hdr["EL"] = 42.3
    hdr["COADDS"] = 1
    hdr["DETGAIN"] = 8.0
    hdr["AOHATCH"] = "open"
    hdr["PCUNAME"] = "telescope"
    hdr["OBJECT"] = "synthetic"
    hdr["DATE-OBS"] = "2026-07-23"
    hdr["UTC"] = "08:01:31.49"      # 22:01:31 HST on the 22nd
    hdr["LSPROP"] = "yes"
    hdr["RA"] = "17:17:40.00"
    hdr["DEC"] = "-22:01:30.5"
    flat, _ = engine.load_nirc2_calibration()
    fits.writeto(os.path.join(dirpath, f"n{imno:04d}.fits"),
                 (frame * flat).astype(np.float32), hdr, overwrite=True)


def make_crowded(dirpath, imno):
    """The SgrA*-style failure: neighbors planted in the sky annulus."""
    from astropy.io import fits
    rng = np.random.default_rng(imno)
    psf = engine.nirc2_dl_psf("narrow", "largehex", 2.2705, 171.3,
                              npix=512, pos=(0.3, 0.2))
    frame = rng.normal(0.0, 0.5, (1024, 1024))
    frame[256:768, 256:768] += 3e6 * psf
    yy, xx = np.mgrid[0:1024, 0:1024]
    for k in range(12):
        ang = 2 * np.pi * k / 12.0
        cx, cy = 511.5 + 125 * np.cos(ang), 511.5 + 125 * np.sin(ang)
        frame += 3.0e4 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2)
                                  / (2 * 3.0 ** 2)))
    hdr = fits.Header()
    for k, v in (("CAMNAME", "narrow"), ("PMSNAME", "largehex"),
                 ("EFFWAVE", 2.2705), ("ROTPPOSN", -1.0), ("EL", 42.3),
                 ("COADDS", 1), ("DETGAIN", 8.0), ("AOHATCH", "open"),
                 ("PCUNAME", "telescope"), ("OBJECT", "crowdy"),
                 ("COADDS", 50)):
        hdr[k] = v
    flat, _ = engine.load_nirc2_calibration()
    fits.writeto(os.path.join(dirpath, f"n{imno:04d}.fits"),
                 (frame * flat).astype(np.float32), hdr, overwrite=True)


def make_osiris(dirpath):
    """Synthetic OSIRIS imager frame: 2048x2048, star in the central-crop
    region, CURRINST/IFILTER headers -- exercises the OSIRIS routing."""
    from astropy.io import fits
    rng = np.random.default_rng(11)
    psf = engine.nirc2_dl_psf("osiris", "open", 2.169, 38.0, npix=512,
                              pos=(0.3, 0.2))
    yy, xx = np.mgrid[0:512, 0:512]
    halo = np.exp(-(((xx - 255.5) ** 2 + (yy - 255.5) ** 2) / (2 * 80.0 ** 2)))
    halo /= halo.sum()
    frame = rng.normal(0.0, 0.5, (2048, 2048))
    frame[768:1280, 768:1280] += 3e6 * (0.85 * psf + 0.15 * halo)
    hdr = fits.Header()
    hdr["CURRINST"] = "OSIRIS"
    hdr["IFILTER"] = "BrGamma"
    hdr["COADDS"] = 3
    hdr["EL"] = 63.7
    hdr["ROTPPOSN"] = 28.1
    hdr["OBJECT"] = "osi-synth"
    hdr["DATE-OBS"] = "2026-07-23"
    hdr["UTC"] = "05:37:25.70"
    hdr["LSPROP"] = "no"
    hdr["RA"] = 235.32
    hdr["DEC"] = -5.71
    # AO TT-sensor stage == active pointing origin: an ON-AXIS TT star
    # for the TSS-vs-PO odometer (engine.tt_star_offset)
    hdr["AOTSX"] = 1.820
    hdr["AOTSY"] = -11.131
    hdr["POXPOS"] = 0.00182
    hdr["POYPOS"] = -0.01113
    hdr["PONAME"] = "osimg"
    fits.writeto(os.path.join(dirpath, "i260723_a000001.fits"),
                 frame.astype(np.float32), hdr, overwrite=True)


def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow()
    win._nighttime_is_night = lambda: True     # gate-pin (house pattern)
    win.resize(1500, 950)
    win.show()
    app.processEvents()

    # ---- structure + width rules -------------------------------------------
    labels = [win.plot_tabs.tabText(i) for i in range(win.plot_tabs.count())]
    assert "Measured SR" in labels, labels
    assert win.n2_nstars.isEnabled() and win.n2_nstars.value() == 5, \
        "Stars spin drives the field-map auto-find"
    assert not hasattr(win, "n2_status"), \
        "green status row removed (mode lives in the image title)"
    assert win.n2_nstars.minimum() == 0 \
        and win.n2_nstars.specialValueText() == "Auto", \
        "Stars spin at 0 must read Auto (quality-gated count)"
    assert not win.n2_auto_rad.isChecked(), \
        "auto aperture defaults off (IDL-faithful fixed radius)"
    assert win.n2_split.count() == 2, "view/results splitter present"
    assert win.n2_map_popout.text() == "Pop out"
    # photometry toggles must not clip: each gets at least its size hint
    # (the overpacked single row truncated "Robust sky (σ-clip)" to
    # "Robust" at real font metrics)
    win.plot_tabs.setCurrentIndex(labels.index("Measured SR"))
    app.processEvents()
    for w in (win.n2_robust_sky, win.n2_auto_rad, win.n2_pick_sky):
        assert w.width() >= w.sizeHint().width(), \
            f"clipped photometry toggle: {w.text()!r} " \
            f"{w.width()} < {w.sizeHint().width()}"
    print("  [ok] tab present, nstars locked, no status row, toggles unclipped")

    # ---- radii-ordering guard (strehl_widget.pro's GO check) ---------------
    win.n2_path.setText("/nonexistent")
    win.n2_photrad.setValue(2.0)     # > inner bg radius 1.2 -> refuse
    win._on_nirc2_go()
    assert "photometry radius" in win.n2_log.toPlainText()
    assert win._n2_worker is None, "worker must not start on bad radii"
    win.n2_photrad.setValue(1.0)
    print("  [ok] radii-ordering guard refuses without starting the worker")

    # ---- full GO! run on a synthetic frame ---------------------------------
    tmp = tempfile.mkdtemp(prefix="gui29_nirc2_")
    make_frame(tmp, 7)
    win.n2_path.setText(tmp)
    win.n2_im1.setValue(7)
    win.n2_nim.setValue(1)
    win.n2_nbg.setValue(0)
    win._on_nirc2_go()
    assert not win.n2_go.isEnabled(), "GO! must disable while measuring"
    pump(lambda: win.n2_go.isEnabled())
    log = win.n2_log.toPlainText()
    assert "Image 7  SR " in log, log
    s = float(win.n2_strehl_out.text())
    assert abs(s - 1.0) < 0.06, f"synthetic DL frame measured S={s}"
    assert win.n2_fwhm_out.text() and win.n2_wfe_out.text()
    assert win.n2_fig.axes and win.n2_fig.axes[0].images, "main image drawn"
    assert win.n2_fig_dl.axes and win.n2_fig_dl.axes[0].images, "DL cutout drawn"
    assert win.n2_fig_star.axes and win.n2_fig_star.axes[0].images, "star cutout"
    circles = [ln for ln in win.n2_fig.axes[0].lines]
    assert len(circles) == 3, "photometry + two annulus circles drawn"
    print(f"  [ok] GO! measured the synthetic frame: S={s:.3f}, "
          "log line + readouts + 3 canvases + aperture circles")

    # ---- display stretch: display-only, ACTUALLY re-renders ---------------
    import io

    def _snap():
        buf = io.BytesIO()
        win.n2_fig.savefig(buf, format="png", dpi=50)
        return buf.getvalue()

    assert win.n2_stretch.currentText() == "IDL ±5σ"
    assert not win.n2_white.isEnabled()
    s_before = win.n2_strehl_out.text()
    png_idl = _snap()
    win.n2_stretch.setCurrentText("asinh")
    assert win.n2_white.isEnabled(), "white point enables for non-IDL modes"
    png_asinh = _snap()
    assert png_asinh != png_idl, \
        "stretch change must change the RENDERED image (the bug Eduardo " \
        "caught: the measure path bypassed the stretch-aware redraw)"
    assert len(win.n2_fig.axes[0].lines) == 3, "circles survive the redraw"
    win.n2_white.setValue(98.0)
    assert _snap() != png_asinh, "white-point change must re-render too"
    assert win.n2_strehl_out.text() == s_before, "stretch never re-measures"
    c0 = win._collect_config()
    assert c0["nirc2"]["stretch"] == "asinh" and c0["nirc2"]["white"] == 98.0
    win.n2_stretch.setCurrentText("IDL ±5σ")
    win._apply_config(c0)
    assert win.n2_stretch.currentText() == "asinh"
    win.n2_stretch.setCurrentText("IDL ±5σ")
    print("  [ok] stretch controls: redraw-only, white-point gating, config")

    # ---- AUTOFIND OFF: click re-measures at the clicked position -----------
    win.n2_autofind.setChecked(False)
    x0, y0 = win._n2_image.shape[1] // 2 - 1, win._n2_image.shape[0] // 2 - 1
    n_before = len(win.n2_log.toPlainText().splitlines())
    ev = SimpleNamespace(xdata=float(x0), ydata=float(y0),
                         inaxes=win.n2_fig.axes[0])
    # This click re-measures a frame ALREADY in the image log, so it hits
    # _nirc2_ask_duplicate_paused -- a modal QMessageBox.exec() that blocks
    # forever headless (it is what timed CI out on 2026-07-28). Stub it the
    # way gui_phase24 does, and assert it actually fired so the duplicate
    # path stays covered rather than merely disarmed. clickedButton() is
    # None under the stub, so the handler takes its declared default,
    # "skip".
    _dup_prompts = []
    _orig_exec = QtWidgets.QMessageBox.exec
    QtWidgets.QMessageBox.exec = lambda self: _dup_prompts.append(1) or 0
    try:
        win._on_nirc2_click(ev)
    finally:
        QtWidgets.QMessageBox.exec = _orig_exec
    assert _dup_prompts, \
        "re-measuring a logged frame must raise the duplicate prompt"
    n_after = len(win.n2_log.toPlainText().splitlines())
    assert n_after == n_before + 1, "click must append one measurement"
    s2 = float(win.n2_strehl_out.text())
    assert abs(s2 - 1.0) < 0.06, f"click re-measure S={s2}"
    # autofind ON ignores clicks
    win.n2_autofind.setChecked(True)
    win._on_nirc2_click(ev)
    assert len(win.n2_log.toPlainText().splitlines()) == n_after, \
        "autofind ON must ignore canvas clicks"
    print(f"  [ok] AUTOFIND-off click re-measures (S={s2:.3f}); ON ignores")

    # ---- AUTOFIND off: GO! must NOT measure — display and wait for click ---
    win.n2_autofind.setChecked(False)
    n_lines = win.n2_log.toPlainText().count("Image 7  SR ")
    win._on_nirc2_go()
    pump(lambda: win.n2_go.isEnabled())
    assert win.n2_log.toPlainText().count("Image 7  SR ") == n_lines, \
        "AUTOFIND off must not auto-measure on GO!"
    title = win.n2_fig.axes[0].get_title()
    assert "CLICK ON THE STAR" in title and "LGS" in title, title
    assert win.n2_strehl_out.text() == "", "measured boxes cleared"
    assert win.n2_object.text() == "synthetic", "identity shown before click"
    # cutout captions + live pick-zoom under the cursor
    assert win.n2_cap_dl.text() == "MODEL PSF"
    win._on_nirc2_motion(ev)
    assert win.n2_cap_star.text() == "PICK ZOOM"
    assert win.n2_fig_star.axes and win.n2_fig_star.axes[0].images, \
        "pick zoom must draw a cutout"
    win._on_nirc2_click(ev)
    assert win.n2_log.toPlainText().count("Image 7  SR ") == n_lines + 1, \
        "click after AUTOFIND-off GO! must measure"
    assert win.n2_cap_star.text() == "MEASURED STAR", "caption restored"
    assert "CLICK ON THE STAR" not in win.n2_fig.axes[0].get_title()
    assert "LGS" in win.n2_fig.axes[0].get_title(), "mode stays in the title"
    win._on_nirc2_motion(ev)     # zoom must stay LOCKED after the click
    assert win.n2_cap_star.text() == "MEASURED STAR", \
        "pick zoom must lock once the star is clicked"
    win.n2_autofind.setChecked(True)
    print("  [ok] AUTOFIND-off GO! displays without measuring; click measures")

    # ---- file list: ALL FITS shown; double-click measures; non-NIRC2
    # frames are refused by header (INSTRUME/CAMNAME rule) ------------------
    import shutil
    from astropy.io import fits as _fits
    shutil.copy(os.path.join(tmp, "n0007.fits"),
                os.path.join(tmp, "N2.test_7.fits"))     # KOA-style name
    _fits.writeto(os.path.join(tmp, "i0001_osiris.fits"),
                  np.zeros((64, 64), dtype=np.float32))  # no INSTRUME/CAMNAME
    win._nirc2_refresh_files()
    names = [win.n2_files.item(i).text() for i in range(win.n2_files.count())]
    assert names == ["N2.test_7.fits", "i0001_osiris.fits", "n0007.fits"], names

    def item_by(name):
        return next(win.n2_files.item(i) for i in range(win.n2_files.count())
                    if win.n2_files.item(i).text() == name)

    win.n2_im1.setValue(1)
    win._on_nirc2_file_dclick(item_by("n0007.fits"))
    assert win.n2_im1.value() == 7 and win.n2_nim.value() == 1
    pump(lambda: win.n2_go.isEnabled())
    assert win.n2_log.toPlainText().count("Image 7  SR ") >= 2
    win._on_nirc2_file_dclick(item_by("N2.test_7.fits"))
    pump(lambda: win.n2_go.isEnabled())
    assert "Image N2.test_7  SR " in win.n2_log.toPlainText()
    win._on_nirc2_file_dclick(item_by("i0001_osiris.fits"))
    pump(lambda: win.n2_go.isEnabled())
    assert "only NIRC2 and OSIRIS frames are supported" in win.n2_log.toPlainText()
    print("  [ok] file list shows all FITS; double-click measures numbered "
          "and KOA-named frames; non-NIRC2 refused by header")

    # ---- measured vs predicted: LSPROP mode, time match, lambda convert ----
    import datetime as dt
    base = dt.datetime(2026, 7, 22, 22, 0, 0)
    win.res = SimpleNamespace(
        times=[base], ngs_bright=np.array([0.60]),
        p_times=[base, base + dt.timedelta(minutes=2)],
        sr_single=np.array([0.30, 0.32]), sr_ltao=np.array([0.35, 0.36]))
    win.prep = SimpleNamespace(lam_nm=2196.0, tomography_on=False)
    win.args_cached = SimpleNamespace(telescope="K2")
    win.n2_autofind.setChecked(False)
    win._on_nirc2_click(ev)
    tip = win.n2_pred_sr.toolTip()
    assert "single-LGS" in tip and "22:02 HST" in tip, tip
    s_conv = 0.32 ** ((2196.0 / 2270.5) ** 2)
    assert win.n2_pred_sr.text() == f"{s_conv:.3f}", win.n2_pred_sr.text()
    assert win.n2_pred_fwhm.text() == "—", "no FWHM series -> em dash"
    assert win.n2_dsr.text().startswith("+"), win.n2_dsr.text()
    log = win.n2_log.toPlainText()
    assert f"predicted   SR {s_conv:.3f}" in log, log.splitlines()[-1]
    assert "ΔSR +" in log and "ΔFWHM" in log, log.splitlines()[-1]
    assert "(single-LGS" not in log, "no mode suffix in the log lines"
    # K1 prediction warns; LTAO series used when THIS FRAME's own decoded
    # AO mode (AOOPSMOD=3) says LTAO -- not whichever mode the currently-
    # loaded run happens to be configured for (Eduardo 2026-07-28: "if
    # the mode is LTAO the estimator changes from single laser to LTAO")
    import dataclasses as _dc
    orig_params = win._n2_params
    win._n2_params = _dc.replace(win._n2_params, aoopsmod=3)
    win.prep = SimpleNamespace(lam_nm=2196.0, tomography_on=True)
    win.args_cached = SimpleNamespace(telescope="K1")
    win._on_nirc2_click(ev)
    tip = win.n2_pred_sr.toolTip()
    assert "prediction is K1" in tip and "LTAO" in tip, tip
    win._n2_params = orig_params
    # out-of-tolerance time -> boxes empty, tooltip says so
    win.res.p_times = [base - dt.timedelta(hours=5)] * 2
    win.args_cached = SimpleNamespace(telescope="K2")
    win.prep = SimpleNamespace(lam_nm=2196.0, tomography_on=False)
    win._on_nirc2_click(ev)
    assert win.n2_pred_sr.text() == "" and win.n2_dsr.text() == ""
    assert "no predicted sample within" in win.n2_pred_sr.toolTip()
    assert "predicted: no predicted sample within" in win.n2_log.toPlainText()
    win.res = win.prep = win.args_cached = None
    print("  [ok] measured-vs-predicted boxes: mode from LSPROP, nearest-"
          "sample match, wavelength conversion, K1 warning, tolerance empty")

    # ---- OBJECT/RA/Dec readouts + target auto-load / Set-as-target ---------
    assert win.n2_object.text() == "synthetic"
    assert win.n2_ra.text() == "17:17:40.00" and win.n2_dec.text() == "-22:01:30.5"
    assert win.n2_set_target.isEnabled()
    win._targets = [{"name": "synthetic", "ra": "17:00:00",
                     "dec": "-20:00:00"}]
    win._refresh_target_combo()
    win.tname_edit.setText("")
    win._on_nirc2_click(ev)         # re-measure -> auto-select from the list
    assert win.tname_edit.text() == "synthetic", win.tname_edit.text()
    assert "target set from list" in win.n2_log.toPlainText()
    win._targets = []
    win._refresh_target_combo()
    win.tname_edit.setText("")
    win.ra_edit.setText("")
    win.dec_edit.setText("")
    win._on_nirc2_set_target()   # no list -> name AND header coords filled
    assert win.tname_edit.text() == "synthetic"
    assert win.ra_edit.text() == "17:17:40.00"
    assert win.dec_edit.text() == "-22:01:30.5"
    print("  [ok] OBJECT/RA/Dec boxes; auto target from loaded list; "
          "Set-as-target fills name + header coords when no list matches")

    # ---- Match SR tool: stats + field map jump to the frame's moment -------
    win.wl_enable.setChecked(False)
    win._nirc2_match_tool("stats")
    assert win.stats_cond.currentText() == "specific time"
    assert win.stats_time.time() == QtCore.QTime(22, 1), \
        win.stats_time.time()      # frame UTC 08:01 -> 22:01 HST
    assert win.wl_enable.isChecked(), "match must enable the nm override"
    assert abs(win.wl_nm.value() - 2270.5) <= 0.5, \
        "match must set the science wavelength to the frame EFFWAVE " \
        "(nm spin has 0 decimals -> nearest nm)"
    win.utc_cb.setChecked(True)    # display zone: entered as UT clock
    win._nirc2_match_tool("fm")
    assert win.fm_cond.currentText() == "specific time"
    assert win.fm_time.time() == QtCore.QTime(8, 1), win.fm_time.time()
    win.utc_cb.setChecked(False)
    saved_t = win._n2_frame_hst
    win._n2_frame_hst = None
    win._nirc2_match_tool("stats")
    assert "Measured SR" in win.status.text(), win.status.text()
    win._n2_frame_hst = saved_t
    print("  [ok] Match SR tool: period/conditions -> frame time, UTC-mode "
          "aware, no-measurement message")

    # ---- crowded field: CROWDED tag, robust sky, Pick sky ------------------
    make_crowded(tmp, 9)
    win._nirc2_refresh_files()
    win.n2_autofind.setChecked(True)
    win.n2_im1.setValue(9)
    win.n2_nim.setValue(1)
    win.n2_robust_sky.setChecked(False)
    win._on_nirc2_go()
    pump(lambda: win.n2_go.isEnabled())
    title = win.n2_fig.axes[0].get_title()
    assert "CROWDED" not in title, "warnings must not be in the title anymore"
    warn = win.n2_warn.text()
    assert "CROWDED" in warn, warn
    assert "UNPHYSICAL SR" in warn, warn      # S < 0 is impossible: say so
    assert win.n2_warn.wordWrap() and win.n2_warn.minimumWidth() == 0
    s_def = float(win.n2_strehl_out.text())
    assert s_def <= 0.0, s_def
    win.n2_robust_sky.setChecked(True)
    win._on_nirc2_go()
    pump(lambda: win.n2_go.isEnabled())
    s_rob = float(win.n2_strehl_out.text())
    assert abs(s_rob - 1.0) < 0.08 and abs(s_rob - 1.0) < abs(s_def - 1.0), \
        (s_def, s_rob)
    # Pick sky: arm, click an empty corner, value sticks; second press clears
    win.n2_pick_sky.setChecked(True)
    win._on_nirc2_pick_sky()
    ev_corner = SimpleNamespace(xdata=40.0, ydata=40.0,
                                inaxes=win.n2_fig.axes[0])
    win._on_nirc2_click(ev_corner)
    assert win._n2_sky_override is not None
    assert "sky picked at (40, 40)" in win.n2_log.toPlainText()
    assert win.n2_pick_sky.text().startswith("Sky ")
    win.n2_robust_sky.setChecked(False)
    win._on_nirc2_go()
    pump(lambda: win.n2_go.isEnabled())
    s_pick = float(win.n2_strehl_out.text())
    assert abs(s_pick - 1.0) < 0.08, s_pick
    win._on_nirc2_pick_sky()          # clear
    assert win._n2_sky_override is None
    assert win.n2_pick_sky.text() == "Pick sky"
    c = win._collect_config()
    assert c["nirc2"]["robust_sky"] is False
    win.n2_robust_sky.setChecked(True)
    assert win._collect_config()["nirc2"]["robust_sky"] is True
    win._apply_config(c)
    assert not win.n2_robust_sky.isChecked()
    win.n2_im1.setValue(7)
    print("  [ok] CROWDED tag; robust sky recovers "
          f"(S {s_def:.3f} -> {s_rob:.3f}); Pick sky sets/clears "
          f"(S {s_pick:.3f}); robust_sky config round-trip")

    # ---- OSIRIS frame: routed by CURRINST, measured via double-click -------
    make_osiris(tmp)
    win._nirc2_refresh_files()
    win._on_nirc2_file_dclick(item_by("i260723_a000001.fits"))
    pump(lambda: win.n2_go.isEnabled())
    assert "Image i260723_a000001  SR " in win.n2_log.toPlainText()
    assert win._n2_image.shape == (2048, 2048), \
        "GUI must show the WHOLE OSIRIS frame, not the tool's crop"
    s_osi = float(win.n2_strehl_out.text())
    assert 0.7 < s_osi < 1.0, f"OSIRIS synthetic (halo) S={s_osi}"
    assert "NGS" in win.n2_fig.axes[0].get_title()
    assert win.n2_object.text() == "osi-synth"
    print(f"  [ok] OSIRIS frame routed and measured (S={s_osi:.3f}, "
          "no flat/mask path, NGS from LSPROP)")

    # ---- TT star: TSS-vs-pointing-origin odometer + catalogue ring match ---
    off = engine.tt_star_offset({"AOTSX": 1.820, "AOTSY": -11.131,
                                 "POXPOS": 0.00182, "POYPOS": -0.01113,
                                 "PONAME": "osimg"})
    assert off is not None and off["on_axis"] and off["sep_arcsec"] < 0.01
    off2 = engine.tt_star_offset({"AOTSX": 12.0, "AOTSY": -11.13,
                                  "POXPOS": 0.00182, "POYPOS": -0.01113})
    assert not off2["on_axis"]
    assert abs(off2["sep_arcsec"] - (12.0 - 1.82) * 1.375) < 0.01
    assert engine.tt_star_offset({}) is None, "no AO keywords -> None"
    fake = [dict(id="RING", dec=-5.71, mags={"R": 15.6},
                 ra=235.32 + 15.0 / 3600.0 / np.cos(np.radians(-5.71))),
            dict(id="CENTER", ra=235.32, dec=-5.71, mags={"R": 12.0})]
    m = engine.tt_ring_match(fake, 235.32, -5.71, 15.0)
    assert [c["star"]["id"] for c in m] == ["RING"], m
    assert engine.best_mag(m[0]["star"]) == ("R", 15.6)
    # GUI: the OSIRIS synth carries the keywords (on-axis) -> button armed;
    # a canned catalogue worker stands in for the Vizier round-trip
    assert win.n2_tt_star.isEnabled(), "TT star button must arm"
    import keck_ao_estimator.gui.workers as _wk

    class FakeCat(QtCore.QObject):
        done = _wk.Signal(str, object, str)

        def __init__(self, cat, ra, dec, radius, parent=None):
            super().__init__(parent)
            self._cat, self._ra, self._dec = cat, ra, dec

        def start(self):
            self.done.emit(self._cat, [dict(id="TT-1", ra=self._ra,
                                            dec=self._dec,
                                            mags={"R": 15.6})], "")
    orig_cat = _wk.CatalogFetchWorker
    _wk.CatalogFetchWorker = FakeCat
    try:
        win._on_nirc2_tt_star()
        app.processEvents()
        lbl = win.n2_ttstar_out.text()
        assert "ON-AXIS" in lbl and "TT-1" in lbl and "R=15.6" in lbl, lbl
        assert "TT star odometer" in win.n2_log.toPlainText()
    finally:
        _wk.CatalogFetchWorker = orig_cat
    print("  [ok] TT star: TSS-vs-PO odometer (on-axis identity), catalogue "
          "ring match delivers the R magnitude; keyword-less frames refuse")

    # ---- measured field map: auto-find, add-by-click, clear ----------------
    assert win.n2_view_tabs.tabText(0) == "Image"
    assert win.n2_view_tabs.tabText(1) == "Field map"
    win._on_nirc2_measure_field()
    assert not win.n2_field_btn.isEnabled(), "field run disables the button"
    pump(lambda: win.n2_field_btn.isEnabled())
    assert win.n2_field_btn.text() == "Measure field", "progress text reset"
    assert len(win._n2_field) == 1, len(win._n2_field)   # one star planted
    assert win.n2_map_fig.axes and win.n2_map_fig.axes[0].collections, \
        "map scatter drawn"
    assert "field: kept 1 of 5 requested" in win.n2_log.toPlainText()
    assert "field exhausted above the detection floors" in \
        win.n2_log.toPlainText(), "smart stop must be reported"
    assert win.n2_cap_star.text() == "MEASURED STAR", "flash caption reset"
    win.n2_add_star.setChecked(True)
    ev_star = SimpleNamespace(xdata=1023.0, ydata=1023.0,
                              inaxes=win.n2_fig.axes[0])
    win._on_nirc2_click(ev_star)
    assert len(win._n2_field) == 2, "add-by-click appends to the map"
    assert "star added (2 on the map)" in win.n2_log.toPlainText()
    win.n2_add_star.setChecked(False)
    win._on_nirc2_field_clear()
    assert len(win._n2_field) == 0
    assert not win.n2_map_fig.axes[0].collections, "cleared map is empty"
    assert win._collect_config()["nirc2"]["nstars"] == 5
    # FWHM metric: same map, brighter=better reversed scale
    win.n2_add_star.setChecked(True)
    win._on_nirc2_click(ev_star)          # repopulate one point
    win.n2_add_star.setChecked(False)
    win.n2_map_metric.setCurrentText("FWHM (mas)")
    assert win.n2_map_fig.axes[0].collections, "FWHM map drawn"
    cb_labels = [a.get_ylabel() for a in win.n2_map_fig.axes[1:]]
    assert any("FWHM" in lb for lb in cb_labels), cb_labels
    assert win._collect_config()["nirc2"]["map_metric"] == "FWHM (mas)"
    win.n2_map_metric.setCurrentText("SR")

    # pop-out: its own resizable figure, redrawn with the embedded map,
    # detached again on close
    win._on_nirc2_map_popout()
    assert win._n2_map_dialog.isVisible()
    ext_fig, _ = win._n2_map_ext
    assert ext_fig.axes and ext_fig.axes[0].collections, \
        "pop-out draws the current map"
    win._on_nirc2_field_clear()
    assert not ext_fig.axes[0].collections, \
        "map redraws propagate into the open pop-out"
    win._n2_map_dialog.close()
    app.processEvents()
    assert win._n2_map_ext is None, "closing the pop-out detaches it"

    # field statistics readout: peak/mean always; theta0 needs >= 4 stars;
    # the pop-out mirrors the text; the field run logged a stats line
    assert win.n2_field_stats.wordWrap()
    win.n2_add_star.setChecked(True)
    win._on_nirc2_click(ev_star)              # repopulate one star
    win.n2_add_star.setChecked(False)
    stats_txt = win.n2_field_stats.text()
    assert "peak SR" in stats_txt and "mean" in stats_txt, stats_txt
    assert "\u03b8\u2080" in stats_txt and "needs" in stats_txt, stats_txt
    win._on_nirc2_map_popout()
    assert win._n2_map_ext_stats.text() == stats_txt, \
        "pop-out mirrors the stats readout"
    win._n2_map_dialog.close()
    app.processEvents()
    assert win._n2_map_ext_stats is None if hasattr(
        win, "_n2_map_ext_stats") else True
    assert "field stats: peak SR" in win.n2_log.toPlainText(), \
        "field run logs the statistics line"
    win._on_nirc2_field_clear()
    assert win.n2_field_stats.text() == "", "cleared map clears the stats"

    # peak/downhill map labels: inject a 5-star field with a real slope
    # (clones of the measured star at synthetic positions/SR values)
    import dataclasses
    base = win._n2_last_draw[1]
    assert base is not None and base.ok
    win._n2_field = [
        dataclasses.replace(base, x=200.0, y=200.0, strehl=0.55,
                            sr_err=0.005),
        dataclasses.replace(base, x=800.0, y=250.0, strehl=0.40,
                            sr_err=0.005),
        dataclasses.replace(base, x=300.0, y=800.0, strehl=0.35,
                            sr_err=0.005),
        dataclasses.replace(base, x=750.0, y=760.0, strehl=0.28,
                            sr_err=0.005),
        dataclasses.replace(base, x=520.0, y=480.0, strehl=0.42,
                            sr_err=0.005),
    ]
    win._nirc2_draw_map()
    map_texts = [t.get_text() for t in win.n2_map_fig.axes[0].texts]
    assert any(t == "peak" for t in map_texts), map_texts
    assert any(t.startswith("downhill") and "SR/" in t
               for t in map_texts), \
        "gradient arrow must be labelled with its magnitude"
    assert "gradient" in win.n2_field_stats.text()

    # ---- click-to-inspect + reject (Eduardo, the i260226_a017005 edge
    #      outlier): clicking a map point shows that star's measurement in
    #      the Results block + MEASURED STAR panel and arms Reject star;
    #      rejecting removes it and REGENERATES every field statistic
    sc0 = win.n2_map_fig.axes[0].collections[0]
    assert sc0.get_picker() == 6, "map points must be pickable"
    n5 = len(win._n2_field)
    win._on_nirc2_map_pick(SimpleNamespace(ind=[0]))
    assert win._n2_sel_star == 0
    assert win.n2_reject_star.isEnabled(), "selection arms Reject star"
    assert win.n2_strehl_out.text() == "0.550", win.n2_strehl_out.text()
    assert win.n2_cap_star.text().startswith("FIELD STAR 1"), \
        win.n2_cap_star.text()
    win._on_nirc2_reject_star()
    assert len(win._n2_field) == n5 - 1
    assert win._n2_sel_star is None and not win.n2_reject_star.isEnabled()
    assert "rejected by user" in win.n2_log.toPlainText()
    assert win._n2_field_st is not None and win._n2_field_st.n == n5 - 1, \
        "field statistics must regenerate without the rejected star"
    assert abs(win._n2_field_st.peak_sr - 0.42) < 1e-9, \
        "the peak must move off the rejected 0.55 star"
    print("  [ok] map click selects + inspects a field star; Reject star "
          "drops it and the field statistics regenerate")
    win._on_nirc2_field_clear()

    # Auto star count: 0 = quality decides; the one planted star is bright
    # (tiny propagated SR noise) so it is kept and the run reports the gate
    win.n2_nstars.setValue(0)
    win._on_nirc2_measure_field()
    pump(lambda: win.n2_field_btn.isEnabled())
    assert len(win._n2_field) >= 1, "auto mode keeps the quality star"
    assert all(r.sr_err <= engine.SR_ERR_MAX for r in win._n2_field)
    assert "quality star(s) — auto stop at SR noise" in \
        win.n2_log.toPlainText(), "auto summary names the gate"
    assert win._collect_config()["nirc2"]["nstars"] == 0
    win.n2_nstars.setValue(5)
    win._on_nirc2_field_clear()
    print("  [ok] field map: async auto-find with progress+flash, "
          "add-by-click, clear, nstars config, FWHM metric, pop-out, "
          "Auto star count")

    # ---- field rejection: saturated star never reaches the map -------------
    win.n2_autofind.setChecked(True)
    win._on_nirc2_file_dclick(item_by("n0007.fits"))    # COADDS=1 -> saturated
    pump(lambda: win.n2_go.isEnabled())
    assert len(win._n2_field) == 0, "new frame clears the map"
    win.n2_add_star.setChecked(True)
    ev_c = SimpleNamespace(xdata=float(win._n2_image.shape[1] // 2 - 1),
                           ydata=float(win._n2_image.shape[0] // 2 - 1),
                           inaxes=win.n2_fig.axes[0])
    win._on_nirc2_click(ev_c)
    assert len(win._n2_field) == 0, "saturated star must NOT join the map"
    assert "not added — saturated" in win.n2_log.toPlainText()
    win.n2_add_star.setChecked(False)
    print("  [ok] field rejection: saturated (and unphysical) stars are "
          "refused with a reason")

    # ---- EDGE guard: auto-aperture must not optimize on a clipped curve ----
    # geometry: interior ~0, mid-edge ~0.5, corner ~0.75 of the disc off
    assert engine.aperture_edge_clip_frac((1000, 1000), 500, 500, 100) < 0.01
    assert abs(engine.aperture_edge_clip_frac((1000, 1000), 500, 0, 100)
               - 0.5) < 0.05
    assert abs(engine.aperture_edge_clip_frac((1000, 1000), 0, 0, 100)
               - 0.75) < 0.05
    # a star 30 px from the array edge: the sky annulus is mostly
    # off-array, so the growth curve is truncated -- it settles early and
    # inflates peak/flux (the real i260226_a017005 star read SR 0.50 auto
    # vs 0.24 fixed).  auto_radius must FALL BACK to the caller's fixed
    # radius and the result must carry the EDGE flag.
    from astropy.io import fits as _fits
    hdr_e = _fits.Header()
    for k, v in (("CAMNAME", "narrow"), ("PMSNAME", "largehex"),
                 ("EFFWAVE", 2.2705), ("ROTPPOSN", -1.0), ("EL", 42.3),
                 ("COADDS", 1), ("DETGAIN", 8.0), ("AOHATCH", "open"),
                 ("PCUNAME", "telescope")):
        hdr_e[k] = v
    par_e = engine.nirc2_frame_params(hdr_e)
    rng_e = np.random.default_rng(42)
    psf_e = engine.nirc2_dl_psf("narrow", "largehex", 2.2705, 171.3,
                                npix=512)
    fr_e = rng_e.normal(0.0, 0.5, (1024, 1024))
    fr_e[0:286, 256:768] += 3e6 * psf_e[226:512, :]      # core at y=30
    r_edge = engine.measure_strehl(fr_e, params=par_e, pos=(512.0, 30.0),
                                   robust_sky=True, auto_radius=True)
    assert r_edge.ok, r_edge.error
    assert r_edge.edge and r_edge.edge_clip > engine.EDGE_CLIP_WARN_FRAC, \
        (r_edge.edge, r_edge.edge_clip)
    r_fix = engine.measure_strehl(fr_e, params=par_e, pos=(512.0, 30.0),
                                  robust_sky=True, auto_radius=False)
    assert abs(r_edge.photrad_used_arcsec
               - r_fix.photrad_used_arcsec) < 1e-9, \
        "edge star must keep the caller's fixed radius (no optimization)"
    assert abs(r_edge.strehl - r_fix.strehl) < 1e-9, \
        "auto at the edge must equal the fixed-aperture measurement"
    fr_i = rng_e.normal(0.0, 0.5, (1024, 1024))
    fr_i[256:768, 256:768] += 3e6 * psf_e
    r_int = engine.measure_strehl(fr_i, params=par_e, pos=(512.0, 512.0),
                                  robust_sky=True, auto_radius=True)
    assert r_int.ok and not r_int.edge and r_int.edge_clip < 1e-9, \
        (r_int.edge_clip,)
    print(f"  [ok] EDGE guard: clip geometry, auto-radius fallback at the "
          f"edge (clip {r_edge.edge_clip:.2f}, SR {r_edge.strehl:.3f} == "
          f"fixed), interior unaffected")

    # ---- autofind fallback: brightest pixel on an unmeasurable plateau ----
    # the 20260528 M13 failure: the frame's brightest pixel was the TT
    # star's saturated/bled plateau -- flat-topped, so the centroid
    # degenerates and autofind died with "centroid failed". The worker
    # must fall back to the detection-floor star finder and measure a
    # real star, with a log note.
    tmp_pl = tempfile.mkdtemp(prefix="gui29_bleed_")
    make_frame(tmp_pl, 9)
    from astropy.io import fits as _fits_pl
    fpath = os.path.join(tmp_pl, "n0009.fits")
    arr = np.asarray(_fits_pl.getdata(fpath), dtype=np.float32)
    hdr_pl = _fits_pl.getheader(fpath)
    # a 2-px-wide charge-bleed stripe, far brighter than the star --
    # reproduces the real failure mode (a flat 70x70 plateau does NOT:
    # cntrd survives it and returns a saturated junk measurement,
    # which is displayed with its SATURATED banner as intended)
    arr[60:260, 100:102] = float(arr.max()) * 50.0
    _fits_pl.writeto(fpath, arr, hdr_pl, overwrite=True)
    win.n2_path.setText(tmp_pl)
    win.n2_im1.setValue(9)
    win.n2_nim.setValue(1)
    win.n2_autofind.setChecked(True)
    win._on_nirc2_go()
    pump(lambda: win.n2_go.isEnabled())
    log_pl = win.n2_log.toPlainText()
    assert "fell back to a detected star" in log_pl, \
        "autofind must report the bleed-stripe fallback"
    assert win.n2_strehl_out.text() != "", "fallback must measure"
    assert float(win.n2_strehl_out.text()) > 0.5, \
        f"fallback should land on a real star (S={win.n2_strehl_out.text()})"
    win.n2_path.setText(tmp)
    print("  [ok] autofind falls back to a detected star when the "
          "brightest pixel is unmeasurable (charge-bleed stripe)")

    # ---- missing frame: error logged, GO! re-enabled -----------------------
    win.n2_im1.setValue(99)
    win._on_nirc2_go()
    pump(lambda: win.n2_go.isEnabled())
    assert "Image 99" in win.n2_log.toPlainText()
    print("  [ok] missing frame logs an error and re-enables GO!")

    # ---- config round-trip --------------------------------------------------
    win.n2_im1.setValue(207)
    win.n2_photrad.setValue(0.8)
    c = win._collect_config()
    assert c["nirc2"]["im1"] == 207 and c["nirc2"]["photrad"] == 0.8
    win.n2_im1.setValue(1)
    win.n2_photrad.setValue(1.0)
    win._apply_config(c)
    assert win.n2_im1.value() == 207
    assert abs(win.n2_photrad.value() - 0.8) < 1e-9
    assert c["nirc2"]["path"] == tmp
    win.n2_auto_rad.setChecked(True)
    assert win._collect_config()["nirc2"]["auto_radius"] is True
    win._apply_config(c)                     # c collected with it off
    assert not win.n2_auto_rad.isChecked(), "auto_radius round-trips"
    print("  [ok] nirc2 config keys collect and re-apply")

    # ---- native filenames: KOA-renamed frames displayed as the observer's
    # own numbers, via each frame's own DATAFILE header card (2026-07-25) --
    koa_dir = tempfile.mkdtemp(prefix="gui29_koa_")
    from astropy.io import fits as _fits2
    koa_hdr = _fits2.Header()
    koa_hdr["CURRINST"] = "OSIRIS"
    koa_hdr["IFILTER"] = "BrGamma"
    koa_hdr["OBJECT"] = "koa-synth"
    koa_hdr["DATE-OBS"] = "2026-01-12"
    koa_hdr["UTC"] = "08:59:45.38"
    koa_hdr["LSPROP"] = "yes"
    koa_hdr["RA"] = 81.04414
    koa_hdr["DEC"] = -24.52464
    koa_hdr["DATAFILE"] = "i260112_a000061.fits"
    koa_hdr["FRAMENO"] = 61
    _fits2.writeto(os.path.join(koa_dir, "OI.20260112.32385.38.fits"),
                   np.zeros((64, 64), dtype=np.float32), koa_hdr,
                   overwrite=True)
    plain_hdr = _fits2.Header()
    plain_hdr["CURRINST"] = "OSIRIS"
    plain_hdr["OBJECT"] = "plain-synth"
    _fits2.writeto(os.path.join(koa_dir, "plain_frame.fits"),
                   np.zeros((64, 64), dtype=np.float32), plain_hdr,
                   overwrite=True)

    win.n2_native_names.setChecked(True)
    win.n2_path.setText(koa_dir)
    app.processEvents()
    pump(lambda: win._n2_names_worker is not None
         and not win._n2_names_worker.isRunning())
    for _ in range(10):        # drain the queued cross-thread done signal
        app.processEvents()
        QtCore.QThread.msleep(20)
    assert win.n2_files.count() == 2

    def item_for(disk_name):
        for row in range(win.n2_files.count()):
            it = win.n2_files.item(row)
            if it.data(QtCore.Qt.ItemDataRole.UserRole) == disk_name:
                return it
        raise AssertionError(f"{disk_name} not in file list")

    koa_item = item_for("OI.20260112.32385.38.fits")
    assert koa_item.text() == "i260112_a000061.fits", koa_item.text()
    assert koa_item.toolTip() == "OI.20260112.32385.38.fits"
    plain_item = item_for("plain_frame.fits")
    assert plain_item.text() == "plain_frame.fits", \
        "a file with no DATAFILE card must keep showing its disk name"
    print("  [ok] native-filenames toggle: DATAFILE header shown, disk "
          "name preserved as item data + tooltip; no-DATAFILE file "
          "falls back to its own name")

    win._on_nirc2_file_dclick(koa_item)
    pump(lambda: win._n2_worker is None or not win._n2_worker.isRunning())
    app.processEvents()
    log = win.n2_log.toPlainText()
    assert "i260112_a000061" in log, \
        f"double-click must label the run with the NATIVE name: {log!r}"
    assert "OI.20260112.32385.38" not in log, \
        "the KOA disk name should not leak into the results log"
    print("  [ok] double-click on a native-renamed item measures the "
          "real on-disk file, labelled with the native name")

    win.n2_native_names.setChecked(False)
    win.n2_path.setText("")
    app.processEvents()
    win.n2_path.setText(koa_dir)
    app.processEvents()
    assert item_for("OI.20260112.32385.38.fits").text() \
        == "OI.20260112.32385.38.fits", \
        "toggle off must show disk names again on a fresh listing"
    print("  [ok] toggling off reverts to disk names")

    pump(lambda: win.worker is None)
    win.close()
    app.processEvents()
    print("gui_phase29: all checks passed")


if __name__ == "__main__":
    main()
