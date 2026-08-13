#!/usr/bin/env python3
"""PSF-fit neighbour subtraction on the Measured-SR tab (WP-3/WP-4 of the
psf_fit development plan, 2026-07-30): the "PSF-fit neighbour
subtraction" checkbox, its config round-trip, per-star [psf-clean] log
lines (a real subtraction, and the null/isolated outcome VERBATIM from
the engine's own `psf_clean_note`), the D25 high-Strehl envelope
warning, and the D20 field-map exclude/reinsert round-trip (a star
whose cleaning would remove almost all its own aperture flux is left
off the map, same `_n2_field_dropped` / `_on_nirc2_reject_star`
mechanism as a field-consistency outlier -- no second mechanism).

The test frame is built with `psf_fit_synth.synth_frame` (the same
validated builder the psf_fit engine's own CP2 package was measured
against), not a hand-rolled PSF placement: a bare DL-PSF-only frame at
very low noise (the gui_phase29 pattern) makes `deep_star_catalog`
pick up noise fluctuations as spurious "stars" at the SNR floor this
feature's donor/neighbour cuts actually operate at, which would corrupt
the null-outcome case. Six well-separated donors give a `strict`,
converged, 100%-coverage ePSF; an isolated target, a real blended pair,
and an engineered near-100%-neighbour-light pair exercise the three
outcomes above.

Fully offline; run headless (QT_QPA_PLATFORM=offscreen).
"""
import os
import sys
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

import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
import psf_fit_synth as synth


def pump(cond, timeout=120):
    app = QtWidgets.QApplication.instance()
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        QtCore.QThread.msleep(10)


# detector positions, chosen and verified (see WP-3/WP-4 report in
# STATUS.md) to keep every donor >0.5" from every other catalogued star
# and from the three feature stars below, except the two deliberate
# blends.
DONORS = [(200.0, 200.0), (824.0, 200.0), (200.0, 824.0), (824.0, 824.0),
          (512.0, 200.0), (512.0, 824.0)]
ISO_POS = (300.0, 512.0)                 # null-outcome: no catalogued neighbour
BLEND_TARGET = (512.0, 512.0)            # real subtraction, ~0.3" companion
BLEND_NEIGHBOUR = (542.0, 512.0)
# D20: neighbour dominates the target's aperture (subtracted_frac ~0.98,
# comfortable margin over the 0.95 exclusion floor) while the UNCLEANED
# radial profile still peaks near the target (fwhm0 ~44 mas, comfortable
# margin over radial_profile_fwhm's off-centre failure -- a closer/
# fainter combination made the uncleaned profile fail outright, which
# rejects the star on "invalid FWHM" before psf_clean's own D20 check
# ever runs; found empirically, see the WP-3/WP-4 report in STATUS.md).
EXTREME_TARGET = (750.0, 512.0)
EXTREME_NEIGHBOUR = (810.0, 512.0)


def make_psf_clean_field(dirpath, imno):
    """A crowded NIRC2-narrow frame exercising all three psf_clean
    outcomes at once -- see the module docstring. Returns the params
    used to build it (so the test can cross-check headers), NOT
    written to the FITS (the GUI re-derives params from the header)."""
    params = synth.synth_params()   # narrow, effwave 2.1245, coadds 1, detgain 4.0
    stars = [(x, y, 1.0e5) for x, y in DONORS]
    stars.append((*ISO_POS, 1.0e5, "target"))
    stars.append((*BLEND_TARGET, 1.0e5, "target"))
    stars.append((*BLEND_NEIGHBOUR, 0.6e5, "neighbour"))
    stars.append((*EXTREME_TARGET, 1.0e4, "target"))
    stars.append((*EXTREME_NEIGHBOUR, 2.0e5, "neighbour"))
    raw, truth = synth.synth_frame(stars, params, sr=0.30,
                                   case="gui_phase34", seed=34)

    from astropy.io import fits
    hdr = fits.Header()
    # ROTPPOSN/EL solved so nirc2_frame_params' pmrangl formula
    # (-(ROTPPOSN-EL)+38+90) reproduces synth_params' pmrangl_deg=0.0.
    for k, v in (("CAMNAME", "narrow"), ("PMSNAME", "largehex"),
                 ("EFFWAVE", params.effwave_um), ("ROTPPOSN", 173.0),
                 ("EL", 45.0), ("COADDS", params.coadds),
                 ("DETGAIN", 4.0), ("AOHATCH", "open"),
                 ("PCUNAME", "telescope"), ("OBJECT", "psf-clean-synth")):
        hdr[k] = v
    fits.writeto(os.path.join(dirpath, f"n{imno:04d}.fits"),
                 raw.astype("float32"), hdr, overwrite=True)
    return params


def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow()

    # ---- checkbox present, default off, mentions the envelope -------------
    assert hasattr(win, "n2_psf_clean")
    assert not win.n2_psf_clean.isChecked(), "default off (byte-faithful)"
    assert "0.30" in win.n2_psf_clean.toolTip()
    print("  [ok] checkbox present, default off")

    # ---- config round-trip --------------------------------------------------
    c = win._collect_config()
    assert c["nirc2"]["psf_clean"] is False
    win.n2_psf_clean.setChecked(True)
    assert win._collect_config()["nirc2"]["psf_clean"] is True
    win._apply_config(c)               # c was collected with it off
    assert not win.n2_psf_clean.isChecked(), "psf_clean round-trips"
    print("  [ok] psf_clean config round-trip")

    # ---- build and load the crowded field ------------------------------------
    import tempfile
    tmp = tempfile.mkdtemp(prefix="gui34_psfclean_")
    make_psf_clean_field(tmp, 1)
    win.n2_path.setText(tmp)
    win.n2_im1.setValue(1)
    win.n2_nim.setValue(1)
    win.n2_nbg.setValue(0)
    win.n2_autofind.setChecked(False)   # deterministic: click known positions
    win.n2_add_star.setChecked(True)
    win._on_nirc2_go()
    pump(lambda: win.n2_go.isEnabled())
    assert win._n2_image is not None, win.n2_log.toPlainText()

    def click_at(x, y):
        # re-measuring an already-logged position raises the duplicate-
        # frame prompt (_nirc2_ask_duplicate_paused), a modal
        # QMessageBox.exec() that blocks forever headless -- this test
        # deliberately re-measures the SAME positions (OFF baseline, then
        # ON), so stub it exactly as gui_phase29.py does. clickedButton()
        # is None under the stub, so the handler takes its declared
        # default ("skip"), same as an unattended batch run would.
        ev = SimpleNamespace(xdata=float(x), ydata=float(y),
                             inaxes=win.n2_fig.axes[0])
        orig_exec = QtWidgets.QMessageBox.exec
        QtWidgets.QMessageBox.exec = lambda self: 0
        try:
            win._on_nirc2_click(ev)
        finally:
            QtWidgets.QMessageBox.exec = orig_exec

    # ---- OFF: byte-faithful baseline, no [psf-clean] lines -------------------
    win._on_nirc2_field_clear()
    click_at(*BLEND_TARGET)
    s_default = float(win.n2_strehl_out.text())
    assert "[psf-clean]" not in win.n2_log.toPlainText(), \
        "psf_clean OFF must never touch the log (RULES section 1)"
    print(f"  [ok] psf_clean OFF: byte-faithful, S={s_default:.4f}, "
          "no [psf-clean] lines")

    # ---- ON, auto-find FIELD measure: the ePSF is built ONCE for the
    # field (not per star -- WP-3: "the engine builds the ePSF once; your
    # job is progress messaging"), logged before any star is measured -----
    win.n2_psf_clean.setChecked(True)
    win._on_nirc2_field_clear()
    win.n2_autofind.setChecked(True)
    win.n2_add_star.setChecked(False)
    win.n2_nstars.setValue(6)
    win._on_nirc2_measure_field()
    pump(lambda: win.n2_field_btn.isEnabled())
    log = win.n2_log.toPlainText()
    assert "[psf-clean] field ePSF:" in log, log
    assert "tag='strict'" in log and "converged=True" in log, log
    print("  [ok] field measure: ePSF built ONCE, tag/delta/converged/"
          "coverage logged before per-star results")

    # Each blocking prologue stage must announce itself BEFORE it runs.
    # On a crowded field the catalogue build alone takes ~14 s on the GUI
    # thread; without this the button sits dead and the user cannot tell
    # the tool from a hang. Order matters: the announcement is worthless
    # if it only lands after the work it describes.
    i_find = log.find("field: finding stars")
    i_cat = log.find("field: building the deep neighbour catalogue")
    i_epsf = log.find("field: building the field ePSF")
    i_done = log.find("[psf-clean] field ePSF:")
    assert i_find >= 0, f"no find-stars stage line:\n{log}"
    assert i_cat > i_find, f"catalogue stage missing/out of order:\n{log}"
    assert i_epsf > i_cat, f"ePSF stage missing/out of order:\n{log}"
    assert i_done > i_epsf, \
        f"ePSF result must follow its own progress line:\n{log}"
    print("  [ok] measure-field prologue announces each blocking stage "
          "before running it (find -> catalogue -> ePSF)")
    win._on_nirc2_field_clear()
    win.n2_autofind.setChecked(False)
    win.n2_add_star.setChecked(True)

    # ---- ON, single-star click: null outcome, VERBATIM engine note -------
    click_at(*ISO_POS)
    log = win.n2_log.toPlainText()
    assert ("0 neighbours above the 0.1% contamination floor" in log
            and "effectively isolated" in log), log
    print("  [ok] null outcome printed verbatim from psf_clean_note")

    # ---- real subtraction + D25 envelope warning (this pair's cleaned SR
    #      lands just above 0.30, the validated ceiling) -------------------
    n_before = len(win.n2_log.toPlainText())
    click_at(*BLEND_TARGET)
    added = win.n2_log.toPlainText()[n_before:]
    assert "neighbour(s) subtracted" in added, added
    s_clean = float(win.n2_strehl_out.text())
    assert s_clean > s_default, \
        f"cleaning must recover flux the neighbour was stealing: " \
        f"default={s_default:.4f} clean={s_clean:.4f}"
    assert s_clean > engine.PSF_FIT_SR_VALIDATED_MAX
    assert engine.PSF_FIT_SR_ENVELOPE_NOTE in added, added
    assert "PSF-CLEAN ABOVE VALIDATED SR" in win.n2_warn.text(), \
        win.n2_warn.text()
    # D27: the DIRECTION of the likely residual error must reach the user,
    # not just its magnitude. This pair cleans to above the validated
    # ceiling, so the engine's explicit OVERESTIMATE warning is the one
    # that has to appear -- erring low is only the "safe" direction if the
    # observer is told when it is NOT what happened.
    assert engine.PSF_FIT_BIAS_UNSAFE_NOTE in added, added
    assert "OVERESTIMATE" in added and "upper bound" in added, added
    print("  [ok] D27: expected-bias direction logged (OVERESTIMATE, "
          "above the validated ceiling)")
    print(f"  [ok] real subtraction (default S={s_default:.4f} -> "
          f"cleaned S={s_clean:.4f}); D25 envelope warning fired "
          "(log + n2_warn tag)")

    # ---- D20: engineered near-total-neighbour-light star is left OFF the
    #      map, not appended, with the exact log phrasing ----------------
    win._on_nirc2_field_clear()
    click_at(*EXTREME_TARGET)
    assert len(win._n2_field) == 0, \
        "an over-contaminated star must not join the kept field"
    dropped = getattr(win, "_n2_field_dropped", None) or []
    assert len(dropped) == 1, dropped
    assert dropped[0].psf_clean_excluded
    log = win.n2_log.toPlainText()
    assert "left off the map" in log and "neighbour light" in log \
        and "reinsertable" in log, log
    import re
    m = re.search(r"left off the map — ([\d.]+)% of its aperture flux",
                  log)
    assert m and float(m.group(1)) > 95.0, log
    print(f"  [ok] D20: {m.group(1)}% neighbour-light star left off the "
          "map with the specced log line, not appended to the field")

    # ---- D20 reinsertion: the SAME _on_nirc2_reject_star mechanism,
    #      no second one ---------------------------------------------------
    win._nirc2_draw_map()
    win._on_nirc2_map_pick(SimpleNamespace(
        ind=[0],
        artist=SimpleNamespace(_n2_pool="dropped")))
    assert win._n2_sel_dropped == 0
    assert win.n2_reject_star.text() == "Reinsert star"
    win._on_nirc2_reject_star()
    assert len(win._n2_field) == 1, "reinsertion must use the field-map slot"
    assert (getattr(win, "_n2_field_dropped", None) or []) == []
    assert "reinserted into the fit by user" in win.n2_log.toPlainText()
    print("  [ok] D20 reinsertion: same × / Reinsert mechanism as a "
          "field-consistency outlier, no second mechanism")

    win.n2_psf_clean.setChecked(False)
    win._on_nirc2_field_clear()
    win.close()
    app.processEvents()
    print("gui_phase34: all checks passed")


if __name__ == "__main__":
    main()
