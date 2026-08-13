#!/usr/bin/env python3
"""Target RA/Dec: colon-separated and decimal-degree entry (in addition to the
original hms/dms), proper motion (propagates the RA/Dec to the observing date
-- engine.apply_proper_motion -- with per-target storage so switching targets
never leaks one target's PM onto another) and SIMBAD name resolution (the
Resolve button; driven via _on_resolved directly, like _on_catalog_loaded
elsewhere -- the live network query itself is untested offline), the
persistent Target-offset control (shift the base RA/Dec by an arcsec offset
before the engine sees it), and "fix to base position" on the TT/NGS/laser
offsets (anchor their ABSOLUTE sky position so they don't silently follow a
target-offset exploration). Run headless."""
import os, sys, time, json, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
from astropy.coordinates import SkyCoord
import astropy.units as u
np = engine.np
DATA = os.path.join(HERE, "data")


def pump(cond, timeout=90):
    app = QtWidgets.QApplication.instance(); t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents(); QtCore.QThread.msleep(10)


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1500, 950); win.show(); app.processEvents()

    # --- coordinate format flexibility: hms/dms, colon, decimal degrees ------
    ref = SkyCoord("15h49m57.7s", "-03d55m16s")
    for ra, dec in (("15h49m57.7s", "-03d55m16s"),   # original hms/dms
                    ("15:49:57.7", "-03:55:16"),       # colon-separated
                    ("237.490417", "-3.921111"),       # decimal degrees
                    ("15h49m57.7s", "-03:55:16")):     # mixed, independent parsing
        c = engine.parse_radec(ra, dec)
        sep = c.separation(ref).arcsec
        assert sep < 0.05, f"{ra!r}/{dec!r} -> {sep:.4f}\" from reference"
        assert win._radec_ok(ra, dec), f"_radec_ok rejected {ra!r}/{dec!r}"
    assert not win._radec_ok("garbage", "-03d55m16s"), "unparseable RA must fail"
    print("  [ok] RA/Dec formats: hms/dms, colon-separated, decimal degrees "
          "all agree to <0.05\" (mixed formats too)")

    # --- target offset: default (0) leaves the position unchanged ------------
    win.ra_edit.setText("15h49m57.7s"); win.dec_edit.setText("-03d55m16s")
    app.processEvents()
    eff = win._effective_target_coords()
    assert eff is not None
    c0 = engine.parse_radec(*eff)
    assert c0.separation(ref).arcsec < 1e-6, "zero offset must not move the target"
    print(f"  [ok] zero target offset: effective position = {eff[0]} {eff[1]} "
          f"(0.0000\" from typed RA/Dec)")

    # --- control-panel width floors (2026-07-20 horizontal-scrollbar fix):
    #     a QDoubleSpinBox's minimumSizeHint is an unshrinkable layout floor
    #     (its default h-policy is Minimum) unless an explicit minimumWidth
    #     overrides it, and a QFormLayout's minimum width is max(label col) +
    #     max(field col) -- so one over-wide row forces a horizontal scrollbar
    #     across the whole tab. Guard the exact floors that regressed (font-
    #     independent, unlike asserting pixel widths under the offscreen QPA).
    #     <= not <: the invariant is "the explicit floor never exceeds what
    #     the box actually needs" (so it can't widen the row past its natural
    #     size) -- equality is fine, the widget still gets exactly enough
    #     room. Some Qt style/font combinations (Fusion, on this machine)
    #     compute a smaller natural minimumSizeHint than others for the same
    #     range/decimals/suffix -- 2026-07-25: HOMEAS et al ties minimumWidth
    #     exactly (70==70) despite an unrelated Qt build/style difference,
    #     not a code change (verified: ADJUSTABLE_BUDGET_PARAMS' ranges for
    #     every affected row are byte-identical across the budget-
    #     recalibration commit that touched this tab).
    for spin in (win.pmra_spin, win.pmdec_spin):
        assert 0 < spin.minimumWidth() <= spin.minimumSizeHint().width(), \
            "PM spinboxes need an explicit minimumWidth floor AT OR BELOW " \
            "their natural minimumSizeHint, or the Target tab regains its " \
            "scrollbar"
    for name, r in win.wfe_rows.items():
        assert 0 < r["spin"].minimumWidth() <= r["spin"].minimumSizeHint().width(), \
            f"WFE slider spinbox [{name}] lost its explicit width floor"
    probe = win._wrap(QtWidgets.QHBoxLayout())
    m = probe.layout().contentsMargins()
    assert (m.left(), m.top(), m.right(), m.bottom()) == (0, 0, 0, 0), \
        "_wrap must zero the QLayout margins (9px/side widened+indented every " \
        "wrapped form row; part of the horizontal-scrollbar fix)"
    assert win.lgs_offset_enable.text() == "override", \
        "LGS override checkbox text must stay short (long text -> wide field " \
        "column -> horizontal scrollbar); details belong in its tooltip"
    # --- 2026-08-12: the GLOBAL invariant, not just the widgets that bit
    #     before. Guarding only known offenders let SIX new rows regress the
    #     panel (all six tabs were over the floor). Every control tab's
    #     content minimum must fit the panel's guaranteed viewport --
    #     tabs.minimumWidth() minus the frame and a vertical scrollbar,
    #     queried from the live style so the bound tracks fonts/styles.
    #     If this fires: find the row whose minimumSizeHint grew (a new
    #     label/button/checkbox/group-title) and give it a width floor or
    #     move the parenthetical to a tooltip -- see 631045c.
    _sbw = win.style().pixelMetric(
        QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent)
    _guaranteed = win.tabs.minimumWidth() - _sbw - 4
    for _i in range(win.tabs.count()):
        _sa = win.tabs.widget(_i)
        _content = (_sa.widget() if isinstance(_sa, QtWidgets.QScrollArea)
                    else _sa)
        _m = _content.minimumSizeHint().width()
        assert _m <= _guaranteed, \
            (f"control tab '{win.tabs.tabText(_i)}' minimum {_m}px exceeds "
             f"the panel's guaranteed viewport {_guaranteed}px -- the "
             f"horizontal scrollbar is back (NEVER allowed; see 631045c)")
    print(f"  [ok] all {win.tabs.count()} control tabs fit the "
          f"{_guaranteed}px guaranteed viewport (no h-scrollbar possible)")
    # 2026-07-21: this class bit AGAIN -- the Nighttime-mode status label
    # ("last pull HH:MM:SS HST · next pull HH:MM:SS HST") shipped as a plain
    # QLabel, whose full-text minimumSizeHint re-widened the Data tab into a
    # horizontal scrollbar. Guard it BEHAVIOURALLY (and font-independently):
    # growing any dynamic status label's text must not move its tab's width
    # floor. Applies to every _shrinkable_label-managed readout.
    for tab_idx, label in ((0, win.nighttime_status),):
        inner = win.tabs.widget(tab_idx).widget()
        w0 = inner.minimumSizeHint().width()
        old = label.text()
        label.setText("last pull 00:00:00 HST  ·  next pull 00:00:00 HST" * 3)
        app.processEvents()
        assert inner.minimumSizeHint().width() == w0, \
            f"tab {tab_idx}: a status label's text length moved the tab's " \
            "width floor -- it must be _shrinkable_label-managed (clip + " \
            "tooltip), or the tab regains its horizontal scrollbar"
        label.setText(old)
    print("  [ok] control-panel width floors hold (PM + WFE spin minimums, "
          "_wrap margins, LGS checkbox text, dynamic status labels)")

    # --- proper motion: 0/0 (the default) leaves the position unchanged,
    #     exactly like zero target offset above (same byte-identity contract) -
    assert win.pmra_spin.value() == 0.0 and win.pmdec_spin.value() == 0.0
    eff0 = win._effective_target_coords()
    c0 = engine.parse_radec(*eff0)
    assert c0.separation(ref).arcsec < 1e-6, "0/0 PM must not move the target"
    print("  [ok] zero proper motion: effective position unchanged (0.0000\")")

    # --- nonzero PM shifts the effective position, matching a direct
    #     engine.apply_proper_motion call (same J2000 epoch, same "today") ----
    win.pmra_spin.setValue(50.0); win.pmdec_spin.setValue(-20.0)
    app.processEvents()
    eff_pm = win._effective_target_coords()
    got_pm = engine.parse_radec(*eff_pm)
    want_pm = engine.apply_proper_motion(ref, 50.0, -20.0, win._pm_obs_date())
    sep_pm = got_pm.separation(want_pm).arcsec
    assert sep_pm < 0.01, f"PM-shifted effective coords mismatch: {sep_pm:.4f}\""
    assert got_pm.separation(ref).arcsec > 0.05, \
        "this check is vacuous unless the PM actually moved the answer"
    print(f"  [ok] proper motion (50,-20 mas/yr) shifts the effective target "
          f"{got_pm.separation(ref).arcsec:.3f}\" from J2000, matching "
          f"engine.apply_proper_motion directly")
    win.pmra_spin.setValue(0.0); win.pmdec_spin.setValue(0.0)
    app.processEvents()

    # --- Resolve button: drive _on_resolved directly (like _on_catalog_loaded
    #     elsewhere) -- the live SIMBAD query itself is untested offline -------
    win.tname_edit.setText("my nickname for this star")
    win._on_resolved({"name": "HD 141569", "ra_deg": 237.490618,
                      "dec_deg": -3.921206, "pmra": -17.42, "pmdec": -19.113}, "")
    assert win.tname_edit.text() == "my nickname for this star", \
        "Resolve must NOT overwrite a user-typed name with SIMBAD's canonical id"
    resolved_c = engine.parse_radec(win.ra_edit.text(), win.dec_edit.text())
    assert resolved_c.separation(SkyCoord(237.490618 * u.deg, -3.921206 * u.deg)) \
        .arcsec < 0.01, "Resolve must fill RA/Dec from the SIMBAD result"
    assert abs(win.pmra_spin.value() - (-17.4)) < 1e-6, \
        "pmra_spin has 1 decimal, -17.42 rounds to -17.4"
    assert abs(win.pmdec_spin.value() - (-19.1)) < 1e-6, \
        "pmdec_spin has 1 decimal, -19.113 rounds to -19.1"
    assert "HD 141569" in win.status.text() and "mas/yr" in win.status.text()
    print(f"  [ok] Resolve fills RA/Dec + PM from a SIMBAD result, keeps the "
          f"typed name ({win.status.text()})")

    # a target with NO measured SIMBAD proper motion must leave the PM fields
    # AS THEY WERE (nothing to fill in) rather than crash or silently zero them
    win._on_resolved({"name": "NAME Sgr A*", "ra_deg": 266.416817,
                      "dec_deg": -29.007825, "pmra": None, "pmdec": None}, "")
    assert abs(win.pmra_spin.value() - (-17.4)) < 1e-6 and \
        abs(win.pmdec_spin.value() - (-19.1)) < 1e-6, \
        "no measured PM -> PM fields left as they were (nothing to fill in)"
    assert "no proper motion" in win.status.text()
    print("  [ok] Resolve on a target with no SIMBAD proper motion leaves the "
          "PM fields alone and says so")

    # a name that doesn't resolve reports the error without crashing
    win._on_resolved(None, "ValueError: “bogus” did not resolve in SIMBAD")
    assert "Resolve failed" in win.status.text()
    assert win.resolve_btn.isEnabled(), "button must re-enable after failure"
    print("  [ok] a failed resolve reports the error and re-enables the button")
    win.pmra_spin.setValue(0.0); win.pmdec_spin.setValue(0.0)

    # --- per-target proper motion: switching targets must not leak PM --------
    win.ra_edit.setText("15h49m57.7s"); win.dec_edit.setText("-03d55m16s")
    win.tname_edit.setText("Star A")
    win.pmra_spin.setValue(50.0); win.pmdec_spin.setValue(-20.0)
    idx_a = win._save_current_target()
    win.ra_edit.setText("16h00m00.0s"); win.dec_edit.setText("-04d00m00s")
    win.tname_edit.setText("Star B")
    win.pmra_spin.setValue(0.0); win.pmdec_spin.setValue(0.0)
    idx_b = win._save_current_target()
    win._on_target_selected(idx_a)
    assert abs(win.pmra_spin.value() - 50.0) < 1e-6 and \
        abs(win.pmdec_spin.value() - (-20.0)) < 1e-6, \
        "selecting Star A must restore ITS proper motion"
    win._on_target_selected(idx_b)
    assert win.pmra_spin.value() == 0.0 and win.pmdec_spin.value() == 0.0, \
        "selecting Star B must NOT carry over Star A's proper motion"
    print("  [ok] per-target proper motion: switching targets restores each "
          "target's own PM, no leakage")

    # config round-trip preserves proper motion
    win._on_target_selected(idx_a)
    cfg_pm = os.path.join(tempfile.gettempdir(), "p18_pm.json")
    with open(cfg_pm, "w") as fh:
        json.dump(win._collect_config(), fh)
    win.pmra_spin.setValue(0.0); win.pmdec_spin.setValue(0.0)
    with open(cfg_pm) as fh:
        win._apply_config(json.load(fh))
    assert abs(win.pmra_spin.value() - 50.0) < 1e-6, "PM RA not restored"
    assert abs(win.pmdec_spin.value() - (-20.0)) < 1e-6, "PM Dec not restored"
    print("  [ok] config round-trip preserves proper motion")
    win.pmra_spin.setValue(0.0); win.pmdec_spin.setValue(0.0)
    win.ra_edit.setText("15h49m57.7s"); win.dec_edit.setText("-03d55m16s")
    app.processEvents()

    # --- target offset: ΔRA/ΔDec mode, checked against astropy directly ------
    win.target_offset.mode.setCurrentIndex(1)
    win.target_offset.dra.setValue(10.0)    # arcsec East
    win.target_offset.ddec.setValue(-5.0)   # arcsec South
    app.processEvents()
    eff = win._effective_target_coords()
    got = engine.parse_radec(*eff)
    # spherical_offsets_by(d_lon, d_lat): +d_lon = increasing RA = East,
    # matching the ΔRA (E) sign convention directly (no flip needed here).
    want = ref.spherical_offsets_by(10.0 * u.arcsec, -5.0 * u.arcsec)
    assert got.separation(want).arcsec < 0.01, \
        f"ΔRA/ΔDec offset mismatch: got {got.to_string('hmsdms')}, want {want.to_string('hmsdms')}"
    print("  [ok] target offset (10\" E, 5\" S) matches astropy's own "
          "spherical_offsets_by to <0.01\"")
    assert "effective target" in win.target_offset_readout.text()

    # --- a fixed backdrop must NOT move when the target offset changes -------
    # (the field-map centre/backdrop stays on the BASE target; only the
    # target marker -- via the drawing shift -- moves within it)
    fc_before = win._field_center_deg()
    win.target_offset.dra.setValue(20.0); win.target_offset.ddec.setValue(15.0)
    app.processEvents()
    fc_after = win._field_center_deg()
    assert fc_before == fc_after, \
        f"field/backdrop centre must not move with the target offset: " \
        f"{fc_before} -> {fc_after}"
    shift_x, shift_y = win._backdrop_shift_arcsec()
    # a target offset of (dRA east, dDec north) must shift the FIXED backdrop
    # by (+dRA, -dDec) in the plot's (x=West+, y=North+) frame -- i.e. the
    # target visually moves the OPPOSITE way, (east, north), within it.
    assert abs(shift_x - 20.0) < 0.01, shift_x
    assert abs(shift_y - (-15.0)) < 0.01, shift_y
    print(f"  [ok] backdrop/field centre fixed under a target offset "
          f"(shift {shift_x:.2f}\",{shift_y:.2f}\" applied to the drawing only)")

    # --- NGS/TT star-coordinate offset mode measures from the EFFECTIVE
    #     target (the one the target offset produces), not the base one -----
    win.ngs_offset.mode.setCurrentIndex(2)
    win.ngs_offset.sra.setText("15h50m10.0s"); win.ngs_offset.sdec.setText("-03d55m16s")
    app.processEvents()
    eff_now = win._effective_target_coords()
    want_ngs = SkyCoord(*eff_now).separation(
        SkyCoord("15h50m10.0s", "-03d55m16s")).arcsec
    base_only = ref.separation(SkyCoord("15h50m10.0s", "-03d55m16s")).arcsec
    assert abs(win.ngs_offset.value() - want_ngs) < 0.01, \
        f"NGS star-offset must measure from the effective target: " \
        f"got {win.ngs_offset.value():.3f}, want {want_ngs:.3f}"
    assert abs(win.ngs_offset.value() - base_only) > 1.0, \
        "this check is vacuous unless the target offset actually moves the answer"
    print(f"  [ok] NGS star-coordinate offset measures from the effective "
          f"target ({win.ngs_offset.value():.2f}\", not the base {base_only:.2f}\")")
    win.ngs_offset.setValue(0.0)                    # back to a clean total
    win.target_offset.dra.setValue(10.0); win.target_offset.ddec.setValue(-5.0)

    # --- offset actually reaches the engine via collect_args -----------------
    win.target_enable.setChecked(True)
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(os.path.join(DATA, "20260525_dimm.dat"))
    win.mass_edit.setText(os.path.join(DATA, "20260525_mass.dat"))
    win.masspro_edit.setText(os.path.join(DATA, "20260525_masspro.dat"))
    win.tel_k1.setChecked(True)
    win._validate()
    a = win.collect_args("/tmp/x.png")
    a_coord = engine.parse_radec(a.ra, a.dec)
    assert a_coord.separation(want).arcsec < 0.01, \
        "collect_args must use the offset-applied position, not the raw fields"
    print(f"  [ok] collect_args ra/dec reflects the target offset: {a.ra} {a.dec}")

    # --- star RA/Dec mode with bad coordinates blocks Run --------------------
    assert win.run_btn.isEnabled(), "should run with a valid offset"
    win.target_offset.mode.setCurrentIndex(2)
    win.target_offset.sra.setText("not-a-coordinate")
    win.target_offset.sdec.setText("also not one")
    win._validate()
    assert not win.target_offset.ok() and not win.run_btn.isEnabled(), \
        "bad target-offset star coordinates must block Run"
    assert "Target offset" in win.status.text()
    print("  [ok] invalid target-offset star coordinates disable Run")

    # star RA/Dec mode with GOOD coordinates: effective position = typed coords
    win.target_offset.sra.setText("15h50m10.0s")
    win.target_offset.sdec.setText("-04d00m00s")
    app.processEvents()
    eff = win._effective_target_coords()
    got = engine.parse_radec(*eff)
    want2 = SkyCoord("15h50m10.0s", "-04d00m00s")
    assert got.separation(want2).arcsec < 0.01, "star-coord mode -> typed position"
    print("  [ok] target-offset star-coordinate mode uses the typed position directly")

    # --- config round-trip preserves the target offset ------------------------
    win.target_offset.mode.setCurrentIndex(1)
    win.target_offset.dra.setValue(6.0); win.target_offset.ddec.setValue(8.0)
    app.processEvents()
    cfg = os.path.join(tempfile.gettempdir(), "p18.json")
    with open(cfg, "w") as fh:
        json.dump(win._collect_config(), fh)
    win.target_offset.setValue(0.0)                # disturb
    with open(cfg) as fh:
        win._apply_config(json.load(fh))
    assert win.target_offset.mode.currentIndex() == 1, "mode not restored"
    assert abs(win.target_offset.dra.value() - 6.0) < 1e-9, "dRA not restored"
    assert abs(win.target_offset.ddec.value() - 8.0) < 1e-9, "dDec not restored"
    print("  [ok] config round-trip preserves the target offset")

    # --- "fix to base position": TT/NGS/laser stay put in absolute sky terms
    #     as the target offset changes, instead of following it ---------------
    win.target_offset.setValue(0.0)               # clean slate
    win.ra_edit.setText("15h49m57.7s"); win.dec_edit.setText("-03d55m16s")
    app.processEvents()

    win.tt_offset.mode.setCurrentIndex(1)
    win.tt_offset.dra.setValue(19.3); win.tt_offset.ddec.setValue(0.0)
    win.tt_offset.fix_to_base.setChecked(True)
    tt_anchor = win.tt_offset._anchor
    win.target_offset.mode.setCurrentIndex(1)
    win.target_offset.dra.setValue(5.0); win.target_offset.ddec.setValue(0.0)
    app.processEvents()
    assert abs(win.tt_offset.value() - 14.3) < 0.01, \
        f"TT offset should shrink 19.3->14.3 as the target moves 5\" toward " \
        f"it, got {win.tt_offset.value():.2f}"
    eff = win._effective_target_coords()
    tt_drift = win.tt_offset.resolved_skycoord(engine.parse_radec(*eff)) \
                             .separation(tt_anchor).arcsec
    assert tt_drift < 0.01, f"TT star must stay fixed in absolute position, drifted {tt_drift:.4f}\""
    print(f"  [ok] TT-offset fix-to-base: 19.3\" -> {win.tt_offset.value():.1f}\" "
          f"as target moves 5\" E, absolute position drift {tt_drift:.4f}\"")

    win.ngs_offset.mode.setCurrentIndex(1)
    win.ngs_offset.dra.setValue(10.0); win.ngs_offset.ddec.setValue(0.0)
    win.ngs_offset.fix_to_base.setChecked(True)
    ngs_anchor = win.ngs_offset._anchor
    win.target_offset.ddec.setValue(3.0)           # add 3" N to the target too
    app.processEvents()
    eff = win._effective_target_coords()
    ngs_drift = win.ngs_offset.resolved_skycoord(engine.parse_radec(*eff)) \
                              .separation(ngs_anchor).arcsec
    assert ngs_drift < 0.01, f"NGS star must stay fixed, drifted {ngs_drift:.4f}\""
    print(f"  [ok] NGS-offset fix-to-base: absolute position drift "
          f"{ngs_drift:.4f}\" after a further target-offset change")

    win.lgs_offset_enable.setChecked(True); win.lgs_offset.setValue(7.0)
    win.laser_pa.setValue(90.0)
    win.laser_fix_to_base.setChecked(True)
    laser_anchor = win._laser_anchor
    win.target_offset.dra.setValue(8.0)
    app.processEvents()
    eff = win._effective_target_coords()
    laser_drift = win._laser_absolute(engine.parse_radec(*eff)) \
                      .separation(laser_anchor).arcsec
    assert laser_drift < 0.01, f"laser must stay fixed, drifted {laser_drift:.4f}\""
    print(f"  [ok] laser fix-to-base: absolute position drift "
          f"{laser_drift:.4f}\" after a target-offset change "
          f"(lgs_offset now {win.lgs_offset.value():.2f}\")")

    # config round-trip preserves all three fix-to-base checkboxes
    cfg2 = os.path.join(tempfile.gettempdir(), "p18_fixbase.json")
    with open(cfg2, "w") as fh:
        json.dump(win._collect_config(), fh)
    win.tt_offset.fix_to_base.setChecked(False)
    win.ngs_offset.fix_to_base.setChecked(False)
    win.laser_fix_to_base.setChecked(False)
    with open(cfg2) as fh:
        win._apply_config(json.load(fh))
    assert win.tt_offset.fix_to_base.isChecked(), "TT fix-to-base not restored"
    assert win.ngs_offset.fix_to_base.isChecked(), "NGS fix-to-base not restored"
    assert win.laser_fix_to_base.isChecked(), "laser fix-to-base not restored"
    print("  [ok] config round-trip preserves all three fix-to-base states")

    # --- a run works end-to-end with an offset target position ---------------
    win.on_run()
    pump(lambda: win.res is not None, timeout=90)
    assert win.res is not None
    print("  [ok] full run with an offset-shifted target position")


if __name__ == "__main__":
    main()
