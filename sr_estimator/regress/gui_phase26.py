#!/usr/bin/env python3
"""Guide-star-follows-target (2026-07-22): "anytime we define a new target
we need to also define the guide star." NGS/LGS default to on-axis (guide
star = the target) whenever a target is newly defined (SIMBAD Resolve,
Save, a starlist pick with no linked candidate) -- the exception is a
starlist target=<name>-linked TT-star candidate for LGS, picked directly if
there is exactly one, or by the SAME delivered-Strehl ranking the field
map's Rank button uses if there are several. Each SAVED target remembers
its OWN guide star (tt_offset_cfg/tt_mag), like proper motion already does,
so switching between saved targets restores each one's own pick rather than
leaking. Run headless."""
import os, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
DATA = os.path.join(HERE, "data")


def pump(cond, timeout=90):
    app = QtWidgets.QApplication.instance(); t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents(); QtCore.QThread.msleep(10)


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1500, 950); win.show(); app.processEvents()

    # keep the suite offline AND fail-safe: any UNEXPECTED SIMBAD query from
    # a pick that should have been satisfied by the starlist's own magnitudes
    # surfaces as a loud "unexpected SIMBAD query" warning in the status
    # rather than a silent live network call. NOTE the patch target: the
    # CLASS (type(Simbad)) -- astroquery.simbad.Simbad is an INSTANCE, and
    # Simbad() mints a NEW instance via BaseQuery.__call__, so an
    # instance-level patch is silently bypassed (2026-07-22 lesson: the
    # proper_motion_model fakes ran against the real network for days).
    import astroquery.simbad
    from astropy.table import Table
    import numpy.ma as ma
    simbad_cls = type(astroquery.simbad.Simbad)
    orig_query = simbad_cls.query_object
    # query_object is NOT the only network call. resolve_target_name() first
    # does sim.add_votable_fields("pmra","pmdec","allfluxes"), and in current
    # astroquery that validates the field names against SIMBAD's live TAP
    # capabilities endpoint. Stubbing only query_object therefore left this
    # "offline" suite making a real request to simbad.cds.unistra.fr on every
    # ResolveWorker run -- which is what broke CI on 2026-07-28: with CDS
    # unreachable the call spends ~14 s before raising, blowing
    # wait_gs_worker's 15 s budget, and even when it returns in time the
    # worker reports an error so the SIMBAD magnitude is never applied. The
    # fake tables below already carry every column the real call would have
    # requested, so declaring the fields is a no-op here.
    orig_fields = simbad_cls.add_votable_fields
    simbad_cls.add_votable_fields = lambda self, *fields: None

    def stub_simbad(mag_cols):
        """None -> unresolvable (empty table); dict -> one row with fluxes."""
        def _q(self, name):
            if mag_cols is None:
                return Table({"main_id": [], "ra": [], "dec": []})
            cols = {"main_id": ["X"], "ra": [1.0], "dec": [2.0],
                    "pmra": ma.array([0.0], mask=[True]),
                    "pmdec": ma.array([0.0], mask=[True])}
            cols.update({k: [v] for k, v in mag_cols.items()})
            return Table(cols)
        simbad_cls.query_object = _q

    def _unexpected(self, name):
        raise AssertionError(f"unexpected SIMBAD query for {name!r}")
    simbad_cls.query_object = _unexpected

    def wait_gs_worker():
        pump(lambda: win._gs_mag_worker is None, timeout=15)
        app.processEvents()

    # --- Resolve resets the guide star to on-axis AND applies the resolved
    #     star's own magnitudes (it is now its own guide star) ---------------
    win.tt_offset.total.setValue(12.0)          # a leftover custom offset
    win.ngs_offset.total.setValue(3.0)
    win._on_resolved({"name": "HD 141569", "ra_deg": 237.490618,
                      "dec_deg": -3.921206, "pmra": None, "pmdec": None,
                      "mags": {"V": 7.1, "K": 6.8}}, "")
    app.processEvents()
    cfg = win.tt_offset.get_config()
    assert cfg["mode"] == 0 and cfg["total"] == 0.0, cfg
    ngs_cfg = win.ngs_offset.get_config()
    assert ngs_cfg["mode"] == 0 and ngs_cfg["total"] == 0.0, ngs_cfg
    assert abs(win.tt_mag.value() - 7.1) < 0.05, win.tt_mag.value()
    assert abs(win.ngs_bright.value() - 7.1) < 0.05, win.ngs_bright.value()
    print("  [ok] Resolve resets TT/NGS guide star to on-axis AND applies "
          "the star's own SIMBAD magnitudes (R~V=7.1)")

    # ...and with NO magnitudes in the result, it says so instead of leaving
    # a stale value looking deliberate
    win.tt_mag.setValue(11.1)
    win._on_resolved({"name": "NoMagStar", "ra_deg": 50.0, "dec_deg": 5.0,
                      "pmra": None, "pmdec": None}, "")
    app.processEvents()
    assert "NOT updated" in win.status.text() and \
        "NOT adjusted" in win.status.text(), win.status.text()
    assert win.tt_mag.value() == 11.1, "no-mag resolve must not touch tt_mag"
    print("  [ok] Resolve with no usable magnitude WARNS (TT not updated / "
          "NGS not adjusted) and touches nothing")

    # --- Save CAPTURES whatever is currently in the fields (like pm_ra/dec) -
    win.tname_edit.setText("StarA")
    win.ra_edit.setText("15h49m57.7s"); win.dec_edit.setText("-03d55m16s")
    win.tt_offset.total.setValue(7.0); win.tt_mag.setValue(11.5)
    idx_a = win._save_current_target()
    assert win._targets[idx_a]["tt_offset_cfg"]["total"] == 7.0
    assert win._targets[idx_a]["tt_mag"] == 11.5
    print("  [ok] Save captures the LIVE tt_offset/tt_mag (like pm_ra/pm_dec)")

    # --- per-target persistence: switching targets restores EACH one's own
    #     guide star, never leaking -- and NGS stays on-axis regardless -----
    win._on_resolved({"name": "StarB", "ra_deg": 100.0, "dec_deg": 10.0,
                      "pmra": None, "pmdec": None}, "")
    win.tname_edit.setText("StarB")
    win.ngs_offset.total.setValue(9.0)          # must NOT survive -- NGS has
                                                 # no per-target exception
    idx_b = win._save_current_target()
    assert win._targets[idx_b]["tt_offset_cfg"]["total"] == 0.0

    win._on_target_selected(idx_a)
    assert win.tt_offset.get_config()["total"] == 7.0, \
        "switching back to A must restore ITS OWN guide star"
    assert win.tt_mag.value() == 11.5
    assert win.ngs_offset.get_config()["total"] == 0.0, \
        "NGS has no exception -- always on-axis regardless of which target"
    win._on_target_selected(idx_b)
    assert win.tt_offset.get_config()["total"] == 0.0, \
        "selecting B must NOT carry over A's guide star"
    print("  [ok] per-target guide-star restore: switching targets never "
          "leaks one target's TT-star onto another; NGS always on-axis")

    # --- starlist: 0 linked candidates -> on-axis ----------------------------
    text = (
        "MyTarget         15 49 57.7 -03 55 16.0 2000.0 vmag=13.5 lgs=1\n"
        "Far_bright       15 50 10.0 -03 55 16.0 2000.0 rmag=9.0 target=MyTarget\n"
        "Close_faint      15 49 58.5 -03 55 20.0 2000.0 rmag=16.5 target=MyTarget\n"
        "Close_bright     15 49 58.7 -03 55 12.0 2000.0 rmag=10.0 target=MyTarget\n"
        "Solo             16 00 00.0 +10 00 00.0 2000.0 vmag=12.5 lgs=1\n"
        "SoloTT           16 00 01.0 +10 00 05.0 2000.0 rmag=12.0 target=Solo\n"
        # eng361-style: a bare vmag=... lgs=1 row with no linked TT star --
        # picking it must make it its OWN guide star, magnitude included
        "Alone            17 00 00.0 +20 00 00.0 2000.0 vmag=11.0 lgs=1\n"
        # no magnitude at all (HD 214019-style): triggers the SIMBAD lookup
        "NoMag            18 00 00.0 +20 00 00.0 2000.0 lgs=1\n"
        "NoMag2           18 30 00.0 +20 00 00.0 2000.0 lgs=1\n"
        # a SPACED target name linked via the underscore convention (a
        # target= value is a whitespace-delimited token, so it can't hold a
        # literal space) -- mirrors the real hand-edited-list rows that
        # exposed the 2026-07-22 exact-match bug
        "IRAS Demo Star   23 09 43.6 +67 23 38.9 2000.0 vmag=13.0 pa=65.00 lgs=1\n"
        "tt001            23 09 46.2 +67 23 42.5 2000.0 rmag=15.77 "
        "sep=15.44 target=IRAS_Demo_Star\n")
    lst = os.path.join(tempfile.gettempdir(), "p26_multi.lst")
    with open(lst, "w") as fh:
        fh.write(text)

    win._open_starlist(lst); app.processEvents()
    table = win._starlist_table

    def row_of(name):
        return next(r for r in range(table.rowCount())
                    if table.item(r, 1).text() == name)

    win.tt_offset.total.setValue(42.0)          # leftover values to prove
    win.tt_mag.setValue(15.2); win.ngs_bright.setValue(8.0)   # they get replaced
    table.cellDoubleClicked.emit(row_of("Alone"), 0); app.processEvents()
    cfg = win.tt_offset.get_config()
    assert cfg["mode"] == 0 and cfg["total"] == 0.0, \
        "no linked TT candidate -> on-axis, not a leftover value"
    assert win._pending_gs is None
    # ...AND the target's own magnitude, not the previous star's: "a star
    # should be its own TT or NGS star" means position + magnitude both
    assert abs(win.tt_mag.value() - 11.0) < 0.05, win.tt_mag.value()
    assert abs(win.ngs_bright.value() - 11.0) < 0.05, win.ngs_bright.value()
    assert win._gs_mag_worker is None, "list HAD a magnitude -- no SIMBAD call"
    assert win._targets[win.target_select.currentIndex()]["tt_mag"] == \
        win.tt_mag.value(), "the self-star magnitude must persist per-target"
    print("  [ok] starlist target with no linked candidate: guide star = "
          "target itself -- on-axis AND its own magnitude (R~V=11.0)")

    # --- no magnitude in the list at all -> ONE SIMBAD lookup ---------------
    stub_simbad({"V": 6.32, "B": 6.31, "K": 6.321})       # HD 214019's reals
    table.cellDoubleClicked.emit(row_of("NoMag"), 0); app.processEvents()
    assert "asking SIMBAD" in win.status.text(), win.status.text()
    wait_gs_worker()
    assert abs(win.tt_mag.value() - 6.3) < 0.05, win.tt_mag.value()
    assert abs(win.ngs_bright.value() - 6.3) < 0.05, win.ngs_bright.value()
    assert "SIMBAD" in win.status.text()
    assert abs(win._targets[win.target_select.currentIndex()]["tt_mag"]
               - 6.3) < 0.05, "the SIMBAD magnitude must persist per-target"
    print("  [ok] no magnitude in the list: SIMBAD fills TT+NGS (R~V=6.3) "
          "and it persists onto the saved target")

    # ...and a FAILED lookup warns loudly and touches nothing
    stub_simbad(None)
    win.tt_mag.setValue(12.3); win.ngs_bright.setValue(9.9)   # stale sentinels
    table.cellDoubleClicked.emit(row_of("NoMag2"), 0); app.processEvents()
    wait_gs_worker()
    assert "NOT updated" in win.status.text() and \
        "NOT adjusted" in win.status.text(), win.status.text()
    assert win.tt_mag.value() == 12.3 and win.ngs_bright.value() == 9.9, \
        "a failed lookup must leave the magnitudes strictly untouched"
    simbad_cls.query_object = _unexpected
    print("  [ok] failed SIMBAD lookup: explicit 'NOT updated / NOT "
          "adjusted' warning, stale values untouched")

    # --- starlist: exactly 1 linked candidate -> picked directly ------------
    table.cellDoubleClicked.emit(row_of("Solo"), 0); app.processEvents()
    cfg = win.tt_offset.get_config()
    assert cfg["mode"] == 2, "a single candidate must set STAR RA/Dec mode"
    c = engine.parse_radec(cfg["sra"], cfg["sdec"])
    want = engine.parse_radec("16:00:01.0", "+10:00:05.0")
    assert c.separation(want).arcsec < 0.5
    assert abs(win.tt_mag.value() - 12.0) < 0.05, win.tt_mag.value()
    assert "SoloTT" in win.status.text() and "only candidate" in win.status.text()
    print(f"  [ok] a single linked candidate is picked directly "
          f"({win.status.text().split('—')[-1].strip()})")

    # --- starlist: a SPACED name linked with underscores (the target= token
    #     can't contain a literal space) must still match --------------------
    table.cellDoubleClicked.emit(row_of("IRAS Demo Star"), 0); app.processEvents()
    cfg = win.tt_offset.get_config()
    assert cfg["mode"] == 2, \
        "target=IRAS_Demo_Star must link to the name 'IRAS Demo Star'"
    c = engine.parse_radec(cfg["sra"], cfg["sdec"])
    want_tt = engine.parse_radec("23:09:46.2", "+67:23:42.5")
    assert c.separation(want_tt).arcsec < 0.5
    assert abs(win.tt_mag.value() - 15.77) < 0.05, win.tt_mag.value()
    assert "tt001" in win.status.text()
    print("  [ok] underscore-for-space target= link (IRAS_Demo_Star -> "
          "'IRAS Demo Star') matches; tt001 auto-set (R=15.8)")

    # --- starlist: 2+ candidates BEFORE any Run -> deferred -----------------
    prev_cfg_total = win.tt_offset.get_config()["total"]
    table.cellDoubleClicked.emit(row_of("MyTarget"), 0); app.processEvents()
    cfg = win.tt_offset.get_config()
    assert cfg["mode"] == 0 and cfg["total"] == 0.0, \
        "no prepared night yet -> safe on-axis placeholder, not a guess"
    assert win._pending_gs is not None and win._pending_gs["target"] == "MyTarget"
    assert len(win._pending_gs["candidates"]) == 3
    assert "run the estimator" in win.status.text()
    print("  [ok] 2+ candidates with no prepared night yet: deferred (safe "
          "on-axis placeholder + pending-resolution note)")

    # --- ...and it resolves automatically once a Run completes -------------
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(f"{DATA}/20260525_dimm.dat")
    win.mass_edit.setText(f"{DATA}/20260525_mass.dat")
    win.masspro_edit.setText(f"{DATA}/20260525_masspro.dat")
    win.tel_k2.setChecked(True)
    win.on_run()
    pump(lambda: win.res is not None)
    app.processEvents()
    assert win._pending_gs is None, "must resolve once a result exists"
    cfg = win.tt_offset.get_config()
    assert cfg["mode"] == 2, "the deferred pick must have applied by now"
    c = engine.parse_radec(cfg["sra"], cfg["sdec"])
    want_bright = engine.parse_radec("15:49:58.7", "-03:55:12.0")   # Close_bright
    assert c.separation(want_bright).arcsec < 0.5, \
        "close + bright must win over far-bright and close-faint"
    assert abs(win.tt_mag.value() - 10.0) < 0.05
    # persisted onto the (now-current) target's saved entry too
    idx = win.target_select.currentIndex()
    assert win._targets[idx]["tt_offset_cfg"]["mode"] == 2
    assert abs(win._targets[idx]["tt_mag"] - 10.0) < 0.05
    print("  [ok] deferred pick auto-resolves after Run: Close_bright wins "
          "(close AND bright beats far-bright / close-but-faint), and is "
          "persisted onto the target's saved entry")

    # --- the SAME ranking, done SYNCHRONOUSLY (prep/res already exist) -----
    # cross-checked directly against engine.rank_guide_stars (not just "some
    # star got picked") -- re-pick MyTarget now that a result exists
    table.cellDoubleClicked.emit(row_of("MyTarget"), 0); app.processEvents()
    cfg = win.tt_offset.get_config()
    c = engine.parse_radec(cfg["sra"], cfg["sdec"])
    assert c.separation(want_bright).arcsec < 0.5
    assert win._pending_gs is None, "prep/res exist -> no deferral needed"

    entries, _ = engine.parse_starlist(lst)
    candidates = [e for e in entries if e["target"] == "MyTarget"]
    mytarget = next(e for e in entries if e["name"] == "MyTarget")
    stars = [{"id": i, "ra": e["ra_deg"], "dec": e["dec_deg"],
             "mags": {"R": engine.entry_float(e, "rmag")}}
            for i, e in enumerate(candidates)]
    stars = engine.stars_field_xy(stars, mytarget["ra_deg"], mytarget["dec_deg"])
    snap = engine.field_snapshot(win.args_cached, win.prep, win.res,
                                 *win._fm_when_time())
    mode = "ltao" if win.prep.tomography_on else "single"
    with engine.budget_overrides(**(win.last_offsets or {})):
        ranked = engine.rank_guide_stars(win.args_cached, win.prep, snap, mode,
                                         stars, (0.0, 0.0),
                                         win._tt_sensor_band(), metric="strehl")
    usable = [e for e in ranked if e["rank"] is not None]
    winner_name = candidates[usable[0]["id"]]["name"]
    assert winner_name == "Close_bright", winner_name
    print(f"  [ok] synchronous ranking matches a DIRECT engine.rank_guide_"
          f"stars call exactly (winner: {winner_name})")

    # --- config round-trip: per-target guide star survives save/load -------
    cfg_saved = win._collect_config()
    win2 = gui.MainWindow()
    win2._apply_config(cfg_saved)
    idx2 = next(i for i, t in enumerate(win2._targets) if t["name"] == "StarA")
    assert win2._targets[idx2]["tt_offset_cfg"]["total"] == 7.0
    assert win2._targets[idx2]["tt_mag"] == 11.5
    idxm = next(i for i, t in enumerate(win2._targets) if t["name"] == "MyTarget")
    assert win2._targets[idxm]["tt_offset_cfg"]["mode"] == 2
    win2.close()
    print("  [ok] per-target guide star (tt_offset_cfg/tt_mag) round-trips "
          "through a saved config")

    simbad_cls.query_object = orig_query
    simbad_cls.add_votable_fields = orig_fields
    win.grab().save(os.path.join(HERE, "gui_phase26.png"))
    print("  [ok] screenshot saved")


if __name__ == "__main__":
    main()
