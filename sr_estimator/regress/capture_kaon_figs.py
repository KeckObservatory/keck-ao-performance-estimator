#!/usr/bin/env python3
"""One-off capture script for the KAON documentation figures: drives
the real GUI (on-screen, not headless) through each control state
and grabs a QPixmap of the window, saved as PNG straight into
regress/kaon_update_figs/ (untracked output; regenerate at will).
Not a regression test -- run manually:

    python sr_estimator/regress/capture_kaon_figs.py [name ...]

With no arguments it captures everything; pass one or more of the keys in
SHOTS (below) to redo just those.
"""
import os, sys, time, types
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore, QtGui
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "kaon_update_figs")
# The tool's own built-in default target (Sgr A* / Galactic Center) -- reuse
# it rather than inventing coordinates, so the name/RA/Dec/observing-window
# annotation are all self-consistent (see engine.constants.DEF_TARGET_*).
TARGET_RA, TARGET_DEC = engine.DEF_TARGET_RA, engine.DEF_TARGET_DEC


def pump(cond, timeout=90):
    app = QtWidgets.QApplication.instance(); t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents(); QtCore.QThread.msleep(10)


def settle(app, n=6):
    for _ in range(n):
        app.processEvents()
        QtCore.QThread.msleep(15)


def run_sync(win, app):
    win.on_run()
    pump(lambda: win.worker is None)
    pump(lambda: win.res is not None)
    settle(app)


def grab(win, app, name, w, h):
    win.resize(w, h)
    settle(app)
    path = os.path.join(OUT, name)
    ok = win.grab().save(path, "PNG")
    print(f"  [{'ok' if ok else 'FAIL'}] {name}  ({w}x{h})")
    return path


def scroll_tab(win, idx, frac=None, px=None):
    """The Data/Target/NGS/LGS/WFE-sliders tabs are each wrapped in a
    QScrollArea (MainWindow._scroll) -- win.tabs.widget(idx) IS that scroll
    area. frac in [0,1] scrolls to that fraction of the range; px an
    absolute value."""
    win.tabs.setCurrentIndex(idx)
    sa = win.tabs.widget(idx)
    bar = sa.verticalScrollBar()
    if px is not None:
        bar.setValue(px)
    elif frac is not None:
        bar.setValue(int(bar.maximum() * frac))


def setup_common(win):
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(f"{DATA}/20260525_dimm.dat")
    win.mass_edit.setText(f"{DATA}/20260525_mass.dat")
    win.masspro_edit.setText(f"{DATA}/20260525_masspro.dat")
    win.tel_k1.setChecked(True)
    win.band_combo.setCurrentText("K")
    win.tomo_combo.setCurrentText("auto (per telescope)")
    win.target_enable.setChecked(True)     # RA/Dec/name stay the tool defaults
    win._validate()


def _offset_star_radec(win, e_arcsec, n_arcsec):
    """A guide-star position e_arcsec East / n_arcsec North of the current
    base target, as (hms string, dms string)."""
    import astropy.units as u
    base = engine.parse_radec(win.ra_edit.text(), win.dec_edit.text())
    p = base.spherical_offsets_by(e_arcsec * u.arcsec, n_arcsec * u.arcsec)
    return (p.ra.to_string(unit="hourangle", sep="hms", precision=1),
            p.dec.to_string(unit="deg", sep="dms", precision=0, alwayssign=True))


def _offset_star_radec_colon(win, e_arcsec, n_arcsec):
    import astropy.units as u
    base = engine.parse_radec(win.ra_edit.text(), win.dec_edit.text())
    p = base.spherical_offsets_by(e_arcsec * u.arcsec, n_arcsec * u.arcsec)
    return (p.ra.to_string(unit="hourangle", sep=":", precision=1),
            p.dec.to_string(unit="deg", sep=":", precision=0, alwayssign=True))


def reset_wfe(win):
    win._reset_all_wfe()


def reset_offset(entry, default=0.0):
    """Restore an OffsetEntry to total mode at `default` (the tool's own
    widget default, not a bare 0" -- a leftover 0" leaks a degenerate,
    unillustrative TT-at-field-centre marker into every later field-map
    shot that doesn't explicitly override it)."""
    entry.mode.setCurrentIndex(0)
    entry.total.setValue(default)
    entry.pa.setValue(0.0)
    if getattr(entry, "_fixable", False):
        entry.fix_to_base.setChecked(False)


def fm_render(win, app):
    """Force a field-map redraw and wait for it (both the debounce and the
    trailing full-resolution settle) to finish."""
    win._fieldmap_dirty = True
    win._render_field_map_if_visible()
    pump(lambda: not win._fm_debounce.isActive() and not win._fm_settle.isActive())
    settle(app, 10)


# ---------------------------------------------------------------------------
def shot_a2_main(win, app):
    win.tabs.setCurrentIndex(0)               # Data
    win.plot_tabs.setCurrentIndex(0)           # Timeline
    win.report_combo.setCurrentText("Strehl")
    run_sync(win, app)
    grab(win, app, "fig_gui_main.png", 1944, 950)


def shot_a3_fwhm(win, app):
    win.tabs.setCurrentIndex(0)
    win.plot_tabs.setCurrentIndex(0)
    win.report_combo.setCurrentText("FWHM")
    win.fwhm_curves_combo.setCurrentText("half-max")
    run_sync(win, app)
    grab(win, app, "fig_gui_fwhm_mode.png", 1944, 950)
    win.report_combo.setCurrentText("Strehl")
    run_sync(win, app)


def shot_a1_wfe(win, app):
    reset_wfe(win)
    win.plot_tabs.setCurrentIndex(0)
    run_sync(win, app)
    scroll_tab(win, 4, px=120)                 # WFE sliders tab
    settle(app)
    grab(win, app, "fig_gui_wfe_sliders.png", 1944, 950)


def shot_a4_ngs(win, app):
    sra, sdec = _offset_star_radec(win, 10.0, 5.0)
    win.ngs_offset.mode.setCurrentIndex(2)     # star RA/Dec
    win.ngs_offset.sra.setText(sra)
    win.ngs_offset.sdec.setText(sdec)
    win.ngs_offset.fix_to_base.setChecked(True)
    scroll_tab(win, 2, px=0)                   # NGS tab, top
    settle(app)
    grab(win, app, "fig_ngs_tab_fixed.png", 1500, 950)
    reset_offset(win.ngs_offset, win.defaults.ngs_offset)


def shot_a5_offset_modes(win, app):
    win.tabs.setCurrentIndex(3)                # LGS
    tt = win.tt_offset
    tt.mode.setCurrentIndex(0)                 # total + PA
    tt.total.setValue(20.0)
    tt.pa.setValue(35.0)
    tt.fix_to_base.setChecked(True)
    scroll_tab(win, 3, px=0)
    settle(app)
    grab(win, app, "fig_gui_offset_modes_a.png", 1500, 950)

    tt.fix_to_base.setChecked(False)
    tt.mode.setCurrentIndex(2)                 # star RA/Dec, colon-sexagesimal
    sra, sdec = _offset_star_radec_colon(win, -18.0, 4.0)
    tt.sra.setText(sra)
    tt.sdec.setText(sdec)
    tt.fix_to_base.setChecked(True)
    settle(app)
    grab(win, app, "fig_gui_offset_modes_b.png", 1500, 950)
    reset_offset(tt, win.defaults.tt_offset)


def shot_b4_lgs_tab(win, app):
    win.tt_sensor.setCurrentText("TRICK (K)")
    settle(app)
    run_sync(win, app)
    scroll_tab(win, 3, frac=1.0)               # bottom: Cn2 profile in frame
    settle(app)
    grab(win, app, "fig_gui_lgs_tab.png", 1500, 950)
    win.tt_sensor.setCurrentText("STRAP (R)")
    # STRAP doesn't dictate a science band (by design -- see
    # _TT_SENSOR_MAP), so switching back from TRICK leaves band_combo at
    # TRICK's forced complement (H) instead of reverting it: restore the
    # baseline K explicitly, and clear the sensor-switch status message
    # _sync_tt_mag_for_band leaves behind, so later shots don't inherit
    # either.
    win.band_combo.setCurrentText("K")
    win.fm_catalog_status.setText("")
    settle(app)
    run_sync(win, app)


def shot_b3_target_tab(win, app):
    win._targets = []
    win._add_target("Galactic Center", TARGET_RA, TARGET_DEC, select=False)
    win._add_target("HD 141569", "15h49m57.7s", "-03d55m16s", select=False)
    win._add_target("HD 141943", "15h49m09.9s", "-03d51m30s", select=True)
    win.target_offset.mode.setCurrentIndex(0)
    win.target_offset.total.setValue(8.0)
    win.target_offset.pa.setValue(90.0)
    scroll_tab(win, 1, px=0)                   # Target tab
    settle(app)
    grab(win, app, "fig_gui_target_tab.png", 1500, 950)
    win.target_offset.total.setValue(0.0)
    # restore the base target so later shots (field map etc.) see the
    # tool's usual default again, not whichever target was selected here
    win._targets = []
    win._add_target(engine.DEF_TARGET_NAME, TARGET_RA, TARGET_DEC, select=True)


def shot_a6_field_map(win, app):
    win.plot_tabs.setCurrentIndex(1)
    win.fm_mode.setCurrentText("LTAO")
    win.fm_metric.setCurrentText("Strehl")
    win.fm_for.setChecked(False)
    win.fm_sky.setCurrentText("off")
    win._clear_catalog()
    win._fm_clear_targets()
    settle(app)
    fm_render(win, app)
    grab(win, app, "fig_field_map.png", 2035, 950)


def shot_a7_field_map_sky(win, app):
    win.plot_tabs.setCurrentIndex(1)
    win.fm_mode.setCurrentText("LTAO")
    win.fm_metric.setCurrentText("Strehl")
    win.fm_for.setChecked(True)
    win.fm_sky.setCurrentText("DSS2 red")
    pump(lambda: win._sky_worker is None, timeout=30)
    settle(app, 6)
    fm_render(win, app)
    grab(win, app, "fig_field_map_sky.png", 2035, 950)


def shot_b1_field_map_catalog(win, app):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    win.plot_tabs.setCurrentIndex(1)
    win.fm_mode.setCurrentText("LTAO")
    win.fm_metric.setCurrentText("Strehl")
    win.fm_for.setChecked(True)
    win.fm_sky.setCurrentText("off")
    win.tt_sensor.setCurrentText("STRAP (R)")
    fc = win._field_center_deg()
    cen = SkyCoord(fc[0] * u.deg, fc[1] * u.deg)

    def star(id_, e_arcsec, n_arcsec, mags):
        p = cen.spherical_offsets_by(e_arcsec * u.arcsec, n_arcsec * u.arcsec)
        return {"id": id_, "ra": float(p.ra.deg), "dec": float(p.dec.deg),
                "mags": mags}

    stars = [
        star("A", 5, 20, {"G": 10.8, "BP": 11.0, "RP": 10.4}),
        star("B", -15, 10, {"G": 12.5, "BP": 12.9, "RP": 12.0}),
        star("C", 25, -5, {"G": 14.2, "BP": 14.7, "RP": 13.6}),
        star("D", -30, -25, {"G": 18.8, "BP": 19.3, "RP": 18.1}),
        star("E", 40, 30, {"G": 19.6, "BP": 20.2, "RP": 18.9}),
        star("F", -10, -40, {"G": 16.0, "BP": 16.4, "RP": 15.5}),
        star("G", 12, -18, {"G": 20.5, "BP": 21.0, "RP": 19.7}),
    ]
    win._on_catalog_loaded("Gaia DR2", stars, "")
    settle(app)
    sx = win._catalog_stars_xy()
    a = next(s for s in sx if s["id"] == "A")
    win._inspect_catalog_star(a["x"], a["y"])
    fm_render(win, app)
    grab(win, app, "fig_field_map_catalog.png", 2035, 950)
    win._clear_catalog()


def shot_b2_field_map_targets(win, app):
    win.plot_tabs.setCurrentIndex(1)
    win._clear_catalog()
    win.fm_mode.setCurrentText("single-LGS")
    win.fm_metric.setCurrentText("Strehl")
    win.fm_for.setChecked(False)
    win.fm_sky.setCurrentText("off")
    win.lgs_offset_enable.setChecked(True)
    win.lgs_offset.setValue(0.0)               # laser on the field centre
    win._fm_clear_targets()
    win._fm_add_target(4.0, 0.0)
    win._fm_add_target(-4.0, 0.0)
    win._fm_add_target(6.0, -6.0)
    settle(app)
    fm_render(win, app)
    grab(win, app, "fig_field_map_targets.png", 2035, 950)
    win._fm_clear_targets()
    win.lgs_offset_enable.setChecked(False)


def shot_b5_field_map_pa(win, app):
    """Field PA rotation (2026-07-18 addendum (g)): unrotated vs rotated,
    same field-of-regard + DSS2 backdrop scene so the rotation is obvious."""
    win.plot_tabs.setCurrentIndex(1)
    win._clear_catalog()
    win._fm_clear_targets()
    win.fm_mode.setCurrentText("LTAO")
    win.fm_metric.setCurrentText("Strehl")
    win.fm_for.setChecked(True)
    win.fm_sky.setCurrentText("DSS2 red")
    pump(lambda: win._sky_worker is None, timeout=30)
    settle(app, 6)

    win.fm_pa.setValue(0.0)
    fm_render(win, app)
    grab(win, app, "fig_field_map_pa_a.png", 2035, 950)

    win.fm_pa.setValue(60.0)
    fm_render(win, app)
    grab(win, app, "fig_field_map_pa_b.png", 2035, 950)

    win.fm_pa.setValue(0.0)
    fm_render(win, app)


def shot_e1_nighttime_mode(win, app):
    """Nighttime mode (2026-07-21): forced fetch-mode, disabled source
    controls, live pull-timestamp status. The day/night gate and on_run are
    stubbed (same technique as regress/gui_phase24.py) so the shot doesn't
    depend on real Keck nighttime hours or an actual network fetch -- the
    ALREADY-LOADED local-file run's Timeline stays on screen; only the
    controls/status change."""
    win.tabs.setCurrentIndex(0)                # Data tab
    win.plot_tabs.setCurrentIndex(0)
    orig_is_night, orig_on_run = win._nighttime_is_night, win.on_run
    win._nighttime_is_night = lambda: True
    win.on_run = lambda: None                  # no real fetch during the toggle
    win.nighttime_enable.setChecked(True)
    win._on_nighttime_pull_done()              # records "last pull" for the status
    scroll_tab(win, 0, px=0)
    settle(app)
    grab(win, app, "fig_gui_nighttime_mode.png", 1944, 950)
    win.nighttime_enable.setChecked(False)
    win._nighttime_is_night, win.on_run = orig_is_night, orig_on_run
    win.mode_local.setChecked(True)
    run_sync(win, app)                          # restore normal state


def shot_e2_dark_theme(win, app):
    """Dark theme (2026-07-21): widgets restyle dark; the on-screen (and
    exported) matplotlib figure stays light/print-style -- both in one
    frame is the whole point of the feature."""
    win.tabs.setCurrentIndex(0)
    win.plot_tabs.setCurrentIndex(0)
    win.dark_action.setChecked(True)
    settle(app)
    grab(win, app, "fig_gui_dark_theme.png", 1944, 950)
    win.dark_action.setChecked(False)
    settle(app)


def shot_e3_summary_stats(win, app):
    """Data-tab summary-stats panel (2026-07-21): mean NGS(bright)/single-
    LGS Strehl for BOTH telescopes, LTAO (K1), DIMM/MASS/r0/theta0. STRAP so
    every column is populated (not n/a)."""
    win.tt_sensor.setCurrentText("STRAP (R)")
    win.band_combo.setCurrentText("K")
    run_sync(win, app)
    win.tabs.setCurrentIndex(0)
    scroll_tab(win, 0, frac=1.0)                # bottom: summary-stats panel
    settle(app)
    grab(win, app, "fig_gui_summary_stats.png", 1944, 950)


def shot_e3b_summary_stats_trick_na(win, app):
    """Optional companion: TRICK selected -> the OTHER telescope's Strehl
    cells read n/a (TRICK is K1-only; a synthetic K2 number at K1's forced
    dichroic band would be physically meaningless)."""
    win.tt_sensor.setCurrentText("TRICK (K)")
    run_sync(win, app)
    win.tabs.setCurrentIndex(0)
    scroll_tab(win, 0, frac=1.0)
    settle(app)
    grab(win, app, "fig_gui_summary_stats_trick_na.png", 1944, 950)
    win.tt_sensor.setCurrentText("STRAP (R)")
    win.band_combo.setCurrentText("K")
    run_sync(win, app)


def shot_e4_target_guide_star(win, app):
    """Guide star follows the target (2026-07-22): (a) the starlist picker
    with a target row about to be picked; (b) the LGS tab right after,
    TT-star offset already filled in. A synthetic 3-candidate starlist (the
    bundled real one only has 1:1 target= links) so the "ranked, top
    candidate wins" behavior -- not just a single direct pick -- is the one
    shown, per the request."""
    import tempfile
    text = ("GuideDemo        15 49 57.7 -03 55 16.0 2000.0 lgs=1\n"
            "Far_bright       15 50 10.0 -03 55 16.0 2000.0 rmag=9.0 "
            "target=GuideDemo\n"
            "Close_faint      15 49 58.5 -03 55 20.0 2000.0 rmag=16.5 "
            "target=GuideDemo\n"
            "Close_bright     15 49 58.7 -03 55 12.0 2000.0 rmag=10.0 "
            "target=GuideDemo\n")
    lst = os.path.join(tempfile.gettempdir(), "kaon_e4_guide_demo.lst")
    with open(lst, "w") as fh:
        fh.write(text)
    win.tabs.setCurrentIndex(1)                 # Target tab
    scroll_tab(win, 1, px=0)
    win._open_starlist(lst)
    settle(app)
    table = win._starlist_table
    row = next(r for r in range(table.rowCount())
              if table.item(r, 1).text() == "GuideDemo")
    table.selectRow(row)
    settle(app)
    # the picker is its OWN top-level QDialog, not embedded in win -- grab
    # IT, not the main window (which wouldn't show it at all)
    dlg = win._starlist_dialog
    dlg.resize(900, 420)
    settle(app)
    ok = dlg.grab().save(os.path.join(OUT, "fig_gui_target_guide_star_a.png"), "PNG")
    print(f"  [{'ok' if ok else 'FAIL'}] fig_gui_target_guide_star_a.png  (900x420)")

    table.cellDoubleClicked.emit(row, 0)
    settle(app)
    win._starlist_dialog.close()
    win.tabs.setCurrentIndex(3)                 # LGS tab
    scroll_tab(win, 3, px=0)
    settle(app)
    grab(win, app, "fig_gui_target_guide_star_b.png", 1500, 950)

    # restore the base target for later shots
    win._targets = []
    win._add_target(engine.DEF_TARGET_NAME, TARGET_RA, TARGET_DEC, select=True)
    run_sync(win, app)


def shot_e5_fa_advisory(win, app):
    """FA timing advisory (2026-07-22) with the event warning TRIGGERED:
    the bundled 20260525 night's final 40 minutes genuinely trip the
    volatile cue (FA climbing 0.23-0.58"), and the GFS winds come from the
    per-UT-date cache when present (mkwc_cache/gfs_winds_20260525.json;
    a fresh checkout fetches once, live, via the historical host), so the
    per-layer lead/lag line renders too. Shows the winds row + advisory +
    summary panel in one frame."""
    win.tabs.setCurrentIndex(0)
    win.plot_tabs.setCurrentIndex(0)
    run_sync(win, app)
    prev = win.res
    win._fetch_gfs_winds()
    pump(lambda: win._winds_worker is None, timeout=30)
    pump(lambda: win.res is not prev, timeout=30)
    settle(app)
    assert "⚠" in win.fa_advisory.text(), \
        "the event cue must be TRIGGERED in this figure"
    assert "Keck first" in win.fa_advisory.text(), \
        "the lead/lag line must be present (winds + target)"
    scroll_tab(win, 0, frac=0.5)
    settle(app)
    grab(win, app, "fig_gui_fa_advisory.png", 1944, 950)
    # restore the stock winds so no later shot inherits this night's values
    win.wind_ground.setValue(8.0)
    win.wind_free.setValue(25.0)
    run_sync(win, app)


def shot_e6_fa_geometry(win, app):
    """FA pierce-point geometry dialog (the live Figure-3 remake): plan +
    side view for the current target, with per-layer GFS wind vectors and
    lead/lag annotations. The DIALOG is its own top-level window -- grab
    it directly (the e4 lesson).

    Set on the 2026-01-31 UT M79 night at the frames' own time rather
    than the generic default target (Eduardo's Rev D-F figure request):
    that is the case the KAON text describes, where the MKAM monitor
    model picks Aldebaran (alpha Tau) and collapses the 16 km lead's
    +/-43 min pointing-ignorance band to -1...+6 min. The reference
    time drives which monitor stars are up, so it has to be pinned to
    the frame time -- at the default end-of-night reference M79 is
    already below the horizon and a different asterism is returned.

    NOTE (2026-07-28): the star identity and the -1...+6 min lead range
    reproduce exactly, but this model now gives Aldebaran P = 70 %,
    where the Rev D text quotes P ~ 41 %. See the figure-request reply
    -- the caption number needs reconciling, the figure is correct."""
    win.tabs.setCurrentIndex(0)
    win.mode_fetch.setChecked(True)
    win.fetch_date.setDate(QtCore.QDate(2026, 1, 31))
    win._targets = []
    win._add_target("M79", "05h24m10.6s", "-24d31m27s", select=True)
    run_sync(win, app)
    if win._gfs_winds_result is None:
        prev = win.res
        win._fetch_gfs_winds()
        pump(lambda: win._winds_worker is None, timeout=60)
        pump(lambda: win.res is not prev, timeout=60)
    # the M79 frames' own time (06:50 UT = 20:50 HST); the FA dialog
    # follows the Data-tab Period selector when it names one instant
    win.stats_cond.setCurrentText("specific time")
    win.stats_time.setTime(QtCore.QTime(20, 50))
    settle(app, 8)
    win._show_fa_geometry()
    settle(app, 10)
    dlg = win._fa_geo_dialog
    assert dlg is not None
    dlg.resize(980, 540)
    settle(app, 6)
    ok = dlg.grab().save(os.path.join(OUT, "fig_gui_fa_geometry.png"), "PNG")
    print(f"  [{'ok' if ok else 'FAIL'}] fig_gui_fa_geometry.png  (980x540)")
    dlg.close()
    settle(app)
    win.wind_ground.setValue(8.0)
    win.wind_free.setValue(25.0)
    run_sync(win, app)


def shot_b6_field_map_flip(win, app):
    """Backdrop/frame flip (2026-07-18 addendum (h)): the NIRC2 example frame
    unflipped vs with Frame-flip X checked -- also exercises the addendum
    (a) WCS-string-card load fix."""
    win.plot_tabs.setCurrentIndex(1)
    win._clear_catalog()
    win._fm_clear_targets()
    win.fm_for.setChecked(False)               # zoom to the frame, not the 60" FoR
    win.fm_sky.setCurrentText("off")
    frame_path = os.path.join(os.path.dirname(HERE), "examples",
                              "N2.20210821_42637.fits")
    win._load_local_sky(frame_path)
    settle(app)
    assert win._sky_fg_img is not None, "NIRC2 example frame failed to load"

    win.fm_fg_flip_x.setChecked(False)
    win.fm_fg_flip_y.setChecked(False)
    fm_render(win, app)
    grab(win, app, "fig_field_map_flip_a.png", 1700, 950)

    win.fm_fg_flip_x.setChecked(True)
    fm_render(win, app)
    grab(win, app, "fig_field_map_flip_b.png", 1700, 950)

    win.fm_fg_flip_x.setChecked(False)
    win._clear_frame()
    # restore the base target -- _load_local_sky made the frame's own
    # target active, same cleanup shot_b3_target_tab does
    win._targets = []
    win._add_target(engine.DEF_TARGET_NAME, TARGET_RA, TARGET_DEC, select=True)


def _measured_sr_tab_index(win):
    return next(i for i in range(win.plot_tabs.count())
                if win.plot_tabs.tabText(i) == "Measured SR")


def _nirc2_measure_file(win, app, fname):
    """Measure one example frame through the real worker path (the file
    list's double-click flow) and wait for it to finish.  The M79
    worked-example frames moved to the keck_ao_experiments repo (split
    2026-07-25): i260131 frames resolve there ($M79_DATA overrides)."""
    if fname.startswith("i260131"):
        ex_dir = os.environ.get("M79_DATA", os.path.expanduser(
            "~/keck_ao_experiments/m79_slgs_vs_ltao_20260131/data"))
    else:
        ex_dir = os.path.join(os.path.dirname(HERE), "examples")
    win.n2_path.setText(ex_dir)                # populates the file list too
    label = os.path.splitext(fname)[0]
    win._nirc2_start(files=[(label, os.path.join(ex_dir, fname))])
    pump(lambda: win._n2_worker is None or not win._n2_worker.isRunning(),
         timeout=120)
    pump(lambda: win.n2_go.isEnabled(), timeout=120)
    settle(app, 10)


def _m79_context(win, app):
    """The 2026-01-31 UT M79 off-axis-experiment context both g-shots
    need: that night's REAL MKWC data (fetched; cached in mkwc_cache/
    after the first run) + the frames' own pointing as the target, plus
    the crowded-field photometry settings the worked example used."""
    win.mode_fetch.setChecked(True)
    win.fetch_date.setDate(QtCore.QDate(2026, 1, 31))
    # the frames' own pointing (header RA/DEC): M79, near transit that night
    win._targets = []
    win._add_target("M79", "05h24m10.6s", "-24d31m27s", select=True)
    run_sync(win, app)
    win.plot_tabs.setCurrentIndex(_measured_sr_tab_index(win))
    win.n2_robust_sky.setChecked(True)         # globular cluster: crowding
    win.n2_auto_rad.setChecked(True)           # is the norm (the example's
    settle(app)                                # own settings)


def shot_g3_measured_sr(win, app):
    """Measured SR tab (2026-07-23), Image view: a real OSIRIS M79 frame
    from the 2026-01-31 UT off-axis experiment (the KAON worked example),
    measured with the crowded-field settings the example used, against the
    night's REAL fetched MKWC prediction — so the MODEL PSF / MEASURED STAR
    cutouts, the MEASURED/PREDICTED/Δ readouts, and the two aligned log
    lines are all authentically populated in one frame."""
    _m79_context(win, app)
    # AUTOFIND off = the IDL tool's CLICK-ON-THE-STAR flow: the brightest
    # pixel is the R=10 NGS itself, whose core profile doesn't yield a
    # clean FWHM here -- click a bright, uncrowded cluster star instead
    # (SR 0.395, FWHM 49 mas; the frame's field mean is 0.41)
    win.n2_autofind.setChecked(False)
    _nirc2_measure_file(win, app, "i260131_a028002.fits")
    win._nirc2_measure_at(964.3, 1575.1)
    settle(app, 6)
    assert win.n2_strehl_out.text(), "measurement must have landed"
    assert float(win.n2_fwhm_out.text()) > 0, "FWHM must be physical"
    assert win.n2_pred_sr.text(), \
        "PREDICTED SR must be filled (frame time inside the fetched night)"
    grab(win, app, "fig_gui_measured_sr_tab.png", 1944, 950)

    # companion: the live pick-zoom magnifier (AUTOFIND off, cursor over
    # the field) -- it REPLACES the MEASURED STAR panel while hovering, so
    # it cannot coexist with the cutout pair in a single grab
    import types
    win.n2_autofind.setChecked(False)
    win._n2_pick_locked = False
    r = win._n2_last_draw[1]
    ev = types.SimpleNamespace(xdata=r.x + 55.0, ydata=r.y + 40.0,
                               inaxes=win.n2_fig.axes[0])
    win._on_nirc2_motion(ev)
    settle(app, 6)
    assert win.n2_cap_star.text() == "PICK ZOOM"
    grab(win, app, "fig_gui_measured_sr_pickzoom.png", 1944, 950)
    win.n2_autofind.setChecked(True)


def shot_g4_measured_field_map(win, app):
    """Measured field map (2026-07-23): auto-find field run on the crowded
    M79 core frame (a026002, the 30-arcsec-North LTAO step) with the Auto
    star count -- stars marked and labelled, the gold peak ring + downhill
    gradient arrow, and the field-stats readout including the fitted
    theta0(eff) line.
    Frame choice: a026002 is the one whose auto run yields a clean theta0
    fit (30 stars); a027002's field measures flat (no fit), and a028002's
    never fitted in the worked example either. The PREDICTED boxes stay
    empty here -- a026002's timestamp is 653 s from the nearest MASS
    profile sample, past the 600 s match tolerance -- which is the
    tool's honest answer, tooltip'd as such. The _m79_context crowded-
    field settings are load-bearing, not cosmetics: without robust sky +
    auto apertures the field scatter doubles and the theta0 fit
    degenerates to "flat or noise-dominated"."""
    _m79_context(win, app)
    win.n2_autofind.setChecked(True)
    win.n2_nstars.setValue(0)                  # Auto: quality decides
    # EE aperture correction ON so the highlighted h badge beside the
    # field stats is populated -- one of the two things the Rev D/E/F
    # figure request wants this retake to show (the other is Reject star)
    win.n2_ee_corr.setChecked(True)
    _nirc2_measure_file(win, app, "i260131_a026002.fits")
    win._on_nirc2_measure_field()
    pump(lambda: getattr(win, "_n2_field_queue", None) is None
         and win.n2_field_btn.isEnabled(), timeout=300)
    settle(app, 10)
    assert len(win._n2_field) >= engine.GRADIENT_MIN_STARS
    stats = win.n2_field_stats.text()
    assert "θ₀(eff)" in stats, f"theta0 fit missing: {stats}"
    win.n2_view_tabs.setCurrentIndex(1)        # Field map sub-tab
    settle(app, 6)
    grab(win, app, "fig_gui_measured_field_map.png", 1944, 950)

    # restore the stock state for anything that runs after
    win.n2_view_tabs.setCurrentIndex(0)
    win.n2_nstars.setValue(5)
    win.n2_robust_sky.setChecked(False)
    win.n2_auto_rad.setChecked(False)
    win._on_nirc2_field_clear()
    win.mode_local.setChecked(True)
    win._targets = []
    win._add_target(engine.DEF_TARGET_NAME, TARGET_RA, TARGET_DEC, select=True)
    win.plot_tabs.setCurrentIndex(0)
    run_sync(win, app)


def shot_h6_starlist_picker(win, app):
    """Rev F: the starlist picker with a row selected so the detail panel
    is populated -- Az/El, Moon separation + illumination, the HA column,
    and the "Evaluate at" date/time pair with its (HST)/(UT) label.

    The colour cues live on the DETAIL lines (not the table rows), so the
    row chosen is one that actually trips a non-green cue at the chosen
    evaluation time -- scanned for rather than hard-coded, since which
    starlist entries are down/vignetted depends on the date."""
    lst = os.path.join(os.path.dirname(HERE), "examples", "synthetic_k1lgs.lst")
    win.tabs.setCurrentIndex(1)                     # Target tab
    win._open_starlist(lst)
    settle(app, 4)
    dlg, table = win._starlist_dialog, win._starlist_table
    assert dlg is not None and table is not None, "picker must be open"
    # an arbitrary fixed observing instant (synthetic list, any night works)
    win._starlist_date_edit.setDate(QtCore.QDate(2026, 7, 28))
    win._starlist_time_edit.setTime(QtCore.QTime(22, 30))
    settle(app, 4)
    # prefer a row whose pointing state is not 'open' (red/amber cue), else
    # fall back to the first row rather than failing the capture
    chosen = 0
    for row in range(min(table.rowCount(), 60)):
        win._starlist_show_detail(row)
        cue = win._starlist_detail_azel.property("cue")
        if cue in ("err", "warn"):
            chosen = row
            break
    win._starlist_show_detail(chosen)
    table.selectRow(chosen)
    settle(app, 4)
    dlg.resize(900, 640)
    settle(app, 6)
    ok = dlg.grab().save(os.path.join(OUT, "fig_gui_starlist_picker.png"), "PNG")
    print(f"  [{'ok' if ok else 'FAIL'}] fig_gui_starlist_picker.png  "
          f"(900x640, row {chosen}, cue "
          f"{win._starlist_detail_azel.property('cue')})")
    dlg.close()
    settle(app)


def shot_h7_image_log(win, app):
    """Rev F: the log popped out on its Image log tab -- the sortable
    table with real measured rows, the Guide mag source column populated
    (the column that makes the resolved-magnitude story visible), and the
    Delete selected / Export CSV buttons in frame.

    Measures three M79 frames so the table has more than one row and the
    duplicate guard never fires (distinct frames, distinct positions).

    The M79 target is given its real tip-tilt magnitude (the field's
    R = 10 NGS -- the same star g3's docstring identifies as the frame's
    brightest pixel) so the Guide mag source column shows a RESOLVED
    entry rather than the ASSUMED fallback. With no magnitude attached
    the column would read "estimator default -- ASSUMED" on every row,
    which is the honest output for an unconfigured target but shows
    none of what the column exists to convey."""
    _m79_context(win, app)
    idx = win._nirc2_target_index("M79")
    if idx is not None:
        win._targets[idx]["tt_mag"] = 10.0
    win.tt_mag.setValue(10.0)
    # AUTOFIND's brightest pixel on these frames is the R=10 NGS, whose
    # core does not yield a half-max crossing -- radial_profile_fwhm then
    # returns its -1 px failure sentinel and the row carries no FWHM. Use
    # the click flow on the same clean cluster star g3 measures (SR 0.395,
    # FWHM 49 mas) so the captured rows show real values in every column.
    win.n2_autofind.setChecked(False)
    _nirc2_measure_file(win, app, "i260131_a028002.fits")
    # several clean cluster stars from the ONE verified frame (the other
    # dither pointings put the star elsewhere, so a fixed click misses).
    # Rows sharing a frame number but differing in pixel position is also
    # the honest picture of measuring a field -- and exactly what the
    # duplicate guard is position-aware for.
    # A scripted run must never be able to raise the duplicate-frame
    # dialog: it is MODAL, so with no one to answer it the capture hangs
    # forever (hit while building this shot -- find_stars can return a
    # candidate within the guard's 3 px of the star already clicked in
    # this same frame). Pre-setting the run's policy skips silently.
    win._n2_dup_batch_policy = "skip"
    win._nirc2_measure_at(964.3, 1575.1)          # g3's verified star
    settle(app, 4)
    photrad_px = (win.n2_photrad.value() * 1000.0
                  / win._n2_params.plate_scale_mas)
    for (sx, sy) in engine.find_stars(win._n2_image, n_stars=8,
                                      exclude_px=photrad_px):
        if len(win._n2_csv_rows) >= 4:
            break
        win._nirc2_measure_at(float(sx), float(sy))
        settle(app, 2)
        # drop anything whose FWHM fit failed (sentinel -> None) so the
        # captured table shows real values in every column
        if win._n2_csv_rows and win._n2_csv_rows[-1]["measured_fwhm"] is None:
            del win._n2_csv_rows[-1]
    assert len(win._n2_csv_rows) >= 2, "need a few real rows to show"
    win._on_nirc2_log_popout()
    settle(app, 6)
    dlg = win._n2_log_dialog
    assert dlg is not None
    tabs = dlg.findChild(QtWidgets.QTabWidget)
    tabs.setCurrentIndex(1)                         # the Image log tab
    settle(app, 4)
    tbl = win._n2_csv_table
    tbl.resizeColumnsToContents()
    settle(app, 4)
    dlg.resize(1800, 380)
    settle(app, 6)
    ok = dlg.grab().save(os.path.join(OUT, "fig_gui_image_log.png"), "PNG")
    print(f"  [{'ok' if ok else 'FAIL'}] fig_gui_image_log.png  "
          f"(1500x620, {tbl.rowCount()} row(s))")
    dlg.close()
    settle(app)


def shot_h8_fieldmap_rightclick(win, app):
    """Rev F: the Field-map right-click menu open, with both "Drop science
    target here" and the new "Redefine field centre here" visible -- the
    pair the KAON and the 1556 manual warn are easy to confuse.

    A QMenu is its own top-level window, so win.grab() cannot include it
    and menu.exec() would block the capture. The menu is therefore shown
    non-blockingly (popup) and COMPOSITED onto the window grab at the
    click point. Both halves are real widget renders at their real
    positions -- nothing is drawn by hand -- but note it is a composite,
    not a single-surface screenshot."""
    win.status.setText("")      # don't inherit a previous shot's status line
    win.plot_tabs.setCurrentIndex(1)                # Field map
    win._fieldmap_dirty = True
    win._render_field_map_if_visible()
    settle(app, 8)
    canvas0 = win._fm_holder["canvas"]
    assert canvas0.figure.axes, "field map must have rendered"
    ax = canvas0.figure.axes[0]
    # a click INSIDE the plotted field (the axes span the 20" FOV, so a
    # point outside it both reads oddly in the menu text and pushes the
    # popup off the right edge of the grab), in DISPLAY pixels
    xd, yd = -4.0, 4.0
    px, py = ax.transData.transform((xd, yd))
    canvas = win._fm_holder["canvas"]
    h = canvas.figure.bbox.height
    pt = canvas.mapTo(win, QtCore.QPoint(int(px / canvas.devicePixelRatioF()),
                                         int((h - py) / canvas.devicePixelRatioF())))
    shown = {}

    def _popup_instead(self, *a, **k):
        self.popup(win.mapToGlobal(pt))
        shown["menu"] = self
        return None
    orig_exec = QtWidgets.QMenu.exec
    QtWidgets.QMenu.exec = _popup_instead
    try:
        ev = types.SimpleNamespace(inaxes=ax, xdata=xd, ydata=yd, button=3)
        win._on_fm_canvas_click(ev)
        settle(app, 8)
        menu = shown.get("menu")
        assert menu is not None, "right-click menu must have opened"
        win.resize(1944, 950)
        settle(app, 6)
        base = win.grab()
        mshot = menu.grab()
        p = QtGui.QPainter(base)
        p.drawPixmap(pt, mshot)
        p.end()
        ok = base.save(os.path.join(OUT, "fig_gui_fieldmap_rightclick.png"),
                       "PNG")
        print(f"  [{'ok' if ok else 'FAIL'}] fig_gui_fieldmap_rightclick.png "
              f"(1944x950, composited menu {mshot.width()}x{mshot.height()})")
        menu.close()
    finally:
        QtWidgets.QMenu.exec = orig_exec
    settle(app)


def shot_f1_pred_norun(win, app):
    """2026-08-12 features: prediction scenario with NO night loaded.
    Uses a FRESH MainWindow (the shared one has a run) so the grabs show
    the true no-Run state: (a) the field map rendered straight from the
    controls + scenario, (b) the Error-terms tab's predicted snapshot
    breakdown. The scenario spinboxes stay at the tab's defaults so the
    numbers match the KAON walkthrough text."""
    w2 = gui.MainWindow()
    w2.resize(1944, 950)
    w2.show()
    settle(app)
    w2.tel_k1.setChecked(True)
    w2.band_combo.setCurrentText("K")
    w2.tabs.setCurrentIndex(5)               # Prediction tab in the panel
    w2.pred_enable.setChecked(True)          # jumps to the Field map tab
    settle(app, 10)
    path = os.path.join(OUT, "fig_gui_pred_norun_fieldmap.png")
    ok = w2.grab().save(path, "PNG")
    print(f"  [{'ok' if ok else 'FAIL'}] fig_gui_pred_norun_fieldmap.png")
    w2.plot_tabs.setCurrentIndex(2)          # Error terms (predicted bars)
    settle(app, 10)
    path = os.path.join(OUT, "fig_gui_pred_terms.png")
    ok = w2.grab().save(path, "PNG")
    print(f"  [{'ok' if ok else 'FAIL'}] fig_gui_pred_terms.png")
    w2.close()
    settle(app)


def shot_f2_pred_with_night(win, app):
    """Prediction tab + field map WITH a night loaded (1542 Fig 11 class):
    the with-night companion to f1 -- shows the updated note text ("field
    map and error terms"), the new checkbox wording, and the scenario
    driving the field map over a loaded night. Restores pred-off after."""
    win.tabs.setCurrentIndex(5)                    # Prediction tab
    scroll_tab(win, 5, px=0)
    win.pred_enable.setChecked(True)               # jumps to Field map tab
    fm_render(win, app)
    settle(app, 8)
    grab(win, app, "fig_gui_pred_with_night_fieldmap.png", 1944, 950)
    win.pred_enable.setChecked(False)
    settle(app)


def shot_f3_lgs_trick_k(win, app):
    """LGS tab with TRICK (K) selected and the night's Cn2 profile plot
    (1556 Fig 11 class): the gain spinbox reads 1.00 (KAON 1303 section 5.5
    default) and the sensor-swap side effects (mag label, dichroic note)
    are visible. Restores STRAP (R) after."""
    win.tabs.setCurrentIndex(3)                    # LGS tab
    win.tt_sensor.setCurrentText("TRICK (K)")
    run_sync(win, app)
    scroll_tab(win, 3, px=0)
    settle(app, 6)
    grab(win, app, "fig_gui_lgs_trick_k.png", 1944, 950)
    win.tt_sensor.setCurrentText("STRAP (R)")
    run_sync(win, app)


SHOTS = {
    "a2": shot_a2_main,
    "a3": shot_a3_fwhm,
    "a1": shot_a1_wfe,
    "a4": shot_a4_ngs,
    "a5": shot_a5_offset_modes,
    "b4": shot_b4_lgs_tab,
    "b3": shot_b3_target_tab,
    "a6": shot_a6_field_map,
    "a7": shot_a7_field_map_sky,
    "b1": shot_b1_field_map_catalog,
    "b2": shot_b2_field_map_targets,
    "b5": shot_b5_field_map_pa,
    "b6": shot_b6_field_map_flip,
    "e1": shot_e1_nighttime_mode,
    "e2": shot_e2_dark_theme,
    "e3": shot_e3_summary_stats,
    "e3b": shot_e3b_summary_stats_trick_na,
    "e4": shot_e4_target_guide_star,
    "e5": shot_e5_fa_advisory,
    "e6": shot_e6_fa_geometry,
    "g3": shot_g3_measured_sr,
    "g4": shot_g4_measured_field_map,
    "h6": shot_h6_starlist_picker,
    "h7": shot_h7_image_log,
    "h8": shot_h8_fieldmap_rightclick,
    "f1": shot_f1_pred_norun,
    "f2": shot_f2_pred_with_night,
    "f3": shot_f3_lgs_trick_k,
}
# order matters: b4/tt_sensor and a5/tt_offset touch shared state, restored
# at the end of each; field-map shots (a6/a7/b1/b2/b5/b6) go last since
# they're the slowest (network fetch for a7/b5) and most order-sensitive.
# e1/e2/e3/e3b/e4 (2026-07-22) go after those: e1 touches dark-theme state
# (auto-dark) which e2 then sets explicitly, and e4 touches tt_sensor/
# tt_offset/_targets like b4/b3/a5 before it.
ORDER = ["a2", "a3", "a1", "a4", "a5", "b4", "b3", "a6", "a7", "b1", "b2",
         "b5", "b6", "e1", "e2", "e3", "e3b", "e4", "e5", "e6", "g3", "g4",
         "f1", "f2", "f3"]   # f1 spawns its own no-run window; f2/f3 use
                             # the shared run and restore their state
# g3/g4 go last: they swap the WHOLE data context (fetched 20260131 night +
# M79 target) and restore the local-20260525 baseline when done.


def main():
    keys = [k for k in sys.argv[1:]] or ORDER
    os.makedirs(OUT, exist_ok=True)   # QPixmap.save fails silently without it
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow()
    win.resize(1944, 950)
    win.show()
    app.processEvents()
    setup_common(win)
    run_sync(win, app)
    for k in keys:
        print(f"-- {k} --")
        SHOTS[k](win, app)
    win.close()


if __name__ == "__main__":
    main()
