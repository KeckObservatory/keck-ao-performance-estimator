#!/usr/bin/env python3
"""Guide-star catalogue overlay: the engine registry + pure table parser +
best-available-band magnitude picker + RA/Dec->field-xy converter, and the GUI
overlay (load a catalogue -> stars on the field map, right-click one to select
it as the TT/NGS guide star, taking its position and best-band magnitude).
Also covers the Rank button (auto-ranking by delivered performance at the
science target -- gs_ranking.rank_guide_stars via FieldMapOverlaysMixin;
see gs_ranking_model.py for the engine-only physics contract).

The live Vizier query (engine.query_guide_stars / CatalogFetchWorker) is NOT
exercised here -- like the DSS/2MASS backdrop fetch it is offline-untested; the
overlay is driven with synthetic stars via _on_catalog_loaded. Run headless."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
np = engine.np
DATA = os.path.join(HERE, "data")


def pump(cond, timeout=90):
    app = QtWidgets.QApplication.instance(); t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents(); QtCore.QThread.msleep(10)


def engine_catalog():
    # the observers' five catalogues are registered, in order
    assert list(engine.CATALOGS) == ["GSC 2.4", "2MASS", "UCAC4",
                                     "PanSTARRS DR2", "Gaia DR2"]
    spec = engine.CATALOGS["Gaia DR2"]
    # synthetic Vizier table: 3 rows, one masked Gmag, one un-parseable Dec
    t = Table({
        "RA_ICRS": np.ma.array([150.0, 150.001, 149.998], mask=[0, 0, 0]),
        "DE_ICRS": np.ma.array([36.0, 36.0005, 0.0], mask=[0, 0, 1]),
        "Source": np.array([111, 222, 333]),
        "Gmag": np.ma.array([12.3, 99.0, 14.1], mask=[0, 1, 0]),
        "BPmag": np.array([12.6, 15.0, 14.4]),
        "RPmag": np.array([11.9, 14.2, 13.5]),
    })
    stars = engine.parse_catalog_table(t, spec)
    assert len(stars) == 2, "the masked-Dec row must be dropped"
    assert stars[0]["id"] == "111" and abs(stars[0]["ra"] - 150.0) < 1e-9
    assert stars[0]["mags"]["G"] == 12.3 and stars[1]["mags"]["G"] is None
    # best-available band: optical R(0.65) -> G(0.60); IR K(2.2) -> RP(0.80)
    b, v = engine.pick_mag(stars[0]["mags"], engine.SENSOR_BAND_UM["R"])
    assert b == "G" and abs(v - 12.3) < 1e-9, (b, v)
    b, _ = engine.pick_mag(stars[0]["mags"], engine.SENSOR_BAND_UM["K"])
    assert b == "RP", b
    # a star with only IR mags: R still resolves to the nearest (J)
    b, v = engine.pick_mag({"J": 9.0, "H": 8.5, "K": 8.2},
                           engine.SENSOR_BAND_UM["R"])
    assert b == "J" and abs(v - 9.0) < 1e-9, (b, v)
    b, _ = engine.pick_mag({"J": 9.0, "H": 8.5, "K": 8.2},
                           engine.SENSOR_BAND_UM["K"])
    assert b == "K", b
    assert engine.pick_mag({"G": None}, 0.65) == (None, None)
    # field xy: plot frame x = West+ (−East), y = North+
    xy = engine.stars_field_xy(stars, 150.0, 36.0)
    assert abs(xy[0]["x"]) < 1e-3 and abs(xy[0]["y"]) < 1e-3, xy[0]
    assert xy[1]["x"] < 0 and xy[1]["y"] > 0, "star E+N of centre -> x<0, y>0"
    print("  [ok] engine: registry + parse (masked/ids) + best-band pick + "
          "field-xy signs")

    # sensing-band magnitude estimate: exact / colour-transform / rough / none
    assert engine.estimate_sensing_mag({"R": 13.4}, "R") == (13.4, "exact", "R")
    # Gaia G,BP,RP -> Cousins R (Evans 2018): G − R = a + b·c + d·c² , c=BP−RP
    c = 11.3 - 10.6
    want = 11.0 - (-0.003226 + 0.3833 * c - 0.1345 * c * c)
    v, kind, lab = engine.estimate_sensing_mag(
        {"G": 11.0, "BP": 11.3, "RP": 10.6}, "R")
    assert kind == "est" and abs(v - want) < 1e-9 and "Gaia" in lab, (v, kind, lab)
    # PanSTARRS r,i -> R (Lupton) is an estimate; 2MASS gives K exactly
    v, kind, _ = engine.estimate_sensing_mag({"r": 12.0, "i": 11.8}, "R")
    assert kind == "est", (v, kind)
    assert engine.estimate_sensing_mag({"J": 9, "H": 8.5, "K": 8.2}, "K") \
        == (8.2, "exact", "K")
    # only IR photometry but R wanted -> rough nearest-band fallback (flagged)
    v, kind, _ = engine.estimate_sensing_mag({"J": 9, "H": 8.5, "K": 8.2}, "R")
    assert kind == "near" and v == 9.0, (v, kind)
    assert engine.estimate_sensing_mag({"G": None}, "R") == (None, None, None)
    print("  [ok] engine: sensing-band magnitude estimate "
          "(exact / colour-transform / rough / none)")


def gui_overlay():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1500, 950); win.show(); app.processEvents()
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(f"{DATA}/20260525_dimm.dat")
    win.mass_edit.setText(f"{DATA}/20260525_mass.dat")
    win.masspro_edit.setText(f"{DATA}/20260525_masspro.dat")
    win.tel_k1.setChecked(True)
    win.ra_edit.setText("10h00m00s"); win.dec_edit.setText("+20d00m00s")
    win._validate(); win.on_run()
    pump(lambda: win.res is not None)

    def idle():
        return (not win._fm_debounce.isActive()
                and not win._fm_settle.isActive())
    win.plot_tabs.setCurrentIndex(1); app.processEvents(); pump(idle)
    assert [win.fm_catalog.itemText(i) for i in range(win.fm_catalog.count())] \
        == list(engine.CATALOGS)

    # drive the overlay with synthetic stars (no network): A = 5"E/3"N bright
    # optical; B = 10"W, IR-only
    fc = win._field_center_deg(); assert fc is not None
    cen = SkyCoord(fc[0] * u.deg, fc[1] * u.deg)
    a = cen.spherical_offsets_by(5 * u.arcsec, 3 * u.arcsec)
    b = cen.spherical_offsets_by(-10 * u.arcsec, 0 * u.arcsec)
    stars = [
        {"id": "A", "ra": float(a.ra.deg), "dec": float(a.dec.deg),
         "mags": {"G": 11.0, "BP": 11.3, "RP": 10.6}},
        {"id": "B", "ra": float(b.ra.deg), "dec": float(b.dec.deg),
         "mags": {"J": 9.0, "H": 8.5, "K": 8.2, "G": None}},
    ]
    # A: bright optical (Gaia -> R est ~10.8, usable); B: bright IR but for R
    # sensing only a rough IR fallback (~J 9, usable); make a 3rd very-faint
    # star to exercise the too-faint hollow branch
    faint = cen.spherical_offsets_by(8 * u.arcsec, -6 * u.arcsec)
    stars.append({"id": "F", "ra": float(faint.ra.deg), "dec": float(faint.dec.deg),
                  "mags": {"G": 19.5, "BP": 20.0, "RP": 18.8}})
    win.tt_sensor.setCurrentText("STRAP (R)")           # size by R
    win._on_catalog_loaded("Gaia DR2", stars, ""); pump(idle); app.processEvents()
    assert win._catalog_name == "Gaia DR2" and len(win._catalog_stars) == 3
    ax = next(a for a in win._fm_holder["canvas"].figure.axes if a.images)
    cols = [c for c in ax.collections if len(c.get_offsets()) == 3]
    assert cols, "3 catalogue stars drawn as one scatter"
    sc = cols[0]
    sizes = np.asarray(sc.get_sizes(), float)
    fcs = sc.get_facecolors()
    # order matches insertion: A (R~10.8, usable), B (usable), F (R>17.5 limit).
    # F is the too-faint one: smallest; every marker (A included) is UNFILLED
    # -- a filled dot, however small, would hide the real star underneath a
    # sky backdrop (Rev C follow-up, 2026-07-18) -- distinguished by size and
    # edge colour instead: both are the same red hue (FM_C_CATSTAR -- reads
    # over a grayscale backdrop, 2026-07-18 follow-up), too-faint just dimmer
    # (lower alpha) than usable.
    assert sizes[2] < sizes[0], (sizes, "faint star must be smaller")
    assert fcs[2][3] == 0.0, "too-faint star F must be hollow (transparent face)"
    assert fcs[0][3] == 0.0, "usable star A must ALSO be unfilled (edge only)"
    ecs = sc.get_edgecolors()
    assert tuple(ecs[0][:3]) == tuple(ecs[2][:3]), \
        "usable and too-faint markers must share the same red hue"
    assert ecs[0][0] > ecs[0][1] and ecs[0][0] > ecs[0][2], \
        f"catalogue star edge must be red-dominant, got {tuple(ecs[0])}"
    assert ecs[0][3] > ecs[2][3], \
        "too-faint star F must be dimmer (lower alpha) than usable star A"
    sx = win._catalog_stars_xy()
    assert abs(sx[0]["x"] + 5.0) < 0.05 and abs(sx[0]["y"] - 3.0) < 0.05, sx[0]
    print(f"  [ok] overlay: stars sized by sensing-band brightness "
          f"(A={sizes[0]:.0f}, F too-faint hollow={sizes[2]:.0f})")

    # right-click hit-test
    assert win._catalog_star_near(sx[0]["x"], sx[0]["y"])["id"] == "A"
    assert win._catalog_star_near(50.0, 50.0) is None
    print("  [ok] right-click hit-test finds a star / misses empty field")

    # left-click to INSPECT a star: it highlights + reports mags without
    # touching the guide-star controls
    tt_before = win.tt_mag.value()
    win._inspect_catalog_star(sx[0]["x"], sx[0]["y"]); pump(idle); app.processEvents()
    assert win._catalog_inspected == "A"
    assert win.tt_mag.value() == tt_before, "inspect must not change the mag"
    txt = win.fm_catalog_status.text()
    assert "“A”" in txt and "G=11.0" in txt, txt
    ax = next(a for a in win._fm_holder["canvas"].figure.axes if a.images)
    rings = [ln for ln in ax.get_lines() if ln.get_marker() == "o"
             and ln.get_markerfacecolor() in ("none", (0, 0, 0, 0))
             and len(ln.get_xdata()) == 1]
    assert rings, "inspected star must be ringed"
    # clicking empty space clears the inspection
    win._inspect_catalog_star(50.0, 50.0); pump(idle)
    assert win._catalog_inspected is None
    print("  [ok] left-click inspects (highlights + reports mags), no selection")

    # select A as TT (STRAP R): A has no R band -> Cousins R ESTIMATED from
    # Gaia G,BP,RP (Evans 2018), not the raw G
    win.tt_sensor.setCurrentText("STRAP (R)")
    win._fm_select_star(stars[0], "tt"); pump(idle)
    assert win.tt_offset.mode.currentIndex() == 2, "must land in star RA/Dec mode"
    assert win.tt_offset.sra.text() and win.tt_offset.sdec.text(), \
        "a catalogue pick must store the star's real RA/Dec, not just an offset"
    txy = win.tt_offset.offset_xy(win._sky_field_center())
    assert abs(txy[0] + 5.0) < 0.1 and abs(txy[1] - 3.0) < 0.1, txy
    cc = 11.3 - 10.6
    want_r = 11.0 - (-0.003226 + 0.3833 * cc - 0.1345 * cc * cc)
    # tt_mag spinbox quantises to 0.1; the estimate (~10.80) must land there,
    # and crucially NOT be the raw G (11.0)
    assert abs(win.tt_mag.value() - want_r) < 0.06, (win.tt_mag.value(), want_r)
    assert abs(win.tt_mag.value() - 11.0) > 0.1, "must be estimated R, not raw G"
    assert "estimated" in win.fm_catalog_status.text(), win.fm_catalog_status.text()
    # select B as NGS (optical R): only IR mags -> rough nearest-band (J)
    win._fm_select_star(stars[1], "ngs"); pump(idle)
    nxy = win.ngs_offset.offset_xy(win._sky_field_center())
    assert abs(nxy[0] - 10.0) < 0.1 and abs(nxy[1]) < 0.1, nxy
    assert abs(win.ngs_bright.value() - 9.0) < 1e-6, win.ngs_bright.value()
    print("  [ok] select as TT/NGS: position + estimated sensing-band magnitude")

    # TRICK K sensor pulls the K-band magnitude exactly
    win.tt_sensor.setCurrentText("TRICK (K)"); app.processEvents()
    win._fm_select_star(stars[1], "tt"); pump(idle)
    assert abs(win.tt_mag.value() - 8.2) < 1e-6, win.tt_mag.value()
    print("  [ok] TRICK (K) sensor -> selection pulls the exact K-band magnitude")

    # --- guide-star auto-ranking (Rank button) -------------------------------
    win.tt_sensor.setCurrentText("STRAP (R)"); app.processEvents()
    assert win.fm_catalog_rank.isEnabled(), "Rank must be enabled with stars loaded"
    win._rank_guide_stars(); pump(idle); app.processEvents()
    assert win._gs_ranking, "ranking must populate after Rank"
    assert {e["id"] for e in win._gs_ranking} == {"A", "B", "F"}
    by_id = {e["id"]: e for e in win._gs_ranking}
    # B: bright (R~9 via nearest-band J) but far; A: fainter est. R but close;
    # B wins narrowly at STRAP(R) -- both usable, F excluded (too faint)
    assert by_id["B"]["rank"] == 1 and by_id["A"]["rank"] == 2, by_id
    assert by_id["F"]["rank"] is None and "too faint" in by_id["F"]["excluded_reason"], \
        by_id["F"]
    assert by_id["A"]["offset_arcsec"] < by_id["B"]["offset_arcsec"], \
        "A is the closer star (offset must reflect that even though B ranks higher)"
    print(f"  [ok] rank_guide_stars: B #1 / A #2 (STRAP R), F excluded "
          f"({by_id['F']['excluded_reason']})")

    ax = next(a for a in win._fm_holder["canvas"].figure.axes if a.images)
    badges = [t for t in ax.texts
             if t.get_text() in ("1", "2") and t.get_bbox_patch() is not None]
    assert len(badges) == 2, "top-2 usable stars (of 3, one excluded) must be badged"
    print(f"  [ok] top-ranked stars badged on the map ({len(badges)} badges)")

    assert win._gs_rank_dialog is not None, "ranking table dialog must open"
    print("  [ok] ranking table dialog opened")

    # the table includes RA/Dec columns (per Eduardo, 2026-07-19)
    headers = [win._gs_rank_table.horizontalHeaderItem(i).text()
              for i in range(win._gs_rank_table.columnCount())]
    assert "RA" in headers and "Dec" in headers, headers

    def row_of(star_id):
        # visual row of a star id, scanning the id column: the table is
        # user-sortable (2026-07-21), so no fixed id->row map exists
        t = win._gs_rank_table
        return next(r for r in range(t.rowCount())
                    if t.item(r, 1).text() == star_id)

    a_row = row_of("A")
    ra_col, dec_col = headers.index("RA"), headers.index("Dec")
    assert "h" in win._gs_rank_table.item(a_row, ra_col).text()
    assert "d" in win._gs_rank_table.item(a_row, dec_col).text()
    print(f"  [ok] ranking table includes RA/Dec columns ({headers})")

    # left-click a ranked star on the map HIGHLIGHTS its row (not selects it
    # as the guide star -- tt_mag must be untouched)
    sx2 = win._catalog_stars_xy()
    a_xy = next(s for s in sx2 if s["id"] == "A")
    win.tt_mag.setValue(0.0)                      # sentinel
    win._inspect_catalog_star(a_xy["x"], a_xy["y"]); pump(idle)
    sel_rows = {ix.row() for ix in win._gs_rank_table.selectedIndexes()}
    assert sel_rows == {a_row}, (sel_rows, a_row)
    assert win.tt_mag.value() == 0.0, "left-click must only highlight, never select"
    print("  [ok] left-click on the map highlights (not selects) the matching row")

    # clicking empty space clears the highlight too
    win._inspect_catalog_star(50.0, 50.0); pump(idle)
    assert not win._gs_rank_table.selectedIndexes(), \
        "clicking empty space must clear the table highlight"
    print("  [ok] clicking empty space clears the table highlight")

    # the REVERSE direction: single-clicking a table row highlights that star
    # on the map (ring it) -- highlight only, tt_mag untouched
    a_row2 = row_of("A")
    tt_before2 = win.tt_mag.value()
    win._gs_rank_row_clicked(win._gs_ranking, a_row2); pump(idle)
    assert win._catalog_inspected == "A", "table row click must highlight the map star"
    assert win.tt_mag.value() == tt_before2, "row click highlights only, never selects"
    ax2 = next(a for a in win._fm_holder["canvas"].figure.axes if a.images)
    rings2 = [ln for ln in ax2.get_lines() if ln.get_marker() == "o"
              and ln.get_markerfacecolor() in ("none", (0, 0, 0, 0))
              and len(ln.get_xdata()) == 1]
    assert rings2, "the row-clicked star must be ringed on the map"
    print("  [ok] clicking a table row highlights the star on the map (reverse)")

    # double-clicking a ranked row selects that star as the TT star, exactly
    # like right-click "Set as TT star" -- absolute RA/Dec, sensing-band mag
    b_row = next(i for i, e in enumerate(win._gs_ranking) if e["id"] == "B")
    win._gs_rank_select(win._gs_ranking, b_row, "tt"); pump(idle)
    assert win.tt_offset.mode.currentIndex() == 2, "must select via star RA/Dec"
    assert abs(win.tt_mag.value() - 9.0) < 0.06, win.tt_mag.value()
    print("  [ok] double-clicking a ranked row selects it as the TT star")

    # --- column sorting (2026-07-21): headers sort by VALUE, and the
    #     row<->star mapping survives the re-ordering ------------------------
    t = win._gs_rank_table
    mag_col = headers.index("mag")
    t.sortItems(mag_col, QtCore.Qt.SortOrder.AscendingOrder)
    mag_texts = [t.item(r, mag_col).text() for r in range(t.rowCount())]
    nums = [float(s.rstrip("*")) for s in mag_texts if s != "—"]
    assert nums == sorted(nums), f"mag column must sort numerically: {mag_texts}"
    assert all(s == "—" for s in mag_texts[len(nums):]), \
        f"no-mag rows must sort last (ascending): {mag_texts}"
    # map->table highlight still finds the star wherever its row moved
    win._gs_rank_highlight("A"); app.processEvents()
    sel_after = {ix.row() for ix in t.selectedIndexes()}
    assert sel_after == {row_of("A")}, (sel_after, row_of("A"))
    # double-click through the REAL signal on the SORTED table still selects
    # the star that row displays, not whatever ranked[row] happens to be
    win.tt_mag.setValue(0.0)
    t.cellDoubleClicked.emit(row_of("B"), 0); pump(idle)
    assert abs(win.tt_mag.value() - 9.0) < 0.06, \
        "double-click after sorting must still select the DISPLAYED star"
    print("  [ok] header sorting: numeric order, '—' last, highlight and "
          "double-click still hit the right star after re-ordering")

    # a sensing-band switch invalidates a ranking computed for the old band
    win.tt_sensor.setCurrentText("TRICK (K)"); app.processEvents()
    assert not win._gs_ranking, "sensor switch must invalidate the ranking"
    assert win._gs_rank_dialog is None, "the stale ranking dialog must close"
    print("  [ok] sensor switch invalidates a stale ranking (dialog closes)")
    win.tt_sensor.setCurrentText("STRAP (R)"); app.processEvents()

    # --- optical-reddening safety in the GUI ranking -------------------------
    # a reddened IR-only star (J-K=3.0, like a dusty-GC 2MASS source) close to
    # centre: naive R from J looks usable, but STRAP(R) must flag+exclude it
    # (dust -> optically invisible), while TRICK(K) keeps it.
    rd = cen.spherical_offsets_by(3 * u.arcsec, 0 * u.arcsec)
    red_star = {"id": "RED", "ra": float(rd.ra.deg), "dec": float(rd.dec.deg),
                "mags": {"J": 12.4, "H": 11.7, "K": 9.4}}
    good_off = cen.spherical_offsets_by(-4 * u.arcsec, 0 * u.arcsec)
    good_star = {"id": "GOOD", "ra": float(good_off.ra.deg),
                 "dec": float(good_off.dec.deg), "mags": {"R": 12.0}}
    win._on_catalog_loaded("Gaia DR2", [good_star, red_star], ""); pump(idle)
    win._rank_guide_stars(); pump(idle); app.processEvents()
    rby = {e["id"]: e for e in win._gs_ranking}
    assert rby["RED"]["rank"] is None and "IR-red" in rby["RED"]["excluded_reason"], \
        rby["RED"]
    assert rby["GOOD"]["rank"] == 1
    red_row = row_of("RED")
    status_col = win._gs_rank_table.columnCount() - 1
    stxt = win._gs_rank_table.item(red_row, status_col).text()
    assert "IR-red" in stxt, stxt
    print(f"  [ok] STRAP(R): reddened star flagged+excluded in the table ({stxt})")
    # left-click inspect warns too (the standing 'verify vs imagery' string)
    rxy = next(s for s in win._catalog_stars_xy() if s["id"] == "RED")
    win._inspect_catalog_star(rxy["x"], rxy["y"]); pump(idle)
    assert "IR-red" in win.fm_catalog_status.text(), win.fm_catalog_status.text()
    print("  [ok] left-click inspect warns on the reddened star")
    # TRICK(K): the same star is usable, unflagged (direct K photometry)
    win.tt_sensor.setCurrentText("TRICK (K)"); app.processEvents()
    win._rank_guide_stars(); pump(idle); app.processEvents()
    rbyk = {e["id"]: e for e in win._gs_ranking}
    assert rbyk["RED"]["rank"] is not None and rbyk["RED"]["reddening_note"] is None, \
        rbyk["RED"]
    print("  [ok] TRICK(K): the same reddened star is usable (no warning)")
    win.tt_sensor.setCurrentText("STRAP (R)"); app.processEvents()

    # clear, and the selected catalogue round-trips through config
    win._rank_guide_stars(); pump(idle)          # repopulate, then verify clear invalidates
    assert win._gs_ranking
    win._clear_catalog(); pump(idle)
    assert len(win._catalog_stars) == 0
    assert not win.fm_catalog_rank.isEnabled(), "Rank must disable with no stars"
    assert not win._gs_ranking, "clearing the catalogue must invalidate the ranking"
    win.fm_catalog.setCurrentText("2MASS")
    cfg = win._collect_config(); assert cfg["fm_catalog"] == "2MASS"
    win.fm_catalog.setCurrentText("Gaia DR2")
    win._loading = True; win._apply_config(cfg); win._loading = False
    assert win.fm_catalog.currentText() == "2MASS"
    print("  [ok] clear (+ ranking invalidation) + config round-trip")


def main():
    engine_catalog()
    gui_overlay()
    print("  [ok] guide-star catalogue overlay + click-to-select")


if __name__ == "__main__":
    main()
