#!/usr/bin/env python3
"""Field-map display/interaction follow-ups (post gui_phase20/21):

  * dropped science targets report their own RA/Dec in the Targets list, not
    just an offset + predicted value.
  * catalogue guide-star markers are unfilled and red-outlined (a filled dot
    hides the real star underneath a sky backdrop, and red reads over a
    grayscale one); the too-faint/unknown ones stay dimmer, same hue.
  * the field map supports mouse-wheel zoom.
  * the field map supports rotating the whole field's position angle.
  * a loaded FITS (survey backdrop or inscribed frame) can be flipped in X
    and/or Y, for cases where the file's own WCS parity is untrustworthy.

Run headless."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
import astropy.units as u
from astropy.coordinates import SkyCoord
np = engine.np
DATA = os.path.join(HERE, "data")


def pump(cond, timeout=90):
    app = QtWidgets.QApplication.instance(); t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents(); QtCore.QThread.msleep(10)


def make_window():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1500, 950); win.show(); app.processEvents()
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(f"{DATA}/20260525_dimm.dat")
    win.mass_edit.setText(f"{DATA}/20260525_mass.dat")
    win.masspro_edit.setText(f"{DATA}/20260525_masspro.dat")
    win.tel_k1.setChecked(True)
    win.ra_edit.setText("10h00m00s"); win.dec_edit.setText("+20d00m00s")
    win._validate(); win.on_run()
    pump(lambda: win.res is not None)
    return app, win


def idle(win):
    return (not win._fm_debounce.isActive() and not win._fm_settle.isActive())


def dropped_target_radec():
    app, win = make_window()
    win.plot_tabs.setCurrentIndex(1); app.processEvents(); pump(lambda: idle(win))

    fc = win._field_center_deg(); assert fc is not None
    cen = SkyCoord(fc[0] * u.deg, fc[1] * u.deg)
    win._fm_add_target(4.0, 3.0)               # East 4, North 3 of centre
    pump(lambda: idle(win))
    assert len(win._fm_markers) == 1
    txt = win.fm_target_list.item(0).text()
    # coordinates must actually be IN the list text, and be the right ones
    want = cen.spherical_offsets_by(-4.0 * u.arcsec, 3.0 * u.arcsec)
    want_ra = want.ra.to_string(unit="hourangle", sep="hms", precision=1)
    assert want_ra in txt, (txt, want_ra)
    # and the direct helper resolves the same way _draw_fm_targets would use
    c = win._fm_target_radec(4.0, 3.0)
    assert c.separation(want).arcsec < 1e-6
    print(f"  [ok] dropped target reports its own RA/Dec: {txt!r}")
    win.close()


def catalog_markers_unfilled():
    app, win = make_window()
    win.plot_tabs.setCurrentIndex(1); app.processEvents(); pump(lambda: idle(win))

    fc = win._field_center_deg()
    cen = SkyCoord(fc[0] * u.deg, fc[1] * u.deg)
    bright = cen.spherical_offsets_by(5 * u.arcsec, 3 * u.arcsec)
    faint = cen.spherical_offsets_by(-8 * u.arcsec, 6 * u.arcsec)
    stars = [
        {"id": "A", "ra": float(bright.ra.deg), "dec": float(bright.dec.deg),
         "mags": {"G": 11.0, "BP": 11.3, "RP": 10.6}},
        {"id": "F", "ra": float(faint.ra.deg), "dec": float(faint.dec.deg),
         "mags": {"G": 19.5, "BP": 20.0, "RP": 18.8}},
    ]
    win.tt_sensor.setCurrentText("STRAP (R)")
    win._on_catalog_loaded("Gaia DR2", stars, ""); pump(lambda: idle(win))
    ax = next(a for a in win._fm_holder["canvas"].figure.axes if a.images)
    sc = next(c for c in ax.collections if len(c.get_offsets()) == 2)
    fcs = sc.get_facecolors()
    # every catalogue star marker must be unfilled (alpha 0 face), bright or
    # faint alike -- only the size/edge conveys brightness/usability now
    assert all(fc[3] == 0.0 for fc in fcs), \
        f"catalogue star markers must be unfilled, got face alphas {[fc[3] for fc in fcs]}"
    ecs = sc.get_edgecolors()
    # both are the same red hue (reads over a grayscale backdrop); the
    # too-faint star (F, index 1) is just dimmer (lower alpha) than usable (A)
    assert tuple(ecs[0][:3]) == tuple(ecs[1][:3]), \
        "usable and too-faint markers must share the same red hue"
    assert ecs[0][3] > ecs[1][3], \
        "too-faint star F must be dimmer (lower alpha) than usable star A"
    print("  [ok] catalogue star markers unfilled, red-outlined; "
          "faint/unusable ones stay dimmer")
    win.close()


class _FakeScroll:
    """Just enough of a matplotlib MouseEvent for _on_fm_scroll -- driven
    directly (like _on_fm_canvas_click's tests) rather than through real Qt
    wheel-event dispatch."""
    def __init__(self, inaxes, xdata, ydata, button):
        self.inaxes = inaxes; self.xdata = xdata; self.ydata = ydata
        self.button = button


def scroll_zoom():
    app, win = make_window()
    win.plot_tabs.setCurrentIndex(1); app.processEvents(); pump(lambda: idle(win))
    ax = next(a for a in win._fm_holder["canvas"].figure.axes if a.images)
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    half0 = (x1 - x0) / 2.0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    # scroll "up" (zoom in) centred on the view centre -> view shrinks,
    # centre stays put
    win._on_fm_scroll(_FakeScroll(ax, cx, cy, "up"))
    nx0, nx1 = ax.get_xlim(); ny0, ny1 = ax.get_ylim()
    half1 = (nx1 - nx0) / 2.0
    assert half1 < half0, (half0, half1)
    assert abs((nx0 + nx1) / 2.0 - cx) < 1e-6
    assert abs((ny0 + ny1) / 2.0 - cy) < 1e-6

    # scroll "down" (zoom out) undoes it back to (about) the original extent
    win._on_fm_scroll(_FakeScroll(ax, cx, cy, "down"))
    x0b, x1b = ax.get_xlim()
    assert abs((x1b - x0b) / 2.0 - half0) < 1e-6

    # off-centre scroll: the point under the cursor stays fixed, not the
    # view centre
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    px, py = x0 + (x1 - x0) * 0.25, y0 + (y1 - y0) * 0.25
    win._on_fm_scroll(_FakeScroll(ax, px, py, "up"))
    nx0, nx1 = ax.get_xlim(); ny0, ny1 = ax.get_ylim()
    assert abs((px - nx0) / (nx1 - nx0) - 0.25) < 1e-6, "cursor point must stay fixed"

    # clamps: many "up" scrolls must not shrink past the minimum half-extent
    for _ in range(200):
        win._on_fm_scroll(_FakeScroll(ax, px, py, "up"))
    x0c, x1c = ax.get_xlim()
    assert (x1c - x0c) / 2.0 >= win._FM_ZOOM_MIN_HALF - 1e-6
    print("  [ok] field map: mouse-wheel zoom in/out about the cursor, clamped")
    win.close()


def field_pa_rotation():
    app, win = make_window()
    win.plot_tabs.setCurrentIndex(1); app.processEvents(); pump(lambda: idle(win))
    ax = next(a for a in win._fm_holder["canvas"].figure.axes if a.images)

    # --- transform math: rotate_deg's CCW-positive sense must match this
    # plot's North->East convention (East=-x, North=+y) for free ---
    trans = win._fm_rotation_transform(ax)
    # PA=0 default: identity (composed only with the axes' own transData)
    assert np.allclose(trans.transform((3.0, 4.0)),
                       ax.transData.transform((3.0, 4.0)))
    win.fm_pa.setValue(90.0)
    pump(lambda: idle(win)); app.processEvents()
    trans = win._fm_rotation_transform(ax)
    # North (0, 10) rotated 90 deg (North->East) must land where East (which
    # is -x, so raw data point (-10, 0)) used to be, in RAW (unrotated) data
    # coordinates -- i.e. trans(0,10) == transData(-10,0)
    got = trans.transform((0.0, 10.0))
    want = ax.transData.transform((-10.0, 0.0))
    assert np.allclose(got, want, atol=0.5), (got, want)
    print("  [ok] field PA rotation: rotate_deg matches this plot's "
          "North->East convention")

    # --- an actually-drawn marker uses the SAME transform, i.e. the field
    # really did rotate on screen, not just some unused helper ---
    win.tt_sensor.setCurrentText("STRAP (R)")
    win.tt_offset.setValue(10.0)               # total mode, PA=0 -> due North
    win._fieldmap_dirty = True
    win._render_field_map_if_visible()
    pump(lambda: not win._fm_debounce.isActive() and not win._fm_settle.isActive())
    for _ in range(8):
        app.processEvents(); QtCore.QThread.msleep(15)
    ax = next(a for a in win._fm_holder["canvas"].figure.axes if a.images)
    star_line = next(ln for ln in ax.get_lines()
                     if "TT star" in (ln.get_label() or ""))
    xd, yd = star_line.get_xdata()[0], star_line.get_ydata()[0]
    drawn_px = star_line.get_transform().transform((xd, yd))
    want_px = ax.transData.transform((-yd, xd))    # 90 deg CCW of (xd,yd)
    assert np.allclose(drawn_px, want_px, atol=1.0), (drawn_px, want_px)
    print("  [ok] a real drawn marker (TT star) is rotated on screen, not just the helper")
    win.fm_pa.setValue(0.0)

    # --- the live/scrub scaffold must invalidate (rebuild) when PA changes,
    # or a rotated view would silently snap back during scrubbing ---
    win._fm_live = None
    win.fm_for.setChecked(False)
    win._fieldmap_dirty = True; win._fm_settle.start()   # force the live path
    win._render_field_map()
    key0 = win._fm_live["key"]
    win.fm_pa.setValue(45.0)
    win._fieldmap_dirty = True; win._fm_settle.start()
    win._render_field_map()
    assert win._fm_live["key"] != key0, "PA change must invalidate the live scaffold"
    win.fm_pa.setValue(0.0)
    print("  [ok] the interactive (scrubbing) field-map scaffold rebuilds on a PA change")
    win.close()


def frame_flip():
    """A loaded frame with flip X/Y toggled must actually mirror the
    rendered array (not just flip some unused cached copy) -- uses the
    bundled public-KOA NIRC2 example frame (M15 cluster field, narrow
    camera), whose KOA CD-matrix-as-string-cards export quirk the
    imaging.py _coerce_wcs_numeric_strings fix makes loadable (also
    pinned synthetically in tests/test_correctness_physics.py)."""
    app, win = make_window()
    win.plot_tabs.setCurrentIndex(1); app.processEvents(); pump(lambda: idle(win))
    path = os.path.join(os.path.dirname(HERE), "examples", "N2.20210821_42637.fits")
    if not os.path.exists(path):
        print("  [skip] frame flip: NIRC2 example FITS not present locally")
        win.close(); return
    win.fm_sky.setCurrentText("off")            # isolate to the frame layer
    win._load_local_sky(path)
    pump(lambda: idle(win))
    assert win._sky_fg_img is not None, "frame failed to load"

    def frame_array():
        win._fieldmap_dirty = True
        win._render_field_map_if_visible()
        pump(lambda: not win._fm_debounce.isActive() and not win._fm_settle.isActive())
        for _ in range(6):
            app.processEvents(); QtCore.QThread.msleep(15)
        ax = next(a for a in win._fm_holder["canvas"].figure.axes if a.images)
        assert len(ax.images) == 1, "expected only the frame layer (no backdrop)"
        return np.asarray(ax.images[0].get_array())

    def same(a, b):
        return bool(np.all((np.isnan(a) & np.isnan(b)) | (a == b)))

    base = frame_array()
    win.fm_fg_flip_x.setChecked(True)
    flip_x = frame_array()
    assert same(flip_x, np.fliplr(base)), "flip X must mirror columns"
    win.fm_fg_flip_x.setChecked(False)
    win.fm_fg_flip_y.setChecked(True)
    flip_y = frame_array()
    assert same(flip_y, np.flipud(base)), "flip Y must mirror rows"
    win.fm_fg_flip_y.setChecked(False)
    back = frame_array()
    assert same(back, base), "unchecking both flips must restore the original orientation"
    print("  [ok] frame flip X/Y mirrors the rendered array, reversibly")
    win.close()


def image_pa_override():
    """The manual image-only PA override must rotate ONLY the loaded image
    (its imshow transform), NOT the catalogue/markers (still on the Field-PA
    transform), and must flag itself on the map while non-zero."""
    import math
    app, win = make_window()
    win.plot_tabs.setCurrentIndex(1); app.processEvents(); pump(lambda: idle(win))
    path = os.path.join(os.path.dirname(HERE), "examples", "N2.20210821_42637.fits")
    if not os.path.exists(path):
        print("  [skip] image PA: NIRC2 example FITS not present locally")
        win.close(); return
    win.fm_pa.setValue(0.0)
    win.fm_sky.setCurrentText("off")
    win._load_local_sky(path)
    pump(lambda: idle(win))
    assert win._sky_fg_img is not None, "frame failed to load"

    def render_ax():
        win._fieldmap_dirty = True
        win._render_field_map_if_visible()
        pump(lambda: not win._fm_debounce.isActive() and not win._fm_settle.isActive())
        for _ in range(8):
            app.processEvents(); QtCore.QThread.msleep(15)
        return next(a for a in win._fm_holder["canvas"].figure.axes if a.images)

    # img PA 0: the image's imshow transform matches the Field-PA transform
    # (imagery lined up with everything else), and there's no warning badge
    win.fm_img_pa.setValue(0.0)
    ax = render_ax()
    field_t = win._fm_rotation_transform(ax)
    im = ax.images[0]
    assert np.allclose(im.get_transform().transform((3.0, 4.0)),
                       field_t.transform((3.0, 4.0)), atol=1.0)
    assert not any("ROTATED" in (t.get_text() or "") for t in ax.texts), \
        "no image-PA warning should show at 0°"

    # img PA 30: the image rotates 30° (N->E, about its centre) relative to
    # the field, so its imshow transform now differs; a non-image marker (the
    # field-centre star, on the Field-PA transform) is NOT rotated by it
    win.fm_img_pa.setValue(30.0)
    ax = render_ax()
    field_t = win._fm_rotation_transform(ax)
    im = ax.images[0]
    th = math.radians(30.0)
    # image point (0,10) -> 30° CCW about (0,0): (-10 sinθ, 10 cosθ) in the
    # field frame (matches _fm_image_transform's rotate-then-field-PA)
    got = im.get_transform().transform((0.0, 10.0))
    want = field_t.transform((-10.0 * math.sin(th), 10.0 * math.cos(th)))
    assert np.allclose(got, want, atol=1.5), (got, want)
    assert not np.allclose(im.get_transform().transform((5.0, 5.0)),
                           field_t.transform((5.0, 5.0)), atol=1.0), \
        "image transform must differ from the field transform at 30°"
    centre = next(ln for ln in ax.get_lines()
                  if "field centre" in (ln.get_label() or ""))
    assert np.allclose(centre.get_transform().transform((5.0, 5.0)),
                       field_t.transform((5.0, 5.0)), atol=1.0), \
        "catalogue/markers must NOT be rotated by the image-PA override"
    assert any("ROTATED" in (t.get_text() or "") for t in ax.texts), \
        "a non-zero image PA must be flagged on the map"
    print("  [ok] image PA override rotates only the loaded image (not the "
          "catalogue), and flags itself on the map")

    # config round-trips the image PA
    win.fm_img_pa.setValue(25.0)
    cfg = win._collect_config()
    assert cfg["fm_img_pa"] == 25.0
    win.fm_img_pa.setValue(0.0)
    win._apply_config(cfg)
    assert win.fm_img_pa.value() == 25.0
    win.fm_img_pa.setValue(0.0)
    print("  [ok] image PA persists through save/load config")
    win.close()


def main():
    dropped_target_radec()
    catalog_markers_unfilled()
    scroll_zoom()
    field_pa_rotation()
    frame_flip()
    image_pa_override()
    print("  [ok] field-map display/interaction follow-ups")


if __name__ == "__main__":
    main()
