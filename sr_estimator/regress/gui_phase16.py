#!/usr/bin/env python3
"""Field map: 2D anisoplanatism map (SR/FWHM) vs field position. Engine contract
(on-axis reproduces the science value; best at the reference; K1/K2 field size;
SW laser) + the GUI tab. Run headless."""
import os, sys, time, math
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
np = engine.np
DATA = os.path.join(HERE, "data")


def pump(cond, timeout=120):
    app = QtWidgets.QApplication.instance(); t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents(); QtCore.QThread.msleep(20)


def fm_idle(win):
    """Both field-map timers quiet: the throttle (coarse live frames) AND the
    settle timer (trailing full-resolution redraw). Waiting on both means the
    map on screen is the settled full-res render, not a mid-scrub coarse one."""
    return not win._fm_debounce.isActive() and not win._fm_settle.isActive()


def engine_contract():
    from astropy.utils.iers import conf; conf.auto_max_age = None
    a = engine.build_parser().parse_args([
        "--dimm", f"{DATA}/20260525_dimm.dat", "--mass", f"{DATA}/20260525_mass.dat",
        "--masspro", f"{DATA}/20260525_masspro.dat", "--telescope", "K1",
        "--out", "/tmp/f.png", "--force"])
    p = engine.prepare_night(a); r = engine.compute_timeline(a, p)
    snap = engine.field_snapshot(a, p, r, "night")
    assert snap is not None and snap["theta0_los"] > 0
    lgs = 7.0
    laser = (+lgs / math.sqrt(2), -lgs / math.sqrt(2))     # radial SW
    tt = (0.0, a.tt_offset)                                # assume North
    ngs = (0.0, 0.0)

    # field size is telescope-specific
    ext, Z, meta = engine.field_map_grid(a, p, snap, "ltao", "strehl",
                                         ngs, tt, laser, n_grid=41)
    assert meta["fov"] == 20.0 and ext == [-10, 10, -10, 10], "K1 field 20x20"

    # on-axis (target = grid center) reproduces the science-direction value
    for mode in ("single", "ltao"):
        _, Z, _ = engine.field_map_grid(a, p, snap, mode, "strehl", ngs, tt,
                                        laser, n_grid=41)
        sci = engine.lgs_strehl(
            snap["eps_tot_los"], snap["eps_fa_los"], "K1", mode, p.lam_nm,
            cn2_bins=(snap["cn2_bins"] if mode == "ltao" else None),
            tt_mag=a.tt_mag, tt_offset=a.tt_offset, lgs_offset=lgs,
            legacy=a.legacy_budget, bw_factor=p._ltao_bw_fac,
            v_ground=a.wind_ground, v_free=a.wind_free)
        assert abs(Z[20, 20] - sci) < 1e-9, f"{mode} center != science"
    print("  [ok] engine: center reproduces science-direction Strehl (single+LTAO)")

    # NGS map peaks AT the guide star, degrades away
    a.ngs_offset = 5.0
    _, Z, _ = engine.field_map_grid(a, p, snap, "ngs", "strehl",
                                    (0.0, 5.0), tt, laser, n_grid=41)
    imax = np.unravel_index(np.nanargmax(Z), Z.shape)
    ax = np.linspace(-10, 10, 41)
    assert abs(ax[imax[1]]) < 0.6 and abs(ax[imax[0]] - 5.0) < 0.6, \
        "NGS Strehl should peak at the guide-star position"
    assert Z[imax] > Z[0, 0], "far corner must be worse than the star"
    print(f"  [ok] engine: NGS Strehl peaks at the guide star (5\" N), "
          f"corner {Z[0,0]:.3f} < peak {Z[imax]:.3f}")

    # NGS budget what-if (Marechal variance swap): a NEGATIVE delta-var (e.g.
    # a denser DM lowering FITTING_ERR) must RAISE the whole NGS map, positive
    # must lower it, 0 must be untouched; LGS modes ignore it (their budget
    # applies directly). DM-upgrade example: K1 141 -> 60 nm swaps out
    # 141^2 - 60^2 = 16281 nm^2.
    dv = 60.0**2 - 141.0**2
    _, Zb, _ = engine.field_map_grid(a, p, snap, "ngs", "strehl",
                                     (0.0, 5.0), tt, laser, n_grid=21)
    _, Zu, _ = engine.field_map_grid(a, p, snap, "ngs", "strehl",
                                     (0.0, 5.0), tt, laser, n_grid=21,
                                     ngs_delta_var=dv)
    _, Zw, _ = engine.field_map_grid(a, p, snap, "ngs", "strehl",
                                     (0.0, 5.0), tt, laser, n_grid=21,
                                     ngs_delta_var=-dv)
    _, Z0, _ = engine.field_map_grid(a, p, snap, "ngs", "strehl",
                                     (0.0, 5.0), tt, laser, n_grid=21,
                                     ngs_delta_var=0.0)
    assert np.all(Zu >= Zb) and np.nanmax(Zu - Zb) > 0.01, "upgrade must raise"
    assert np.all(Zw <= Zb) and np.nanmax(Zb - Zw) > 0.01, "added error lowers"
    assert np.array_equal(Z0, Zb), "delta_var=0 must be the pure NGS model"
    # exactness at one pixel: swap in nm-space matches the closed form
    lam = p.lam_nm
    sig = (lam / (2 * np.pi)) * np.sqrt(-np.log(Zb[10, 10]))
    want = np.exp(-(2 * np.pi / lam) ** 2 * (sig**2 + dv))
    assert abs(Zu[10, 10] - want) < 1e-12, (Zu[10, 10], want)
    _, Zl, _ = engine.field_map_grid(a, p, snap, "single", "strehl",
                                     ngs, tt, laser, n_grid=21)
    _, Zl2, _ = engine.field_map_grid(a, p, snap, "single", "strehl",
                                      ngs, tt, laser, n_grid=21,
                                      ngs_delta_var=dv)
    assert np.array_equal(Zl, Zl2), "LGS map must ignore ngs_delta_var"
    print(f"  [ok] engine: NGS map variance swap (DM upgrade 141->60 nm "
          f"raises peak {np.nanmax(Zb):.3f} -> {np.nanmax(Zu):.3f}); "
          f"LGS unaffected")

    # K2 field is 10x10
    a2 = engine.build_parser().parse_args([
        "--dimm", f"{DATA}/20260525_dimm.dat", "--mass", f"{DATA}/20260525_mass.dat",
        "--masspro", f"{DATA}/20260525_masspro.dat", "--telescope", "K2",
        "--out", "/tmp/f2.png", "--force"])
    p2 = engine.prepare_night(a2); r2 = engine.compute_timeline(a2, p2)
    s2 = engine.field_snapshot(a2, p2, r2, "night")
    _, _, m2 = engine.field_map_grid(a2, p2, s2, "single", "fwhm",
                                     ngs, tt, (0, 0), n_grid=21)
    assert m2["fov"] == 10.0, "K2 field 10x10"
    print("  [ok] engine: field size 20\" (K1) / 10\" (K2); FWHM grid works")


def gui_tab():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1550, 950); win.show(); app.processEvents()
    # 3 plot tabs incl. Field map
    assert win.plot_tabs.count() == 4 and win.plot_tabs.tabText(1) == "Field map"

    win.mode_local.setChecked(True)
    win.dimm_edit.setText(f"{DATA}/20260525_dimm.dat")
    win.mass_edit.setText(f"{DATA}/20260525_mass.dat")
    win.masspro_edit.setText(f"{DATA}/20260525_masspro.dat")
    win.tel_k1.setChecked(True)

    # laser default is the K1 pointing-offset campaign direction
    # (2026-08-07): PA 254.8 deg = 4.8" W, 1.3" S of the pointing origin,
    # at the DEF_LGS_OFFSET["K1"] = 4.97" radius. Still West+South, as the
    # old placeholder PA 225 (radial SW) was -- but now measured, not assumed.
    assert win.laser_pa.value() == engine.DEF_LASER_PA_DEG
    lx, ly = win._laser_xy()
    assert lx > 0 and ly < 0, "the K1 laser sits West+South of the target"
    assert abs(lx - 4.8) < 0.05 and abs(ly + 1.3) < 0.05, \
        f"laser should be 4.8\" W, 1.3\" S; got {lx:.2f}, {ly:.2f}"
    # NGS star 5" N via ΔRA/ΔDec; TT star 8" NE
    win.ngs_offset.mode.setCurrentIndex(1)
    win.ngs_offset.dra.setValue(0.0); win.ngs_offset.ddec.setValue(5.0)
    win._validate(); win.on_run()
    pump(lambda: win.res is not None)

    # offset_xy: ΔDec 5 N -> (0, +5, known)
    x, y, known = win.ngs_offset.offset_xy()
    assert abs(x) < 1e-9 and abs(y - 5.0) < 1e-9 and known
    # total-only mode -> assume North (PA 0), dir not known
    win.tt_offset.setValue(9.0)
    x, y, known = win.tt_offset.offset_xy()
    assert abs(x) < 1e-9 and abs(y - 9.0) < 1e-9 and not known
    # ...and the PA is adjustable: PA=90 -> due East (x=-9), magnitude unchanged
    win.tt_offset.pa.setValue(90.0)
    x, y, known = win.tt_offset.offset_xy()
    assert abs(x + 9.0) < 1e-6 and abs(y) < 1e-6, "total-mode PA must steer marker"
    assert abs(win.tt_offset.value() - 9.0) < 1e-9, "PA must not change magnitude"
    win.tt_offset.pa.setValue(0.0)
    print("  [ok] offset_xy: ΔRA/ΔDec 2D known; total offset steered by PA")

    # render the field map (lazy) and confirm imshow + colorbar
    win.fm_mode.setCurrentText("LTAO")
    win.plot_tabs.setCurrentIndex(1); app.processEvents()
    for _ in range(6):
        app.processEvents(); QtCore.QThread.msleep(30)
    fig = win._fm_holder["canvas"].figure
    assert any(ax.images for ax in fig.axes), "field map should draw an image"
    assert len(fig.axes) >= 2, "expected an image axis + a colorbar"
    # LTAO -> 4-LGS asterism ring + beacons; science target is blue
    imgax = next(ax for ax in fig.axes if ax.images)
    labels = [l.get_label() for l in imgax.get_lines()] + \
             [c.get_label() for c in imgax.collections]
    assert any("asterism" in s for s in labels), "LTAO should draw the asterism"
    from matplotlib.patches import Circle
    assert any(isinstance(p, Circle) for p in imgax.patches), "asterism ring"
    tgt = next(l for l in imgax.get_lines()
               if l.get_label().startswith("field centre"))
    assert tgt.get_markerfacecolor() == gui.FM_C_TARGET, "centre must be blue"
    # the field-centre metric value is reported on the marker/legend
    assert "—" in tgt.get_label(), "centre value must be reported"
    assert any("centre:" in t.get_text() for t in imgax.texts), "centre annotation"
    print(f"  [ok] GUI: LTAO asterism; field centre is blue; value reported "
          f"({tgt.get_label().split('—')[1].strip()})")

    # metric toggle -> all three FWHM conventions available on the field map
    for mname in ("FWHM (half-max)", "FWHM (Gaussian fit)",
                  "FWHM (Gaussian fit +background)"):
        win.fm_metric.setCurrentText(mname); app.processEvents()
        pump(lambda: fm_idle(win)); app.processEvents()
        lbl = [a.get_ylabel() for a in win._fm_holder["canvas"].figure.axes]
        assert any("FWHM" in s for s in lbl), f"{mname} colorbar"
    print("  [ok] GUI: field-map metric offers Strehl / FWHM half-max / gaussfit / gaussfit+bg")

    # OSIRIS spectrograph FOV = (lenslet x scale) x (64 x scale) -- rectangular
    win.fm_metric.setCurrentText("Strehl")
    win.fm_osiris_mode.setCurrentText("spectrograph")
    win.fm_osiris_scale.setCurrentText("0.05")
    win.fm_osiris_lenslet.setCurrentText("32")
    app.processEvents()
    pump(lambda: fm_idle(win)); app.processEvents()
    assert win._current_fov() == (1.6, 3.2), win._current_fov()
    imgax = next(ax for ax in win._fm_holder["canvas"].figure.axes if ax.images)
    x0, x1 = imgax.images[0].get_extent()[:2]
    y0, y1 = imgax.images[0].get_extent()[2:]
    assert abs((x1 - x0) - 1.6) < 1e-6 and abs((y1 - y0) - 3.2) < 1e-6, \
        "spectrograph FOV must map a 1.6x3.2 rectangle"
    print("  [ok] GUI: OSIRIS spectrograph FOV rectangular (0.05\"x32 -> 1.6x3.2\")")

    # imager is a fixed 20x20 square
    win.fm_osiris_mode.setCurrentText("imager (20×20″)"); app.processEvents()
    assert win._current_fov() == (20.0, 20.0)
    print("  [ok] GUI: OSIRIS imager fixed 20x20; spectrograph controls hidden")

    # field of regard: 60" patrol circle drawn, view zooms out past the FOV
    from matplotlib.patches import Circle
    win.fm_for.setChecked(True); app.processEvents()
    pump(lambda: fm_idle(win)); app.processEvents()
    fmax = next(ax for ax in win._fm_holder["canvas"].figure.axes if ax.images)
    assert any(isinstance(p, Circle)
               and abs(p.get_radius() - gui.FIELD_OF_REGARD_RADIUS_ARCSEC) < 1e-6
               for p in fmax.patches), "60\" field-of-regard circle missing"
    assert fmax.get_xlim()[1] >= gui.FIELD_OF_REGARD_RADIUS_ARCSEC, \
        "view must zoom out to the field of regard"
    # the SR heatmap now covers the WHOLE field of regard (not just the FOV)
    ext = fmax.images[0].get_extent()
    R = gui.FIELD_OF_REGARD_RADIUS_ARCSEC
    assert abs(ext[1] - R) < 1e-6 and abs(ext[3] - R) < 1e-6, \
        f"heatmap must span the field of regard, got extent {ext}"
    # a TT star out at 30" now shows at its true position (inside the FoR),
    # no longer clamped to the tiny FOV edge
    win.tt_offset.mode.setCurrentIndex(0); win.tt_offset.setValue(30.0)
    app.processEvents()
    pump(lambda: fm_idle(win)); app.processEvents()
    fmax = next(ax for ax in win._fm_holder["canvas"].figure.axes if ax.images)
    star = [ln for ln in fmax.get_lines()
            if ln.get_label().startswith("TT star")
            and len(ln.get_ydata())]
    assert any(abs(ln.get_ydata()[0] - 30.0) < 0.5 for ln in star), \
        "TT star at 30\" should plot at its true position inside the FoR"
    print("  [ok] GUI: field of regard draws the 60\" patrol circle; star placed true")

    # ---- NGS budget what-if reaches the field map -----------------------------
    # (a) with masspro loaded, theta0 is measured -> the Assumed-theta0 fallback
    # spin greys out (a dead knob must look dead)
    assert not win.assumed_theta0.isEnabled(), \
        "assumed theta0 must grey out on a MASS night"
    # (b) a common-path slider projects onto the NGS map (variance swap) and the
    # map is flagged; the map values actually move
    win.fm_mode.setCurrentText("NGS")
    pump(lambda: fm_idle(win)); app.processEvents()
    fmax = next(ax for ax in win._fm_holder["canvas"].figure.axes if ax.images)
    z_before = np.asarray(fmax.images[0].get_array(), float).copy()
    win.wfe_rows["FITTING_ERR_K1"]["spin"].setValue(60.0)   # denser-DM what-if
    # wait for the what-if to reach the budget AND for the field-map timers to
    # settle, so z_after is the trailing full-res render (same grid as z_before,
    # not a mid-scrub coarse frame)
    pump(lambda: win.last_offsets and fm_idle(win))
    fmax = next(ax for ax in win._fm_holder["canvas"].figure.axes if ax.images)
    z_after = np.asarray(fmax.images[0].get_array(), float)
    assert np.nanmax(np.abs(z_after - z_before)) > 0.005, \
        "NGS field map must respond to a common-path budget what-if"
    notes = [t.get_text() for t in fmax.texts if "MODIFIED" in t.get_text()]
    assert notes and "projected NGS" in notes[0], notes
    win.wfe_rows["FITTING_ERR_K1"]["spin"].setValue(
        win.wfe_rows["FITTING_ERR_K1"]["default"])
    for _ in range(30):
        app.processEvents(); QtCore.QThread.msleep(30)
        if not win.last_offsets:
            break
    win.fm_mode.setCurrentText("LTAO"); app.processEvents()
    print("  [ok] GUI: NGS field map takes the budget what-if (flagged "
          "'projected NGS'); assumed-theta0 greyed on MASS night")

    # ---- sky-image overlay (offline: URL builder + FITS resample + display) --
    u = gui._hips2fits_url("CDS/P/DSS2/red", 250.4235, 36.4613, 0.0333)
    assert "hips=CDS/P/DSS2/red" in u and "format=jpg" in u and "ra=250.42" in u
    # synthetic FITS (N-up/E-left TAN) with a feature 20" NORTH of centre: the
    # resampled overlay must place it in the upper rows (North=top), centre col
    from astropy.io import fits as _fits
    ra0, dec0, sc = 250.0, 36.0, 1.0 / 3600
    dat = np.zeros((101, 101)); dat[50 + 20, 50] = 100.0
    hdr = _fits.Header()
    hdr["CTYPE1"] = "RA---TAN"; hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = 51; hdr["CRPIX2"] = 51
    hdr["CRVAL1"] = ra0; hdr["CRVAL2"] = dec0
    hdr["CD1_1"] = -sc; hdr["CD1_2"] = 0; hdr["CD2_1"] = 0; hdr["CD2_2"] = sc
    fp = os.path.join(HERE, "synth_sky.fits")
    _fits.PrimaryHDU(dat, hdr).writeto(fp, overwrite=True)
    simg, note, ctr, _t, _nm, _half = gui.sky_image_from_fits(fp, n=101)
    r, c = np.unravel_index(np.nanargmax(simg), simg.shape)
    assert r < 45 and abs(c - 50) < 6, \
        f"20\" N feature at row {r} col {c} (want upper-centre / N-up E-left)"
    assert note == "file WCS", note
    # the image's OWN pointing defines the field centre (crval), regardless of
    # any typed target
    assert abs(ctr.ra.deg - ra0) < 1e-3 and abs(ctr.dec.deg - dec0) < 1e-3, \
        f"field centre {ctr} must be the image pointing ({ra0},{dec0})"
    os.remove(fp)

    # a PNG carrying its astrometry in a 'SKYWCS' text chunk places just like the
    # FITS (the shippable light-image path, e.g. the bundled GSAOI GC master).
    # Same 20"-N feature + same WCS header -> same upper-centre placement.
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    pdat = np.zeros((101, 101), np.uint8); pdat[50 + 20, 50] = 255
    _meta = PngInfo(); _meta.add_text("SKYWCS", hdr.tostring())
    _meta.add_text("OBJECT", "T")
    fpp = os.path.join(HERE, "synth_sky.png")
    Image.fromarray(pdat, "L").save(fpp, pnginfo=_meta)
    pimg, pnote, pctr, _pt, _pnm, _ph = gui.sky_image_from_png(fpp, n=101)
    pr, pc = np.unravel_index(np.nanargmax(pimg), pimg.shape)
    assert pr < 45 and abs(pc - 50) < 6, f"PNG 20\"N feature at ({pr},{pc})"
    assert abs(pctr.ra.deg - ra0) < 1e-3 and abs(pctr.dec.deg - dec0) < 1e-3, pctr
    # sky_image_from_file dispatches by extension (PNG here, FITS elsewhere)
    fimg, *_ = gui.sky_image_from_file(fpp, n=101)
    assert np.allclose(np.nan_to_num(fimg), np.nan_to_num(pimg)), "dispatch mismatch"
    # a PNG WITHOUT the SKYWCS chunk has no sky position -> clear ValueError
    Image.fromarray(pdat, "L").save(fpp)                     # no pnginfo
    try:
        gui.sky_image_from_png(fpp, n=101)
        raise AssertionError("a PNG with no SKYWCS must raise")
    except ValueError as e:
        assert "SKYWCS" in str(e), e
    os.remove(fpp)
    print("  [ok] sky overlay: PNG-with-embedded-WCS places like FITS "
          "(sky_image_from_file dispatch; no-WCS PNG rejected)")

    # a header with NO celestial WCS but Keck-style pointing + plate-scale + PA
    # (as OSIRIS frames have) must synthesize a WCS and still place the image,
    # centred on its own pointing
    h2 = _fits.Header()
    h2["NAXIS"] = 2; h2["NAXIS1"] = 200; h2["NAXIS2"] = 200
    h2["TARGRA"] = ra0; h2["TARGDEC"] = dec0
    h2["PSCALE"] = 0.5; h2["PA_IMAG"] = 0.0; h2["INSTR"] = "imag"
    h2["MJD-OBS"] = 61071.26518462                       # 2026-01-31 06:21 UTC
    d2 = np.zeros((200, 200)); d2[100 + 20, 100] = 100.0     # 10" N (0.5"/px)
    fp2 = os.path.join(HERE, "synth_nowcs.fits")
    _fits.PrimaryHDU(d2, h2).writeto(fp2, overwrite=True)
    s2, note2, ctr2, t2, _nm2, half2 = gui.sky_image_from_fits(fp2, n=101)
    assert "no file WCS" in note2 and np.any(np.isfinite(s2)), note2
    assert abs(ctr2.ra.deg - ra0) < 1e-3 and abs(ctr2.dec.deg - dec0) < 1e-3
    # obs time comes back in HST (= UTC − 10h): 06:21 UTC -> 20:21 HST prev day
    assert t2 is not None and t2.hour == 20 and t2.minute == 21, t2
    r2, c2 = np.unravel_index(np.nanargmax(s2), s2.shape)
    assert r2 < 50 and abs(c2 - 50) < 8, \
        f"synthesized-WCS feature at row {r2} col {c2} (want upper-centre)"
    os.remove(fp2)
    print("  [ok] sky overlay: local FITS defines the field (image pointing "
          "= centre; real & synthesized WCS)")

    # a multi-extension FITS (MEF, e.g. GSAOI's 4 detectors) must MOSAIC every
    # science extension via its own WCS -- not just load extension 1 -- with
    # pointing/target/time taken from the (data-less) primary header
    ra0, dec0 = 250.0, 36.0
    prim = _fits.Header()                       # metadata only, no pixels
    prim["OBJECT"] = "MEF-TGT"; prim["MJD-OBS"] = 61071.26518462  # 20:21 HST
    hdus = [_fits.PrimaryHDU(header=prim)]
    dx = 30.0 / 3600.0                          # two 30" tiles, E and W of centre
    for k, off in enumerate((+dx, -dx)):        # tile centres 30" E / 30" W
        d = np.zeros((60, 60)); d[30, 30] = 100.0 * (k + 1)   # bright centre pixel
        eh = _fits.Header()
        eh["CTYPE1"] = "RA---TAN"; eh["CTYPE2"] = "DEC--TAN"
        eh["CRPIX1"] = 30.5; eh["CRPIX2"] = 30.5
        eh["CRVAL1"] = ra0 + off / np.cos(np.radians(dec0)); eh["CRVAL2"] = dec0
        eh["CD1_1"] = -0.5 / 3600; eh["CD1_2"] = 0
        eh["CD2_1"] = 0; eh["CD2_2"] = 0.5 / 3600
        hdus.append(_fits.ImageHDU(d, eh))
    fpm = os.path.join(HERE, "synth_mef.fits")
    _fits.HDUList(hdus).writeto(fpm, overwrite=True)
    sm, notem, ctrm, tm, nmm, halfm = gui.sky_image_from_fits(fpm, n=161)
    assert "2-extension mosaic" in notem, notem       # BOTH tiles, not just ext 1
    assert nmm == "MEF-TGT" and tm is not None and tm.hour == 20, (nmm, tm)
    assert abs(ctrm.ra.deg - ra0) < 2e-3, f"centre between the tiles ({ctrm.ra.deg})"
    assert halfm > 25.0, f"extent spans both 30\" tiles ({halfm})"
    # both detectors landed: a bright pixel to the EAST (left) AND the WEST
    # (right) of centre -- extension 1 alone could only give one
    mid = sm.shape[1] // 2
    east = np.nanmax(sm[:, :mid - 10]); west = np.nanmax(sm[:, mid + 10:])
    assert np.isfinite(east) and np.isfinite(west) and east > 10 and west > 10, \
        f"both tiles must appear (E={east}, W={west}); MEF was not mosaicked"
    os.remove(fpm)
    print("  [ok] sky overlay: MEF mosaicked (all extensions + primary-header "
          "pointing), not just extension 1")

    # a local FITS can also be used as a BACKDROP (like DSS/2MASS) instead of
    # the inscribed role: same file, two roles selected by center/half. Backdrop
    # mode resamples onto the field grid centred on the SCIENCE FIELD, not the
    # image itself.
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    raw, decw = 250.0, 36.0
    wide = np.zeros((200, 200)); wide[100, 100] = 50.0     # bright at own centre
    wh = _fits.Header()
    wh["CTYPE1"] = "RA---TAN"; wh["CTYPE2"] = "DEC--TAN"
    wh["CRPIX1"] = 100.5; wh["CRPIX2"] = 100.5
    wh["CRVAL1"] = raw; wh["CRVAL2"] = decw
    wh["CD1_1"] = -0.5 / 3600; wh["CD1_2"] = 0
    wh["CD2_1"] = 0; wh["CD2_2"] = 0.5 / 3600
    fpw = os.path.join(HERE, "synth_wide.fits")
    _fits.PrimaryHDU(wide, wh).writeto(fpw, overwrite=True)
    # inscribe (default): centred on the image, native ~50" half-extent
    _a, _n, cin, _o, _nm, hin = gui.sky_image_from_fits(fpw, n=101)
    assert abs(cin.ra.deg - raw) < 1e-3 and abs(hin - 50.0) < 1.0, (cin, hin)
    # backdrop: centred on a target 20" off the image centre, at field-of-regard
    # half -- the grid follows the caller, the pixels follow the image's own WCS
    tgt = SkyCoord((raw - 20 / 3600 / np.cos(np.radians(decw))) * u.deg,
                   decw * u.deg)
    _a2, _n2, cbg, _o2, _nm2, hbg = gui.sky_image_from_fits(
        fpw, n=101, center=tgt, half=gui.FIELD_OF_REGARD_RADIUS_ARCSEC)
    assert abs(cbg.ra.deg - tgt.ra.deg) < 1e-6, "backdrop uses the given centre"
    assert abs(hbg - gui.FIELD_OF_REGARD_RADIUS_ARCSEC) < 1e-6, "backdrop half=FoR"
    # GUI: loading it as a local backdrop sets the combo + centres per the
    # precedence frame > target > image; independent of any inscribed frame
    win._clear_frame()
    win.ra_edit.setText("16h40m00s"); win.dec_edit.setText("+36d00m00s")  # far off
    win._load_bg_local(fpw)                       # no frame -> centre on target
    assert win.fm_sky.currentText() == gui.LOCAL_BACKDROP
    assert win._sky_bg_local_path == fpw and "centred on target" in win._sky_bg_note
    # turning the backdrop off drops the local-backdrop state (no fetch)
    win.fm_sky.setCurrentText("off")
    assert win._sky_bg_local_path is None and win._sky_bg_img is None
    os.remove(fpw)
    print("  [ok] sky overlay: local FITS usable as a backdrop (own WCS, field-"
          "centred), distinct from the inscribe role")

    # A correct WCS is TRUSTED as-is: the placement goes sky->pixel through the
    # WCS onto a fixed N-up/E-left grid, so parity/rotation are already handled
    # and there is NO instrument-specific auto-flip. A frame tagged
    # INSTRUME=GSAOI must therefore place IDENTICALLY to the same frame
    # untagged (an earlier auto-flip mirrored correctly-WCS'd reduced GSAOI
    # mosaics -- see the field-map Backdrop/Frame flip controls, which are now
    # the by-eye correction path for a genuinely mislabeled file).
    rg, dg = 250.0, 36.0
    feat = np.zeros((101, 101)); feat[50, 70] = 100.0     # off-centre in X
    def _mk_instr(instr):
        ph = _fits.Header(); ph["INSTRUME"] = instr
        eh = _fits.Header()
        eh["CTYPE1"] = "RA---TAN"; eh["CTYPE2"] = "DEC--TAN"
        eh["CRPIX1"] = 51; eh["CRPIX2"] = 51; eh["CRVAL1"] = rg; eh["CRVAL2"] = dg
        eh["CD1_1"] = -1.0 / 3600; eh["CD1_2"] = 0    # standard N-up/E-left parity
        eh["CD2_1"] = 0; eh["CD2_2"] = 1.0 / 3600
        fpi = os.path.join(HERE, f"synth_{instr.lower()}.fits")
        _fits.HDUList([_fits.PrimaryHDU(header=ph),
                       _fits.ImageHDU(feat, eh)]).writeto(fpi, overwrite=True)
        return fpi
    fpo = _mk_instr("OTHER"); fpg = _mk_instr("GSAOI")
    so, no, *_ = gui.sky_image_from_fits(fpo, n=101)
    sg, ng, *_ = gui.sky_image_from_fits(fpg, n=101)
    co = np.unravel_index(np.nanargmax(so), so.shape)[1] - 50
    cg = np.unravel_index(np.nanargmax(sg), sg.shape)[1] - 50
    assert "X-flip" not in no and "X-flip" not in ng, (no, ng)
    assert co == cg, f"INSTRUME must not change WCS placement (cols {co} vs {cg})"
    # CD1_1<0 puts the col-70 feature West of centre -> +x (right) in the output
    assert cg > 0, f"feature must land per the WCS (West, +col), got {cg}"
    os.remove(fpo); os.remove(fpg)
    print("  [ok] sky overlay: WCS trusted as-is (no instrument auto-flip); "
          "parity from the file, not INSTRUME")

    # the window title carries the app name + version
    assert gui.APP_NAME in win.windowTitle() and gui.__version__ in \
        win.windowTitle(), win.windowTitle()
    assert gui.APP_NAME == "Keck AO Performance Estimator"
    print(f"  [ok] GUI: title bar shows '{gui.APP_NAME} v{gui.__version__}'")

    # bundled documentation resolves and is reachable from the Help menu
    for doc in (gui.DOC_USER_MANUAL, gui.DOC_TECH_NOTE, gui.DOC_BENCH_DIAGRAMS):
        p = gui._bundled_doc(doc)
        assert p and os.path.isfile(p) and p.endswith(".pdf"), (doc, p)
    helps = [m for m in win.menuBar().findChildren(QtWidgets.QMenu)
             if m.title() == "&Help"][0]
    labels = [a.text() for a in helps.actions() if a.text()]
    assert any("User Manual" in t for t in labels) and \
        any("About" in t for t in labels), labels
    print("  [ok] GUI: bundled docs resolve + Help menu (User Manual / About)")

    # display path (no network): with a sky overlay the performance is
    # CONTOUR-ONLY (a filled heatmap over the image is unreadable), so the only
    # image is the gray backdrop
    win.plot_tabs.setCurrentIndex(1); app.processEvents()
    win._sky_bg_img = np.random.default_rng(0).random((200, 200))
    win._sky_bg_half = gui.FIELD_OF_REGARD_RADIUS_ARCSEC
    win._on_fieldmap_input_changed()
    pump(lambda: fm_idle(win)); app.processEvents()
    fmax = next(ax for ax in win._fm_holder["canvas"].figure.axes if ax.images)
    assert len(fmax.images) == 1 and fmax.images[0].get_cmap().name == "gray", \
        "sky overlay -> only the gray backdrop (no filled heatmap on top)"
    assert len(fmax.collections) >= 1, "performance contours must be drawn"
    # inscribe a smaller local frame ON TOP of the survey backdrop: both layers
    # draw as gray images, the frame at its own (smaller) angular extent, and a
    # dotted outline marks the inscribed frame's footprint
    win._sky_fg_img = np.random.default_rng(1).random((80, 80))
    win._sky_fg_half = 10.0                       # 20" frame inside the 60" FoR
    win._on_fieldmap_input_changed()
    pump(lambda: fm_idle(win)); app.processEvents()
    fmax = next(ax for ax in win._fm_holder["canvas"].figure.axes if ax.images)
    grays = [im for im in fmax.images if im.get_cmap().name == "gray"]
    assert len(grays) == 2, "survey backdrop + inscribed frame both drawn"
    exts = sorted(abs(im.get_extent()[1]) for im in grays)
    assert abs(exts[0] - 10.0) < 1e-6 and exts[1] > exts[0], \
        f"inscribed frame smaller than the survey backdrop ({exts})"
    win._sky_fg_img = None
    # remove the overlay -> the filled heatmap returns
    win._sky_bg_img = None
    win._on_fieldmap_input_changed()
    pump(lambda: fm_idle(win)); app.processEvents()
    fmax = next(ax for ax in win._fm_holder["canvas"].figure.axes if ax.images)
    assert any(im.get_cmap().name.startswith("viridis") for im in fmax.images), \
        "without a sky overlay the filled heatmap returns"
    print("  [ok] sky overlay: frame inscribed in survey; contour-only; heatmap off")

    # a loaded image DEFINES the field centre (its pointing), overriding the
    # typed target for the field map's star-coordinate offsets
    win.ra_edit.setText("10h00m00s"); win.dec_edit.setText("+20d00m00s")  # typed
    img_ra, img_dec = 150.05, 20.0                       # image pointing (E of it)
    hh = _fits.Header()
    hh["NAXIS"] = 2; hh["NAXIS1"] = 200; hh["NAXIS2"] = 200
    hh["CTYPE1"] = "RA---TAN"; hh["CTYPE2"] = "DEC--TAN"
    hh["CRPIX1"] = 100.5; hh["CRPIX2"] = 100.5
    hh["CRVAL1"] = img_ra; hh["CRVAL2"] = img_dec
    hh["CD1_1"] = -0.5 / 3600; hh["CD1_2"] = 0
    hh["CD2_1"] = 0; hh["CD2_2"] = 0.5 / 3600
    hh["MJD-OBS"] = 61071.26518462               # 2026-01-31 06:21 UTC = 20:21 HST
    hh["TARGNAME"] = "FRAME-TGT"
    fp3 = os.path.join(HERE, "synth_field.fits")
    _fits.PrimaryHDU(np.ones((200, 200)), hh).writeto(fp3, overwrite=True)
    QtWidgets.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (fp3, ""))
    n_tgt0 = len(win._targets)
    win._load_local_sky()                        # QFileDialog mocked -> fp3
    for _ in range(6):
        app.processEvents(); QtCore.QThread.msleep(20)
    assert win._sky_center is not None
    assert abs(win._sky_center.ra.deg - img_ra) < 1e-3, "field centre = image"
    # the frame's target joined the night's target list (non-destructive: the
    # typed target/fields are untouched)
    assert any(t["name"] == "FRAME-TGT" for t in win._targets), \
        "loaded frame must add its target to the list"
    assert len(win._targets) == n_tgt0 + 1
    # the frame becomes the ACTIVE target: its name + coordinates fill the
    # Target-tab fields (was typed "10h00m00s"; image pointing is 150.05 deg)
    fc = win._sky_field_center()
    assert win.tname_edit.text() == "FRAME-TGT", win.tname_edit.text()
    assert win.ra_edit.text() == fc[0] and win.ra_edit.text().startswith("10h00m12"), \
        f"frame must become the active target ({win.ra_edit.text()})"
    # the field-map snapshot is driven by the frame's own timestamp
    assert win.fm_cond.currentText() == "specific time", "time mode from header"
    assert win.fm_time.time().hour() == 20 and win.fm_time.time().minute() == 21, \
        win.fm_time.time().toString()
    # a TT star AT the image pointing sits at ~zero offset from the (now frame-
    # defined) field centre, consistent whether measured from fc or the fields
    win.tt_offset.mode.setCurrentIndex(2)
    win.tt_offset.sra.setText("10h00m12s"); win.tt_offset.sdec.setText("+20d00m00s")
    xi, yi, _ = win.tt_offset.offset_xy(fc)
    assert abs(xi) < 2 and abs(yi) < 2, f"star at image centre -> ~0 offset ({xi},{yi})"
    win._clear_frame()
    assert win._sky_center is None, "clearing the frame clears the frame centre"
    os.remove(fp3)
    print("  [ok] sky overlay: loaded frame becomes the active target + field centre")
    win.fm_for.setChecked(False); app.processEvents()

    # multiple targets for one night: Save from the fields, switch between them
    n0 = len(win._targets)
    win.tname_edit.setText("Alpha")
    win.ra_edit.setText("01h00m00s"); win.dec_edit.setText("+10d00m00s")
    win._save_current_target()
    win.tname_edit.setText("Beta")
    win.ra_edit.setText("02h00m00s"); win.dec_edit.setText("-05d00m00s")
    win._save_current_target()
    assert len(win._targets) == n0 + 2, "two targets saved"
    ia = next(i for i, t in enumerate(win._targets) if t["name"] == "Alpha")
    win.target_select.setCurrentIndex(ia); app.processEvents()
    assert win.ra_edit.text() == "01h00m00s" and win.tname_edit.text() == "Alpha", \
        "selecting a target loads its coordinates"
    # Save again with same name updates in place (no duplicate)
    win.ra_edit.setText("01h30m00s"); win._save_current_target()
    assert len(win._targets) == n0 + 2 and \
        win._targets[ia]["ra"] == "01h30m00s", "same-name Save updates in place"
    # Delete removes it
    win.target_select.setCurrentIndex(ia)
    win._delete_target()
    assert not any(t["name"] == "Alpha" for t in win._targets), "delete works"
    print("  [ok] targets: save / select-loads-coords / update / delete")


def window_label_placement():
    """The observing-window label must not cover data in the GUI: it hangs in
    the margin ABOVE the panel (window_label_margin=True). The CLI default
    (False) keeps the frozen in-data position."""
    from astropy.utils.iers import conf; conf.auto_max_age = None
    a = engine.build_parser().parse_args([
        "--dimm", f"{DATA}/20260525_dimm.dat", "--mass", f"{DATA}/20260525_mass.dat",
        "--masspro", f"{DATA}/20260525_masspro.dat", "--telescope", "K1",
        "--target", "--target-name", "TGT", "--ra", "05h24m10.6s",
        "--dec=-24d31m27s", "--window", "22:00-23:00",
        "--out", "/tmp/wl.png", "--force"])
    p = engine.prepare_night(a); r = engine.compute_timeline(a, p)

    def anno_y(margin):
        fig = engine.render_main_figure(a, p, r, window_label_margin=margin)
        for ax in fig.axes:
            for t in ax.texts:
                if "observations" in t.get_text():
                    return t.xy[1]
        return None
    y_cli = anno_y(False)
    y_gui = anno_y(True)
    assert y_cli is not None and abs(y_cli - 0.575) < 1e-9, \
        f"CLI default keeps the in-data label position ({y_cli})"
    assert y_gui is not None and y_gui > 1.0, \
        f"GUI label must sit above the axes (margin), got y={y_gui}"
    print(f"  [ok] window label: CLI in-data (y={y_cli}) / GUI in margin (y={y_gui})")


def real_cn2_profile():
    """field_cn2_profile: the night's real Cn² profile, MEAN over whole night /
    observing window, or the exact profile at a time. Feeds the LGS-tab plot."""
    from astropy.utils.iers import conf; conf.auto_max_age = None
    a = engine.build_parser().parse_args([
        "--dimm", f"{DATA}/20260525_dimm.dat", "--mass", f"{DATA}/20260525_mass.dat",
        "--masspro", f"{DATA}/20260525_masspro.dat", "--telescope", "K1",
        "--out", "/tmp/c.png", "--force"])
    p = engine.prepare_night(a); r = engine.compute_timeline(a, p)
    whole = engine.field_cn2_profile(a, p, r, "night")
    assert whole["n"] == len(r.col_cn2) and whole["cn2_mean"].shape == (6,)
    assert "whole-night mean" in whole["when_desc"]
    # the whole-night mean equals the direct mean of every profile
    assert np.allclose(whole["cn2_mean"],
                       np.asarray(r.col_cn2, float).mean(axis=0))
    # a specific time picks ONE profile (n=1) nearest that time
    t = r.p_times[len(r.p_times) // 3]
    one = engine.field_cn2_profile(a, p, r, "time", t)
    assert one["n"] == 1 and f"{t:%H:%M}" in one["when_desc"]
    assert np.allclose(one["cn2_mean"], np.asarray(r.col_cn2[len(r.p_times)//3]))
    print(f"  [ok] real Cn²: whole-night mean over {whole['n']} profiles / "
          f"exact at a time (n=1)")

    # GUI: the LGS-tab profile plot renders and follows the Conditions selector
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.show(); app.processEvents()
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(f"{DATA}/20260525_dimm.dat")
    win.mass_edit.setText(f"{DATA}/20260525_mass.dat")
    win.masspro_edit.setText(f"{DATA}/20260525_masspro.dat")
    win.tel_k1.setChecked(True); win._validate(); win.on_run()
    pump(lambda: win.res is not None)
    assert len(win.lgs_prof_fig.axes) == 2, "LGS profile draws twin panels"
    st = win.lgs_prof_fig._suptitle.get_text()
    assert "whole-night mean" in st, st
    win.fm_cond.setCurrentText("specific time")
    win.fm_time.setTime(QtCore.QTime(1, 0)); app.processEvents()
    for _ in range(4):
        app.processEvents(); QtCore.QThread.msleep(20)
    assert "01:00 HST" in win.lgs_prof_fig._suptitle.get_text(), \
        "LGS profile must follow the field-map Conditions (specific time)"
    print("  [ok] GUI: LGS-tab real-Cn² plot follows the field-map Conditions")


def tt_sensor_trick():
    """TRICK IR tip-tilt sensor: engine model (faint-limit + band swap) and the
    GUI control (band-swap, K1-only, mag relabel)."""
    from astropy.utils.iers import conf; conf.auto_max_age = None
    s = (0.69 / engine.REF_TOTAL) ** (5.0 / 6.0)
    # TRICK holds tip-tilt near-flat to sensing-mag ~14, then rolls off steeply;
    # it beats STRAP's R quadcell at a faint sensing magnitude
    tk = {m: engine.tt_wfe_nm(s, m, 0.0, sensor="trick") for m in (10, 14, 15)}
    # knee ~14.7 (TRICK_KNEE_MAG): steep core-loss slope below it, bounded
    # photon-limited slope beyond -- no longer an unbounded runaway (6267797)
    assert tk[10] < 80 and tk[14] < 130 and 220 < tk[15] < 320, tk
    assert 2.0 < tk[15] / tk[14] < 3.0, "steep-but-bounded faint-end roll-off"
    # band swap on K1: trick-k -> science H, trick-h -> science K, strap free
    for sensor, want in (("strap", None), ("trick-h", "K"), ("trick-k", "H")):
        a = engine.build_parser().parse_args([
            "--dimm", f"{DATA}/20260525_dimm.dat", "--mass", f"{DATA}/20260525_mass.dat",
            "--masspro", f"{DATA}/20260525_masspro.dat", "--telescope", "K1",
            "--tt-sensor", sensor, "--out", "/tmp/t.png", "--force"])
        engine.resolve_tt_sensor(a)
        base = "trick" if sensor.startswith("trick") else "strap"
        assert a._tt_sensor_base == base
        if want:
            assert a.band == want, f"{sensor} must force science {want}"
    print(f"  [ok] engine: TRICK faint-limit (mag14 {tk[14]:.0f}nm, mag15 "
          f"{tk[15]:.0f}nm) + K1 dichroic band swap")

    # GUI: sensor combo swaps + locks the band, relabels the mag, K1-only
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.show(); app.processEvents()
    win.tel_k1.setChecked(True); app.processEvents()
    assert win.tt_mag_label.text() == "TT-star R mag:" and win.band_combo.isEnabled()
    win.tt_sensor.setCurrentText("TRICK (K)"); app.processEvents()
    assert win.tt_mag_label.text() == "TT-star K mag:"
    assert win.band_combo.currentText() == "H" and not win.band_combo.isEnabled(), \
        "TRICK-K must swap science to H and lock the band"
    a = win.collect_args("/tmp/t.png")
    assert a.tt_sensor == "trick-k" and a.band == "H"
    win.tel_k2.setChecked(True); app.processEvents()
    assert win.tt_sensor.currentText() == "STRAP (R)", "K2 falls back to STRAP"
    _m = win.tt_sensor.model()
    _tk = [_m.item(i) for i in range(win.tt_sensor.count())
           if _m.item(i).text().startswith("TRICK")]
    assert _tk and all(not it.isEnabled() for it in _tk), \
        "TRICK entries must be greyed out on K2"
    _st = [_m.item(i) for i in range(win.tt_sensor.count())
           if _m.item(i).text().startswith("STRAP")]
    assert all(it.isEnabled() for it in _st), "STRAP choices stay available on K2"
    print("  [ok] GUI: TRICK combo swaps/locks the science band; TRICK K1-only")

    # the refined-vs-legacy STRAP revert: legacy sensor reproduces the frozen
    # sheet; --legacy-budget forces it regardless of the sensor default
    tt_new = engine.tt_wfe_nm(1.0)
    tt_old = engine.tt_wfe_nm(1.0, sensor="strap-legacy")
    assert abs(tt_old - 163.6) < 1.5 and tt_new > tt_old + 50, (tt_old, tt_new)
    tleg = engine.lgs_budget_terms(0.5, 0.3, "K1", "single", legacy=True)
    assert abs(tleg["tt"] - tt_old) < 1e-9, "--legacy-budget must use the old row"
    print("  [ok] engine: STRAP legacy revert reproduces the frozen 163.6 nm")


def static_breakout():
    """The static/calibration error, previously one lumped STATIC=91 nm term, is
    broken out into five physical sub-groups; the telescope-aberration group is
    per-telescope (K1 66 nm worse than K2 47 nm)."""
    from astropy.utils.iers import conf; conf.auto_max_age = None
    # the old lumped constant is gone; the five sub-groups sum (in quadrature)
    # to the per-telescope subtotal -- under the v3_1_3 value set (2026-07-24)
    # 109.2 nm on K2, 122.5 nm on K1 (was 91.3 / 102.4 under v3_1_1)
    assert not hasattr(engine, "STATIC"), "the lumped STATIC term must be gone"
    assert abs(engine.static_subtotal("K2") - 109.24) < 0.05
    assert abs(engine.static_subtotal("K1") - 122.53) < 0.05
    assert engine.STATIC_TEL["K1"] > engine.STATIC_TEL["K2"], "K1 optics worse"
    # the budget dict carries the five groups (not one 'static'); telescope group
    # tracks the telescope; the rest are common
    tk1 = engine.lgs_budget_terms(0.6, 0.35, "K1", "single")
    tk2 = engine.lgs_budget_terms(0.6, 0.35, "K2", "single")
    assert "static" not in tk1 and {"stat_tel", "stat_calib", "stat_dm",
        "stat_inst", "stat_reg"} <= set(tk1), sorted(tk1)
    assert tk1["stat_tel"] > tk2["stat_tel"] and tk1["stat_calib"] == tk2["stat_calib"]
    # the K1 correction is a real Strehl penalty and applies to the legacy budget
    # too (a bug-fix, not a methodology change)
    s_k1 = float(engine.lgs_strehl(0.6, 0.35, "K1", "single"))
    with engine.budget_overrides(STATIC_TEL_K1=engine.STATIC_TEL["K2"]):
        s_k1_asK2 = float(engine.lgs_strehl(0.6, 0.35, "K1", "single"))
    assert s_k1 < s_k1_asK2, "the worse K1 telescope term must lower Strehl"
    tleg = engine.lgs_budget_terms(0.6, 0.35, "K1", "single", legacy=True)
    assert tleg["stat_tel"] == engine.STATIC_TEL["K1"], "legacy K1 gets the fix too"
    print(f"  [ok] engine: static broken into 5 groups; K1 subtotal "
          f"{engine.static_subtotal('K1'):.1f} > K2 {engine.static_subtotal('K2'):.1f} nm")

    # GUI: six static sliders, the telescope pair gated by the active telescope,
    # and no spurious off-default at rest (defaults are 0.1-nm representable)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.show(); app.processEvents()
    for n in ("STATIC_TEL_K1", "STATIC_TEL_K2", "STATIC_CALIB", "STATIC_DM",
              "STATIC_INST", "STATIC_REG"):
        assert n in win.wfe_rows, n
    win.tel_k1.setChecked(True); app.processEvents()
    assert all(r.isEnabled() for r in win._wfe_tel_rows["K1"]) and \
        not any(r.isEnabled() for r in win._wfe_tel_rows["K2"]), "K1 gates K1 rows"
    win.tel_k2.setChecked(True); app.processEvents()
    assert all(r.isEnabled() for r in win._wfe_tel_rows["K2"]) and \
        not any(r.isEnabled() for r in win._wfe_tel_rows["K1"]), "K2 gates K2 rows"
    assert win.current_offsets() == {}, \
        f"static defaults must be at-rest (0.1-nm exact): {win.current_offsets()}"
    print("  [ok] GUI: 6 static sliders, telescope-gated, no spurious modified-budget")


def main():
    engine_contract()
    gui_tab()
    window_label_placement()
    real_cn2_profile()
    tt_sensor_trick()
    static_breakout()
    print("  [ok] field map: engine contract + GUI tab")


if __name__ == "__main__":
    main()
