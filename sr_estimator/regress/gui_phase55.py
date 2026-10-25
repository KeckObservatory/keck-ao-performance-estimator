#!/usr/bin/env python3
"""ORKID through its instrument parameter file (Eduardo 2026-10-21: the
reworked Strehl tool is the default way, ORKID follows it). The built-in
ORKID models reproduce orkid.py exactly (parameters, cube reduction, bench
reference with the measured source blur); ORKID headers (CAMNAME = orkid,
no INSTRUME) are not taken by NIRC2's rule for old frames; frame names
carry the _0000 suffix; auto-measure keeps to the PREFIX in ORKID's shared
directory; the GUI switches to ORKID's apertures and PREFIX, measures a
synthetic ORKID frame at S ~ 1, and Latest gives the camera server's
UT-dated night folder; tracking frames get the sky reference.
Offline; QT_QPA_PLATFORM=offscreen.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from astropy.io import fits

TMP = tempfile.mkdtemp(prefix="gui55_")
os.environ.pop("KECK_AO_INSTRUMENTS", None)
os.environ.setdefault("KECK_AO_SR_LOG_DIR", os.path.join(TMP, "logs"))

from qtcompat import QtWidgets

import keck_ao_estimator.gui as gui
from keck_ao_estimator import frame_watch, instruments, orkid
from keck_ao_estimator.image_strehl import measure_strehl
from gui_phase29 import pump


def orkid_header(fwpos="FW_OPEN", pwpos="PW_JOVIAN"):
    return fits.Header({"CAMNAME": "orkid", "FWPOS": fwpos, "PWPOS": pwpos,
                        "DATE-OBS": "2026-10-21", "UTC": "22:35:11.0",
                        "TARGNAME": "bench"})


def write_frame(path, header, scale=1.0):
    """A one-plane ORKID cube (16 bit, BZERO 32768) built from the bench
    reference itself: S = scale."""
    psf = orkid.orkid_dl_psf(0.889, npix=256, daytime=True)
    c = 128
    cut = psf[c - 75:c + 75, c - 75:c + 75]
    img = 100.0 + 20000.0 * scale * cut / cut.max()
    if scale < 1.0:     # move the rest of the light into a broad halo
        yy, xx = np.mgrid[0:150, 0:150]
        halo = np.exp(-((xx - 75) ** 2 + (yy - 75) ** 2) / (2 * 15.0 ** 2))
        img += halo / halo.sum() * (1 - scale) * 20000.0 * cut.sum() / cut.max()
    hdu = fits.PrimaryHDU(np.round(img)[np.newaxis].astype(np.int32),
                          header=header)
    hdu.scale("int16", bzero=32768)
    hdu.writeto(path)


def check_engine():
    o = instruments.get("orkid")
    h = orkid_header()
    assert instruments.identify(h) is o
    old = fits.Header({"CAMNAME": "narrow"})            # pre-INSTRUME NIRC2
    assert instruments.identify(old) is instruments.get("nirc2")
    p = o.frame_params(h)
    assert p == orkid.orkid_frame_params(h)
    assert p.plate_scale_mas == orkid.ORKID_PLATE_SCALE_MAS and p.daytime
    assert np.array_equal(o.dl_psf(p, npix=128), orkid.orkid_dl_psf(
        p.effwave_um, p.plate_scale_mas, p.pmsname, p.pmrangl_deg, npix=128,
        daytime=True, mfd_um=orkid.ORKID_DEFAULT_MFD_UM))
    cube = np.stack([np.full((8, 8), 10.0), np.full((8, 8), 30.0)])
    assert np.array_equal(o.reduce_frame(cube), orkid.orkid_reduce(cube))
    assert o.frame_name("test_", 14) == "test_0014_0000.fits"
    assert o.frame_number("test_0014_0000.fits") == 14
    assert not instruments.get("nirc2").is_frame("test_0014_0000.fits")
    assert o.apertures == (0.35, 0.40, 0.48, 0.1)
    from datetime import datetime, timezone
    night = datetime(2026, 10, 23, 7, 0, tzinfo=timezone.utc)
    assert instruments.find_latest_night(o, utc_now=night) == (
        "k2orkid:/sdata1651/261023", True)
    # daytime: auto -- the bench PCU reference unless the telescope tracked
    sky = orkid_header()
    sky["AXESTAT"] = "tracking"
    ps = o.frame_params(sky)
    assert not ps.daytime and ps == orkid.orkid_frame_params(sky, daytime=False)
    sky["AXESTAT"] = "not controlling"
    assert o.frame_params(sky).daytime
    print("  [ok] ORKID file reaches orkid.py's parameters, reduction and "
          "reference; identity, names, apertures, the UT-dated Latest and "
          "the sky reference for tracking frames from the file")

    d = os.path.join(TMP, "shared")
    os.makedirs(d)
    for name in ("test_0017_0000.fits", "wlOn_0006_0000.fits"):
        open(os.path.join(d, name), "wb").write(b"\0" * 2880)
    assert frame_watch.newest_frame(d, "orkid")[0] == "wlOn_0006_0000.fits"
    assert frame_watch.newest_frame(d, "orkid", "test_")[0] == \
        "test_0017_0000.fits"
    print("  [ok] auto-measure keeps to the PREFIX in a shared directory")

    path = os.path.join(TMP, "one_0001_0000.fits")
    write_frame(path, orkid_header())
    raw = np.asarray(fits.getdata(path), float)
    red = o.reduce_frame(raw)
    r_new = measure_strehl(red, params=p, photometry_radius_arcsec=0.35,
                           bg_inner_arcsec=0.40, bg_outer_arcsec=0.48,
                           peak_radius_arcsec=0.1, dl_psf=o.dl_psf(p))
    r_old = orkid.measure_orkid_frame(path)
    assert r_new.ok and r_new.strehl == r_old.strehl, (r_new.strehl, r_old.strehl)
    print(f"  [ok] file path = orkid.measure_orkid_frame: S={r_new.strehl:.4f}")


def main():
    check_engine()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv[:1])
    d = os.path.join(TMP, "bench")
    os.makedirs(d)
    write_frame(os.path.join(d, "test_0003_0000.fits"), orkid_header(), 0.6)
    win = gui.MainWindow()
    win._nirc2_prefetch_guide_stars = lambda seq, cont: cont()
    items = [win.n2_instrument.itemText(i)
             for i in range(win.n2_instrument.count())]
    assert items == ["NIRC2", "OSIRIS", "OSIRIS IFU", "ORKID"], items
    win.n2_instrument.setCurrentText("ORKID")
    assert win._nirc2_kind() == "orkid" and win.n2_prefix.text() == "test_"
    assert win._nirc2_radii() == (0.35, 0.40, 0.48, 0.1), win._nirc2_radii()
    assert win._nirc2_numbered_name(3) == "test_0003_0000"
    win.n2_instrument.setCurrentText("NIRC2")
    import keck_ao_estimator as engine
    assert win._nirc2_radii() == (engine.NIRC2_PHOTOMETRY_RADIUS_ARCSEC,
                                  engine.NIRC2_BG_INNER_RADIUS_ARCSEC,
                                  engine.NIRC2_BG_OUTER_RADIUS_ARCSEC,
                                  engine.NIRC2_PEAK_RADIUS_ARCSEC)
    win.n2_instrument.setCurrentText("ORKID")
    assert win._nirc2_radii() == (0.35, 0.40, 0.48, 0.1)
    print("  [ok] ORKID in the menu; its PREFIX, apertures and _0000 names")
    win.n2_path.setText(d)
    win.n2_im1.setValue(3)
    win.n2_nim.setValue(1)
    win.n2_nbg.setValue(0)
    win._n2_loaded_files = None
    win._on_nirc2_go()
    pump(lambda: win.n2_go.isEnabled())
    sr = float(win.n2_strehl_out.text())
    assert 0.5 < sr < 0.7, (sr, win.n2_log.toPlainText()[-600:])
    print(f"  [ok] GO! measures the numbered ORKID frame: S={sr:.3f} "
          "(built with 0.6 of the bench reference core)")
    win.n2_path.setText("")
    win._on_nirc2_latest()
    pump(lambda: win.n2_latest.isEnabled())
    from datetime import datetime, timezone
    want = datetime.now(timezone.utc).strftime("k2orkid:/sdata1651/%y%m%d")
    assert win.n2_path.text() == want, (win.n2_path.text(), want)
    print("  [ok] Latest gives the camera server's folder for tonight")
    win.close()
    print("gui_phase55: PASS")


if __name__ == "__main__":
    main()
