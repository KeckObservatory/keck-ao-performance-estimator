#!/usr/bin/env python3
"""NIRC2 coadds and OSIRIS spectrograph cubes in the engine (Eduardo
2026-10-25: "lets do 1 and 2"): nirc2_coadds reads each coadd of a NIRC2
_unp file (CDS and MCDS) so the coadds sum to the frame, counts only the
complete ones of a cut-short file and measures a coadd's short-exposure SR
against the frame's flux; osiris_ifs reads an ORP cube (DQ 9 = good),
collapses it whole or in wavelength bins, and measures FWHM with the model
fit of model_fwhm.py -- the same value wherever the star sits in its spaxel
-- and a relative SR; the instrument files route ORP cubes and raw
spectrograph frames to OSIRIS IFU, keep imager frames on OSIRIS, and give
Latest the night's SPEC/ORP. nirc2_dl_psf is unchanged without the new
keyword. Offline; no window.
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

TMP = tempfile.mkdtemp(prefix="gui56_")
os.environ.pop("KECK_AO_INSTRUMENTS", None)

from keck_ao_estimator import (frame_watch, instruments, model_fwhm,
                               nirc2_coadds, osiris_ifs)
from keck_ao_estimator.image_strehl import measure_strehl
from keck_ao_estimator.nirc2 import nirc2_frame_params
from keck_ao_estimator.nirc2_psf import nirc2_dl_psf

RNG = np.random.default_rng(56)


def nirc2_header(coadds, sampmode=2, multisam=1):
    return fits.Header({"INSTRUME": "NIRC2", "CAMNAME": "narrow",
                        "EFFWAVE": 2.1245, "FILTER": "Kp", "PMSNAME": "largehex",
                        "ROTPPOSN": 100.0, "EL": 60.0, "COADDS": coadds,
                        "SAMPMODE": sampmode, "MULTISAM": multisam,
                        "ITIME": 0.05, "DATE-OBS": "2026-07-27",
                        "UTC": "06:00:00.0", "OBJECT": "test"})


def star_image(n=256, x=128.3, y=127.6, scale=1000.0):
    """A diffraction-limited NIRC2 star (narrow camera) plus a little noise."""
    dl = nirc2_dl_psf("narrow", "largehex", 2.1245, 0.0, npix=n,
                      pos=(x - n / 2 + 0.5, y - n / 2 + 0.5))
    return scale * 1e3 * dl


def write_unp(path, coadds, sampmode=2, multisam=1, cut=None):
    """A _unp file: one int16 (reads, y, x) extension per coadd; returns the
    coadds' images. `cut` truncates the file to that many bytes."""
    hdr = nirc2_header(coadds, sampmode, multisam)
    hdus = [fits.PrimaryHDU(header=hdr)]
    images = []
    nreads = 2 * multisam if sampmode == 3 else 2
    for k in range(coadds):
        img = np.round(star_image(64, 32.2, 31.7, scale=50.0 + 5 * k))
        reset = RNG.integers(-60, -40, size=(64, 64)).astype(float)
        reads = np.empty((nreads, 64, 64))
        half = nreads // 2
        for i in range(nreads):
            reads[i] = reset + (img if i >= half else 0.0)
        hdus.append(fits.ImageHDU(reads.astype(np.int16)))
        images.append(img)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    if cut is not None:
        with open(path, "r+b") as fh:
            fh.truncate(cut)
    return images


def check_coadds():
    d = os.path.join(TMP, "nirc2")
    os.makedirs(d)
    imgs = write_unp(os.path.join(d, "n_unp_0007.fits"), 5)
    fits.writeto(os.path.join(d, "n0007.fits"), np.sum(imgs, axis=0),
                 header=nirc2_header(5), overwrite=True)
    f = os.path.join(d, "n0007.fits")
    u = nirc2_coadds.unp_path(f)
    assert u == os.path.join(d, "n_unp_0007.fits"), u
    assert nirc2_coadds.unp_path(u) is None
    assert nirc2_coadds.unp_path(os.path.join(d, "n0008.fits")) is None
    assert nirc2_coadds.coadd_count(u) == 5
    total = sum(nirc2_coadds.read_coadd(u, k) for k in range(1, 6))
    assert np.array_equal(total, fits.getdata(f).astype(float))
    win = nirc2_coadds.coadd_window(fits.open(u, memmap=True), 3, 10, 20, 5, 9)
    assert np.array_equal(win, imgs[2][10:20, 5:9])
    print("  [ok] CDS coadds = read[1] - read[0], sum to the frame; windows")
    # KOA name
    k = os.path.join(d, "N2.20260727.21600.fits")
    fits.writeto(k, np.zeros((4, 4)), overwrite=True)
    open(os.path.join(d, "N2.20260727.21600_unp.fits"), "wb").close()
    assert nirc2_coadds.unp_path(k).endswith("N2.20260727.21600_unp.fits")
    # MCDS
    m = write_unp(os.path.join(d, "n_unp_0009.fits"), 2, sampmode=3,
                  multisam=4)
    assert np.allclose(nirc2_coadds.read_coadd(os.path.join(
        d, "n_unp_0009.fits"), 2), m[1])
    print("  [ok] MCDS coadd = mean(last N) - mean(first N); KOA _unp name")
    # a file cut short in its 4th coadd: 3 complete
    p = os.path.join(d, "n_unp_0011.fits")
    write_unp(p, 6)
    size = os.path.getsize(p)
    write_unp(p, 6, cut=int(size * 0.62))
    assert nirc2_coadds.coadd_count(p) == 3, nirc2_coadds.coadd_count(p)
    print("  [ok] a cut-short _unp file: only its complete coadds count")
    # short-exposure SR: coadd peak / (frame flux / N)
    big = [np.zeros((1024, 1024)) for _ in range(4)]
    for i, b in enumerate(big):
        b[384:640, 384:640] = star_image(scale=200.0 * (1 + 0.02 * i))
        b += RNG.normal(0, 0.3, b.shape)
    frame = np.sum(big, axis=0)
    params = nirc2_frame_params(nirc2_header(4))
    dl = nirc2_dl_psf("narrow", "largehex", 2.1245, params.pmrangl_deg)
    kw = dict(photometry_radius_arcsec=1.0, bg_inner_arcsec=1.2,
              bg_outer_arcsec=1.4, peak_radius_arcsec=0.1, dl_psf=dl)
    fr = measure_strehl(frame, params=params, **kw)
    rs = [nirc2_coadds.measure_coadd(b, fr, 4, **kw) for b in big]
    for r in rs:
        assert r.ok and r.mode == "coadd" and r.params.coadds == 1, r.error
        assert r.sky_mode == nirc2_coadds.COADD_SKY_MODE
        want = fr.strehl * 4 * r.peak / fr.peak
        assert abs(r.strehl - want) < 1e-12 and abs(r.flux - fr.flux / 4) < 1e-6
    assert abs(np.mean([r.strehl for r in rs]) - fr.strehl) < 0.02, \
        ([r.strehl for r in rs], fr.strehl)
    print(f"  [ok] coadd SR against the frame's flux: frame {fr.strehl:.3f}, "
          f"coadds {', '.join(f'{r.strehl:.3f}' for r in rs)}")


def ifs_header(scale="0.035", filt="Kn3", nchan=40, lam0=2121.0):
    return fits.Header({"INSTRUME": "OSIRIS", "CURRINST": "OSIRIS",
                        "INSTR": "spec", "SSCALE": scale, "SFILTER": filt,
                        "IFILTER": "Drk", "CTYPE1": "WAVE", "CUNIT1": "nm",
                        "CRVAL1": lam0, "CDELT1": 2.5, "CRPIX1": 1,
                        "DATE-OBS": "2026-07-14", "UTC": "06:28:59.5",
                        "OBJECT": "A Star", "COADDS": 1})


def ifs_cube(scale_mas, x, y, sigma_mas=0.0, ny=41, nx=50, nchan=40,
             lam0=2121.0):
    """A cube of a blurred diffraction-limited star, built by brute force on
    a fine grid (not model_fwhm's own Fourier path), DQ 9 except a corner."""
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import shift as nd_shift
    q = 10
    nf = 64 * q
    lam = (lam0 + 2.5 * (nchan - 1) / 2) / 1000.0
    p = nirc2_dl_psf("narrow", "open", lam, 38.0, npix=nf, pos=(0.5, 0.5),
                     plate_scale_mas=scale_mas / q)
    if sigma_mas:
        p = gaussian_filter(p, sigma_mas / (scale_mas / q), mode="wrap")
    # fine pixel nf/2 is coarse coordinate (nf/2 - (q-1)/2) / q
    c0 = (nf / 2 - (q - 1) / 2) / q
    p = nd_shift(p, ((y - c0) * q, (x - c0) * q), order=3, mode="wrap")
    img = p.reshape(64, q, 64, q).sum(axis=(1, 3))[:ny, :nx] * 5e4 + 2.0
    data = np.repeat(img[:, :, None], nchan, axis=2).astype(np.float32)
    data += RNG.normal(0, 0.05, data.shape).astype(np.float32)
    dq = np.full(data.shape, 9, np.uint8)
    dq[:4, :6, :] = 0                       # an unfilled corner
    dq[:, :, 0] = 0                         # a bad channel
    return data, dq


def write_cube(path, data, dq, header):
    fits.HDUList([fits.PrimaryHDU(data, header=header),
                  fits.ImageHDU(np.ones_like(data)),
                  fits.ImageHDU(dq)]).writeto(path, overwrite=True)


def check_ifs():
    d = os.path.join(TMP, "ORP")
    os.makedirs(d)
    data, dq = ifs_cube(35, 24.3, 20.6)
    hdr = ifs_header()
    f = os.path.join(d, "s260714_a010002_Kn3_035.fits")
    write_cube(f, data, dq, hdr)
    cube = osiris_ifs.read_cube(f)
    assert cube.data.shape == (41, 50, 40) and not cube.good[0, 0, 5]
    assert not cube.good[10, 10, 0] and cube.good[10, 10, 1]
    assert np.allclose(cube.wavelengths_um[[0, -1]], [2.121, 2.2185])
    img, valid = osiris_ifs.collapse(cube)
    assert not valid[0, 0] and valid[20, 20]
    assert np.isclose(img[20, 20], data[20, 20, 1:].mean(), rtol=1e-5)
    assert osiris_ifs.channel_bins(40, 10) == [(0, 10), (10, 20), (20, 30), (30, 40)]
    assert osiris_ifs.channel_bins(43, 10)[-1] == (30, 43)
    assert osiris_ifs.channel_bins(46, 10)[-1] == (40, 46)
    print("  [ok] ORP cube: DQ 9 good, mean over good channels, bins, axis")
    inst = instruments.get("osiris_ifs")
    assert instruments.identify(hdr) is inst and inst.is_ifs
    raw = fits.Header({"CURRINST": "OSIRIS", "INSTR": "spec", "NAXIS": 2})
    assert instruments.identify(raw) is inst and osiris_ifs.is_ifs_raw(raw)
    img_hdr = fits.Header({"CURRINST": "OSIRIS", "INSTR": "imag"})
    assert instruments.identify(img_hdr) is instruments.get("osiris")
    old_img = fits.Header({"CURRINST": "OSIRIS"})
    assert instruments.identify(old_img) is instruments.get("osiris")
    assert inst.is_frame("s260714_a010002_Kn3_035.fits")
    assert not inst.is_frame("s260714_a010002.fits")
    assert inst.is_skipped("s260714_a010002.fits") and "SPEC/ORP" in inst.skip_reason
    assert not instruments.get("osiris").is_frame("s260714_a010002_Kn3_035.fits")
    assert inst.frame_name("s260714_a", 10002) == "s260714_a010002_*.fits"
    assert inst.apertures == (0.25, 0.30, 0.40, 0.1)
    assert instruments.get("nirc2").coadds_file == "unp"
    assert instruments.get("osiris").coadds_file == ""
    print("  [ok] cubes and raw spectrograph frames -> OSIRIS IFU; imager "
          "frames stay OSIRIS; names, apertures, NIRC2 coadds file")
    p = osiris_ifs.ifs_frame_params(hdr, 2.17)
    assert p.plate_scale_mas == 35.0 and p.effwave_um == 2.17
    assert p.filter_name == "Kn3" and p.pmsname == "open"
    assert np.array_equal(inst.dl_psf(p, npix=128), nirc2_dl_psf(
        "narrow", "open", 2.17, p.pmrangl_deg, npix=128, plate_scale_mas=35.0))
    # the FWHM does not depend on where the star sits in its spaxel
    fw = []
    for x, y in ((24.0, 20.0), (24.5, 20.5), (24.25, 20.8)):
        data, dq = ifs_cube(35, x, y, sigma_mas=10.0)
        write_cube(f, data, dq, hdr)
        cube = osiris_ifs.read_cube(f)
        img, valid = osiris_ifs.collapse(cube)
        pp = osiris_ifs.ifs_frame_params(hdr, float(cube.wavelengths_um.mean()))
        r = osiris_ifs.measure_ifs(img, valid, pp)
        assert r.ok and r.mode == "ifs" and r.note == "", (r.error, r.note)
        # the photometry centre is the tools' centroid (cntrd), within a
        # spaxel on 35 mas; the model fit has its own position
        assert abs(r.x - x) < 0.5 and abs(r.y - y) < 0.5, (r.x, r.y, x, y)
        fw.append((r.fwhm_mas, r.fwhm_dl_mas, r.strehl, r.edge_clip))
    a = np.array(fw)
    truth = model_fwhm._grid(35.0, pp.effwave_um, "open", 38.0, 7).optical_fwhm(10.0)
    assert np.ptp(a[:, 0]) < 0.3 and abs(a[:, 0].mean() - truth) < 0.6, (a, truth)
    assert abs(a[0, 1] - 44.4) < 0.5 and np.all(a[:, 3] == 0.0), a
    print(f"  [ok] model FWHM {a[:, 0].mean():.1f} mas (truth {truth:.1f}) at "
          f"three spaxel phases, spread {np.ptp(a[:, 0]):.2f}; DL {a[0, 1]:.1f}")
    # wavelength bins: the reference follows the bin's wavelength
    b = osiris_ifs.channel_bins(cube.nchan, 20)
    res = []
    for lo, hi in b:
        im_b, v_b = osiris_ifs.collapse(cube, lo, hi)
        pb = osiris_ifs.ifs_frame_params(hdr, float(cube.wavelengths_um[lo:hi].mean()))
        res.append(osiris_ifs.measure_ifs(im_b, v_b, pb))
    assert res[1].fwhm_dl_mas > res[0].fwhm_dl_mas
    print("  [ok] per-bin reference at the bin's wavelength "
          f"({res[0].fwhm_dl_mas:.2f} -> {res[1].fwhm_dl_mas:.2f} mas)")
    # no star in the field
    empty = np.where(valid, 2.0 + RNG.normal(0, 0.05, img.shape), 0.0)
    r = osiris_ifs.measure_ifs(empty, valid, pp)
    assert not r.ok and "IFU field" in r.error, r.error
    print("  [ok] a cube without its star fails and says so")


def check_latest_and_psf():
    real = frame_watch.find_latest_osiris_night
    frame_watch.find_latest_osiris_night = lambda **kw: (
        "k2ao:/s/sdata1100/osiris6/261023/IMAG/raw", True)
    try:
        assert instruments.find_latest_night(instruments.get("osiris_ifs")) == (
            "k2ao:/s/sdata1100/osiris6/261023/SPEC/ORP", True)
        assert instruments.find_latest_night(instruments.get("osiris")) == (
            "k2ao:/s/sdata1100/osiris6/261023/IMAG/raw", True)
    finally:
        frame_watch.find_latest_osiris_night = real
    a = nirc2_dl_psf("narrow", "largehex", 2.1245, 12.0, npix=128)
    b = nirc2_dl_psf("narrow", "largehex", 2.1245, 12.0, npix=128,
                     plate_scale_mas=None)
    assert np.array_equal(a, b)
    print("  [ok] Latest: the night's SPEC/ORP for OSIRIS IFU; nirc2_dl_psf "
          "unchanged without plate_scale_mas")


def check_estimator_worker():
    """The Measured SR worker measures an IFU cube whole, the same way as
    PyAO's strehl_tool (osiris_ifs.measure_ifs), and refuses a raw frame."""
    from qtcompat import QtWidgets
    from keck_ao_estimator.gui.workers import Nirc2MeasureWorker
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    d = os.path.join(TMP, "est")
    os.makedirs(d)
    data, dq = ifs_cube(20, 24.3, 20.6, sigma_mas=8.0)
    f = os.path.join(d, "s261020_a001002_Kn3_020.fits")
    write_cube(f, data, dq, ifs_header(scale="0.020"))
    raw = os.path.join(d, "s261020_a001003.fits")
    fits.PrimaryHDU(np.zeros((8, 8), np.float32), header=fits.Header(
        {"CURRINST": "OSIRIS", "INSTR": "spec"})).writeto(raw)
    w = Nirc2MeasureWorker(d, "", 0, 1, 0, 0, (0.25, 0.30, 0.40, 0.1),
                           files=[("cube", f), ("raw", raw)])
    done, failed = [], []
    w.frame_done.connect(lambda *a: done.append(a))
    w.frame_failed.connect(lambda *a: failed.append(a))
    w.run()
    app.processEvents()
    r = done[0][1]
    cube = osiris_ifs.read_cube(f)
    img, valid = osiris_ifs.collapse(cube)
    p = osiris_ifs.ifs_frame_params(cube.header, float(cube.wavelengths_um.mean()))
    want = osiris_ifs.measure_ifs(img, valid, p)
    assert r.mode == "ifs" and r.fwhm_mas == want.fwhm_mas and r.strehl == want.strehl
    assert len(done) == 1 and "raw spectrograph frame" in failed[0][1]
    print(f"  [ok] Measured SR worker: IFU cube measured whole (FWHM "
          f"{r.fwhm_mas:.1f} / DL {r.fwhm_dl_mas:.1f} mas), raw frame refused")


def main():
    check_coadds()
    check_ifs()
    check_latest_and_psf()
    check_estimator_worker()
    print("gui_phase56: PASS")


if __name__ == "__main__":
    main()
