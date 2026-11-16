#!/usr/bin/env python3
"""OSIRIS bench frames as the K1 IDL tool measures them since keckAOIDL #965/#966 (asurendran 2026-11-12/13).

The checks:

- A frame is a bench frame when PCUNAME is anything but 'telescope'.
- The bench reference is a 10.2 m circle; NIRC2 keeps 11.0 m.
- Bench frames have the H2RG row/column stripes removed (osiris_destripe.py):
  column medians, then row medians, both measured more than 250 px from the star.

The checks run on a synthetic striped bench frame with SR 1, through the scripting call and the GUI reduce.

The bundled superflat opens without astropy's invalid-keyword warning: its 9-character AOOPSMODE card was renamed
AOOPSMOD in place, and the pixel data are unchanged. Offline; no window.
"""
import os
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))

import numpy as np
from astropy.io import fits

from keck_ao_estimator import image_strehl, instruments
from keck_ao_estimator.nirc2 import NIRC2_DAYTIME_PUPIL_M, nirc2_frame_params
from keck_ao_estimator.nirc2_psf import nirc2_dl_psf, nirc2_pupil
from keck_ao_estimator.osiris import OSIRIS_DAYTIME_PUPIL_M, osiris_daytime, osiris_frame_params
from keck_ao_estimator.osiris_destripe import _idl_median, destripe_star_position, osiris_destripe

STAR = (1100, 950)


def bench_frame(seed=3):
    """A 2048^2 striped bench frame (8 counts rms rows, 1 rms columns, 2 noise), the SR-1 bench PSF at STAR."""
    rng = np.random.default_rng(seed)
    psf = nirc2_dl_psf("osiris", "open", 2.169, 38.0, npix=256, daytime=True,
                       daytime_pupil_m=OSIRIS_DAYTIME_PUPIL_M)
    stripes = rng.normal(0.0, 8.0, (2048, 1)) + rng.normal(0.0, 1.0, (1, 2048))
    img = stripes + rng.normal(0.0, 2.0, (2048, 2048))
    x, y = STAR
    img[y - 128:y + 128, x - 128:x + 128] += 2e5 * psf
    return img, stripes


def check_destripe():
    """The IDL median, the star position, the stripes removed and the star's flux kept."""
    a = np.array([[4.0, 1.0, 3.0, 2.0], [9.0, 7.0, 8.0, 0.0]])
    assert list(_idl_median(a, axis=1)) == [3.0, 8.0] and list(_idl_median(a, axis=0)) == [9.0, 7.0, 8.0, 2.0]
    img, stripes = bench_frame()
    assert np.allclose(destripe_star_position(img), STAR, atol=1)
    out = osiris_destripe(img)
    far = np.ones(img.shape, bool)
    far[STAR[1] - 300:STAR[1] + 300, STAR[0] - 300:STAR[0] + 300] = False
    before = np.std(np.nanmedian(np.where(far, img, np.nan), axis=1))
    after = np.std(np.nanmedian(np.where(far, out, np.nan), axis=1))
    assert before > 6.0 and after < 0.3, (before, after)
    x, y = STAR
    box = (slice(y - 20, y + 20), slice(x - 20, x + 20))
    assert abs(out[box].sum() / (img - stripes)[box].sum() - 1.0) < 2e-3
    full = image_strehl.osiris_reduce(img, crop=False, destripe=True)
    assert np.array_equal(image_strehl.osiris_reduce(img, crop=True, destripe=True), full[512:1536, 512:1536])
    print(f"  [ok] destripe: row offsets {before:.1f} -> {after:.2f} counts rms away from the star, star flux kept, "
          "whole frame destriped before the crop")


def check_bench_rule_and_pupil():
    """PCUNAME alone decides; OSIRIS bench pupil 10.2 m, NIRC2 11.0 m; the GUI reduce destripes bench frames only."""
    for pcu, axestat, bench in [("osimg", "not controlling", True), ("osimglgs1", "tracking", True),
                                ("home", "not controlling", True), ("telescope", "not controlling", False),
                                ("", "tracking", False)]:
        assert osiris_daytime(fits.Header({"PCUNAME": pcu, "AXESTAT": axestat})) is bench, (pcu, axestat)
    assert osiris_frame_params(fits.Header({"PCUNAME": "osimg"})).daytime_pupil_m == OSIRIS_DAYTIME_PUPIL_M == 10.2
    n2 = nirc2_frame_params({"CAMNAME": "narrow", "FILTER": "Kp", "PMSNAME": "largehex", "EFFWAVE": 2.12,
                             "PCUNAME": "pcu", "AOHATCH": "closed"})
    assert n2.daytime and n2.daytime_pupil_m == NIRC2_DAYTIME_PUPIL_M == 11.0
    a11 = nirc2_pupil(npix=700, du=0.02, daytime=True).sum()
    a102 = nirc2_pupil(npix=700, du=0.02, daytime=True, daytime_pupil_m=10.2).sum()
    assert abs(a102 / a11 / (10.2 / 11.0) ** 2 - 1.0) < 2e-3
    img, _ = bench_frame()
    osi = instruments.get("osiris")
    bench = osi.frame_params(fits.Header({"CURRINST": "OSIRIS", "IFILTER": "BrGamma", "PCUNAME": "osimg"}))
    sky = osi.frame_params(fits.Header({"CURRINST": "OSIRIS", "IFILTER": "BrGamma", "PCUNAME": "telescope"}))
    assert np.array_equal(osi.reduce_frame(img, params=bench),
                          image_strehl.osiris_reduce(img, crop=False, destripe=True))
    assert np.array_equal(osi.reduce_frame(img, params=sky), image_strehl.osiris_reduce(img, crop=False))
    assert np.array_equal(osi.dl_psf(bench, npix=128), nirc2_dl_psf("osiris", "open", bench.effwave_um, 38.0,
                                                                    npix=128, daytime=True, daytime_pupil_m=10.2))
    print("  [ok] bench = PCUNAME not 'telescope' (tracking or not); OSIRIS bench circle 10.2 m (area x"
          f" {a102 / a11:.3f}), NIRC2 11.0 m; GUI reduce destripes bench frames only")


def check_measure():
    """measure_osiris_frame: SR 1 on the striped bench frame by default; off without the destripe."""
    img, _ = bench_frame()
    path = os.path.join(tempfile.mkdtemp(prefix="gui59_"), "i261112_a001002.fits")
    fits.PrimaryHDU(img.astype(np.float32), header=fits.Header(
        {"CURRINST": "OSIRIS", "INSTR": "imag", "IFILTER": "BrGamma", "COADDS": 1, "PCUNAME": "osimg"})).writeto(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = image_strehl.measure_osiris_frame(path, crop=False, idl_version="2026-10")
        raw = image_strehl.measure_osiris_frame(path, crop=False, destripe=False, idl_version="2026-10")
    assert r.ok and r.params.daytime and abs(r.strehl - 1.0) < 0.02 and abs(r.sky) < 0.1, (r.strehl, r.sky)
    assert abs(raw.strehl - 1.0) > 0.04, raw.strehl
    print(f"  [ok] striped SR-1 bench frame: SR {r.strehl:.3f} destriped, {raw.strehl:.3f} not")


def check_superflat_header():
    """The superflat opens with no astropy header warning; its card is AOOPSMOD = 2."""
    flat_path = os.path.join(os.path.dirname(ROOT), "src", "keck_ao_estimator", "data", "superflat.fits.gz")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with fits.open(flat_path) as hdul:
            hdul.verify("exception")
            hdr = hdul[0].header
            assert hdr["AOOPSMOD"] == 2 and "AOOPSMODE" not in hdr
            assert hdul[0].data.shape == (1024, 1024)
    print("  [ok] superflat header: AOOPSMOD = 2, no invalid-keyword warning")


def main():
    check_destripe()
    check_bench_rule_and_pupil()
    check_measure()
    check_superflat_header()
    print("gui_phase59: PASS")


if __name__ == "__main__":
    main()
