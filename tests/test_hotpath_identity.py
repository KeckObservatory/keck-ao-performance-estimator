"""The crowded-field hot paths were narrowed for speed.  They must return
BIT-identical values, not merely close ones.

Both optimizations restrict a computation to the domain where its result
can be non-zero:

* `aperture_flux` builds its radius map on the bounding box of the widest
  radius it needs instead of the whole frame (7x on NIRC2, 31x on OSIRIS,
  and now constant in image size).
* `EpsfModel._interp` interpolates only the samples inside `r_stamp_px`;
  the rest were being computed and then overwritten with 0.0.

Neither is an approximation, and neither may be allowed to become one.
The reference implementations below are the pre-optimization code kept
verbatim; if a future change makes the fast path disagree in the last
ulp, these fail.  That matters because the tool's numbers are
cross-validated against the summit IDL widget -- "close enough" is not
the standard the rest of the suite holds.
"""
import numpy as np
import pytest
from scipy.ndimage import map_coordinates

import keck_ao_estimator as engine
from keck_ao_estimator.epsf import EpsfModel
from keck_ao_estimator.image_strehl import radius_map, sigma_clipped_median


def _aperture_flux_fullframe(image, aper_px, x, y, insky_px=None,
                             outsky_px=None, skyval=None, robust=False):
    """The original whole-frame implementation, verbatim."""
    image = np.asarray(image, dtype=float)
    rmap = radius_map(image.shape, x, y)
    ap_sel = rmap <= aper_px
    crowding = 0.0
    sky_sigma = 0.0
    n_ann = 0
    if skyval is None:
        ann = image[(rmap >= insky_px) & (rmap <= outsky_px)]
        mean_sky = float(ann.mean())
        clipped_sky = sigma_clipped_median(ann)
        sky_sigma = 1.4826 * float(np.median(np.abs(ann - clipped_sky)))
        n_ann = int(ann.size)
        skyval = clipped_sky if robust else mean_sky
        flux = float((image[ap_sel] - skyval).sum())
        if flux != 0.0:
            crowding = abs(mean_sky - clipped_sky) * ap_sel.sum() / abs(flux)
    else:
        flux = float((image[ap_sel] - skyval).sum())
    return flux, float(skyval), crowding, sky_sigma, int(ap_sel.sum()), n_ann


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(7)
    img = rng.normal(100.0, 20.0, (512, 512))
    for _ in range(40):
        yy, xx = rng.integers(0, 512, 2)
        img[max(yy - 3, 0):yy + 3, max(xx - 3, 0):xx + 3] += 5e4
    return img


def test_aperture_flux_is_bit_identical_to_fullframe(frame):
    rng = np.random.default_rng(11)
    positions = [(float(rng.uniform(-20, 532)), float(rng.uniform(-20, 532)))
                 for _ in range(120)]
    # corners, edges and off-array centres: the box has to clip exactly as
    # the full-frame mask did
    positions += [(0.0, 0.0), (511.0, 511.0), (0.5, 511.5), (5.0, 256.0),
                  (507.0, 256.0), (256.0, 5.0), (-5.0, 256.0),
                  (256.0, 518.0), (256.0, 256.0)]
    for x, y in positions:
        for robust in (False, True):
            got = engine.aperture_flux(frame, 40.6, x, y, insky_px=48.8,
                                       outsky_px=56.9, robust=robust)
            ref = _aperture_flux_fullframe(frame, 40.6, x, y, insky_px=48.8,
                                           outsky_px=56.9, robust=robust)
            assert got == ref, f"aperture_flux drifted at ({x}, {y})"
        got = engine.aperture_flux(frame, 40.6, x, y, skyval=99.5)
        ref = _aperture_flux_fullframe(frame, 40.6, x, y, skyval=99.5)
        assert got == ref, f"aperture_flux(skyval) drifted at ({x}, {y})"


def _interp_fullfootprint(model, grid, yy, xx, x, y):
    """The original: interpolate everywhere, then zero outside the stamp."""
    u = np.asarray(xx, dtype=float) - float(x)
    v = np.asarray(yy, dtype=float) - float(y)
    out = map_coordinates(
        grid, [v * model.oversample + model.n_half,
               u * model.oversample + model.n_half],
        order=3, prefilter=False, mode="constant", cval=0.0)
    out[u * u + v * v > model.r_stamp_px ** 2] = 0.0
    return out


def test_epsf_interp_is_bit_identical_to_full_footprint():
    rng = np.random.default_rng(3)
    grid = np.abs(rng.normal(0.0, 1.0, (129, 129)))
    grid /= grid.sum()
    model = EpsfModel(grid=grid, grad_y=grid.copy(), grad_x=grid.copy(),
                      oversample=4, r_stamp_px=16.0, fwhm_px=4.5,
                      ee_photrad=0.9)
    for _ in range(200):
        n = int(rng.integers(50, 2000))
        yy = rng.uniform(-60.0, 60.0, n)
        xx = rng.uniform(-60.0, 60.0, n)
        x = float(rng.uniform(-30.0, 30.0))
        y = float(rng.uniform(-30.0, 30.0))
        got = model._interp(grid, yy, xx, x, y)
        ref = _interp_fullfootprint(model, grid, yy, xx, x, y)
        assert np.array_equal(got, ref), "_interp drifted"

    # every sample outside the stamp: must be exactly zero, no NaN, and
    # must not fall over on an empty interpolation
    far_y = np.full(64, 1e4)
    far_x = np.full(64, 1e4)
    got = model._interp(grid, far_y, far_x, 0.0, 0.0)
    assert np.array_equal(got, np.zeros(64))
