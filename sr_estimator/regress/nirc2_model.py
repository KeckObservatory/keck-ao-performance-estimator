#!/usr/bin/env python3
"""NIRC2 measured-Strehl port (nirc2.py / nirc2_psf.py / image_strehl.py):
synthetic contract checks that run fully offline, plus -- when the
proprietary calibration/image directory is present -- validation against
the IDL tool's golden outputs.

The golden data (IDL sources, NIRC2 frames, and the IDL-run golden CSV)
is proprietary and lives OUTSIDE the repository, in $NIRC2_STREHL_DATA
(default ~/nirc2_strehl).  When that directory or its pieces are missing
the golden section is skipped with a notice; the synthetic section always
runs.  The superflat/supermask calibration pair ships with the package
(Eduardo OK'd bundling it, 2026-07-23) so reduction is exercised offline.

Known, accepted deviations from IDL (see module docstrings): natural cubic
spline instead of IDL SPLINE's tension spline in the FWHM profile
(<=0.4 mas on the goldens), raw sub-image max for the saturation flag, and
a pupil rasterizer that matches POLYFILLV to 15/262144 pixels (half-integer
tie handling on the sextant web seams), leaving measured Strehl within
0.001 of the IDL tool on every golden frame.  When the IDL-written oracle
arrays (idl_pupil.fits / idl_dlpsf.fits, written by the summit tool) are
present in the golden directory they are checked directly.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import warnings
warnings.filterwarnings("ignore")

import numpy as np

import keck_ao_estimator as engine

FAILURES = []


def check(name, cond, detail=""):
    tag = "ok" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------- synthetic

def synthetic_checks():
    print("synthetic contract checks:")
    rng = np.random.default_rng(20260723)

    # --- cntrd recovers a known sub-pixel centroid
    yy, xx = np.mgrid[0:64, 0:64]
    g = np.exp(-(((xx - 31.62) ** 2 + (yy - 33.28) ** 2) / (2 * 2.2 ** 2)))
    cx, cy = engine.cntrd(g, 31, 33, 11)
    # the DAOPHOT derivative centroid has an inherent sub-pixel bias of up
    # to ~0.2 px on offset sources (IDL behaves identically); 0.25 px still
    # catches axis swaps and sign errors
    check("cntrd sub-pixel centroid", abs(cx - 31.62) < 0.25 and abs(cy - 33.28) < 0.25,
          f"({cx:.3f}, {cy:.3f}) vs (31.62, 33.28)")

    # --- find_peak undoes pixelation: pixel-integrated Gaussian's true peak
    sig = 2.5
    amp = 1000.0
    eff = sig ** 2 / (sig ** 2 + 1.0 / 12.0)     # 1D pixel-integration loss
    gpix = amp * eff * np.exp(-(((xx - 31.5) ** 2 + (yy - 33.5) ** 2) / (2 * (sig ** 2 + 1 / 12.0))))
    pk = engine.find_peak(gpix, 31.5, 33.5, 23)
    check("find_peak deconvolves pixelation", abs(pk - amp) / amp < 0.02,
          f"peak {pk:.1f} vs true {amp:.1f}")

    # --- radial-profile FWHM of a Gaussian
    fw = engine.radial_profile_fwhm(g, cx, cy)
    check("radial_profile_fwhm gaussian", abs(fw - 2.355 * 2.2) < 0.15,
          f"{fw:.3f} px vs {2.355 * 2.2:.3f}")

    # --- sigma filter: kills an isolated hot pixel, keeps a real PSF core
    field = rng.normal(100.0, 5.0, (64, 64))
    field[20, 20] = 5000.0
    filt = engine.sigma_filter3(field)
    check("sigma_filter3 removes hot pixel", filt[20, 20] < 200.0,
          f"{filt[20, 20]:.1f}")
    # a diffraction-limited core must pass through untouched: with the
    # correct astrolib variance (no /(bw^2-1) on it) zero core-box pixels
    # are replaced; the once-shipped /8 variant replaced ~180 including
    # the peak (-13% Strehl on real frames)
    kpsf_small = engine.nirc2_dl_psf("narrow", "largehex", 2.2705, 171.3,
                                     npix=256)
    core = 3e6 * kpsf_small + rng.normal(0.0, 0.5, kpsf_small.shape)
    filt2 = engine.sigma_filter3(core)
    iy, ix = np.unravel_index(core.argmax(), core.shape)
    box = np.s_[iy - 12:iy + 13, ix - 12:ix + 13]
    check("sigma_filter3 keeps PSF core (astrolib variance, no /8)",
          int((filt2[box] != core[box]).sum()) == 0,
          f"core-box replacements {(filt2[box] != core[box]).sum()}")

    # --- deadpix_fill replaces with the neighbor median and converges
    im = np.ones((16, 16))
    im[5, 5] = 1e9
    bad = np.zeros((16, 16), bool)
    bad[5, 5] = True
    fixed = engine.deadpix_fill(im, bad)
    check("deadpix_fill neighbor median", fixed[5, 5] == 1.0)

    # --- pupil sanity: daytime circle area, hex stop within expected range
    du = 0.092
    pd = engine.nirc2_pupil(npix=512, du=du, daytime=True)
    area = np.pi * (5.5 / du) ** 2
    check("daytime pupil circle area", abs(pd.sum() - area) / area < 0.01,
          f"{pd.sum()} vs {area:.0f}")
    ph = engine.nirc2_pupil(npix=512, du=du, pmsname="largehex", pmrangl=171.3)
    disc = np.pi * (5.0 / du) ** 2
    check("largehex pupil fill fraction", 0.75 < ph.sum() / disc < 0.95,
          f"{ph.sum() / disc:.3f} of 10 m disc")

    # --- DL PSF: normalized, centered, blue PSF more concentrated than red
    kpsf = engine.nirc2_dl_psf("narrow", "largehex", 2.2705, 171.3, npix=512)
    check("dl psf normalized", abs(kpsf.sum() - 1.0) < 1e-9)
    py, px = np.unravel_index(kpsf.argmax(), kpsf.shape)
    check("dl psf centered", abs(py - 255.5) < 1.1 and abs(px - 255.5) < 1.1,
          f"argmax ({py}, {px})")
    hpsf = engine.nirc2_dl_psf("narrow", "largehex", 1.5804, 171.3, npix=512)
    check("dl psf chromatic scaling", hpsf.max() > kpsf.max(),
          f"H peak {hpsf.max():.4f} vs K {kpsf.max():.4f}")

    # --- end-to-end: a pure DL PSF embedded in a frame must measure S ~ 1
    frame = np.zeros((1024, 1024))
    psf_shift = engine.nirc2_dl_psf("narrow", "largehex", 2.2705, 171.3,
                                    npix=512, pos=(0.3, 0.2))
    frame[500:1012, 100:612] = 1e8 * psf_shift
    frame += rng.normal(0.0, 0.5, frame.shape)
    from astropy.io import fits
    hdr = fits.Header()
    hdr["CAMNAME"] = "narrow"
    hdr["PMSNAME"] = "largehex"
    hdr["EFFWAVE"] = 2.2705
    hdr["ROTPPOSN"] = -1.0
    hdr["EL"] = 42.3
    hdr["COADDS"] = 1
    hdr["DETGAIN"] = 8.0
    hdr["AOHATCH"] = "open"
    hdr["PCUNAME"] = "telescope"
    r = engine.measure_strehl(frame, header=hdr)
    check("measure_strehl of DL PSF ~ 1", r.ok and abs(r.strehl - 1.0) < 0.03,
          f"S={r.strehl:.4f} err={r.error!r}")

    # --- header parsing: pmrangl formula, gain default, daytime logic
    p = engine.nirc2_frame_params(hdr)
    check("pmrangl convention", abs(p.pmrangl_deg - 171.3) < 1e-9,
          f"{p.pmrangl_deg}")
    check("saturation from DETGAIN", abs(p.max_counts - 4000.0) < 1e-9)
    check("night frame not daytime", not p.daytime)
    hdr2 = hdr.copy()
    hdr2["PCUNAME"] = "pcu"
    hdr2["AOHATCH"] = "closed"
    hdr2["DETGAIN"] = 0.0
    p2 = engine.nirc2_frame_params(hdr2)
    check("daytime PCU frame", p2.daytime and abs(p2.max_counts - 8000.0) < 1e-9)
    hdr3 = hdr.copy()
    del hdr3["EFFWAVE"]
    hdr3["FILTER"] = "PK50_1.5 + Kcont"
    check("filter LUT fallback",
          abs(engine.nirc2_frame_params(hdr3).effwave_um - 2.2706) < 1e-9)

    # --- packaged K2 calibration pair loads with the widget's semantics
    flat, mask = engine.load_nirc2_calibration()
    check("packaged calibration pair",
          flat.shape == (1024, 1024) and flat.min() >= 0.2
          and bool(mask[411, 1023]) and mask.sum() > 1000,
          f"bad px {int(mask.sum())}")

    # --- robust sky: crowded annulus (the SgrA* failure mode, synthetic)
    vals = np.concatenate([rng.normal(50.0, 2.0, 2000),
                           np.full(100, 5000.0)])   # sky + star spikes
    med = engine.sigma_clipped_median(vals)
    check("sigma_clipped_median rejects spikes", abs(med - 50.0) < 1.0,
          f"{med:.2f} (mean {vals.mean():.1f})")

    psf512 = engine.nirc2_dl_psf("narrow", "largehex", 2.2705, 171.3,
                                 npix=512, pos=(0.3, 0.2))
    crowd = rng.normal(0.0, 0.5, (1024, 1024))
    crowd[256:768, 256:768] += 3e6 * psf512
    yy, xx = np.mgrid[0:1024, 0:1024]
    for k in range(12):                      # neighbors inside the annulus
        ang = 2 * np.pi * k / 12.0
        cx = 511.5 + 125.0 * np.cos(ang)
        cy = 511.5 + 125.0 * np.sin(ang)
        crowd += 3.0e4 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2)
                                  / (2 * 3.0 ** 2)))
    from astropy.io import fits as _f
    hdr2 = _f.Header()
    for k, v in (("CAMNAME", "narrow"), ("PMSNAME", "largehex"),
                 ("EFFWAVE", 2.2705), ("ROTPPOSN", -1.0), ("EL", 42.3),
                 ("COADDS", 1), ("DETGAIN", 8.0), ("AOHATCH", "open"),
                 ("PCUNAME", "telescope")):
        hdr2[k] = v
    r_def = engine.measure_strehl(crowd, header=hdr2)
    r_rob = engine.measure_strehl(crowd, header=hdr2, robust_sky=True)
    r_pick = engine.measure_strehl(crowd, header=hdr2, sky_override=0.0)
    check("crowded annulus flags CROWDED",
          r_def.crowded and r_def.crowding > engine.CROWDING_WARN_FRAC,
          f"crowding {r_def.crowding:.3f}")
    check("robust sky recovers S in crowded field",
          abs(r_rob.strehl - 1.0) < 0.06
          and abs(r_rob.strehl - 1.0) < abs(r_def.strehl - 1.0),
          f"default {r_def.strehl:.3f} robust {r_rob.strehl:.3f}")
    check("picked-sky override bypasses the annulus",
          abs(r_pick.strehl - 1.0) < 0.06
          and r_pick.sky_mode == "picked" and r_pick.crowding == 0.0,
          f"S {r_pick.strehl:.3f}")
    # physics flag: SR outside (0, 1] is impossible and must say so
    import dataclasses as _dc
    check("unphysical-SR flag",
          r_def.unphysical
          and _dc.replace(r_pick, strehl=1.2).unphysical
          and not _dc.replace(r_pick, strehl=0.55).unphysical,
          f"default S {r_def.strehl:.3f} flagged")

    # --- OSIRIS-readiness: nothing NIRC2-hardcoded in the shared core
    big = rng.normal(0.0, 1.0, (2048, 2048))
    check("2048x2048 frame reduces (no 1024 hardcode)",
          engine.reduce_frame(big).shape == (2048, 2048))
    try:
        engine.reduce_frame(big, flat=np.ones((1024, 1024)))
        shape_guard = False
    except ValueError:
        shape_guard = True
    check("wrong-instrument calibration shape raises", shape_guard)
    opsf = engine.nirc2_dl_psf("osimg", "open", 2.12, 0.0, npix=256)
    check("osimg camera in the shared PSF model",
          abs(opsf.sum() - 1.0) < 1e-9 and opsf.max() > 0.05)

    # --- OSIRIS fork: instrument detect, filter LUT, WL pupil, fix_image
    from astropy.io import fits as _ff
    h_osi = _ff.Header()
    h_osi["CURRINST"] = "OSIRIS"
    h_n2 = hdr.copy()
    h_n2["INSTRUME"] = "NIRC2"
    check("detect_instrument",
          engine.detect_instrument(h_n2) == "nirc2"
          and engine.detect_instrument(h_osi) == "osiris"
          and engine.detect_instrument(_ff.Header({"INSTRUME": "GSAOI"})) == ""
          and engine.detect_instrument(hdr) == "",
          "nirc2 / osiris / '' routing")
    h_osi["IFILTER"] = "BrGamma"
    po = engine.osiris_frame_params(h_osi)
    check("osiris params: IFILTER LUT + pinned pupil",
          abs(po.effwave_um - 2.169) < 1e-9 and po.camname == "osiris"
          and po.pmsname == "open" and po.pmrangl_deg == 38.0
          and po.max_counts == float("inf"),
          f"{po.effwave_um} um")
    h_osi["IFILTER"] = "nosuchfilter"
    check("osiris unknown filter fallback",
          abs(engine.osiris_frame_params(h_osi).effwave_um - 2.2) < 1e-9)
    du2 = 0.05
    pw = engine.nirc2_pupil(npix=512, du=du2, sfp=True)
    aw = np.pi * (11.14 / 2.0 / du2) ** 2
    check("sfp white-light pupil area", abs(pw.sum() - aw) / aw < 0.01,
          f"{pw.sum()} vs {aw:.0f}")

    fx = rng.normal(100.0, 5.0, (256, 256))
    fx[40, 40] = 9000.0                      # isolated spike -> repaired
    fx[100:108, 100:108] = 4000.0            # extended core -> spared
    fixed_im = engine.fix_image(fx)
    check("fix_image repairs spikes, spares cores",
          abs(fixed_im[40, 40] - 100.0) < 30.0
          and fixed_im[103, 103] == fx[103, 103],
          f"spike -> {fixed_im[40, 40]:.1f}")
    big2 = np.arange(2048.0 * 2048.0).reshape(2048, 2048)
    red2 = engine.osiris_reduce(big2)
    check("osiris_reduce crops 2048 to the central 1024",
          red2.shape == (1024, 1024))

    # --- multi-star field: halo/noise-proof auto-find + measure_field
    psf3 = engine.nirc2_dl_psf("narrow", "largehex", 2.2705, 171.3,
                               npix=256, pos=(0.3, 0.2))
    field3 = rng.normal(0.0, 0.5, (1024, 1024))
    yy3, xx3 = np.mgrid[0:256, 0:256]
    halo = np.exp(-(((xx3 - 127.5) ** 2 + (yy3 - 127.5) ** 2)
                    / (2 * 40.0 ** 2)))
    halo /= halo.sum()
    spots = [(200, 300, 3e6), (700, 650, 2e6), (450, 820, 1e6)]
    for cy, cx, amp in spots:
        field3[cy - 128:cy + 128, cx - 128:cx + 128] += \
            amp * (0.85 * psf3 + 0.15 * halo)
    hdr_f = hdr.copy()
    hdr_f["COADDS"] = 50            # keep the planted stars unsaturated
    pts = engine.find_stars(field3, n_stars=10, exclude_px=100.0)
    got = sorted((round(x), round(y)) for x, y in pts)
    want = sorted((cx, cy) for cy, cx, _ in spots)
    check("find_stars: 3 planted stars, no halo knots, SNR+relative floors",
          len(pts) == 3 and all(abs(a - c) <= 1 and abs(b - d) <= 1
                                for (a, b), (c, d) in zip(got, want)),
          f"{got}")
    fres = engine.measure_field(field3, engine.nirc2_frame_params(hdr_f),
                                n_stars=10)
    # and with the shared COADDS=1 header (4000 ADU/coadd ceiling) every
    # planted star saturates -> the field must come back EMPTY
    check("measure_field drops saturated stars entirely",
          engine.measure_field(field3, engine.nirc2_frame_params(hdr),
                               n_stars=10) == [])
    check("measure_field: 3 stars, physical S, brightest first",
          len(fres) == 3
          and all(0.7 < r.strehl < 1.0 for r in fres)
          and round(fres[0].x) == 300 and round(fres[2].x) == 820,
          f"S={[round(r.strehl, 3) for r in fres]}")
    check("measure_field rejects saturated/unphysical/broken-FWHM points",
          all(not r.saturated and not r.unphysical and r.fwhm_mas > 0
              for r in fres))
    # backfill-to-N: a saturated interloper BRIGHTER than two good stars
    # must not consume a slot -- asking for 3 still yields the 3 good ones
    field4 = field3.copy()
    yc, xc = 850, 150                       # clear of the three good stars
    field4[yc - 128:yc + 128, xc - 128:xc + 128] += 4e8 * psf3  # saturates
    f4 = engine.measure_field(field4, engine.nirc2_frame_params(hdr_f),
                              n_stars=3)
    # field self-consistency: a fabricated impossible-high point among a
    # coherent field must be dropped; small fields pass untouched
    import dataclasses as _dc2
    base_r = fres[0]
    coherent = [_dc2.replace(base_r, strehl=v)
                for v in (0.33, 0.35, 0.36, 0.38, 0.40, 0.37)]
    spiked = coherent + [_dc2.replace(base_r, strehl=0.71),
                         _dc2.replace(base_r, strehl=0.09)]
    keep_c, out_c = engine.field_consistent(spiked)
    check("field_consistent drops impossible outliers",
          len(out_c) == 2
          and sorted(round(r.strehl, 2) for r in out_c) == [0.09, 0.71],
          f"dropped {[round(r.strehl, 2) for r in out_c]}")
    keep_s, out_s = engine.field_consistent(spiked[:4])
    check("field_consistent passes small fields through",
          len(keep_s) == 4 and not out_s)

    check("backfill past rejected stars to N accepted",
          len(f4) == 3 and all(0.6 < r.strehl < 1.0 for r in f4)
          and not any(abs(r.x - xc) < 5 for r in f4),
          f"kept {[(round(r.x), round(r.strehl, 2)) for r in f4]}")

    # --- curve-of-growth aperture: three physical regimes
    r_iso, conv_iso = engine.optimize_photometry_radius(
        field3, 300.0, 200.0, 9.942)
    # a seeing halo keeps adding real flux all the way out, so the curve
    # of growth need not "settle" -- what matters is no false clip
    check("auto aperture: isolated star keeps the full aperture",
          r_iso > 0.9, f"r={r_iso:.2f}\" settled={conv_iso}")
    pair = rng.normal(0.0, 0.5, (1024, 1024))
    star_im = 0.85 * psf3 + 0.15 * halo
    pair[384:640, 384:640] += 3e6 * star_im
    pair[444:700, 384:640] += 1.5e6 * star_im     # neighbor 0.6" away
    p_pair = engine.nirc2_frame_params(hdr_f)
    r_fix = engine.measure_strehl(pair, params=p_pair, pos=(511.5, 511.5))
    r_auto = engine.measure_strehl(pair, params=p_pair, pos=(511.5, 511.5),
                                   auto_radius=True)
    check("auto aperture: close pair stops before the neighbor and "
          "recovers S",
          r_auto.photrad_used_arcsec < 0.6
          and abs(r_auto.strehl - 0.86) < abs(r_fix.strehl - 0.86),
          f"fixed S={r_fix.strehl:.3f}, "
          f"auto r={r_auto.photrad_used_arcsec:.2f}\" "
          f"S={r_auto.strehl:.3f}")
    check("auto aperture: rings are not contamination (isolated S "
          "preserved)",
          abs(engine.measure_strehl(field3, params=p_pair,
                                    pos=(300.0, 200.0),
                                    auto_radius=True).strehl
              - fres[0].strehl) < 0.05)

    # --- sr_err: propagated sky-noise uncertainty grows for faint stars
    noisy = rng.normal(0.0, 5.0, (1024, 1024))
    noisy[384:640, 384:640] += 3e6 * star_im
    noisy[128:384, 640:896] += 8e3 * star_im      # faint star
    rb = engine.measure_strehl(noisy, params=p_pair, pos=(511.5, 511.5))
    rf2 = engine.measure_strehl(noisy, params=p_pair, pos=(767.5, 255.5))
    check("sr_err: faint star carries a larger propagated uncertainty",
          0.0 < rb.sr_err < rf2.sr_err and rf2.sr_err > engine.SR_ERR_MAX,
          f"bright ±{rb.sr_err:.4f}, faint ±{rf2.sr_err:.4f}")

    # --- auto star count: quality decides, not brute force
    auto_field = rng.normal(0.0, 5.0, (1024, 1024))
    for cy, cx, amp in [(200, 300, 3e6), (700, 650, 2e6), (450, 820, 1e6),
                        (850, 200, 9e3), (150, 700, 7e3)]:
        auto_field[cy - 128:cy + 128, cx - 128:cx + 128] += amp * star_im
    fa = engine.measure_field(auto_field, p_pair, n_stars=None)
    check("auto star count keeps only quality stars",
          len(fa) == 3 and all(r.sr_err <= engine.SR_ERR_MAX for r in fa),
          f"kept {len(fa)}: ±{[round(r.sr_err, 3) for r in fa]}")

    # --- field statistics: known anisoplanatic field must be recovered
    frng = np.random.default_rng(7)
    fx = frng.uniform(-14, 14, 12)
    fy = frng.uniform(-14, 14, 12)
    fx[0], fy[0] = -3.0, 2.0                       # the reference star
    fth = np.hypot(fx + 3.0, fy - 2.0)
    fs = 0.60 * np.exp(-(fth / 15.0) ** (5.0 / 3.0)) \
        * frng.normal(1.0, 0.01, 12)
    fst = engine.field_statistics(fx, fy, fs, sr_errs=0.01 * fs,
                                  wavelength_um=2.27)
    check("field stats: peak star located",
          abs(fst.peak_dx_arcsec + 3.0) < 1e-9
          and abs(fst.peak_dy_arcsec - 2.0) < 1e-9)
    check("field stats: theta0 recovered from the falloff",
          fst.theta0_arcsec is not None
          and abs(fst.theta0_arcsec - 15.0) < 1.0
          and fst.theta0_500nm_arcsec < fst.theta0_arcsec,
          f"theta0 {fst.theta0_arcsec:.2f}\" (true 15.0), "
          f"500nm {fst.theta0_500nm_arcsec:.2f}\"")
    check("field stats: gradient points downhill (away from the peak)",
          fst.grad_sr_per_arcmin is not None
          and fst.grad_sr_per_arcmin > 0
          and abs(((fst.grad_pa_deg - np.degrees(np.arctan2(
              np.mean(fy) - 2.0, np.mean(fx) + 3.0))) + 180) % 360 - 180)
          < 60,
          f"{fst.grad_sr_per_arcmin:.3f} SR/arcmin at "
          f"{fst.grad_pa_deg:+.0f} deg")
    flat = engine.field_statistics(fx[:6], fy[:6], np.full(6, 0.5))
    check("field stats: flat field refuses a theta0",
          flat.theta0_arcsec is None and "flat" in flat.theta0_note)
    coll = engine.field_statistics([0, 5, 10], [0, 0, 0],
                                   [0.6, 0.5, 0.4])
    check("field stats: collinear stars refuse a gradient",
          coll.grad_sr_per_arcmin is None
          and coll.theta0_note.startswith("needs"))

    # --- theta0_from_ratios: drift-immune through-origin variant
    rth = np.array([0.0, 15.0, 30.0, 45.0])
    rr = np.exp(-(rth / 40.0) ** (5.0 / 3.0))
    t0r, t0e, note = engine.theta0_from_ratios(rth, rr, 0.01 * rr)
    check("theta0_from_ratios recovers a known falloff",
          t0r is not None and abs(t0r - 40.0) < 0.5,
          f"theta0 {t0r:.2f}\" (true 40)")
    t0f, _, notef = engine.theta0_from_ratios(rth, np.ones(4))
    check("theta0_from_ratios refuses non-decreasing ratios",
          t0f is None and "decrease" in notef)

    # --- Marechal WFE round-trip against the engine's forward direction
    wfe = np.sqrt(-np.log(0.552)) * 2.2705e3 / (2 * np.pi)
    check("marechal inversion consistency",
          abs(engine.marechal_strehl(wfe, 2270.5) - 0.552) < 1e-9,
          f"wfe {wfe:.1f} nm")


# ------------------------------------------------------------------ golden

def golden_checks():
    data_dir = os.environ.get(
        "NIRC2_STREHL_DATA", os.path.expanduser("~/nirc2_strehl"))
    csvs = [f for f in (os.listdir(data_dir) if os.path.isdir(data_dir) else [])
            if f.startswith("idl_strehl_golden") and f.endswith(".csv")]
    if not csvs:
        print(f"golden checks: SKIPPED (no golden data under {data_dir})")
        return
    print(f"golden checks against {csvs[0]}:")
    flat, mask = engine.load_nirc2_calibration()      # packaged pair
    with open(os.path.join(data_dir, csvs[0])) as fh:
        rows = list(csv.DictReader(fh))
    for g in rows:
        path = os.path.join(data_dir, g["frame_file"])
        if not os.path.exists(path):
            print(f"  [skip] {g['frame_file']} missing")
            continue
        r = engine.measure_nirc2_frame(path, flat=flat, mask=mask)
        ds = r.strehl - float(g["strehl"])
        df = r.fwhm_mas - float(g["fwhm_mas"])
        dx = r.x - float(g["x_pix"])
        dy = r.y - float(g["y_pix"])
        check(f"frame {g['image']}",
              r.ok and abs(ds) <= 0.0025 and abs(df) <= 0.5
              and abs(dx) <= 0.1 and abs(dy) <= 0.1 and not r.saturated
              and not r.crowded,
              f"S {r.strehl:.4f} (dS {ds:+.4f})  FWHM {r.fwhm_mas:.2f} "
              f"(dF {df:+.2f})  pos ({r.x:.2f}, {r.y:.2f})")

    # oracle arrays written by the summit IDL tool, if the user copied them
    pupil_p = os.path.join(data_dir, "test_images", "idl_pupil.fits")
    dl_p = os.path.join(data_dir, "test_images", "idl_dlpsf.fits")
    if os.path.exists(pupil_p):
        from astropy.io import fits as _fits
        idl_pupil = np.asarray(_fits.getdata(pupil_p)).astype(np.uint8)
        du = 2.2705e-6 / (512.0 * 0.009942 / 206265.0)
        ours = engine.nirc2_pupil(npix=512, du=du, pmsname="largehex",
                                  pmrangl=171.3)
        mm = int((ours != idl_pupil).sum())
        check("pupil vs IDL oracle", mm <= 20, f"{mm}/262144 pixels differ")
    if os.path.exists(dl_p):
        from astropy.io import fits as _fits
        idl_dl = np.asarray(_fits.getdata(dl_p), dtype=float)
        ours = engine.nirc2_dl_psf("narrow", "largehex", 2.2705, 171.3,
                                   npix=512)
        rel = np.sqrt(((ours - idl_dl) ** 2).mean()) / idl_dl.max()
        check("dl psf vs IDL oracle", rel < 1e-4, f"rms/max {rel:.2e}")


def osiris_golden_checks():
    data_dir = os.environ.get(
        "OSIRIS_STREHL_DATA", os.path.expanduser("~/osiris_strehl"))
    csvs = [f for f in (os.listdir(data_dir) if os.path.isdir(data_dir) else [])
            if f.startswith("idl_strehl_golden") and f.endswith(".csv")]
    if not csvs:
        print(f"osiris golden checks: SKIPPED (no golden data under {data_dir})")
        return
    print(f"osiris golden checks against {csvs[0]}:")
    with open(os.path.join(data_dir, csvs[0])) as fh:
        rows = list(csv.DictReader(fh))
    for g in rows:
        path = os.path.join(data_dir, g["frame_file"])
        if not os.path.exists(path):
            print(f"  [skip] {g['frame_file']} missing")
            continue
        r = engine.measure_osiris_frame(path)
        ds = r.strehl - float(g["strehl"])
        df = r.fwhm_mas - float(g["fwhm_mas"])
        dx = r.x - float(g["x_pix"])
        dy = r.y - float(g["y_pix"])
        check(f"osiris frame {g['image']}",
              r.ok and abs(ds) <= 0.004 and abs(df) <= 0.2
              and abs(dx) <= 0.15 and abs(dy) <= 0.15,
              f"S {r.strehl:.4f} (dS {ds:+.4f})  FWHM {r.fwhm_mas:.2f} "
              f"(dF {df:+.2f})  pos ({r.x:.1f}, {r.y:.1f})")


if __name__ == "__main__":
    synthetic_checks()
    golden_checks()
    osiris_golden_checks()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("\nnirc2_model: all checks passed")
