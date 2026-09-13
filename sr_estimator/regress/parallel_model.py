#!/usr/bin/env python3
"""parallel (estimator parallelism) identity checks.

Every optimization on `parallel/engine` is an implementation change that
must not move a single number (parallel PLAN PR-D3); each gets a section
here that compares the new code path against the old one, bit for bit.

  (a) D.3  `blank_disc` -- the catalogue builders blank each detected
      peak's disc on the rows it can reach, not with a full-frame
      `radius_map`: the mask itself on random draws, and
      `deep_star_catalog` / `find_stars` on real and synthetic frames,
      against the pre-D.3 code patched back in as the reference.
  (b) D.4  `solve_field` installs the models it renders in the ePSF's
      cache: exactly its keys, each bit-equal to a fresh `at()`, and every
      target's result (both engines) identical with and without the
      install (`install_models=False` on an identical ePSF build).

Default run is the CI-wired subset; --full adds frames. Needs no network
and no proprietary data: the bundled example frame, the packaged K2
calibration and psf_fit_synth.py frames only.
"""
import argparse
import glob
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
warnings.filterwarnings("ignore")

import numpy as np

import keck_ao_estimator as engine
import keck_ao_estimator.epsf as epsf_mod
import keck_ao_estimator.image_strehl as strehl_mod
import psf_fit_synth as synth

FAILURES = []


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ------------------------------------------------------------ (a) D.3 box

def _full_frame_blank(array, x, y, radius, value):
    """The pre-D.3 code, verbatim: the identity reference."""
    array[strehl_mod.radius_map(array.shape, x, y) <= radius] = value


class _PreD3:
    """Patch the catalogue builders back onto the full-frame mask."""

    def __enter__(self):
        self._saved = (epsf_mod.blank_disc, strehl_mod.blank_disc)
        epsf_mod.blank_disc = strehl_mod.blank_disc = _full_frame_blank
        return self

    def __exit__(self, *exc):
        epsf_mod.blank_disc, strehl_mod.blank_disc = self._saved
        return False


def _frames(full):
    """(label, reduced frame, params): the bundled real frame(s), S5-moderate
    seeds and the S2 donor frames."""
    from astropy.io import fits

    flat, mask = engine.load_nirc2_calibration()
    for path in sorted(glob.glob(os.path.join(ROOT, "examples", "*.fits"))):
        with fits.open(path) as h:
            raw = np.asarray(h[0].data, dtype=float)
            hdr = h[0].header
        yield (os.path.basename(path),
               engine.reduce_frame(raw, background=None, flat=flat, badmask=mask),
               engine.nirc2_frame_params(hdr))
    params = synth.synth_params()
    for s in (range(300, 324) if full else range(300, 302)):
        raw, _t = list(synth.build_s5_moderate(params, seed=synth.SEED + s, n_noise=1))[0]
        yield f"S5-moderate SEED+{s}", engine.reduce_frame(raw, flat=flat), params
    for sr in (0.15, 0.30, 0.60):
        raw, _t = synth.build_s2_donor_frame(params, sr)
        yield f"S2 donor frame sr {sr}", engine.reduce_frame(raw, flat=flat), params


def blank_disc_checks(full=False):
    print("parallel (a) -- blank_disc == the full-frame radius_map mask (D.3):")
    rng = np.random.default_rng(20260913)
    radii = (1.0, 3.0, 4.56, 9.1, 17.3, 40.0)
    n_draws, bad = 0, []
    for shape in ((1024, 1024), (37, 53), (1024, 512)):
        base = rng.standard_normal(shape)
        for _ in range(300 if full else 100):
            x = int(rng.integers(-60, shape[1] + 60))
            y = int(rng.integers(-60, shape[0] + 60))
            if rng.random() < 0.3:
                x, y = x + float(rng.random()), y + float(rng.random())
            r = float(rng.choice(radii))
            a, b = base.copy(), base.copy()
            _full_frame_blank(a, x, y, r, -7.0)
            strehl_mod.blank_disc(b, x, y, r, -7.0)
            n_draws += 1
            if not np.array_equal(a, b):
                bad.append((shape, x, y, r))
    check("(a) blank_disc array-equal to the full-frame mask on random draws "
          "(edges, off-array centres, fractional positions)",
          not bad, f"{n_draws} draws; mismatches {bad[:3]}")

    n_frames, mism = 0, []
    for label, red, p in _frames(full):
        work = engine.sigma_filter3(red)
        excl = strehl_mod.NIRC2_PHOTOMETRY_RADIUS_ARCSEC * 1000.0 / p.plate_scale_mas
        t0 = time.perf_counter()
        cat_new = engine.deep_star_catalog(work, p)
        stars_new = engine.find_stars(red, n_stars=60, exclude_px=excl)
        t_new = time.perf_counter() - t0
        with _PreD3():
            t0 = time.perf_counter()
            cat_old = engine.deep_star_catalog(work, p)
            stars_old = engine.find_stars(red, n_stars=60, exclude_px=excl)
            t_old = time.perf_counter() - t0
        same = (list(cat_new) == list(cat_old)
                and cat_new.truncated == cat_old.truncated
                and stars_new == stars_old)
        n_frames += 1
        if not same:
            mism.append(label)
        print(f"    {label}: catalogue {len(cat_new)} stars, find_stars {len(stars_new)}; "
              f"{'identical' if same else 'DIFFERENT'} ({t_old:.2f} s full-frame, "
              f"{t_new:.2f} s box)")
        if full and label.endswith(".fits"):
            dl = engine.nirc2_dl_psf(p.camname, p.pmsname, p.effwave_um, p.pmrangl_deg,
                                     npix=512, daytime=p.daytime, sfp=getattr(p, "sfp", False))
            f_new = engine.measure_field(red, p, n_stars=6, dl_psf=dl, psf_clean=True)
            with _PreD3():
                f_old = engine.measure_field(red, p, n_stars=6, dl_psf=dl, psf_clean=True)
            same_f = [repr(r) for r in f_new] == [repr(r) for r in f_old]
            check(f"(a) {label}: measure_field(psf_clean=True) repr-identical",
                  same_f, f"n={len(f_new)}")
    check("(a) deep_star_catalog and find_stars identical, box vs full-frame",
          not mism, f"{n_frames} frames; different: {mism}")


# ---------------------------------------------------- (b) D.4 cache install

def _models_equal(m1, m2):
    return (all(np.array_equal(getattr(m1, a), getattr(m2, a))
                for a in ("grid", "grad_y", "grad_x"))
            and all(getattr(m1, s) == getattr(m2, s)
                    for s in ("oversample", "r_stamp_px", "fwhm_px", "ee_photrad",
                              "peak_value")))


def cache_install_checks(full=False):
    print("parallel (b) -- the solve's models installed in the ePSF cache change "
          "no result (D.4):")
    params = synth.synth_params()
    flat = engine.load_nirc2_calibration()[0]
    n_fields = n_compared = 0
    key_bad, stars_bad, diffs = [], [], []
    for s in (range(300, 324) if full else range(300, 304)):
        raw, truth = list(synth.build_s5_moderate(params, seed=synth.SEED + s, n_noise=1))[0]
        work = engine.sigma_filter3(engine.reduce_frame(raw, flat=flat))
        ep_a = engine.build_epsf(work, params)
        if not ep_a.usable:
            continue
        n_fields += 1
        ep_b = engine.build_epsf(work, params)      # the same build, never installed into
        cat = engine.deep_star_catalog(work, params)
        inner = engine.sigma_filter3(work)
        sol_a = engine.solve_field(inner, params, ep_a, cat)
        sol_b = engine.solve_field(inner, params, ep_b, cat, install_models=False)
        if repr(sol_a.stars) != repr(sol_b.stars):
            stars_bad.append(s)
        cache = ep_a.__dict__.get("_model_cache", {})
        fresh = engine.build_epsf(work, params)
        bin_px = 1000.0 / float(fresh.plate_scale_mas)
        for k in sorted({st.model_key for st in sol_a.stars}):
            fresh.__dict__.pop("_model_cache", None)
            if k not in cache or not _models_equal(cache[k], fresh.at(k[0] * bin_px,
                                                                      k[1] * bin_px)):
                key_bad.append((s, k))
        if ep_b.__dict__.get("_model_cache"):
            key_bad.append((s, "install_models=False wrote the cache"))
        for tid in truth["target_ids"]:
            st = truth["stars"][tid]
            base = dict(params=params, pos=(st["x"], st["y"]), psf_clean=True,
                        robust_sky=True, star_catalog=cat)
            for eng in ("native", "field"):
                ka = {} if eng == "native" else dict(psf_clean_engine="field", field_solution=sol_a)
                kb = {} if eng == "native" else dict(psf_clean_engine="field", field_solution=sol_b)
                ra = engine.measure_strehl(work, epsf=ep_a, **base, **ka)
                rb = engine.measure_strehl(work, epsf=ep_b, **base, **kb)
                n_compared += 1
                if repr(ra) != repr(rb):
                    diffs.append((s, tid, eng))
    check("(b) solve_field installs exactly its models, each bit-equal to a fresh at()",
          not key_bad, f"{n_fields} built fields; problems {key_bad[:3]}")
    check("(b) the solution itself is identical with and without the install",
          not stars_bad, f"different on {stars_bad}")
    check("(b) every target, both engines: result repr-identical with and without "
          "the installed models", not diffs,
          f"{n_compared} comparisons; different {diffs[:5]}")


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="more frames and draws")
    args = ap.parse_args()
    t_start = time.time()
    for name, fn in (("(a)", lambda: blank_disc_checks(args.full)),
                     ("(b)", lambda: cache_install_checks(args.full))):
        t0 = time.time()
        fn()
        print(f"  ({name} section: {time.time() - t0:.1f}s)\n")
    print(f"total runtime {time.time() - t_start:.1f}s")
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("\nparallel_model: all checks passed")


if __name__ == "__main__":
    main()
