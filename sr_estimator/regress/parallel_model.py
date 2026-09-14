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
  (c) D.2 / D.5  `workers > 1`: OPEN-8's near-singular SEED+34 targets
      through a 2-worker pool, repr-identical to this process (PR-D11) --
      the one pool call of the default run (PR-OPEN-6). --full adds, at
      min(8, cpu_count // 2) workers: the render pool's models bit-equal to
      the serial render (with its timing), `solve_field(workers=N)`
      identical to `workers=1`, and `measure_field(workers=N)`
      repr-identical to `workers=1` on >= 20 explicit targets, the
      default path, both engines, the capped find_stars path and the auto
      path.
  (d) PR-D12  `field_clean` computes the frame's robust sky only when the
      target's annulus gives no scatter: every field-engine target (frame
      scope through `measure_strehl`, footprint scope directly, OPEN-8's
      near-singular SEED+34 targets included) identical to the pre-PR-D12
      code patched back in as the reference.
  (e) D.7 / PR-D4  worker count: `workers=None` -> min(8, cpu_count // 2),
      KECK_AO_WORKERS, an explicit count winning, < 1 and > 8 refused
      (naming KECK_AO_WORKERS_UNCAPPED) in resolve_workers, get_pool and
      measure_field, and the override allowing it.

Default run is the CI-wired subset; --full adds frames and, in (c), the pool
timing and full-width worker comparisons (PR-OPEN-6: CI runners have 2
vCPUs, and those pools were most of this script's CI growth). Needs no network
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


# ------------------------------------------------- (c) D.2 / D.5 workers

def workers_checks(full=False):
    print("parallel (c) -- workers > 1 returns exactly what workers = 1 returns "
          "(D.2 render pool, D.5 measure_field; T2 identity):")
    import keck_ao_estimator.field_solve as fs_mod
    import keck_ao_estimator.parallel as par

    # PR-OPEN-6 (T7): the default (CI) run makes ONE small pool call -- the
    # PR-D11 case at the end, 2 workers -- so CI still exercises the spawn
    # path, shared memory and a worker computing exactly as this process;
    # the render-pool timing and the solve_field / measure_field pool
    # comparisons at min(8, cpu_count // 2) workers run under --full
    n = max(2, min(8, (os.cpu_count() or 2) // 2)) if full else 2
    params = synth.synth_params()
    flat = engine.load_nirc2_calibration()[0]
    dl = engine.nirc2_dl_psf(params.camname, params.pmsname, params.effwave_um,
                             params.pmrangl_deg, npix=512, daytime=params.daytime,
                             sfp=getattr(params, "sfp", False))

    if full:
        raw, _truth = list(synth.build_s5_moderate(params, seed=synth.SEED + 300, n_noise=1))[0]
        red = engine.reduce_frame(raw, flat=flat)
        work = engine.sigma_filter3(red)
        cat = engine.deep_star_catalog(work, params)

        # --- the render pool: the same models as the serial render
        ep = engine.build_epsf(work, params)
        keys = sorted({fs_mod._bin_key(ep, c["x"], c["y"]) for c in cat})
        t0 = time.perf_counter()
        m_1 = par.render_models(ep, keys, 1)
        t_1 = time.perf_counter() - t0
        t0 = time.perf_counter()
        m_n = par.render_models(ep, keys, n)
        t_n = time.perf_counter() - t0
        bad = [k for k in keys if not _models_equal(m_1[k], m_n[k])]
        check(f"(c) render_models: {n} workers bit-equal to serial",
              not bad, f"{len(keys)} models; different {bad[:3]}; {t_1:.2f} s serial, "
              f"{t_n:.2f} s pool (incl. pool start if first)")

        # --- solve_field
        inner = engine.sigma_filter3(work)
        ep1, epn = engine.build_epsf(work, params), engine.build_epsf(work, params)
        s_1 = engine.solve_field(inner, params, ep1, cat, workers=1)
        s_n = engine.solve_field(inner, params, epn, cat, workers=n)
        same = (repr(s_1.stars) == repr(s_n.stars) and len(s_1.models) == len(s_n.models)
                and all(_models_equal(a, b) for a, b in zip(s_1.models, s_n.models)))
        check(f"(c) solve_field: {n} workers identical to serial (stars and models)",
              same, f"{s_1.n_live} stars, {len({id(m) for m in s_1.models})} models")

        # --- measure_field, >= 20 explicit targets and the capped/auto paths
        positions = [(c["x"], c["y"]) for c in cat]
        cases = [("default path, explicit positions", dict(positions=positions)),
                 ("native psf_clean, explicit positions",
                  dict(positions=positions, psf_clean=True, robust_sky=True)),
                 ("field psf_clean, explicit positions",
                  dict(positions=positions, psf_clean=True, robust_sky=True,
                       psf_clean_engine="field")),
                 ("native psf_clean, find_stars n_stars=6",
                  dict(n_stars=6, psf_clean=True, robust_sky=True)),
                 ("field psf_clean, auto n_stars=None",
                  dict(n_stars=None, psf_clean=True, robust_sky=True, psf_clean_engine="field"))]
        for label, kw in cases:
            f_1 = engine.measure_field(red, params, dl_psf=dl, workers=1, **kw)
            f_n = engine.measure_field(red, params, dl_psf=dl, workers=n, **kw)
            check(f"(c) measure_field {label}: {n} workers repr-identical to serial",
                  [repr(r) for r in f_1] == [repr(r) for r in f_n],
                  f"{len(f_1)} results serial, {len(f_n)} with workers")

    # --- the sensitive case (PR-D11): OPEN-8's near-singular SEED+34 targets,
    # whose refusal note carries a cleaned SR that BLAS threading moves in
    # its last digit; a worker that does not compute exactly as this process
    # does shows up here, where the well-conditioned targets above did not
    raw34, truth34 = list(synth.build_s5_moderate(params, seed=synth.SEED + 34))[0]
    work34 = engine.sigma_filter3(engine.reduce_frame(raw34, flat=flat))
    ep34 = engine.build_epsf(work34, params)
    cat34 = engine.deep_star_catalog(work34, params)
    pos34 = [(truth34["stars"][t]["x"], truth34["stars"][t]["y"]) for t in (1, 6)]
    kw34 = dict(psf_clean=True, robust_sky=True, epsf=ep34, star_catalog=cat34)
    r_1 = [engine.measure_strehl(work34, params=params, pos=p, dl_psf=dl, **kw34)
           for p in pos34]
    batch = par.MeasureBatch(work34, params, dl, kw34, pos34, n)
    try:
        r_n = [batch.result(i) for i in range(len(pos34))]
    finally:
        batch.close()
    check(f"(c) OPEN-8's near-singular SEED+34 t1/t6: {n} pool results repr-identical "
          "to this process's (PR-D11)",
          [repr(r) for r in r_1] == [repr(r) for r in r_n],
          f"notes {[r.psf_clean_note[40:70] for r in r_n]}")


# ------------------------------------------ (d) PR-D12 frame sky scatter

_D12_NEW = """\
    if sky_sigma0 > 0.0:
        sky_sigma = sky_sigma0
    else:
        _sky_g, sky_sigma = _robust_sky(work)
"""
_D12_OLD = """\
    _sky_g, sky_sigma = _robust_sky(work)
    if sky_sigma0 > 0.0:
        sky_sigma = sky_sigma0
"""


class _PreD12:
    """Patch `field_clean` back to the pre-PR-D12 code, verbatim: the frame's
    robust sky computed for every target, then overwritten by the annulus
    scatter whenever that is positive."""

    def __enter__(self):
        import inspect

        import keck_ao_estimator.field_solve as fs_mod
        src = inspect.getsource(fs_mod.field_clean)
        if src.count(_D12_NEW) != 1:
            raise RuntimeError("(d) PR-D12 block not found once in field_clean")
        ns = {}
        exec(compile(src.replace(_D12_NEW, _D12_OLD), fs_mod.__file__, "exec"),
             fs_mod.__dict__, ns)
        self._mod, self._saved = fs_mod, fs_mod.field_clean
        fs_mod.field_clean = ns["field_clean"]
        return self

    def __exit__(self, *exc):
        self._mod.field_clean = self._saved
        return False


def frame_sky_checks(full=False):
    print("parallel (d) -- field_clean computes the frame's robust sky only when "
          "it is used (PR-D12):")
    import keck_ao_estimator.field_solve as fs_mod

    params = synth.synth_params()
    flat = engine.load_nirc2_calibration()[0]
    dl = engine.nirc2_dl_psf(params.camname, params.pmsname, params.effwave_um,
                             params.pmrangl_deg, npix=512, daytime=params.daytime,
                             sfp=getattr(params, "sfp", False))
    seeds = list(range(300, 306) if full else range(300, 302)) + [34]
    n_targets, different, t_new, t_old = 0, [], 0.0, 0.0
    for s in seeds:
        raw, truth = list(synth.build_s5_moderate(params, seed=synth.SEED + s, n_noise=1))[0]
        red = engine.reduce_frame(raw, flat=flat)
        work = engine.sigma_filter3(red)
        cat = engine.deep_star_catalog(work, params)
        ep = engine.build_epsf(work, params, catalog=cat)
        sol = engine.solve_field(engine.sigma_filter3(np.asarray(red, dtype=float)),
                                 params, ep, cat)
        positions = [(c["x"], c["y"]) for c in cat]
        if s == 34:     # OPEN-8's near-singular targets ride along
            positions += [(truth["stars"][t]["x"], truth["stars"][t]["y"]) for t in (1, 6)]
        for scope in ("frame", "footprint"):
            kw = dict(psf_clean=True, robust_sky=True, epsf=ep, star_catalog=cat,
                      psf_clean_engine="field", field_solution=sol)
            t0 = time.perf_counter()
            r_new = [engine.measure_strehl(red, params=params, pos=p, dl_psf=dl, **kw)
                     for p in positions] if scope == "frame" else [
                fs_mod.field_clean(work, sol, p, params, ep, scope="footprint",
                                   catalog=cat, robust_sky=True) for p in positions]
            t_new += time.perf_counter() - t0
            with _PreD12():
                t0 = time.perf_counter()
                r_old = [engine.measure_strehl(red, params=params, pos=p, dl_psf=dl, **kw)
                         for p in positions] if scope == "frame" else [
                    fs_mod.field_clean(work, sol, p, params, ep, scope="footprint",
                                       catalog=cat, robust_sky=True) for p in positions]
                t_old += time.perf_counter() - t0
            if scope == "footprint":    # (array, report): compare both, bit for bit
                same = [np.array_equal(a[0], b[0]) and repr(a[1]) == repr(b[1])
                        for a, b in zip(r_new, r_old)]
            else:
                same = [repr(a) == repr(b) for a, b in zip(r_new, r_old)]
            n_targets += len(same)
            different += [f"SEED+{s} {scope} t{i}" for i, ok in enumerate(same) if not ok]
    check("(d) field engine: every target identical to the pre-PR-D12 field_clean "
          "(measure_strehl frame scope; field_clean footprint scope, array and report)",
          not different, f"{n_targets} comparisons over SEED+{seeds}; different "
          f"{different[:3]}; {t_new:.1f} s vs {t_old:.1f} s pre-PR-D12")


# ------------------------------------------------ (e) D.7 worker count

def _with_env(**env):
    """Context manager: set/unset environment variables for one check."""
    import contextlib

    @contextlib.contextmanager
    def cm():
        saved = {k: os.environ.get(k) for k in env}
        try:
            for k, v in env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            yield
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return cm()


def _refusal(fn):
    try:
        fn()
    except ValueError as exc:
        return str(exc)
    return None


def cap_guard_checks(full=False):
    print("parallel (e) -- worker count: default, KECK_AO_WORKERS, the cap of 8 and "
          "its override (D.7, PR-D4; T4):")
    import keck_ao_estimator.parallel as par

    cpu = os.cpu_count() or 2
    expect = min(8, max(1, cpu // 2))
    with _with_env(KECK_AO_WORKERS=None, KECK_AO_WORKERS_UNCAPPED=None):
        check("(e) workers=None -> min(8, max(1, cpu_count // 2))",
              par.resolve_workers(None) == expect == par.default_workers(),
              f"cpu_count {cpu} -> {par.resolve_workers(None)}")
        check("(e) an explicit count is used as given", par.resolve_workers(3) == 3)
        msg0 = _refusal(lambda: par.resolve_workers(0))
        check("(e) workers < 1 refused", msg0 is not None, repr(msg0))
        msg9 = _refusal(lambda: par.resolve_workers(9))
        check("(e) workers=9 refused, naming the cap and the override variable",
              msg9 is not None and "8" in msg9 and "KECK_AO_WORKERS_UNCAPPED" in msg9,
              repr(msg9))
        msg_pool = _refusal(lambda: par.get_pool(9))
        check("(e) get_pool(9) refused before any process starts",
              msg_pool is not None and par._POOL_WORKERS != 9, repr(msg_pool))
        msg_mf = _refusal(lambda: engine.measure_field(
            np.zeros((64, 64)), synth.synth_params(), positions=[(32, 32)], workers=9))
        check("(e) measure_field(workers=9) refused up front", msg_mf is not None,
              repr(msg_mf))
    with _with_env(KECK_AO_WORKERS="3", KECK_AO_WORKERS_UNCAPPED=None):
        check("(e) KECK_AO_WORKERS=3 sets the default", par.resolve_workers(None) == 3)
        check("(e) an explicit count wins over KECK_AO_WORKERS",
              par.resolve_workers(2) == 2)
    with _with_env(KECK_AO_WORKERS="12", KECK_AO_WORKERS_UNCAPPED=None):
        check("(e) KECK_AO_WORKERS=12 is capped like an argument",
              _refusal(lambda: par.resolve_workers(None)) is not None)
    with _with_env(KECK_AO_WORKERS=None, KECK_AO_WORKERS_UNCAPPED="1"):
        check("(e) KECK_AO_WORKERS_UNCAPPED=1 allows 9 (no pool started)",
              par.resolve_workers(9) == 9)


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="more frames and draws")
    args = ap.parse_args()
    t_start = time.time()
    for name, fn in (("(a)", lambda: blank_disc_checks(args.full)),
                     ("(b)", lambda: cache_install_checks(args.full)),
                     ("(c)", lambda: workers_checks(args.full)),
                     ("(d)", lambda: frame_sky_checks(args.full)),
                     ("(e)", lambda: cap_guard_checks(args.full))):
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
