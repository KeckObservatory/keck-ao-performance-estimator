#!/usr/bin/env python3
"""field_solve validation harness (fieldsolve PLAN.md Phase 1, WP-1):
the check()/skip()/FAILURES pattern of psf_fit_model.py (and, before it,
nirc2_model.py), run against the same synthetic battery
(psf_fit_synth.py) plus a field-level solve.

Skeleton: Sonnet (WP-1). Assertions: Opus (O1), against `solve_field()` /
`field_clean()` and the `engine="field"` dispatch in `clean_star` /
`measure_strehl`.

The four slots are FS-CP2's own acceptance criteria (fieldsolve
CHECKPOINTS.md): a frame with nothing to clean is a no-op; a controlled
S2 pair is cleaned to the pre-registered tolerance; a realistic sparse
field converges quickly; a pathological field refuses instead of
guessing.

Needs no network stubs (RULES section 4, inherited from psf_fit): all
synthetic data is local, no astroquery/pyvo/requests import anywhere in
this module.
"""
import argparse
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
import psf_fit_synth as synth

FAILURES = []
_CACHE = {}


def _calibration():
    if "cal" not in _CACHE:
        _CACHE["cal"] = (synth.synth_params(),
                         engine.load_nirc2_calibration()[0])
    return _CACHE["cal"]


def _work(raw, flat):
    return engine.sigma_filter3(engine.reduce_frame(raw, flat=flat))


def _donor_epsf_030():
    """The sr 0.30 ePSF from the S2 clean-donor frame -- the same source
    psf_fit_model.py's S2 surface uses. Built once."""
    if "epsf030" not in _CACHE:
        params, flat = _calibration()
        raw, _truth = synth.build_s2_donor_frame(params, 0.30)
        _CACHE["epsf030"] = engine.build_epsf(_work(raw, flat), params)
    return _CACHE["epsf030"]


def check(name, cond, detail=""):
    tag = "ok" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def skip(name, reason):
    print(f"  [skip] {name}  ({reason})")


# ------------------------------------------------------------- (a) no-op

def noop_isolated_star_checks():
    """A single isolated star: nothing in the frame for the field solver
    to subtract, so `field_clean` must be a no-op -- bit-identical to the
    uncleaned measurement, same as psf_fit_model.py's S1 no-op guard.
    Build: `psf_fit_synth.build_s1(params)`.
    """
    print("field_solve (a) -- single isolated star is a no-op:")
    params, flat = _calibration()
    raw, truth = synth.build_s1(params)[0]
    reduced = engine.reduce_frame(raw, flat=flat)
    work = engine.sigma_filter3(reduced)
    target = next(s for s in truth["stars"] if s["role"] == "target")
    pos = (target["x"], target["y"])
    epsf = engine.build_epsf(work, params)
    cat = engine.deep_star_catalog(work, params)
    sol = engine.solve_field(work, params, epsf, cat)
    check("field_solve (a): the S1 frame's solution converges",
          sol.converged, sol.note)
    # the S1 donors sit 340-484 px away, beyond the 282 px a stamp can
    # reach into the aperture+annulus (FS-D12), so even scope="frame" has
    # nothing to subtract
    for scope in ("frame", "footprint"):
        cleaned, rep = engine.clean_star(work, pos, params, epsf, catalog=cat,
                                         engine="field", field_solution=sol,
                                         field_scope=scope)
        check(f"field_solve (a): scope={scope} refuses with the 0-components "
              "note, array bit-exact",
              not rep.cleaned and rep.engine == "field"
              and rep.note.startswith("0 components to subtract")
              and np.array_equal(cleaned, work),
              f"note={rep.note!r}")
    r0 = engine.measure_strehl(reduced, params=params, pos=pos)
    r1 = engine.measure_strehl(reduced, params=params, pos=pos,
                               psf_clean=True, psf_clean_engine="field")
    check("field_solve (a): measure_strehl(psf_clean_engine='field') == "
          "psf_clean=False (bit-exact)",
          r0.ok and r1.ok and not r1.cleaned and r1.strehl == r0.strehl,
          f"default={r0.strehl!r} field={r1.strehl!r}")


# ------------------------------------------------------ (b) S2 pair, ±0.02

def s2_pair_checks():
    """A two-star S2 pair at 0.45" separation, sr 0.30: the field solve
    must clean the target within +/-0.02 SR bias, the pre-registered S2
    tolerance (fieldsolve PLAN.md section 5, rule 4).
    Build: `psf_fit_synth.build_s2(params, separations=[0.45], srs=[0.30])`
    (or the single-pair equivalent Opus's O1 wires up).
    """
    print("field_solve (b) -- S2 pair at 0.45\" sr 0.30, cleaned within +/-0.02:")
    params, flat = _calibration()
    epsf = _donor_epsf_030()
    check("field_solve (b): sr 0.30 donor-frame ePSF usable", epsf.usable,
          f"tag={epsf.tag!r}")
    # one pair, contrast 0, seed SEED: the S2 surface's own frame builder
    raw, truth = synth._s2_lattice_frame(params, 0.30, 0, (0.45,),
                                         seed=synth.SEED)
    work = _work(raw, flat)
    cat = engine.deep_star_catalog(work, params)
    tstar = truth["stars"][truth["pairs"][0]["target_id"]]
    pos = (tstar["x"], tstar["y"])
    sol = engine.solve_field(work, params, epsf, cat)
    check("field_solve (b): the pair frame's solution converges",
          sol.converged, sol.note)
    cleaned, rep = engine.clean_star(work, pos, params, epsf, catalog=cat,
                                     engine="field", field_solution=sol)
    # measured exactly as psf_fit_model.py's S2 surface measures a cleaned
    # pair, so the +/-0.02 means the same thing it means there
    bias = float("nan")
    if rep.cleaned:
        r = engine.measure_strehl(cleaned, params=params, pos=pos)
        if r.ok:
            bias = r.strehl - truth["sr_truth_isolated"]
    r_def = engine.measure_strehl(work, params=params, pos=pos)
    check("field_solve (b): S2 pair 0.45\" sr 0.30 contrast 0 cleaned "
          "|bias| <= 0.02 (scope=frame)",
          rep.cleaned and abs(bias) <= 0.02,
          f"bias={bias:+.4f} (uncleaned {r_def.strehl - truth['sr_truth_isolated']:+.4f}); "
          f"{rep.note[:90]}")


# --------------------------------------------------- (c) S5-sparse, <=3 sweeps

def s5_sparse_convergence_checks():
    """A 20-star S5-sparse frame must converge (max relative amplitude
    change over all stars below `tol`) in at most 3 sweeps.
    Build: `psf_fit_synth.build_s5_sparse(params, n_noise=1)`.
    """
    print("field_solve (c) -- 20-star S5-sparse frame converges in <= 3 sweeps:")
    f = _s5_sparse()
    sol = engine.solve_field(f["work"], f["params"], f["epsf"], f["cat"])
    check("field_solve (c): S5-sparse frame converges, n_sweeps <= 3",
          sol.converged and sol.n_sweeps <= 3,
          f"{sol.note}; per-sweep change "
          f"{tuple(round(c, 4) for c in sol.max_rel_change)}")
    side_effect_checks()


def side_effect_checks():
    """FS-D13: a field solution must not change the SHIPPED measurement of
    any star measured after it. `EmpiricalPsf.at` caches the model for a
    1-arcsec bin from whichever position asks first, so a solve that asked
    for its own positions changed later clean_star results (caught by the
    FS-E1 driver's cross-tree identity check: 2 of 7 targets on this frame,
    up to 2.3e-3 SR). A moderate-density frame, not a registered FS-E1
    seed; its own ePSF."""
    params, flat = _calibration()
    raw, truth = list(synth.build_s5_moderate(params, seed=synth.SEED + 200))[0]
    work = _work(raw, flat)
    cat = engine.deep_star_catalog(work, params)

    def native(epsf):
        out = []
        for tid in truth["target_ids"]:
            s = truth["stars"][tid]
            r = engine.measure_strehl(work, params=params, pos=(s["x"], s["y"]),
                                      psf_clean=True, robust_sky=True,
                                      epsf=epsf, star_catalog=cat)
            out.append((r.strehl, r.n_subtracted))
        return out

    ep_fresh = engine.build_epsf(work, params)
    if not ep_fresh.usable:
        check("field_solve (c'): side-effect frame builds a usable ePSF", False,
              ep_fresh.note)
        return
    fresh = native(ep_fresh)
    ep = engine.build_epsf(work, params)
    keys0 = set(ep.__dict__.get("_model_cache", {}).keys())
    engine.solve_field(engine.sigma_filter3(work), params, ep, cat)
    keys1 = set(ep.__dict__.get("_model_cache", {}).keys())
    check("field_solve (c'): solve_field leaves the ePSF's model cache untouched",
          keys0 == keys1, f"{len(keys0)} -> {len(keys1)} cached models")
    after = native(ep)
    n_diff = sum(1 for a, b in zip(fresh, after) if a != b)
    check("field_solve (c'): the shipped measurement after a field solve is "
          "bit-identical to one without",
          n_diff == 0, f"{n_diff} of {len(fresh)} targets differ")


def _s5_sparse():
    """The S5-sparse frame with the sr 0.30 donor-frame ePSF. Its own ePSF
    is uncalibrated (too few usable donors), and solve_field correctly
    refuses to build on that, so (c) and (d) supply the clean-donor model
    -- the S2 surface's convention (fieldsolve STATUS, FS-CP2 slot note)."""
    if "s5sparse" not in _CACHE:
        params, flat = _calibration()
        raw, truth = synth.build_s5_sparse(params)[0]
        work = _work(raw, flat)
        _CACHE["s5sparse"] = dict(
            params=params, raw=raw, truth=truth, work=work,
            epsf=_donor_epsf_030(),
            cat=engine.deep_star_catalog(work, params))
    return _CACHE["s5sparse"]


# ------------------------------------------------ (d) non-convergence refusal

def non_convergence_refusal_checks():
    """A frame constructed NOT to converge (fieldsolve PLAN.md section 4,
    point 3: max_sweeps exhausted without the relative-change tolerance
    being met) must be a REFUSAL for every target in it -- naming the
    sweep and the change in the note -- never a silent fallback to an
    unconverged solution.
    """
    print("field_solve (d) -- a non-converging frame refuses legibly:")
    f = _s5_sparse()
    params, work, epsf, cat = f["params"], f["work"], f["epsf"], f["cat"]
    # one sweep and an unreachable tolerance: the solve cannot converge
    sol = engine.solve_field(work, params, epsf, cat, max_sweeps=1, tol=1e-12)
    check("field_solve (d): solution reports NOT converged, naming the "
          "sweep and the change",
          not sol.converged
          and sol.note.startswith("field solution did not converge")
          and "at sweep 1" in sol.note,
          sol.note)
    truth = f["truth"]
    pos = None
    for tid in truth["target_ids"]:
        s = truth["stars"][tid]
        if engine.measure_strehl(work, params=params, pos=(s["x"], s["y"])).ok:
            pos = (s["x"], s["y"])
            break
    if pos is None:
        check("field_solve (d): a measurable target exists on the frame",
              False)
        return
    cleaned, rep = engine.clean_star(work, pos, params, epsf, catalog=cat,
                                     engine="field", field_solution=sol)
    check("field_solve (d): field_clean refuses with the non-convergence "
          "note, array bit-exact",
          not rep.cleaned
          and rep.note.startswith("cleaning refused: field solution did "
                                  "not converge")
          and np.array_equal(cleaned, work),
          f"note={rep.note[:110]!r}")
    r0 = engine.measure_strehl(work, params=params, pos=pos)
    r1 = engine.measure_strehl(work, params=params, pos=pos, psf_clean=True,
                               epsf=epsf, star_catalog=cat,
                               psf_clean_engine="field", field_solution=sol)
    check("field_solve (d): no silent fallback -- measure_strehl keeps the "
          "uncleaned number and says it refused",
          r1.ok and not r1.cleaned and r1.strehl == r0.strehl
          and "did not converge" in r1.psf_clean_note,
          f"SR {r1.strehl!r} vs {r0.strehl!r}")


# ------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="reserved for a slower battery, once O1 lands "
                             "(mirrors psf_fit_model.py's --full; no effect "
                             "on the skip stubs today)")
    args = parser.parse_args()

    t_start = time.time()
    for name, fn in (("(a) no-op", noop_isolated_star_checks),
                      ("(b) S2 pair", s2_pair_checks),
                      ("(c) S5-sparse", s5_sparse_convergence_checks),
                      ("(d) non-convergence", non_convergence_refusal_checks)):
        t0 = time.time()
        fn()
        print(f"  ({name} section: {time.time() - t0:.1f}s)\n")

    print(f"total runtime {time.time() - t_start:.1f}s")
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("\nfield_solve_model: all checks passed")


if __name__ == "__main__":
    main()
