#!/usr/bin/env python3
"""field_solve validation harness (fieldsolve PLAN.md Phase 1, WP-1):
the check()/skip()/FAILURES pattern of psf_fit_model.py (and, before it,
nirc2_model.py), run against the same synthetic battery
(psf_fit_synth.py) plus a field-level solve.

`field_solve.py` does not exist yet -- it is Opus's O1 (fieldsolve
PLAN.md section 4/6). Every slot below is therefore a named `skip()`
stub, not a real check: this is the SKELETON only (fieldsolve
WORKORDER_SONNET.md, "Your task list" item 4 / WP-1), so this script is
runnable and CI-green today, and Opus fills in the real assertions
against `solve_field()` / `field_clean()` once they land, without
restructuring anything here.

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

import psf_fit_synth as synth  # noqa: F401 -- imported for Opus's O1 fill-in;
                                # not yet used by the skip stubs below.

FAILURES = []


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
    skip("field_solve (a): isolated star, field_clean is a no-op",
         "awaiting O1")


# ------------------------------------------------------ (b) S2 pair, ±0.02

def s2_pair_checks():
    """A two-star S2 pair at 0.45" separation, sr 0.30: the field solve
    must clean the target within +/-0.02 SR bias, the pre-registered S2
    tolerance (fieldsolve PLAN.md section 5, rule 4).
    Build: `psf_fit_synth.build_s2(params, separations=[0.45], srs=[0.30])`
    (or the single-pair equivalent Opus's O1 wires up).
    """
    print("field_solve (b) -- S2 pair at 0.45\" sr 0.30, cleaned within +/-0.02:")
    skip("field_solve (b): S2 pair 0.45\" sr 0.30 cleaned |bias| <= 0.02",
         "awaiting O1")


# --------------------------------------------------- (c) S5-sparse, <=3 sweeps

def s5_sparse_convergence_checks():
    """A 20-star S5-sparse frame must converge (max relative amplitude
    change over all stars below `tol`) in at most 3 sweeps.
    Build: `psf_fit_synth.build_s5_sparse(params, n_noise=1)`.
    """
    print("field_solve (c) -- 20-star S5-sparse frame converges in <= 3 sweeps:")
    skip("field_solve (c): S5-sparse frame converges, n_sweeps <= 3",
         "awaiting O1")


# ------------------------------------------------ (d) non-convergence refusal

def non_convergence_refusal_checks():
    """A frame constructed NOT to converge (fieldsolve PLAN.md section 4,
    point 3: max_sweeps exhausted without the relative-change tolerance
    being met) must be a REFUSAL for every target in it -- naming the
    sweep and the change in the note -- never a silent fallback to an
    unconverged solution.
    """
    print("field_solve (d) -- a non-converging frame refuses legibly:")
    skip("field_solve (d): non-converging frame refuses, never a silent fallback",
         "awaiting O1")


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
    print("\nfield_solve_model: all checks passed (4 skipped, awaiting O1)")


if __name__ == "__main__":
    main()
