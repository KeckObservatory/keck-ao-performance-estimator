#!/usr/bin/env python3
"""psf_fit validation harness (WP-2 of the psf_fit development plan,
repo history): the
check()/FAILURES pattern of nirc2_model.py, run against the synthetic
S1-S4 battery built by psf_fit_synth.py.

The engine (`epsf.py` / `psf_fit.py`, and `measure_strehl(psf_clean=True)`)
is still `NotImplementedError` pending Opus's O2-O4 -- every section that
needs it catches that specific exception and prints a clear [skip] line
rather than crashing, so this script is runnable (and CI-green) today and
starts asserting real numbers the moment the engine lands. Sections that
don't touch the new engine at all (S1's default-path recovery, S4a flat
consistency, S4b saturation clipping) are REAL checks now.

Default run (no --full) is S1, S4a/S4d, S5-sparse, and S6, and stays
fast -- this is the CI-wired subset (mirrors how nirc2_model.py is
invoked). The S2 bias surface and S3 field comparison, plus S4b/S4c and
S5-moderate/extreme, are behind --full: those build dozens of frames or
several seeds and are not meant to run on every push. S6 (WP-7, D50) is
the exception among the heavier S5/S6 machinery -- one fixed-seed
crowded field, cheap enough to stay in the default run.

Needs no network stubs (RULES section 4): everything here is local
synthetic data plus the packaged K2 superflat/supermask pair -- no
astroquery, pyvo, or requests import anywhere in this module or in
psf_fit_synth.py.
"""
import argparse
import csv
import os
import sys
import time
import warnings
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
warnings.filterwarnings("ignore")

import numpy as np

import keck_ao_estimator as engine
import psf_fit_synth as synth

FAILURES = []
DATA_DIR = os.path.join(HERE, "data")


def check(name, cond, detail=""):
    tag = "ok" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def skip(name, reason):
    print(f"  [skip] {name}  ({reason})")


# --------------------------------------------------------------------- S1

def s1_checks():
    print("S1 -- no-op guard:")
    params = synth.synth_params()
    flat = engine.load_nirc2_calibration()[0]
    raw, truth = synth.build_s1(params)[0]
    reduced = engine.reduce_frame(raw, flat=flat)
    target = next(s for s in truth["stars"] if s["role"] == "target")

    r_default = engine.measure_strehl(reduced, params=params,
                                      pos=(target["x"], target["y"]))
    check("S1 default measurement recovers sr_truth_isolated",
          r_default.ok
          and abs(r_default.strehl - truth["sr_truth_isolated"]) < 0.03,
          f"S={r_default.strehl:.4f} truth={truth['sr_truth_isolated']:.4f}")
    check("S1 default measurement not spuriously CROWDED",
          not r_default.crowded, f"crowding={r_default.crowding:.4f}")

    # WP-1c: the frame now carries real, distant donors, so build_epsf /
    # clean_star exercise the actual no-op path directly -- an isolated
    # star measured with a WORKING ePSF, not the "no usable ePSF" skip a
    # donor-less S1 frame was stuck in (measure_strehl's own psf_clean
    # kwarg is still NotImplementedError pending O4, so these call the
    # engine functions directly, same as Opus's s2_surface_driver.py).
    work = engine.sigma_filter3(reduced)
    try:
        epsf = engine.build_epsf(work, params)
    except NotImplementedError as e:
        skip("S1 WP-1c: build_epsf usable, tag == strict", str(e))
        return
    check("S1 WP-1c: build_epsf usable, tag == strict",
          epsf.usable and epsf.tag == "strict",
          f"usable={epsf.usable} tag={epsf.tag!r} note={epsf.note!r}")

    cat = engine.deep_star_catalog(work, params)
    cleaned, report = engine.clean_star(work, (target["x"], target["y"]),
                                        params, epsf, catalog=cat)
    check("S1 WP-1c: clean_star refuses with the zero-neighbour note "
          "(not the 'no usable ePSF' note)",
          not report.cleaned and "0 neighbours above" in report.note
          and "no usable ePSF" not in report.note,
          f"note={report.note!r}")
    check("S1 WP-1c: no-op guard -- returned array bit-exact to input",
          np.array_equal(cleaned, work))

    try:
        r_clean = engine.measure_strehl(reduced, params=params,
                                        pos=(target["x"], target["y"]),
                                        psf_clean=True)
    except NotImplementedError as e:
        skip("S1 no-op guard: psf_clean=True == psf_clean=False (<=1e-6 SR)",
             str(e))
        return
    check("S1 no-op guard: psf_clean=True == psf_clean=False (<=1e-6 SR)",
          r_clean.ok and abs(r_clean.strehl - r_default.strehl) <= 1e-6,
          f"default={r_default.strehl:.8f} clean={r_clean.strehl:.8f}")


# --------------------------------------------------------------------- S2

def _build_s2_epsf(params, flat, sr, halo_beta):
    """WP-2b: one ePSF per (sr, halo_beta), built ONCE from a clean
    9-isolated-singles donor frame -- never from the pair frames
    themselves (O3 finding: every star on a pair frame is itself part of
    a blend, so an ePSF built from one never converges)."""
    raw, _truth = synth.build_s2_donor_frame(params, sr, halo_beta=halo_beta)
    work = engine.sigma_filter3(engine.reduce_frame(raw, flat=flat))
    return engine.build_epsf(work, params)


def _s2_bias_rows(params, flat, frames):
    """One row per (frame, pair): default-path bias always; psf_clean
    bias via the WP-2b donor-frame ePSF (one build per (sr, halo_beta),
    reused for every pair sharing it) when the engine supports it, else
    None with one [skip] line. Returns (rows, clean_skip_reason,
    epsf_reports) -- epsf_reports maps (sr, halo_beta) -> the ePSF's
    tag/delta/converged/phase_coverage, for the table header (WP-2b:
    "the reader knows what model produced the surface")."""
    rows = []
    epsf_cache = {}
    epsf_reports = {}
    clean_skip_reason = None
    for raw, truth in frames:
        if truth["case"] not in ("S2", "S2_broadwing"):
            continue
        work = engine.sigma_filter3(engine.reduce_frame(raw, flat=flat))
        group = truth["sr_contrast_group"]
        sr = group["sr"]
        beta = truth["halo"]["beta"]
        key = (sr, beta)

        if key not in epsf_cache:
            epsf = None
            if clean_skip_reason is None:
                try:
                    epsf = _build_s2_epsf(params, flat, sr, beta)
                    epsf_reports[key] = {
                        "tag": epsf.tag, "delta": epsf.delta,
                        "converged": epsf.converged,
                        "phase_coverage": epsf.phase_coverage,
                    }
                except NotImplementedError as e:
                    clean_skip_reason = str(e)
            epsf_cache[key] = epsf
        epsf = epsf_cache[key]
        cat = engine.deep_star_catalog(work, params) if epsf is not None else None

        for pair in truth["pairs"]:
            tstar = truth["stars"][pair["target_id"]]
            pos = (tstar["x"], tstar["y"])
            r_def = engine.measure_strehl(work, params=params, pos=pos)
            row = {
                "sr": sr, "halo_beta": beta,
                "contrast_mag": group["contrast_mag"],
                "sep_arcsec": pair["sep_arcsec"],
                "bias_default": (r_def.strehl - truth["sr_truth_isolated"])
                if r_def.ok else None,
                "bias_psf_clean": None,
                "epsf_tag": epsf.tag if epsf is not None else "",
                "refusal_note": "",
            }
            if epsf is not None:
                cleaned, report = engine.clean_star(work, pos, params, epsf,
                                                    catalog=cat)
                if report.cleaned:
                    r_clean = engine.measure_strehl(cleaned, params=params,
                                                     pos=pos)
                    row["bias_psf_clean"] = (
                        r_clean.strehl - truth["sr_truth_isolated"]
                        if r_clean.ok else None)
                else:
                    row["refusal_note"] = report.note
            rows.append(row)
    return rows, clean_skip_reason, epsf_reports


def _write_s2_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["sr", "halo_beta", "contrast_mag", "sep_arcsec",
                  "bias_default", "bias_psf_clean", "epsf_tag",
                  "refusal_note"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _s2_markdown_table(rows, key, epsf_reports=None):
    """rows = separation, columns = contrast, one block per (sr,
    halo_beta) -- WP-2b: the broad-wing slice (beta=2.5) shares sr=0.30
    with the main grid (beta=4.765) and must not be merged into the same
    cells. Cells = mean(bias) +/- sem over noise realizations sharing
    (sr, beta, contrast, sep). Each block header reports the ePSF that
    produced it when `epsf_reports` is given."""
    groups = defaultdict(list)
    for row in rows:
        v = row[key]
        if v is None:
            continue
        groups[(row["sr"], row["halo_beta"], row["contrast_mag"],
               row["sep_arcsec"])].append(v)
    if not groups:
        return f"  (no data for {key} -- engine not yet implemented)"

    blocks = sorted({(k[0], k[1]) for k in groups})
    contrasts = sorted({k[2] for k in groups})
    seps = sorted({k[3] for k in groups})
    lines = [f"\n#### {key}"]
    for sr, beta in blocks:
        lines.append(f"\nsr = {sr}, halo beta = {beta}")
        if epsf_reports is not None and (sr, beta) in epsf_reports:
            ep = epsf_reports[(sr, beta)]
            lines.append(
                f"(ePSF: tag={ep['tag']!r} delta={ep['delta']:.4f} "
                f"converged={ep['converged']} "
                f"phase_coverage={ep['phase_coverage']:.3f})")
        header = "| sep\\\" | " + " | ".join(f"{c:g} mag" for c in contrasts) + " |"
        lines.append(header)
        lines.append("|---" * (len(contrasts) + 1) + "|")
        for sep in seps:
            cells = []
            for c in contrasts:
                vals = groups.get((sr, beta, c, sep))
                if not vals:
                    cells.append("--")
                    continue
                mean = float(np.mean(vals))
                sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) \
                    if len(vals) > 1 else 0.0
                cells.append(f"{mean:+.4f}±{sem:.4f}")
            lines.append(f"| {sep:g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def s2_checks():
    print("S2 -- blended pairs (bias surface):")
    t0 = time.time()
    params = synth.synth_params()
    flat = engine.load_nirc2_calibration()[0]
    frames = synth.build_s2(params)
    rows, clean_skip_reason, epsf_reports = _s2_bias_rows(params, flat, frames)

    csv_path = os.path.join(DATA_DIR, "psf_fit_s2_bias.csv")
    _write_s2_csv(csv_path, rows)
    print(f"  wrote {csv_path} ({len(rows)} rows)")
    print(_s2_markdown_table(rows, "bias_default", epsf_reports))
    if clean_skip_reason is None:
        print(_s2_markdown_table(rows, "bias_psf_clean", epsf_reports))
    else:
        skip("S2 psf_clean bias surface", clean_skip_reason)
    print(f"  ({time.time() - t0:.1f}s)")
    return rows


# --------------------------------------------------------------------- S3

def s3_checks():
    print("S3 -- GC-like field (path comparison):")
    t0 = time.time()
    params = synth.synth_params()
    flat = engine.load_nirc2_calibration()[0]
    raw, truth = synth.build_s3(params, n_noise=1)[0]
    reduced = engine.reduce_frame(raw, flat=flat)
    work = engine.sigma_filter3(reduced)
    target_ids = truth["target_ids"]
    sr_truth = truth["sr_truth_isolated"]

    results = {}
    for name, kw in (("default", {}), ("robust_sky", {"robust_sky": True})):
        biases, n_crowded = [], 0
        for tid in target_ids:
            star = truth["stars"][tid]
            r = engine.measure_strehl(reduced, params=params,
                                      pos=(star["x"], star["y"]), **kw)
            if not r.ok:
                continue
            biases.append(r.strehl - sr_truth)
            if r.crowded:
                n_crowded += 1
        n = len(biases)
        med = float(np.median(biases)) if n else float("nan")
        mad = float(np.median(np.abs(np.array(biases) - med))) if n else float("nan")
        results[name] = {
            "n": n, "median_bias": med, "mad": mad,
            "frac_crowded": (n_crowded / n) if n else float("nan"),
        }

    # WP-2b: S3 builds its ePSF from its OWN field -- unlike S2's clean
    # donor-frame shortcut, GC density means there IS no clean-donor
    # frame here, so this is the honest/realistic case. Must still
    # report the same four numbers the S2 donor ePSF reports.
    try:
        epsf = engine.build_epsf(work, params)
    except NotImplementedError as e:
        skip("S3 path=psf_clean", str(e))
        epsf = None
    if epsf is not None:
        print(f"  S3 field ePSF: tag={epsf.tag!r} delta={epsf.delta:.4f} "
              f"converged={epsf.converged} "
              f"phase_coverage={epsf.phase_coverage:.3f}")
        cat = engine.deep_star_catalog(work, params)
        biases, n_crowded = [], 0
        for tid in target_ids:
            star = truth["stars"][tid]
            pos = (star["x"], star["y"])
            cleaned, report = engine.clean_star(work, pos, params, epsf,
                                                catalog=cat)
            r = engine.measure_strehl(
                cleaned if report.cleaned else work, params=params, pos=pos)
            if not r.ok:
                continue
            biases.append(r.strehl - sr_truth)
            if r.crowded:
                n_crowded += 1
        n = len(biases)
        med = float(np.median(biases)) if n else float("nan")
        mad = float(np.median(np.abs(np.array(biases) - med))) if n else float("nan")
        results["psf_clean"] = {
            "n": n, "median_bias": med, "mad": mad,
            "frac_crowded": (n_crowded / n) if n else float("nan"),
        }

    if results:
        print("  | path        |   n | median bias |    MAD | frac CROWDED |")
        print("  |-------------|----:|------------:|-------:|-------------:|")
        for name, r in results.items():
            print(f"  | {name:<11} | {r['n']:3d} | {r['median_bias']:+.4f}"
                  f"      | {r['mad']:.4f} | {r['frac_crowded']:.3f}        |")
    print(f"  ({time.time() - t0:.1f}s)")
    return results


# --------------------------------------------------------------------- S5

def s5_checks(full=False):
    """PLAN section 10.3 / WP-5 handoff (STATUS.md, Lane A -> Lane C,
    2026-07-31): three density classes, NO supplied ePSF anywhere --
    `build_epsf` runs on the field itself, closing the acceptance-
    criterion hole D30 found in Phase 1's S3 (which the WP-5 handoff
    calls out as ALREADY doing this correctly for its own field, but S5
    formalizes it as a real, CI-wired battery across three specific,
    committable-number densities instead of one).

    Ownership split (WP-5 handoff, explicit): the S5 field BUILDERS and
    this WIRING are Sonnet's; the numeric pass/fail ASSERTIONS for
    moderate/extreme are Lane A's. `sparse` and `extreme` get a real
    `check()` -- sparse must reproduce the S1 no-op bit-identically,
    extreme must refuse legibly at model level (D32). moderate now does
    too: D42/D43 found the |bias| <= 0.02 target belongs to the S2
    controlled-pair geometry, not a random field (most moderate targets
    carry several neighbours across aperture/annulus/beyond, and psi
    quality was never the binding constraint -- target difficulty was).
    D44/D47 derived a criterion that fits what the class actually is:
    cleaned signed median bounded at <= +0.12 (measured anchor +0.1038),
    and |median| must fall by >= 40% vs a robust_sky-only baseline arm
    (measured 44.6%). Scored over multiple seeds (D44 used 12) and only
    over targets where cleaning actually fired, matching D44's own
    "builds only" methodology."""
    print("S5 -- end-to-end at real densities (no supplied ePSF, PLAN "
          "section 10.3):")
    params = synth.synth_params()
    flat = engine.load_nirc2_calibration()[0]

    # --- sparse: real check(), CI-wired (cheap, 20 stars) -- must
    # reproduce the S1 no-op BIT-IDENTICALLY, before and after D31.
    t0 = time.time()
    raw, truth = synth.build_s5_sparse(params)[0]
    reduced = engine.reduce_frame(raw, flat=flat)
    mismatches = []
    skip_reason = None
    for tid in truth["target_ids"]:
        s = truth["stars"][tid]
        pos = (s["x"], s["y"])
        r0 = engine.measure_strehl(reduced, params=params, pos=pos)
        try:
            r1 = engine.measure_strehl(reduced, params=params, pos=pos,
                                       psf_clean=True)
        except NotImplementedError as e:
            skip_reason = str(e)
            break
        # a star this sparse field places near the edge can fail centroid/
        # photometry on BOTH paths identically (e.g. tid 15/16/19 at
        # seed=SEED) -- that is not a psf_clean divergence, so only a
        # differing ok/not-ok VERDICT or a differing strehl on two ok
        # results counts as a mismatch.
        if r0.ok != r1.ok:
            mismatches.append(tid)
        elif r0.ok and r1.ok and abs(r0.strehl - r1.strehl) > 1e-6:
            mismatches.append(tid)
    if skip_reason is not None:
        skip("S5 sparse: psf_clean == default (bit-identical no-op)",
             skip_reason)
    else:
        check("S5 sparse: psf_clean == default (bit-identical no-op)",
              not mismatches,
              f"{len(mismatches)}/{len(truth['target_ids'])} targets differ"
              if mismatches else "")
    print(f"  (S5 sparse: {time.time() - t0:.1f}s)")

    if not full:
        return

    # --- moderate: D44/D47's derived criterion, scored across multiple
    # seeds since build success is seed-sensitive (D40/D42) and any single
    # field gives too few cleaned targets for a stable median. N_SEEDS
    # matches D44's own methodology. Both arms come from ONE
    # measure_strehl(psf_clean=True, robust_sky=True, ...) call per
    # target: `.strehl` is the cleaned bias (robust_sky auto-ignored per
    # D47 once cleaning succeeds) and `.strehl_uncleaned` is the SAME
    # target's robust_sky-only control at the matched aperture (D9) --
    # exactly D44's two comparison arms, without a second call.
    N_SEEDS_MODERATE = 12
    MIN_CLEANED_MODERATE = 5   # Sonnet's own floor, not D47's: guards
    # against a near-empty sample driving a hard pass/fail; D44 itself
    # measured on n=23.
    t0 = time.time()
    clean_biases, robust_biases = [], []
    n_builds, n_fields, skip_reason = 0, 0, None
    for raw, truth in synth.build_s5_moderate(params, n_noise=N_SEEDS_MODERATE):
        n_fields += 1
        reduced = engine.reduce_frame(raw, flat=flat)
        work = engine.sigma_filter3(reduced)
        sr_truth = truth["sr_truth_isolated"]
        try:
            epsf = engine.build_epsf(work, params)
        except NotImplementedError as e:
            skip_reason = str(e)
            break
        if not epsf.usable:
            continue
        n_builds += 1
        cat = engine.deep_star_catalog(work, params)
        for tid in truth["target_ids"]:
            s = truth["stars"][tid]
            pos = (s["x"], s["y"])
            r = engine.measure_strehl(work, params=params, pos=pos,
                                      psf_clean=True, robust_sky=True,
                                      epsf=epsf, star_catalog=cat)
            if r.ok and r.cleaned:
                clean_biases.append(r.strehl - sr_truth)
                robust_biases.append(r.strehl_uncleaned - sr_truth)
    if skip_reason is not None:
        skip("S5 moderate: D44/D47 criterion", skip_reason)
    else:
        print(f"  S5 moderate: {n_builds}/{n_fields} fields built, "
              f"{len(clean_biases)} targets cleaned")
        if len(clean_biases) < MIN_CLEANED_MODERATE:
            print(f"  S5 moderate: only {len(clean_biases)} cleaned targets "
                  f"(< {MIN_CLEANED_MODERATE}) -- sample too small to "
                  "assert, not a failure")
        else:
            clean_med = float(np.median(clean_biases))
            clean_med_abs = float(np.median(np.abs(clean_biases)))
            robust_med_abs = float(np.median(np.abs(robust_biases)))
            reduction = ((robust_med_abs - clean_med_abs) / robust_med_abs
                        if robust_med_abs else float("nan"))
            check("S5 moderate: cleaned signed median <= +0.12 (D47)",
                  clean_med <= 0.12,
                  f"n={len(clean_biases)} signed median={clean_med:+.4f}")
            check("S5 moderate: |median| reduction vs robust_sky >= 40% "
                  "(D44/D47)",
                  reduction >= 0.40,
                  f"robust |median|={robust_med_abs:.4f} "
                  f"clean |median|={clean_med_abs:.4f} "
                  f"reduction={reduction:.1%}")
    print(f"  (S5 moderate: {time.time() - t0:.1f}s)")

    # --- extreme: GC-class. D32's pass criterion is a LEGIBLE, MODEL-
    # level refusal -- one line for the whole field, not N per-star
    # refusals -- so THIS is assertable now regardless of D31. If D31
    # has already made the build usable, that is ALSO a documented pass
    # (WP-5 handoff): report either outcome, assert only the shape of a
    # refusal when one happens.
    t0 = time.time()
    raw, truth = synth.build_s5_extreme(params)[0]
    reduced = engine.reduce_frame(raw, flat=flat)
    work = engine.sigma_filter3(reduced)
    try:
        epsf = engine.build_epsf(work, params)
    except NotImplementedError as e:
        skip("S5 extreme: legible model-level refusal", str(e))
        epsf = None
    if epsf is not None:
        if not epsf.usable:
            check("S5 extreme: legible model-level refusal (one line, "
                  "not per-star)",
                  bool(epsf.note) and "\n" not in epsf.note,
                  f"tag={epsf.tag!r} note={epsf.note!r}")
        else:
            print(f"  S5 extreme: build_epsf SUCCEEDED (D31 already "
                  f"landed?) tag={epsf.tag!r} delta={epsf.delta:.4f} "
                  f"converged={epsf.converged} -- a documented pass "
                  "either way per D32; numeric bias is Lane A's to judge")
    print(f"  (S5 extreme: {time.time() - t0:.1f}s)")


# --------------------------------------------------------------------- S6

def s6_checks():
    """WP-7 handoff (STATUS.md, D50 -> Lane C, 2026-08-04): D50's
    ultrareview found two defects (F1, F2) that lived in the cleaning
    path for a full day and were INVISIBLE to the existing battery,
    because S2's donor frame is isolated enough that `auto_radius` never
    shrinks and predicted neighbour light is always 0.000 -- both bugs
    need `auto_radius=True` TOGETHER WITH `psf_clean=True` on a CROWDED
    field to show up at all. This is that missing case: the S5-moderate
    builder's default (fixed) seed, which D50 measured shrinks
    `auto_radius` on several targets and is confirmed here to build a
    usable ePSF.

    Both assertions are written to FAIL on the pre-D50 engine -- the
    numbers in each `check()` below are D50's own measured before/after
    on this class of field, not invented thresholds:

    F1 -- the D46 over-amplitude gate must not depend on the aperture.
    `_predicted_frac` (imported directly from `epsf.py` -- it is the
    EXACT metric the gate uses and D50 measured, not a reimplementation
    that could silently drift from what the gate actually computes) is
    evaluated at photometry_radius_arcsec = 1.0, 0.8, 0.6" against ONE
    already-built model. Before the fix the ratio was roughly flat/
    inflating as the aperture shrank (D50: 0.053/0.055/0.059); the
    discriminating assertion is that frac(0.6") must be SMALLER than
    frac(1.0"), not larger or roughly equal.

    F2 -- the post-clean re-optimized aperture may EXCEED the uncleaned
    optimum. Before the fix this was arithmetically impossible (the
    optimizer's `r_max` was clamped to the shrunken uncleaned radius);
    the assertion is that at least one target's `auto_radius`-optimized
    CLEANED `photrad_used_arcsec` is strictly greater than its own
    independently-optimized UNCLEANED radius, on the same field.

    Ownership unchanged: this file is Sonnet's; `epsf.py` is Lane A's --
    `_predicted_frac` is called, never edited."""
    from keck_ao_estimator.epsf import _predicted_frac

    print("S6 -- auto_radius + psf_clean on a crowded field (D50/WP-7):")
    params = synth.synth_params()
    flat = engine.load_nirc2_calibration()[0]

    raw, truth = synth.build_s5_moderate(params)[0]
    reduced = engine.reduce_frame(raw, flat=flat)
    work = engine.sigma_filter3(reduced)
    cat = engine.deep_star_catalog(work, params)
    try:
        epsf = engine.build_epsf(work, params, catalog=cat)
    except NotImplementedError as e:
        skip("S6: auto_radius + psf_clean crowded regression", str(e))
        return
    check("S6 precondition: build_epsf succeeds on the fixed "
          "S5-moderate seed", epsf.usable,
          f"tag={epsf.tag!r} note={epsf.note!r}")
    if not epsf.usable:
        return

    # --- F1: the D46 gate metric must not inflate as the aperture shrinks
    ps = params.plate_scale_mas
    model = epsf.at()
    fracs = {}
    for r in (1.0, 0.8, 0.6):
        photrad_px = r * 1000.0 / ps
        fracs[r] = _predicted_frac(
            work, params, model, cat, photrad_px,
            engine.NIRC2_BG_INNER_RADIUS_ARCSEC,
            engine.NIRC2_BG_OUTER_RADIUS_ARCSEC, engine.EPSF_GATE_SAMPLE,
            photometry_radius_arcsec=r)
    check("S6 F1: D46 gate metric does not inflate as aperture shrinks",
          np.isfinite(fracs[0.6]) and np.isfinite(fracs[1.0])
          and fracs[0.6] < fracs[1.0],
          f'frac(1.0")={fracs[1.0]:.4f} frac(0.8")={fracs[0.8]:.4f} '
          f'frac(0.6")={fracs[0.6]:.4f}')

    # --- F2: cleaned auto_radius may exceed the uncleaned optimum
    growth = []
    for tid in truth["target_ids"]:
        s = truth["stars"][tid]
        pos = (s["x"], s["y"])
        r_unclean = engine.measure_strehl(work, params=params, pos=pos,
                                          auto_radius=True)
        r_clean = engine.measure_strehl(work, params=params, pos=pos,
                                        auto_radius=True, psf_clean=True,
                                        epsf=epsf, star_catalog=cat)
        if (r_unclean.ok and r_clean.ok and r_clean.cleaned
                and r_clean.photrad_used_arcsec
                > r_unclean.photrad_used_arcsec):
            growth.append((tid, r_unclean.photrad_used_arcsec,
                          r_clean.photrad_used_arcsec))
    detail = (f"{len(growth)}/{len(truth['target_ids'])} targets grew, "
             f"e.g. target {growth[0][0]}: {growth[0][1]:.2f}\" -> "
             f"{growth[0][2]:.2f}\"" if growth else
             "no target's cleaned aperture exceeded its uncleaned optimum")
    check("S6 F2: cleaned auto_radius exceeds the uncleaned optimum on "
          "at least one target", bool(growth), detail)


# --------------------------------------------------------------------- S4

def s4_checks(full=False):
    print("S4 -- hygiene:")
    params = synth.synth_params()
    flat = engine.load_nirc2_calibration()[0]

    # --- S4a flat-consistency: real check, no engine dependency
    (raw_f, truth_f), (raw_nf, truth_nf) = synth._s4a_pair(params)
    r_f = engine.measure_strehl(
        engine.reduce_frame(raw_f, flat=flat), params=params,
        pos=(truth_f["stars"][0]["x"], truth_f["stars"][0]["y"]))
    r_nf = engine.measure_strehl(
        engine.reduce_frame(raw_nf, flat=None), params=params,
        pos=(truth_nf["stars"][0]["x"], truth_nf["stars"][0]["y"]))
    d = abs(r_f.strehl - r_nf.strehl)
    check("S4a flat-consistency |dS| <= 0.002", r_f.ok and r_nf.ok and d <= 0.002,
          f"with-flat={r_f.strehl:.4f} without-flat={r_nf.strehl:.4f} d={d:.4f}")

    # --- S4d donor ladder: needs build_epsf (engine-dependent)
    for raw, truth in synth.build_s4_ladder(params):
        expected = truth["expected_epsf_tag"]
        work = engine.sigma_filter3(engine.reduce_frame(raw, flat=flat))
        try:
            epsf = engine.build_epsf(work, params)
        except NotImplementedError as e:
            skip(f"S4d ladder tag == {expected}", str(e))
            continue
        check(f"S4d ladder tag == {expected}", epsf.tag == expected,
              f"got {epsf.tag!r}, note={epsf.note!r}")

    # --- WP-1d AC1: S2 donor frame builds usable=True, tag=strict at
    # EACH sr -- cheap (9-star build_epsf per sr), so CI-wired like S4d
    # rather than gated behind --full.
    for sr in (0.15, 0.30, 0.60):
        raw_d, truth_d = synth.build_s2_donor_frame(params, sr)
        work_d = engine.sigma_filter3(engine.reduce_frame(raw_d, flat=flat))
        try:
            epsf_d = engine.build_epsf(work_d, params)
        except NotImplementedError as e:
            skip(f"WP-1d: S2 donor frame usable/strict at sr={sr}", str(e))
            continue
        check(f"WP-1d: S2 donor frame usable/strict at sr={sr}",
              epsf_d.usable and epsf_d.tag == "strict",
              f"usable={epsf_d.usable} tag={epsf_d.tag!r} "
              f"window={truth_d['donor_peak_window_adu']} note={epsf_d.note!r}")

    if not full:
        return

    # --- S4b saturation: real check, no engine dependency
    raw_b, truth_b = synth.build_s4_saturation(params)[0]
    sat_star = next(s for s in truth_b["stars"] if "saturated" in s["role"])
    iy, ix = int(round(sat_star["y"])), int(round(sat_star["x"]))
    peak = float(raw_b[iy - 2:iy + 3, ix - 2:ix + 3].max())
    check("S4b saturated stars hard-clipped at saturation_adu",
          abs(peak - truth_b["saturation_adu"]) < 1e-6,
          f"peak={peak} saturation_adu={truth_b['saturation_adu']}")

    # --- S4c wide camera: WP-1d AC2 asserts usable=True at each
    # oversample (donor flux/geometry now scaled for the wide window --
    # see build_s4_wide's docstring). `tag` itself is NOT asserted: what
    # "adequate at oversample 2" means beyond usable -- shape residual?
    # epsf_strehl accuracy? -- still isn't specified (WP-2 handoff: "I
    # own the assertions"), so tag/delta/etc are printed as [info] for
    # Opus to judge, same deliberate slot as before.
    wide_params = synth._derive_params(params, camname="wide")
    for raw_w, truth_w in synth.build_s4_wide(params):
        contrast = truth_w["sr_contrast_group"]["contrast_mag"]
        work_w = engine.sigma_filter3(engine.reduce_frame(raw_w, flat=flat))
        for oversample in (2, 4):
            try:
                epsf_w = engine.build_epsf(work_w, wide_params,
                                          oversample=oversample)
            except NotImplementedError as e:
                skip(f"WP-1d: S4c wide usable, contrast={contrast} "
                     f"oversample={oversample}", str(e))
                continue
            check(f"WP-1d: S4c wide usable, contrast={contrast} "
                  f"oversample={oversample}", epsf_w.usable,
                  f"tag={epsf_w.tag!r} note={epsf_w.note!r}")


# ------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="also run S2/S3 and S4b/S4c (slow, not CI-wired)")
    args = parser.parse_args()

    t_start = time.time()
    for name, fn in (("S1", s1_checks), ("S4", lambda: s4_checks(args.full)),
                     ("S5", lambda: s5_checks(args.full)),
                     ("S6", s6_checks)):
        t0 = time.time()
        fn()
        print(f"  ({name} section: {time.time() - t0:.1f}s)\n")

    if args.full:
        for name, fn in (("S2", s2_checks), ("S3", s3_checks)):
            t0 = time.time()
            fn()
            print(f"  ({name} section: {time.time() - t0:.1f}s)\n")

    print(f"total runtime {time.time() - t_start:.1f}s")
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("\npsf_fit_model: all checks passed")


if __name__ == "__main__":
    main()
