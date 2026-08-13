#!/usr/bin/env python3
# =============================================================================
#  regression harness for keck_ao_estimator.cli (the AO Strehl estimator CLI)
#
#  Purpose: the tool's outputs (CSV and PNG) are byte-deterministic, so any
#  refactor -- in particular the GUI engine refactor -- must reproduce the
#  frozen reference outputs EXACTLY. A mismatch means a silently changed
#  number, which is the one failure mode this tool must never have.
#
#  Usage:
#     python3 harness.py freeze    # run all scenarios, store refs in ./ref
#     python3 harness.py check     # re-run all scenarios, byte-compare to ref
#     python3 harness.py check --local
#                                  # same, but for a machine that is NOT the
#                                  # reference box: CSVs (the numbers) must still
#                                  # be byte-identical, PNGs are compared by
#                                  # pixel content within a tight tolerance
#                                  # (matplotlib stamps version/timestamp bytes
#                                  # into the PNG container that differ per host
#                                  # even when the rendered pixels are identical),
#                                  # and astropy's IERS staleness guard is
#                                  # suppressed so target scenarios can run.
#     python3 harness.py check --local --no-png-pixels
#                                  # same CSV byte-exact check, but PNGs are only
#                                  # sanity-checked (valid image, dimensions within
#                                  # a loose tolerance of the reference) rather
#                                  # than pixel-compared. For a genuinely
#                                  # different RENDERING environment (e.g. a CI
#                                  # runner with different installed fonts):
#                                  # bbox_inches="tight" bakes font-metric-
#                                  # dependent text extents into the saved pixel
#                                  # dimensions, so an exact-shape pixel compare
#                                  # can legitimately fail by a handful of pixels
#                                  # with the numbers (the CSVs) still exactly
#                                  # right. Do NOT use this to paper over a real
#                                  # box-specific mismatch -- prefer plain
#                                  # --local (this script's main safety net) on
#                                  # any machine capable of it.
#
#  Scenarios cover: both telescopes, tomography on/off, --target windows,
#  graceful MASS-less degradation, high airmass, the legacy budget, and a
#  non-default NGS magnitude + seeing law.
# =============================================================================
import subprocess, sys, os, hashlib, shutil

# Pixel-content tolerance for --local PNG comparison. The mean absolute per-
# channel pixel difference between a frozen reference and a fresh local render.
# Empirically the two renders are pixel-identical (MAD 0.000) or differ only by
# sub-count antialiasing rounding; any real change to a plotted number moves
# many pixels by a large amount, so this threshold passes identical renders
# while still catching a genuinely changed figure.
PNG_MAD_TOL = 0.002

# --no-png-pixels: how far a PNG's dimensions may drift from the reference
# (relative, per axis) before it's treated as a real mismatch rather than
# renderer-environment jitter. Observed cross-machine drift (this dev box vs
# a GitHub Actions Ubuntu runner, same matplotlib major version) was <1% per
# axis; 3% leaves comfortable margin while still catching a genuinely wrong
# figure (a dropped legend/panel moves dimensions far more than that).
PNG_SHAPE_RTOL = 0.03

HERE = os.path.dirname(os.path.abspath(__file__))
# Invoked as `python3 -m keck_ao_estimator.cli` (a module, not a script path)
# now that the engine lives in the src/ package; SRC goes on PYTHONPATH for
# the subprocess since the package is not installed. See _run_tool.py for the
# --local path's equivalent runpy.run_module() invocation.
TOOL_MODULE = "keck_ao_estimator.cli"
SRC = os.path.join(HERE, "..", "..", "src")
SUBPROCESS_ENV = dict(os.environ)
SUBPROCESS_ENV["PYTHONPATH"] = SRC + os.pathsep + SUBPROCESS_ENV.get("PYTHONPATH", "")
# Directory holding the night .dat files. Default: the in-repo
# regress/data/ copies (fetched from the MKWC archive, verified
# byte-identical to the frozen refs -- see LOCAL_NOTES.md). The old
# default was the web-chat sandbox's /mnt/user-data/uploads; that era
# is over (Eduardo 2026-07-25: web chat is report-generation only --
# nothing in the GUI/engine/harness may depend on it). Override with
# SR_HARNESS_DATA to point elsewhere. Paths are quoted at the point
# of interpolation, so a data dir with spaces is fine.
UP   = os.environ.get("SR_HARNESS_DATA", os.path.join(HERE, "data"))
MAY  = (f'--dimm "{UP}/20260525_dimm.dat" --mass "{UP}/20260525_mass.dat" '
        f'--masspro "{UP}/20260525_masspro.dat"')
JUL  = f'--dimm "{UP}/20260707_dimm.dat"'

SCENARIOS = {
    # name: CLI args (without --out/--force)
    "A_may_K1_full":     f"{MAY} --telescope K1",
    "B_may_K2_full":     f"{MAY} --telescope K2",
    "C_may_K1_target":   (f"{MAY} --telescope K1 --target --target-name HD141569 "
                          f"--ra 15h49m57.7s --dec=-03d55m16s --window 09:00-10:00"),
    "D_jul_K2_himass":   (f"{JUL} --telescope K2 --target --target-name HIP_88553 "
                          f"--ra 18h04m53.72s --dec=-44d39m52.8s "
                          f"--window 23:45-23:52 --ngs-bright 8.2"),
    "E_may_K1_legacy":   f"{MAY} --telescope K1 --legacy-budget",
    "F_may_K1_gaussian": f"{MAY} --telescope K1 --ngs-seeing-law gaussian",
}
# outputs produced per scenario (relative to the --out stem)
SUFFIXES = [".png", ".csv", "_terms.png", "_all.png"]


def outputs_for(workdir, name):
    """All files the scenario produced, in its own subdirectory: the --out
    stem outputs plus the UT-date-named companion figures."""
    d = os.path.join(workdir, name)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d))


def run_scenario(name, args, workdir, local=False):
    d = os.path.join(workdir, name)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, name + ".png")
    # --local routes through _run_tool.py, which suppresses the IERS staleness
    # guard before importing the engine (numbers are unaffected -- see that
    # file). The strict path invokes the tool directly, exactly as before.
    # paths are QUOTED: this box's user dir contains a space ("Eduardo
    # Marin"), and shell=True splits an unquoted script/--out path there
    # (first seen 2026-07-23 -- every scenario "RUN FAILED" trying to open
    # 'C:\Users\Eduardo'). Pass SR_HARNESS_DATA space-free (e.g. the 8.3
    # short form) since the scenario arg strings interpolate it unquoted.
    if local:
        runner = f'python3 "{os.path.join(HERE, "_run_tool.py")}" {TOOL_MODULE}'
    else:
        runner = f"python3 -m {TOOL_MODULE}"
    cmd = f'{runner} {args} --out "{out}" --force'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       env=SUBPROCESS_ENV)
    if r.returncode != 0:
        print(f"  {name}: RUN FAILED\n{r.stderr[-800:]}")
        return False
    return True


def sha(path):
    if not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def png_mad(ref, test):
    """(mad, detail). mad is the mean absolute per-channel pixel difference
    between two PNGs, or None if either is missing / unreadable / a different
    shape (treated as a failure); detail explains which, and -- for a shape
    mismatch -- the actual two shapes (e.g. a different font/matplotlib
    renderer on another machine changing a bbox_inches="tight" save size is a
    very different failure mode than a genuinely wrong plot)."""
    if not (ref and test and os.path.exists(ref) and os.path.exists(test)):
        return None, "missing file"
    import numpy as np
    import matplotlib.image as mpimg
    try:
        a = mpimg.imread(ref).astype("float64")
        b = mpimg.imread(test).astype("float64")
    except Exception as e:
        return None, f"unreadable ({e})"
    if a.shape != b.shape:
        return None, f"shape changed: ref {a.shape} vs test {b.shape}"
    return float(np.mean(np.abs(a - b))), None


def png_shape_sanity(ref, test):
    """(ok, detail): loose cross-environment PNG check for --no-png-pixels --
    the test image must exist, be a valid, readable image, and have the same
    number of channels with each spatial dimension within PNG_SHAPE_RTOL of
    the reference. No pixel-content comparison (a shape difference of even one
    pixel makes raw pixel alignment meaningless, so there is nothing
    meaningful left to MAD-compare once the strict path has been declined)."""
    if not (ref and test and os.path.exists(ref) and os.path.exists(test)):
        return False, "missing file"
    import matplotlib.image as mpimg
    try:
        a = mpimg.imread(ref)
        b = mpimg.imread(test)
    except Exception as e:
        return False, f"unreadable ({e})"
    if len(a.shape) != len(b.shape) or a.shape[2:] != b.shape[2:]:
        return False, f"shape changed: ref {a.shape} vs test {b.shape}"
    for ra, rb in zip(a.shape[:2], b.shape[:2]):
        if abs(ra - rb) > PNG_SHAPE_RTOL * ra:
            return False, (f"dimensions drifted beyond {PNG_SHAPE_RTOL:.0%}: "
                           f"ref {a.shape} vs test {b.shape}")
    return True, f"shape ref {a.shape} vs test {b.shape} (within tolerance)"


def files_match(fname, ref_path, test_path, local, png_pixels=True):
    """Return (ok, detail). CSVs must always be byte-identical. PNGs are
    byte-identical in strict mode; in --local mode either pixel-close
    (png_pixels=True, the default) or shape-sanity-only (png_pixels=False --
    see png_shape_sanity, intended for a rendering environment genuinely
    different from the one the references were frozen on, e.g. CI)."""
    if local and fname.lower().endswith(".png"):
        if not png_pixels:
            return png_shape_sanity(ref_path, test_path)
        mad, detail = png_mad(ref_path, test_path)
        if mad is None:
            return False, detail
        return (mad <= PNG_MAD_TOL), f"MAD={mad:.6f}"
    rs = sha(ref_path or "/nonexistent")
    ts = sha(test_path or "/nonexistent")
    ok = (rs == ts and rs is not None)
    return ok, (f"ref {'missing' if rs is None else rs[:10]}"
                f" vs test {'missing' if ts is None else ts[:10]}")


def freeze(local=False):
    """Regenerate ./ref. --local routes through _run_tool.py (IERS staleness
    guard suppressed; numerically neutral -- see that file) so target scenarios
    can be frozen on a box with a stale bundled IERS table."""
    ref = os.path.join(HERE, "ref")
    shutil.rmtree(ref, ignore_errors=True)
    os.makedirs(ref)
    ok = True
    for name, args in SCENARIOS.items():
        if not run_scenario(name, args, ref, local=local):
            ok = False
            continue
        present = [os.path.basename(p) for p in outputs_for(ref, name)
                   if os.path.exists(p)]
        print(f"  froze {name}: {len(present)} files")
    print("FREEZE " + ("OK" if ok else "INCOMPLETE"))
    return 0 if ok else 1


def check(local=False, png_pixels=True):
    ref = os.path.join(HERE, "ref")
    test = os.path.join(HERE, "test")
    shutil.rmtree(test, ignore_errors=True)
    os.makedirs(test)
    n_pass = n_fail = 0
    for name, args in SCENARIOS.items():
        if not run_scenario(name, args, test, local=local):
            n_fail += 1
            continue
        refs  = {os.path.basename(p): p for p in outputs_for(ref, name)}
        tests = {os.path.basename(p): p for p in outputs_for(test, name)}
        for fname in sorted(set(refs) | set(tests)):
            ok, detail = files_match(fname, refs.get(fname), tests.get(fname),
                                     local, png_pixels=png_pixels)
            if ok:
                n_pass += 1
            else:
                n_fail += 1
                print(f"  MISMATCH {name}/{fname}  ({detail})")
    if not local:
        kind = "byte-identical"
    elif png_pixels:
        kind = "CSV byte-exact + PNG pixel-close"
    else:
        kind = "CSV byte-exact + PNG shape-sanity (no pixel compare)"
    print(f"CHECK ({kind}): {n_pass} files OK, {n_fail} mismatches "
          + ("-- PASS" if n_fail == 0 else "-- FAIL"))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    argv = sys.argv[1:]
    local = "--local" in argv
    png_pixels = "--no-png-pixels" not in argv
    argv = [a for a in argv if not a.startswith("--")]
    mode = argv[0] if argv else "check"
    if mode == "freeze":
        sys.exit(freeze(local=local))
    sys.exit(check(local=local, png_pixels=png_pixels))
