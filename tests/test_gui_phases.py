"""pytest wrapper over the existing regress/gui_phase*.py + gui_smoke.py end-to-
end scripts, plus the fwhm_model.py physics-contract script.

These are CHARACTERIZATION / behavioural regression tests (each script asserts
specific, documented behaviour of the GUI and engine as it was built) rather
than independent-ground-truth correctness tests -- see
test_correctness_physics.py for the latter. Each script is run as its own
subprocess (matches how it has always been verified, and keeps Qt/engine state
from one script leaking into the next) with pytest reporting pass/fail per
script and showing captured output on failure.
"""
import glob
import os
import subprocess
import sys

import pytest

from conftest import REGRESS

SCRIPTS = sorted(
    glob.glob(os.path.join(REGRESS, "gui_phase*.py"))
    + [os.path.join(REGRESS, "gui_smoke.py")],
    key=lambda p: os.path.basename(p),
)


def _looks_like_crash(returncode, stdout, stderr):
    """fieldsolve FS-OPEN-4: best-effort detection of a process CRASH
    (segfault / access violation under headless Qt) as opposed to a real
    assertion failure, which every script wired through `_run()` reports
    via `sys.exit(1)` after printing a `[FAIL]` / `FAILURE(S)` line (or,
    for the gui_phase*.py scripts, after printing whatever `[ok]` lines
    it got through) -- always to STDOUT.

    - POSIX: a NEGATIVE returncode means the process was killed by a
      signal (e.g. -11 = SIGSEGV), which is exactly what was observed
      for gui_phase12.py on hosted GitHub Actions runners (WP-1 CI run
      34741109947, first attempt). This alone covers the observed case.
    - Windows crashes surface differently (a large positive exit code,
      no signal), so as a portable fallback: a nonzero returncode with
      BOTH stdout and stderr empty is also treated as a crash -- the
      process died before it could report anything at all.
    - Deliberately NOT using "stderr empty" alone (the literal
      suggestion in the brief): every script's own sys.exit(1) failure
      path ALSO has empty stderr, since everything is printed to
      stdout. Using stderr alone would retry genuine assertion
      failures, which this must not do -- only empty STDOUT (nothing
      reported at all) is a safe proxy for "the process never got to
      report a real failure".
    """
    if returncode < 0:
        return True
    if returncode != 0 and not (stdout or "").strip() and not (stderr or "").strip():
        return True
    return False


def _run(script):
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    # fieldsolve FS-OPEN-4: CI now runs on every push to the public repo
    # (fieldsolve P0-1), so hosted-runner slowness gets a real ceiling
    # instead of painting unrelated pushes red -- double the timeout
    # under GITHUB_ACTIONS, unchanged locally.
    timeout = 360 if os.environ.get("GITHUB_ACTIONS") else 180

    def _once():
        return subprocess.run(
            [sys.executable, script], cwd=REGRESS, env=env,
            capture_output=True, text=True, timeout=timeout,
        )

    r = _once()
    if r.returncode != 0 and _looks_like_crash(r.returncode, r.stdout, r.stderr):
        # One automatic retry, crash signature ONLY (FS-OPEN-4): a real
        # assertion failure or meaningful stderr fails on the first
        # attempt, no blanket retry.
        r = _once()
    assert r.returncode == 0, (
        f"{os.path.basename(script)} failed:\n{r.stdout}\n{r.stderr}"
    )


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.parametrize("script", SCRIPTS, ids=[os.path.basename(s) for s in SCRIPTS])
def test_gui_phase_script(script):
    _run(script)


@pytest.mark.slow
def test_fwhm_physics_contract_script():
    _run(os.path.join(REGRESS, "fwhm_model.py"))


@pytest.mark.slow
def test_psf_fit_model_script():
    _run(os.path.join(REGRESS, "psf_fit_model.py"))


@pytest.mark.slow
def test_field_solve_model_script():
    """fieldsolve WP-1 skeleton: four named skip() stubs today (awaiting
    Opus's O1 field_solve.py), still asserting the script itself runs
    clean and exits 0 -- see field_solve_model.py's own docstring."""
    _run(os.path.join(REGRESS, "field_solve_model.py"))


@pytest.mark.slow
def test_parallel_model_script():
    """parallel project: every optimization commit on the engine proves it
    changes no number -- see parallel_model.py."""
    _run(os.path.join(REGRESS, "parallel_model.py"))


@pytest.mark.slow
def test_fwhm_srtool_model_script():
    """The 4th FWHM convention must stay the SR tool's own process --
    see fwhm_srtool_model.py."""
    _run(os.path.join(REGRESS, "fwhm_srtool_model.py"))


@pytest.mark.slow
def test_vignetting_model_script():
    """TSS reachability/vignetting (KAON 913) + the outer-scale tilt ceiling
    (KAON 1318) -- see vignetting_model.py."""
    _run(os.path.join(REGRESS, "vignetting_model.py"))


@pytest.mark.slow
def test_onaxis_sr_one_model_script():
    """The field map, the summary stats and the SR tool's Predicted box must
    stay ONE model with three input sets -- see onaxis_sr_model.py."""
    _run(os.path.join(REGRESS, "onaxis_sr_model.py"))
