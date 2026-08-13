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


def _run(script):
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    r = subprocess.run(
        [sys.executable, script], cwd=REGRESS, env=env,
        capture_output=True, text=True, timeout=180,
    )
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
