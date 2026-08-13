"""pytest wrapper over the byte-identity regression harness (regress/harness.py).

This is a CHARACTERIZATION test, not a correctness test: it proves the CLI's
numeric/pixel output hasn't silently drifted from the frozen references in
regress/ref/, not that those references are physically right. See
test_correctness_physics.py for independent-ground-truth checks.
"""
import os
import subprocess
import sys

import pytest

from conftest import REGRESS, DATA

# On a genuinely different rendering environment (e.g. a hosted CI runner with
# different installed fonts than the box the PNG references were frozen on),
# bbox_inches="tight" bakes font-metric-dependent text extents into the saved
# pixel dimensions, so an exact pixel/shape PNG compare can legitimately fail
# by a handful of pixels while the CSVs (the actual computed numbers) stay
# byte-exact -- see harness.py's --no-png-pixels for the full rationale. GitHub
# Actions sets CI=true automatically; every other environment (this dev box,
# any contributor's machine) keeps the full strict pixel-compare, which is the
# project's primary safety net and must not be silently loosened elsewhere.
_ON_CI = os.environ.get("CI", "").lower() == "true"


@pytest.mark.slow
def test_harness_local_byte_identity():
    env = dict(os.environ)
    env["SR_HARNESS_DATA"] = DATA
    cmd = [sys.executable, os.path.join(REGRESS, "harness.py"), "check", "--local"]
    if _ON_CI:
        cmd.append("--no-png-pixels")
    r = subprocess.run(
        cmd, cwd=REGRESS, env=env, capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, (
        f"harness --local check failed (byte-identity/pixel-closeness broken):\n"
        f"{r.stdout}\n{r.stderr}"
    )
    assert "-- PASS" in r.stdout, r.stdout
