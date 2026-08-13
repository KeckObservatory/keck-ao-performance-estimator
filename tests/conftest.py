"""pytest configuration shared by every test module.

Puts sr_estimator/ (so `qtcompat` resolves -- the GUI package still imports
it as a flat module) and src/ (so `import keck_ao_estimator as engine` /
`import keck_ao_estimator.gui as gui` work without installing the package)
on sys.path, and forces the offscreen Qt platform for anything that touches
the GUI, so the suite runs headless in CI with no display.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR_ESTIMATOR = os.path.join(REPO_ROOT, "sr_estimator")
REGRESS = os.path.join(SR_ESTIMATOR, "regress")
SRC = os.path.join(REPO_ROOT, "src")
DATA = os.path.join(REGRESS, "data")

for p in (SR_ESTIMATOR, REGRESS, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)
