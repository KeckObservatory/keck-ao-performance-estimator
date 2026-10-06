#!/usr/bin/env python3
"""Keep PyAO's standalone Strehl tool (kaotools/strehl_tool) in sync with
this package's measurement engine.

PyAO carries a VERBATIM copy of the engine modules in
``kaotools/strehl_tool/_vendored_estimator/``. Engine changes are made HERE
first and then copied over with this script; the vendored copy is never
edited in PyAO (Eduardo 2026-10-06: "keep both of these tools in sync").

    python sr_estimator/sync_pyao_strehl_tool.py --check ~/PyAO-1693
    python sr_estimator/sync_pyao_strehl_tool.py --copy  ~/PyAO-1693

--check exits 1 and names every file that differs; --copy overwrites the
vendored files with this checkout's versions (review with git diff in PyAO,
then run its tests and .github/check_health.py).

Deliberate differences, never copied:
- ``__init__.py`` and ``constants.py``: PyAO keeps a measurement-only
  subset (no prediction code).
- ``data/superflat.fits.gz``: PyAO stores the flat RICE-compressed to fit
  its 1 MB file limit (golden-frame SR within 2e-4 of this full-precision
  flat).

The two GUIs are separate code; GUI features are mirrored by hand (see the
CHANGELOG entries that mention PyAO).
"""
import argparse
import filecmp
import os
import shutil
import sys

ENGINE_MODULES = (
    "image_strehl.py", "nirc2.py", "nirc2_psf.py", "osiris.py", "epsf.py",
    "psf_fit.py", "field_solve.py", "field_stats.py", "ee_correction.py",
    "parallel.py", "frame_watch.py", "keck_network.py", "series_stats.py",
)
DATA_FILES = ("supermask.fits.gz",)
VENDOR_DIR = os.path.join("kaotools", "strehl_tool", "_vendored_estimator")
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "src", "keck_ao_estimator")


def pairs(pyao_root):
    """(source, vendored copy) for every synced file."""
    dst = os.path.join(pyao_root, VENDOR_DIR)
    for name in ENGINE_MODULES:
        yield os.path.join(SRC, name), os.path.join(dst, name)
    for name in DATA_FILES:
        yield (os.path.join(SRC, "data", name),
               os.path.join(dst, "data", name))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--copy", action="store_true")
    ap.add_argument("pyao_root", help="PyAO checkout (repo root)")
    a = ap.parse_args(argv)
    if not os.path.isdir(os.path.join(a.pyao_root, VENDOR_DIR)):
        sys.exit(f"no {VENDOR_DIR} under {a.pyao_root}")
    drift = []
    for src, dst in pairs(a.pyao_root):
        same = os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False)
        if same:
            continue
        rel = os.path.relpath(dst, a.pyao_root)
        if a.copy:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"copied  {rel}")
        else:
            drift.append(rel)
            print(f"DIFFERS {rel}")
    if a.check:
        print("in sync" if not drift else f"{len(drift)} file(s) out of sync")
        return 1 if drift else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
