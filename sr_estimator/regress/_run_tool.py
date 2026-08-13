#!/usr/bin/env python3
"""Thin launcher for keck_ao_estimator.cli used by the harness's --local mode.

It suppresses astropy's IERS-staleness guard before importing the engine, so
target (airmass) scenarios can run on a machine whose bundled IERS predictive
table is >30 days old and offline. This changes NOTHING numeric: the target
geometry that comes out is byte-identical to the frozen references (verified),
astropy simply refuses to *compute* it under the freshness guard otherwise.

Not used by the default (strict) harness path, and not imported by the engine
or the GUI -- it exists only to make the regression scenarios runnable here.
"""
import runpy
import sys

from astropy.utils.iers import conf as _iers_conf

# astropy's own documented remedy for "predictive values ... more than 30 days
# old" when offline (see the error text it raises).
_iers_conf.auto_max_age = None

_TOOL_MODULE = sys.argv[1]
sys.argv = [_TOOL_MODULE] + sys.argv[2:]
runpy.run_module(_TOOL_MODULE, run_name="__main__")
