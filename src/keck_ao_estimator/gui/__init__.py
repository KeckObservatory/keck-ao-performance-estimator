"""GUI subpackage (Qt) for the Keck AO Performance Estimator.

Kept separate from the core engine package: install with the `[gui]` extra
to pull in the Qt binding. This module is the curated public API: it
re-exports every name the historical flat `ao_strehl_gui.py` module used to
expose bare, so existing code that did `import ao_strehl_gui as gui` can do
`import keck_ao_estimator.gui as gui` instead and keep every `gui.NAME`
reference working unchanged.
"""
# ruff: noqa: F401 -- every import in this file is an intentional re-export.
from qtcompat import BINDING

from .._version import APP_NAME, __version__
from ..imaging import (
    _hips2fits_url, sky_image_from_file, sky_image_from_fits, sky_image_from_png,
)
from .about import DOC_BENCH_DIAGRAMS, DOC_TECH_NOTE, DOC_USER_MANUAL, _bundled_doc
from .constants import (
    FIELD_OF_REGARD_RADIUS_ARCSEC, FM_C_CATSTAR, FM_C_MARKER, FM_C_TARGET,
    LOCAL_BACKDROP, NIGHTTIME_FM_COND, WFE_SCALING,
)
from .mainwindow import MainWindow
from .tabs.nighttime import NIGHTTIME_PULL_INTERVAL_MS
from .theme import apply_theme, dark_palette, is_dark, set_cue
