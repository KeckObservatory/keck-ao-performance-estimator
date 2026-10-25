"""Single-source identity + version for KAPE, the Keck AO Performance
Estimator.

Imported by the engine, the GUI, and the packaging metadata (pyproject.toml
reads __version__ from here), so the version is defined in exactly one place.
"""

__version__ = "1.6.0"

# the name in titles, the About box and the CLI (since 1.6.0); the package
# stays keck_ao_estimator (PyAO vendors it) and the old commands remain
APP_NAME = "KAPE"
APP_FULL_NAME = "Keck AO Performance Estimator"
MAINTAINER = "Eduardo Marin"
ORGANIZATION = "W. M. Keck Observatory (WMKO) AO group"
CONTACT = "W. M. Keck Observatory"
