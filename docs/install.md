# Install

!!! note "Not yet on PyPI"
    Install from a clone of this repository for now. Once the tool is published
    to PyPI, installation will simply be
    `pip install "keck-ao-performance-estimator[gui]"`.

Clone and install into a Python environment (a virtualenv or conda env is
recommended):

```bash
git clone https://github.com/KeckObservatory/keck-ao-performance-estimator.git
cd keck-ao-performance-estimator

# with the GUI (pulls in a Qt binding + astroquery for catalogue lookups)
pip install ".[gui]"

# CLI / engine only
pip install .
```

Both provide the `keck-ao-estimator` and `keck-ao-estimator-gui` commands. Use
`pip install -e ".[gui]"` for an editable / development install.

## Run in place (without installing)

You can run it from the clone after installing the dependencies:

```bash
pip install -r requirements.txt      # numpy, scipy, matplotlib, astropy, pillow, PyQt6
export PYTHONPATH=src:sr_estimator   # src/ for the package, sr_estimator/ for qtcompat
python3 -m keck_ao_estimator.gui.app       # GUI
python3 -m keck_ao_estimator.cli --help    # CLI
```

## Python and Qt

- Requires Python ≥ 3.9.
- The engine/CLI needs only numpy, scipy, matplotlib, astropy, and pillow.
- The GUI needs a Qt binding. `pip install ".[gui]"` pulls in **PyQt6**;
  **PyQt5** also works where PyQt6 is unavailable (a one-line binding shim,
  `qtcompat.py`, selects whichever is present).
- The field map's guide-star catalogue lookups use **astroquery** (Vizier),
  installed with the `[gui]` extra. It is imported lazily, so the CLI, the
  engine, and the offline test suite never require it.
