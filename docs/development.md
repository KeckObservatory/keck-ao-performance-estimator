# Development

## Layout

The tool is a single installable package, `keck_ao_estimator`, under `src/`:

- **Engine** — `src/keck_ao_estimator/` (`constants`, `atmosphere`, `psf`,
  `tiptilt`, `ngs`, `budget`, `geometry`, `fieldmap`, `catalogs`,
  `config`, `pipeline`, `plots`, `export`, `cli`). Qt-free and headless.
  Image measurement lives in `image_strehl` (the port of the summit IDL
  Strehl widget) with `nirc2`/`osiris` for the per-instrument frame
  parameters, `nirc2_psf` for the diffraction-limited reference, and
  `ee_correction` for the encircled-energy aperture correction.
  Crowded-field measurement adds `epsf` (empirical PSF construction) and
  `psf_fit` (simultaneous neighbour fitting and subtraction); both are
  opt-in and neither is reachable from the default measurement path.
  `vignetting` models the K1 tip-tilt stage's reachability and vignetting
  (KAON 913) for guide-star ranking and the field map — explicitly a model,
  not the observatory's measured map; see its module docstring.
- **GUI** — `src/keck_ao_estimator/gui/` (its own subpackage): `mainwindow`,
  `app`, `widgets`, `workers`, `constants`, and one mixin per tab under
  `gui/tabs/` composed into the main window.
- `sr_estimator/` holds the regression harness and its data (`regress/`), the
  bundled KAON PDFs (`keck_ao_docs/`), and the standalone Qt-binding shim
  (`qtcompat.py`).

The engine and GUI import nothing from each other's private internals; the
package `__init__` files are the curated public API.

## Tests

Run the whole suite with:

```bash
pip install -e ".[gui,dev]"
QT_QPA_PLATFORM=offscreen pytest -q
```

There are three complementary layers:

- **Correctness tests** (`tests/test_correctness_physics.py`) — assert actual
  physical relationships (Strehl/FWHM monotonicity, Maréchal exactness,
  budget/band swaps, airmass geometry). They are written to fail if the model is
  wrong, not merely to exercise the code.
- **Regression harness** (`tests/test_regression.py` +
  `sr_estimator/regress/harness.py`) — re-runs a fixed set of CLI scenarios and
  checks the CSV outputs **byte-for-byte** and the PNG figures pixel-close
  against committed references. Every engine change must keep these identical.
- **GUI behaviour tests** (`tests/test_gui_phases.py` running the
  `sr_estimator/regress/gui_phase*.py` scripts) — drive the real widgets
  headless (offscreen Qt) and assert on the rendered/derived state.

Running the harness directly:

```bash
SR_HARNESS_DATA="$PWD/sr_estimator/regress/data" \
    python3 sr_estimator/regress/harness.py check --local
```

The crowded-field engine has its own model-level harness, which builds
empirical PSFs on synthetic frames and asserts on the recovered Strehl,
the donor ladder, and the refusal paths:

```bash
python3 sr_estimator/regress/psf_fit_model.py
```

Two further model-level scripts guard cross-cutting contracts:

```bash
# the 4th FWHM convention must BE the Measured-SR tab's own process
python3 sr_estimator/regress/fwhm_srtool_model.py
# field map / summary stats / SR tool must stay ONE model, three input sets
python3 sr_estimator/regress/onaxis_sr_model.py
# TSS reachability/vignetting model + the outer-scale open-loop tilt ceiling
python3 sr_estimator/regress/vignetting_model.py
```

Some validation reads proprietary NIRC2 material (IDL sources, flats,
reference frames and goldens). It is never committed; those tests read
`$NIRC2_STREHL_DATA` and skip cleanly when it is unset.

## Lint

```bash
ruff check .
```

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request to `master`:

- a **ruff** lint job, and
- a **pytest** job (installing the headless Qt runtime libraries and running the
  full suite under `QT_QPA_PLATFORM=offscreen`).

Two further workflows are wired but **do not run automatically**:

- `docs.yml` builds this documentation site on every push (to catch build
  breaks); it only **deploys** to GitHub Pages when triggered manually
  (`workflow_dispatch`).
- `publish.yml` builds and publishes the package to PyPI, and runs **only** on a
  published GitHub Release (or manual dispatch). It uses PyPI Trusted Publishing,
  so it stays dormant until that is configured on the PyPI side.

## Building the package

```bash
python3 -m build       # sdist + wheel into dist/
```

The version is read dynamically from `keck_ao_estimator._version.__version__`.
