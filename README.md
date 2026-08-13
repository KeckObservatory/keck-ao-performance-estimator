# Keck AO Performance Estimator

Estimate W. M. Keck Observatory adaptive-optics performance — NGS, single-beacon
LGS, and LTAO **Strehl** and **FWHM** at a chosen science wavelength — across a
night from Mauna Kea MASS/DIMM seeing, using a semi-analytical error budget. A
command-line tool and a PyQt GUI share one engine.

Developed by the **W. M. Keck Observatory (WMKO) AO group**; current maintainer
**Eduardo Marin** (W. M. Keck Observatory).

## Install

> **Not yet on PyPI.** Install from a clone of this repository for now.

Clone and install into a Python environment (a virtualenv or conda env is
recommended):

```bash
git clone https://github.com/KeckObservatory/keck-ao-performance-estimator.git
cd keck-ao-performance-estimator

# with the GUI (pulls in a Qt binding)
pip install ".[gui]"

# CLI / engine only
pip install .
```

Both provide the `keck-ao-estimator` and `keck-ao-estimator-gui` commands. Use
`pip install -e ".[gui]"` for an editable/development install.

Prefer not to install? You can run it in place from the clone after installing
the dependencies:

```bash
pip install -r requirements.txt      # numpy, scipy, matplotlib, astropy, pillow, PyQt6
export PYTHONPATH=src:sr_estimator   # src/ for the package, sr_estimator/ for qtcompat
python3 -m keck_ao_estimator.gui.app       # GUI
python3 -m keck_ao_estimator.cli --help    # CLI
```

You can also install straight from GitHub without cloning:

```bash
pip install "keck-ao-performance-estimator[gui] @ git+https://github.com/KeckObservatory/keck-ao-performance-estimator"
```

Pin a release with `...keck-ao-performance-estimator@v1.0.0` at the end of the
URL. If the tool is later published to PyPI, installation will simply be
`pip install "keck-ao-performance-estimator[gui]"`.

## Usage

```bash
# GUI
keck-ao-estimator-gui

# CLI — e.g. a night's K-band LGS Strehl on K1
keck-ao-estimator --telescope K1 --target --ra 17h45m40s --dec -29d00m28s \
    --dimm 20260525_dimm.dat --mass 20260525_mass.dat --masspro 20260525_masspro.dat

keck-ao-estimator --version
keck-ao-estimator --help
```

## Features

- NGS / single-beacon LGS / LTAO Strehl and FWHM timelines at any science
  wavelength, with the target's airmass overlaid.
- Field-of-regard performance map, with a DSS/2MASS survey or a local FITS
  (incl. multi-extension mosaics, e.g. GSAOI) as a backdrop and an OSIRIS/NIRC2
  science frame inscribed at its true angular size.
- **Measured** Strehl, FWHM and wavefront error from a NIRC2 or OSIRIS frame —
  a port of the summit IDL Strehl widget, byte-faithful to it on the default
  path — measured either on one star or across the field, and compared against
  the predicted value for the same instant and wavelength.
- Crowded-field measurement (opt-in): σ-clipped or hand-picked sky, and
  **PSF-fit neighbour subtraction**, which builds the frame's own empirical PSF
  and subtracts fitted neighbours from inside the photometry aperture. Every
  cleaned measurement reports which way it is likely to be wrong; validated to
  |SR bias| ≤ 0.02 for Strehl ≤ 0.30. *Neighbour subtraction is still
  developmental*: on real NIRC2 frames tested so far the empirical PSF has too
  few usable donor stars, so it declines and leaves the default number
  untouched. See the bundled KAON 1556 GUI manual (`keck_ao_docs/`).
- Interactive error-budget (WFE) sliders for what-if analysis; modified budgets
  are flagged in every plot and export.
- Reproducible CLI outputs (CSV + figures), guarded by a byte-identity
  regression harness.
- A user manual and technical note (KAON 1542) ship with the tool and open from
  the GUI **Help** menu.

## Limitations

This is a semi-analytical **estimator** calibrated to specific Keck on-sky data,
not an end-to-end AO simulation. Strehl uses the extended Maréchal approximation
(most reliable at moderate-to-high Strehl); NGS performance is an empirical
K-band fit extrapolated to other bands; some tip-tilt and static/calibration
allocations are not yet fully on-sky validated. Treat outputs as planning
estimates and verify against delivered on-sky performance. See the in-app
**Help ▸ About** for the full list.

## Acknowledgments

The baseline analytical AO error budget used for the LGS modes was developed by
**Richard G. Dekany** (California Institute of Technology).

The W. M. Keck Observatory is operated as a scientific partnership among the
California Institute of Technology, the University of California, and the
National Aeronautics and Space Administration. The Observatory was made possible
by the generous financial support of the W. M. Keck Foundation.

This tool was partially funded by the HAKA project, an NSF Major Research
Instrumentation Program award **AST-2320038**.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
