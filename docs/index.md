# Keck AO Performance Estimator

Estimate W. M. Keck Observatory adaptive-optics performance — NGS, single-beacon
LGS, and LTAO **Strehl** and **FWHM** at a chosen science wavelength — across a
night from Mauna Kea MASS/DIMM seeing, using a semi-analytical error budget. A
command-line tool and a PyQt GUI share one engine.

Developed by the **W. M. Keck Observatory (WMKO) AO group**; current maintainer
**Eduardo Marin** (W. M. Keck Observatory).

## Features

- NGS / single-beacon LGS / LTAO Strehl and FWHM timelines at any science
  wavelength, with the target's airmass overlaid.
- A field-of-regard performance map with a DSS/2MASS survey or a local FITS
  (including multi-extension mosaics, e.g. GSAOI) as a backdrop and an
  OSIRIS/NIRC2 science frame inscribed at its true angular size.
- **Guide-star catalogue lookups** on the field map (GSC 2.4, 2MASS, UCAC4,
  PanSTARRS DR2, Gaia DR2): candidates are plotted sized by their brightness in
  the tip-tilt sensing band; right-click one to set it as the TT/NGS guide star
  (position + magnitude, estimated in the sensing band from published colour
  transforms when the catalogue lacks that band).
- Multiple science targets droppable on the field map, each reading its own
  predicted performance at that field position.
- Interactive error-budget (WFE) sliders for what-if analysis; modified budgets
  are flagged in every plot and export.
- Reproducible CLI outputs (CSV + figures), guarded by a byte-identity
  regression harness.
- A user manual and technical note (KAON 1542) ship with the tool and open from
  the GUI **Help** menu.

## Where to next

- [Install](install.md) — how to get it running.
- [Usage](usage.md) — the CLI and the GUI.
- [Development](development.md) — layout, tests, and the regression harness.

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

GNU General Public License v3.0.
