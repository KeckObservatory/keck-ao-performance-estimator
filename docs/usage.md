# Usage

## GUI

```bash
keck-ao-estimator-gui
```

The GUI is organised as tabs that feed one shared engine:

- **Data** — load the night's MASS/DIMM/MASSpro seeing (from a local file or a
  MKWC fetch).
- **Target** — RA/Dec (hms/dms, colon-separated, or decimal degrees), an
  optional arcsec target offset, and a saved-target list.
- **NGS / LGS / WFE** — guide-star magnitudes, the laser/LTAO configuration, the
  tip-tilt sensor (STRAP or TRICK), and the error-budget sliders.
- **Prediction** — a hypothetical-conditions scenario.
- **SR tool** — measure Strehl, FWHM and wavefront error directly from a NIRC2
  or OSIRIS frame and compare them against the prediction for that instant.

Plot tabs show the **Timeline**, the **Field map**, and the **Error terms**.

### SR tool

**Measure** re-measures the frame already loaded (change an aperture or a sky
setting and re-run it directly); it measures the numbered FIRST IMAGE / N
IMAGES sequence when nothing is loaded, or as soon as you touch either spin.

Load a frame and the tool measures the brightest star, or "Measure field" to
find and measure the N brightest and plot Strehl (or FWHM) against field
position — the observational twin of the Field-map tab's model map. Stars that
are saturated, measure outside a physical Strehl of (0, 1], or have a broken
radial profile are rejected with a per-star reason in the log.

With the default settings the numbers are byte-identical to the summit IDL
widget. Three opt-in mitigations handle crowded fields, each logged whenever it
changes anything:

- **Robust sky (σ-clip)** and **Pick sky** — replace the annulus mean sky, which
  a neighbour in the annulus drags high (this is how a dense field yields
  nonsense or negative Strehl).
- **PSF-fit neighbour subtraction** (**"Measure field" only**) — for the case
  those two cannot reach, a neighbour *inside* the photometry aperture. The frame's own empirical PSF is
  built from its isolated stars and fitted simultaneously to the target and its
  neighbours; the neighbours are subtracted before measuring. Validated to
  |SR bias| ≤ 0.02 for Strehl ≤ 0.30; every cleaned measurement logs the
  expected *direction* of the residual error, and above 0.30 it warns that the
  value should be read as an upper bound.

  **Developmental.** The empirical PSF needs at least 4 isolated, well-exposed
  donor stars, and on the real NIRC2 frames tested so far — sparse standard
  fields through the Galactic Centre — it has not found them. It then reports
  `uncalibrated`, skips cleaning, and the default number stands. Enabling the
  option on such a frame changes nothing and says so in the log. It can also
  reject a model it has already built, if that model predicts more neighbour
  light than a target's aperture physically contains.

  Note that on any star cleaning succeeds on, **Robust sky is ignored** — both
  remove the same neighbour light, and using them together doubles the residual
  bias. The log says so per star.

  The checkbox applies to **"Measure field"** and to clicks made afterwards on
  the same frame (which reuse the model it built) — a plain **Measure** skips
  it, and says so in the log. Building the field's empirical PSF costs seconds,
  and doing it per frame just to look at one star delayed every single-frame
  measurement for a correction that usually declined to run.

See the bundled KAON 1556 GUI manual (`keck_ao_docs/`) for the full
description.

### Field map

The field map shows Strehl or FWHM across the field of regard. Where a FWHM
convention is offered, **"as the SR tool reads it" leads and is the default** —
it is what the SR tool itself measures, so it is the one to compare against a
measured number. On it you can:

- Load a **survey backdrop** (DSS/2MASS) or a local FITS science frame.
- **Drop science targets** (right-click) and read each one's predicted
  performance at its field position.
- Place the **laser / TT star / NGS star** at a clicked point (right-click).
- Toggle **TSS vignetting** to see where the K1 tip-tilt sensor can actually
  be placed: a solid ring at the radius reachable at every rotator angle, a
  dashed one at the stage's longest reach (the band between them depends on
  the bench angle, which the app does not carry), and modelled vignetting
  contours. Guide-star ranking uses the same model — stars outside the stage
  travel are excluded, and vignetting is charged as lost flux. It is a
  **model** reconstructed from KAON 913, not the observatory's measured map.
- **Load a guide-star catalogue** (GSC 2.4, 2MASS, UCAC4, PanSTARRS DR2, Gaia
  DR2): candidate stars are plotted **sized by their brightness in the tip-tilt
  sensing band** (bigger = brighter = better guide stars; stars fainter than the
  sensor's practical limit are drawn hollow). **Left-click** a star to inspect
  its magnitudes; **right-click** it to set it as the TT or NGS guide star,
  which fills in its position and its magnitude — the catalogue's own band if it
  has it, otherwise an estimate of the sensing band from a published colour
  transform (flagged as an estimate).

## CLI

```bash
# a night's K-band LGS Strehl on K1
keck-ao-estimator --telescope K1 --target --ra 17h45m40s --dec -29d00m28s \
    --dimm 20260525_dimm.dat --mass 20260525_mass.dat --masspro 20260525_masspro.dat

keck-ao-estimator --version
keck-ao-estimator --help
```

The CLI writes a CSV table and figures. These outputs are the frozen reference
guarded by the byte-identity regression harness (see
[Development](development.md)).
