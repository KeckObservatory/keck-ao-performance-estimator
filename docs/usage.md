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
  |SR bias| ≤ 0.02 for Strehl ≤ 0.30 (S2 bias surface, target region sep ≥
  0.3″ / contrast ≤ 3 mag: 20/20 cells met at Strehl 0.15, 24/24 at 0.30);
  every cleaned measurement logs the expected *direction* of the residual
  error, and above 0.30 it warns that the value should be read as an upper
  bound — the shipped surface currently meets 15/24 of the same cells at
  Strehl 0.60 (worst case +0.04), and degrades further above that; the full
  table is in the private ledger.

  **Engine: field (default) or native.** Two engines solve the
  neighbours, an "Engine" selector next to the checkbox — **field**
  (fieldsolve H2c, default since 2026-09-13, FS-D17) solves every
  catalogued star in the frame together, one simultaneous fit, then
  subtracts every solved component but the target; **native** is the
  original per-target group fit described above (≤ 16 neighbours inside
  the target's own footprint). Both keep every gate and refusal above —
  a field solution that fails to converge is a refusal, logged
  `[psf-clean:field] cleaning refused: field solution did not converge
  (...)`, never a silent fallback, same as any other null outcome; native
  outcomes keep the plain `[psf-clean]` tag. On a real S5-moderate
  battery (94 built fields), field measured lower and flatter bias than
  native: own-set signed median +0.0135 (n 347, 2 of 347 beyond ±0.30)
  against native's +0.0428 (n 248, 16 beyond ±0.30); on the 236 targets
  both cleaned, field's median is +0.0121 against native's +0.0417, a
  74–78 % reduction of the robust-sky bias against native's 54–55 %. The
  field engine's own S2 target-region cells (sep ≥ 0.3″ / contrast ≤ 3
  mag, the same criterion as above): 21/21 met at Strehl 0.15, 24/24 at
  0.30, 14/24 at 0.60 — native measures 20/20, 24/24 and 14/24 on the
  same grid. The cost: about 3.9× native's total time over a field (the
  simultaneous solve is the expensive part, ~3.6 s per frame on a typical
  box, built once per frame and shared across every target in it, same
  as the empirical PSF above). field's own residual still reads slightly
  high (+0.01, D27's OVERESTIMATE direction, its own log wording says so
  explicitly). native's note names two regimes below Strehl 0.30: a small
  UNDERESTIMATE on isolated pairs, and an OVERESTIMATE of order +0.04 on
  crowded fields with 3 or more subtracted neighbours (+0.043 on the same
  battery, from over-subtracted aperture flux). Read whichever engine's
  direction note the log actually printed, not the other engine's.

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

**StarFinder evaluated, not adopted.** The INAF Python StarFinder package
(v0.5.1) was evaluated 2026-09-12 as an alternative PSF source and as an
alternative neighbour-subtraction fitter for the mitigation above. Neither
produced a usable model on real NIRC2 fields, and neither improved on the
shipped fitter where both did produce a model. The shipped PSF-fit
neighbour-subtraction path above is unchanged by this evaluation; the
detailed numbers live in the project's private ledger, not in this repo.

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

## Parallelism

`measure_field()` can spread its per-target measurements, and the field
engine's group-ePSF renders (`solve_field`), across multiple worker
**processes**. On the SR tool's NIRC2 tab, a **Workers** spin box next to
the engine selector shows what this actually resolves to — not the literal
`None` the GUI passes: the default is `$KECK_AO_WORKERS` if that
environment variable is set, otherwise `min(8, cpu_count() // 2)`. The
engine itself enforces a cap of 8 workers for every caller, GUI or
otherwise (`KECK_AO_WORKERS_UNCAPPED=1` is the deliberate, environment-only
override for offline batteries) — the spin box simply stops at the same
cap, it is not a separate GUI-only ceiling. Setting it to 1 runs serially,
in-process, with no worker pool at all — bit-identical to every result
produced before this feature existed. The choice round-trips through the
usual config save/load.

**What this release parallelizes:** per-target `measure_strehl` calls in
the "Measure field" flow; the field engine's group-ePSF renders inside
`solve_field` (the single largest parallel saving in the engine); and,
for developers, `--workers` in `psf_fit_model.py` and the FS-E2 battery
driver's native stage. All are embarrassingly parallel and
order-independent (see the note on `EmpiricalPsf.at()` below).

**What it does NOT parallelize:** `build_epsf` and the field engine's own
Gauss-Seidel simultaneous-solve sweeps are serial **by design** — each is
one sequential computation, not a set of independent per-target jobs, so
there is no per-task work to distribute regardless of the worker count.
(The group-ePSF renders and the per-target measurements are the
bandwidth-bound ones: real, parallel work whose speedup is limited by
memory bandwidth, not by design.) Together `build_epsf` and the sweeps
account for roughly 41% of a warm, psf_clean field-engine call measured
serially — about 55% of the same call's wall time once the parallel
portions are sped up on 8 workers, since shrinking the parallel part
raises the serial remainder's share of the (now shorter) total. In the
GUI, this shows up as the field ePSF and field solution stages always
running before the (optionally parallel) per-target loop, each still
logged as its own stage line.

**Why parallel per-target order doesn't change the answer:** `EmpiricalPsf.
at()` weights donor stars at the centre of its 1-arcsec cache bin rather
than favouring whichever request reaches it first, so which worker happens
to finish first no longer changes a downstream star's cleaned result —
without that fix, parallelizing the per-target loop would have made
results depend on scheduling, which is exactly what this release avoids.

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
