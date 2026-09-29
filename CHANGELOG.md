# Changelog

All notable changes to the Keck AO Performance Estimator. The project
is released on GitHub only (pin `@v1.3.0` in the install URL).

## Unreleased

### Changed
- **K2 NGS: the 57x57 fit is now the HAKA N53 fit.** The ceiling is 0.747,
  A 0.661, m₀ 13.62 and w 1.58 (it was N49: 0.755 / 0.738 / 13.76 / 1.71).
  N53 adds the UT 2026-09-23 points (HAKA NGS report rev27).
  - At 0.5″ DIMM the K2 NGS Strehl moves by +0.004 at R 8, −0.001 at
    R 12, −0.025 at R 14 and −0.024 at R 15.
  - K1 is unchanged.
  - The regression references for the two K2 scenarios were updated, and
    only their NGS columns changed.
  - A config saved earlier keeps the fit values it saved. **Reset fit**
    loads N53.

### Added
- **K2 NGS WFS mode.** `--ngs-wfs {57x57,29x29}` on the CLI; on the GUI,
  the **K2 NGS WFS** selector on the NGS tab, which shares a row with
  **Reset fit**.
  - `29x29` selects a **preliminary** fit: ceiling 0.614, A 0.661,
    m₀ 14.64, w 1.58.
  - The fit uses four stars from one night (2026-09-23), R 11.7–15.3.
    A and w are tied to 57x57, and the ceiling is capped at 0.965× the
    57x57 ceiling.
  - It crosses 57x57 at R 12.2. It is ×1.5 at R 14 and ×2.6 at R 15, and
    about 17 % lower on bright stars. Outside R 11.7–15.3 it is an
    extrapolation.
  - The mode exists on K2 only: the CLI refuses it on K1, and the GUI
    disables the selector there.
  - A non-default mode is recorded in the CSV provenance
    (`ngs_wfs=29x29(PRELIMINARY_fit)`) and in the run summary.
  - The Gompertz fields still override on top of the selected fit.
  - **NGS fit preview on K2:**
    - The other mode's fit is drawn dashed at 0.5″ K seeing.
    - Its crossover with the active mode is marked ("29x29 better fainter
      than R 12.2").
    - A bar on the R axis shows the range the 29x29 fit has data for, and
      the 29x29 curve is thinner where it is extrapolated.
    - One seeing is enough: A and w are tied between the modes, so the
      seeing term cancels in their ratio and the crossover is R 12.2 at
      any seeing.
    - The title is shorter and no longer clips at the dock's width.
  - Engine: `ngs_strehl(..., ngs_wfs=)`, `ngs_fit_params()`,
    `NGS_PARAMS_K2_WFS`, `NGS_WFS_MODES`, `DEF_NGS_WFS`.
  - Regression test `gui_phase46`.

## 1.3.0 — 2026-09-22

### Added
- LGS tab: **LGS flux** sub-tab with an option to **scale the measurement
  error with the modelled LGS return** at the target's pointing
  (`--lgs-flux-model` on the CLI). The return relative to zenith is
  F = (1/X) · T^(2(X−1)) · g(θ_B)/g_zenith, with g = 1 − C·sin²θ_B and θ_B
  the angle between the beam and the geomagnetic field line (T = 0.84,
  C = 0.45, dip 36°, declination 9.5° E). The measurement term becomes
  HOMEAS · F^(−1/2).
  - C was calibrated on the 2026-09-21 LGS WFS counts: the model gives a
    TYC 5858-779-1 / UCAC4 748-00066 ratio of 0.74 against 0.75
    measured. It also matches the 2016 return-vs-pointing data to ~0.1
    mag.
  - The pointing comes from: the target's az/el (timeline); the snapshot
    time (field map); the frame header (Measured SR comparison); the
    azimuth average at the zenith angle (Prediction scenarios).
  - The page shows a sky map of the modelled return.
  - Off by default. Off and under the legacy budget, every result is
    unchanged. Saved in configs.
- `--instrument {osiris-imager, osiris-spec, nirc2}` for the LGS-offset
  default (below).
- Starlist dialog: a **sky plot** beside the table, showing where the
  selected target sits in Keck's pointing space. It is a polar map, zenith
  at the centre, N up and E right, and shows:
  - the selected telescope's pointing limits: the Nasmyth-deck wedge
    (blocked below its floor), the vignetted band below 18°, the zenith
    ceiling, and the ring above which guiding is not guaranteed;
  - the target's path across the night (17:00–08:00 HST; faint in
    twilight, Sun above −12°), with hour ticks;
  - the target at the "Evaluate at" time, coloured open / vignetted /
    blocked;
  - the Moon, with the dialog's 15° and 30° avoidance circles.

  It redraws on row clicks and time changes. New engine helpers:
  `night_track`, and `moon_altaz_deg` (topocentric).

### Changed
- **LGS offset default: the laser is offset only on K1 with the OSIRIS
  imager** (4.97″). The OSIRIS spectrograph and K2/NIRC2 are on axis (0″).
  - In the GUI, the instrument follows the Field map tab's OSIRIS imager /
    spectrograph selector on K1. Switching it recomputes and moves the
    laser on the field map.
  - The Measured SR comparison uses the frame's own instrument.
  - An explicit LGS-offset override still wins.
  - Defaults are unchanged for existing runs: K1 without `--instrument` is
    the imager.

Regress `gui_phase44`, `gui_phase45`.

## 1.2.2 — 2026-09-22

### Added
- Measured SR tab: **Auto-measure new frames**, the summit IDL Strehl
  tool's "find the Strehl of the last image automatically". PATH is
  polled off the GUI thread and each new, completely written frame of the
  selected instrument is measured with the page's settings; the current
  last frame is measured on switch-on. A frame landing mid-measurement or
  mid-"Measure field" waits; if a newer one lands too, only the newest is
  measured and the log names the skipped one. Not saved in configs.
- Measured SR tab: **Instrument** selector (NIRC2 / OSIRIS; both can be in
  use on one night). It sets which frames are looked for (NIRC2
  `n####.fits`, OSIRIS imager `i<YYMMDD>_a######.fits`) and where Latest
  searches, and each instrument remembers its own PATH (saved in configs).
  Switching instrument stops Auto-measure.
- Measured SR tab: **Latest** points PATH at tonight's directory — NIRC2
  on the NFS-mounted `/s/sdata900`–`907` disks (any account), OSIRIS on
  the AO server's `/s/sdata1100` via a one-off rsync listing. It says so
  when tonight's directory does not exist yet and the most recent night
  was picked.
- **Remote PATH** (`host:/dir`, e.g. `k2ao:/s/sdata1100/...`): read with
  rsync over ssh (listed every 3 s, new frames copied to
  `~/.cache/keck-ao-estimator/remote_frames`, newest 20 kept). Starting
  it asks for confirmation; while it runs a red "RSYNC POLLING" banner
  sits above the image and the tab reads "Measured SR ⚠ rsync". Any PATH
  change to or from a remote directory stops it.
- Auto-measure is **only available on the Keck network**: the checkbox is
  greyed out (reason in its tooltip) unless this machine can open a TCP
  connection to the AO server's ssh or the NIRC2 data server's NFS port —
  DNS is no test, keck.hawaii.edu names resolve publicly. Checked in the
  background at start-up and each time the tab is shown; losing the
  network while polling stops it. `KECK_AO_KECK_NETWORK=1/0` forces the
  answer. Engine: `keck_network.py`. Regress `gui_phase43`.

## 1.2.1 — 2026-09-15

### Docs
- Bundled manuals (Help menu) updated to the September 2026 editions:
  KAON 1556 (GUI manual) documents the Layers page (§8, Figure 26)
  and the 2.00" seeing-slider cap; KAON 1542 (technical note) adds the
  layer-mode paragraph to the prediction section. No code change.

## 1.2.0 — 2026-09-15

### Changed
- Prediction tab: the DIMM / MASS seeing sliders stop at 2.00" (was
  3.00"); Mauna Kea seeing is never worse than that. Saved configs with a
  larger value load clamped to 2.00".

### Added
- Prediction tab, new **Layers** sub-tab: turbulence **by layer** as
  turbulence fractions on the LTAO reconstructor's altitude grid (ground +
  the six MASS bins, the reconstructor's own units, summing to exactly 1:
  moving one row rescales the others proportionally in either direction;
  an "Exact" text row + Apply sets all seven verbatim, normalized only if
  they do not sum to 1). The total (DIMM) seeing sets the scale (its row is
  repeated on the Layers page) and the free-atm (MASS) seeing follows from
  the aloft fractions. "Reset layers to reconstructor prior" loads the
  reconstructor's static fractions (KAON 1542 §3.4; layer mismatch m = 0)
  and undoes layer edits. θ₀, m, the Cn² profile plot (repeated on the
  Layers page), the field map and the error terms follow live. Engine:
  `layers.py` (`fractions_to_layers`, `recon_prior_layers`,
  `layers_seeing`, `layer_seeing`, `layer_from_seeing`,
  `layers_from_seeing_pair`), `integrated_cn2_to_seeing`, and
  `synthetic_field_snapshot(..., cn2_layers=)`. Saved configs carry the
  mode and the exact fractions; older configs load with layer mode off.
  Regress `gui_phase42`.

## 1.1.0 — 2026-09-14

### Added
- Field engine for crowded-field Strehl: `psf_clean_engine="field"`
  solves every catalogued star of a frame together and subtracts every
  neighbour before the aperture measurement. GUI default; the library
  default stays `native`. Validated on 94 synthetic moderate-density
  fields (matched-target bias +0.012 vs +0.042 for the native engine,
  tails 2 vs 16).
- Parallel measurement: `measure_field(..., workers=N)` and
  `solve_field(..., workers=N)` run per-target measurements and the
  field solve's ePSF renders in a persistent process pool; results are
  bit-identical to `workers=1`. `workers=None` resolves to
  `$KECK_AO_WORKERS`, else `min(8, cpu_count // 2)`; counts above 8
  need `KECK_AO_WORKERS_UNCAPPED=1`.
- GUI: "Measure field" runs on worker threads and no longer blocks the
  window; the button doubles as Cancel while busy; a Workers spin box
  with config round-trip. Engine selector (field / native) next to the
  PSF-fit checkbox, saved per target.
- Regress batteries take `--workers N` (`psf_fit_model.py --full`).
- Public CI on every push (Python 3.11 and 3.14).

### Changed
- The empirical PSF model for a target is now weighted at the centre of
  its 1-arcsec bin, so a cleaned result no longer depends on the order
  in which stars were measured. Cleaned numbers move by <= 0.003 SR in
  the median; individual crowded targets can differ by up to ~0.085 SR
  from 1.0.0. Uncleaned numbers and the IDL goldens are unchanged.
- PSF-fit cleaning now refuses an unphysical cleaned result (cleaned
  Strehl > 1 or <= 0, or aperture flux collapsing below 20.6 % of the
  uncleaned value) and keeps the uncleaned measurement, saying so.
- The native engine's bias note names both regimes: a small
  underestimate on isolated pairs, an overestimate of order +0.04 on
  crowded fields.
- The S2 bias surface shipped with the regress suite was regenerated
  with the shipped engine (it was stale).

### Fixed
- The CLI crashed on a Windows console using the cp1252 code page when
  printing θ₀/d₀; stdout/stderr are now UTF-8 with replacement.
- The GUI could re-enable "Measure field" while a field solve was still
  running, and could accept results computed after the flow had decided
  to stop.
- The GUI failed to open when `KECK_AO_WORKERS` held an invalid value.
- `gui_phase12` regress no longer fails on slow hosted CI runners.

### Known limits
- Speedup at 8 workers on a 16-core desktop: native field measurement
  0.46x serial, field engine 0.61x (the ePSF build and the solve's
  sweeps are serial by design; the parallel parts are memory-bandwidth
  bound). On a 12-vCPU Linux VM at 6 workers: 0.38x and 0.52x.
- During "Measure field" the window can still pause for 83-281 ms
  between paints on each result (per-result panel redraw).
- On near-singular cleaning targets the last digits of a cleaned Strehl
  can depend on the machine's OpenBLAS thread count.
- No native ePSF has yet been built on a real NIRC2 frame; on such
  frames both engines refuse cleaning and the uncleaned number stands.

## 1.0.0 — 2026-08-12

First public release.
