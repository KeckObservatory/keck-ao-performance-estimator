# Changelog

All notable changes to the Keck AO Performance Estimator. The project
is released on GitHub only (pin `@v1.2.2` in the install URL).

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
