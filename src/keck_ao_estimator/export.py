"""CSV export: write_csv_table(), the predicted-Strehl timeline table writer
shared by the CLI and the GUI's export path.
"""
import os
from datetime import timedelta

import numpy as np

from . import budget
from .budget import DEF_LGS_OFFSET, STATIC_TEL, active_budget_overrides, static_subtotal
from .constants import (
    DM_ACTUATORS_ACROSS, HST_TO_UTC_HOURS, MOFFAT_BETA_KOLM, TEL_DIAMETER_M,
    V_FREE, V_GROUND,
)
from .ngs import NGS_K1_QUADCELL_PENALTY, NGS_PARAMS, NGS_SK_ANCHOR
from .tiptilt import DEF_LTAO_TT_THETA0_GAIN, NGS_TILT_SERVO_MAS


def write_csv_table(args, prep, res, csv_path):
    """Write the predicted-Strehl timeline CSV to csv_path (honoring
    --force). Extracted verbatim from main() so the GUI exports through
    the identical code path; harness-verified byte-identical."""
    _ltao_bw_fac = prep._ltao_bw_fac
    baseline_zen_factor = prep.baseline_zen_factor
    dimm_dt = prep.dimm_dt
    dimm_sec = prep.dimm_sec
    dimm_see = prep.dimm_see
    fixed_zen_factor = prep.fixed_zen_factor
    in_any_window = prep.in_any_window
    lam_label = prep.lam_label
    lam_nm = prep.lam_nm
    mass_dt = prep.mass_dt
    mass_sec = prep.mass_sec
    mass_see = prep.mass_see
    night_date = prep.night_date
    out_path = prep.out_path
    profiles = prep.profiles
    show_target = prep.show_target
    tomography_on = prep.tomography_on
    ut_stamp = prep.ut_stamp
    windows = prep.windows
    zen_factor_by_time = prep.zen_factor_by_time
    col_airmass = res.col_airmass
    col_ang1 = res.col_ang1
    col_d0 = res.col_d0
    col_dimm = res.col_dimm
    col_mass = res.col_mass
    col_mm = res.col_mm
    col_ngs_th0 = res.col_ngs_th0
    col_r0_cm = res.col_r0_cm
    col_terms = res.col_terms
    col_theta0 = res.col_theta0
    col_tt10 = res.col_tt10
    col_zf = res.col_zf
    n_fb = res.n_fb
    ngs_bright = res.ngs_bright
    ngs_faint = res.ngs_faint
    ngs_fb = res.ngs_fb
    p_airmass = res.p_airmass
    p_dimm_in = res.p_dimm_in
    p_secs = res.p_secs
    p_times = res.p_times
    p_zf = res.p_zf
    sr_ltao = res.sr_ltao
    sr_single = res.sr_single
    th0_assumed = res.th0_assumed
    times = res.times
    # honor the same overwrite policy as the PNG
    if os.path.exists(csv_path) and not args.force:
        print(f"  (CSV '{csv_path}' exists; use --force to overwrite — skipping)")
    else:
        _tel = args.telescope
        header = [
            "utc_iso", "hst_iso",
            "dimm_seeing_arcsec_los", "mass_seeing_arcsec_los",
            "zenith_factor", "airmass", "r0_cm_500nm",
            f"ngs_R{args.ngs_bright:g}_strehl", f"ngs_R{args.ngs_faint:g}_strehl",
            "single_lgs_strehl", "ltao_strehl",
            "theta0_arcsec", "d0_m", "layer_mismatch", "ngs_theta0_arcsec",
            "err_fit_nm", "err_scint_nm", "err_ang_nm",
            "err_bw_single_nm", "err_bw_ltao_nm",
            "err_focal_single_nm", "err_alt_ltao_nm", "err_tt_nm",
        ]
        # FWHM columns appended at the END so the base layout is unchanged;
        # present only under --report fwhm/both (reference CSVs untouched).
        if getattr(args, "report", "strehl") != "strehl":
            # FOUR FWHM conventions side by side, all on the same 3-component
            # model PSF: *_fwhm_mas is the half-max (psf_fwhm_mas; no confirmed
            # real-tool analog); *_fwhm_gaussfit_mas is a no-background
            # Gaussian LSQ fit (fwhm_gaussfit_mas; models the OSIRIS quicklook
            # tool's rarely-used Strehl button, OSIRISSTREHL_QL2.pro);
            # *_fwhm_gaussfit_sky_mas is a free-background Gaussian LSQ fit
            # (fwhm_gaussfit_sky_mas; models the OSIRIS quicklook tool's
            # hand-drawn-box fit feature -- a separate, independent tool from
            # the AO Strehl tool).
            header += [f"ngs_R{args.ngs_bright:g}_fwhm_mas",
                       f"ngs_R{args.ngs_faint:g}_fwhm_mas",
                       "single_lgs_fwhm_mas", "ltao_fwhm_mas",
                       f"ngs_R{args.ngs_bright:g}_fwhm_gaussfit_mas",
                       f"ngs_R{args.ngs_faint:g}_fwhm_gaussfit_mas",
                       "single_lgs_fwhm_gaussfit_mas",
                       "ltao_fwhm_gaussfit_mas",
                       f"ngs_R{args.ngs_bright:g}_fwhm_gaussfit_sky_mas",
                       f"ngs_R{args.ngs_faint:g}_fwhm_gaussfit_sky_mas",
                       "single_lgs_fwhm_gaussfit_sky_mas",
                       "ltao_fwhm_gaussfit_sky_mas",
                       # and *_fwhm_srtool_mas is what THIS package's own
                       # Measured-SR tab reads off the same PSF -- its
                       # find_fwhm.pro port on a rendered detector frame
                       # (fwhm_srtool_mas). The one column directly
                       # comparable to a measured FWHM, so it is the one a
                       # predicted-vs-delivered join should use.
                       f"ngs_R{args.ngs_bright:g}_fwhm_srtool_mas",
                       f"ngs_R{args.ngs_faint:g}_fwhm_srtool_mas",
                       "single_lgs_fwhm_srtool_mas",
                       "ltao_fwhm_srtool_mas"]

        def _f(x, nd):
            """Format a value, blank if NaN (cols not applicable to this row)."""
            return "" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.{nd}f}"

        with open(csv_path, "w") as fh:
            # a couple of comment lines documenting provenance
            fh.write("# ao_strehl_timeline predicted Strehl vs time\n")
            fh.write(f"# night={night_date.date()} telescope={_tel} "
                     f"tomography={'on' if tomography_on else 'off'} "
                     f"target={'on' if show_target else 'off'} "
                     f"wavelength={lam_label}\n")
            _lo = DEF_LGS_OFFSET[args.telescope] if args.lgs_offset is None else args.lgs_offset
            fh.write(f"# budget={'legacy' if args.legacy_budget else 'refined-2026-07'} "
                     f"tt_star=R{args.tt_mag:g}@{args.tt_offset:g}arcsec "
                     f"lgs_offset={_lo:g}arcsec ngs_offset={float(args.ngs_offset or 0):g}arcsec"
                     + (f" ltao_bw_factor={_ltao_bw_fac:.3f}(floor_frac={args.ltao_bw_floor_frac:g})"
                        if not args.legacy_budget else "")
                     + (f" ltao_tt_theta0_gain="
                        f"{getattr(args, 'ltao_tt_theta0_gain', DEF_LTAO_TT_THETA0_GAIN):g}"
                        if not args.legacy_budget else "")
                     + f" ngs_seeing_law={args.ngs_seeing_law}"
                     + (f"(anchor_sK={NGS_SK_ANCHOR:g})"
                        if args.ngs_seeing_law == "kolmogorov" else "")
                     + (f" assumed_theta0={args.assumed_theta0:g}arcsec(K,zenith)"
                        if args.assumed_theta0 is not None else "") + "\n")
            # A what-if run (GUI WFE sliders) records the moved parameters here
            # so a modified budget can never masquerade as the reference (§0.3).
            # Absent entirely when nothing is overridden, so reference outputs
            # -- and the CLI, which never overrides -- are byte-unchanged.
            _bov = active_budget_overrides()
            if _bov:
                fh.write("# budget_overrides="
                         + ",".join(f"{k}={v:g}" for k, v in sorted(_bov.items()))
                         + "\n")
            # non-default wind estimate (feeds the wind-weighted bandwidth);
            # absent at the default 8/25 m/s so reference outputs are unchanged.
            if args.wind_ground != V_GROUND or args.wind_free != V_FREE:
                fh.write(f"# winds: ground={args.wind_ground:g}m/s "
                         f"free={args.wind_free:g}m/s (non-default; "
                         f"reference {V_GROUND:g}/{V_FREE:g})\n")
            # non-default NGS Gompertz fit (recalibration, not budget what-if);
            # absent at the reference values so reference outputs are unchanged.
            _gp = NGS_PARAMS[_tel]
            _gov = []
            for _flag, _key in (("ngs_s0", "S0"), ("ngs_a", "A"),
                                ("ngs_m0", "m0"), ("ngs_w", "w")):
                _v = getattr(args, _flag)
                if _v is not None and _v != _gp[_key]:
                    _gov.append(f"{_key}={_v:g}")
            if _tel == "K1" and args.k1_quadcell_penalty != NGS_K1_QUADCELL_PENALTY:
                _gov.append(f"quadcell={args.k1_quadcell_penalty:g}")
            if _gov:
                fh.write(f"# ngs_fit_override({_tel}): {' '.join(_gov)} "
                         f"(reference S0={_gp['S0']:g} A={_gp['A']:g} "
                         f"m0={_gp['m0']:g} w={_gp['w']:g}"
                         + (f" quadcell={NGS_K1_QUADCELL_PENALTY:g}"
                            if _tel == "K1" else "") + ")\n")
            # FWHM model provenance, present only when FWHM columns are (so
            # reference outputs are unchanged).
            if getattr(args, "report", "strehl") != "strehl":
                _N = DM_ACTUATORS_ACROSS[args.telescope]
                fh.write(f"# fwhm_model=airy_core(D={TEL_DIAMETER_M:g}m)"
                         f"*gauss_tt_jitter + moffat_shoulder(FWHM=theta_c="
                         f"{_N:g}/2*lam/D, corrected-band energy) + "
                         f"moffat_wings(beta={MOFFAT_BETA_KOLM:g}, kolmogorov "
                         f"lam^-1/5 LOS seeing, fitting-error energy) "
                         f"core_energy=S_ho=S_total/marechal(tt) "
                         f"ngs_tilt_servo={getattr(args,'ngs_tilt_servo',NGS_TILT_SERVO_MAS):g}mas"
                         f"(fwhm-path only; strehl budget unchanged) "
                         f"report={args.report}\n")
                _bm = float(getattr(args, "fwhm_box_mas", 300.0))
                fh.write(f"# fwhm_gaussfit=simulated NO-background Gaussian LSQ "
                         f"fit (free amplitude, no sky, r<{_bm:g}mas) on the "
                         f"same PSF; models the OSIRIS quicklook tool's "
                         f"rarely-used Strehl button, OSIRISSTREHL_QL2.pro "
                         f"(its own box is auto-sized, ~30.7mas at K -- "
                         f"box_mas is a real, adjustable parameter here, "
                         f"default kept at 300 to preserve the validation "
                         f"below). "
                         f"fwhm_gaussfit_sky=simulated FREE-background "
                         f"Gaussian LSQ fit (free amplitude+constant, "
                         f"r<{_bm:g}mas); models the OSIRIS quicklook tool's "
                         f"hand-drawn-box fit feature (a separate, "
                         f"independent tool from the AO Strehl tool; its box "
                         f"is drawn by hand, not fixed). "
                         f"Validated 20260701 00:35-01:36 K1 (box_mas=300): "
                         f"half-max 66.0 vs \"AO Strehl tool\" 62.8; gaussfit "
                         f"71.8 vs \"OSIRIS QL\" 76.7 -- this pairing is "
                         f"empirical, not a confirmed tool-mechanism match "
                         f"(see psf.py \"REAL MEASUREMENT TOOLS\")\n")
                fh.write("# fwhm_srtool=THIS package's own Measured-SR tab "
                         "run on the same model PSF: rendered on a NIRC2 "
                         "narrow-camera pixel grid, annulus sky (1.2-1.4\") "
                         "subtracted, then image_strehl.radial_profile_fwhm "
                         "(the find_fwhm.pro port) -- the tool's own code, "
                         "not a model of it. Validated 2026-08-07 on 60 "
                         "isolated-standard NIRC2 frames (20260727 o Her) at "
                         "their MEASURED Strehl: median convention error "
                         "-0.4mas vs -1.4 half-max / -3.8 gaussfit / -4.1 "
                         "gaussfit-sky. Use THIS column against a measured "
                         "FWHM\n")
            fh.write("# merged timeline, two row types by source= column:\n")
            if float(args.ngs_offset or 0.0) > 0.0:
                fh.write("#   source=dimm : one row per DIMM sample -- dimm seeing, "
                         "r0, NGS Strehl including the off-axis anisoplanatism "
                         "factor (blank where no MASS profile within tolerance "
                         "supplies theta0; MASS-derived cols blank)\n")
            else:
                fh.write("#   source=dimm : one row per DIMM sample -- dimm seeing, "
                         "r0, NGS Strehl (MASS-derived cols blank)\n")
            fh.write("#   source=mass : one row per MASS profile -- mass seeing, "
                     "LGS/LTAO Strehl, theta0, d0, plus the matched DIMM total "
                     "seeing used as input (NGS cols blank)\n")
            fh.write("# Strehl columns + theta0 are at the science wavelength; "
                     "seeing columns are 500nm LINE-OF-SIGHT (zenith x airmass^(3/5)); "
                     "r0 is 500nm at ZENITH; d0 is wavelength-independent (m)\n")
            fh.write("# err_*_nm columns: per-sample error-budget terms in nm "
                     "RMS on mass rows (line-of-sight-projected inputs). "
                     "err_focal_single is the single-beacon focal-aniso (cone) term; "
                     "err_alt_ltao is tomography + quadratic layer-mismatch. "
                     f"Fixed terms not tabulated: HO measurement {budget.HOMEAS:g}, "
                     f"Na focus {budget.NAFOC:g}, static "
                     f"{static_subtotal(args.telescope):.1f} "
                     f"(={args.telescope}: tel-aberr {STATIC_TEL[args.telescope]:.1f} "
                     f"+ WFS-calib {budget.STATIC_CALIB:.1f} + DM {budget.STATIC_DM:.1f} "
                     f"+ AO/instr {budget.STATIC_INST:.1f} + registration "
                     f"{budget.STATIC_REG:.1f}, quad), margin {budget.MARGIN:g} nm.\n")
            fh.write(",".join(["source"] + header) + "\n")

            rows = []
            for i in range(len(times)):
                t_hst = times[i]
                t_utc = t_hst + timedelta(hours=HST_TO_UTC_HOURS)
                rows.append((t_hst, ["dimm",
                    t_utc.strftime("%Y-%m-%dT%H:%M:%S"),
                    t_hst.strftime("%Y-%m-%dT%H:%M:%S"),
                    _f(col_dimm[i] * col_zf[i], 4), "",
                    _f(col_zf[i], 4), _f(col_airmass[i], 4), _f(col_r0_cm[i], 2),
                    _f(ngs_bright[i], 4), _f(ngs_faint[i], 4),
                    "", "", "", "", "",
                    _f(float(col_ngs_th0[i]), 3),
                    "", "", "", "", "", "", "", "",
                ]))
            for i in range(len(p_times)):
                t_hst = p_times[i]
                t_utc = t_hst + timedelta(hours=HST_TO_UTC_HOURS)
                rows.append((t_hst, ["mass",
                    t_utc.strftime("%Y-%m-%dT%H:%M:%S"),
                    t_hst.strftime("%Y-%m-%dT%H:%M:%S"),
                    _f(p_dimm_in[i] * p_zf[i], 4), _f(col_mass[i] * p_zf[i], 4),
                    _f(p_zf[i], 4), _f(p_airmass[i], 4), "",
                    "", "",
                    _f(sr_single[i], 4), _f(sr_ltao[i], 4),
                    _f(col_theta0[i], 3), _f(col_d0[i], 3),
                    _f(col_mm[i], 3), "",
                ] + [_f(float(v), 1) for v in col_terms[i]]))
            # append the FWHM cells before the time-sort: rows[:len(times)]
            # are the dimm rows in i-order, the rest the mass rows in i-order.
            if getattr(args, "report", "strehl") != "strehl":
                for i in range(len(times)):
                    rows[i][1].extend([
                        _f(float(res.fwhm_ngs_bright[i]), 1),
                        _f(float(res.fwhm_ngs_faint[i]), 1), "", "",
                        _f(float(res.fwhm_gauss_ngs_bright[i]), 1),
                        _f(float(res.fwhm_gauss_ngs_faint[i]), 1), "", "",
                        _f(float(res.fwhm_sky_ngs_bright[i]), 1),
                        _f(float(res.fwhm_sky_ngs_faint[i]), 1), "", "",
                        _f(float(res.fwhm_tool_ngs_bright[i]), 1),
                        _f(float(res.fwhm_tool_ngs_faint[i]), 1), "", ""])
                for i in range(len(p_times)):
                    rows[len(times) + i][1].extend([
                        "", "",
                        _f(float(res.fwhm_single[i]), 1),
                        _f(float(res.fwhm_ltao[i]), 1),
                        "", "",
                        _f(float(res.fwhm_gauss_single[i]), 1),
                        _f(float(res.fwhm_gauss_ltao[i]), 1),
                        "", "",
                        _f(float(res.fwhm_sky_single[i]), 1),
                        _f(float(res.fwhm_sky_ltao[i]), 1),
                        "", "",
                        _f(float(res.fwhm_tool_single[i]), 1),
                        _f(float(res.fwhm_tool_ltao[i]), 1)])
            rows.sort(key=lambda r: r[0])
            for _, row in rows:
                fh.write(",".join(row) + "\n")
        print(f"  Wrote table {csv_path}  ({len(times)} DIMM rows + "
              f"{len(p_times)} MASS rows)")
    return csv_path
