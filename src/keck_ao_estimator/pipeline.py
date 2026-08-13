"""The two-stage estimation pipeline: prepare_night() (load/fetch a night's
data, resolve settings, build target geometry) and compute_timeline() (the
per-sample Strehl/FWHM estimation core). See each function's docstring.
"""
import os
from types import SimpleNamespace
from datetime import datetime, timedelta

import numpy as np

from . import budget
from .atmosphere import theta0_d0_from_profile, zenith_seeing_factor
from .budget import (
    ANG_REF_OFFSET, FITTING_ERR, layer_mismatch, lgs_budget_terms, lgs_strehl,
    ltao_bw_factor,
)
from .config import (
    default_output_name, parse_night, parse_windows, resolve_tomography,
    resolve_tt_sensor, resolve_wavelength,
)
from .constants import (
    DEF_WINDOWS, DM_ACTUATORS_ACROSS, LAMBDA_K_NM, REF_FREEATM, REF_TOTAL,
)
from .geometry import compute_airmass_curve
from .io import fetch_mkwc_files, load_mass_profile, load_seeing_series
from .ngs import ngs_strehl
from .psf import (fwhm_gaussfit_mas, fwhm_gaussfit_sky_mas,
                  fwhm_srtool_mas, psf_fwhm_mas)
from .tiptilt import NGS_TILT_SERVO_MAS, ngs_tt_nm, tt_wfe_nm


def prepare_night(args):
    """Stage 1 of 2 (see main): resolve settings, load/fetch the night's
    data files, and build the target geometry (windows, airmass, zenith
    factors). Expensive (network + astropy); a GUI should call this ONCE
    per night/target and cache the result."""
    # --- resolve telescope-dependent and opt-in settings ---------------------
    tomography_on = resolve_tomography(args)   # K2 off / K1 on unless forced
    # TT sensor (K1): 'strap' | 'trick-h' | 'trick-k'. On K1 the TRICK IR
    # sensor and OSIRIS share the dichroic, so sensing in K forces science H
    # and vice versa. Records the base sensor + sensing band, and swaps the
    # science band (unless --wavelength pins the nm explicitly).
    resolve_tt_sensor(args)
    # LTAO bandwidth penalty: latency-floor-aware multiplier on the single-beacon
    # bandwidth term for the slower LTAO loop (see ltao_bw_factor / --ltao-bw-
    # floor-frac). Legacy budget keeps the old pure half-rate 2^(5/6) factor.
    _ltao_bw_fac = (2.0 ** (5.0 / 6.0) if args.legacy_budget
                    else ltao_bw_factor(args.ltao_bw_floor_frac))
    show_target   = args.show_target           # airmass + window boxes: opt-in
    lam_nm, lam_label = resolve_wavelength(args)   # science wavelength

    # --- choose input files: download from MKWC, or use the local paths -------
    if args.fetch_date:
        print(f"Fetching MKWC seeing data for {args.fetch_date} "
              f"into '{args.cache_dir}/' ...")
        dimm_path, mass_path, masspro_path = fetch_mkwc_files(
            args.fetch_date, args.cache_dir, refetch=args.refetch,
            trust_cache=args.trust_cache)
    else:
        dimm_path, mass_path, masspro_path = args.dimm, args.mass, args.masspro

    # --- load the three data files -------------------------------------------
    dimm_dt, dimm_sec, dimm_see = load_seeing_series(dimm_path)
    mass_dt, mass_sec, mass_see = load_seeing_series(mass_path)
    profiles = load_mass_profile(masspro_path)

    # DIMM is required (it drives NGS and the time axis). MASS + profile are
    # optional: without them we simply skip the LGS / LTAO / theta0 / d0
    # estimates and produce an NGS-only run.
    if len(dimm_sec) == 0:
        raise SystemExit(
            "ERROR: no usable DIMM data — cannot proceed.\n"
            f"       DIMM file: {dimm_path}\n"
            "       (DIMM total seeing is required; it drives NGS and the time "
            "axis.)")
    if len(mass_see) == 0 or len(profiles) == 0:
        missing = []
        if len(mass_see) == 0:  missing.append("MASS")
        if len(profiles) == 0:  missing.append("MASS-profile")
        print(f"  NOTE: no usable {' / '.join(missing)} data — proceeding with "
              f"DIMM only (NGS estimates; LGS/LTAO/θ₀/d₀ skipped).")

    # --- determine the civil (evening) date of the night ---------------------
    #  Robust default: take it from the DATA itself. A night's data begins in
    #  the evening (before midnight), so the earliest sample's date IS the civil
    #  evening date. This removes any chance of a --night / file-date mismatch
    #  (the MKWC files are named by the morning/UT date, which trips people up).
    #  --night is kept only as an explicit override for unusual cases.
    if args.night:
        night_date = parse_night(args.night)
        if night_date.date() != dimm_dt.min().date():
            print(f"  WARNING: --night {night_date.date()} does not match the "
                  f"data's first timestamp date {dimm_dt.min().date()}.\n"
                  f"           Using --night as given, but double-check this is "
                  f"intended (the time axis is anchored to --night).")
    else:
        night_date = datetime(dimm_dt.min().year,
                              dimm_dt.min().month,
                              dimm_dt.min().day)
        print(f"  Night date derived from data: {night_date.date()} (evening)")

    # --- resolve the output filename -----------------------------------------
    #  Default: auto-name from the data's UT date stamp + telescope. The UT date
    #  is the morning/UT date = the night_date + 1 day (the MKWC file convention),
    #  unless the user fetched, in which case use that explicit stamp.
    ut_stamp = (args.fetch_date if args.fetch_date
                else (night_date + timedelta(days=1)).strftime("%Y%m%d"))
    if args.out:
        out_path = args.out
        #  matplotlib silently appends ".png" when saving without a recognized
        #  extension, which would desynchronize out_path from the file actually
        #  written (breaking the console message and the combined-figure
        #  compositor). Normalize here so name-on-disk == name-in-hand.
        if os.path.splitext(out_path)[1].lower() not in (
                ".png", ".pdf", ".svg", ".jpg", ".jpeg"):
            out_path += ".png"
            print(f"  note: --out had no image extension; writing '{out_path}'")
    else:
        out_path = default_output_name(ut_stamp, args.telescope)

    # refuse to overwrite unless --force (now that we know the final name)
    if os.path.exists(out_path) and not args.force:
        raise SystemExit(
            f"ERROR: output file '{out_path}' already exists.\n"
            f"       Use --force to overwrite, or --out NAME to write elsewhere.")

    # --- parse the observing windows (only used when --target is given) ------
    windows = parse_windows(args.window if args.window else DEF_WINDOWS,
                            night_date)

    # --- zenith-angle setup ---------------------------------------------------
    #  Baseline ZA = --zenith-angle (default 0). Inside the observing window(s),
    #  if --target is given, the baseline is OVERRIDDEN by the target's actual
    #  per-sample line-of-sight airmass. Outside the window(s) the baseline ZA
    #  applies. So:
    #    * --target only           -> ZA=0 outside, target ZA inside.
    #    * --target --zenith 50     -> ZA=50 outside, target ZA inside.
    #    * --zenith 50 (no target)  -> ZA=50 everywhere.
    #  The factor scales BOTH total (DIMM) and free-atm (MASS) seeing.
    def in_any_window(t):
        return any(w0 <= t <= w1 for (w0, w1) in windows)

    baseline_zen_factor = zenith_seeing_factor(args.zenith_angle)   # default 0->1.0

    # Zenith factor is a function of TIME only, so compute it on the DIMM
    # time-base (NGS is estimated at every DIMM sample; see loop below).
    dimm_dts = [d for d in dimm_dt]
    if show_target:
        am_d, el_d, _az_d = compute_airmass_curve(args.ra, args.dec, dimm_dts)
        zen_factor_by_time = {}
        for pt, am, el in zip(dimm_dts, am_d, el_d):
            if in_any_window(pt) and el > 0 and np.isfinite(am):
                zen_factor_by_time[pt] = float(am) ** (3.0 / 5.0)  # on-target ZA
            else:
                zen_factor_by_time[pt] = baseline_zen_factor       # baseline ZA
        fixed_zen_factor = None
        if args.zenith_angle:
            print(f"  Zenith projection: {args.target_name} ZA inside observing "
                  f"window(s); ZA={args.zenith_angle:g} deg elsewhere")
        else:
            print(f"  Zenith projection: {args.target_name} ZA inside observing "
                  f"window(s); ZA=0 (zenith) elsewhere")
    else:
        fixed_zen_factor = baseline_zen_factor
        zen_factor_by_time = None
        if args.zenith_angle:
            print(f"  Zenith projection: fixed zeta={args.zenith_angle:g} deg "
                  f"(airmass {1.0/np.cos(np.radians(min(abs(args.zenith_angle),85))):.2f}, "
                  f"seeing x{fixed_zen_factor:.3f})")

    return SimpleNamespace(**{k: v for k, v in locals().items()
                              if k != "args"})


def compute_timeline(args, prep):
    """Stage 2 of 2: the per-sample estimation core. Pure math on the
    prepared data -- fast (milliseconds), safe to re-run per parameter
    change (e.g. GUI budget sliders via budget_overrides())."""
    _ltao_bw_fac = prep._ltao_bw_fac
    baseline_zen_factor = prep.baseline_zen_factor
    dimm_dt = prep.dimm_dt
    dimm_sec = prep.dimm_sec
    dimm_see = prep.dimm_see
    fixed_zen_factor = prep.fixed_zen_factor
    in_any_window = prep.in_any_window
    lam_nm = prep.lam_nm
    profiles = prep.profiles
    show_target = prep.show_target
    zen_factor_by_time = prep.zen_factor_by_time
    # --- per-sample estimation -----------------------------------------------
    #  Two timebases, matching what each estimate physically depends on:
    #    * NGS needs only the TOTAL (DIMM) seeing -> computed at EVERY DIMM
    #      sample, on the DIMM timebase.
    #    * LGS / LTAO / theta0 / d0 need the MASS free-atmosphere seeing and
    #      Cn2 profile -> computed at EVERY MASS-profile sample, on the
    #      PROFILE timebase, using the nearest DIMM total seeing (within
    #      --match-tol) as the total-seeing input. Reusing one DIMM sample as
    #      the input for two nearby profiles is fine -- each plotted point is
    #      still backed by its own distinct MASS measurement. (The reverse --
    #      reusing one MASS profile for several DIMM samples -- is NOT fine:
    #      it fabricates "fresh" LGS/theta0 points from stale MASS data, most
    #      visibly continuing the curves after MASS stopped for the night.)
    #  Result: one theta0/LGS point per real MASS profile, no more, no less;
    #  a profile is skipped only if there is no DIMM sample within tolerance
    #  to provide its total-seeing input.
    times = []
    ngs_bright, ngs_faint = [], []
    col_dimm, col_zf, col_airmass = [], [], []

    dimm_order = np.argsort(dimm_sec) if len(dimm_sec) else np.array([], dtype=int)
    dimm_dt_sorted  = np.asarray(dimm_dt)[dimm_order]
    dimm_sec_sorted = np.asarray(dimm_sec)[dimm_order]
    dimm_see_sorted = np.asarray(dimm_see)[dimm_order]

    # ---- NGS on the DIMM timebase ----
    #  Off-axis NGS (--ngs-offset > 0): the science direction suffers full
    #  angular anisoplanatism against the NGS-referenced correction, a Strehl
    #  factor exp(-(theta/theta0)^(5/3)) with theta0 at the science wavelength.
    #  theta0 comes from the MASS profile, so each NGS sample borrows the
    #  nearest profile within --match-tol as input (bounded staleness, the
    #  same rule as the DIMM total-seeing input to the LGS/LTAO points);
    #  samples with no profile in tolerance become honest gaps.
    ngs_off = float(args.ngs_offset or 0.0)
    th0_assumed = (args.assumed_theta0
                   if (args.assumed_theta0 is not None and args.assumed_theta0 > 0)
                   else None)
    if ngs_off > 0.0 and not profiles and th0_assumed is None:
        raise SystemExit("--ngs-offset requires theta0: no MASS profile "
                         "(masspro) is available for this night and the "
                         "assumed-theta0 fallback is disabled (<= 0). Either "
                         "supply the masspro file or re-enable the fallback "
                         "(--assumed-theta0 15).")
    prof_secs_arr = np.array([p[1] for p in profiles]) if profiles else np.array([])
    # fallback theta0 at the science wavelength, zenith (scaled per sample below)
    th0_fb_zen = (th0_assumed * (lam_nm / LAMBDA_K_NM) ** 1.2
                  if th0_assumed is not None else None)
    n_fb = 0                       # samples that used the assumed theta0
    col_ngs_th0 = []               # theta0 actually used per NGS sample (or nan)
    ngs_fb = []                    # per-sample: True where assumed theta0 used
    for dt, sec, eps_tot in zip(dimm_dt_sorted, dimm_sec_sorted, dimm_see_sorted):
        # zenith factor at this DIMM time
        zf = (zen_factor_by_time[dt] if zen_factor_by_time is not None
              else fixed_zen_factor)
        eps_tot_los = eps_tot * zf

        _nkw = dict(seeing_law=args.ngs_seeing_law,
                    ngs_s0=args.ngs_s0, ngs_a=args.ngs_a,
                    ngs_m0=args.ngs_m0, ngs_w=args.ngs_w,
                    k1_quadcell=args.k1_quadcell_penalty)
        s_b = ngs_strehl(eps_tot_los, args.ngs_bright, args.telescope, lam_nm,
                         **_nkw)
        s_f = ngs_strehl(eps_tot_los, args.ngs_faint, args.telescope, lam_nm,
                         **_nkw)
        th0_used = np.nan
        if ngs_off > 0.0:
            th0 = None
            if prof_secs_arr.size:
                j = int(np.argmin(np.abs(prof_secs_arr - sec)))
                if abs(prof_secs_arr[j] - sec) <= args.match_tol:
                    za_here = np.degrees(np.arccos(min(1.0, zf ** (-5.0 / 3.0))))
                    th0, _d0 = theta0_d0_from_profile(profiles[j][3], za_here, lam_nm)
            used_fb = False
            if th0 is None and th0_fb_zen is not None:
                # assumed theta0: zenith value projected onto this line of
                # sight (theta0 ~ airmass^(-3/5) = 1/zf)
                th0 = th0_fb_zen / zf
                n_fb += 1
                used_fb = True
            if th0 is not None:
                aniso = np.exp(-(ngs_off / th0) ** (5.0 / 3.0)) if th0 > 0 else 0.0
                s_b *= aniso
                s_f *= aniso
                th0_used = th0
            else:
                s_b = s_f = np.nan          # no theta0 available -> honest gap
        col_ngs_th0.append(th0_used)
        ngs_fb.append(used_fb if ngs_off > 0.0 else False)

        times.append(dt)
        ngs_bright.append(s_b)
        ngs_faint.append(s_f)
        col_dimm.append(eps_tot)
        col_zf.append(zf)
        col_airmass.append(zf ** (5.0 / 3.0))

    # ---- LGS / LTAO / theta0 / d0 on the PROFILE timebase ----
    p_times, p_secs = [], []
    sr_single, sr_ltao = [], []
    col_mass, col_theta0, col_d0 = [], [], []
    col_mm = []                      # layer mismatch m vs reconstructor prior
    col_cn2 = []                     # per-profile 6-bin Cn2 (for the field map)
    col_terms = []                   # per-sample error-budget terms (nm RMS)
    col_tt10  = []                   # reference TT error: R=10 star on-axis
    col_ang1  = []                   # reference ang aniso at a 1" laser offset
    p_zf, p_airmass, p_dimm_in = [], [], []

    if profiles and len(dimm_sec_sorted):
        # zenith factor at the profile times (same baseline/window rules as DIMM)
        prof_dts = [p[0] for p in profiles]
        if show_target:
            am_p, el_p, _az_p = compute_airmass_curve(args.ra, args.dec, prof_dts)
            zf_prof = [float(am) ** (3.0 / 5.0)
                       if (in_any_window(pt) and el > 0 and np.isfinite(am))
                       else baseline_zen_factor
                       for pt, am, el in zip(prof_dts, am_p, el_p)]
        else:
            zf_prof = [fixed_zen_factor] * len(prof_dts)

        for (pdt, psec, eps_fa_raw, cn2_bins), zf in zip(profiles, zf_prof):
            j = int(np.argmin(np.abs(dimm_sec_sorted - psec)))
            if abs(dimm_sec_sorted[j] - psec) > args.match_tol:
                continue      # no DIMM total-seeing input near this profile
            eps_tot = float(dimm_see_sorted[j])
            eps_fa  = min(eps_fa_raw, 0.99 * eps_tot)
            eps_tot_los = eps_tot * zf
            eps_fa_los  = eps_fa * zf
            za_here = np.degrees(np.arccos(min(1.0, zf ** (-5.0 / 3.0))))
            th0, d0 = theta0_d0_from_profile(cn2_bins, za_here, lam_nm)

            p_times.append(pdt)
            p_secs.append(psec)
            _bkw = dict(tt_mag=args.tt_mag, tt_offset=args.tt_offset,
                        lgs_offset=args.lgs_offset, legacy=args.legacy_budget,
                        bw_factor=_ltao_bw_fac,
                        v_ground=args.wind_ground, v_free=args.wind_free,
                        tt_sensor=getattr(args, "_tt_sensor_base", "strap"),
                        ltao_tt_theta0_gain=getattr(
                            args, "ltao_tt_theta0_gain", None))
            sr_single.append(lgs_strehl(eps_tot_los, eps_fa_los, args.telescope,
                                        "single", lam_nm, **_bkw))
            sr_ltao.append(lgs_strehl(eps_tot_los, eps_fa_los, args.telescope,
                                      "ltao", lam_nm, cn2_bins=cn2_bins, **_bkw))
            # per-term budget values for the CSV (nm RMS); mode-dependent
            # terms (bw, alt) are recorded for BOTH modes
            _ts = lgs_budget_terms(eps_tot_los, eps_fa_los, args.telescope,
                                   "single", **_bkw)
            _tl = lgs_budget_terms(eps_tot_los, eps_fa_los, args.telescope,
                                   "ltao", cn2_bins=cn2_bins, **_bkw)
            col_terms.append((_ts["fit"], _ts["scint"], _ts["ang"],
                              _ts["bw"], _tl["bw"],
                              _ts["alt"], _tl["alt"], _ts["tt"]))
            col_tt10.append(tt_wfe_nm((eps_tot_los / REF_TOTAL) ** (5.0 / 6.0),
                                      10.0, 0.0))
            col_ang1.append(budget.ANG_REF
                            * (eps_fa_los / REF_FREEATM) ** (5.0 / 6.0)
                            * (1.0 / ANG_REF_OFFSET) ** (5.0 / 6.0))
            col_mm.append(0.0 if args.legacy_budget else layer_mismatch(cn2_bins))
            col_mass.append(eps_fa)
            col_theta0.append(th0)
            col_d0.append(d0)
            col_cn2.append(np.asarray(cn2_bins, dtype=float))
            p_zf.append(zf)
            p_airmass.append(zf ** (5.0 / 3.0))
            p_dimm_in.append(eps_tot)

    # sort each timebase chronologically
    times      = np.array(times)
    order      = np.argsort(times)
    times      = times[order]
    ngs_bright = np.array(ngs_bright)[order]
    ngs_faint  = np.array(ngs_faint)[order]
    col_dimm   = np.array(col_dimm)[order]
    col_zf     = np.array(col_zf)[order]
    col_airmass = np.array(col_airmass)[order]
    col_ngs_th0 = np.array(col_ngs_th0)[order]
    ngs_fb      = np.array(ngs_fb)[order]

    p_times    = np.array(p_times)
    p_order    = np.argsort(p_times) if len(p_times) else np.array([], dtype=int)
    p_times    = p_times[p_order]
    p_secs     = np.array(p_secs)[p_order]
    sr_single  = np.array(sr_single)[p_order]
    sr_ltao    = np.array(sr_ltao)[p_order]
    col_mass   = np.array(col_mass)[p_order]
    col_mm     = np.array(col_mm)[p_order]
    col_terms  = (np.array(col_terms)[p_order]
                  if len(col_terms) else np.zeros((0, 8)))
    col_tt10   = np.array(col_tt10)[p_order]
    col_ang1   = np.array(col_ang1)[p_order]
    col_theta0 = np.array(col_theta0)[p_order]
    col_d0     = np.array(col_d0)[p_order]
    col_cn2    = (np.array(col_cn2)[p_order] if len(col_cn2)
                  else np.zeros((0, 6)))
    p_zf       = np.array(p_zf)[p_order]
    p_airmass  = np.array(p_airmass)[p_order]
    p_dimm_in  = np.array(p_dimm_in)[p_order]
    # r0 at 500 nm from the (zenith-corrected) DIMM total seeing, for the CSV
    col_r0_cm  = np.array([0.98 * 500e-9 / (e / 206265.0) * 100.0
                           for e in col_dimm])

    # --- FWHM estimates (only when --report fwhm/both; see psf_fwhm_mas) -----
    #  Halo width input is the LINE-OF-SIGHT total (DIMM) seeing at 500 nm:
    #  col_dimm*col_zf on the NGS/DIMM timebase, the matched p_dimm_in*p_zf on
    #  the LGS/profile timebase. None (not empty arrays) in strehl mode, so a
    #  consumer that forgets the gate fails loudly rather than plotting junk.
    #  Each curve is smeared by ITS OWN residual tip-tilt (see psf_fwhm_mas):
    #    * NGS   -- the guide star IS the tilt reference, so use the engine's
    #               parameterized TT model at the NGS magnitude and offset;
    #    * LGS / LTAO -- the budgeted TT star, already tabulated per sample as
    #               the 8th column of col_terms (err_tt_nm).
    fwhm_ngs_bright = fwhm_ngs_faint = fwhm_single = fwhm_ltao = None
    tt_ngs_bright = tt_ngs_faint = fit_ngs = None
    if getattr(args, "report", "strehl") != "strehl":
        _eps_d = col_dimm * col_zf
        _s_tot_d = (_eps_d / REF_TOTAL) ** (5.0 / 6.0)
        _ngs_off = float(args.ngs_offset or 0.0)
        _servo = float(getattr(args, "ngs_tilt_servo", NGS_TILT_SERVO_MAS))
        _tt_b = np.array([ngs_tt_nm(s, args.ngs_bright, _ngs_off, _servo)
                          for s in _s_tot_d])
        _tt_f = np.array([ngs_tt_nm(s, args.ngs_faint, _ngs_off, _servo)
                          for s in _s_tot_d])
        # shoulder inputs: per-sample fitting term + the telescope's DM sampling
        _nact = DM_ACTUATORS_ACROSS[args.telescope]
        _fit_d = FITTING_ERR[args.telescope] * _s_tot_d
        # exposed on `res` so a consumer (e.g. the Measured-SR comparison) can
        # re-evaluate psf_fwhm_mas at a DIFFERENT wavelength than lam_nm
        # (e.g. the actual band of a measured frame) without re-deriving the
        # per-sample tt/fit inputs from scratch
        tt_ngs_bright, tt_ngs_faint, fit_ngs = _tt_b, _tt_f, _fit_d
        fwhm_ngs_bright = np.array(
            [psf_fwhm_mas(s, e, lam_nm, t, fit_nm=f, n_act=_nact)
             for s, e, t, f in zip(ngs_bright, _eps_d, _tt_b, _fit_d)])
        fwhm_ngs_faint  = np.array(
            [psf_fwhm_mas(s, e, lam_nm, t, fit_nm=f, n_act=_nact)
             for s, e, t, f in zip(ngs_faint, _eps_d, _tt_f, _fit_d)])
        _eps_p = p_dimm_in * p_zf if len(p_times) else np.array([])
        _tt_p = (col_terms[:, 7] if len(col_terms)
                 else np.zeros(len(p_times)))
        _fit_p = (col_terms[:, 0] if len(col_terms)
                  else np.zeros(len(p_times)))
        fwhm_single = np.array(
            [psf_fwhm_mas(s, e, lam_nm, t, fit_nm=f, n_act=_nact)
             for s, e, t, f in zip(sr_single, _eps_p, _tt_p, _fit_p)])
        fwhm_ltao   = np.array(
            [psf_fwhm_mas(s, e, lam_nm, t, fit_nm=f, n_act=_nact)
             for s, e, t, f in zip(sr_ltao, _eps_p, _tt_p, _fit_p)])
        # companion metric: what a quick-look GAUSSIAN FIT would report on the
        # same PSF (OSIRIS-style; see fwhm_gaussfit_mas)
        _box_mas = float(getattr(args, "fwhm_box_mas", 300.0))
        fwhm_gauss_ngs_bright = np.array(
            [fwhm_gaussfit_mas(s, e, lam_nm, t, f, _nact, box_mas=_box_mas)
             for s, e, t, f in zip(ngs_bright, _eps_d, _tt_b, _fit_d)])
        fwhm_gauss_ngs_faint = np.array(
            [fwhm_gaussfit_mas(s, e, lam_nm, t, f, _nact, box_mas=_box_mas)
             for s, e, t, f in zip(ngs_faint, _eps_d, _tt_f, _fit_d)])
        fwhm_gauss_single = np.array(
            [fwhm_gaussfit_mas(s, e, lam_nm, t, f, _nact, box_mas=_box_mas)
             for s, e, t, f in zip(sr_single, _eps_p, _tt_p, _fit_p)])
        fwhm_gauss_ltao = np.array(
            [fwhm_gaussfit_mas(s, e, lam_nm, t, f, _nact, box_mas=_box_mas)
             for s, e, t, f in zip(sr_ltao, _eps_p, _tt_p, _fit_p)])
        # third convention: same fit, but with a FREE BACKGROUND term (the
        # closer mechanism match to the live interactive "AO Strehl tool";
        # see fwhm_gaussfit_sky_mas)
        fwhm_sky_ngs_bright = np.array(
            [fwhm_gaussfit_sky_mas(s, e, lam_nm, t, f, _nact, box_mas=_box_mas)
             for s, e, t, f in zip(ngs_bright, _eps_d, _tt_b, _fit_d)])
        fwhm_sky_ngs_faint = np.array(
            [fwhm_gaussfit_sky_mas(s, e, lam_nm, t, f, _nact, box_mas=_box_mas)
             for s, e, t, f in zip(ngs_faint, _eps_d, _tt_f, _fit_d)])
        fwhm_sky_single = np.array(
            [fwhm_gaussfit_sky_mas(s, e, lam_nm, t, f, _nact, box_mas=_box_mas)
             for s, e, t, f in zip(sr_single, _eps_p, _tt_p, _fit_p)])
        fwhm_sky_ltao = np.array(
            [fwhm_gaussfit_sky_mas(s, e, lam_nm, t, f, _nact, box_mas=_box_mas)
             for s, e, t, f in zip(sr_ltao, _eps_p, _tt_p, _fit_p)])
        # FOURTH convention: what THIS package's own Measured-SR tab reads --
        # its find_fwhm.pro port run on the model PSF rendered as a detector
        # frame (see fwhm_srtool_mas). The only one of the four directly
        # comparable to a measured number without a convention caveat, so it
        # is what a predicted-vs-delivered join should use.
        fwhm_tool_ngs_bright = np.array(
            [fwhm_srtool_mas(s, e, lam_nm, t, f, _nact)
             for s, e, t, f in zip(ngs_bright, _eps_d, _tt_b, _fit_d)])
        fwhm_tool_ngs_faint = np.array(
            [fwhm_srtool_mas(s, e, lam_nm, t, f, _nact)
             for s, e, t, f in zip(ngs_faint, _eps_d, _tt_f, _fit_d)])
        fwhm_tool_single = np.array(
            [fwhm_srtool_mas(s, e, lam_nm, t, f, _nact)
             for s, e, t, f in zip(sr_single, _eps_p, _tt_p, _fit_p)])
        fwhm_tool_ltao = np.array(
            [fwhm_srtool_mas(s, e, lam_nm, t, f, _nact)
             for s, e, t, f in zip(sr_ltao, _eps_p, _tt_p, _fit_p)])
    else:
        fwhm_gauss_ngs_bright = fwhm_gauss_ngs_faint = None
        fwhm_gauss_single = fwhm_gauss_ltao = None
        fwhm_sky_ngs_bright = fwhm_sky_ngs_faint = None
        fwhm_sky_single = fwhm_sky_ltao = None
        fwhm_tool_ngs_bright = fwhm_tool_ngs_faint = None
        fwhm_tool_single = fwhm_tool_ltao = None

    _res_names = ['col_airmass', 'col_ang1', 'col_cn2', 'col_d0', 'col_dimm', 'col_mass', 'col_mm', 'col_ngs_th0', 'col_r0_cm', 'col_terms', 'col_theta0', 'col_tt10', 'col_zf', 'n_fb', 'ngs_bright', 'ngs_faint', 'ngs_fb', 'p_airmass', 'p_dimm_in', 'p_secs', 'p_times', 'p_zf', 'sr_ltao', 'sr_single', 'th0_assumed', 'times']
    _res_names = _res_names + ['fwhm_ngs_bright', 'fwhm_ngs_faint',
                               'fwhm_single', 'fwhm_ltao',
                               'fwhm_gauss_ngs_bright', 'fwhm_gauss_ngs_faint',
                               'fwhm_gauss_single', 'fwhm_gauss_ltao',
                               'fwhm_sky_ngs_bright', 'fwhm_sky_ngs_faint',
                               'fwhm_sky_single', 'fwhm_sky_ltao',
                               'fwhm_tool_ngs_bright', 'fwhm_tool_ngs_faint',
                               'fwhm_tool_single', 'fwhm_tool_ltao',
                               'tt_ngs_bright', 'tt_ngs_faint', 'fit_ngs']
    _loc = locals()
    return SimpleNamespace(**{k: _loc[k] for k in _res_names})
