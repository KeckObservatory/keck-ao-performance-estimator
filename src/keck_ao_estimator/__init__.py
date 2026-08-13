"""Keck AO Performance Estimator -- engine + GUI package.

This top-level module is the curated public API: it re-exports every name
the historical flat `ao_strehl_timeline.py` engine module used to expose
bare (physics functions, budget/constants, and the pipeline/plots/export/
cli layers), so existing code that did `import ao_strehl_timeline as engine`
can do `import keck_ao_estimator as engine` instead and keep every
`engine.NAME` reference working unchanged.
"""
# ruff: noqa: F401 -- every import in this file is an intentional re-export.
import os
import sys
import warnings

import matplotlib
import numpy as np
from astropy.utils import iers

# All-NaN slices are expected when a night has DIMM but no MASS (NGS-only run);
# numpy's nanmax/nanmean warn on those. Silence just that specific warning.
warnings.filterwarnings("ignore", message="All-NaN", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice",
                        category=RuntimeWarning)
# headless backend; safe on servers. Harmless for the GUI too -- it always
# wraps a Figure in FigureCanvasQTAgg directly rather than going through
# matplotlib.pyplot's backend-dependent state machine.
matplotlib.use("Agg")
# astropy is only needed for the airmass calculation
iers.conf.auto_download = False

from ._version import APP_NAME, CONTACT, MAINTAINER, ORGANIZATION, __version__
from .atmosphere import (
    cn2_density_profile, seeing_to_integrated_cn2, theta0_d0_from_profile,
    zenith_seeing_factor,
)
from .budget import (
    ADJUSTABLE_BUDGET_PARAMS, ANG_REF_OFFSET, BUDGET_DEFAULTS, BUDGET_VERSIONS,
    DEF_LASER_PA_DEG, DEF_LGS_OFFSET, DEFAULT_BUDGET_VERSION, FITTING_ERR,
    LTAO_BW_FLOOR_FRAC,
    LTAO_RATE_SINGLE, LTAO_RATE_TOMO, STATIC_TEL, active_budget_overrides,
    apply_budget_version, budget_overrides, get_budget_param, layer_mismatch,
    lgs_budget_terms, lgs_strehl, ltao_bw_factor, static_subtotal,
)
from .catalogs import (
    CATALOGS, parse_catalog_table, query_guide_stars, stars_field_xy,
)
from .cli import build_parser, main, _cli
from .config import (
    default_output_name, parse_night, parse_windows, resolve_tomography,
    resolve_tt_sensor, resolve_wavelength,
)
from .constants import (
    DEF_AIRMASS_PAD, DEF_CACHE_DIR, DEF_DIMM, DEF_ELEV_CUT, DEF_MASS,
    DEF_MASSPRO, DEF_MATCH_TOL, DEF_NGS_BRIGHT, DEF_NGS_FAINT,
    DEF_NIGHT_DATE, DEF_OUTPUT, DEF_TARGET_DEC, DEF_TARGET_NAME,
    DEF_TARGET_RA, DEF_TELESCOPE, DEF_WINDOWS, DM_ACTUATORS_ACROSS,
    FIELD_FOV_ARCSEC, HST_TO_UTC_HOURS, KECK_HEIGHT_M, KECK_LAT_DEG,
    KECK_LON_DEG, LAMBDA_K_NM, MASS_HEIGHTS_M, MKWC_BASE, MKWC_DIMM_URL,
    MKWC_MASS_URL, MKWC_MASSPRO_URL, MOFFAT_BETA_KOLM, NM_PER_MAS,
    PHOTOMETRIC_BANDS, POINTING_LIMITS, REF_FREEATM, REF_TOTAL,
    TEL_DIAMETER_M, V2K, V_FREE, V_GROUND,
)
from .export import write_csv_table
from .field_stats import (
    GRADIENT_MIN_STARS, THETA0_MIN_STARS, FieldStats, field_statistics,
    theta0_from_ratios,
)
from .fieldmap import (
    field_cn2_profile, field_map_grid, field_metric_at, field_snapshot,
    synthetic_field_snapshot,
)
from .geometry import (
    apply_proper_motion, compute_airmass_curve, hour_angle_hours, in_wedge,
    is_night_at_keck, moon_illumination_fraction, moon_separation_deg,
    parse_radec, pointing_state, sun_altitude_deg,
)
from .gs_ranking import rank_guide_stars
from .image_strehl import (
    CROWDING_WARN_FRAC, EDGE_CLIP_WARN_FRAC, SR_ERR_MAX, Nirc2StrehlResult,
    aperture_edge_clip_frac, aperture_flux, cntrd,
    deadpix_fill, find_peak, load_nirc2_calibration,
    field_consistent, find_stars, fix_image, measure_field, measure_nirc2_frame,
    measure_osiris_frame, measure_strehl, optimize_photometry_radius,
    osiris_reduce,
    radial_profile_fwhm, radius_map, reduce_frame, sigma_clipped_median,
    sigma_filter3,
)
from .ee_correction import ee_calibrate_h, ee_correct, ee_expected_small
from .epsf import (
    EPSF_CONVERGE_TOL, EPSF_DEFAULT_OVERSAMPLE, EPSF_DONOR_MIN_SNR,
    EPSF_ISOLATION_LOOSE_ARCSEC, EPSF_ISOLATION_STRICT_ARCSEC,
    EPSF_COLLAPSE_CROWD_MIN, EPSF_COLLAPSE_PRED_MIN,
    EPSF_CORE_NEG_MAX_FRAC, EPSF_CORE_NEG_RADIUS_FWHM,
    EPSF_GATE_MAX_PREDICTED_FRAC,
    EPSF_GATE_SAMPLE,
    EPSF_MASK_RADIUS_FWHM, EPSF_MAX_DONORS, EPSF_MIN_DONORS,
    EPSF_N_CYCLES, EPSF_NORM_RADIUS_FWHM,
    EPSF_PEAK_CEILING_FRAC, EPSF_WEIGHT_SCALE_ARCSEC, EmpiricalPsf,
    EpsfDonor, EpsfModel, StarCatalog, build_epsf, deep_star_catalog,
    donor_candidates, epsf_strehl, estimate_psf_shape, theoretical_psf,
)
from .psf_fit import (
    PSF_FIT_FOOTPRINT_FWHM, PSF_FIT_MAX_NEIGHBOURS,
    PSF_FIT_NEIGHBOUR_FLOOR_FRAC, PSF_FIT_POS_TOL_FWHM,
    PSF_FIT_BIAS_SAFE_NOTE, PSF_FIT_BIAS_UNSAFE_NOTE,
    PSF_FIT_SIGMA_REJECT, PSF_FIT_SR_ENVELOPE_NOTE,
    PSF_FIT_SR_VALIDATED_MAX, CleanReport, Neighbour, clean_star,
    component_footprint, group_fit, select_neighbours,
)
from .io import fetch_mkwc_files, load_mass_profile, load_seeing_series, parse_dt, parse_secs
from .marechal import marechal_strehl
from .nirc2 import (
    AO_OPS_MODE_NAMES, NIRC2_BG_INNER_RADIUS_ARCSEC,
    NIRC2_BG_OUTER_RADIUS_ARCSEC, NIRC2_FILTER_NAMES,
    NIRC2_FILTER_WAVELENGTH_UM, NIRC2_PEAK_RADIUS_ARCSEC,
    NIRC2_PHOTOMETRY_RADIUS_ARCSEC, NIRC2_PLATE_SCALE_MAS, NIRC2_PUPIL_STOPS,
    Nirc2FrameParams, decode_ao_ops_mode, nirc2_frame_params,
    trick_sensor_active,
)
from .nirc2_psf import nirc2_dl_psf, nirc2_pupil
from .osiris import (
    OSIRIS_FILTER_WAVELENGTH_UM, OSIRIS_PLATE_SCALE_MAS, detect_instrument,
    osiris_frame_params,
)
from .ngs import NGS_K1_QUADCELL_PENALTY, NGS_PARAMS, NGS_SEEING_LAW, NGS_SK_ANCHOR, ngs_strehl
from .photometry import (
    SENSOR_BAND_UM, SENSOR_FAINT_LIMIT, estimate_sensing_mag,
    optical_extinction_lower_bound, pick_mag,
)
from .night_stats import masked_mean, time_selection_mask
from .pipeline import compute_timeline, prepare_night
from .plots import (
    FWHM_COLLAPSE_MULT, FWHM_MIN_SPAN_FRAC, _SplitSwatch, _SplitSwatchHandler,
    _annotate_fwhm_axis, _fwhm_axis_limits, apply_utc_display, gapline,
    overlay_fwhm_on_main, render_fwhm_figure, render_main_figure,
    render_predicted_terms_figure, render_terms_figure, shift_hst_text,
)
#  OPEN_LOOP_TILT_ONEAXIS_MAS / OUTER_SCALE_M are SNAPSHOTS of psf globals
#  that psf.set_outer_scale() rebinds; that function keeps these two
#  re-exports in step deliberately, so `engine.OPEN_LOOP_TILT_ONEAXIS_MAS`
#  never disagrees with what `engine.tt_wfe_nm` is actually applying.
from .psf import (OPEN_LOOP_TILT_ONEAXIS_MAS,
                  OPEN_LOOP_TILT_ONEAXIS_MAS_KOLMOGOROV, OUTER_SCALE_M,
                  fwhm_gaussfit_mas, fwhm_gaussfit_sky_mas, fwhm_srtool_mas,
                  psf_fwhm_mas, set_outer_scale)
from .starlist import (
    entry_float, format_starlist_line, parse_starlist, parse_starlist_text,
    same_star_name, write_starlist,
)
from .target_resolve import resolve_target_name
from .tiptilt import (DEF_LTAO_TT_THETA0_GAIN, DEF_TT_MAG, DEF_TT_OFFSET,
                      NGS_TILT_SERVO_MAS, ngs_tt_nm, tt_wfe_nm)
from .ttstar import (
    RING_TOL_ARCSEC, TRICK_PLATE_SCALE_MAS, TRICK_ROI_CENTER_PX,
    TSS_ARCSEC_PER_MM, TT_ONAXIS_MAX_ARCSEC, best_mag, tt_ring_match,
    tt_star_offset, trick_roi_offset, trick_roi_sky_offset,
)
from .fa_advisory import event_lead_lag, pierce_points, trailing_fa_stats
from .fa_geometry_plot import draw_cone_effect, draw_fa_geometry
from .mkam_catalog import (
    dimm_star_probabilities, load_mkam_catalog, star_altaz,
    top_monitor_orientations,
)
from .vignetting import (
    INSTRUMENT_CENTRE_MM, TSS_CIRCUMSCRIBED_ARCSEC, TSS_INSCRIBED_ARCSEC,
    TSS_TRAVEL_MM, UNVIGNETTED_RADIUS_ARCSEC, VIGNETTE_COEFF, VIGNETTE_EXP,
    VIGNETTE_SAMPLES, VIGNETTE_UNUSABLE_FRAC, field_centre_mm,
    instrument_centre_offset_arcsec, tss_box_reachable, tss_reachable,
    vignetting_fraction, vignetting_mag_penalty, vignetting_note,
)
from .winds import WindsError, night_winds, tau0_seconds
