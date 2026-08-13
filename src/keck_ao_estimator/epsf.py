"""Empirical PSF (ePSF) built from a field's own stars.

Purpose: supply a model of the instrumental+atmospheric PSF good enough to
SUBTRACT a star's neighbours before the aperture measurement runs.  It is
never used to measure a star (`psf_fit.py` explains why: one fitted PSF
shape makes peak/flux a field constant, which is not a per-star Strehl).

Definition (Anderson & King 2000 "effective PSF"): psi(u, v) is the value a
detector pixel whose CENTRE lies at offset (u, v) detector-pixels from the
star centre would record, for a star of unit flux.  Under this definition

    model_pixel(px, py) = amp * psi(px - x_star, py - y_star)

with no pixel-integration step anywhere -- the pixel response is baked into
psi itself.  The oversampled grid exists only to represent psi's sub-pixel
structure; it is never block-averaged.  Construction is the inverse scatter:
each donor pixel is one sample of psi at its own sub-pixel offset, and the
donors' random sub-pixel phases fill the oversampled grid between them.

Conventions that make the model composable with `image_strehl`:

- psi is defined on the SAME sky convention as the measurement: each donor
  stamp has the clipped-median of the [bg_inner, bg_outer] annulus removed
  before stacking, exactly as `aperture_flux(robust=True)` would.  psi
  therefore falls to ~0 near the annulus, which is why truncating the stamp
  at bg_outer costs almost nothing, and why the residual pedestal is
  absorbed by the group fit's free background rather than pretending to be
  a physical halo.
- psi is normalized so that its sum over a unit-spaced detector grid inside
  the stamp is 1.0, i.e. a fitted amplitude IS the star's stamp flux in the
  measurement's own sky convention.
- Arrays are numpy [row, col] = [y, x]; every (x, y) argument and result is
  IDL/detector convention, matching `image_strehl`.

Qt-free by rule (numpy/scipy/astropy only).
"""
from dataclasses import dataclass, field

import numpy as np

from .constants import TEL_DIAMETER_M
from .image_strehl import (
    CROWDING_WARN_FRAC, aperture_flux, cntrd, find_peak, radius_map,
    sigma_clipped_median,
)
from .nirc2 import (
    NIRC2_BG_INNER_RADIUS_ARCSEC, NIRC2_BG_OUTER_RADIUS_ARCSEC,
    NIRC2_PEAK_RADIUS_ARCSEC, NIRC2_PHOTOMETRY_RADIUS_ARCSEC,
)

__all__ = [
    "EPSF_DEFAULT_OVERSAMPLE", "EPSF_DONOR_MIN_SNR", "EPSF_MIN_DONORS",
    "EPSF_MAX_DONORS", "EPSF_N_CYCLES", "EPSF_CONVERGE_TOL",
    "EPSF_WEIGHT_SCALE_ARCSEC", "EPSF_ISOLATION_STRICT_ARCSEC",
    "EPSF_ISOLATION_LOOSE_ARCSEC", "EPSF_PEAK_CEILING_FRAC",
    "EPSF_MIN_PHASE_COVERAGE", "EPSF_ISOLATION_MIN_SNR",
    "EpsfDonor", "EpsfModel", "EmpiricalPsf",
    "build_epsf", "epsf_strehl", "donor_candidates", "deep_star_catalog",
    "theoretical_psf", "estimate_psf_shape",
    "StarCatalog",
]

# --- defaults, each with the reason it has this value -----------------
#
# oversampling: NIRC2 narrow at Kp samples lambda/D = 43.8 mas with 9.942
# mas pixels = 4.41 px per lambda/D, so the detector is already 2.2x
# Nyquist; psi at 2x oversampling carries ~9 samples across the DL core
# FWHM (1.03 lambda/D = 4.5 px), which resolves every structure the
# detector actually recorded.  Going finer costs N^2 in fit evaluation and
# demands more donors for sub-pixel phase coverage without adding
# information.  The WIDE camera (39.686 mas/px = 1.10 px per lambda/D) is
# genuinely sub-Nyquist and 2x may not be enough; that is measured in S4,
# not assumed here (PLAN Section 5).
EPSF_DEFAULT_OVERSAMPLE = 2

# A donor is a MODEL, not a detection -- find_stars' own floors
# (min_snr=10) are detection floors and far too permissive here.  The
# value is DERIVED, not argued (D37): swept 50/35/25/20/15/10 at moderate
# density over 6 seeds, recording both donor supply and psi quality.
# Degradation crosses between 35 and 25 -- mean delta jumps 0.0038 ->
# 0.7652, the same non-convergence signature as the Galactic-Centre
# failure in D30 -- while donor supply saturates at 25 (the field has no
# fainter stars to admit).  35 is the last floor whose builds converge.
#
# The window this bounds from below is narrow: with the
# EPSF_PEAK_CEILING_FRAC ceiling above, it spans ~2.5 mag, and that width
# is what limits donor supply in EVERY density class (D36).
EPSF_DONOR_MIN_SNR = 35.0

# donor peak ceiling as a fraction of the frame saturation level
# (params.max_counts * params.coadds).  Detector response departs from
# linear well below the hard ceiling -- the GC 20260728 frames measurably
# flatten before it -- and a flattened donor core biases psi's peak LOW,
# which under-subtracts every neighbour at exactly the pixel that matters
# most.  Donors are plentiful; be conservative.  Callers with a per-frame
# measured onset (the `nirc2_reduce.saturation_onset` peak-vs-flux curve)
# should pass `donor_peak_max` explicitly instead.
EPSF_PEAK_CEILING_FRAC = 0.5

# DERIVED (D39), where the previous 5 was a round number from O1.  Varying
# ONLY the donor count on a fixed field and cleaning a fixed S2 pair set
# with each psi_k (4 seeds, 24 measurements per k) gives a cleaned-SR bias
# that is FLAT in k: every k from 2 to 9 lands inside the S2 target of
# 0.02, and k=4 is the best of them (worst |dSR| 0.0167 vs 0.0175 at 5).
# `delta` does improve 7x from k=2 to k=9, but that is an INTERNAL
# convergence statistic and it does not translate into SR accuracy -- the
# old threshold was gating on the wrong quantity.
#
# Not lower than 4, despite k=2 and k=3 measuring fine there: that
# experiment used a CLEAN donor frame, and the stack is sigma-clipped
# across the donors reaching each cell, so rejecting a contaminated sample
# needs redundancy.  Contaminated donors are the regime that matters and
# the experiment says nothing about it.  4 keeps a rejection margin and is
# the measured optimum anyway.
EPSF_MIN_DONORS = 4

# D31 -- catalogue-masked stacking.  Pixels within this many FWHM of ANY
# catalogued star other than the donor itself contribute NO samples to the
# stack.  Neighbours then contribute nothing rather than light, which
# breaks the D30 chain at step 1: blended donors -> wing-heavy psi ->
# suppressed peak_value -> amp = peak/psi_peak inflated past 100 % of the
# aperture flux.  Different donors have different neighbour geometries, so
# the sigma-clipped stack fills each masked cell from the donors that are
# clean there; only a cell masked in EVERY donor is lost, and the existing
# phase-coverage guard already reports that.
EPSF_MASK_RADIUS_FWHM = 1.5

# Donors must be normalized over a region that masking can never eat, or
# a donor with a masked neighbour enters the stack on a different scale
# from one without -- the same failure that made partially off-array
# donors unusable (D40).  The isolation cut guarantees no CATALOGUED star
# within isolation_arcsec (0.25" = 25 px at worst on NIRC2 narrow), and a
# mask reaches only EPSF_MASK_RADIUS_FWHM * FWHM (~7 px) back toward the
# donor, so a radius of 2 FWHM (~9 px) is always clean.  The absolute
# scale is free -- psi -> c*psi is absorbed exactly by the fitted
# amplitude (D24) -- so only cross-donor CONSISTENCY matters.
EPSF_NORM_RADIUS_FWHM = 2.0

# D46 -- the model-level over-amplitude gate.  D30's confirmed pathology
# is `clean_star` refusing star after star with predicted neighbour light
# at 98.9 %, 160.4 %, 868.9 % OF THE TARGET'S OWN APERTURE FLUX -- values
# above 100 % are physically impossible and mean the model's amplitudes
# are wrong, not that the field is crowded.  Detecting it per star costs
# ~4 s each and produces N identical refusals; the same quantity comes
# from `select_neighbours` alone, with NO fitting, so it can be sampled
# once per FIELD and reported in one line.
#
# The threshold is the physical one, not a fitted one: predicted
# neighbour light cannot exceed the target's own flux.  Measured medians
# over the brightest EPSF_GATE_SAMPLE stars -- sound builds (clean donor
# frames at sr 0.15/0.30/0.60, moderate fields) span 0.000-0.797; real
# crowded frames with the veto relaxed span 0.814-5.796.  A gate at 1.0
# clears every sound build with margin and catches the GC-class builds
# whose amplitudes have actually broken.
#
# NOT a psi-quality gate: psi quality was measured not to bind (D42) and
# `delta` is retired (D39), so this tests the amplitude pathology only.
EPSF_GATE_MAX_PREDICTED_FRAC = 1.0
EPSF_GATE_SAMPLE = 12

# Over-subtraction guard: the deepest NEGATIVE excursion allowed inside
# EPSF_CORE_NEG_RADIUS_FWHM of psi's centre, as a fraction of psi's peak.
#
# DEPTH, not count: sound builds do contain negative cells there (88 of
# them at sr=0.60 on a clean S2 donor frame), because a sharp core leaves
# the outer part of the disc carrying little signal. And the radius
# matters -- the damage does NOT live in the inner core. Measured depth
# vs radius, sound builds against a known runaway (hand-picked M92
# donors, cycle 2):
#
#   r/FWHM        2       3       4       6
#   sound S2 .30  0.017   0.021   0.022   0.031
#   sound S2 .60  0.043   0.043   0.043   0.043
#   sound M92 c1  0.000   0.011   0.015   0.015
#   RUNAWAY  c2   0.069   0.087   0.118   0.118
#
# At 2 FWHM the classes nearly touch (0.043 vs 0.069); by 4 FWHM they
# separate 0.043 vs 0.118. Hence radius 4 and a threshold of 0.08,
# roughly midway in log space.
#
# HONEST LIMIT (D51): that is a 2.7x separation from FOUR samples, which
# is thin -- the same weakness that made me discard D46's first metric.
# It is shipped anyway because the failure is ASYMMETRIC: exceeding the
# threshold makes the build keep the PREVIOUS cycle's model, and D42
# measured psi quality as barely affecting the answer, so stopping a
# cycle early is nearly free while a runaway is fatal. Erring toward
# stopping is cheap insurance, not a tuned optimum. Widen the sample
# before treating 0.08 as physics.
# D52 -- the OTHER side of the D46 gate. D46 catches a model predicting
# MORE neighbour light than the aperture holds. A COLLAPSED model has the
# opposite signature: it predicts almost nothing, and sails through.
# Measured on the n0290 runaway, pred/flux 0.021 and 0.000 -- "passes".
#
# The referee is `crowding`, which is MODEL-FREE (annulus mean-vs-clipped
# disagreement, no PSF involved). Sound builds on UNCROWDED fields also
# predict ~0, and correctly so, which is why the test cannot key on the
# prediction alone:
#
#   case                    pred    crowd
#   sound S2 sr=0.30        0.000   0.013
#   sound S2 sr=0.60        0.000   0.035
#   sound moderate seed 0   0.797   0.015
#   sound moderate seed 6   0.053   0.032
#   COLLAPSED n0290         0.021   0.643
#   COLLAPSED n0290 (k=6)   0.000   0.643
#
# So: only when the field is UNAMBIGUOUSLY crowded (median crowding above
# EPSF_COLLAPSE_CROWD_MIN, four times the tool's own CROWDING_WARN_FRAC)
# is a near-zero prediction treated as collapse.
#
# DELIBERATELY NARROW. Unlike the D51 guard, a false positive here is NOT
# cheap -- it would refuse on exactly the crowded fields the feature
# exists for. And there is no sound-model-on-a-crowded-field exemplar to
# calibrate against, because those are the builds that do not yet
# succeed. So the bound is set to catch only the egregious case and no
# more; it must be re-derived the moment such an exemplar exists.
EPSF_COLLAPSE_CROWD_MIN = 0.20
EPSF_COLLAPSE_PRED_MIN = 0.05

EPSF_CORE_NEG_RADIUS_FWHM = 4.0
EPSF_CORE_NEG_MAX_FRAC = 0.08
EPSF_MAX_DONORS = 15        # sqrt(15) = 3.9x SNR; more only adds fainter,
                            # more crowded donors and stamp memory
EPSF_N_CYCLES = 3           # extract -> clean -> re-register, 3 passes
EPSF_CONVERGE_TOL = 0.01    # sum|psi_k - psi_(k-1)| / sum psi_(k-1)

# distance weighting toward the target.  The EE campaign measured the halo
# parameter h drifting 0.42 -> 0.51 from on-axis to 30" off, i.e. a ~20%
# shape change over 30": the PSF-shape correlation length is tens of
# arcsec, not arcsec.  A Lorentzian w = 1/(1 + (d/d0)^2) with d0 = 10"
# down-weights a 30" donor ~10x relative to a co-located one and never
# zeroes anyone (no cliff when the near donors run out).  On a NIRC2
# narrow frame (10.2" across) the weighting is nearly flat -- correct,
# since shape variation over 10" is small -- while medium/wide fields get
# real weighting.
EPSF_WEIGHT_SCALE_ARCSEC = 10.0

# donor isolation ladder (mirrors the EE feature's strict -> LOOSE ->
# UNCALIBRATED, tags and all -- never a silent skip)
EPSF_ISOLATION_STRICT_ARCSEC = 0.5      # no catalogued neighbour this close
EPSF_ISOLATION_LOOSE_ARCSEC = 0.25      # ~5.7 lambda/D at Kp: outside the
                                        # first few Airy rings, so what is
                                        # left is cleanable by iteration

# minimum fraction of oversampled cells that must receive a real donor
# sample before the model counts as fully calibrated (D12).  Donors fill
# one sub-pixel phase class each (D11), so coverage measures how much of
# psi is MEASURED rather than interpolated by the empty-cell fill.  Below
# this the model is provisional and the rung is downgraded, exactly as
# for a build that did not converge -- and unlike convergence, which is
# blind to it: a field whose donors share a phase converges beautifully
# onto a model that is three-quarters interpolation.
EPSF_MIN_PHASE_COVERAGE = 0.5

# SNR a catalogue entry needs before it may DISQUALIFY a donor on the
# isolation cut.  The deep catalogue serves two purposes that want opposite
# errors: neighbour subtraction wants depth (a missed contaminant is an
# uncorrected bias), while donor isolation wants reliability (a spurious
# detection next to a good donor throws the donor away).  Running the
# catalogue at 5 sigma for the first purpose made the second fail outright
# -- on the S4d ladder field, 16 spurious entries appeared and every one of
# the 8 real donors was then rejected "not isolated", leaving the build
# uncalibrated.  So isolation is judged only against entries at or above
# this floor, the threshold the catalogue itself used to run at.  Anything
# fainter is both untrustworthy as a veto and negligible as a contaminant
# of the donor's own stamp -- and the build's iteration subtracts it anyway.
EPSF_ISOLATION_MIN_SNR = 8.0


class StarCatalog(list):
    """The neighbour catalogue, which knows whether its cap bound.

    A plain `list` subclass on purpose: every existing caller and driver
    keeps working unchanged, and the extra fact is there for anything that
    looks.  The alternative -- returning a dataclass -- would have churned
    two engine callers and eleven committed drivers for one boolean.

    Why the boolean has to exist at all: hitting `n_max` returned a list
    INDISTINGUISHABLE from "the field ran out of stars".  The other two
    exits are physical floors (SNR, and the diffraction-halo relative
    floor); the cap is not, and an uncatalogued neighbour is simply never
    subtracted.  NIRC2 narrow at GC density holds ~295 stars against a
    400 cap -- a 35 % margin -- so this can bind on real data today, and
    an OSIRIS field (20.4", ~1180 stars at the same density) would blow
    straight through it.  Never silent (RULES section 5).
    """

    truncated = False
    n_max = 0

    @property
    def cap_note(self):
        return ("" if not self.truncated else
                f" CATALOGUE TRUNCATED at n_max={self.n_max}: neighbours "
                f"beyond it are NOT subtracted, and the field is denser "
                f"than the cap allows -- raise n_max for this field.")


@dataclass(frozen=True)
class EpsfDonor:
    """One registered, cleaned, normalized donor -- as a SPARSE sample set.

    A donor is not a dense stamp and cannot be (D11).  Every pixel of one
    star sits at the SAME sub-pixel phase relative to that star's centre,
    because pixel offsets differ by exact integers.  One donor therefore
    populates exactly ONE of the `oversample**2` phase classes of the
    oversampled grid and leaves the rest empty -- 75% empty at
    oversample=2, 94% at oversample=4.  Phase diversity across donors is
    what fills the grid in; that is the whole mechanism of an empirical
    PSF, and it only works if the donors are combined as samples.

    So a donor carries its samples, not a picture of itself:

      `cell`   int32 (n,) -- flat index into the oversampled grid
      `value`  float32 (n,) -- (pixel - sky) / flux at that offset
      `weight` float32 (n,) -- bilinear share of this sample for that cell

    Samples are scattered BILINEARLY (D21), not binned to the nearest cell.
    Nearest-cell binning quantizes each sample's sub-pixel offset by up to
    +/-1/(2*oversample) -- +/-0.25 detector px at oversample 2 -- which
    distorts psi in a phase-dependent way and was measured to make the
    amplitude a fit needs vary by 5.2% with nothing but the star's sub-pixel
    phase.  Raising oversample shrinks the quantization but starves cell
    coverage (9 donors reached only 50% of the grid at oversample 4);
    bilinear scatter removes the quantization AND spreads each sample over
    4 cells, so coverage improves with oversampling instead of collapsing.
    It is the exact adjoint of the bilinear gather.

    `cell` is computed from the donor's FINAL registered position, so
    re-registration between cycles moves the samples, not just a label.
    """
    x: float                # final registered detector position
    y: float
    peak: float             # sky-subtracted core max, ADU (linearity gate)
    flux: float             # stamp flux, measurement sky convention
    sky: float
    crowding: float         # existing annulus-contamination metric
    isolation_arcsec: float # distance to the nearest catalogued neighbour
    n_cleaned: int          # neighbours subtracted from this donor's stamp
    reg_dx: float           # final registration shift from the cntrd seed
    reg_dy: float
    cell: np.ndarray = field(repr=False)
    value: np.ndarray = field(repr=False)
    weight: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class EpsfModel:
    """A concrete, evaluable PSF at one field point.

    Holds the spline-PREFILTERED oversampled grid (scipy.ndimage.
    spline_filter applied once at construction) so that every evaluation
    is a `map_coordinates(..., order=3, prefilter=False)` call: the fit
    evaluates the model thousands of times and re-prefiltering per call
    would dominate the runtime.
    """
    grid: np.ndarray = field(repr=False)        # prefiltered, oversampled
    grad_y: np.ndarray = field(repr=False)      # prefiltered d psi/d y
    grad_x: np.ndarray = field(repr=False)      # prefiltered d psi/d x
    oversample: int
    r_stamp_px: float
    fwhm_px: float
    ee_photrad: float       # fraction of stamp flux inside the phot radius
    peak_value: float = 0.0  # psi's peak on a unit-spaced detector grid, so
                             # amp ~ star_peak / peak_value converts a
                             # measured core height into a model amplitude

    @property
    def n_half(self):
        """Half-size of the oversampled grid, in CELLS."""
        return (self.grid.shape[0] - 1) // 2

    def _interp(self, grid, yy, xx, x, y):
        from scipy.ndimage import map_coordinates
        u = np.asarray(xx, dtype=float) - float(x)
        v = np.asarray(yy, dtype=float) - float(y)
        # Interpolate ONLY where the component can be non-zero.  The stamp
        # cut below is not a post-hoc tidy-up: the model carries no flux
        # past r_stamp_px by construction, so those samples were always
        # going to be overwritten with 0.0.  Computing them first is pure
        # waste, and in a group fit it is the dominant waste -- the fit
        # footprint is the UNION of every component's neighbourhood, so
        # each component was being evaluated across all the others' pixels
        # too (measured on a Galactic-Centre frame: 16 components sharing
        # one footprint, ~90 % of each column discarded).
        #
        # BIT-IDENTICAL: map_coordinates with prefilter=False is pointwise
        # -- each output sample is a fixed cubic B-spline combination of
        # grid values chosen by its own coordinate alone -- so evaluating
        # a subset returns exactly the values the full call returned at
        # those positions.  The omitted samples keep the 0.0 they were
        # assigned anyway.
        inside = (u * u + v * v) <= self.r_stamp_px ** 2
        out = np.zeros(u.shape, dtype=float)
        if not inside.any():
            return out
        out[inside] = map_coordinates(
            grid,
            [v[inside] * self.oversample + self.n_half,
             u[inside] * self.oversample + self.n_half],
            order=3, prefilter=False, mode="constant", cval=0.0)
        return out

    def evaluate(self, shape, x, y, amp=1.0, origin=(0, 0)):
        """Render `amp * psi` centred at detector (x, y) onto `shape`.

        `origin` is the (y0, x0) detector index of `shape`'s [0, 0] pixel,
        so a caller can render onto a cutout without shifting coordinates.
        Pixels farther than r_stamp_px from the centre are exactly 0.0.
        """
        ny, nx = shape
        yy, xx = np.mgrid[origin[0]:origin[0] + ny, origin[1]:origin[1] + nx]
        flat = self.evaluate_at(yy.ravel(), xx.ravel(), x, y, amp=amp)
        return flat.reshape(ny, nx)

    def evaluate_at(self, yy, xx, x, y, amp=1.0):
        """Same, but at an explicit list of detector pixel coordinates.

        `yy`, `xx` are 1-D arrays of the same length (the fit footprint);
        returns a 1-D array.  This is the hot path of the group fit.
        """
        return float(amp) * self._interp(self.grid, yy, xx, x, y)

    def gradient_at(self, yy, xx, x, y, amp=1.0):
        """(d/dx, d/dy) of `evaluate_at` -- the analytic Jacobian columns.

        Interpolated from the pre-computed, pre-filtered gradient grids
        rather than finite-differenced, so the group fit costs 3
        interpolations per component instead of 3 model evaluations per
        parameter.  Sign convention: derivative with respect to the
        COMPONENT position (x, y), i.e. d/dx psi(px - x) = -psi'(u), and
        the stored grids hold d psi / d CELL, hence the oversample factor.
        """
        s = -float(amp) * self.oversample
        return (s * self._interp(self.grad_x, yy, xx, x, y),
                s * self._interp(self.grad_y, yy, xx, x, y))

    def detector_stamp(self, npix=None):
        """psi resampled onto a unit-spaced detector grid, centred.

        Used by `epsf_strehl` and for FWHM: the returned array is what an
        infinitely bright, noiseless star at the exact pixel centre would
        look like, so `find_peak` / `aperture_flux` / `radial_profile_fwhm`
        apply to it unchanged.
        """
        if npix is None:
            npix = 2 * int(np.ceil(self.r_stamp_px)) + 1
        npix = int(npix) | 1                    # keep the centre on a pixel
        c = npix // 2
        return self.evaluate((npix, npix), c, c)


@dataclass(frozen=True)
class EmpiricalPsf:
    """The field's ePSF: the donor set plus the rules for combining it.

    `tag` is the ladder rung, mirroring the EE feature's vocabulary:

      "strict"       -- >= min_donors donors with no catalogued neighbour
                        inside isolation_strict and crowding <= the
                        existing CROWDING_WARN_FRAC.
      "loose"        -- the strict cut yielded too few donors, so the
                        isolation requirement was relaxed to
                        isolation_loose and the donors' own neighbours were
                        subtracted during the build.  Also the rung an
                        otherwise-strict build is DOWNGRADED to when the
                        iteration failed to converge.
      "uncalibrated" -- fewer than min_donors even at loose.  `donors` is
                        empty, `usable` is False, and callers MUST skip
                        cleaning and say so.  There is no silent path.

    `note` always carries a human sentence naming the counts and the
    reason, ready to go straight into a GUI log line.
    """
    donors: tuple
    tag: str
    note: str
    usable: bool
    oversample: int
    r_stamp_px: float
    plate_scale_mas: float
    weight_scale_px: float
    n_candidates: int
    n_cycles_run: int
    converged: bool
    delta: float                    # final convergence residual
    fwhm_px: float                  # of the unweighted field model
    donor_peak_max: float           # ceiling actually used, ADU
    isolation_used_arcsec: float
    phase_coverage: float           # fraction of oversampled cells that
                                    # got >= 1 donor sample (see D11)
    n_filled_cells: int             # cells that had none and were filled
    photrad_px: float               # photometry radius the ee_photrad
                                    # diagnostic is reported against
    delta_pixel: float = float("nan")   # pixel-wise L1 change, DIAGNOSTIC
                                        # only -- `delta` (encircled) is
                                        # what gates convergence (D18)
    reasons: dict = field(default_factory=dict, repr=False)

    def at(self, x=None, y=None):
        """Distance-weighted `EpsfModel` for a target at detector (x, y).

        Weights w_i = 1 / (1 + (d_i / weight_scale_px)^2) are applied to
        the donors' SAMPLES, which are then accumulated into the
        oversampled grid, robustly combined per cell, and the cells no
        donor reached are filled (D11).  Weighting the samples rather
        than dense per-donor stamps is what keeps the weighting from
        varying between sub-pixel phase classes.

        x=y=None returns the unweighted field model.  Results are cached
        on position rounded to 1 arcsec, so `measure_field` pays the
        recombination once per neighbourhood, not once per star.
        """
        fixed = self.__dict__.get("_fixed_model")
        if fixed is not None:
            return fixed          # theoretical model: same everywhere
        if not self.usable or not self.donors:
            raise ValueError(
                "ePSF is unusable (tag=%r); callers must skip cleaning "
                "and say so rather than asking for a model" % self.tag)
        bin_px = 1000.0 / self.plate_scale_mas          # 1 arcsec
        key = (None if x is None or y is None
               else (round(float(x) / bin_px), round(float(y) / bin_px)))
        cache = self.__dict__.setdefault("_model_cache", {})
        if key in cache:
            return cache[key]

        if key is None:
            w = np.ones(len(self.donors))
        else:
            d = np.array([np.hypot(dn.x - x, dn.y - y)
                          for dn in self.donors])
            w = 1.0 / (1.0 + (d / self.weight_scale_px) ** 2)
        grid_n = 2 * int(np.ceil(self.r_stamp_px)) * self.oversample + 1
        g, _cov, _nf = _assemble(self.donors, w, grid_n, self.oversample,
                                 self.r_stamp_px, 3.0)
        model = _make_model(g, self.oversample, self.r_stamp_px,
                            self.photrad_px)
        cache[key] = model
        return model


# ------------------------------------------------------------- internals

def dl_fwhm_px(params):
    """Diffraction-limited core FWHM in detector pixels, 1.03 lambda/D.

    Used for the detection exclusion radius, the fit footprint and the
    position bounds -- all of which want "how big is a star here", and all
    of which would otherwise hardcode a narrow-camera Kp number."""
    lam_over_d_mas = (float(params.effwave_um) * 1e-6 / TEL_DIAMETER_M
                      * 206264.806 * 1000.0)
    return 1.03 * lam_over_d_mas / float(params.plate_scale_mas)


def _box(shape, x, y, r):
    """Integer slices of the square that contains the radius-r disc, and
    the (yy, xx) coordinate grids over it -- clipped to the array."""
    ny, nx = shape
    y0 = max(int(np.floor(y - r)), 0)
    y1 = min(int(np.ceil(y + r)) + 1, ny)
    x0 = max(int(np.floor(x - r)), 0)
    x1 = min(int(np.ceil(x + r)) + 1, nx)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return (slice(y0, y1), slice(x0, x1)), yy, xx


def _predicted_frac(work, params, model, catalog, photrad_px,
                    bg_inner_arcsec, bg_outer_arcsec, n_sample,
                    photometry_radius_arcsec=NIRC2_PHOTOMETRY_RADIUS_ARCSEC):
    """Median predicted-neighbour-light / own-aperture-flux over the
    brightest `n_sample` catalogued stars (D46).

    Uses `select_neighbours`' own prediction and NO fitting, so it costs a
    handful of aperture integrals rather than a group fit per star -- the
    point being to detect the D30 amplitude pathology once for the field
    instead of N times at ~4 s each.
    """
    from .psf_fit import select_neighbours
    ps = float(params.plate_scale_mas)
    r_in = bg_inner_arcsec * 1000.0 / ps
    r_out = bg_outer_arcsec * 1000.0 / ps
    stars = sorted(catalog, key=lambda c: -float(c.get("peak", 0.0)))
    fracs = []
    for c in stars[:int(n_sample)]:
        x, y = float(c["x"]), float(c["y"])
        flux = aperture_flux(work, photrad_px, x, y, insky_px=r_in,
                             outsky_px=r_out, robust=True)[0]
        if not np.isfinite(flux) or flux <= 0.0:
            continue
        # The prediction MUST integrate over the same aperture the flux
        # was measured in. `select_neighbours` defaults to
        # NIRC2_PHOTOMETRY_RADIUS_ARCSEC (1.0"), so calling it
        # positionally compared a prediction over 1.0" against a flux
        # over the caller's radius -- and `measure_strehl` passes an
        # auto-optimized radius here, which on crowded fields is
        # SMALLER. At 0.6" that inflates gate_frac by ~(1.0/0.6)^2 = 2.8x
        # from the mismatch alone, refusing sound builds as
        # "over-amplitude" on exactly the crowded fields the gate exists
        # to serve. `clean_star` forwards these correctly; this now does
        # the same.
        kept, dropped = select_neighbours(
            catalog, (x, y), params, model, flux,
            photometry_radius_arcsec=photometry_radius_arcsec,
            bg_outer_arcsec=bg_outer_arcsec)
        pred = (sum(r["predicted"] for r in kept)
                + sum(r["predicted"] for r in dropped))
        fracs.append(pred / flux)
    return float(np.median(fracs)) if fracs else float("nan")


def _robust_sky(image):
    sky = sigma_clipped_median(image)
    sig = 1.4826 * float(np.median(np.abs(np.asarray(image) - sky)))
    if not np.isfinite(sig) or sig <= 0.0:
        sig = float(np.std(image)) or 1.0
    return float(sky), float(sig)


def _core_peak(image, x, y, fwhm_px):
    """Sky-free core maximum, raw (no sinc deconvolution).

    Deliberately the RAW max, matching `measure_strehl`'s saturation
    check: this number is compared against a detector linearity ceiling,
    and `find_peak`'s pixelation correction would inflate it past what the
    detector actually recorded."""
    r = max(1.0, 0.75 * fwhm_px)
    sl, _, _ = _box(np.asarray(image).shape, x, y, r)
    sub = np.asarray(image)[sl]
    return float(sub.max()) if sub.size else 0.0


def _combine_samples(donors, weights, n_cells, clip_sigma):
    """Weighted, sigma-clipped per-cell combination of donor SAMPLES.

    This is the operation D11 exists for: weights multiply the samples,
    not dense per-donor stamps, so every cell's estimate is drawn from the
    same weighting law regardless of which sub-pixel phase class it is in.

    Returns (values, filled_mask) with `values` NaN where no donor reached
    the cell at all -- the caller fills those and reports how many.
    """
    def _accumulate(keepers):
        w = np.zeros(n_cells)
        wv = np.zeros(n_cells)
        wv2 = np.zeros(n_cells)
        for d, wi, keep in zip(donors, weights, keepers):
            cell = d.cell if keep is None else d.cell[keep]
            val = d.value if keep is None else d.value[keep]
            sw = d.weight if keep is None else d.weight[keep]
            if cell.size == 0:
                continue
            # per-sample bilinear share x the donor's distance weight
            ww = np.asarray(sw, dtype=float) * float(wi)
            w += np.bincount(cell, weights=ww, minlength=n_cells)
            wv += np.bincount(cell, weights=ww * val, minlength=n_cells)
            wv2 += np.bincount(cell, weights=ww * val * val,
                               minlength=n_cells)
        return w, wv, wv2

    w, wv, wv2 = _accumulate([None] * len(donors))
    hit = w > 0
    mean = np.full(n_cells, np.nan)
    mean[hit] = wv[hit] / w[hit]

    if clip_sigma and clip_sigma > 0:
        var = np.zeros(n_cells)
        var[hit] = np.maximum(wv2[hit] / w[hit] - mean[hit] ** 2, 0.0)
        std = np.sqrt(var)
        # only clip where there is enough support for a scatter to mean
        # anything; two samples can always look like an outlier pair
        n_hit = np.zeros(n_cells)
        for d in donors:
            # count only samples with a MAJORITY share in this cell; a
            # bilinear tail of 0.02 is not independent support for a clip
            sig = d.weight >= 0.25
            if sig.any():
                n_hit += np.bincount(d.cell[sig], minlength=n_cells)
        clippable = hit & (n_hit >= 3) & (std > 0)
        keepers = []
        for d in donors:
            m = mean[d.cell]
            s = std[d.cell]
            c = clippable[d.cell]
            keepers.append(~c | (np.abs(d.value - m) <= clip_sigma * s))
        w, wv, _ = _accumulate(keepers)
        hit = w > 0
        mean = np.full(n_cells, np.nan)
        mean[hit] = wv[hit] / w[hit]
    return mean, hit


def _fill_empty(grid, hit):
    """Fill cells no donor reached, by the iterative 4-neighbour mean.

    Same shape as `fix_image`'s repair loop (image_strehl.fix_image), and
    for the same reason: it is a local, conservative fill that cannot
    invent structure.  Non-optional -- with 5 donors at oversample 2 a
    given phase class is empty ~24 % of the time (D11)."""
    g = np.where(hit, np.nan_to_num(grid), 0.0)
    good = hit.astype(float)
    for _ in range(4 * grid.shape[0]):
        if good.all():
            break
        n_good = (np.roll(good, 1, 0) + np.roll(good, -1, 0)
                  + np.roll(good, 1, 1) + np.roll(good, -1, 1))
        n_val = (np.roll(g, 1, 0) + np.roll(g, -1, 0)
                 + np.roll(g, 1, 1) + np.roll(g, -1, 1))
        fillable = (good == 0) & (n_good > 0)
        if not fillable.any():
            break
        g[fillable] = n_val[fillable] / n_good[fillable]
        good[fillable] = 1.0
    return g


def _disc_mask(grid_n, oversample, r_stamp_px):
    """Cells of the oversampled grid that lie inside the stamp disc.

    The grid is square but psi is only defined on the inscribed disc, so
    the corners (out to r_stamp*sqrt(2)) are OUTSIDE the model.  They must
    be held at exactly zero: they carry no samples, so the empty-cell fill
    would otherwise extrapolate into them, and the normalization would
    then divide by a sum that includes the invention."""
    n_half = (grid_n - 1) // 2
    ax = (np.arange(grid_n) - n_half) / float(oversample)
    return np.hypot(ax[None, :], ax[:, None]) <= r_stamp_px


def _core_negative_depth(grid, oversample, r_core_px):
    """Deepest NEGATIVE excursion inside the core disc, as a fraction of
    psi's peak.  Returns 0.0 when the core is entirely non-negative.

    DEPTH, not count: sound models DO contain negative core cells --
    measured on clean S2 donor frames, 0 at sr=0.15 but 88 at sr=0.60,
    because a sharper core leaves the outer part of the disc carrying
    little signal.  What distinguishes over-subtraction is how DEEP the
    hole is relative to the star: clean builds reach 0.0386 of the peak
    at worst, while a runaway digs holes comparable to or deeper than
    the peak itself.  This is a physical test -- a PSF core is positive
    -- unlike `delta`, which D39/D42 retired as a quality measure after
    it moved 400x while cleaned-SR bias moved 9 %.
    """
    n = grid.shape[0] // 2
    ax = (np.arange(grid.shape[0]) - n) / float(oversample)
    rr = np.hypot(ax[None, :], ax[:, None])
    core = grid[rr <= r_core_px]
    peak = float(grid.max())
    if peak <= 0.0 or core.size == 0:
        return float("inf")
    return float(abs(min(0.0, float(core.min()))) / peak)


def _assemble(donors, weights, grid_n, oversample, r_stamp_px, clip_sigma):
    """Samples -> a normalized psi grid, plus its coverage diagnostics.

    Shared by `build_epsf`'s stacking and `EmpiricalPsf.at`'s
    distance-weighted recombination so the two cannot drift apart.
    Returns (grid, phase_coverage, n_filled) with coverage measured over
    the DISC only -- counting the corners would dilute the number with
    cells that are meant to be empty."""
    disc = _disc_mask(grid_n, oversample, r_stamp_px)
    mean, hit = _combine_samples(donors, weights, grid_n * grid_n,
                                 clip_sigma)
    mean = mean.reshape(grid_n, grid_n)
    hit = hit.reshape(grid_n, grid_n)
    # outside the disc psi is known to be zero -- that is data, not a gap
    mean = np.where(disc, mean, 0.0)
    g = _fill_empty(mean, hit | ~disc)
    g = np.where(disc, g, 0.0)
    total = g.sum()
    if total > 0:
        g = g * (oversample ** 2) / total
    n_disc = int(disc.sum()) or 1
    return (g, float((hit & disc).sum()) / n_disc,
            int((~hit & disc).sum()))


def _encircled(grid, oversample, r_stamp_px):
    """Radially integrated profile of psi, in detector-pixel radii.

    The convergence statistic (D18).  A pixel-wise L1 difference is NOT
    scale-invariant: it is dominated by core pixels, so the same sub-pixel
    re-registration registers as a large change when the core is sharp.
    Measured on an isolated-donor field at sr=0.60, pixel-wise delta rose
    0.1965 -> 0.3318 while the encircled profile moved only 0.0022 ->
    0.0039 and the donors' registration shifts were SHRINKING (0.082 ->
    0.058 px) -- the iteration was converging and the metric said
    otherwise, which downgraded a sound model to `loose`/PROVISIONAL.
    Integrating over annuli removes the sub-pixel sensitivity while still
    responding to any real change in the model's shape.
    """
    n = grid.shape[0]
    n_half = (n - 1) // 2
    ax = (np.arange(n) - n_half) / float(oversample)
    rr = np.hypot(ax[None, :], ax[:, None])
    radii = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 14.0, 27.0])
    radii = radii[radii <= max(r_stamp_px, 1.0)]
    if radii.size == 0:
        radii = np.array([max(r_stamp_px, 1.0)])
    return np.array([float(grid[rr <= r].sum()) for r in radii])


def _profile_delta(new_grid, old_grid, oversample, r_stamp_px):
    a = _encircled(new_grid, oversample, r_stamp_px)
    b = _encircled(old_grid, oversample, r_stamp_px)
    denom = float(np.abs(b).sum()) or 1.0
    return float(np.abs(a - b).sum() / denom)


def _make_model(grid_2d, oversample, r_stamp_px, photrad_px):
    """Prefilter psi and its gradients ONCE and wrap them in an EpsfModel."""
    from scipy.ndimage import spline_filter

    g = np.asarray(grid_2d, dtype=float)
    gy, gx = np.gradient(g)
    n_half = (g.shape[0] - 1) // 2
    ax = (np.arange(g.shape[0]) - n_half) / float(oversample)
    rr = np.hypot(ax[None, :], ax[:, None])
    inside = rr <= r_stamp_px
    total = float(g[inside].sum())
    # photrad can exceed the stamp; the model carries no flux past the
    # stamp, so the honest answer there is 1.0, not >1 from grid corners
    ee = (float(g[inside & (rr <= photrad_px)].sum()) / total
          if total else 0.0)

    model = EpsfModel(
        grid=spline_filter(g, order=3, mode="constant"),
        grad_y=spline_filter(gy, order=3, mode="constant"),
        grad_x=spline_filter(gx, order=3, mode="constant"),
        oversample=int(oversample), r_stamp_px=float(r_stamp_px),
        fwhm_px=1.0, ee_photrad=float(ee))
    # FWHM from the model itself rather than from theory: it is used for
    # fit footprints and position bounds, so it should describe the PSF
    # actually built, seeing halo included
    from .image_strehl import radial_profile_fwhm
    stamp = model.detector_stamp()
    c = stamp.shape[0] // 2
    fw = radial_profile_fwhm(stamp, c, c)
    return EpsfModel(grid=model.grid, grad_y=model.grad_y,
                     grad_x=model.grad_x, oversample=model.oversample,
                     r_stamp_px=model.r_stamp_px,
                     fwhm_px=float(fw) if fw > 0 else 1.0,
                     ee_photrad=model.ee_photrad,
                     peak_value=float(stamp.max()))


def deep_star_catalog(image, params, sky=None, n_max=400,
                      min_snr=5.0, rel_floor=3e-4, exclude_fwhm=2.0):
    """Neighbour catalogue: deeper than the measured-star list, floored
    where the diffraction halo lives.

    `find_stars`' defaults are tuned for picking MAP points (a handful of
    good stars); the neighbour catalogue must instead be as complete as the
    frame allows.  Two floors bound it, and neither is negotiable:
    `min_snr` x the robust scatter, and `rel_floor` x the brightest star's
    peak.  `min_snr` is 5, not `find_stars`' 10 and not the 8 this function
    first shipped with, because this is a catalogue for SUBTRACTION rather
    than for measurement, and the two want opposite errors.  Measured on an
    S2 pair frame: a 3-mag-fainter companion had a core 180.4 ADU against
    an 8-sigma floor of 191.0 -- undetected, therefore never subtracted,
    while the uncleaned SR bias it caused reached -0.036.  A missed real
    neighbour is an uncorrected bias; a spurious 5-sigma one costs a fitted
    component whose amplitude comes out near zero or negative and is
    dropped by `group_fit`'s own non-negativity pass.  The fit is
    self-protecting against false positives; nothing protects against a
    contaminant that was never catalogued.  The relative floor is set by the diffraction HALO KNOTS, which
    sit at ~0.03% of their star's peak and scale with it, so no SNR floor
    can reject them; 3e-4 clears the knots while still reaching 8.8 mag
    below the brightest star -- deeper than the 5.6 mag spread measured in
    the GC 20260728 field.

    Neighbours fainter than this floor are NOT catalogued and therefore not
    subtracted.  That is a real, bounded limitation and it is exactly what
    `psf_fit`'s residual_frac reports; it is never papered over.

    Returns a list of dicts (x, y, peak, flux), brightest first.  `flux`
    is a small-aperture catalogue flux for RANKING and for the neighbour
    floor, not a photometric measurement -- the group fit solves each
    amplitude properly.
    """
    work = np.asarray(image, dtype=float)
    fwhm = dl_fwhm_px(params)
    exclude_px = max(exclude_fwhm * fwhm, 3.0)
    if sky is None:
        sky, sig = _robust_sky(work)
    else:
        _, sig = _robust_sky(work)
        sky = float(sky)

    r_ap = max(2.0 * fwhm, 3.0)
    r_in, r_out = 3.0 * fwhm, 5.0 * fwhm
    masked = work.copy()
    out = StarCatalog()
    out.n_max = int(n_max)
    first_peak = None
    while len(out) < int(n_max):
        iy, ix = np.unravel_index(int(masked.argmax()), masked.shape)
        peak = float(masked[iy, ix])
        if peak < sky + min_snr * sig:
            break
        if first_peak is None:
            first_peak = peak
        elif peak - sky < rel_floor * (first_peak - sky):
            break
        masked[radius_map(masked.shape, ix, iy) <= exclude_px] = sky
        x, y = cntrd(work, ix, iy, fwhm)
        if x < 0 or y < 0:
            continue
        if any(np.hypot(x - p["x"], y - p["y"]) <= exclude_px for p in out):
            continue            # halo knot / re-detection of a kept star
        sl, yy, xx = _box(work.shape, x, y, r_out)
        sub = work[sl]
        rr = np.hypot(xx - x, yy - y)
        ann = sub[(rr >= r_in) & (rr <= r_out)]
        s_loc = sigma_clipped_median(ann) if ann.size else sky
        flux = float((sub[rr <= r_ap] - s_loc).sum())
        out.append({"x": float(x), "y": float(y),
                    "peak": float(_core_peak(work, x, y, fwhm) - s_loc),
                    "flux": flux})
    # the loop exits on the SNR floor, the halo-knot floor, or the CAP.
    # Only the cap is an arbitrary bound, so only the cap has to announce
    # itself -- the floors are physics and are already documented.
    out.truncated = len(out) >= int(n_max)
    return out


def donor_candidates(image, params, catalog, sky=None, sky_sigma=None,
                     donor_min_snr=EPSF_DONOR_MIN_SNR, donor_peak_max=None,
                     donor_max_crowding=CROWDING_WARN_FRAC,
                     isolation_arcsec=EPSF_ISOLATION_STRICT_ARCSEC,
                     isolation_min_snr=EPSF_ISOLATION_MIN_SNR,
                     photometry_radius_arcsec=NIRC2_PHOTOMETRY_RADIUS_ARCSEC,
                     bg_inner_arcsec=NIRC2_BG_INNER_RADIUS_ARCSEC,
                     bg_outer_arcsec=NIRC2_BG_OUTER_RADIUS_ARCSEC,
                     badmask=None):
    """Rank `catalog` entries as ePSF donors; returns rows worst-cut-first.

    A donor must clear, in this order (the first failure is recorded so the
    build can report WHY a field had no donors):

      1. fully inside the array with its whole stamp + sky annulus;
      2. peak >= donor_min_snr x the robust sky scatter;
      3. peak <= donor_peak_max (linearity, see EPSF_PEAK_CEILING_FRAC);
      4. no core pixel flagged in `badmask`;
      5. annulus-contamination `crowding` <= donor_max_crowding;
      6. nearest catalogued neighbour farther than `isolation_arcsec`.

    Returns (rows, reasons) where rows is a list of dicts carrying x, y,
    peak, flux, sky, crowding, isolation_arcsec -- sorted by descending
    peak, i.e. best SNR first -- and reasons is a {cut_name: count} tally
    of everything rejected.
    """
    work = np.asarray(image, dtype=float)
    ps = float(params.plate_scale_mas)
    fwhm = dl_fwhm_px(params)
    if sky is None or sky_sigma is None:
        sky, sky_sigma = _robust_sky(work)
    if donor_peak_max is None:
        donor_peak_max = (EPSF_PEAK_CEILING_FRAC * float(params.max_counts)
                          * float(params.coadds))
    r_in = bg_inner_arcsec * 1000.0 / ps
    r_out = bg_outer_arcsec * 1000.0 / ps
    iso_px = isolation_arcsec * 1000.0 / ps
    # crowding must be measured over the PHOTOMETRY aperture, not the
    # annulus outer radius: `crowding` is (mean-sky minus clipped-sky)
    # integrated over n_ap and divided by flux, so it scales with n_ap,
    # and `donor_max_crowding` defaults to the tool's own
    # CROWDING_WARN_FRAC -- a threshold calibrated against
    # measure_strehl's 1.0" aperture. Using 1.4" here inflated n_ap by
    # (1.4/1.0)^2 = 1.96 and applied a 1.0"-calibrated threshold to a
    # nearly-doubled statistic. The metric has a pure-noise floor of
    # roughly sigma_sky/sqrt(N_ann) * n_ap / flux; on NIRC2 narrow with
    # 10 ADU read noise and a 1e5 ADU donor that is 0.049 at 1.4" --
    # indistinguishable from the 0.05 threshold -- against 0.025 at 1.0".
    r_crowd = photometry_radius_arcsec * 1000.0 / ps
    cat = list(catalog)
    # isolation is judged only against entries bright enough to be trusted
    # as a veto (EPSF_ISOLATION_MIN_SNR); see that constant for why
    iso_floor = isolation_min_snr * sky_sigma
    trusted = [c for c in cat if float(c.get("peak", 0.0)) >= iso_floor]
    cx = np.array([c["x"] for c in trusted]) if trusted else np.zeros(0)
    cy = np.array([c["y"] for c in trusted]) if trusted else np.zeros(0)
    ny, nx = work.shape
    rows, reasons = [], {}

    def _reject(name):
        reasons[name] = reasons.get(name, 0) + 1

    for c in cat:
        x, y = c["x"], c["y"]
        # The whole stamp must be on-array because `_samples` normalizes
        # each donor by its own stamp flux -- a clipped donor would enter
        # the stack on a different scale.  Relaxing this to the photometry
        # disc (plus a matching common-normalization change) was BUILT AND
        # MEASURED in D40: it bought ~1 build in 10 and left psi quality
        # unchanged, so it was reverted rather than shipped for nothing.
        # The real limit at moderate density is model CONVERGENCE, not
        # donor supply.
        if (x - r_out < 0 or y - r_out < 0
                or x + r_out >= nx or y + r_out >= ny):
            _reject("off-array")
            continue
        d = np.hypot(cx - x, cy - y) if cx.size else np.zeros(0)
        if d.size:
            d[d <= 1e-6] = np.inf       # the star's own trusted entry
        iso = float(d.min()) if d.size and np.isfinite(d).any() else np.inf
        peak = _core_peak(work, x, y, fwhm) - sky
        if peak < donor_min_snr * sky_sigma:
            _reject("faint")
            continue
        if peak > donor_peak_max:
            _reject("saturated/nonlinear")
            continue
        if badmask is not None:
            sl, _, _ = _box(work.shape, x, y, max(1.0, 0.75 * fwhm))
            if np.asarray(badmask, dtype=bool)[sl].any():
                _reject("bad-pixel core")
                continue
        flux, s_loc, crowding, *_ = aperture_flux(
            work, r_crowd, x, y, insky_px=r_in, outsky_px=r_out,
            robust=True)
        if not np.isfinite(flux) or flux <= 0.0:
            _reject("non-positive flux")
            continue
        if crowding > donor_max_crowding:
            _reject("crowded annulus")
            continue
        if iso <= iso_px:
            _reject("not isolated")
            continue
        rows.append({"x": float(x), "y": float(y), "peak": float(peak),
                     "flux": float(flux), "sky": float(s_loc),
                     "crowding": float(crowding),
                     "isolation_arcsec": float(iso * ps / 1000.0)})
    rows.sort(key=lambda r: -r["peak"])
    return rows, reasons


def build_epsf(image, params, catalog=None, *,
               oversample=None,
               r_stamp_arcsec=None,
               photometry_radius_arcsec=NIRC2_PHOTOMETRY_RADIUS_ARCSEC,
               bg_inner_arcsec=NIRC2_BG_INNER_RADIUS_ARCSEC,
               bg_outer_arcsec=NIRC2_BG_OUTER_RADIUS_ARCSEC,
               donor_min_snr=EPSF_DONOR_MIN_SNR,
               donor_peak_max=None,
               donor_max_crowding=CROWDING_WARN_FRAC,
               min_donors=EPSF_MIN_DONORS, max_donors=EPSF_MAX_DONORS,
               mask_radius_fwhm=EPSF_MASK_RADIUS_FWHM,
               donor_positions=None,
               gate_max_predicted_frac=EPSF_GATE_MAX_PREDICTED_FRAC,
               core_neg_max_frac=EPSF_CORE_NEG_MAX_FRAC,
               gate_sample=EPSF_GATE_SAMPLE,
               isolation_strict_arcsec=EPSF_ISOLATION_STRICT_ARCSEC,
               isolation_loose_arcsec=EPSF_ISOLATION_LOOSE_ARCSEC,
               n_cycles=EPSF_N_CYCLES, converge_tol=EPSF_CONVERGE_TOL,
               min_phase_coverage=EPSF_MIN_PHASE_COVERAGE,
               weight_scale_arcsec=EPSF_WEIGHT_SCALE_ARCSEC,
               clip_sigma=3.0, badmask=None):
    """Build the field ePSF.  Returns an `EmpiricalPsf`, always -- a field
    with no usable donors returns tag="uncalibrated", usable=False and a
    note saying so, never an exception and never a quietly bad model.

    `image` must be the SIGMA-FILTERED work array (`sigma_filter3` output),
    the same array the measurement and the subtraction operate on, so that
    the hot-pixel treatment is identical in the model and in the data.

    `r_stamp_arcsec` defaults to `bg_outer_arcsec`: under this module's sky
    convention psi has already fallen to ~0 by the annulus, so a larger
    stamp adds cost and no signal.  The consequence -- neighbour wings
    beyond the stamp are not subtracted -- biases cleaned flux HIGH and so
    cleaned SR LOW, the conservative direction, and is reported rather than
    corrected.

    Iteration (`n_cycles`, default 3):

      cycle 0  extract stamps at the `cntrd` positions, subtract the
               annulus sky, normalize by stamp flux, scatter into the
               oversampled grid, sigma-clip across the samples reaching
               each cell, fill the cells no donor reached -> psi_0.
      cycle k  subtract every catalogued neighbour inside each donor's
               stamp using psi_(k-1); RE-REGISTER each donor by fitting
               (dx, dy) against psi_(k-1) rather than re-running `cntrd`;
               re-stack -> psi_k.

    Re-registration is the point of iterating.  `cntrd` carries an inherent
    sub-pixel bias up to ~0.2 px (documented in the regress model, IDL
    behaves the same); stacking at that accuracy smears the core and biases
    psi's peak ~1% low, which under-subtracts neighbour cores.  Fitting
    against the current psi drives registration to the photon-noise limit.

    Convergence: delta = sum|psi_k - psi_(k-1)| / sum psi_(k-1) over the
    stamp, converged when delta <= converge_tol.  NOT converging is not a
    failure and not an exception: psi from the last cycle is kept, the
    delta is recorded, and the tag is downgraded one rung (strict ->
    loose) so every consumer sees that the model is provisional.
    """
    from .psf_fit import (
        PSF_FIT_FOOTPRINT_FWHM, component_footprint, group_fit,
    )

    work = np.asarray(image, dtype=float)
    ps = float(params.plate_scale_mas)
    oversample = int(oversample or EPSF_DEFAULT_OVERSAMPLE)
    if r_stamp_arcsec is None:
        r_stamp_arcsec = bg_outer_arcsec
    r_stamp_px = r_stamp_arcsec * 1000.0 / ps
    photrad_px = photometry_radius_arcsec * 1000.0 / ps
    weight_scale_px = weight_scale_arcsec * 1000.0 / ps
    fwhm = dl_fwhm_px(params)
    # D31 masking radius and the masking-proof normalization radius
    mask_r = float(mask_radius_fwhm) * fwhm
    r_norm_px = EPSF_NORM_RADIUS_FWHM * fwhm
    sky, sky_sigma = _robust_sky(work)
    if donor_peak_max is None:
        donor_peak_max = (EPSF_PEAK_CEILING_FRAC * float(params.max_counts)
                          * float(params.coadds))
    if catalog is None:
        catalog = deep_star_catalog(work, params, sky=sky)

    n_half = int(np.ceil(r_stamp_px)) * oversample
    grid_n = 2 * n_half + 1
    n_cells = grid_n * grid_n

    def _empty(tag, note):
        return EmpiricalPsf(
            donors=(), tag=tag, note=note, usable=False,
            oversample=oversample, r_stamp_px=r_stamp_px,
            plate_scale_mas=ps, weight_scale_px=weight_scale_px,
            n_candidates=len(catalog), n_cycles_run=0, converged=False,
            delta=float("nan"), fwhm_px=fwhm,
            donor_peak_max=float(donor_peak_max),
            isolation_used_arcsec=float(isolation_loose_arcsec),
            phase_coverage=0.0, n_filled_cells=0, photrad_px=photrad_px,
            reasons=dict(reasons_seen))

    reasons_seen = {}
    # --- hand-picked donors (StarFinder practice) ---------------------
    # When the caller names the PSF stars, the automatic isolation and
    # crowding cuts are NOT applied to them: those cuts exist to guess
    # which stars a human would choose, and the human has now said. The
    # PHYSICAL cuts still apply (off-array, saturation, non-positive
    # flux) because those are not judgement calls. The full `catalog` is
    # still used for neighbour subtraction and masking -- only donor
    # SELECTION is overridden.
    if donor_positions:
        want = [(float(x), float(y)) for x, y in donor_positions]
        picked, missing = [], []
        for wx, wy in want:
            best, bd = None, 1e9
            for c in catalog:
                d = np.hypot(float(c["x"]) - wx, float(c["y"]) - wy)
                if d < bd:
                    bd, best = d, c
            if best is not None and bd <= max(2.0, 0.5 * fwhm):
                picked.append(best)
            else:
                missing.append((round(wx, 1), round(wy, 1)))
        rows, reasons_seen = donor_candidates(
            work, params, picked, sky=sky, sky_sigma=sky_sigma,
            donor_min_snr=donor_min_snr, donor_peak_max=donor_peak_max,
            donor_max_crowding=float("inf"), isolation_arcsec=0.0,
            photometry_radius_arcsec=photometry_radius_arcsec,
            bg_inner_arcsec=bg_inner_arcsec,
            bg_outer_arcsec=bg_outer_arcsec, badmask=badmask)
        if len(rows) < min_donors:
            why = ", ".join(f"{n}: {c}" for n, c in
                            sorted(reasons_seen.items(), key=lambda kv: -kv[1]))
            return _empty(
                "uncalibrated",
                f"hand-picked donors: only {len(rows)} of "
                f"{len(want)} requested stars are usable, below the "
                f"{min_donors} needed."
                + (f" Rejections: {why}." if why else "")
                + (f" Not found in the catalogue: {missing}." if missing
                   else ""))
        tag, iso_used = "hand-picked", 0.0
        rows = rows[:max_donors]
    else:
        # --- ladder: strict, then loose, then refuse ------------------
        tag, iso_used = None, isolation_strict_arcsec
        for rung, iso in (("strict", isolation_strict_arcsec),
                          ("loose", isolation_loose_arcsec)):
            rows, reasons_seen = donor_candidates(
                work, params, catalog, sky=sky, sky_sigma=sky_sigma,
                donor_min_snr=donor_min_snr, donor_peak_max=donor_peak_max,
                donor_max_crowding=donor_max_crowding, isolation_arcsec=iso,
                photometry_radius_arcsec=photometry_radius_arcsec,
                bg_inner_arcsec=bg_inner_arcsec,
                bg_outer_arcsec=bg_outer_arcsec, badmask=badmask)
            if len(rows) >= min_donors:
                tag, iso_used = rung, iso
                break
        if tag is None:
            why = ", ".join(f"{n}: {c}" for n, c in
                            sorted(reasons_seen.items(), key=lambda kv: -kv[1]))
            return _empty(
                "uncalibrated",
                f"only {len(rows)} of {len(catalog)} catalogued stars are "
                f"usable ePSF donors at the loose isolation cut "
                f"({isolation_loose_arcsec:.2f}\"), below the {min_donors} "
                f"needed -- cleaning skipped."
                + (f" Rejections: {why}." if why else
                   " Nothing was rejected -- the field simply has too few"
                   " stars."))
        rows = rows[:max_donors]

    cx = np.array([c["x"] for c in catalog])
    cy = np.array([c["y"] for c in catalog])

    def _samples(x, y, sky_d, subtract=None):
        """One donor's sample set: (cell, value, flux).  `subtract` is an
        optional model array already removed over the stamp box."""
        sl, yy, xx = _box(work.shape, x, y, r_stamp_px)
        sub = work[sl].astype(float)
        if subtract is not None:
            sub = sub - subtract
        u = xx - x
        v = yy - y
        d2 = u * u + v * v
        inside = d2 <= r_stamp_px ** 2
        # D31: drop every pixel within EPSF_MASK_RADIUS_FWHM * FWHM of a
        # CATALOGUED star other than this donor.  Only entries whose mask
        # can actually reach the stamp are considered.
        if mask_r > 0.0 and cx.size:
            near = (np.abs(cx - x) <= r_stamp_px + mask_r) & \
                   (np.abs(cy - y) <= r_stamp_px + mask_r)
            for nx_, ny_ in zip(cx[near], cy[near]):
                if (nx_ - x) ** 2 + (ny_ - y) ** 2 <= (0.5 * fwhm) ** 2:
                    continue                     # the donor's own entry
                inside &= ((xx - nx_) ** 2 + (yy - ny_) ** 2) > mask_r ** 2
        vals = sub[inside] - sky_d
        # Normalize over a disc masking can never reach (see
        # EPSF_NORM_RADIUS_FWHM), so a donor with a masked neighbour and
        # one without are on the same scale.
        core = d2 <= r_norm_px ** 2
        flux = float((sub[core] - sky_d).sum())
        if not np.isfinite(flux) or flux <= 0 or vals.size == 0:
            return None, None, None, 0.0
        # bilinear scatter (D21): each sample lands on the 4 surrounding
        # cells with the same weights the gather would use, so no sub-pixel
        # offset is quantized away
        fj = u[inside] * oversample + n_half
        fi = v[inside] * oversample + n_half
        j0 = np.floor(fj).astype(int)
        i0 = np.floor(fi).astype(int)
        tj = fj - j0
        ti = fi - i0
        norm = (vals / flux).astype(np.float64)
        cells, values, weights = [], [], []
        for di, wi in ((0, 1.0 - ti), (1, ti)):
            for dj, wj in ((0, 1.0 - tj), (1, tj)):
                ii, jj = i0 + di, j0 + dj
                w = wi * wj
                ok = ((ii >= 0) & (ii < grid_n) & (jj >= 0) & (jj < grid_n)
                      & (w > 0.0))
                if not ok.any():
                    continue
                cells.append((ii[ok] * grid_n + jj[ok]).astype(np.int32))
                values.append(norm[ok].astype(np.float32))
                weights.append(w[ok].astype(np.float32))
        if not cells:
            return None, None, None, 0.0
        return (np.concatenate(cells), np.concatenate(values),
                np.concatenate(weights), flux)

    # --- cycle 0: scatter the raw stamps ------------------------------
    donors = []
    for r in rows:
        cell, val, wgt, flux = _samples(r["x"], r["y"], r["sky"])
        if cell is None:
            continue
        donors.append(EpsfDonor(
            x=r["x"], y=r["y"], peak=r["peak"], flux=flux, sky=r["sky"],
            crowding=r["crowding"], isolation_arcsec=r["isolation_arcsec"],
            n_cleaned=0, reg_dx=0.0, reg_dy=0.0, cell=cell, value=val,
            weight=wgt))
    if len(donors) < min_donors:
        return _empty(
            "uncalibrated",
            f"only {len(donors)} donor stamps survived extraction, below "
            f"the {min_donors} needed -- cleaning skipped.")

    def _stack(dons):
        return _assemble(dons, np.ones(len(dons)), grid_n, oversample,
                         r_stamp_px, clip_sigma)

    grid, coverage, n_filled = _stack(donors)
    model = _make_model(grid, oversample, r_stamp_px, photrad_px)

    # --- cycles 1..n-1: clean the donors, re-register, re-stack -------
    delta = float("nan")
    delta_pixel = float("nan")
    converged = False
    over_subtracted = 0.0
    diverged = False
    cycles_run = 1
    for _cycle in range(1, max(int(n_cycles), 1)):
        new_donors = []
        for d in donors:
            # Exclude the donor's OWN catalogue entry by a physical
            # radius, not by dd > 1e-6.  Re-registration moves the donor
            # off its catalogue position by ~0.1 px, after which an exact
            # -zero test stops matching and the donor is fitted as its own
            # neighbour -- and then SUBTRACTED, gutting itself (observed:
            # donor flux 100085 -> 5108 in cycle 3, delta 0.002 -> 3.3).
            # 0.5 x FWHM is the same self-exclusion rule
            # `psf_fit.select_neighbours` uses, and it says the same thing:
            # a catalogue entry this close IS this star.
            dd = np.hypot(cx - d.x, cy - d.y)
            near = np.flatnonzero((dd <= r_stamp_px)
                                  & (dd > 0.5 * model.fwhm_px))
            comps = [(d.x, d.y)] + [(cx[k], cy[k]) for k in near]
            # fit only the components' own neighbourhoods, clipped to the
            # donor's stamp.  Handing group_fit the whole stamp box costs
            # ~16x the pixels (80 089 vs ~4 900 here) for residual
            # sensitivity the sky noise has already swamped, and it was
            # 14.3 s of a 15.5 s build before this was restricted.
            ffy, ffx = component_footprint(
                work.shape, comps, PSF_FIT_FOOTPRINT_FWHM * model.fwhm_px,
                clip_center=(d.x, d.y), clip_radius=r_stamp_px)
            if ffy.size < 2 * len(comps) + 4:
                new_donors.append(d)
                continue
            amps, pos, bg, _res, info = group_fit(
                work, comps, model, (ffy, ffx), sky_sigma,
                pos_tol_px=0.35 * model.fwhm_px, badmask=badmask,
                saturation=donor_peak_max / EPSF_PEAK_CEILING_FRAC
                if np.isfinite(donor_peak_max) else None)
            if info["status"] <= 0:
                new_donors.append(d)    # keep this cycle's samples rather
                continue                # than register against a bad fit
            nx_, ny_ = float(pos[0][0]), float(pos[0][1])
            # render the neighbour models on the box of the RE-REGISTERED
            # position: the fit moves the donor, and _box's floor/ceil can
            # then return a differently-shaped window than the one the fit
            # ran on.  Rendering here keeps `subtract` aligned to the
            # stamp the samples are actually taken from.
            _sl2, yy2, xx2 = _box(work.shape, nx_, ny_, r_stamp_px)
            sub = np.zeros(yy2.size, dtype=float)
            for k in range(1, len(comps)):
                if amps[k] <= 0:
                    continue
                sub += model.evaluate_at(yy2.ravel(), xx2.ravel(),
                                         pos[k][0], pos[k][1], amp=amps[k])
            cell, val, wgt, flux = _samples(
                nx_, ny_, d.sky, subtract=sub.reshape(yy2.shape))
            if cell is None:
                new_donors.append(d)
                continue
            new_donors.append(EpsfDonor(
                x=nx_, y=ny_, peak=d.peak, flux=flux, sky=d.sky,
                crowding=d.crowding, isolation_arcsec=d.isolation_arcsec,
                n_cleaned=len(comps) - 1, reg_dx=nx_ - d.x,
                reg_dy=ny_ - d.y, cell=cell, value=val, weight=wgt))
        new_grid, new_cov, new_filled = _stack(new_donors)
        new_delta = _profile_delta(new_grid, grid, oversample, r_stamp_px)
        new_delta_px = float(np.abs(new_grid - grid).sum()
                             / (float(np.abs(grid).sum()) or 1.0))
        cycles_run += 1
        # Divergence guard (D15): the iteration exists to clean the donors'
        # own companions, and on heavily blended donors it can instead run
        # away -- an over-estimated companion amplitude eats donor flux, the
        # normalized samples inflate, psi distorts, and the next cycle is
        # worse (observed delta 11.25 on the 0.35"-companion ladder field).
        # Never return a psi worse than the best one already seen: if delta
        # grows, keep the previous cycle's model and stop.
        if np.isfinite(delta) and new_delta > delta:
            diverged = True
            delta, delta_pixel = new_delta, new_delta_px
            break
        # Over-subtraction guard: the delta test above cannot fire on the
        # FIRST iteration (delta is NaN until two cycles exist), which is
        # exactly where a hand-picked or heavily blended donor set runs
        # away.  Measured on M92 with 4 hand-picked donors: cycle 1 has
        # ZERO negative cells in the core and a deepest negative 26 % of
        # the peak (ordinary wing noise); cycle 2 has 146 negative core
        # cells and a deepest negative 126 % OF THE PEAK -- a hole deeper
        # than the star is tall.  That is subtracted flux that was never
        # there.  A PSF core is positive, so this is a physical test
        # rather than a tuned one, and it catches the runaway that delta
        # misses.
        core_depth = _core_negative_depth(
            new_grid, oversample, EPSF_CORE_NEG_RADIUS_FWHM * fwhm)
        if core_depth > core_neg_max_frac:
            over_subtracted = core_depth
            break
        donors = new_donors
        grid, coverage, n_filled = new_grid, new_cov, new_filled
        delta, delta_pixel = new_delta, new_delta_px
        model = _make_model(grid, oversample, r_stamp_px, photrad_px)
        if delta <= converge_tol:
            converged = True
            break

    downgrades = []
    if over_subtracted:
        downgrades.append(
            f"iteration OVER-SUBTRACTED (psi's core developed a negative "
            f"hole {100 * over_subtracted:.0f}% as deep as its own peak, "
            f"above the {100 * core_neg_max_frac:.0f}% limit); kept the "
            f"last sound cycle's model")
    if diverged:
        downgrades.append(
            f"iteration DIVERGED (delta rose to {delta:.4f}); kept the last "
            f"improving cycle's model")
    elif not converged:
        downgrades.append(f"NOT converged (delta={delta:.4f})")
    if coverage < min_phase_coverage:
        downgrades.append(
            f"phase coverage {100 * coverage:.0f}% below the "
            f"{100 * min_phase_coverage:.0f}% floor, so {100 * (1 - coverage):.0f}% "
            f"of psi is interpolated rather than measured")
    if downgrades and tag == "strict":
        tag = "loose"           # provisional model: say so via the rung

    # Instrument-agnostic guard, NOT an OSIRIS code path (PLAN section 3
    # keeps OSIRIS a non-goal but asks that the engine stay params-clean).
    # `osiris_frame_params` sets max_counts=inf ("no saturation check in the
    # tool"), which makes the derived donor ceiling infinite and the
    # linearity cut unable to fire at all -- a saturated donor would be
    # accepted and its flattened core would bias psi LOW, under-subtracting
    # every neighbour at its peak.  Any instrument whose params carry no
    # ceiling lands here.  The model may still be fine, so this does not
    # downgrade the rung; what it must not do is stay quiet about a guard
    # that is switched off.
    ceiling_note = ""
    if not np.isfinite(donor_peak_max):
        ceiling_note = (" LINEARITY CUT INACTIVE: this instrument's params "
                        "carry no saturation ceiling, so donor cores were "
                        "not checked for non-linearity -- pass "
                        "donor_peak_max explicitly (e.g. from a measured "
                        "peak-vs-flux onset) to enable it.")

    note = (f"ePSF: {len(donors)} donors ({tag}, isolation "
            f"{iso_used:.2f}\"), {cycles_run} cycle(s), delta="
            f"{delta:.4f} (pixelwise {delta_pixel:.4f}), phase coverage "
            f"{100 * coverage:.0f}% "
            f"({n_filled} cells filled)."
            + (" PROVISIONAL -- " + "; ".join(downgrades) + "."
               if downgrades else "") + ceiling_note
            + getattr(catalog, "cap_note", ""))

    # D46: model-level over-amplitude gate.  One sample per FIELD, before
    # any per-star fit, so a broken model refuses in ONE line instead of
    # N per-star refusals at ~4 s each.
    gate_frac = _predicted_frac(work, params, model, catalog, photrad_px,
                                bg_inner_arcsec, bg_outer_arcsec,
                                gate_sample,
                                photometry_radius_arcsec=(
                                    photometry_radius_arcsec))
    # model-free contamination on the SAME stars, as the collapse referee
    _gc = []
    for _c in sorted(catalog, key=lambda c: -float(c.get("peak", 0.0)))[
            :int(gate_sample)]:
        _f, _sk, _cr, *_ = aperture_flux(
            work, photrad_px, float(_c["x"]), float(_c["y"]),
            insky_px=bg_inner_arcsec * 1000.0 / ps,
            outsky_px=bg_outer_arcsec * 1000.0 / ps, robust=True)
        if np.isfinite(_f) and _f > 0.0:
            _gc.append(_cr)
    gate_crowd = float(np.median(_gc)) if _gc else float("nan")
    if (np.isfinite(gate_frac) and np.isfinite(gate_crowd)
            and gate_crowd > EPSF_COLLAPSE_CROWD_MIN
            and gate_frac < EPSF_COLLAPSE_PRED_MIN):
        return _empty(
            "uncalibrated",
            f"model REFUSED (collapsed build): the field is unambiguously "
            f"crowded (median crowding {gate_crowd:.3f}) yet across the "
            f"{gate_sample} brightest stars the model predicts only "
            f"{100 * gate_frac:.1f}% of the target's own aperture flux in "
            f"neighbour light. A model that sees nothing on a field this "
            f"crowded has collapsed, not succeeded -- subtracting with it "
            f"would leave the contamination in place while reporting that "
            f"it was cleaned. Cleaning skipped for this frame. "
            f"(Donors were {len(donors)} at the {tag} rung.)")
    if np.isfinite(gate_frac) and gate_frac > gate_max_predicted_frac:
        return _empty(
            "uncalibrated",
            f"model REFUSED (over-amplitude build): across the "
            f"{gate_sample} brightest stars the median PREDICTED "
            f"neighbour light is {100 * gate_frac:.0f}% of the target's "
            f"own aperture flux, above the {100 * gate_max_predicted_frac:.0f}% "
            f"physical limit -- the model's amplitudes are wrong, not "
            f"merely the field crowded, so every star would refuse "
            f"individually. Cleaning skipped for this frame. "
            f"(Donors were {len(donors)} at the {tag} rung.)")

    return EmpiricalPsf(
        donors=tuple(donors), tag=tag, note=note, usable=True,
        oversample=oversample, r_stamp_px=r_stamp_px, plate_scale_mas=ps,
        weight_scale_px=weight_scale_px, n_candidates=len(catalog),
        n_cycles_run=cycles_run, converged=converged, delta=delta,
        fwhm_px=float(model.fwhm_px), donor_peak_max=float(donor_peak_max),
        isolation_used_arcsec=float(iso_used), phase_coverage=coverage,
        n_filled_cells=n_filled, photrad_px=photrad_px,
        delta_pixel=delta_pixel, reasons=dict(reasons_seen))


def estimate_psf_shape(image, params, pos=None, *, halo_beta=None,
                       r_fit_arcsec=None,
                       bg_inner_arcsec=NIRC2_BG_INNER_RADIUS_ARCSEC,
                       bg_outer_arcsec=NIRC2_BG_OUTER_RADIUS_ARCSEC,
                       dl_psf=None):
    """Estimate (sr, halo_fwhm_arcsec) for `theoretical_psf` from the frame
    itself, using ONE bright star.  Returns (sr, halo_fwhm_arcsec, info).

    The obvious approach -- read the measured Strehl off the brightest star
    -- does not work in the fields this exists for: a crowded star's
    measured SR is biased LOW by its neighbours' light, which is the very
    problem being solved.  On GC 20260728 the brightest stars measure SR
    0.27-0.65 with crowding 0.67-9.41, so their SR is not the field's SR.

    Instead the two shape parameters are fitted to the star's AZIMUTHALLY
    MEDIAN radial profile.  That is the key: a neighbour sits at ONE
    position angle, so it contaminates a few azimuths at a given radius
    while the median over all azimuths ignores it.  An azimuthal MEAN
    would not -- it is exactly as vulnerable as the aperture photometry
    this feature exists to fix.

    Only the SHAPE is fitted; the star's brightness is divided out, so the
    result does not depend on the star's flux being known.
    """
    from scipy.optimize import least_squares

    from .constants import MOFFAT_BETA_KOLM
    from .nirc2_psf import nirc2_dl_psf

    work = np.asarray(image, dtype=float)
    ps = float(params.plate_scale_mas)
    beta = float(MOFFAT_BETA_KOLM if halo_beta is None else halo_beta)
    if r_fit_arcsec is None:
        r_fit_arcsec = bg_inner_arcsec        # stop before the sky annulus
    r_fit = r_fit_arcsec * 1000.0 / ps
    fwhm = dl_fwhm_px(params)

    if pos is None:
        # The DONOR ceiling (EPSF_PEAK_CEILING_FRAC, half of saturation) is
        # deliberately conservative because a donor's flattened core would
        # bias a whole stack. This is a single SHAPE fit, so the relevant
        # limit is actual saturation with a margin -- applying the donor
        # ceiling here found no star at all on GC 20260728, where the
        # brightest stars sit above it.
        cands = deep_star_catalog(work, params, n_max=200)
        sat = float(params.max_counts) * float(params.coadds)
        margin = bg_outer_arcsec * 1000.0 / ps
        ny, nx = work.shape
        ok = [c for c in cands
              if (not np.isfinite(sat) or c["peak"] < 0.9 * sat)
              and margin < c["x"] < nx - margin
              and margin < c["y"] < ny - margin]
        if not ok:
            raise ValueError(
                f"no unsaturated star clear of the frame edge among "
                f"{len(cands)} candidates; pass `pos` explicitly")
        # brightest such star: best profile SNR out to the halo
        ok.sort(key=lambda c: -c["peak"])
        pos = (ok[0]["x"], ok[0]["y"])
    x, y = float(pos[0]), float(pos[1])

    sl, yy, xx = _box(work.shape, x, y, bg_outer_arcsec * 1000.0 / ps)
    sub = work[sl]
    rr = np.hypot(xx - x, yy - y)
    ann = sub[(rr >= bg_inner_arcsec * 1000.0 / ps)
              & (rr <= bg_outer_arcsec * 1000.0 / ps)]
    sky = sigma_clipped_median(ann) if ann.size else 0.0

    # azimuthal MEDIAN profile -- neighbours occupy single azimuths and are
    # rejected; an azimuthal mean would carry them straight in
    edges = np.arange(0.0, r_fit + 1.0, max(1.0, fwhm / 4.0))
    prof_r, prof_v = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rr >= lo) & (rr < hi)
        if m.sum() >= 4:
            prof_r.append(0.5 * (lo + hi))
            prof_v.append(float(np.median(sub[m])) - sky)
    prof_r = np.array(prof_r)
    prof_v = np.array(prof_v)
    if prof_v.size < 5 or prof_v[0] <= 0:
        raise ValueError("could not build a usable radial profile")
    prof_v = prof_v / prof_v[0]           # shape only; brightness divides out

    if dl_psf is None:
        dl_psf = nirc2_dl_psf(params.camname, params.pmsname,
                              params.effwave_um, params.pmrangl_deg,
                              npix=512, daytime=params.daytime,
                              sfp=getattr(params, "sfp", False))
    # The DL profile must be sampled at EXACTLY the radii the star's
    # profile used. Building it from the same `edges` list and filtering
    # separately misaligns the two -- the star skips bins that are short of
    # pixels in the IMAGE, the DL skips bins short in the DL ARRAY, and the
    # skips are not the same. That misalignment drove the fit to its bound
    # with a 2.6 dex residual.
    dlc = dl_psf.shape[0] / 2.0 - 0.5      # nirc2_dl_psf's own centring
    dax = np.arange(dl_psf.shape[0]) - dlc
    drr = np.hypot(dax[None, :], dax[:, None])
    half = 0.5 * float(np.median(np.diff(edges))) if edges.size > 1 else 0.5

    def _radial(arr2d):
        """Radial median of a 2-D array at exactly `prof_r`."""
        o = np.empty(prof_r.size)
        for i, rc in enumerate(prof_r):
            m = (drr >= max(rc - half, 0.0)) & (drr < rc + half)
            o[i] = float(np.median(arr2d[m])) if m.any() else 0.0
        return o

    # BOTH components must carry their PHYSICAL relative amplitudes, i.e.
    # unit 2-D SUM, because that is what `sr` weights. Normalizing each to
    # its own peak (or its own first bin) makes them equal-peak and `sr`
    # then controls nothing -- the fit ran to its upper bound every time,
    # returning sr ~ 0.98 for injected 0.15/0.30/0.60 while still fitting
    # the profile well, because any sr gave the same normalized shape.
    dl2 = np.asarray(dl_psf, dtype=float)
    dl2 = dl2 / dl2.sum()
    dl_prof = _radial(dl2)

    def model(p):
        sr, fw = p
        alpha = (fw * 1000.0 / ps) / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
        h2 = (1.0 + (drr / alpha) ** 2) ** (-beta)   # 2-D, same grid as DL
        h2 = h2 / h2.sum()                           # unit 2-D sum, as DL is
        m = sr * dl_prof + (1.0 - sr) * _radial(h2)
        return m / max(m[0], 1e-30)

    # Fit only where there is real signal. The annulus sky slightly
    # OVER-subtracts (the D6 convention counts halo as sky), so the outer
    # profile crosses zero and goes negative -- measured, 17 of 106 bins on
    # a clean synthetic. In log space each negative bin contributes an ~8
    # dex residual, which pinned the fit at its bound with rms 2.6 dex.
    # Restricting to bins above a floor of the peak keeps the fit on the
    # part of the profile that carries shape information anyway.
    fit_ok = prof_v > 2e-3 * prof_v[0]
    if fit_ok.sum() < 5:
        raise ValueError(
            f"only {int(fit_ok.sum())} usable profile points above the "
            f"noise floor; cannot fit a PSF shape")

    def resid(p):
        m = np.clip(model(p)[fit_ok], 1e-12, None)
        d = np.clip(prof_v[fit_ok], 1e-12, None)
        return np.log10(m) - np.log10(d)

    r = least_squares(resid, [0.3, 0.5], bounds=([0.01, 0.05], [0.99, 3.0]),
                      xtol=1e-4, ftol=1e-4)
    sr_fit, fw_fit = float(r.x[0]), float(r.x[1])
    info = {"star": (round(x, 2), round(y, 2)), "sky": round(float(sky), 3),
            "n_profile_points": int(prof_r.size),
            "n_fit_points": int(fit_ok.sum()),
            "r_fit_max_px": round(float(prof_r[fit_ok][-1]), 1),
            "fit_status": int(r.status),
            "rms_dex": round(float(np.sqrt(np.mean(resid(r.x) ** 2))), 4)}
    return sr_fit, fw_fit, info


def theoretical_psf(params, sr, halo_fwhm_arcsec, *, halo_beta=None,
                    oversample=None, r_stamp_arcsec=None,
                    photometry_radius_arcsec=NIRC2_PHOTOMETRY_RADIUS_ARCSEC,
                    bg_outer_arcsec=NIRC2_BG_OUTER_RADIUS_ARCSEC,
                    dl_psf=None):
    """A PSF model built from PHYSICS rather than from the field's stars
    (D26).  Returns an `EmpiricalPsf` whose `.at()` yields the usual
    `EpsfModel`, so every consumer -- `clean_star`, `group_fit`, the
    reports, the GUI -- is unchanged.

    psi = sr * DL + (1 - sr) * halo,  the SAME decomposition
    `ee_correction.py` already assumes, so the tool stays consistent with
    itself.  DL is the real rasterized pupil PSF from `nirc2_dl_psf`
    (pupil stop, spiders, rotation, wavelength); the halo is a Moffat of
    the given FWHM and index.

    WHY THIS EXISTS.  `build_epsf` needs uncrowded donor stars, and
    globular-cluster-density fields have none: GC 20260728 yielded ZERO
    donors on four frames and its deep stack, and M92 on the same night
    zero on three more (R2).  This path needs TWO NUMBERS -- an SR and a
    halo width -- not fifteen clean donor stamps, and a field that cannot
    supply one uncrowded donor can still supply those.

    THE HALO IS NOT OPTIONAL.  A bare diffraction-limited PSF is not a
    usable model here: at sr = 0.30 the true PSF's peak is ~30 % of a DL
    peak, so a core-weighted fit matching a pure DL profile returns an
    amplitude ~3x too small and subtracts ~30 % of the flux it should.

    WHAT IT CANNOT DO.  It carries no static speckle, no instrument-
    specific structure, no field-dependent elongation -- all things an
    empirical ePSF captures in principle.  Its errors are different in
    kind from the ePSF's and it does NOT inherit the ePSF's validation;
    the S-battery has to be re-run against it.  The returned model is
    tagged `theoretical` so no consumer can mistake one for the other.
    """
    from .constants import MOFFAT_BETA_KOLM
    from .nirc2_psf import nirc2_dl_psf

    ps = float(params.plate_scale_mas)
    oversample = int(oversample or EPSF_DEFAULT_OVERSAMPLE)
    if r_stamp_arcsec is None:
        r_stamp_arcsec = bg_outer_arcsec
    r_stamp_px = r_stamp_arcsec * 1000.0 / ps
    photrad_px = photometry_radius_arcsec * 1000.0 / ps
    beta = float(MOFFAT_BETA_KOLM if halo_beta is None else halo_beta)
    sr = float(sr)
    if not 0.0 < sr <= 1.0:
        raise ValueError(f"sr must be in (0, 1]; got {sr}")
    if halo_fwhm_arcsec <= 0:
        raise ValueError("halo_fwhm_arcsec must be positive")

    n_half = int(np.ceil(r_stamp_px)) * oversample
    grid_n = 2 * n_half + 1
    # the oversampled grid, in DETECTOR pixels from the centre
    ax = (np.arange(grid_n) - n_half) / float(oversample)
    rr = np.hypot(ax[None, :], ax[:, None])

    # --- diffraction-limited core, resampled onto the oversampled grid.
    # nirc2_dl_psf returns a DETECTOR-sampled PSF, so it is interpolated
    # up rather than recomputed at a finer plate scale: recomputing would
    # change the pupil sampling and stop being the same reference the
    # measurement uses.
    if dl_psf is None:
        dl_psf = nirc2_dl_psf(params.camname, params.pmsname,
                              params.effwave_um, params.pmrangl_deg,
                              npix=512, daytime=params.daytime,
                              sfp=getattr(params, "sfp", False))
    from scipy.ndimage import map_coordinates
    c = dl_psf.shape[0] / 2.0 - 0.5      # nirc2_dl_psf's own centring
    core = map_coordinates(
        np.asarray(dl_psf, dtype=float),
        [(ax[:, None] + np.zeros_like(ax)[None, :]) + c,
         (np.zeros_like(ax)[:, None] + ax[None, :]) + c],
        order=3, mode="constant", cval=0.0)
    core = np.clip(core, 0.0, None)
    if core.sum() > 0:
        core /= core.sum()

    # --- seeing halo: Moffat of the requested FWHM
    fwhm_px = halo_fwhm_arcsec * 1000.0 / ps
    alpha = fwhm_px / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    halo = (1.0 + (rr / alpha) ** 2) ** (-beta)
    halo /= halo.sum()

    g = sr * core + (1.0 - sr) * halo
    disc = rr <= r_stamp_px
    g = np.where(disc, g, 0.0)
    total = g.sum()
    if total > 0:
        g = g * (oversample ** 2) / total

    model = _make_model(g, oversample, r_stamp_px, photrad_px)
    note = (f"THEORETICAL PSF (D26): sr={sr:.3f}, halo FWHM "
            f"{halo_fwhm_arcsec:.3f}\", Moffat beta={beta:.3f}. Built from "
            f"physics, not from the field's stars -- it carries no static "
            f"speckle or instrument-specific structure, and it does NOT "
            f"inherit the empirical ePSF's validation.")
    psf = EmpiricalPsf(
        donors=(), tag="theoretical", note=note, usable=True,
        oversample=oversample, r_stamp_px=r_stamp_px, plate_scale_mas=ps,
        weight_scale_px=float("inf"), n_candidates=0, n_cycles_run=0,
        converged=True, delta=0.0, fwhm_px=float(model.fwhm_px),
        donor_peak_max=float("nan"), isolation_used_arcsec=float("nan"),
        phase_coverage=1.0, n_filled_cells=0, photrad_px=photrad_px,
        reasons={})
    # a theoretical model is the SAME everywhere in the field -- there are
    # no donors to distance-weight -- so `at()` must return it directly
    psf.__dict__["_model_cache"] = {None: model}
    psf.__dict__["_fixed_model"] = model
    return psf


def epsf_strehl(model, params, dl_psf=None,
                photometry_radius_arcsec=NIRC2_PHOTOMETRY_RADIUS_ARCSEC,
                peak_radius_arcsec=NIRC2_PEAK_RADIUS_ARCSEC):
    """Strehl of the ePSF itself against the DL PSF -- a DIAGNOSTIC.

    Computed by exactly `measure_strehl`'s arithmetic on
    `model.detector_stamp()`: pixelation-corrected `find_peak` over
    `aperture_flux` at `photometry_radius_arcsec`, divided by the identical
    ratio on `dl_psf`.  Same convention, same aperture, same reference.

    This is a FIELD-AVERAGE number and must always be presented as one
    (Decision D2).  A single fitted ePSF gives every star the same shape,
    so peak/flux from it is a field constant -- reporting it per star would
    be reporting the same number N times and calling it a measurement.
    """
    from .nirc2_psf import nirc2_dl_psf

    ps = float(params.plate_scale_mas)
    photrad = photometry_radius_arcsec * 1000.0 / ps
    radius = int(np.ceil(peak_radius_arcsec * 1000.0 / ps))
    box = 2 * radius + 1

    if dl_psf is None:
        dl_psf = nirc2_dl_psf(params.camname, params.pmsname,
                              params.effwave_um, params.pmrangl_deg,
                              npix=512, daytime=params.daytime,
                              sfp=getattr(params, "sfp", False))
    # identical arithmetic to measure_strehl's reference branch, so the
    # diagnostic is on the tool's own convention rather than a parallel one
    ctr = dl_psf.shape[0] // 2
    dlpeak = find_peak(dl_psf, ctr, ctr, box)
    crad = max(photrad / 2.0, 6.0)
    cxr, cyr = cntrd(dl_psf, ctr, ctr, crad / 2.0)
    if cxr < 0:
        cxr, cyr = float(ctr), float(ctr)
    strehlone = dlpeak / aperture_flux(dl_psf, photrad, cxr, cyr,
                                       skyval=0.0)[0]

    stamp = model.detector_stamp()
    c = stamp.shape[0] // 2
    peak = find_peak(stamp, c, c, box)
    flux = aperture_flux(stamp, photrad, c, c, skyval=0.0)[0]
    if flux == 0.0 or strehlone == 0.0:
        return float("nan")
    return float((peak / flux) / strehlone)
