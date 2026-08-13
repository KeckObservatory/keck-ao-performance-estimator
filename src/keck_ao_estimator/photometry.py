"""Band-to-band stellar magnitude estimation: given whatever photometry a
star's catalogue entry happens to carry, pick or estimate its magnitude in a
WANTED band. Split out of catalogs.py (which owns the Vizier registry/query/
parsing instead) since this is pure photometry math with no dependency on
where the magnitudes came from -- useful anywhere a magnitude in one band
needs to be read off photometry in others, not just the guide-star catalogue
overlay.

Pure and Qt-free; no network, no lazy imports needed.
"""

# Effective wavelength (microns) of each band label, used to pick the
# available band closest to a wanted wavelength.
_BAND_UM = {"U": 0.36, "B": 0.44, "g": 0.48, "BP": 0.53, "V": 0.55, "G": 0.60,
            "r": 0.62, "R": 0.65, "F": 0.65, "N": 0.66, "i": 0.75, "I": 0.80,
            "RP": 0.80, "z": 0.87, "j": 0.90, "y": 0.96, "J": 1.25, "H": 1.65,
            "Ks": 2.15, "K": 2.20}

# The wavelength (microns) each tip-tilt sensing band works at -- what the GUI
# asks pick_mag()/estimate_sensing_mag() for when auto-filling a guide star's
# magnitude.
SENSOR_BAND_UM = {"R": 0.65, "H": 1.65, "K": 2.20}

# Approximate practical tip-tilt guide magnitudes per sensing band, used only
# for the field-map overlay's usable-vs-too-faint cue (a display heuristic, not
# a hard cutoff). Anchored to the tiptilt model: STRAP's R quadcell is
# calibrated at R=12 and degrades gradually, still usable to ~R 17-18; TRICK is
# floor-limited to a sensing-band magnitude ~14 (TRICK_MAG0) then rolls off
# steeply, so ~15.5 is the practical faint end.
SENSOR_FAINT_LIMIT = {"R": 17.5, "H": 15.5, "K": 15.5}


def pick_mag(mags, want_um):
    """(band_label, value) of the available magnitude whose band is closest in
    wavelength to want_um -- 'best available band', so an IR sensor prefers a
    K/H mag and an optical one prefers R/G. (None, None) if none are usable.
    mags: {band_label: value_or_None}."""
    avail = [(b, v) for b, v in mags.items()
             if v is not None and b in _BAND_UM]
    if not avail:
        return (None, None)
    b, v = min(avail, key=lambda bv: abs(_BAND_UM[bv[0]] - want_um))
    return (b, float(v))


def _est_R(m):
    """Estimate Cousins R from a star's available OPTICAL photometry, using
    published colour transforms. (value, method_label) or None. Ordered by
    reliability; each catalogue only carries one of these band sets so the
    branches are effectively exclusive."""
    g, bp, rp = m.get("G"), m.get("BP"), m.get("RP")
    if None not in (g, bp, rp):
        c = bp - rp                                   # Gaia BP−RP colour
        if -0.5 <= c <= 2.75:                         # validity range
            # Gaia DR2 -> Cousins R (Evans et al. 2018, A&A 616 A4, Table A.2):
            # G − R = −0.003226 + 0.3833(BP−RP) − 0.1345(BP−RP)²
            return (g - (-0.003226 + 0.3833 * c - 0.1345 * c * c),
                    "Gaia G, BP−RP")
    r_, i_ = m.get("r"), m.get("i")
    if None not in (r_, i_):
        # PanSTARRS r,i ≈ SDSS r,i -> Cousins R (Lupton 2005 SDSS transform):
        # R = r − 0.2936(r−i) − 0.1439
        return (r_ - 0.2936 * (r_ - i_) - 0.1439, "PanSTARRS r, i")
    if m.get("F") is not None:
        # GSC photographic F is a red plate ≈ Cousins R (zero-order)
        return (m["F"], "GSC F (photographic red)")
    v, b = m.get("V"), m.get("B")
    if None not in (v, b):
        # V and B−V -> V−R via a mean main-sequence colour (rough): for
        # A–K dwarfs V−R ≈ 0.5(B−V). Weakest of the optical estimates.
        return (v - 0.5 * (b - v), "V, B−V (main-seq)")
    return None


def estimate_sensing_mag(mags, want_band):
    """Best magnitude in the tip-tilt SENSING band, as (value, kind, label):
      kind 'exact' -- the catalogue reports that band directly (label = band);
      kind 'est'   -- derived from a published colour transform (label = the
                      transform used), i.e. a genuine estimate of want_band;
      kind 'near'  -- last-resort nearest available band (label names it), when
                      no transform applies (e.g. only IR mags but R wanted);
      (None, None, None) if nothing is usable.
    want_band in {'R','H','K'}: R for STRAP/NGS optical sensing, H/K for TRICK.
    Near-IR (H/K) is only taken from real near-IR photometry (2MASS/UCAC4),
    never synthesised from optical colours -- that would need a spectral type.
    """
    if mags.get(want_band) is not None:
        return (float(mags[want_band]), "exact", want_band)
    got = _est_R(mags) if want_band == "R" else None
    if got is not None:
        return (float(got[0]), "est", got[1])
    b, v = pick_mag(mags, SENSOR_BAND_UM[want_band])
    if v is not None:
        return (float(v), "near", f"nearest band {b}")
    return (None, None, None)


# --- Interstellar-reddening safety for OPTICAL sensing from IR photometry ----
# Near-IR->optical interstellar extinction ratios (Rieke & Lebofsky 1985:
# A_R/A_V=0.75, A_K/A_V=0.11, A_J/A_K=2.5, A_H/A_K=1.55). Dust dims the optical
# ~7x more than K, so an R magnitude taken from a REDDENED star's near-IR
# photometry (estimate_sensing_mag's 'near' fallback -- R guessed as ~J) is
# drastically too bright: the star can be optically invisible to a STRAP-class
# WFS while still bright in K. This is the dusty-field failure mode (e.g. the
# Galactic Centre). Used ONLY to warn / rank conservatively, never to silently
# "correct" a magnitude -- an exact correction needs the intrinsic colour
# (spectral type), so we can only BOUND the extinction from the colour excess.
A_R_OVER_A_K = 6.7
_E_JK_PER_A_K = 1.52          # E(J-K) = A_K * (A_J/A_K - 1) = A_K * 1.52
_E_HK_PER_A_K = 0.56          # E(H-K) = A_K * (A_H/A_K - 1) = A_K * 0.56
# Reddest ~normal stellar photosphere: colours redder than this are taken as
# reddening. Deliberately conservative -- a genuine late-M giant can reach
# ~1.3 in J-K, so this errs toward warning, which is the point (and an M
# giant's optical mag really IS much fainter than its J anyway, so a nearest-
# band R estimate is optimistic for it too). Tunable.
INTRINSIC_JK_MAX = 1.0
INTRINSIC_HK_MAX = 0.35
_REDDENING_WARN_A_R = 1.0     # below this implied A_R (mag) don't bother warning


def optical_extinction_lower_bound(mags):
    """Approximate LOWER BOUND on optical (R-band) interstellar extinction A_R
    (magnitudes) implied by a star's near-IR colour EXCESS -- for use when an
    optical sensing magnitude has to be guessed from near-IR photometry (see
    estimate_sensing_mag's 'near' fallback). Returns (a_r, note):
      a_r  : >= 0 magnitudes of implied optical dimming (0.0 when the IR colour
             is normal, or when there aren't two IR bands to form a colour);
      note : a short human string ('J-K=3.0, A_R>=12 mag') when a_r exceeds the
             warn threshold, else None.
    LOWER bound because the excess is measured above the REDDEST normal
    photosphere -- an intrinsically bluer star at the same observed colour is
    even more reddened. An honest point estimate needs the spectral type, so
    this only ever drives a 'verify' warning / a conservative ranking, not a
    silent magnitude fix. Dust is the assumed cause (dominant for an IR-only
    source in a crowded field); an intrinsically cool SED would also make the
    band-to-band optical estimate too bright, so flagging is right either way.
    Only meaningful for OPTICAL (R) sensing -- an IR sensor (TRICK H/K) reads
    the near-IR photometry directly and is barely affected by dust."""
    j, h, k = mags.get("J"), mags.get("H"), mags.get("K")
    if j is not None and k is not None:
        a_k = max((j - k) - INTRINSIC_JK_MAX, 0.0) / _E_JK_PER_A_K
        color = f"J-K={j - k:.1f}"
    elif h is not None and k is not None:
        a_k = max((h - k) - INTRINSIC_HK_MAX, 0.0) / _E_HK_PER_A_K
        color = f"H-K={h - k:.1f}"
    else:
        return (0.0, None)
    a_r = A_R_OVER_A_K * a_k
    if a_r < _REDDENING_WARN_A_R:
        return (0.0, None)
    return (a_r, f"{color}, A_R>={a_r:.0f} mag")
