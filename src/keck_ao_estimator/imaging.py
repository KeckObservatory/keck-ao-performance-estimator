"""Sky imagery for the field map: fetching a DSS/2MASS survey cutout from the
CDS hips2fits service, and resampling a local FITS (including multi-extension
mosaics, e.g. GSAOI's four detectors) onto the field-map offset grid.

Deliberately Qt-free (unlike most of the GUI's supporting code) so it is
headless-testable and usable outside the GUI (a deliberate refactor
decision). All heavy third-party dependencies (PIL, astropy.wcs, scipy.ndimage) are
imported lazily inside the functions that need them, exactly as before the
move, so importing this module has no cost beyond numpy.
"""
import io
import re
import urllib.parse
import urllib.request

import numpy as np

HIPS2FITS_URL = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
# CDS runs mirrors of the same service, and they fail INDEPENDENTLY.
# Measured 2026-07-28, when the field-map backdrop stopped loading:
# alasky returned nothing for a 300 px cutout even at a 180 s timeout
# (its bare root page took 36 s to serve 191 bytes), while alaskybis
# served the identical request in 1.1 s.  DNS and TCP connect to BOTH
# were ~0.2 s, so this was the service, not the network -- which is
# exactly the case a single hard-coded host cannot survive.  Hosts are
# tried in order with a SHORT per-host timeout, and the one that works
# is remembered so the steady state costs one request, not two.
HIPS2FITS_HOSTS = ("https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
                   "https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits")
SKY_FETCH_TIMEOUT_S = 25       # per host; failover must stay responsive
_SKY_HOST_OK = None            # last host that worked, tried first next time
SKY_IMAGE_PX = 1000            # cutout size fetched/rendered, pixels


def _hips2fits_url(hips, ra_deg, dec_deg, fov_deg, size_px=SKY_IMAGE_PX):
    """CDS hips2fits cutout URL: a `size_px` square JPEG of `hips`, `fov_deg`
    wide, centred on (ra,dec), TAN projection (North up, East left)."""
    # keep the survey id's slashes literal (CDS wants CDS/P/DSS2/red, not
    # the %2F-escaped form)
    q = urllib.parse.urlencode({
        "hips": hips, "ra": f"{ra_deg:.6f}", "dec": f"{dec_deg:.6f}",
        "fov": f"{fov_deg:.6f}", "width": size_px, "height": size_px,
        "projection": "TAN", "format": "jpg"},
        quote_via=urllib.parse.quote, safe="/")
    return f"{HIPS2FITS_URL}?{q}"


def _sky_host_variants(url):
    """`url` retargeted at each hips2fits host, best-known host first."""
    from urllib.parse import urlsplit, urlunsplit
    parts = urlsplit(url)
    hosts = list(HIPS2FITS_HOSTS)
    if _SKY_HOST_OK in hosts:                 # stick with what just worked
        hosts.remove(_SKY_HOST_OK)
        hosts.insert(0, _SKY_HOST_OK)
    out = []
    for base in hosts:
        b = urlsplit(base)
        out.append((base, urlunsplit((b.scheme, b.netloc, b.path,
                                      parts.query, parts.fragment))))
    return out


def _fetch_sky_jpeg(url, timeout=SKY_FETCH_TIMEOUT_S):
    """Fetch a hips2fits JPEG -> grayscale float array (North up, East left).

    Tries every host in HIPS2FITS_HOSTS (see that constant for why) with
    `timeout` EACH, and remembers the one that answered. Raises the last
    error, with every host's failure in the message, so the GUI's status
    line says which mirrors were tried rather than just "timed out".
    """
    global _SKY_HOST_OK
    import ssl
    import certifi
    from PIL import Image
    ctx = ssl.create_default_context(cafile=certifi.where())
    errs = []
    for base, u in _sky_host_variants(url):
        try:
            data = urllib.request.urlopen(u, timeout=timeout,
                                          context=ctx).read()
            arr = np.asarray(Image.open(io.BytesIO(data)).convert("L"),
                             dtype=float)
            _SKY_HOST_OK = base
            return arr
        except Exception as e:                # try the next mirror
            host = base.split("/")[2]
            errs.append(f"{host}: {type(e).__name__}")
    raise RuntimeError("hips2fits unreachable -- " + "; ".join(errs))


_WCS_NUMERIC_KEY = re.compile(
    r"^(CD\d_\d|PC\d_\d|CDELT\d|CRPIX\d|CRVAL\d|LONPOLE|LATPOLE|PV\d+_\d+)$")


def _coerce_wcs_numeric_strings(hdr):
    """Some archival FITS (e.g. KOA-exported Keck/NIRC2 headers) write WCS
    matrix keywords like CD1_1 as FITS STRING cards ('-0.000002764418')
    rather than floating-point ones. astropy's WCS silently ignores a
    non-numeric CD/PC/CDELT card (has_cd() stays False) and falls back to an
    identity 1 deg/pixel scale -- which still parses as a "valid" celestial
    WCS (has_celestial True) but is wrong by ~9 orders of magnitude, so a
    later arcsec-scale offset blows up astropy's angle validation instead of
    failing where the bad header actually is. Return a copy of hdr with any
    such numeric-looking string values coerced to float."""
    out = hdr.copy()
    for card in hdr.cards:
        key = str(card.keyword)
        if _WCS_NUMERIC_KEY.match(key) and isinstance(card.value, str):
            try:
                out[key] = float(card.value)
            except ValueError:
                pass
    return out


def _celestial_wcs_for_header(hdr):
    """Return (WCS, note) for placing a FITS image on the sky. Prefers a real
    celestial WCS in the header; when there is none (e.g. a Keck/OSIRIS frame
    that carries only pointing + plate-scale + PA keywords) synthesize a TAN
    WCS from those. Raises ValueError if there is not enough to place it."""
    import warnings
    from astropy.wcs import WCS, FITSFixedWarning
    hdr = _coerce_wcs_numeric_strings(hdr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)   # MJD/DATE etc.
        w = WCS(hdr)
    if w.has_celestial:
        cel = w.celestial
        if any(t.startswith(("RA", "GLON", "ELON")) for t in cel.wcs.ctype):
            return cel, "file WCS"

    def _first(*keys):
        for k in keys:
            v = hdr.get(k)
            if v not in (None, "", "0", 0):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return None

    ra = _first("CRVAL1", "RA", "TARGRA")
    dec = _first("CRVAL2", "DEC", "TARGDEC")
    spec = str(hdr.get("INSTR", "")).lower().startswith("spec")
    scale = (_first("SSCALE", "PSCALE", "PIXSCALE", "SECPIX") if spec
             else _first("PSCALE", "SSCALE", "PIXSCALE", "SECPIX"))
    pa = (_first("PA_SPEC", "PA_IMAG", "ROTPOSN") if spec
          else _first("PA_IMAG", "PA_SPEC", "ROTPOSN")) or 0.0
    if ra is None or dec is None or not scale:
        raise ValueError(
            "FITS has no celestial WCS and no usable pointing/plate-scale "
            "keywords (need RA/DEC + a pixel scale) to place it on the field")
    ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])
    s = scale / 3600.0                       # deg / pixel
    th = np.radians(pa)                       # sky position angle (N through E)
    ct, st = np.cos(th), np.sin(th)
    syn = WCS(naxis=2)
    syn.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    syn.wcs.crval = [ra, dec]
    syn.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    # N-up / E-left at PA=0 (CD1_1<0), rotated by the sky PA
    syn.wcs.cd = [[-s * ct, -s * st], [-s * st, s * ct]]
    return syn, (f"no file WCS — placed from pointing + {scale:g}″/px, "
                 f"PA {pa:g}° (orientation assumed; verify vs known stars)")


def _fits_obs_time_hst(hdr):
    """Observation time as an HST datetime from a FITS header, or None.
    Prefers MJD-OBS (unambiguous); falls back to DATE-OBS + UTC/UT. Hawaii has
    no DST, so HST = UTC − 10 h."""
    from datetime import timedelta
    try:
        from astropy.time import Time
        if hdr.get("MJD-OBS") not in (None, ""):
            t = Time(float(hdr["MJD-OBS"]), format="mjd", scale="utc").to_datetime()
        else:
            date = str(hdr.get("DATE-OBS", "")).split("T")[0]
            utc = hdr.get("UTC") or hdr.get("UT") or hdr.get("TIME-OBS")
            if not date or not utc:
                return None
            t = Time(f"{date}T{utc}", format="isot", scale="utc").to_datetime()
        return t - timedelta(hours=10)
    except Exception:
        return None


def sky_image_from_fits(path, n=SKY_IMAGE_PX, center=None, half=None):
    """Resample a local FITS onto the field-map offset grid (x=West+, y=North+).

    Two roles, selected by the optional center/half:
      * INSCRIBED FRAME (center=half=None, the default) -- placed over its OWN
        NATIVE angular extent, centred on the IMAGE's OWN pointing, so a real
        science frame defines the field.
      * BACKDROP (center + half given) -- resampled onto a field grid of the
        given half-extent centred on the given SkyCoord (e.g. an inscribed
        OSIRIS/NIRC2 frame's pointing or the science target), positioned purely
        by its own WCS. This lets a wide local image (e.g. a GSAOI mosaic) be
        used like a DSS/2MASS backdrop, centred on the science field rather than
        on itself. If center is None a fallback centre is the image's own
        pointing, so a backdrop still works with nothing else loaded.

    Returns (array, note, center, obs_hst, name, half_arcsec): an (n,n) float
    array (row 0 = North, col 0 = East -- imshow origin='upper', E-left,
    matching the online cutout), how the image was placed (real file WCS vs a
    synthesized one), the field-centre SkyCoord actually used, the HST
    observation time, the target name, and the half-extent (arcsec) the grid
    spans. Handles arbitrary WCS rotation via astropy + scipy (no reproject
    dependency); a spectral cube is collapsed to a white-light image. A
    multi-extension FITS (MEF, e.g. GSAOI's four detectors) is mosaicked: every
    science image extension is resampled onto the shared sky grid via its OWN
    WCS and combined (gaps are NaN/transparent), with pointing/target/time taken
    from the primary HDU."""
    import warnings
    from astropy.utils.exceptions import AstropyWarning
    # extension names that are NOT science pixels (data-quality / variance /
    # weight planes some reduced MEFs carry) -- never mosaic these in
    _NONSCI = {"DQ", "VAR", "ERR", "WHT", "WEIGHT", "MASK", "VARIANCE",
               "UNCERT", "EXP", "EXPMAP"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        from astropy.io import fits
        with fits.open(path) as hdul:
            prim = hdul[0].header            # a MEF keeps pointing/target/time here
            img_hdus = [h for h in hdul
                        if getattr(h, "data", None) is not None
                        and np.asarray(h.data).ndim >= 2
                        and str(h.header.get("EXTNAME", "")).strip().upper()
                        not in _NONSCI]
            if not img_hdus:
                raise ValueError("FITS has no 2-D image data to place on the field")

            def _merged(hdr):
                """Primary header (shared pointing/scale keys a MEF keeps only in
                HDU 0) overlaid with the extension's own keys (its per-detector
                WCS wins)."""
                m = fits.Header(); m.update(prim); m.update(hdr)
                return m

            # The image is placed purely by its own WCS: the resampling below
            # goes sky -> pixel through wcs.world_to_pixel onto a fixed
            # N-up/E-left sky grid, so any parity/rotation a CORRECT WCS
            # carries is already handled -- no instrument-specific fixup. A
            # file whose stored pixels are genuinely mirrored relative to
            # their WCS (a rare raw-frame quirk) is corrected by eye with the
            # field-map's Backdrop/Frame flip controls, not silently here
            # (an earlier INSTRUME=="GSAOI" auto-flip mirrored correctly-WCS'd
            # reduced GSAOI mosaics -- see the field-map flip controls).
            layers = []                      # (data2d, wcs) per science extension
            note = None
            for h in img_hdus:
                d = np.asarray(h.data, dtype=float)
                while d.ndim > 2:            # OSIRIS cube -> white light
                    d = np.nanmedian(d, axis=0)
                # prefer the extension's own WCS; only consult the primary for
                # pointing/plate-scale when the extension alone can't be placed
                try:
                    w, note = _celestial_wcs_for_header(h.header)
                except ValueError:
                    w, note = _celestial_wcs_for_header(_merged(h.header))
                layers.append((d, w))
            meta = _merged(img_hdus[0].header)
            obs_hst = _fits_obs_time_hst(meta)
            name = str(meta.get("TARGNAME") or meta.get("OBJECT") or "").strip()

    return _place_layers_on_grid(layers, note, obs_hst, name, n, center, half)


def _place_layers_on_grid(layers, note, obs_hst, name, n, center, half):
    """Resample already-loaded (data2d, WCS) layers onto the field-map offset
    grid (x = West+, y = North+) and return the sky_image_from_fits 6-tuple.
    Shared by sky_image_from_fits (FITS extensions) and sky_image_from_png (a
    single PNG layer), so both place an image on the field identically -- the
    only difference is how the pixels + WCS were obtained."""
    import warnings
    from astropy.wcs.utils import proj_plane_pixel_scales
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from scipy.ndimage import map_coordinates

    # the image's OWN centre + native half-extent (arcsec)
    if len(layers) == 1:
        data, wcs = layers[0]
        ny, nx = data.shape
        native_center = wcs.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)
        pixscale = float(np.mean(proj_plane_pixel_scales(wcs))) * 3600.0
        native_half = max(nx, ny) / 2.0 * pixscale
    else:
        # multi-extension mosaic (e.g. GSAOI's four detectors): centre =
        # mean of the extension centres; half = the largest corner
        # separation from it, so a ±half square covers every tile
        cens = [w.pixel_to_world((d.shape[1] - 1) / 2.0,
                                 (d.shape[0] - 1) / 2.0) for d, w in layers]
        xyz = np.mean([c.icrs.cartesian.xyz.value for c in cens], axis=0)
        sph = SkyCoord(x=xyz[0], y=xyz[1], z=xyz[2], frame="icrs",
                       representation_type="cartesian").spherical
        native_center = SkyCoord(ra=sph.lon, dec=sph.lat, frame="icrs")
        native_half = 0.0
        for d, w in layers:
            ny, nx = d.shape
            cw = w.pixel_to_world([0, nx - 1, 0, nx - 1],
                                  [0, 0, ny - 1, ny - 1])
            native_half = max(native_half,
                              float(np.max(native_center.separation(cw).arcsec)))
    # inscribe (default) uses the image's own centre/extent; a backdrop
    # (center/half given) uses the caller's field grid, falling back to the
    # image's own centre when none is supplied
    center = center if center is not None else native_center
    half = float(half) if half is not None else native_half

    # grid: row 0 = +North (top), col 0 = East (x = -half, since x is West+)
    xs = np.linspace(-half, half, n)                        # West+ arcsec
    ys = np.linspace(half, -half, n)                        # North+, top first
    xg, yg = np.meshgrid(xs, ys)
    # East offset (RA direction) = -x; North offset (Dec direction) = +y
    coords = center.spherical_offsets_by(-xg.ravel() * u.arcsec,
                                         yg.ravel() * u.arcsec)

    def _sample(data, wcs):
        px, py = wcs.world_to_pixel(coords)
        return map_coordinates(data, [py, px], order=1, mode="constant",
                               cval=np.nan).reshape(n, n)

    if len(layers) == 1:
        samp = _sample(*layers[0])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN gap pixels
            samp = np.nanmean(np.stack([_sample(d, w) for d, w in layers]),
                              axis=0)
        note = f"{note} · {len(layers)}-extension mosaic"
    return samp, note, center, obs_hst, name, half


def sky_image_from_png(path, n=SKY_IMAGE_PX, center=None, half=None):
    """Place a PNG that carries its sky astrometry in an embedded text chunk
    onto the field-map grid -- the same two roles (inscribed frame / backdrop)
    and the same return contract as sky_image_from_fits.

    The PNG must contain a 'SKYWCS' tEXt/iTXt chunk holding a FITS-WCS header
    string (what a light, shippable sky image -- e.g. the bundled GSAOI
    Galactic-Center master -- is exported with). Grayscale pixel values are the
    image; the embedded WCS positions every pixel, so parity/rotation are
    handled exactly as for a FITS. Raises ValueError if the chunk is absent
    (a plain screenshot has no sky position, so it cannot be placed)."""
    from PIL import Image
    from astropy.io import fits
    img = Image.open(path)
    hdr_str = (getattr(img, "text", {}) or {}).get("SKYWCS")
    if not hdr_str:
        raise ValueError(
            "PNG has no embedded 'SKYWCS' astrometry chunk, so it cannot be "
            "placed on the sky -- export it with a WCS, or load a FITS instead.")
    data = np.asarray(img.convert("L"), dtype=float)
    wcs, note = _celestial_wcs_for_header(fits.Header.fromstring(hdr_str))
    name = (getattr(img, "text", {}) or {}).get("OBJECT", "")
    return _place_layers_on_grid([(data, wcs)], note, None, name, n,
                                 center, half)


def sky_image_from_file(path, **kw):
    """Dispatch a local sky image to the PNG or FITS loader by extension, so
    the field map can accept either. Both return the identical 6-tuple."""
    if str(path).lower().endswith(".png"):
        return sky_image_from_png(path, **kw)
    return sky_image_from_fits(path, **kw)
