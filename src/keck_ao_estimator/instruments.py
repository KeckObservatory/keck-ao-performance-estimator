"""Instrument parameter files: what the Strehl tools need to know about an
imager, as data.

Each instrument is one YAML file (``data/instruments/*.yaml`` ships NIRC2
and OSIRIS; more are read from the directories in $KECK_AO_INSTRUMENTS,
separated by ``os.pathsep``). A file says:

* ``identify`` -- which FITS header marks a frame as this instrument's;
* ``frames``   -- the frame file names (regex, frame-number digits, the
  default PREFIX, names to skip);
* ``night``    -- where tonight's frames are (the Latest button);
* ``params``   -- how a header becomes the measurement parameters (plate
  scale, wavelength, coadds, saturation, pupil rotation, time);
* ``reduce``   -- the reduction (flat / bad-pixel mask, backgrounds);
* ``pupil``    -- the telescope pupil for the diffraction-limited reference.

Each section names a ``model``. NIRC2 and OSIRIS use the built-in models
``nirc2`` / ``osiris`` / ``keck`` -- the existing code, so their numbers are
unchanged (Eduardo 2026-10-21: "build it"; the golden frames are the
parity test). ORKID (the K2 AO bench visible camera) uses the built-in
``orkid`` models (orkid.py: two filter wheels, cube planes averaged, the
bench PCU reference blurred by the measured source size, or the Keck pupil
when the header says the telescope was tracking: ``params.daytime: auto``)
and a UT-dated Latest directory on the camera server. A new instrument uses the generic models, which are pure
data: ``params: {model: generic, ...}``, ``reduce: {model: generic}``,
``night: {model: directories}`` and ``pupil: {model: circular}`` (a circular
primary with central obscuration and spider vanes, pupil_psf.py). See
``data/instruments/TEMPLATE.yaml.example`` for every field.

Optional per-instrument fields: ``frames.suffix`` (text after the number,
ORKID's ``_0000``; a ``*`` in it is matched against the directory, OSIRIS
IFU cubes' ``_<filter>_<scale>``), ``frames.watch_by_prefix`` (auto-measure
considers only the PREFIX's files: ORKID writes every prefix into one
directory), ``frames.coadds: unp`` (each frame's coadds are in its NIRC2
``_unp`` file: nirc2_coadds.py), ``night.subdir`` with the ``osiris`` model
(the night's SPEC/ORP instead of IMAG/raw) and ``apertures`` (the GUI's
default photometry / sky / peak radii). An identify rule may carry
``unless_contains: {KEYWORD: text}`` (the rule fails when that keyword holds
the text: OSIRIS imager rules skip INSTR = spec).

The OSIRIS spectrograph (``params/reduce: osiris_ifs``) measures datacubes,
not frames: osiris_ifs.py (FWHM first, SR relative in a small aperture).

identify() tries the specific rules (``contains``) of every instrument
before the catch-all ones (``present``), so an ORKID frame (CAMNAME = orkid,
no INSTRUME) is not taken by NIRC2's rule for old frames (CAMNAME present,
no INSTRUME).

A file that does not load is skipped and reported in load_errors(); the
tools show those messages instead of failing to start.
"""
import glob
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

__all__ = ["INSTRUMENTS_ENV", "BUILTIN_DIR", "Instrument", "FramePrep",
           "load_registry",
           "registry", "load_errors", "get", "names", "identify",
           "find_latest_night", "reload"]

INSTRUMENTS_ENV = "KECK_AO_INSTRUMENTS"
BUILTIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "instruments")
PARAMS_MODELS = ("nirc2", "osiris", "osiris_ifs", "orkid", "generic")
REDUCE_MODELS = ("nirc2", "osiris", "osiris_ifs", "orkid", "generic")
NIGHT_MODELS = ("nirc2", "osiris", "directories", "fixed", "none")
PUPIL_MODELS = ("keck", "orkid", "circular")
APERTURE_KEYS = ("photometry_arcsec", "sky_inner_arcsec", "sky_outer_arcsec",
                 "peak_arcsec")


def _header_str(header, key):
    return str(header.get(key, "") or "").strip()


def _rule_matches(rule, header):
    key = rule["keyword"]
    for other in rule.get("unless", ()):
        if _header_str(header, other):
            return False
    for other, text in rule.get("unless_contains", {}).items():
        if str(text).upper() in _header_str(header, other).upper():
            return False
    if "contains" in rule:
        return str(rule["contains"]).upper() in _header_str(header, key).upper()
    if rule.get("present"):
        return header.get(key) is not None
    return False


@dataclass(frozen=True)
class Instrument:
    """One instrument parameter file, validated. `key` is the lowercase id
    the code uses ("nirc2"); `name` is what the Instrument menu shows."""
    key: str
    name: str
    telescope: str
    order: int
    identify: tuple
    frame_regex: re.Pattern
    digits: int
    default_prefix: str
    skip_regex: re.Pattern | None
    night: dict
    params: dict
    reduce: dict
    pupil: dict
    source: str
    description: str = ""
    skip_reason: str = "not a frame (matches the instrument's skip pattern)"
    suffix: str = ""
    watch_by_prefix: bool = False
    apertures: tuple | None = None
    coadds_file: str = ""       # "unp": coadds in the NIRC2 _unp file

    @property
    def is_ifs(self):
        """True for the OSIRIS spectrograph: datacubes, osiris_ifs.py."""
        return self.params.get("model") == "osiris_ifs"

    # ---- identity and file names
    def matches(self, header):
        """True when `header` belongs to this instrument (any identify rule)."""
        return any(_rule_matches(r, header) for r in self.identify)

    def is_frame(self, name):
        """True for a frame file name of this instrument (and not skipped)."""
        return (self.frame_regex.match(name) is not None
                and not self.is_skipped(name))

    def is_skipped(self, name):
        """True for files that look like frames but are not (NIRC2 _unp)."""
        return self.skip_regex is not None and \
            self.skip_regex.search(name) is not None

    def frame_number(self, name):
        """The frame number in `name` (the pattern's first group), else None."""
        m = self.frame_regex.match(name)
        if m is None or not m.groups() or m.group(1) is None:
            return None
        try:
            return int(m.group(1))
        except ValueError:
            return None

    def frame_name(self, prefix, number):
        """``<prefix><number zero-padded to digits><suffix>.fits``."""
        return f"{prefix}{int(number):0{self.digits}d}{self.suffix}.fits"

    def watches(self, name, prefix=""):
        """True for a frame auto-measure should consider: a frame of this
        instrument and, when the file sets ``watch_by_prefix`` and a PREFIX
        is given, one of that PREFIX (ORKID writes every prefix into one
        directory, so name order across prefixes is not time order)."""
        return self.is_frame(name) and not (
            self.watch_by_prefix and prefix and not name.startswith(prefix))

    @property
    def backgrounds_supported(self):
        """Whether numbered background frames may be subtracted."""
        return bool(self.reduce.get("backgrounds", True))

    # ---- measurement
    def frame_params(self, header):
        """The measurement parameters (a Nirc2FrameParams) for `header`."""
        model = self.params["model"]
        if model == "nirc2":
            from .nirc2 import nirc2_frame_params
            return nirc2_frame_params(header)
        if model == "osiris":
            from .osiris import osiris_frame_params
            return osiris_frame_params(header)
        if model == "osiris_ifs":
            from .osiris_ifs import ifs_frame_params, wavelengths_um
            wl = wavelengths_um(header)
            return ifs_frame_params(header, float(wl[len(wl) // 2]))
        if model == "orkid":
            from .orkid import orkid_frame_params, orkid_header_on_sky
            daytime = self.params.get("daytime", True)
            if str(daytime).strip().lower() == "auto":
                # bench PCU reference unless the telescope was tracking
                daytime = not orkid_header_on_sky(header)
            return orkid_frame_params(header, daytime=bool(daytime))
        return _generic_frame_params(self, header)

    def calibration(self):
        """(flat, mask) for this instrument; either may be None. Raises
        when a configured file cannot be read (the caller shows frames
        unmeasured, as for NIRC2 without its superflat)."""
        model = self.reduce["model"]
        if model == "nirc2":
            from . import image_strehl
            return image_strehl.load_nirc2_calibration()
        if model in ("osiris", "osiris_ifs", "orkid"):
            return None, None
        return _generic_calibration(self)

    def reduce_frame(self, raw, background=None, flat=None, mask=None):
        """Reduce a raw frame for display and measurement."""
        import numpy as np

        from . import image_strehl
        model = self.reduce["model"]
        if model == "nirc2":
            return image_strehl.reduce_frame(raw, background=background,
                                             flat=flat, badmask=mask)
        if model == "osiris":
            return image_strehl.osiris_reduce(raw, crop=False)
        if model == "osiris_ifs":
            # display only: the cube's mean over wavelength (the workers
            # measure cubes with osiris_ifs.collapse and their quality map)
            a = np.asarray(raw, dtype=float)
            return np.nan_to_num(np.nanmean(a, axis=2)) if a.ndim == 3 else a
        if model == "orkid":
            from .orkid import orkid_reduce
            return orkid_reduce(raw, background=background)
        red = np.asarray(raw, dtype=float)
        if background is not None:
            red = red - np.asarray(background, dtype=float)
        if flat is not None:
            if flat.shape != red.shape:
                raise ValueError(f"flat shape {flat.shape} != frame shape "
                                 f"{red.shape} ({self.source})")
            red = red / flat
        if mask is not None:
            if mask.shape != red.shape:
                raise ValueError(f"bad-pixel mask shape {mask.shape} != frame "
                                 f"shape {red.shape} ({self.source})")
            if mask.any():
                red = image_strehl.deadpix_fill(red, mask)
        if self.reduce.get("fix_image", False):
            red = image_strehl.fix_image(red)
        return red

    def source_mfd_um(self):
        """ORKID bench source blur as the equivalent fiber mode field
        diameter at the PCU focus (um): MFD = 4 sigma x 0.727 um/mas, from
        ``pupil.source_blur_sigma_mas`` (default: the measured 5.8 mas)."""
        from .orkid import ORKID_BENCH_BLUR_SIGMA_MAS
        sigma = float(self.pupil.get("source_blur_sigma_mas",
                                     ORKID_BENCH_BLUR_SIGMA_MAS))
        return round(4 * sigma * 0.727, 1)

    def dl_key(self, params):
        """What the diffraction-limited PSF depends on (a cache key)."""
        return (self.key, params.camname, params.pmsname,
                round(params.effwave_um, 6), round(params.pmrangl_deg, 3),
                params.daytime, params.sfp, params.plate_scale_mas)

    def dl_psf(self, params, npix=512):
        """The diffraction-limited PSF on the detector grid, unit sum."""
        if self.is_ifs:
            from .osiris_ifs import ifs_dl_psf
            return ifs_dl_psf(params, npix=npix)
        if self.pupil["model"] == "keck":
            from .nirc2_psf import nirc2_dl_psf
            return nirc2_dl_psf(params.camname, params.pmsname,
                                params.effwave_um, params.pmrangl_deg,
                                npix=npix, daytime=params.daytime,
                                sfp=params.sfp)
        if self.pupil["model"] == "orkid":
            from .orkid import orkid_dl_psf
            return orkid_dl_psf(params.effwave_um, params.plate_scale_mas,
                                params.pmsname, params.pmrangl_deg, npix=npix,
                                daytime=params.daytime, mfd_um=self.source_mfd_um())
        from .pupil_psf import circular_dl_psf
        p = self.pupil
        spiders = p.get("spiders") or {}
        return circular_dl_psf(
            float(p["diameter_m"]), params.plate_scale_mas,
            params.effwave_um, npix=npix,
            obscuration_m=float(p.get("obscuration_m", 0.0)),
            spiders=int(spiders.get("count", 0)),
            spider_width_m=float(spiders.get("width_m", 0.0)),
            spider_angle_deg=float(spiders.get("angle_deg", 0.0)),
            rotation_deg=params.pmrangl_deg)


class FramePrep:
    """Per-run instrument state for the measurement workers: each
    instrument's calibration is loaded once (a failure is kept as the
    reason, and the frames are then shown but not measured), and one
    diffraction-limited PSF is kept (512^2, rebuilt when what it depends
    on changes)."""

    def __init__(self):
        self._cal = {}
        self._dl_key = None
        self._dl = None

    def calibration(self, inst):
        """(flat, mask, error, first): error is None or the reason the
        calibration is unavailable; first is True the first time this run
        asks for this instrument (report the error once)."""
        first = inst.key not in self._cal
        if first:
            try:
                flat, mask = inst.calibration()
                self._cal[inst.key] = (flat, mask, None)
            except Exception as e:
                self._cal[inst.key] = (None, None, "calibration unavailable "
                                       f"({type(e).__name__}: {e})")
        return self._cal[inst.key] + (first,)

    def dl_psf(self, inst, params, npix=512):
        """The diffraction-limited PSF for `params` (cached while unchanged)."""
        key = inst.dl_key(params) + (npix,)
        if key != self._dl_key:
            self._dl, self._dl_key = None, None     # at most one around
            self._dl = inst.dl_psf(params, npix=npix)
            self._dl_key = key
        return self._dl


# ------------------------------------------------------------ generic models
def _float_from(header, key, scale=1.0):
    v = header.get(key)
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        return float(v) * scale
    except (TypeError, ValueError):
        return None


def _generic_utc(header, p):
    date_obs = _header_str(header, p.get("date_keyword", "DATE-OBS"))
    if not date_obs:
        return None
    if "T" in date_obs:
        text = date_obs
    else:
        tval = _header_str(header, p.get("time_keyword", "UTC"))
        if not tval:
            return None
        text = f"{date_obs}T{tval}"
    try:
        return datetime.fromisoformat(text.split(".")[0])
    except ValueError:
        return None


def _generic_frame_params(inst, header):
    from .nirc2 import Nirc2FrameParams, opt_header_float
    p = inst.params
    scale = p.get("plate_scale_mas")
    if p.get("plate_scale_keyword"):
        scale = _float_from(header, p["plate_scale_keyword"],
                            float(p.get("plate_scale_keyword_scale", 1.0))) \
            or scale
    if not scale or float(scale) <= 0:
        raise ValueError(f"{inst.name}: no plate scale ({inst.source})")
    w = p.get("wavelength") or {}
    effwave = None
    if w.get("keyword"):
        effwave = _float_from(header, w["keyword"],
                              float(w.get("keyword_scale", 1.0)))
    filter_name = _header_str(header, w.get("filter_keyword", "FILTER"))
    if not effwave or effwave <= 0:
        table = {str(k).strip().lower(): float(v)
                 for k, v in (w.get("filters") or {}).items()}
        effwave = table.get(filter_name.lower(), w.get("default_um"))
    if not effwave or float(effwave) <= 0:
        raise ValueError(f"{inst.name}: no wavelength for filter "
                         f"{filter_name!r} ({inst.source})")
    coadds_key = p.get("coadds_keyword", "COADDS")
    coadds = int(_float_from(header, coadds_key) or 1) if coadds_key else 1
    sat = p.get("saturation_adu")
    rot = float(p.get("pupil_rotation_deg", 0.0))
    if p.get("pupil_rotation_keyword"):
        rot += float(p.get("pupil_rotation_sign", 1.0)) * (
            _float_from(header, p["pupil_rotation_keyword"]) or 0.0)
    return Nirc2FrameParams(
        camname=inst.key,
        pmsname=inst.pupil["model"],
        effwave_um=float(effwave),
        pmrangl_deg=rot,
        coadds=max(coadds, 1),
        max_counts=float(sat) if sat else math.inf,
        daytime=False,
        plate_scale_mas=float(scale),
        utc=_generic_utc(header, p),
        object_name=_header_str(header, p.get("object_keyword", "OBJECT")),
        ra=_header_str(header, p.get("ra_keyword", "RA")),
        dec=_header_str(header, p.get("dec_keyword", "DEC")),
        az_deg=opt_header_float(header, p.get("az_keyword", "AZ")),
        el_deg=opt_header_float(header, p.get("el_keyword", "EL")),
        airmass=opt_header_float(header, p.get("airmass_keyword", "AIRMASS")),
        filter_name=filter_name,
    )


def _resolve(inst, path):
    path = os.path.expanduser(str(path))
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(inst.source), path)
    return path


def _generic_calibration(inst):
    import numpy as np
    from astropy.io import fits
    r = inst.reduce
    flat = mask = None
    if r.get("flat"):
        flat = np.asarray(fits.getdata(_resolve(inst, r["flat"])), dtype=float)
        flat = np.maximum(flat, float(r.get("flat_floor", 0.2)))
    if r.get("bad_pixel_mask"):
        mask = np.asarray(fits.getdata(_resolve(inst, r["bad_pixel_mask"]))
                          ).astype(bool)
    return flat, mask


# ------------------------------------------------------------ loading
_REQUIRED = {"key": str, "name": str, "identify": list, "frames": dict,
             "params": dict, "pupil": dict}


def _model(section, allowed, what, source, default=None):
    m = (section or {}).get("model", default)
    if m not in allowed:
        raise ValueError(f"{what}.model must be one of {', '.join(allowed)} "
                         f"(got {m!r})")
    return m


def _build(data, source):
    if not isinstance(data, dict):
        raise ValueError("the file must hold one mapping (key: value lines)")
    for k, t in _REQUIRED.items():
        if not isinstance(data.get(k), t):
            raise ValueError(f"missing or malformed '{k}'")
    key = data["key"].strip().lower()
    if not re.match(r"^[a-z0-9_]+$", key):
        raise ValueError(f"key {key!r}: use lowercase letters, digits, _")
    rules = []
    for r in data["identify"]:
        if not isinstance(r, dict) or "keyword" not in r or not (
                "contains" in r or r.get("present")):
            raise ValueError("each identify rule needs 'keyword' and "
                             "'contains' (or 'present: true')")
        uc = r.get("unless_contains", {}) or {}
        if not isinstance(uc, dict):
            raise ValueError("unless_contains must map KEYWORD: text")
        rules.append(dict(r, unless=tuple(r.get("unless", ())),
                          unless_contains=dict(uc)))
    fr = data["frames"]
    if "pattern" not in fr or "digits" not in fr:
        raise ValueError("frames needs 'pattern' and 'digits'")
    params = dict(data["params"])
    _model(params, PARAMS_MODELS, "params", source)
    reduce = dict(data.get("reduce") or {"model": "generic"})
    _model(reduce, REDUCE_MODELS, "reduce", source)
    night = dict(data.get("night") or {"model": "none"})
    _model(night, NIGHT_MODELS, "night", source)
    pupil = dict(data["pupil"])
    _model(pupil, PUPIL_MODELS, "pupil", source)
    if night["model"] == "fixed" and not str(night.get("path", "")).strip():
        raise ValueError("night.model fixed needs a path")
    apertures = None
    if data.get("apertures") is not None:
        ap = data["apertures"]
        if not isinstance(ap, dict) or any(k not in ap for k in APERTURE_KEYS):
            raise ValueError("apertures needs " + ", ".join(APERTURE_KEYS))
        apertures = tuple(float(ap[k]) for k in APERTURE_KEYS)
        if not (0 < apertures[0] <= apertures[1] < apertures[2]) or apertures[3] <= 0:
            raise ValueError("apertures: need 0 < photometry <= sky_inner "
                             "< sky_outer and peak > 0")
    if pupil["model"] == "circular" and float(pupil.get("diameter_m", 0)) <= 0:
        raise ValueError("pupil.model circular needs diameter_m > 0")
    if params["model"] == "generic" and not (
            params.get("plate_scale_mas") or params.get("plate_scale_keyword")):
        raise ValueError("params.model generic needs plate_scale_mas")
    skip = fr.get("skip")
    return Instrument(
        key=key, name=str(data["name"]).strip(),
        telescope=str(data.get("telescope", "")).strip(),
        order=int(data.get("order", 100)), identify=tuple(rules),
        frame_regex=re.compile(fr["pattern"]), digits=int(fr["digits"]),
        default_prefix=str(fr.get("prefix", "") or ""),
        skip_regex=re.compile(skip, re.IGNORECASE) if skip else None,
        night=night, params=params, reduce=reduce, pupil=pupil,
        source=source, description=str(data.get("description", "")).strip(),
        suffix=str(fr.get("suffix", "") or ""),
        watch_by_prefix=bool(fr.get("watch_by_prefix", False)),
        apertures=apertures,
        coadds_file=_coadds_file(fr),
        **({"skip_reason": str(fr["skip_reason"])}
           if fr.get("skip_reason") else {}))


def _coadds_file(frames):
    c = str(frames.get("coadds", "") or "").strip().lower()
    if c not in ("", "unp"):
        raise ValueError(f"frames.coadds must be 'unp' (got {c!r})")
    return c


def _search_dirs(extra_dirs=None):
    dirs = [BUILTIN_DIR]
    env = os.environ.get(INSTRUMENTS_ENV, "")
    dirs += [d for d in env.split(os.pathsep) if d.strip()]
    dirs += list(extra_dirs or ())
    return [os.path.expanduser(d) for d in dirs]


def load_registry(extra_dirs=None):
    """Read every ``*.yaml`` in the built-in directory, then in
    $KECK_AO_INSTRUMENTS and `extra_dirs` (a later file with the same key
    replaces an earlier one). Returns ({key: Instrument} in menu order,
    [error messages for files that did not load])."""
    import yaml
    found, errors = {}, []
    for d in _search_dirs(extra_dirs):
        for path in sorted(glob.glob(os.path.join(d, "*.yaml"))
                           + glob.glob(os.path.join(d, "*.yml"))):
            try:
                with open(path, encoding="utf-8") as fh:
                    inst = _build(yaml.safe_load(fh), path)
            except Exception as e:
                errors.append(f"{path}: {type(e).__name__}: {e}")
                continue
            found[inst.key] = inst
    ordered = dict(sorted(found.items(), key=lambda kv: (kv[1].order, kv[0])))
    return ordered, errors


_REGISTRY = None


def reload(extra_dirs=None):
    """Re-read the parameter files (after adding one, or in tests)."""
    global _REGISTRY
    _REGISTRY = load_registry(extra_dirs)
    return _REGISTRY[0]


def registry():
    """{key: Instrument} in menu order, loaded once per process."""
    if _REGISTRY is None:
        reload()
    return _REGISTRY[0]


def load_errors():
    """Messages for parameter files that were skipped."""
    if _REGISTRY is None:
        reload()
    return list(_REGISTRY[1])


def get(key):
    """The Instrument for `key` or display name (case-insensitive); KeyError."""
    k = str(key).strip().lower()
    reg = registry()
    if k in reg:
        return reg[k]
    for inst in reg.values():
        if inst.name.lower() == k:
            return inst
    raise KeyError(f"no instrument {key!r} (known: {', '.join(names())})")


def names():
    """Display names in menu order."""
    return [i.name for i in registry().values()]


def identify(header):
    """The instrument whose identify rules match, else None: first any
    instrument (menu order) with a matching ``contains`` rule, then one with
    a matching ``present`` rule -- a specific rule wins over a catch-all."""
    for specific in (True, False):
        for inst in registry().values():
            if any(_rule_matches(r, header) for r in inst.identify
                   if ("contains" in r) == specific):
                return inst
    return None


# ------------------------------------------------------------ Latest
def _night_date(name, fmt):
    try:
        return datetime.strptime(name, fmt).date()
    except ValueError:
        return None


def find_latest_night(inst, utc_now=None):
    """Tonight's frame directory for `inst` as (path, is_tonight), or
    (None, False). Built-in models call the existing NIRC2 / OSIRIS
    finders; ``directories`` scans ``roots`` (globs allowed) for night
    directories named by ``name_format`` (strftime), takes the latest
    dated no later than tomorrow's UT, between equal dates the one written
    last, and appends ``subdir``. ``none`` has no Latest."""
    from . import frame_watch
    model = inst.night["model"]
    kw = {} if utc_now is None else {"utc_now": utc_now}
    if model == "nirc2":
        return frame_watch.find_latest_night_dir(**kw)
    if model == "osiris":
        path, tonight = frame_watch.find_latest_osiris_night(**kw)
        sub = str(inst.night.get("subdir", "") or "").strip("/")
        if path and sub:
            # the same night's spectrograph cubes (SPEC/ORP), not IMAG/raw
            path = re.sub(r"/IMAG/raw/?$", "/" + sub, path)
        return path, tonight
    if model == "none":
        return None, False
    if model == "fixed":
        # one directory (ORKID's camera server output); strftime codes take
        # the UT date (ORKID: /sdata1651/%y%m%d, 261023 for the night of
        # 2026-10-22 HST); host:/dir paths are polled over rsync like any
        # remote PATH
        path = (utc_now or datetime.now(timezone.utc)).strftime(
            str(inst.night["path"]).strip())
        return (path if ":" in path.split("/")[0] else os.path.expanduser(path)), True
    today = (utc_now or datetime.now(timezone.utc)).date()
    horizon = today + timedelta(days=1)
    fmt = inst.night.get("name_format", "%Y%m%d")
    nights = []
    for pattern in inst.night.get("roots", ()):
        for root in sorted(glob.glob(os.path.expanduser(pattern))):
            try:
                with os.scandir(root) as it:
                    for entry in it:
                        d = _night_date(entry.name, fmt)
                        if d is not None and d <= horizon and entry.is_dir():
                            nights.append((d, entry.path))
            except OSError:
                continue
    if not nights:
        return None, False
    latest = max(d for d, _ in nights)
    sub = inst.night.get("subdir", "") or ""

    def _written(p):
        try:
            return os.stat(os.path.join(p, sub) if sub else p).st_mtime
        except OSError:
            return -1.0
    best = max(sorted(p for d, p in nights if d == latest), key=_written)
    return (os.path.join(best, sub) if sub else best), latest >= today


