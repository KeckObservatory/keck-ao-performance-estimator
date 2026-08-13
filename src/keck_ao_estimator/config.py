"""Argument resolution shared by the CLI and the GUI: turning raw --flags
into the derived settings prepare_night() needs (wavelength, tomography
on/off, TT sensor + dichroic band swap, output filename, observing windows).
"""
from datetime import datetime, timedelta

from .constants import PHOTOMETRIC_BANDS, LAMBDA_K_NM


def parse_night(date_str):
    """'YYYY-MM-DD' -> datetime (midnight of the evening/civil date)."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise SystemExit(f"ERROR: --night '{date_str}' is not YYYY-MM-DD") from None


def resolve_wavelength(args):
    """Resolve the science wavelength (nm) from --wavelength or --band.
    --wavelength wins; then --band; default K (2200 nm). Returns (nm, label)."""
    if args.wavelength is not None:
        nm = float(args.wavelength)
        # find a band name if it matches one closely, else just show nm
        label = f"{nm:.0f} nm"
        for b, w in PHOTOMETRIC_BANDS.items():
            if abs(w - nm) < 1.0:
                label = f"{b}-band ({nm:.0f} nm)"
                break
        return nm, label
    if args.band is not None:
        key = args.band.strip()
        # case-insensitive match, but keep Ks distinct
        for b, w in PHOTOMETRIC_BANDS.items():
            if b.lower() == key.lower():
                return w, f"{b}-band ({w:.0f} nm)"
        raise SystemExit(f"ERROR: unknown --band '{args.band}'. "
                         f"Choices: {', '.join(PHOTOMETRIC_BANDS)}")
    return LAMBDA_K_NM, f"K-band ({LAMBDA_K_NM:.0f} nm)"   # default


def resolve_tomography(args):
    """Tomography default depends on the telescope unless the user forced it.
       K2 -> OFF by default, K1 -> ON by default."""
    if args.tomography is None:
        return (args.telescope == "K1")
    return args.tomography


def resolve_tt_sensor(args):
    """Normalize --tt-sensor onto args: sets args._tt_sensor_base ('strap' or
    'trick') and args._tt_wfs_band ('R'/'H'/'K'). For TRICK on K1 the dichroic
    ties the science band to the OTHER of H/K, so unless --wavelength pins the
    nm, the science band is swapped to the complement. STRAP leaves the science
    band untouched (default path -> harness unchanged)."""
    sensor = getattr(args, "tt_sensor", "strap")
    if sensor == "trick-h":
        args._tt_sensor_base, args._tt_wfs_band, sci = "trick", "H", "K"
    elif sensor == "trick-k":
        args._tt_sensor_base, args._tt_wfs_band, sci = "trick", "K", "H"
    else:                              # strap (refined) or strap-legacy
        args._tt_sensor_base, args._tt_wfs_band = sensor, "R"
        return
    if getattr(args, "wavelength", None) in (None, ""):
        args.band = sci                 # dichroic: science gets the other band


def default_output_name(ut_date_stamp, telescope):
    """Auto output filename from the data's UT date stamp + telescope,
       e.g. ('20260525','K1') -> 'ao_strehl_20260525_K1.png'."""
    return f"ao_strehl_{ut_date_stamp}_{telescope}.png"


def parse_windows(window_strs, night_date):
    """Turn ['HH:MM-HH:MM', ...] into [(start_dt, end_dt), ...].

    A clock time whose hour is < 12 is interpreted as the morning AFTER the
    evening date (i.e. next calendar day); 12:00-23:59 stay on the evening date.
    This matches how an observing night spans local midnight."""
    next_day = night_date + timedelta(days=1)

    def to_dt(hhmm):
        try:
            h, m = (int(x) for x in hhmm.split(":"))
        except ValueError:
            raise SystemExit(f"ERROR: bad time '{hhmm}', expected HH:MM") from None
        base = next_day if h < 12 else night_date
        return base.replace(hour=h, minute=m)

    out = []
    for w in window_strs:
        if "-" not in w:
            raise SystemExit(f"ERROR: bad --window '{w}', expected HH:MM-HH:MM")
        a, b = w.split("-", 1)
        out.append((to_dt(a), to_dt(b)))
    return out
