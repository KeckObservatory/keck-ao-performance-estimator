"""Append-only session files for the measured-SR tools: a CSV journal and a
text log per UT night.

The GUIs keep the image log in memory and export it on demand, which loses
a night's measurements when the tool is closed (or dies) without an export
(Eduardo 2026-10-20, the deployed PyAO tool). So every logged measurement
is also appended, as it happens, to ``<log_dir>/sr_tool_<YYYYMMDD>.csv``
-- the same columns and rounding as Export CSV, the header written once
when the file is new -- and the text log is appended to
``<log_dir>/sr_tool_<YYYYMMDD>.log`` when the window closes. Nights are
named by UT date, as Keck names them (a Hawaii night never crosses one).

The journal is append-only: rows deleted or overwritten in the GUI's table
stay in the file. Export CSV still writes the table as shown.

No Qt here; the callers report OSError in their own log.
"""
import csv
import datetime as dt
import os

__all__ = ["FILE_STEM", "night_tag", "session_paths", "resolve_log_dir",
           "csv_cell", "append_csv_row", "append_text"]

FILE_STEM = "sr_tool"
_DECIMALS = 3           # Export CSV's rounding, kept numeric


def night_tag(utc_now=None):
    """``YYYYMMDD`` of the current UT date (or of `utc_now`)."""
    now = utc_now or dt.datetime.now(dt.timezone.utc)
    return now.strftime("%Y%m%d")


def session_paths(log_dir, utc_now=None):
    """(csv_path, log_path) of tonight's journal and text log in `log_dir`."""
    stem = os.path.join(log_dir, f"{FILE_STEM}_{night_tag(utc_now)}")
    return stem + ".csv", stem + ".log"


def resolve_log_dir(env_var, default):
    """The session-log directory: ``$env_var`` when set and non-empty, else
    `default` (``~`` expanded). Nothing is created here -- the append
    functions make the directory on first use."""
    return os.path.expanduser(os.environ.get(env_var) or default)


def csv_cell(value, decimals=_DECIMALS):
    """A CSV cell for `value`: '' for None, numbers rounded to `decimals`
    (kept numeric, bools excepted), everything else unchanged."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    try:
        import numpy as np
        number = isinstance(value, (int, float, np.integer, np.floating))
    except ImportError:         # pragma: no cover - numpy is a dependency
        number = isinstance(value, (int, float))
    if number:
        return round(float(value), decimals)
    return value


def append_csv_row(path, columns, row, decimals=_DECIMALS):
    """Append one measurement `row` (a dict) to the CSV journal `path`,
    writing the column labels first when the file is new or empty.
    `columns` is the GUI's ``[(key, label), ...]``; values are formatted
    with csv_cell. Creates the directory. Raises OSError on failure."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow([label for _key, label in columns])
        w.writerow([csv_cell(row.get(key), decimals) for key, _label in columns])
    return path


def append_text(path, text, banner=None):
    """Append `text` to `path`, preceded by `banner` on its own line when
    given (a session marker), and followed by a blank line. Creates the
    directory. Raises OSError on failure."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        if banner:
            fh.write(banner.rstrip("\n") + "\n")
        fh.write(text.rstrip("\n") + "\n\n")
    return path
