"""Find the newest science frame in a directory, and tonight's NIRC2 or
OSIRIS directory.

The summit IDL Strehl tool (strehl_widget.pro) has an "autoimage" mode:
a 1 s widget timer runs ``file_search(path + '/n????.fits')``, takes the
last name, and measures it whenever it changes. It gets PATH from KTL
(``runname`` for NIRC2, ``show -s osiris ioutdir`` for OSIRIS); those
keyword libraries are not installed off the summit hosts (``show -s nirc2
outdir`` fails on vm-aodev), so tonight's directory is found from the
filesystem instead.

Frame naming, one pattern per instrument:
  * NIRC2: ``n0001.fits`` .. ``n9999.fits``. Written as ``n0017.fits.
    writing`` and renamed when complete; the ``n_unp_NNNN.fits``
    unprocessed copies (up to 134 MB) never match.
  * OSIRIS imager: ``i<YYMMDD>_a<set:3><frame:3>.fits``
    (``i260731_a001002.fits``). Spectrograph ``s...`` frames never match.
Name order is acquisition order for both, so "newest" is the last name.
Both instruments can be in use on the same night, so the caller says which
one it is looking for (`kind`).

NIRC2 nights live at ``/s/sdata9NN/<account>/<YYYYmonDD>[_X]/`` and move
between disks (2023 nights on sdata902/903, 2026 on sdata907). ``/s`` is
an autofs map: only disks already touched show up in a listing, so the
disks are named explicitly rather than globbed; ones that do not mount
here (sdata904-906 on 2026-09-22) fail instantly and are skipped.

OSIRIS data (``/s/sdata1100/<osiris1..20|osrseng>/<YYMMDD>/IMAG/raw/``)
does not mount on vm-aodev, but the AO server k2aoserver-new mounts it and
the read-only rsync key (ssh alias ``k2ao``) can list and copy it. A PATH
of the form ``host:/abs/dir`` is REMOTE: listed with ``rsync --list-only``
and each new frame copied into a local cache (remote_cache_dir) before it
is measured. Remote polling is load on an operations server, so the GUI
asks before starting it and flags it for as long as it runs (Eduardo
2026-09-22).

No Qt here -- the GUI calls this from worker threads (the /s disks are
NFS hard mounts and rsync goes over ssh: a stall must not freeze the
GUI; every rsync call carries a timeout).
"""
import datetime as dt
import glob
import os
import re
import subprocess

__all__ = ["FRAME_PATTERNS", "NIRC2_DATA_DISKS", "FITS_BLOCK",
           "OSIRIS_REMOTE_HOST", "OSIRIS_REMOTE_DISK", "REMOTE_POLL_S",
           "is_frame_name", "nirc2_frame_number", "newest_frame",
           "frame_ready", "night_dir_name", "find_latest_night_dir",
           "split_remote", "parse_rsync_listing", "remote_newest_frame",
           "remote_cache_dir", "remote_fetch", "prune_cache",
           "find_latest_osiris_night"]

FRAME_PATTERNS = {
    "nirc2": re.compile(r"^n(\d{4})\.fits$"),
    "osiris": re.compile(r"^i\d{6}_a\d{6}\.fits$"),
}
NIRC2_DATA_DISKS = tuple(f"/s/sdata{n}" for n in range(900, 908))
FITS_BLOCK = 2880       # a complete FITS file is a whole number of blocks
_NIGHT_RE = re.compile(r"^(\d{4})([a-z]{3})(\d{2})(?:_\w+)?$")
_OSIRIS_NIGHT_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})$")
# host alias + absolute path; deliberately narrow (no spaces, no shell
# metacharacters) since both go on an rsync command line
_REMOTE_RE = re.compile(r"^([A-Za-z0-9_.-]+):(/[A-Za-z0-9_./-]*)$")
OSIRIS_REMOTE_HOST = "k2ao"             # read-only rsync key, vm-aodev only
OSIRIS_REMOTE_DISK = "/s/sdata1100"
REMOTE_POLL_S = 3.0     # one listing is ~0.6 s; 1 s polling is for local
_RSYNC_SSH = "ssh -o BatchMode=yes -o ConnectTimeout=10"
_REMOTE_KEEP = 20       # cached frames kept per remote directory
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun",
           "jul", "aug", "sep", "oct", "nov", "dec")


def is_frame_name(name, kind=None):
    """`name` is a frame of instrument `kind` ("nirc2" / "osiris"), or of
    either when kind is None."""
    if kind is not None:
        return FRAME_PATTERNS[kind].match(name) is not None
    return any(p.match(name) for p in FRAME_PATTERNS.values())


def nirc2_frame_number(name):
    """The frame number of a summit-numbered NIRC2 name, else None."""
    m = FRAME_PATTERNS["nirc2"].match(name)
    return int(m.group(1)) if m else None


def newest_frame(dirpath, kind=None):
    """The last frame (by name) of instrument `kind` -- NIRC2 or OSIRIS
    imager, either when None -- in `dirpath` as (name, path, size_bytes),
    or None when there is none or the directory can't be read."""
    best = None
    try:
        with os.scandir(dirpath) as it:
            for entry in it:
                if is_frame_name(entry.name, kind) and (
                        best is None or entry.name > best.name):
                    best = entry
        if best is None:
            return None
        return best.name, best.path, os.stat(best.path).st_size
    except OSError:
        return None


def frame_ready(size, previous_size):
    """A frame is safe to read once its size is a whole, non-zero number
    of FITS blocks AND did not change since the previous poll -- a frame
    still being written over NFS fails one or the other."""
    return (size > 0 and size % FITS_BLOCK == 0
            and previous_size is not None and size == previous_size)


def night_dir_name(date):
    """NIRC2's directory name for a UT date: 2026sep21."""
    return f"{date.year:04d}{_MONTHS[date.month - 1]}{date.day:02d}"


def _nirc2_night_date(name):
    m = _NIGHT_RE.match(name)
    if m is None or m.group(2) not in _MONTHS:
        return None
    try:
        return dt.date(int(m.group(1)), _MONTHS.index(m.group(2)) + 1,
                       int(m.group(3)))
    except ValueError:
        return None


def _osiris_night_date(name):
    m = _OSIRIS_NIGHT_RE.match(name)
    if m is None:
        return None
    try:
        return dt.date(2000 + int(m.group(1)), int(m.group(2)),
                       int(m.group(3)))
    except ValueError:
        return None


def _utc_today(utc_now):
    return (utc_now or dt.datetime.now(dt.timezone.utc)).date()


def find_latest_night_dir(disks=NIRC2_DATA_DISKS, utc_now=None):
    """The NIRC2 directory frames are being written to tonight: the
    latest-dated night directory (``2026sep21``, ``2026sep02_B``) of any
    account on any disk -- dates after tomorrow's UT are ignored as
    mislabelled -- and, between accounts sharing that date, the one
    written to last (its newest frame, else the directory itself, so an
    empty directory made for tonight still beats last night's). Returns
    (path, is_tonight): is_tonight means dated today's or tomorrow's UT
    (directories can be made before the UT date turns). (None, False)
    when no disk is reachable."""
    today = _utc_today(utc_now)
    horizon = today + dt.timedelta(days=1)
    nights = []
    for disk in disks:
        # glob INSIDE the disk: naming the disk is what makes autofs mount it
        for account in sorted(glob.glob(os.path.join(disk, "nirc*"))):
            try:
                with os.scandir(account) as it:
                    for entry in it:
                        d = _nirc2_night_date(entry.name)
                        if d is not None and d <= horizon and entry.is_dir():
                            nights.append((d, entry.path))
            except OSError:
                continue
    if not nights:
        return None, False
    latest = max(d for d, _ in nights)

    def _last_write(p):
        nf = newest_frame(p, "nirc2")
        try:
            return os.stat(nf[1] if nf else p).st_mtime
        except OSError:
            return -1.0
    best = max((p for d, p in nights if d == latest), key=_last_write)
    return best, latest >= today


# ---- remote (rsync) ---------------------------------------------------------
def split_remote(path):
    """(host, abs_dir) for a ``host:/abs/dir`` PATH, else None."""
    m = _REMOTE_RE.match((path or "").strip())
    if m is None:
        return None
    return m.group(1), (m.group(2).rstrip("/") or "/")


def parse_rsync_listing(text):
    """``rsync --list-only`` lines -> [(is_dir, name, size, mtime)];
    mtime is the server's local time (comparisons only). Lines that are
    not listing entries (warnings, errors) are skipped."""
    out = []
    for line in text.splitlines():
        parts = line.split(None, 4)
        if len(parts) != 5:
            continue
        perms, size, day, clock, name = parts
        try:
            size = int(size.replace(",", ""))
            mtime = dt.datetime.strptime(f"{day} {clock}",
                                         "%Y/%m/%d %H:%M:%S")
        except ValueError:
            continue
        out.append((perms.startswith("d"), name, size, mtime))
    return out


def _rsync(args, timeout):
    cmd = ["rsync", "-e", _RSYNC_SSH] + list(args)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        raise OSError(f"rsync timed out after {timeout:.0f} s") from None
    except FileNotFoundError:
        raise OSError("rsync is not installed") from None
    # 23 = partial transfer: permission-denied subdirectories in a
    # recursive listing (osiriseng, osiris) -- the rest is still valid
    if r.returncode not in (0, 23):
        err = (r.stderr or "").strip().splitlines()
        raise OSError(f"rsync exit {r.returncode}: "
                      f"{err[-1] if err else 'no message'}")
    return r.stdout


def _rsync_list(target, filters=(), recursive=False, timeout=30.0):
    args = ["--list-only"] + (["-r"] if recursive else [])
    args += [f"--{kind}={pattern}" for kind, pattern in filters]
    return parse_rsync_listing(_rsync(args + [target], timeout))


def remote_newest_frame(host, rdir, kind=None, timeout=30.0):
    """newest_frame() for a remote directory: (name, remote_path, size)
    or None when it holds no frame. Raises OSError when the listing
    fails (unreachable host, bad path, timeout)."""
    best = None
    for is_dir, name, size, _ in _rsync_list(f"{host}:{rdir}/",
                                              timeout=timeout):
        if not is_dir and is_frame_name(name, kind) and (
                best is None or name > best[0]):
            best = (name, size)
    if best is None:
        return None
    return best[0], f"{rdir}/{best[0]}", best[1]


def remote_cache_dir(host, rdir):
    """Local directory remote frames of `host:rdir` are copied into
    ($KECK_AO_REMOTE_CACHE, else ~/.cache/keck-ao-estimator/
    remote_frames)."""
    root = os.environ.get("KECK_AO_REMOTE_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "keck-ao-estimator",
        "remote_frames")
    return os.path.join(root, host, rdir.strip("/").replace("/", "_"))


def remote_fetch(host, rpath, local_dir, timeout=180.0):
    """Copy one remote frame into `local_dir` (rsync writes a hidden
    temporary and renames, so a partial copy never looks like a frame).
    Returns the local path; raises OSError on failure."""
    os.makedirs(local_dir, exist_ok=True)
    _rsync(["-t", f"{host}:{rpath}", local_dir + os.sep], timeout)
    local = os.path.join(local_dir, os.path.basename(rpath))
    if not os.path.isfile(local):
        raise OSError(f"rsync reported success but {local} is missing")
    return local


def prune_cache(local_dir, keep=_REMOTE_KEEP):
    """Keep only the `keep` newest-by-name frames in a remote cache
    directory (OSIRIS frames are 16.8 MB; a night would fill a disk)."""
    try:
        names = sorted(n for n in os.listdir(local_dir) if is_frame_name(n))
    except OSError:
        return
    for name in names[:-keep] if keep > 0 else names:
        try:
            os.remove(os.path.join(local_dir, name))
        except OSError:
            pass


def find_latest_osiris_night(host=OSIRIS_REMOTE_HOST,
                             disk=OSIRIS_REMOTE_DISK, utc_now=None,
                             timeout=90.0):
    """Tonight's OSIRIS imager directory, over rsync: the latest-dated
    ``<account>/<YYMMDD>/`` night on `disk` (dates after tomorrow's UT
    ignored), and between accounts sharing it the one whose IMAG/raw was
    written last. One or two rsync listings (2-20 s seen). Returns
    (``host:disk/<account>/<YYMMDD>/IMAG/raw``, is_tonight) or
    (None, False); raises OSError when the host can't be listed."""
    today = _utc_today(utc_now)
    horizon = today + dt.timedelta(days=1)
    account_filters = [("include", "/osiris*/"), ("include", "/osrseng/")]
    nights = []
    for is_dir, name, _, _ in _rsync_list(
            f"{host}:{disk}/", account_filters
            + [("include", "/*/[0-9][0-9][0-9][0-9][0-9][0-9]/"),
               ("exclude", "*")], recursive=True, timeout=timeout):
        parts = name.split("/")
        if is_dir and len(parts) == 2:
            d = _osiris_night_date(parts[1])
            if d is not None and d <= horizon:
                nights.append((d, parts[0], parts[1]))
    if not nights:
        return None, False
    latest = max(n[0] for n in nights)
    night = latest.strftime("%y%m%d")
    accounts = sorted(a for d, a, _ in nights if d == latest)
    written = {a: dt.datetime.min for a in accounts}
    if len(accounts) > 1:
        for _, name, _, mtime in _rsync_list(
                f"{host}:{disk}/", account_filters + [
                    ("include", f"/*/{night}/"),
                    ("include", f"/*/{night}/IMAG/"),
                    ("include", f"/*/{night}/IMAG/raw/"),
                    ("include", f"/*/{night}/IMAG/raw/*.fits"),
                    ("exclude", "*")], recursive=True, timeout=timeout):
            account = name.split("/", 1)[0]
            if account in written and "/IMAG/raw" in name:
                written[account] = max(written[account], mtime)
    best = max(accounts, key=lambda a: written[a])
    return f"{host}:{disk}/{best}/{night}/IMAG/raw", latest >= today
