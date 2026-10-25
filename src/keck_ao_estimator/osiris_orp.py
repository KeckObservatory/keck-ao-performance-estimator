"""The OSIRIS online reduction pipeline (ORP) as the Strehl tools see it: what
each cube had subtracted, and why a raw spectrograph frame has no cube.

Each ORP cube is (raw frame - another raw frame), rectified into a cube; no
flat, telluric or flux calibration. The reduction file (DRF) the pipeline
ran is copied into the cube's COMMENT cards (``DRFC...``) and kept in the
night's ``drf/`` directory, named after the raw frame that triggered it with
its state as the extension: ``NNNNNN.<frame>_ORP.done|failed|working|waiting``.
Its "Subtract Frame" module names the frame subtracted. On 2026-07-14
(osiris6) that was, by the observer's sky setup:

* the frame's dither partner (a004002 - a004003, the second frame flagged as
  sky in the log): an A-B pair, the star also NEGATIVE where it was in the
  partner (0.30-1.05" away on 2026-07-14 / 10-20, as deep as the star);
* a sky frame of the night (a005002, 900 s);
* a frame of another night (s260713_a002003): the sky is not removed.

Why a raw frame has no cube (11 nights surveyed 2026-10-25): it was the sky
of another frame's cube or a dark (by design; never reported here); the
pipeline was never given it (no DRF: whole nights, engineering blocks);
its DRF failed, or hung ('working': 2026-08-05 a033002), or waits in the
queue. A pair's first frame gets its cube only once its partner arrives, so
a frame counts as late only when the NEXT raw frame arrived ORP_GRACE_S ago
(the pipeline takes ~5-6 s per frame) and its cube is still not there.
"""
import datetime as dt
import os
import re

__all__ = ["ORP_GRACE_S", "ORP_STUCK_S", "subtracted_frame",
           "describe_subtraction", "subtraction_kind", "parse_drf",
           "night_paths", "missing_cubes", "group_missing",
           "read_night_state", "check_night"]

ORP_GRACE_S = 30.0      # the next raw frame arrived this long ago, no cube yet
ORP_STUCK_S = 60.0      # a DRF 'working' (or 'waiting') longer than this
_RAW_RE = re.compile(r"^(s(\d{6})_a(\d{3})\d{3})\.fits$")
_CUBE_RE = re.compile(r"^(s\d{6}_a\d{6})_[A-Za-z0-9]+_\d{3}\.fits$")
_DRF_RE = re.compile(r"^\d+\.(s\d{6}_a\d{6})_ORP\.(done|failed|working|waiting)$")
_SUB_RE = re.compile(r'Name="Subtract Frame"[^>]*?CalibrationFile="([^"]*)"')
_DATASET_RE = re.compile(r'<dataset[^>]*?Name="([^"]*)"')


def _stem(name):
    base = os.path.basename(str(name))
    return base[:-5] if base.endswith(".fits") else base


def subtracted_frame(header):
    """The raw frame the ORP subtracted to make this cube (its name without
    .fits), from the DRF copied into the cube's COMMENT cards; None when the
    header carries no DRF (not an ORP cube, or a module other than
    "Subtract Frame")."""
    xml = "".join(str(c)[4:] for c in header.get("COMMENT", [])
                  if str(c).startswith("DRFC"))
    m = _SUB_RE.search(xml)
    return _stem(m.group(1)) if m else None


def subtraction_kind(frame, subtracted):
    """'pair' (the frame's dither partner: same night and set), 'night' (a
    frame of the same night: a sky), 'other-night', or 'none'."""
    if not subtracted:
        return "none"
    a, b = _RAW_RE.match(f"{_stem(frame)}.fits"), _RAW_RE.match(f"{subtracted}.fits")
    if a is None or b is None:
        return "night"
    if a.group(2) != b.group(2):
        return "other-night"
    return "pair" if a.group(3) == b.group(3) else "night"


def describe_subtraction(frame, subtracted):
    """One log line: what the ORP subtracted to make the cube of `frame`."""
    frame = _stem(frame)
    kind = subtraction_kind(frame, subtracted)
    if kind == "none":
        return f"ORP: {frame}, nothing subtracted (no reduction record in the header)"
    if kind == "pair":
        return (f"ORP: {frame} - {subtracted}, its dither partner (A-B: the star is also NEGATIVE where it was "
                f"in {subtracted})")
    if kind == "other-night":
        return f"ORP: {frame} - {subtracted}, a frame of ANOTHER night (the sky is not removed)"
    return f"ORP: {frame} - {subtracted}, a frame of this night (its sky)"


def parse_drf(text):
    """(dataset, subtracted) of a DRF's XML: the raw frame the cube is made
    of and the one subtracted (names without .fits; None if absent)."""
    d = _DATASET_RE.search(text)
    s = _SUB_RE.search(text)
    return (_stem(d.group(1)) if d else None, _stem(s.group(1)) if s else None)


def night_paths(orp_dir):
    """(raw_dir, drf_dir, night_log) of a ``.../<night>/SPEC/ORP`` directory
    (a local path or a remote directory), else None."""
    p = str(orp_dir).rstrip("/")
    if not p.endswith("/SPEC/ORP"):
        return None
    night = p[:-len("/SPEC/ORP")]
    raw = f"{night}/SPEC/raw"
    return raw, f"{night}/drf", f"{raw}/osiris_spec_log"


def missing_cubes(raw, cubes, drfs, now, darks=(), grace_s=ORP_GRACE_S,
                  stuck_s=ORP_STUCK_S):
    """Raw frames that should have a cube by now and do not, with the reason:
    [(frame, reason)] in frame order.

    `raw`: {frame: arrival time}; `cubes`: the frames with a cube; `drfs`:
    {trigger frame: (state, time, dataset, subtracted)} from drf/; `darks`:
    frames to skip; times are datetimes of one clock with `now`. Frames used
    as another cube's sky, darks, frames with a cube, the newest frame and
    frames whose next frame arrived less than `grace_s` ago are not
    reported."""
    skies = {v[3] for v in drfs.values() if v[3]}
    datasets = {v[2] for v in drfs.values() if v[2]}
    order = sorted(raw, key=lambda f: (raw[f], f))
    out = []
    for i, frame in enumerate(order):
        if frame in cubes or frame in skies or frame in darks:
            continue
        if i + 1 >= len(order):
            continue                    # the newest: its partner, or the ORP, may still come
        if (now - raw[order[i + 1]]).total_seconds() < grace_s:
            continue
        d = drfs.get(frame)
        if d is None and frame in datasets:
            # the cube of `frame` is made by its partner's DRF: use that one
            d = next(v for v in drfs.values() if v[2] == frame)
        if d is None:
            out.append((frame, "the pipeline was not given it (no ORP reduction file)"))
            continue
        state, when = d[0], d[1]
        age = (now - when).total_seconds()
        if state == "failed":
            out.append((frame, "its ORP reduction FAILED (drf/...failed)"))
        elif state == "working" and age > stuck_s:
            out.append((frame, f"the ORP has been reducing it since {when:%H:%M:%S} (stuck?)"))
        elif state == "waiting" and age > stuck_s:
            out.append((frame, f"queued in the ORP since {when:%H:%M:%S} (the pipeline is behind or stuck)"))
        elif state == "done":
            out.append((frame, "the ORP reduced it but no cube is in SPEC/ORP"))
    return out


def group_missing(missing):
    """Log lines for `missing`: consecutive frames with one reason in one
    line ("s261020_a042002 .. s261020_a046005 (20 frames): reason")."""
    lines, run = [], []
    for frame, reason in missing + [(None, None)]:
        if run and (reason != run[-1][1]):
            first, last = run[0][0], run[-1][0]
            span = first if len(run) == 1 else f"{first} .. {last} ({len(run)} frames)"
            lines.append(f"{span}: {run[0][1]}")
            run = []
        if frame is not None:
            run.append((frame, reason))
    return lines


def _darks(log_text):
    """Frames the night log lists with the dark filter (Drk)."""
    out = set()
    for line in log_text.splitlines():
        parts = line.split()
        if parts and _RAW_RE.match(f"{parts[0]}.fits") and re.search(r"\bDrk\b", line):
            out.add(parts[0])
    return out


def read_night_state(orp_dir, remote=None, cache_dir=None, drf_cache=None):
    """The inputs of missing_cubes for the night of `orp_dir`: (raw, cubes,
    drfs, darks). `remote` = host when `orp_dir` is a directory on it (rsync
    listings; DRFs and the night log copied into `cache_dir`); `drf_cache`
    {file name: (dataset, subtracted)} keeps parsed DRFs between calls (a
    done DRF does not change). Raises OSError when a listing fails."""
    from . import frame_watch
    paths = night_paths(orp_dir)
    if paths is None:
        raise ValueError(f"{orp_dir} is not a night's SPEC/ORP directory")
    raw_dir, drf_dir, log_path = paths
    drf_cache = {} if drf_cache is None else drf_cache

    def listing(d):
        if remote is None:
            out = []
            try:
                with os.scandir(d) as it:
                    for e in it:
                        if e.is_file():
                            out.append((e.name, dt.datetime.fromtimestamp(e.stat().st_mtime)))
            except FileNotFoundError:
                return []
            return out
        return [(name, mtime) for is_dir, name, _size, mtime
                in frame_watch._rsync_list(f"{remote}:{d}/") if not is_dir]

    raw = {}
    for name, mtime in listing(raw_dir):
        m = _RAW_RE.match(name)
        if m:
            raw[m.group(1)] = mtime
    cubes = set()
    for name, _mtime in listing(orp_dir):
        m = _CUBE_RE.match(name)
        if m:
            cubes.add(m.group(1))
    drfs, todo = {}, []
    for name, mtime in listing(drf_dir):
        m = _DRF_RE.match(name)
        if not m:
            continue
        trigger, state = m.group(1), m.group(2)
        drfs[trigger] = (state, mtime, None, None)
        if name not in drf_cache:
            todo.append(name)
    if todo and remote is not None:
        os.makedirs(cache_dir, exist_ok=True)
        frame_watch._rsync(["-tq"] + [f"{remote}:{drf_dir}/{n}" for n in todo] + [cache_dir + os.sep], 60.0)
    for name in todo:
        path = os.path.join(cache_dir, name) if remote is not None else f"{drf_dir}/{name}"
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                parsed = parse_drf(fh.read())
        except OSError:
            continue
        trigger = _DRF_RE.match(name).group(1)
        drfs[trigger] = drfs[trigger][:2] + parsed
        if name.endswith((".done", ".failed")):
            drf_cache[name] = parsed     # final states: never read again
    for name, parsed in drf_cache.items():
        m = _DRF_RE.match(name)
        if m and m.group(1) in drfs:
            drfs[m.group(1)] = drfs[m.group(1)][:2] + parsed
    darks = set()
    try:
        if remote is None:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                darks = _darks(fh.read())
        else:
            local = frame_watch.remote_fetch(remote, log_path, cache_dir, timeout=30.0)
            with open(local, encoding="utf-8", errors="replace") as fh:
                darks = _darks(fh.read())
    except OSError:
        pass
    return raw, cubes, drfs, darks


def check_night(orp_dir, remote=None, cache_dir=None, drf_cache=None,
                now=None, reported=()):
    """missing_cubes for the night of `orp_dir` now, without the frames in
    `reported`: [(frame, reason)]."""
    raw, cubes, drfs, darks = read_night_state(orp_dir, remote=remote, cache_dir=cache_dir,
                                               drf_cache=drf_cache)
    now = now or dt.datetime.now()
    return [(f, r) for f, r in missing_cubes(raw, cubes, drfs, now, darks=darks) if f not in set(reported)]
