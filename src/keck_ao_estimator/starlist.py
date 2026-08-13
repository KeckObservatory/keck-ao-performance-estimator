"""Keck AO starlist parsing: read a Keck-format starlist file (the format
observers actually hand the OAs -- https://www2.keck.hawaii.edu/inst/ao/
starlist.html) into plain dicts, for the Target tab's "Load starlist" picker.

Qt-free, like catalogs.py/target_resolve.py. The format, as ground-truthed
against real hand-edited K2 LGS starlists rather than only the (self-
contradictory in places) web page -- examples/synthetic_k1lgs.lst is a
synthetic replica of those lists' quirks and is the bundled fixture:

  columns 1-16   target name (fixed width; may itself contain spaces,
                 e.g. "BD+99  1234" or "Syn Cluster A", so the line can NOT
                 be naively token-split from the start)
  then, whitespace-separated:
    RA   as three fields   HH MM SS.ss
    Dec  as three fields  sDD MM SS.s   (sign optional on positives)
    equinox                2000.0 / bare 2000 / APP
    any number of key=value fields (vmag=, rmag=, lgs=, pa=, sep=,
                 vistt=, irtt=, target=, comments=, bare J= H= K=, ...)
    an optional trailing "# ..." comment which itself often carries MORE
                 key=value fields (e.g. "# rmag=9.68 kmag=8.62") -- parsed,
                 not discarded.

A row with a target=<name> field is a tip-tilt-star candidate explicitly
linked to the science-target row of that name; every other row is a target
(or a plain TT-star candidate) in its own right.
"""
import os
import re

# HH MM SS.ss  sDD MM SS.s -- used only as a fallback locator for lines whose
# name field is wider than the standard 16 columns (sloppy hand-edited lists)
_RADEC_RE = re.compile(
    r"\s(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}(?:\.\d*)?)"
    r"\s+([+-]?\d{1,3})\s+(\d{1,2})\s+(\d{1,2}(?:\.\d*)?)\s")

NAME_WIDTH = 16


def _parse_kv(tokens, keys, notes):
    """Split key=value tokens into `keys`; anything else joins `notes`."""
    for tok in tokens:
        if "=" in tok:
            k, _, v = tok.partition("=")
            if k:
                keys[k] = v
                continue
        notes.append(tok)


def _entry_from_line(line, lineno):
    """One starlist line -> entry dict. Raises ValueError on a malformed
    line (caller collects these as skipped lines, it does not abort)."""
    name = line[:NAME_WIDTH].strip()
    rest = line[NAME_WIDTH:]
    tokens = rest.split()
    if len(tokens) < 7 or not re.fullmatch(r"\d{1,2}", tokens[0]):
        # non-standard name width: locate the RA/Dec pattern anywhere in the
        # line and take everything before it as the name
        m = _RADEC_RE.search(line)
        if not m:
            raise ValueError("no 'HH MM SS sDD MM SS equinox' found")
        name = line[:m.start()].strip()
        tokens = line[m.start():].split()
        if len(tokens) < 7:
            raise ValueError("truncated after RA/Dec")
    if not name:
        raise ValueError("empty name field")
    rh, rm, rs, dd, dm, ds, equinox = tokens[:7]

    # trailing "# ..." comment: everything from the first '#' TOKEN on is
    # comment, but its key=value fields (rmag=, kmag=, ...) still count
    kv_tokens = tokens[7:]
    for i, tok in enumerate(kv_tokens):
        if tok.startswith("#"):
            kv_tokens = kv_tokens[:i] + [kv_tokens[i][1:]] + kv_tokens[i + 1:]
            break

    try:
        h, mi, s = int(rh), int(rm), float(rs)
        # keep the Dec DEGREES as its original string: float() would lose the
        # sign of "-00 16 44" (negative zero degrees)
        ddeg, dmi, dsec = int(dd), int(dm), float(ds)
    except ValueError:
        raise ValueError("non-numeric RA/Dec field") from None
    if not (0 <= h < 24 and 0 <= mi < 60 and 0 <= s < 60):
        raise ValueError(f"RA {rh} {rm} {rs} out of range")
    if not (abs(ddeg) <= 90 and 0 <= dmi < 60 and 0 <= dsec < 60):
        raise ValueError(f"Dec {dd} {dm} {ds} out of range")

    keys, notes = {}, []
    _parse_kv([t for t in kv_tokens if t], keys, notes)

    sign = -1.0 if dd.lstrip().startswith("-") else 1.0
    ra_deg = (h + mi / 60.0 + s / 3600.0) * 15.0
    dec_deg = sign * (abs(ddeg) + dmi / 60.0 + dsec / 3600.0)
    dec_str = dd if dd[0] in "+-" else "+" + dd
    return dict(
        name=name,
        ra=f"{rh}:{rm}:{rs}",            # engine.parse_radec-ready strings,
        dec=f"{dec_str}:{dm}:{ds}",      # built from the ORIGINAL tokens
        ra_deg=ra_deg, dec_deg=dec_deg,
        equinox=equinox,
        keys=keys,
        notes=" ".join(notes),
        lgs=keys.get("lgs", "0").strip() == "1",
        target=keys.get("target") or None,   # TT star linked to this target
        lineno=lineno,
    )


def parse_starlist_text(text):
    """Parse starlist `text`. Returns (entries, skipped): `entries` is a list
    of dicts (see _entry_from_line; order preserved), `skipped` a list of
    (lineno, line, reason) for malformed non-blank/non-comment lines --
    starlists are hand-edited files, one bad line must not lose the rest."""
    entries, skipped = [], []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            entries.append(_entry_from_line(line, lineno))
        except ValueError as e:
            skipped.append((lineno, line, str(e)))
    return entries, skipped


def parse_starlist(path):
    """parse_starlist_text() on the contents of `path`."""
    with open(os.fspath(path), encoding="utf-8", errors="replace") as fh:
        return parse_starlist_text(fh.read())


def same_star_name(a, b):
    """True if two starlist names refer to the same star, honouring the
    format's linking convention: key=value fields are whitespace-delimited
    tokens, so a target= value CANNOT contain a space -- a list writes
    'target=Syn_Cluster_A' to link to the name field 'Syn Cluster A'
    (underscores standing in for spaces). Ground truth: a real hand-edited
    list where an exact-match comparison silently missed the link. Also
    case-insensitive and run-collapsing ('BD+99  1234' == 'bd+99_1234'),
    since the 16-column name field pads with space runs a hand-written
    link would never reproduce."""
    def norm(s):
        return " ".join((s or "").replace("_", " ").lower().split())
    return norm(a) == norm(b) and norm(a) != ""


def format_starlist_line(entry):
    """Format an entry dict (same shape parse_starlist_text produces, or a
    freshly-built one with the same keys) back into one Keck-format
    starlist line -- so a session-added target round-trips through
    parse_starlist_text exactly like a hand-written one. `entry["ra"]` must
    already be "HH:MM:SS.ss" and `entry["dec"]` "+DD:MM:SS.s" (see
    _entry_from_line / the GUI's RA/Dec-string conversion for a freshly
    typed target)."""
    name = (entry["name"] or "").ljust(NAME_WIDTH)[:max(NAME_WIDTH, len(entry["name"] or ""))]
    rh, rm, rs = entry["ra"].split(":")
    dd, dm, ds = entry["dec"].split(":")
    equinox = str(entry.get("equinox") or "2000.0")
    kv = " ".join(f"{k}={v}" for k, v in (entry.get("keys") or {}).items()
                 if v not in (None, ""))
    fields = [rh, rm, rs, dd, dm, ds, equinox]
    if kv:
        fields.append(kv)
    return name + " ".join(fields)


def write_starlist(path, entries):
    """Write `entries` (same dict shape parse_starlist_text produces) to
    `path` in Keck starlist format, one per line -- used for the session
    starlist-additions sidecar, so it round-trips through parse_starlist on
    the next load exactly like any other starlist file."""
    with open(os.fspath(path), "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(format_starlist_line(e) + "\n")


def entry_float(entry, *keys):
    """First parseable float among entry key=value fields `keys` (tried in
    order, case-sensitive: Keck lists use both 'kmag=' and bare 'K='), else
    None. For display -- values in a hand-edited list aren't always numeric."""
    for k in keys:
        v = entry["keys"].get(k)
        if v is None:
            continue
        try:
            return float(v)
        except ValueError:
            continue
    return None
