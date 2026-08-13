"""Light/dark theme switching for the GUI (View menu "Dark theme", also
auto-enabled by Nighttime mode -- see gui/tabs/nighttime.py).

WIDGETS-ONLY by design (2026-07-21 decision): the Qt controls restyle; the
on-screen matplotlib figures and everything Export writes keep their light,
print-style rendering (the CLI's outputs are byte/pixel-frozen by the
regression harness and exported PNGs go into reports).

Both themes run on Qt's Fusion style (set once, on the first apply): Fusion
is the one style that honours a custom QPalette everywhere, and using it for
light too keeps the app identical across platforms AND avoids restoring a
style by name -- app.style().objectName() is empty on some Qt builds, so a
save/restore-by-name round-trip silently fails. Light restores the palette
the app started with (the platform default colors, captured on first use).

Semantic label cues
-------------------
Status/readout labels used to hard-code light-theme colors inline
(``color:#555`` etc.), which go near-invisible on a dark background. They now
declare WHAT they are via a dynamic property -- ``set_cue(label, "secondary"
| "ok" | "err")`` -- and an application-level stylesheet, swapped together
with the palette, decides the actual color per theme. A widget-level
stylesheet (e.g. ``font-size:11px``) still merges cleanly with the app-level
cue color, so size-only tweaks stay inline.
"""
from qtcompat import QtGui

# captured on the first apply_theme() call: the platform-default style name
# and palette to restore when switching back to light
_ORIG = {}

_CUE_QSS_LIGHT = """
QLabel[cue="secondary"] { color: #555555; }
QLabel[cue="ok"]        { color: #1B6B3A; }
QLabel[cue="warn"]      { color: #B8860B; }
QLabel[cue="err"]       { color: #C0392B; }
QLabel[cue="info"]      { color: #2A4D7C; }
"""

_CUE_QSS_DARK = """
QLabel[cue="secondary"] { color: #9da3aa; }
QLabel[cue="ok"]        { color: #63c283; }
QLabel[cue="warn"]      { color: #d4a437; }
QLabel[cue="err"]       { color: #e08276; }
QLabel[cue="info"]      { color: #8ab6e8; }
"""


def set_cue(label, cue):
    """Tag a label with a semantic color cue ('secondary' / 'ok' /
    'warn' / 'err' / 'info' -- 'info' being a derived/looked-up result
    rather than a status, e.g. the SR tab's TT-star identification);
    the active theme's app-level stylesheet supplies the actual color.
    Re-polishes so a cue change on a live widget takes effect immediately."""
    label.setProperty("cue", cue)
    st = label.style()
    st.unpolish(label)
    st.polish(label)


def dark_palette():
    """The dark QPalette (standard Fusion-dark recipe: near-black surfaces,
    light text, blue highlight, greyed disabled roles)."""
    R, G, D = (QtGui.QPalette.ColorRole, QtGui.QPalette.ColorGroup,
               QtGui.QColor)
    p = QtGui.QPalette()
    win, base, text = D(53, 53, 53), D(42, 42, 42), D(228, 228, 228)
    hi = D(42, 130, 218)
    p.setColor(R.Window, win)
    p.setColor(R.WindowText, text)
    p.setColor(R.Base, base)
    p.setColor(R.AlternateBase, D(58, 58, 58))
    p.setColor(R.ToolTipBase, D(35, 35, 35))
    p.setColor(R.ToolTipText, text)
    p.setColor(R.Text, text)
    p.setColor(R.PlaceholderText, D(140, 140, 140))
    p.setColor(R.Button, win)
    p.setColor(R.ButtonText, text)
    p.setColor(R.BrightText, D(255, 96, 92))
    p.setColor(R.Link, hi)
    p.setColor(R.Highlight, hi)
    p.setColor(R.HighlightedText, D(20, 20, 20))
    for role in (R.WindowText, R.Text, R.ButtonText):
        p.setColor(G.Disabled, role, D(127, 127, 127))
    p.setColor(G.Disabled, R.Base, D(48, 48, 48))
    p.setColor(G.Disabled, R.Button, D(48, 48, 48))
    return p


def apply_theme(app, dark):
    """Switch the whole application between light (the captured startup
    palette) and dark (dark_palette), both on the Fusion style. Also installs
    the matching semantic-cue stylesheet; call once with dark=False at
    startup so the cue colors work before any theme switch."""
    if not _ORIG:
        _ORIG["palette"] = QtGui.QPalette(app.palette())
        app.setStyle("Fusion")
    if dark:
        app.setPalette(dark_palette())
        app.setStyleSheet(_CUE_QSS_DARK)
    else:
        app.setPalette(_ORIG["palette"])
        app.setStyleSheet(_CUE_QSS_LIGHT)


def is_dark(app):
    """True if the app currently wears the dark palette (windows darker than
    mid-grey) -- a check on actual state, not a stored flag."""
    c = app.palette().color(QtGui.QPalette.ColorRole.Window)
    return c.lightness() < 128


__all__ = ["apply_theme", "dark_palette", "is_dark", "set_cue"]
