#!/usr/bin/env python3
"""parallel PR-CP3 acceptance (Opus, 2026-09-13, blocker B1): an
invalid `$KECK_AO_WORKERS` must never stop the GUI from opening. A
typo in someone's shell profile is not a reason to refuse to start.

Reproduced by Opus on commit 2796394: `MainWindow()` raised `ValueError`
from `_build_nirc2_tab`'s `self.n2_workers.setValue(resolve_workers(None))`
(and the same eager-evaluation pattern in `mainwindow.py`'s
`_apply_config`, `n2.get("workers", resolve_workers(None))`, which
evaluates the default even when "workers" IS present in a saved
config) for `KECK_AO_WORKERS` = "12" (over the cap of 8), "0" (below
the floor of 1), and "abc" (unparseable).

Fix: `nirc2_strehl._resolve_workers_safe()` wraps `resolve_workers(None)`
in a `try`, falls back to `default_workers()` on `ValueError`, and
prints one `WARNING:` line naming the bad variable and the error
(same convention as `io.py`/`pipeline.py`'s own non-fatal recoveries)
-- never crashes, never silently swallows it. `mainwindow.py`'s
config-load default is also made LAZY (only resolved when "workers" is
actually missing from the saved config), so a saved config that
already names an explicit worker count never touches the environment
variable at all, valid or not.

Checks, for each of "12", "0", "abc" (and the unset control): a fresh
`MainWindow()` construction does not raise, the `Workers` spin box
ends up at `default_workers()` (the safe fallback) for every invalid
value, and exactly one `WARNING:` line naming the bad variable reaches
stdout (not silently swallowed). Also: `_apply_config` with a config
that DOES supply an explicit "workers" key is unaffected by an
invalid environment variable at all (the lazy-evaluation half of the
fix) -- no warning, the config's own value wins.

Fully offline; run headless (QT_QPA_PLATFORM=offscreen). Each
KECK_AO_WORKERS case runs in its OWN subprocess (not just a changed
env var in this process) since that is the actual failure mode Opus
reproduced -- a fresh process launched with a bad shell-profile
variable, not a value changed mid-session.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(os.path.dirname(ROOT), "src")

FAILURES = []

_PROBE = """
import sys, os
sys.path.insert(0, {src!r})
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from qtcompat import QtWidgets
app = QtWidgets.QApplication(sys.argv[:1])
import keck_ao_estimator.gui as gui
from keck_ao_estimator.parallel import default_workers
win = gui.MainWindow()
assert win.n2_workers.value() == default_workers(), (
    f"spin box at {{win.n2_workers.value()}}, expected the safe "
    f"fallback default_workers()={{default_workers()}}")
# the lazy half of the fix: a config that DOES supply "workers" must
# never touch the (possibly invalid) environment variable at all
win._apply_config({{"nirc2": {{"workers": 3}}}})
assert win.n2_workers.value() == 3, win.n2_workers.value()
print("PROBE_OK", win.n2_workers.value())
"""


def check(name, cond, detail=""):
    tag = "ok" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def run_probe(env_value):
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    if env_value is None:
        env.pop("KECK_AO_WORKERS", None)
    else:
        env["KECK_AO_WORKERS"] = env_value
    r = subprocess.run([sys.executable, "-c", _PROBE.format(src=SRC)],
                       capture_output=True, text=True, env=env, timeout=60)
    return r


def main():
    for bad in ("12", "0", "abc"):
        r = run_probe(bad)
        check(f"KECK_AO_WORKERS={bad!r}: MainWindow() does not crash",
              r.returncode == 0, f"returncode={r.returncode}\n{r.stderr[-500:]}")
        check(f"KECK_AO_WORKERS={bad!r}: spin box falls back to "
              "default_workers(), lazy config-load unaffected",
              "PROBE_OK" in r.stdout, r.stdout[-200:])
        check(f"KECK_AO_WORKERS={bad!r}: exactly one WARNING line names "
              "the bad variable",
              r.stdout.count("WARNING:") == 1
              and f"KECK_AO_WORKERS={bad!r}" in r.stdout,
              r.stdout.splitlines()[0] if r.stdout else "(no output)")

    # control: unset must build clean, no warning at all
    r = run_probe(None)
    check("KECK_AO_WORKERS unset (control): MainWindow() does not crash",
          r.returncode == 0, f"returncode={r.returncode}\n{r.stderr[-500:]}")
    check("KECK_AO_WORKERS unset (control): no WARNING line",
          "WARNING:" not in r.stdout, r.stdout)

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("gui_phase40: all checks passed")


if __name__ == "__main__":
    main()
