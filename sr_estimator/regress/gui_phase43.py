#!/usr/bin/env python3
"""Measured SR: auto-measure new frames (2026-09-22) -- the IDL Strehl
tool's autoimage timer (strehl_widget.pro: poll PATH/n????.fits every
second, measure the last one when it changes); an Instrument selector
(NIRC2 and OSIRIS can share a night); "Latest", which finds tonight's
directory (NIRC2: the NFS-mounted /s/sdata900-907 disks; OSIRIS: the AO
server over rsync -- the summit tool's KTL lookups are not available off
the summit); and REMOTE (host:/dir) PATHs polled by rsync, which must be
confirmed and stay visibly flagged while they run (Eduardo 2026-09-22).

Engine (frame_watch): newest frame by NAME per instrument (NIRC2
n####.fits, OSIRIS imager i<YYMMDD>_a######.fits), never n_unp_ copies,
NIRC2's in-progress .writing files or spectrograph frames; frame_ready
needs whole FITS blocks AND a size unchanged since the previous poll;
find_latest_night_dir takes the latest-dated night across every account
and disk (an EMPTY directory for tonight beats last night's frames),
the last-written account between same-date ones, ignores dates after
tomorrow; rsync listing parser, remote PATH syntax, cache pruning.

GUI: controls unclipped; switching on without a PATH refuses and
unticks; with the REAL polling thread: the current last frame is
measured on switch-on, a frame written as .writing and renamed is
measured once it lands, a frame landing while busy waits, a second one
supersedes it (logged as skipped) and is measured when the tab is free;
the NIRC2 watcher ignores OSIRIS frames; the instrument switch stops
auto-measure and swaps the per-instrument PATH; a remote PATH (rsync
stubbed onto a local "server" directory): declining the confirmation
unticks, accepting shows the red banner + tab tag, the frame is copied to
the cache and measured, changing PATH stops it and clears the flag;
Latest per instrument (stubbed); switching off stops the thread; the
watch state is not in the saved config, the instrument and its PATHs
are.

Keck-network gate (Eduardo 2026-09-22: polling only on the Keck
network, greyed out otherwise): keck_network_check is a TCP reach test
(DNS is public, so no test at all) -- a listening port answers, a closed
one does not, $KECK_AO_KECK_NETWORK forces either answer; the checkbox
starts greyed, enables on a positive check, greys again with the reason
in its tooltip on a negative one, and a negative check while polling
stops it. The suite forces the network ON (CI is off-site). Fully
offline, headless.
"""
import datetime as dt
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import warnings
warnings.filterwarnings("ignore")

from qtcompat import QtWidgets

import keck_ao_estimator.frame_watch as fw
import keck_ao_estimator.gui as gui
from gui_phase29 import make_frame, make_osiris, pump


def touch(path, nbytes=2880, mtime=None):
    with open(path, "wb") as f:
        f.write(b"\0" * nbytes)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def check_engine():
    d = tempfile.mkdtemp(prefix="gui43_fw_")
    for name in ("n0003.fits", "n0012.fits", "n_unp_0013.fits",
                 "n0014.fits.writing", "notes.txt"):
        touch(os.path.join(d, name))
    name, path, size = fw.newest_frame(d)
    assert name == "n0012.fits" and size == 2880, (name, size)
    assert fw.nirc2_frame_number("n0012.fits") == 12
    assert fw.nirc2_frame_number("i260731_a001002.fits") is None
    o = tempfile.mkdtemp(prefix="gui43_osi_")
    for name in ("i260731_a001002.fits", "i260731_a002001.fits",
                 "s260731_a003001.fits", "i260731_a001009.fits"):
        touch(os.path.join(o, name))
    assert fw.newest_frame(o)[0] == "i260731_a002001.fits", \
        "OSIRIS newest = last set/frame by name; spectrograph ignored"
    assert fw.newest_frame(os.path.join(d, "nope")) is None
    assert fw.newest_frame(tempfile.mkdtemp()) is None
    assert not fw.frame_ready(5760, None), "first sighting is never ready"
    assert not fw.frame_ready(5760, 2880), "still growing"
    assert not fw.frame_ready(3000, 3000), "not whole FITS blocks"
    assert not fw.frame_ready(0, 0)
    assert fw.frame_ready(5760, 5760)
    print("  [ok] newest_frame (NIRC2 + OSIRIS names, n_unp_/.writing/"
          "spectrograph ignored), frame_ready rules")

    now = dt.datetime(2026, 9, 23, 6, 0, tzinfo=dt.timezone.utc)
    assert fw.night_dir_name(now) == "2026sep23"
    root = tempfile.mkdtemp(prefix="gui43_s_")
    disks = [os.path.join(root, f"sdata90{k}") for k in (5, 7)]
    t0 = time.time()

    def night(disk, account, dirname, frame_mtime):
        p = os.path.join(disk, account, dirname)
        os.makedirs(p)
        touch(os.path.join(p, "n0001.fits"), mtime=frame_mtime)
        return p
    night(disks[1], "nirc7", "2026sep21", t0 - 90000)
    last = night(disks[1], "nirc2eng", "2026sep22", t0 - 5000)
    os.makedirs(os.path.join(disks[1], "nirc2eng", "dthompson"))
    night(disks[1], "nirc9", "2027jan01", t0)      # mislabelled future
    got, tonight = fw.find_latest_night_dir(disks, utc_now=now)
    assert (got, tonight) == (last, False), \
        "tonight's directory not made yet: last night, flagged"
    empty = os.path.join(disks[0], "nirc4", "2026sep23")
    os.makedirs(empty)
    got, tonight = fw.find_latest_night_dir(disks, utc_now=now)
    assert (got, tonight) == (empty, True), \
        "an EMPTY directory for tonight beats last night's frames"
    live = night(disks[1], "nirc5", "2026sep23_B", t0 + 10)
    got, tonight = fw.find_latest_night_dir(disks, utc_now=now)
    assert (got, tonight) == (live, True), \
        "same date: the account written to last wins, across disks"
    assert fw.find_latest_night_dir(
        [os.path.join(root, "absent")], utc_now=now) == (None, False)
    print("  [ok] find_latest_night_dir: latest date first (empty tonight "
          "beats last night), last-written account, future dates ignored")

    # instrument filter, rsync listing parser, remote PATH syntax, cache
    both = tempfile.mkdtemp(prefix="gui43_both_")
    for name in ("n0003.fits", "i260731_a001002.fits"):
        touch(os.path.join(both, name))
    assert fw.newest_frame(both, "nirc2")[0] == "n0003.fits"
    assert fw.newest_frame(both, "osiris")[0] == "i260731_a001002.fits"
    listing = fw.parse_rsync_listing(
        "drwxrwsrwx          4,096 2026/08/13 15:47:04 .\n"
        "-rwxr-xr-x     16,816,320 2026/08/13 10:03:30 "
        "i260813_a001002.fits\n"
        "rsync: opendir \"/s/sdata1100/osiriseng\" failed: Permission "
        "denied (13)\n")
    assert listing == [
        (True, ".", 4096, dt.datetime(2026, 8, 13, 15, 47, 4)),
        (False, "i260813_a001002.fits", 16816320,
         dt.datetime(2026, 8, 13, 10, 3, 30))], listing
    assert fw.split_remote("k2ao:/s/sdata1100/osiris4/260921/IMAG/raw/") \
        == ("k2ao", "/s/sdata1100/osiris4/260921/IMAG/raw")
    for bad in ("/s/sdata907", "k2ao:relative", "k2ao:/a b", "a;b:/x", ""):
        assert fw.split_remote(bad) is None, bad
    cache = tempfile.mkdtemp(prefix="gui43_cache_")
    for k in range(25):
        touch(os.path.join(cache, f"i260731_a001{k:03d}.fits"))
    fw.prune_cache(cache, keep=20)
    kept = sorted(os.listdir(cache))
    assert len(kept) == 20 and kept[0] == "i260731_a001005.fits", kept
    print("  [ok] instrument filter, rsync listing parser (errors skipped), "
          "host:/dir syntax, cache pruned to the newest 20")


def check_network_probe():
    import socket
    from keck_ao_estimator.keck_network import (ENV_KECK_NETWORK,
                                                keck_network_check)
    saved = os.environ.pop(ENV_KECK_NETWORK, None)
    try:
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        closed = socket.socket()
        closed.bind(("127.0.0.1", 0))
        dead = closed.getsockname()[1]
        closed.close()
        ok, why = keck_network_check([("127.0.0.1", dead),
                                      ("127.0.0.1", port)], timeout=1.0)
        assert ok and why == f"reached 127:{port}", (ok, why)
        srv.close()
        ok, why = keck_network_check([("127.0.0.1", dead)], timeout=1.0)
        assert not ok and "not reachable" in why, (ok, why)
        os.environ[ENV_KECK_NETWORK] = "0"
        assert keck_network_check()[0] is False
        os.environ[ENV_KECK_NETWORK] = "1"
        assert keck_network_check([("127.0.0.1", dead)])[0] is True
    finally:
        os.environ.pop(ENV_KECK_NETWORK, None)
        if saved is not None:
            os.environ[ENV_KECK_NETWORK] = saved
    print("  [ok] keck_network_check: listening port -> on, closed -> off, "
          "$KECK_AO_KECK_NETWORK forces 0/1")


def main():
    check_engine()
    check_network_probe()
    # off-site CI: the gate is exercised explicitly below, the rest of the
    # suite needs polling available
    os.environ["KECK_AO_KECK_NETWORK"] = "1"

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow()
    win._nighttime_is_night = lambda: True     # gate-pin (house pattern)
    win.resize(1550, 950)
    win.show()
    labels = [win.plot_tabs.tabText(i) for i in range(win.plot_tabs.count())]
    win.plot_tabs.setCurrentIndex(labels.index("Measured SR"))
    app.processEvents()
    # (offscreen font metrics already clip the pre-existing Autofind row
    # in this 330 px column, so the checkbox is held to the full row
    # width rather than its offscreen size hint)
    # (offscreen font metrics already clip the pre-existing Autofind row
    # in this 330 px column, so widgets are held to their siblings rather
    # than to offscreen size hints)
    for w in (win.n2_latest, win.n2_instrument):
        assert w.width() >= w.sizeHint().width(), \
            f"clipped: {w.width()} < {w.sizeHint().width()}"
    assert win.n2_watch.width() >= win.n2_autofind.width()
    assert win.n2_path.width() >= 90, \
        f"path box squeezed to {win.n2_path.width()} px"
    assert win.n2_files.height() >= 40, win.n2_files.height()
    assert not win.n2_watch.isChecked()
    assert win.n2_instrument.currentText() == "NIRC2"
    assert not win.n2_rsync_flag.isVisible()
    print(f"  [ok] Instrument + Latest row, Auto-measure row (path box "
          f"{win.n2_path.width()} px, file list {win.n2_files.height()} px)")

    # ---- Keck-network gate -------------------------------------------------
    pump(lambda: win.n2_watch.isEnabled(), timeout=15)
    assert win.n2_watch.isEnabled() and win._n2_on_keck is True
    assert "UNAVAILABLE" not in win.n2_watch.toolTip()
    win._nirc2_on_network(False, "summit servers not reachable (x:22)")
    assert not win.n2_watch.isEnabled()
    assert "UNAVAILABLE" in win.n2_watch.toolTip() \
        and "x:22" in win.n2_watch.toolTip()
    assert "not on the Keck network (summit servers not reachable" \
        in win.n2_log.toPlainText()
    win.n2_watch.setChecked(True)          # programmatic: still refused
    assert not win.n2_watch.isChecked() and win._n2_watcher is None
    win._nirc2_on_network(True, "reached k2aoserver-new:22")
    assert win.n2_watch.isEnabled()
    assert "auto-measure available" in win.n2_log.toPlainText()
    print("  [ok] Keck-network gate: greyed + reason off-network, enabled "
          "on it, refused programmatically while greyed")

    # ---- refuses without a directory ---------------------------------------
    win.n2_path.setText("")
    win.n2_watch.setChecked(True)
    assert not win.n2_watch.isChecked()
    assert getattr(win, "_n2_watcher", None) is None
    assert "auto-measure needs PATH" in win.n2_log.toPlainText()
    print("  [ok] no PATH: refuses, unticks, no thread")

    # ---- real watcher: current last frame measured on switch-on ------------
    tmp = tempfile.mkdtemp(prefix="gui43_nirc2_")
    make_frame(tmp, 7)
    win.n2_path.setText(tmp)
    win.n2_watch.setChecked(True)
    assert win._n2_watcher.isRunning()
    pump(lambda: "Image 7  SR " in win.n2_log.toPlainText(), timeout=60)
    assert "Image 7  SR " in win.n2_log.toPlainText(), win.n2_log.toPlainText()
    pump(lambda: win.n2_go.isEnabled())
    assert abs(float(win.n2_strehl_out.text()) - 1.0) < 0.06
    print("  [ok] switch-on measures the current last frame (n0007)")

    # ---- a frame written as .writing then renamed --------------------------
    stage = tempfile.mkdtemp(prefix="gui43_stage_")
    make_frame(stage, 8)
    os.replace(os.path.join(stage, "n0008.fits"),
               os.path.join(tmp, "n0008.fits.writing"))
    t_end = time.time() + 2.5
    pump(lambda: time.time() > t_end, timeout=5)
    assert "Image 8" not in win.n2_log.toPlainText(), \
        "a .writing file must never be measured"
    os.replace(os.path.join(tmp, "n0008.fits.writing"),
               os.path.join(tmp, "n0008.fits"))
    pump(lambda: "Image 8  SR " in win.n2_log.toPlainText(), timeout=60)
    assert "Image 8  SR " in win.n2_log.toPlainText()
    assert win.n2_im1.value() == 8 and win.n2_nim.value() == 1
    assert any(win.n2_files.item(i).text() == "n0008.fits"
               for i in range(win.n2_files.count())), \
        "the new frame joins the file list"
    pump(lambda: win.n2_go.isEnabled())
    print("  [ok] .writing ignored; measured once renamed; list refreshed")

    # ---- busy: waits, newer supersedes (logged), drained when free ---------
    win._n2_field_busy = True
    make_frame(tmp, 9)
    pump(lambda: (win._n2_watch_pending or "").endswith("n0009.fits"),
         timeout=30)
    assert (win._n2_watch_pending or "").endswith("n0009.fits")
    assert "Image 9" not in win.n2_log.toPlainText()
    make_frame(tmp, 10)
    pump(lambda: (win._n2_watch_pending or "").endswith("n0010.fits"),
         timeout=30)
    log = win.n2_log.toPlainText()
    assert "skipped n0009.fits -- n0010.fits landed while busy" in log, log
    win._n2_field_busy = False
    win._nirc2_field_finish_summary(setup_failed=True)   # drains on a tick
    pump(lambda: "Image 10  SR " in win.n2_log.toPlainText(), timeout=60)
    assert "Image 10  SR " in win.n2_log.toPlainText()
    assert "Image 9  SR " not in win.n2_log.toPlainText()
    pump(lambda: win.n2_go.isEnabled())
    print("  [ok] busy: n0009 waited, n0010 superseded it (logged), "
          "measured when the field run ended")

    # ---- the NIRC2 watcher ignores OSIRIS frames -------------------------
    make_osiris(tmp)                  # i260723_a000001.fits lands in tmp
    t_end = time.time() + 3.0
    pump(lambda: time.time() > t_end, timeout=6)
    assert "i260723_a000001" not in win.n2_log.toPlainText(), \
        "the NIRC2 watcher must not pick up an OSIRIS frame"
    print("  [ok] NIRC2 watcher ignores an OSIRIS frame in its directory")

    # ---- instrument switch: stops polling, swaps the per-instrument PATH --
    w = win._n2_watcher
    win.n2_instrument.setCurrentText("OSIRIS")
    assert not win.n2_watch.isChecked() and win._n2_watcher is None
    assert w.isFinished()
    assert win.n2_path.text() == "", "OSIRIS has no PATH yet"
    win.n2_instrument.setCurrentText("NIRC2")
    assert win.n2_path.text() == tmp, "NIRC2 PATH remembered"
    assert not win.n2_watch.isChecked(), "switching back does not resume"
    print("  [ok] instrument switch stops auto-measure, PATH per instrument")

    # ---- remote PATH (rsync stubbed onto a local 'server' dir) -------------
    server = tempfile.mkdtemp(prefix="gui43_server_")
    os.environ["KECK_AO_REMOTE_CACHE"] = tempfile.mkdtemp(prefix="gui43_rc_")
    calls = {"list": 0, "fetch": 0}

    def fake_list(host, rdir, kind=None, timeout=30.0):
        assert host == "k2ao"
        calls["list"] += 1
        nf = fw.newest_frame(server, kind)
        return None if nf is None else (nf[0], f"{rdir}/{nf[0]}", nf[2])

    def fake_fetch(host, rpath, local_dir, timeout=180.0):
        import shutil
        calls["fetch"] += 1
        os.makedirs(local_dir, exist_ok=True)
        return shutil.copy(os.path.join(server, os.path.basename(rpath)),
                           local_dir)
    real_list, real_fetch = fw.remote_newest_frame, fw.remote_fetch
    fw.remote_newest_frame, fw.remote_fetch = fake_list, fake_fetch
    try:
        make_osiris(server)
        win.n2_instrument.setCurrentText("OSIRIS")
        rpath = "k2ao:/s/sdata1100/osiris4/260923/IMAG/raw"
        win.n2_path.setText(rpath)
        assert win.n2_rsync_flag.isVisible()
        assert "not polling" in win.n2_rsync_flag.text()
        asked = []
        win._nirc2_confirm_rsync = lambda h, d: asked.append((h, d)) or False
        win.n2_watch.setChecked(True)
        assert asked == [("k2ao", "/s/sdata1100/osiris4/260923/IMAG/raw")]
        assert not win.n2_watch.isChecked() and win._n2_watcher is None
        assert "rsync polling declined" in win.n2_log.toPlainText()
        assert calls["list"] == 0, "declined: nothing may be listed"
        win._nirc2_confirm_rsync = lambda h, d: True
        win.n2_watch.setChecked(True)
        assert "RSYNC POLLING k2ao" in win.n2_rsync_flag.text()
        assert win.n2_rsync_flag.property("cue") == "err"
        idx = win.plot_tabs.indexOf(win._n2_tab_page)
        assert win.plot_tabs.tabText(idx) == "Measured SR ⚠ rsync"
        assert "POLLING " + rpath + " BY RSYNC" in win.n2_log.toPlainText()
        pump(lambda: "Image i260723_a000001  SR " in win.n2_log.toPlainText(),
             timeout=60)
        assert "Image i260723_a000001  SR " in win.n2_log.toPlainText()
        pump(lambda: win.n2_go.isEnabled())
        assert calls["fetch"] == 1
        assert os.path.isfile(os.path.join(win._nirc2_frames_dir(),
                                           "i260723_a000001.fits"))
        assert win.n2_files.count() == 1, "file list shows the cache"
        assert win._n2_image.shape == (2048, 2048)
        w = win._n2_watcher
        win.n2_path.setText("k2ao:/s/sdata1100/osiris4/260924/IMAG/raw")
        assert not win.n2_watch.isChecked() and w.isFinished(), \
            "a new remote target must be re-confirmed"
        assert "not polling" in win.n2_rsync_flag.text()
        assert win.plot_tabs.tabText(idx) == "Measured SR"
        win.n2_path.setText(tmp)
        assert not win.n2_rsync_flag.isVisible()
    finally:
        fw.remote_newest_frame, fw.remote_fetch = real_list, real_fetch
    print("  [ok] remote PATH: declined -> unticked, nothing listed; accepted "
          "-> red banner + tab tag, frame copied to the cache and measured; "
          "PATH change stops it and clears the flag")

    # ---- Latest per instrument (stubbed) -----------------------------------
    real_n, real_o = fw.find_latest_night_dir, fw.find_latest_osiris_night
    fw.find_latest_night_dir = lambda: (tmp, False)
    fw.find_latest_osiris_night = lambda: (rpath, True)
    try:
        win.n2_latest.click()
        assert not win.n2_instrument.isEnabled(), "locked while searching"
        pump(lambda: win.n2_latest.isEnabled(), timeout=30)
        assert win.n2_path.text() == rpath, "OSIRIS Latest -> remote PATH"
        assert "one-off rsync listing" in win.n2_log.toPlainText()
        assert not win.n2_watch.isChecked(), "Latest never starts polling"
        win.n2_instrument.setCurrentText("NIRC2")
        win.n2_latest.click()
        pump(lambda: win.n2_latest.isEnabled(), timeout=30)
        assert win.n2_path.text() == tmp
        assert "most recent night" in win.n2_log.toPlainText()
        fw.find_latest_night_dir = lambda: (None, False)
        win.n2_latest.click()
        pump(lambda: win.n2_latest.isEnabled(), timeout=30)
        assert "no NIRC2 night directory found" in win.n2_log.toPlainText()

        def boom():
            raise OSError("rsync exit 255: Permission denied")
        fw.find_latest_osiris_night = boom
        win.n2_instrument.setCurrentText("OSIRIS")
        win.n2_latest.click()
        pump(lambda: win.n2_latest.isEnabled(), timeout=30)
        assert "Latest (OSIRIS) failed: OSError: rsync exit 255" \
            in win.n2_log.toPlainText()
    finally:
        fw.find_latest_night_dir, fw.find_latest_osiris_night = real_n, real_o
    win.n2_instrument.setCurrentText("NIRC2")
    assert win.n2_path.text() == tmp
    print("  [ok] Latest per instrument (OSIRIS -> remote PATH, never "
          "starts polling), fallback / not-found / rsync failure logged")

    cfg = win._collect_config()["nirc2"]
    assert cfg["instrument"] == "NIRC2"
    assert cfg["instrument_paths"] == {"NIRC2": tmp, "OSIRIS": rpath}, cfg
    win.n2_watch.setChecked(True)
    pump(lambda: win.n2_go.isEnabled())

    # ---- losing the Keck network while polling stops it -------------------
    assert win.n2_watch.isChecked() and win._n2_watcher is not None
    w = win._n2_watcher
    win._nirc2_on_network(False, "summit servers not reachable (x:22)")
    assert not win.n2_watch.isChecked() and w.isFinished()
    assert "auto-measure stopped: Keck network lost" in win.n2_log.toPlainText()
    win._nirc2_on_network(True, "reached k2aoserver-new:22")
    win.n2_watch.setChecked(True)
    print("  [ok] losing the Keck network while polling stops it")

    # ---- off: thread stops; never in the config ----------------------------
    w = win._n2_watcher
    win.n2_watch.setChecked(False)
    assert win._n2_watcher is None and w.isFinished()
    assert "watch" not in str(win._collect_config().get("nirc2", {})).lower()
    print("  [ok] switch-off stops the thread; watch state not saved "
          "(instrument + per-instrument PATHs are)")

    pump(lambda: win.n2_go.isEnabled())
    win.close()
    app.processEvents()
    print("gui_phase43: all checks passed")


if __name__ == "__main__":
    main()
