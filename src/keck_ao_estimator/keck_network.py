"""Is this machine on the Keck network?

Gates the Measured SR tab's auto-measure polling (Eduardo 2026-09-22):
polling reads the summit data disks and the AO server, so it is only
offered where those can be reached.

DNS is NOT a test: keck.hawaii.edu is public, and waikoko-new /
k2aoserver-new resolve identically through 8.8.8.8 from anywhere
(checked 2026-09-22). What the firewall hides is the SERVICES, so the
test is a TCP connection to one: the AO server's ssh or the NIRC2 data
server's NFS port. Inside (or on the VPN) either answers in ~10 ms; from
outside the connection is dropped and times out.

$KECK_AO_KECK_NETWORK=1/0 forces the answer (regress suites, odd
networks).
"""
import os
import socket

__all__ = ["KECK_PROBES", "ENV_KECK_NETWORK", "keck_network_check"]

KECK_PROBES = (("k2aoserver-new.keck.hawaii.edu", 22),   # AO server ssh
               ("waikoko-new.keck.hawaii.edu", 2049))    # NIRC2 data NFS
ENV_KECK_NETWORK = "KECK_AO_KECK_NETWORK"


def _short(host, port):
    return f"{host.split('.')[0]}:{port}"


def keck_network_check(probes=KECK_PROBES, timeout=2.0):
    """(on_keck_network, reason). Tries each probe in turn; worst case
    len(probes) * timeout seconds, so call it off the GUI thread."""
    forced = os.environ.get(ENV_KECK_NETWORK, "").strip()
    if forced in ("0", "1"):
        return forced == "1", f"${ENV_KECK_NETWORK}={forced}"
    for host, port in probes:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True, f"reached {_short(host, port)}"
        except OSError:
            continue
    return False, ("summit servers not reachable ("
                   + ", ".join(_short(h, p) for h, p in probes) + ")")
