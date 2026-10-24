"""Find the local interface/IP that sits on the Tello network (192.168.10.0/24).

Linux uses `ip` and `nmcli`; Windows uses psutil and `netsh wlan`. On Windows the interface
names in fleet.yaml (wlx...) don't exist, so `tello_iface()` finds the adapter by the SSID
it is joined to, and the identity gate still checks that SSID and BSSID against fleet.yaml.
"""

from __future__ import annotations

import ipaddress
import json
import subprocess
import sys
from dataclasses import dataclass

WINDOWS = sys.platform == "win32"

TELLO_SUBNET = ipaddress.ip_network("192.168.10.0/24")
TELLO_IP = "192.168.10.1"


class TelloNetworkError(RuntimeError):
    """No (or no unique) local interface on the Tello subnet."""


@dataclass(frozen=True)
class TelloInterface:
    name: str
    ip: str


def find_tello_interfaces() -> list[TelloInterface]:
    """Return every local IPv4 address inside the Tello subnet, with its interface."""
    if WINDOWS:
        import socket

        import psutil

        return [TelloInterface(name=name, ip=a.address)
                for name, addrs in psutil.net_if_addrs().items() for a in addrs
                if a.family == socket.AF_INET and a.address != TELLO_IP
                and ipaddress.ip_address(a.address) in TELLO_SUBNET]
    result = subprocess.run(
        ["ip", "-json", "-4", "addr", "show"], capture_output=True, text=True, check=True
    )
    found: list[TelloInterface] = []
    for link in json.loads(result.stdout):
        for addr in link.get("addr_info", []):
            ip = addr.get("local")
            if ip and ip != TELLO_IP and ipaddress.ip_address(ip) in TELLO_SUBNET:
                found.append(TelloInterface(name=link["ifname"], ip=ip))
    return found


def resolve_local_ip(iface: str | None = None, local_ip: str | None = None) -> TelloInterface:
    """Pick our address on the Tello subnet.

    With ``local_ip`` or ``iface`` the match must exist; with neither there must be
    exactly one candidate, so we never silently pick the wrong drone's adapter.
    """
    candidates = find_tello_interfaces()
    if local_ip is not None:
        candidates = [c for c in candidates if c.ip == local_ip]
    if iface is not None:
        candidates = [c for c in candidates if c.name == iface]

    if len(candidates) == 1:
        return candidates[0]

    wanted = f"iface={iface!r} local_ip={local_ip!r}"
    if not candidates:
        raise TelloNetworkError(
            f"No local address in {TELLO_SUBNET} ({wanted}). "
            "Is the Wi-Fi adapter connected to the drone's TELLO-XXXXXX network?"
        )
    listing = ", ".join(f"{c.name}={c.ip}" for c in candidates)
    raise TelloNetworkError(
        f"Several interfaces on {TELLO_SUBNET}: {listing}. Set local_ip in fleet.yaml."
    )


@dataclass(frozen=True)
class WifiLink:
    ssid: str
    bssid: str


def split_nmcli_terse(line: str) -> list[str]:
    """Split one `nmcli -t` line on ':' while honouring its '\\:' and '\\\\' escapes."""
    fields: list[str] = []
    current: list[str] = []
    chars = iter(line)
    for ch in chars:
        if ch == "\\":
            current.append(next(chars, ""))
        elif ch == ":":
            fields.append("".join(current))
            current = []
        else:
            current.append(ch)
    fields.append("".join(current))
    return fields


def parse_active_wifi(nmcli_output: str) -> WifiLink | None:
    """Pick the ACTIVE=yes row from `nmcli -t -f ACTIVE,SSID,BSSID dev wifi list`."""
    for line in nmcli_output.splitlines():
        fields = split_nmcli_terse(line)
        if len(fields) == 3 and fields[0] == "yes":
            return WifiLink(ssid=fields[1], bssid=fields[2])
    return None


def parse_netsh_interfaces(output: str) -> list[dict[str, str]]:
    """`netsh wlan show interfaces` -> one {name, state, ssid, bssid} dict per adapter."""
    keys = {"Name": "name", "State": "state", "SSID": "ssid", "AP BSSID": "bssid", "BSSID": "bssid"}
    adapters: list[dict[str, str]] = []
    for line in output.splitlines():
        key, sep, value = line.partition(" : ")
        key = key.strip()
        if not sep or key not in keys or line.startswith("        "):
            continue                    # "Colocated APs" sub-rows are indented further
        if key == "Name":
            adapters.append({})
        if adapters:
            adapters[-1].setdefault(keys[key], value.strip())
    return adapters


def windows_wifi() -> list[dict[str, str]]:
    result = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True,
                            text=True, check=False)
    return parse_netsh_interfaces(result.stdout) if result.returncode == 0 else []


def tello_iface(iface: str, ssid: str) -> str:
    """The adapter to use for a drone: fleet.yaml's name on Linux; on Windows the adapter
    joined to the drone's SSID (fleet.yaml's name if none is, so the error names it)."""
    if not WINDOWS:
        return iface
    for adapter in windows_wifi():
        if adapter.get("state") == "connected" and adapter.get("ssid") == ssid:
            return adapter["name"]
    return iface


def active_wifi(iface: str) -> WifiLink | None:
    """SSID and BSSID the interface is associated with right now, or None."""
    if WINDOWS:
        for adapter in windows_wifi():
            if adapter.get("name") == iface and adapter.get("state") == "connected":
                return WifiLink(ssid=adapter.get("ssid", ""), bssid=adapter.get("bssid", ""))
        return None
    result = subprocess.run(
        ["nmcli", "-t", "-f", "ACTIVE,SSID,BSSID", "dev", "wifi", "list",
         "ifname", iface, "--rescan", "no"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return None
    return parse_active_wifi(result.stdout)
