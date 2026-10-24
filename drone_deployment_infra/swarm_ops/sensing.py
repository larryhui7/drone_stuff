"""Sensing for an umbrella policy: where our drones are, and where the attackers are.

Our drones (Vision, one per drone): the drone's video over its own dongle, decoded and run
through core/positional_kalman_filter.FilteredPositioner -- the localizer + Kalman filter
the simulator trains on -- with config/map.yaml and that drone's calibration
config/camera/tello_<n>.yaml (default.yaml if it has none).

Attackers (AttackerFeed + drone_telemetry/track.Tracker): the other team's drone_gui
telemetry (UDP JSON, drone_telemetry/README.md) arrives on --telemetry-port. Each attacker's
last minute of fixes is kept; when its pose goes stale the tracker coasts it in a straight
line. Optionally (spot_model) a drone detector runs on our cameras and each box becomes a
room-frame sighting (range from the Tello's width, our drone's own camera pose) that
refreshes the attacker it matches.

Everything is in the room frame of map.yaml: origin tag 10's centre, x right, y up, z out
of the wall (in front of it is +z); metres, m/s. Times are this laptop's time.time(): the
feed restamps each datagram on arrival, so the two laptops' clocks never meet.
"""
from __future__ import annotations

import json
import math
import os
import socket
import sys
import threading
import time

import numpy as np

DD_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in ("core", "drone_telemetry", os.path.join("cal_and_map")):
    _p = os.path.join(DD_ROOT, _p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from positional_kalman_filter import FilteredPositioner  # noqa: E402
from track import Tracker, box_to_camera  # noqa: E402

DRONE_LABELS = {"drone", "quadcopter"}   # names the baked YOLO-World vocabulary uses for a Tello
SPOT_CONF = 0.3


VIDEO_CMDS = {"streamon", "streamoff"}  # read-only; swarm_policies' command filter predates video


def raw_send(drone, cmd: str, timeout: float = 5.0) -> str:
    """Send a video command on the drone's own socket (same lock and reply handling as
    Drone.send). Only VIDEO_CMDS: every flight command still goes through Drone.send's check."""
    assert cmd in VIDEO_CMDS, cmd
    with drone.lock:
        drone.sock.setblocking(False)
        try:
            while True:
                drone.sock.recv(1024)
        except (BlockingIOError, OSError):
            pass
        drone.sock.settimeout(timeout)
        try:
            drone.sock.sendto(cmd.encode(), ("192.168.10.1", 8889))
            return drone.sock.recv(1024).decode(errors="ignore").strip().lower()
        except (socket.timeout, OSError):
            return "timeout"


def heading_deg(R) -> float:
    """Room-frame heading of a camera (R: camera axes in the room, x right, y down, z forward):
    0 = looking at the tag-10 wall (along -z), + = turned toward +x."""
    f = np.asarray(R, float)[:, 2]
    return math.degrees(math.atan2(f[0], -f[2]))


class Vision:
    """One drone: video thread + localization thread -> .est (FilteredPositioner output)."""

    def __init__(self, drone, on_fix=None) -> None:
        self.drone = drone
        self.fpos = FilteredPositioner(f"tello_{drone.num}")
        self.on_fix = on_fix               # called (vision, frame, raw fix, t) after each located frame
        self.frame, self.frame_t, self.frame_id = None, 0.0, 0
        self.est, self.est_t = None, 0.0
        self.yaw = None                    # last heading from a fix, deg
        self.frames = self.decode_errors = 0
        self.stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._video_loop, daemon=True).start()
        threading.Thread(target=self._locate_loop, daemon=True).start()

    def _video_loop(self) -> None:
        import av
        vs = self.drone._sock(11111)               # bound to this drone's dongle
        vs.settimeout(1.0)
        codec = av.CodecContext.create("h264", "r")
        while not self.stop.is_set():
            try:
                data = vs.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                for pkt in codec.parse(data):
                    for fr in codec.decode(pkt):
                        self.frame, self.frame_t = fr.to_ndarray(format="bgr24"), time.time()
                        self.frame_id += 1
                        self.frames += 1
            except Exception:  # noqa: BLE001  H.264 errors before each keyframe are normal
                self.decode_errors += 1
        vs.close()

    def _locate_loop(self) -> None:
        last = 0
        while not self.stop.is_set():
            if self.frame_id == last or self.frame is None:
                time.sleep(0.005)
                continue
            last, frame, t = self.frame_id, self.frame, self.frame_t
            est = self.fpos.locate(frame, t=t)
            raw = est["raw"]
            if raw is not None:
                self.yaw = heading_deg(raw["R"])
            self.est, self.est_t = est, t
            if self.on_fix is not None and raw is not None:
                self.on_fix(self, frame, raw, t)

    def state(self, now: float) -> dict:
        """This drone now: xyz/vel (Kalman), yaw, fix_age, lost; xyz None before the first fix."""
        est = self.est
        if est is None or est["xyz"] is None:
            return {"xyz": None, "vel": None, "yaw": self.yaw, "fix_age": float("inf"), "lost": True,
                    "tags": [], "frames": self.frames}
        fix_age = est["fix_age"] + (now - self.est_t)
        return {"xyz": est["xyz"], "vel": est["vel"], "yaw": self.yaw, "fix_age": round(fix_age, 3),
                "lost": fix_age > self.fpos.cfg.lost_after,
                "tags": est["raw"]["tags"] if est["raw"] else [],
                "position_variance": est["position_variance"], "frames": self.frames}


class AttackerFeed:
    """UDP listener for the other team's telemetry -> Tracker (restamped with our clock)."""

    def __init__(self, port: int, tracker: Tracker, lock: threading.Lock) -> None:
        self.port, self.tracker, self.lock = port, tracker, lock
        self.last_t = None
        self.datagrams = self.bad = 0
        self.stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.port))
        sock.settimeout(1.0)
        while not self.stop.is_set():
            try:
                data = sock.recv(65535)
            except socket.timeout:
                continue
            try:
                msg = json.loads(data)
            except ValueError:
                self.bad += 1
                continue
            if msg.get("v") != 1:
                self.bad += 1
                continue
            msg["t"] = time.time()             # our clock: pose ages are relative to arrival
            with self.lock:
                self.tracker.update(msg)
            self.last_t, self.datagrams = msg["t"], self.datagrams + 1
        sock.close()


class Spotter:
    """One detector over our drones' newest localized frames, batched -> attacker sightings."""

    def __init__(self, model_path: str, sensing: "Sensing", conf: float = SPOT_CONF) -> None:
        import torch
        from ultralytics import YOLO
        if not os.path.exists(model_path):
            sys.exit(f"spot model {model_path} not found (vision_and_mapping/teleop_detect.py --bake makes one)")
        self.model = YOLO(model_path)
        if not set(self.model.names.values()) & DRONE_LABELS:
            sys.exit(f"spot model {model_path} has no drone class")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sensing, self.conf = sensing, conf
        self.pending: dict[int, tuple] = {}   # drone num -> (vision, frame, R, p, t), newest only
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True).start()

    def offer(self, vision: Vision, frame, raw: dict, t: float) -> None:
        with self.lock:
            self.pending[vision.drone.num] = (vision, frame, np.asarray(raw["R"], float),
                                              np.asarray(raw["xyz"], float), t)

    def _loop(self) -> None:
        while not self.stop.is_set():
            with self.lock:
                batch, self.pending = list(self.pending.values()), {}
            if not batch:
                time.sleep(0.01)
                continue
            try:
                results = self.model.predict([b[1] for b in batch], conf=self.conf, device=self.device,
                                             verbose=False)
            except Exception as exc:  # noqa: BLE001
                print(f"[sensing] spot error: {exc}", flush=True)
                time.sleep(0.5)
                continue
            for (vision, frame, R, p, t), result in zip(batch, results):
                for xyxy, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
                    if result.names[int(cls)] not in DRONE_LABELS:
                        continue
                    cam = box_to_camera(xyxy, vision.drone_K, vision.drone_dist, image_width=frame.shape[1])
                    if cam is not None:
                        self.sensing.add_sighting(vision.drone.num, t, R @ cam + p)


class Sensing:
    """Our drones' Visions + the attacker tracker; snapshot() is what a policy reads."""

    def __init__(self, drones, telemetry_port: int | None = 9100, attacker_team: str | None = "attack",
                 spot_model: str | None = None) -> None:
        self.lock = threading.Lock()
        self.tracker = Tracker(team=attacker_team)
        self.feed = AttackerFeed(telemetry_port, self.tracker, self.lock) if telemetry_port else None
        self.spotter = Spotter(spot_model, self) if spot_model else None
        self.visions = {}
        for d in drones:
            v = Vision(d, on_fix=self.spotter.offer if self.spotter else None)
            det = v.fpos.positioner.detector                       # its calibration, for box ranging
            v.drone_K, v.drone_dist = det.K, det.dist
            self.visions[d.num] = v
        self._snap, self._snap_t = None, 0.0

    def start(self) -> None:
        """Start every thread. Each drone's video comes on once its status stream is live
        (swarm_policies' preflight has put it in SDK mode), so positions run before takeoff."""
        for v in self.visions.values():
            threading.Thread(target=self._stream_on, args=(v,), daemon=True).start()
            v.start()
        if self.feed:
            self.feed.start()
        if self.spotter:
            self.spotter.start()

    def _stream_on(self, v: Vision, timeout: float = 120.0) -> None:
        d, t0 = v.drone, time.time()
        while not d.st and time.time() - t0 < timeout:      # preflight not done yet
            time.sleep(0.2)
        for _ in range(3):
            if raw_send(d, "streamon") == "ok":
                d.log("video on")
                return
        d.log("streamon got no ok: no video, no position")

    def add_sighting(self, viewer: int, t: float, xyz) -> None:
        now = time.time()
        teammates = [s["xyz"] for n, s in ((n, v.state(now)) for n, v in self.visions.items())
                     if n != viewer and s["xyz"] is not None and not s["lost"]]
        with self.lock:
            self.tracker.add_sighting(t, xyz, teammates)

    def snapshot(self, max_age: float = 0.02) -> dict:
        """{"t", "defenders": {num: state}, "attackers": {name: estimate}, "feed_age", "sightings"}.
        Cached for max_age s, so every drone in one tick sees the same picture."""
        now = time.time()
        if self._snap is not None and now - self._snap_t < max_age:
            return self._snap
        with self.lock:
            attackers = {name: self.tracker.estimate(name, now) for name in self.tracker.tracks}
            sightings = dict(self.tracker.sightings)
            unmapped = sorted(self.tracker.unmapped)
        feed_age = None if not self.feed or self.feed.last_t is None else round(now - self.feed.last_t, 2)
        self._snap = {"t": now, "defenders": {n: v.state(now) for n, v in self.visions.items()},
                      "attackers": attackers, "feed_age": feed_age, "sightings": sightings,
                      "unmapped_tags": unmapped}
        self._snap_t = now
        return self._snap

    def close(self) -> None:
        for v in self.visions.values():
            v.stop.set()
            raw_send(v.drone, "streamoff", 2)
        for part in (self.feed, self.spotter):
            if part:
                part.stop.set()
