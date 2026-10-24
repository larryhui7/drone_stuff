#!/usr/bin/env python3
"""Keep tracking a drone after it loses its tag: 60 s of fixes, then a straight-line coast.

drone_gui republishes every drone's last pose each tick and only `age` grows, so a lost
attacker looks like a pose that stops changing. This module keeps the last minute of
fresh fixes per drone in the room frame (tag 10, from config/map.yaml), and when a
pose goes stale (age > 0.5 s) it extrapolates at the velocity fitted to the last second
of fixes before the loss: the drone is assumed to keep flying straight.

Run as a relay between drone_gui and a policy that reads telemetry; the policy needs no
code change, only the port:

  ../../.venv/bin/python -u track.py --listen 9100 --forward 127.0.0.1:9101
  # drone_gui -> :9100 (its default), policy listens on :9101 instead of :9100

The relay forwards every datagram unchanged except the drones of `--team` (attack):
  - every such drone gains "track": {"xyz", "vel" (room frame: tag 10, x right, y up,
    z out of the wall; m, m/s), "predicted", "coast_s", "fix_age", "samples"}
  - while coasting, its "pose" is replaced by the predicted pose, written back in the
    frame of the last tag it saw (so a policy reading "pose" just sees it keep moving),
    with "age" 0.0 and "predicted": true; "fix_age" keeps the real age.
--no-fill-pose keeps "pose" as published and only adds "track".

Defender sightings (drone_gui --spot): a defender's camera runs a drone detector, and each
box becomes a camera-frame position (box_to_camera: calibrated intrinsics + distortion, range
from the Tello's 98 mm width). Here it is placed in the room with that defender's own pose
and becomes a fix ("source": "seen") for the nearest attacker its own tags no longer track,
so a lost attacker in a defender's view is tracked instead of coasting. Sightings that land
on another defender are dropped; those near no attacker are counted as unmatched.

As a library, feed it datagrams and query:
  tracker = Tracker()
  tracker.update(msg)                # each decoded telemetry dict
  tracker.estimate("TELLO-2")        # {"xyz", "vel", "predicted", "coast_s", ...} or None

  python track.py --selftest         # pass: SELFTEST PASSED
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from paths import MAP_PATH  # noqa: E402

from telemetry import parse_destinations  # noqa: E402

STALE_S = 0.5          # pose older than this: lost, coast (listen.py's STALE)
BUFFER_S = 60.0        # fixes kept per drone
FIT_S = 1.0            # velocity is fitted over the last this-many seconds of fixes
MIN_FIT_SPAN_S = 0.3   # shorter than this: too few fixes to trust a velocity, hold still
OUTLIER_M = 0.3        # a fix this far off the first line fit (a pose flip) is dropped
MAX_SPEED = 3.0        # m/s, clamp on the fitted velocity
MAX_COAST_S = BUFFER_S  # coast at most this long, then no estimate
FLOOR_Y = -1.22        # room-frame floor (tag 10 centre is 1.22 m up): coasting stops there

# Sightings: a defender's camera sees an attacker (drone_gui --spot publishes the boxes,
# already turned into camera-frame positions by box_to_camera).
TELLO_WIDTH_M = 0.098  # Tello body width (98 x 92.5 x 41 mm); range = f * width / box width
EDGE_PX = 3            # a box this close to the image border is cut off: its width lies
MIN_BOX_PX = 12        # narrower boxes are too far for a usable range
MAX_SEEN_RANGE_M = 6.0
SEEN_GATE_M = 0.6      # a sighting joins the nearest coasting attacker within this ...
SEEN_GATE_GROWTH = 0.3  # ... plus this per second of coasting (the prediction drifts) ...
SEEN_GATE_MAX_M = 2.5  # ... up to this
DEFENDER_CLEAR_M = 0.5  # a sighting this close to another defender is that defender

# drone_gui's pose is pupil-apriltags' tag frame (x right, y down, z into the wall);
# map.yaml uses x right, y up, z out of the face. The two differ by 180 deg about x.
FLIP = np.diag([1.0, -1.0, -1.0])


def load_map(path: Path = MAP_PATH) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """map.yaml -> {tag: (R, t)} with X_room = R @ X_tag + t."""
    data = yaml.safe_load(Path(path).read_text())
    return {int(k): (np.asarray(v["R"], float), np.asarray(v["t"], float))
            for k, v in (data.get("tags") or {}).items()}


def box_to_camera(box, K, dist, image_width: int = 960, width_m: float = TELLO_WIDTH_M):
    """Detector box (x1, y1, x2, y2 px) of a Tello -> its centre in the camera frame (x right,
    y down, z forward; m), or None when the box can't give a range.

    K and dist are the drone's calibration (config/camera/<drone>.yaml, fitted at 960 px wide);
    K is scaled to the frame width. Range is pinhole similar triangles on the undistorted box
    width: z = width_m / (x_right - x_left) in normalised image coordinates.
    """
    import cv2

    x1, y1, x2, y2 = (float(v) for v in box)
    K = np.asarray(K, float).copy()
    K[:2] *= image_width / 960.0
    if x2 - x1 < MIN_BOX_PX or x1 < EDGE_PX or x2 > image_width - EDGE_PX:
        return None
    yc = (y1 + y2) / 2
    pts = np.array([[[(x1 + x2) / 2, yc]], [[x1, yc]], [[x2, yc]]], np.float64)
    (u, v), (xl, _), (xr, _) = cv2.undistortPoints(pts, K, np.asarray(dist, float)).reshape(3, 2)
    if xr - xl <= 0:
        return None
    z = width_m / (xr - xl)
    if z > MAX_SEEN_RANGE_M:
        return None
    return np.array([u * z, v * z, z])


def camera_in_room(pose: dict, tag_map: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """(R, p): camera axes (x right, y down, z forward) and position in the room frame, from
    a telemetry pose. The camera is taken as level: the pose carries only yaw (a hovering
    Tello stays within a few degrees of level and its camera looks straight ahead)."""
    tag = int(pose["tag"])
    if tag not in tag_map:
        return None
    psi = np.radians(float(pose.get("yaw") or 0.0))
    c, s = np.cos(psi), np.sin(psi)
    cam_in_tag = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])  # yaw + = toward tag +x
    R_map, t_map = tag_map[tag]
    R = R_map @ FLIP @ cam_in_tag
    p = R_map @ (FLIP @ np.asarray(pose["xyz"], float)) + t_map
    return R, p


class Track:
    """One drone's fixes (sender time, room xyz, source) and the straight-line fit used to coast.

    source is "tag" (its own camera's pose) or "seen" (a defender's camera saw it)."""

    def __init__(self) -> None:
        self.fixes: deque[tuple[float, np.ndarray, str]] = deque()
        self.last_tag: int | None = None       # tag of its last own pose: the frame for fill-in
        self.last_yaw: float | None = None
        self.t_msg = 0.0               # sender time of the newest datagram for this drone
        self.flying = False
        self._vel: np.ndarray | None = None    # cached fit, cleared on each new fix

    def add(self, t_meas: float, xyz: np.ndarray, tag: int | None = None, yaw: float | None = None,
            source: str = "tag") -> None:
        if self.fixes and t_meas - self.fixes[-1][0] < 0.02:
            return                     # the same measurement republished, or older than the last
        self.fixes.append((t_meas, xyz, source))
        self._vel = None
        if tag is not None:
            self.last_tag, self.last_yaw = tag, yaw
        while self.fixes and t_meas - self.fixes[0][0] > BUFFER_S:
            self.fixes.popleft()

    def velocity(self) -> np.ndarray:
        """Least-squares line over the last FIT_S of fixes, refit without outliers."""
        if self._vel is not None:
            return self._vel
        t_end = self.fixes[-1][0]
        recent = [(t, p) for t, p, _ in self.fixes if t_end - t <= FIT_S]
        vel = np.zeros(3)
        if len(recent) >= 3 and recent[-1][0] - recent[0][0] >= MIN_FIT_SPAN_S:
            ts = np.array([t for t, _ in recent]) - t_end
            ps = np.array([p for _, p in recent])
            for _ in range(2):
                A = np.stack([ts, np.ones_like(ts)], axis=1)
                coef = np.linalg.lstsq(A, ps, rcond=None)[0]
                vel, p0 = np.asarray(coef[0], float), np.asarray(coef[1], float)
                keep = np.linalg.norm(ps - (p0 + ts[:, None] * vel), axis=1) <= OUTLIER_M
                if keep.all() or keep.sum() < 3:
                    break
                ts, ps = ts[keep], ps[keep]
            speed = float(np.linalg.norm(vel))
            if speed > MAX_SPEED:
                vel = vel * (MAX_SPEED / speed)
        self._vel = vel
        return vel

    def estimate(self, t: float) -> dict | None:
        if not self.fixes:
            return None
        t_fix, p_fix, source = self.fixes[-1]
        coast = max(t - t_fix, 0.0)    # age of the last usable fix (unmapped-tag fixes don't count)
        predicted = coast > STALE_S
        if predicted and coast > MAX_COAST_S:
            return None
        vel = self.velocity() if self.flying else np.zeros(3)
        xyz = p_fix + vel * coast if predicted else p_fix
        if predicted and xyz[1] < FLOOR_Y:     # flew into the floor: it has landed or crashed
            xyz = np.array([xyz[0], FLOOR_Y, xyz[2]])
        return {"xyz": [round(float(v), 3) for v in xyz],
                "vel": [round(float(v), 3) for v in vel],
                "predicted": predicted, "coast_s": round(coast, 2) if predicted else 0.0,
                "fix_age": round(coast, 2), "source": source, "samples": len(self.fixes)}


class Tracker:
    """Tracks for every drone in the telemetry, keyed by name."""

    def __init__(self, tag_map: dict | None = None, team: str | None = "attack") -> None:
        self.tag_map = load_map() if tag_map is None else tag_map
        self.team = team
        self.tracks: dict[str, Track] = {}
        self.unmapped: set[int] = set()        # tags seen that map.yaml doesn't have
        self.sightings = {"used": 0, "unmatched": 0, "teammate": 0}

    def update(self, msg: dict) -> None:
        for d in msg.get("drones", []):
            if self.team and d.get("team") != self.team:
                continue
            track = self.tracks.setdefault(d["name"], Track())
            track.t_msg, track.flying = msg["t"], bool(d.get("fly"))
            pose = d.get("pose")
            if pose is None:
                continue
            age = float(pose["age"])
            if age > STALE_S:
                continue
            tag = int(pose["tag"])
            if tag not in self.tag_map:
                self.unmapped.add(tag)
                continue
            R, t = self.tag_map[tag]
            xyz = R @ (FLIP @ np.asarray(pose["xyz"], float)) + t
            track.add(msg["t"] - age, xyz, tag, pose.get("yaw"))
        self._add_sightings(msg)

    def _add_sightings(self, msg: dict) -> None:
        """Defender sightings ("seen") -> fixes for the attacker each one matches.

        A sighting is placed with the viewer's own pose, dropped if it lands on another
        defender, and joins the nearest attacker that its own tags no longer track (within a
        gate that widens with coasting time). Sightings carry no identity, so an attacker
        that still has its tags is left to them."""
        viewers = {}                               # non-tracked drone -> its camera in the room
        for d in msg.get("drones", []):
            pose = d.get("pose")
            if (self.team and d.get("team") == self.team) or pose is None or pose["age"] > STALE_S:
                continue
            cam = camera_in_room(pose, self.tag_map)
            if cam is not None:
                viewers[d["name"]] = cam
        for d in msg.get("drones", []):
            seen = d.get("seen")
            if not seen or d["name"] not in viewers or seen["age"] > STALE_S:
                continue
            R, p_cam = viewers[d["name"]]
            t_seen = msg["t"] - float(seen["age"])
            teammates = [p for name, (_, p) in viewers.items() if name != d["name"]]
            for det in seen["dets"]:
                self.add_sighting(t_seen, R @ np.asarray(det["cam"], float) + p_cam, teammates)

    def add_sighting(self, t: float, xyz, teammates=()) -> str:
        """One room-frame sighting at time t -> a fix for the attacker it matches.

        Dropped if it lands on a teammate (positions in `teammates`); else it joins the
        nearest attacker that its own tags no longer track, within a gate that widens with
        coasting time. Returns "used", "teammate" or "unmatched" (also counted in .sightings)."""
        xyz = np.asarray(xyz, float)
        if any(np.linalg.norm(xyz - np.asarray(p, float)) < DEFENDER_CLEAR_M for p in teammates):
            self.sightings["teammate"] += 1
            return "teammate"
        best = None
        for track in self.tracks.values():
            est = track.estimate(t)
            if est is None or (not est["predicted"] and est["source"] == "tag"):
                continue
            gate = min(SEEN_GATE_M + SEEN_GATE_GROWTH * est["coast_s"], SEEN_GATE_MAX_M)
            gap = float(np.linalg.norm(xyz - np.asarray(est["xyz"])))
            if gap <= gate and (best is None or gap < best[0]):
                best = (gap, track)
        if best is None:
            self.sightings["unmatched"] += 1
            return "unmatched"
        best[1].add(t, xyz, source="seen")
        self.sightings["used"] += 1
        return "used"

    def estimate(self, name: str, t: float | None = None) -> dict | None:
        """Room-frame estimate at sender time t (default: this drone's newest datagram)."""
        track = self.tracks.get(name)
        if track is None:
            return None
        return track.estimate(track.t_msg if t is None else t)

    def tag_pose(self, name: str, est: dict) -> dict | None:
        """An estimate written back as a telemetry pose in the drone's last tag frame."""
        track = self.tracks[name]
        if track.last_tag is None:
            return None
        R, t = self.tag_map[track.last_tag]
        xyz = FLIP @ (R.T @ (np.asarray(est["xyz"], float) - t))
        return {"tag": track.last_tag, "xyz": [round(float(v), 2) for v in xyz],
                "yaw": track.last_yaw, "age": 0.0, "predicted": True, "source": est["source"],
                "fix_age": est["fix_age"]}

    def annotate(self, msg: dict, fill_pose: bool = True) -> dict:
        """Add "track" to every tracked drone; when its own pose is stale, swap in the estimate
        (from a defender's sighting, or the straight-line coast)."""
        for d in msg.get("drones", []):
            if d["name"] not in self.tracks or (self.team and d.get("team") != self.team):
                continue
            est = self.estimate(d["name"])
            d["track"] = est
            pose = d.get("pose")
            own_stale = pose is None or pose["age"] > STALE_S
            if fill_pose and est is not None and own_stale and (est["predicted"] or est["source"] == "seen"):
                d["pose"] = self.tag_pose(d["name"], est) or pose
        return msg


def relay(args) -> int:
    tracker = Tracker(load_map(args.map), args.team)
    destinations = parse_destinations(args.forward)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind((args.host, args.listen))
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[track] {args.host}:{args.listen} -> {', '.join(args.forward)}  team {args.team}  "
          f"map {len(tracker.tag_map)} tags {sorted(tracker.tag_map)}", flush=True)
    last_print, reported = 0.0, set()
    while True:
        msg = json.loads(rx.recvfrom(65535)[0])
        if msg.get("v") == 1:
            tracker.update(msg)
            msg = tracker.annotate(msg, fill_pose=not args.no_fill_pose)
        payload = json.dumps(msg, separators=(",", ":")).encode()
        for dest in destinations:
            tx.sendto(payload, dest)
        if tracker.unmapped - reported:
            print(f"[track] tags not in map.yaml, their fixes are skipped: {sorted(tracker.unmapped)}",
                  flush=True)
            reported = set(tracker.unmapped)
        if time.time() - last_print > 1.0:
            last_print = time.time()
            for name in sorted(tracker.tracks):
                est = tracker.estimate(name)
                state = ("no fix yet" if est is None else
                         f"COAST {est['coast_s']:5.1f}s" if est["predicted"] else
                         "SEEN by def" if est["source"] == "seen" else "tracking   ")
                where = "" if est is None else f"  xyz {est['xyz']}  vel {est['vel']}"
                print(f"[track] {name:<13} {state}{where}", flush=True)
            if any(tracker.sightings.values()):
                print(f"[track] sightings {tracker.sightings}", flush=True)


def _selftest() -> None:
    # Tag 10 is the origin; tag 5 sits 2 m right on the side wall, facing -x.
    rot_y = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # tag 5 z -> room -x
    tag_map = {10: (np.eye(3), np.zeros(3)), 5: (rot_y, np.array([2.0, 0.0, 1.0]))}
    tracker = Tracker(tag_map, "attack")
    p0, v = np.array([-1.0, 0.0, 3.0]), np.array([0.5, 0.0, -0.25])   # true straight flight

    def as_pose(tag, room_xyz):
        R, t = tag_map[tag]
        return (FLIP @ (R.T @ (room_xyz - t))).tolist()

    stamp, last = None, None
    for i in range(41):                                   # 4 s at 10 Hz
        t = 100.0 + 0.1 * i
        if i < 20:
            tag = 10 if i < 10 else 5                     # hand-off between tags mid-track
            last = {"tag": tag, "xyz": as_pose(tag, p0 + v * (t - 100.0)), "yaw": 0.0}
            stamp = t
        pose = dict(last, age=round(t - stamp, 2))        # drone_gui keeps the last pose, age grows
        msg = {"v": 1, "t": t, "drones": [
            {"name": "A", "team": "attack", "fly": True, "pose": pose},
            {"name": "D", "team": "defense", "fly": True, "pose": pose}]}
        tracker.update(msg)
        out = tracker.annotate(json.loads(json.dumps(msg)))
    est = tracker.estimate("A")
    truth = p0 + v * 4.0
    err = float(np.linalg.norm(np.asarray(est["xyz"]) - truth))
    assert est["predicted"] and abs(est["coast_s"] - 2.1) < 1e-6, est
    assert err < 1e-6, (est, truth)
    assert np.allclose(est["vel"], v, atol=1e-6), est
    assert "D" not in tracker.tracks, "defense must not be tracked"
    a = out["drones"][0]
    assert a["pose"]["predicted"] and a["pose"]["age"] == 0.0 and a["pose"]["tag"] == 5, a
    R, tt = tag_map[5]
    assert np.allclose(R @ (FLIP @ np.asarray(a["pose"]["xyz"])) + tt, truth, atol=0.01), a
    assert "track" not in out["drones"][1]
    # Fresh poses from a tag missing from map.yaml give no fix: the drone still coasts.
    tracker.update({"v": 1, "t": 104.5, "drones": [{"name": "A", "team": "attack", "fly": True,
                    "pose": {"tag": 6, "xyz": [0.0, 0.0, -2.0], "yaw": 0.0, "age": 0.0}}]})
    est = tracker.estimate("A")
    assert est["predicted"] and 6 in tracker.unmapped, est
    assert np.allclose(est["xyz"], p0 + v * 4.5, atol=1e-6), est
    # A pose flip inside the fit window is dropped, not fitted.
    track = Track()
    track.flying = True
    for i in range(10):
        p = p0 + v * 0.1 * i + (np.array([0.0, 0.8, 0.0]) if i == 6 else 0.0)
        track.add(0.1 * i, p, 10, 0.0)
    assert np.allclose(track.velocity(), v, atol=1e-6), track.velocity()
    # Landed drones do not coast; coasting stops at the floor.
    track.flying, track._vel = False, None
    assert track.estimate(10.0)["xyz"] == [round(float(x), 3) for x in track.fixes[-1][1]]
    track.flying = True
    track.fixes.append((20.0, np.array([0.0, -1.0, 2.0]), "tag"))
    track._vel = np.array([0.0, -1.0, 0.0])
    assert track.estimate(25.0)["xyz"][1] == FLOOR_Y

    # box_to_camera inverts the camera model: project a Tello's edges, read the range back.
    import cv2
    K = np.array([[931.8, 0.0, 491.9], [0.0, 933.7, 356.5], [0.0, 0.0, 1.0]])   # tello_4
    dist = np.array([0.00177, 0.0279, -0.00165, 0.00502, 0.0])
    for p_true in (np.array([0.3, -0.1, 2.0]), np.array([-1.2, 0.4, 3.5])):
        edges = np.array([p_true - [TELLO_WIDTH_M / 2, 0, 0], p_true + [TELLO_WIDTH_M / 2, 0, 0], p_true])
        px = cv2.projectPoints(edges, np.zeros(3), np.zeros(3), K, dist)[0].reshape(3, 2)
        box = (px[0, 0], px[2, 1] - 15, px[1, 0], px[2, 1] + 15)
        got = box_to_camera(box, K, dist)
        assert got is not None and np.linalg.norm(got - p_true) < 0.01 * p_true[2], (got, p_true)
        small = box_to_camera([b * 640 / 960 for b in box], K, dist, image_width=640)  # 640x480 frame
        assert small is not None and np.linalg.norm(small - p_true) < 0.01 * p_true[2], (small, p_true)
    assert box_to_camera((0, 100, 40, 130), K, dist) is None, "box cut off at the border"
    assert box_to_camera((500, 300, 506, 306), K, dist) is None, "too small to range"

    # camera_in_room: 4 m in front of tag 10 looking at it, then turned 90 deg toward +x.
    cam_pose = {"tag": 10, "xyz": [0.0, 0.0, -4.0], "yaw": 0.0}
    R, p = camera_in_room(cam_pose, tag_map)
    assert np.allclose(R @ [0, 0, 1.0] + p, [0, 0, 3.0]), "forward is toward the wall (-z room)"
    assert np.allclose(R @ [0, 1.0, 0] + p, [0, -1.0, 4.0]), "image down is room down"
    R, p = camera_in_room(dict(cam_pose, yaw=90.0), tag_map)
    assert np.allclose(R @ [0, 0, 1.0] + p, [1.0, 0, 4.0]), "yaw +90 looks along room +x"

    # A defender sees coasting attacker A: the sighting becomes A's fix and fills its pose.
    t = 105.0
    a_true = p0 + v * 5.0
    cam_a = np.diag([1.0, -1.0, -1.0]) @ (a_true - [0, 0, 4.0])      # D2 at room (0,0,4), yaw 0
    e_room = np.array([1.0, 0.0, 2.0])                              # teammate E, also in view
    cam_e = np.diag([1.0, -1.0, -1.0]) @ (e_room - [0, 0, 4.0])
    msg = {"v": 1, "t": t, "drones": [
        {"name": "A", "team": "attack", "fly": True, "pose": dict(last, age=round(t - stamp, 2))},
        {"name": "D2", "team": "defense", "fly": True, "pose": dict(cam_pose, age=0.05),
         "seen": {"age": 0.1, "dets": [{"cam": cam_a.tolist(), "conf": 0.7},
                                        {"cam": cam_e.tolist(), "conf": 0.6}]}},
        {"name": "E", "team": "defense", "fly": True,
         "pose": {"tag": 10, "xyz": (FLIP @ e_room).tolist(), "yaw": 0.0, "age": 0.0}}]}
    tracker.update(msg)
    est = tracker.estimate("A")
    assert est["source"] == "seen" and not est["predicted"], est
    assert np.allclose(est["xyz"], a_true, atol=1e-3), (est, a_true)
    assert tracker.sightings == {"used": 1, "unmatched": 0, "teammate": 1}, tracker.sightings
    a = tracker.annotate(json.loads(json.dumps(msg)))["drones"][0]
    assert a["pose"]["source"] == "seen" and a["pose"]["age"] == 0.0, a
    # A sighting far from every attacker's prediction is not forced onto one.
    far = np.diag([1.0, -1.0, -1.0]) @ (np.array([-2.0, 0.5, 1.0]) - [0, 0, 4.0])
    msg["t"], msg["drones"][1]["seen"]["dets"] = 105.3, [{"cam": far.tolist(), "conf": 0.5}]
    tracker.update(msg)
    assert tracker.sightings["unmatched"] == 1, tracker.sightings
    print("SELFTEST PASSED")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", type=int, default=9100, help="port drone_gui publishes to")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--forward", nargs="+", default=["127.0.0.1:9101"], help="host:port of each consumer")
    ap.add_argument("--team", default="attack", help="team to track")
    ap.add_argument("--map", type=Path, default=MAP_PATH)
    ap.add_argument("--no-fill-pose", action="store_true", help="only add 'track', keep 'pose' as published")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return 0
    try:
        return relay(args)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
