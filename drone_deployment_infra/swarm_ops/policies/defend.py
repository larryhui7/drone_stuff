"""defend: our trained defender policy, run by the umbrella on its drones.

  python swarm_policies.py --run formation:4,5 --run defend:6,7            # dry-run first
  python swarm_policies.py --run formation:4,5 --run defend:6,7 --fly

Every tick, for each of its drones:
  1. OWN POSITION (AprilTags): the drone's video, over its own dongle, goes through
     core/positional_kalman_filter.FilteredPositioner -- the localizer + Kalman filter the
     simulator trains with -- using config/map.yaml, config/tags.yaml and the drone's
     calibration config/camera/tello_<n>.yaml (default.yaml if it has none).
  2. ATTACKERS (telemetry bus): every "formation/drone<N>" topic that is not one of ours.
     Each attacker's last BUFFER_S of tag fixes is kept. When its fix goes stale (fix_age >
     STALE_S) it is coasted from its last fix: COAST_MODE "stop" (default) slows it to a
     halt within ~STOP_TAU_S, as swarm_formation does when it loses its tag (freeze, recover
     in place); "straight" keeps its fitted velocity. A coast is dropped after MAX_COAST_S.
  3. MODEL: observation -> trained network -> rc (see the MODEL section: the observation
     layout and action scaling there MUST match how the policy was trained).
  4. SAFETY: rc 0 0 0 0 (the Tello holds in place on its own) when the drone has no tag fix,
     the model is missing or fails, or another drone is closer than STOP_DIST_M; sticks are
     clipped to MAX_RC and height kept within MIN_CM..MAX_CM. The runner still lands a drone
     that loses its status stream or battery.

Publishes "defend/drone<N>": its position, mode and the attackers it sees.
Room frame everywhere: origin tag 10's centre, x right, y up, z out of the wall; m, m/s.

  python policies/defend.py --selftest      # offline: fake bus and drones, no network
"""
import math
import os
import socket
import sys
import threading
import time
from collections import deque

import numpy as np

try:
    from .base import Policy, clip
except ImportError:                       # run as a script for --selftest
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from policies.base import Policy, clip

HERE = os.path.dirname(os.path.abspath(__file__))
DD_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# ======================= SETTINGS =======================
MODEL_PATH = os.path.join(HERE, "defend_model.pt")   # TorchScript export of the trained policy
MAX_RC = 40                  # |rc| cap on every stick
MAX_RC_Z = 30
MIN_CM, MAX_CM = 60, 220     # height band (Tello tof/h): outside it, climb/sink regardless of the model
STOP_DIST_M = 0.5            # any other drone closer than this: hold
N_ATTACKERS = 4              # attackers in the observation (nearest first, zero-padded)
ATTACKER_IDS = range(1, 17)  # drone numbers probed for "formation/drone<N>"

BUS_MAX_AGE_S = 0.5          # a topic must have been published this recently
STALE_S = 0.5                # attacker fix older than this: coast
BUFFER_S = 60.0              # fixes kept per attacker
FIT_S = 1.0                  # velocity fitted over the last this-many seconds of fixes
MIN_FIT_SPAN_S = 0.3
OUTLIER_M = 0.3              # a fix this far off the line fit (a pose flip) is dropped
MAX_SPEED = 3.0              # m/s cap on a fitted velocity
COAST_MODE = "stop"          # "stop": decay to a halt (formation freezes when lost); "straight"
STOP_TAU_S = 0.5             # "stop" decay time constant
MAX_COAST_S = 10.0           # coast this long at most, then the attacker is unknown
FLOOR_Y = -1.22              # room-frame floor (tag 10's centre is 1.22 m up)
LOST_AFTER_S = 0.4           # own drone: no tag fix this long = lost (the simulator's value)
FLYING_STATES = ("CLIMB", "SEARCH", "LOCKED", "FACE10", "APPROACH", "HOLDING")
VIDEO_CMDS = ("streamon", "streamoff")
HOLD = (0, 0, 0, 0)
# ========================================================


# ----------------------- attackers -----------------------
class AttackerTrack:
    """One attacker: its fixes (time, room xyz) and the coast used when they stop."""

    def __init__(self):
        self.fixes = deque()
        self.flying = False
        self._vel = None

    def add(self, t, p):
        if self.fixes and t - self.fixes[-1][0] < 0.02:
            return                                   # the same fix, republished
        self.fixes.append((t, np.asarray(p, float)))
        self._vel = None
        while t - self.fixes[0][0] > BUFFER_S:
            self.fixes.popleft()

    def velocity(self):
        """Least-squares line over the last FIT_S of fixes, refit once without outliers."""
        if self._vel is not None:
            return self._vel
        t_end = self.fixes[-1][0]
        recent = [(t, p) for t, p in self.fixes if t_end - t <= FIT_S]
        vel = np.zeros(3)
        if len(recent) >= 3 and recent[-1][0] - recent[0][0] >= MIN_FIT_SPAN_S:
            ts = np.array([t for t, _ in recent]) - t_end
            ps = np.array([p for _, p in recent])
            for _ in range(2):
                coef = np.linalg.lstsq(np.stack([ts, np.ones_like(ts)], axis=1), ps, rcond=None)[0]
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

    def estimate(self, now):
        """{"xyz", "vel", "coasting", "coast_s"} or None (no fix yet, landed, or coasted too long)."""
        if not self.fixes or not self.flying:
            return None
        t_fix, p_fix = self.fixes[-1]
        coast = max(now - t_fix, 0.0)
        if coast <= STALE_S:
            return {"xyz": p_fix, "vel": self.velocity(), "coasting": False, "coast_s": 0.0}
        if coast > MAX_COAST_S:
            return None
        v = self.velocity()
        if COAST_MODE == "straight":
            p, v_now = p_fix + v * coast, v
        else:                                        # "stop": v decays as exp(-coast / tau)
            decay = math.exp(-coast / STOP_TAU_S)
            p, v_now = p_fix + v * STOP_TAU_S * (1.0 - decay), v * decay
        if p[1] < FLOOR_Y:
            p = np.array([p[0], FLOOR_Y, p[2]])
        return {"xyz": p, "vel": v_now, "coasting": True, "coast_s": coast}


class AttackerTracker:
    """Reads "formation/drone<N>" from the bus each tick and keeps a track per attacker."""

    def __init__(self, bus, ours):
        self.bus, self.ours = bus, set(ours)
        self.tracks = {}

    def update(self, now):
        for n in ATTACKER_IDS:
            if n in self.ours:
                continue
            hit = self.bus.latest(f"formation/drone{n}", BUS_MAX_AGE_S)
            if hit is None:                          # quiet publisher: an existing track keeps coasting
                continue
            data, pub_age = hit
            if not isinstance(data, dict) or not _finite(pub_age) or pub_age < 0:
                continue
            track = self.tracks.setdefault(n, AttackerTrack())
            track.flying = data.get("state") in FLYING_STATES and data.get("flying", True) is True
            p, fix_age = data.get("p"), data.get("fix_age")
            if p is None or not _finite(fix_age) or fix_age < 0 or fix_age > STALE_S:
                continue                             # no fix yet, or a stale one (no new information)
            if len(p) != 3 or not all(_finite(v) for v in p):
                continue
            track.add(now - pub_age - fix_age, p)

    def estimates(self, now):
        out = {}
        for n, track in self.tracks.items():
            est = track.estimate(now)
            if est is not None:
                out[n] = est
        return out


# ----------------------- our drones -----------------------
def raw_send(d, cmd, timeout=5.0):
    """A video command on the drone's own command socket (the runner's command filter only
    knows flight commands). Never anything but VIDEO_CMDS."""
    assert cmd in VIDEO_CMDS, cmd
    with d.lock:
        d.sock.setblocking(False)
        try:
            while True:
                d.sock.recv(1024)
        except (BlockingIOError, OSError):
            pass
        d.sock.settimeout(timeout)
        try:
            d.sock.sendto(cmd.encode(), ("192.168.10.1", 8889))
            return d.sock.recv(1024).decode(errors="ignore").strip().lower()
        except (socket.timeout, OSError):
            return "timeout"


def heading_deg(R):
    """Camera heading in the room: 0 = facing the tag-10 wall (-z), + = turned toward +x."""
    f = np.asarray(R, float)[:, 2]
    return math.degrees(math.atan2(f[0], -f[2]))


class OwnVision:
    """One of our drones: video thread + localizer thread -> its Kalman position."""

    def __init__(self, d):
        for sub in ("core", "cal_and_map"):
            path = os.path.join(DD_ROOT, sub)
            if path not in sys.path:
                sys.path.insert(0, path)
        from positional_kalman_filter import FilteredPositioner
        self.d = d
        self.fpos = FilteredPositioner(f"tello_{d.num}")
        self.frame, self.frame_t, self.frame_id = None, 0.0, 0
        self.est, self.est_t, self.yaw = None, 0.0, None
        self.video_on = False
        self.stop = threading.Event()

    def start(self):
        for target in (self._stream_on, self._video_loop, self._locate_loop):
            threading.Thread(target=target, daemon=True).start()

    def _stream_on(self):
        t0 = time.time()
        while not getattr(self.d, "st", None) and time.time() - t0 < 120:   # until preflight is done
            time.sleep(0.2)
        for _ in range(3):
            if raw_send(self.d, "streamon") == "ok":
                self.video_on = True
                self.d.log("defend: video on")
                return
        self.d.log("defend: streamon got no ok -> no tag position, this drone will hold")

    def _video_loop(self):
        import av
        vs = self.d._sock(11111)                    # bound to this drone's dongle
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
            except Exception:  # noqa: BLE001  H.264 errors before each keyframe are normal
                pass
        vs.close()

    def _locate_loop(self):
        last = 0
        while not self.stop.is_set():
            if self.frame_id == last or self.frame is None:
                time.sleep(0.005)
                continue
            last, frame, t = self.frame_id, self.frame, self.frame_t
            est = self.fpos.locate(frame, t=t)
            if est["raw"] is not None:
                self.yaw = heading_deg(est["raw"]["R"])
            self.est, self.est_t = est, t

    def state(self, now):
        est = self.est
        if est is None or est["xyz"] is None or self.yaw is None:
            return None
        fix_age = est["fix_age"] + (now - self.est_t)
        return {"xyz": np.asarray(est["xyz"], float), "vel": np.asarray(est["vel"], float),
                "yaw": self.yaw, "fix_age": fix_age, "lost": fix_age > LOST_AFTER_S}


# ----------------------- MODEL -----------------------
# The observation and action below are a reasonable layout, NOT necessarily the one the
# policy was trained on. Make observation() and to_rc() reproduce the training env exactly
# (order, frames, units, normalisation, VecNormalize stats) before flying with a model.
def observation(me, attackers, d):
    """Per drone: own state + the N_ATTACKERS nearest attackers relative to it."""
    yaw = math.radians(me["yaw"])
    own = [*me["xyz"], *me["vel"], math.sin(yaw), math.cos(yaw), float(me["lost"]),
           min(me["fix_age"], 2.0), (d.height_cm() or 0.0) / 100.0]
    near = sorted(attackers.values(), key=lambda a: float(np.linalg.norm(a["xyz"] - me["xyz"])))
    rows = []
    for a in near[:N_ATTACKERS]:
        rows += [*(a["xyz"] - me["xyz"]), *a["vel"], float(a["coasting"]), min(a["coast_s"], MAX_COAST_S)]
    rows += [0.0] * (8 * N_ATTACKERS - len(rows))
    return np.asarray(own + rows, np.float32)


def to_rc(action):
    """Network output (4 values in [-1, 1]: right, forward, up, yaw) -> rc sticks."""
    a = np.clip(np.asarray(action, float).reshape(-1)[:4], -1.0, 1.0)
    return (a[0] * MAX_RC, a[1] * MAX_RC, a[2] * MAX_RC_Z, a[3] * MAX_RC)


class Model:
    def __init__(self, path=MODEL_PATH):
        self.net, self.error = None, None
        if not os.path.exists(path):
            self.error = f"no model at {path}"
            return
        try:
            import torch
            self.torch = torch
            self.net = torch.jit.load(path, map_location="cpu").eval()
        except Exception as exc:  # noqa: BLE001
            self.error = f"model load failed: {exc!r}"

    def act(self, obs):
        with self.torch.no_grad():
            out = self.net(self.torch.as_tensor(obs).unsqueeze(0))
        return out[0].numpy() if hasattr(out, "numpy") else np.asarray(out)[0]


# ----------------------- policy -----------------------
def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class DefendPolicy(Policy):
    name = "defend"

    def __init__(self, model_path=MODEL_PATH, vision=True):
        self.model_path, self.use_vision = model_path, vision

    def setup(self, drones):
        super().setup(drones)
        self.model = Model(self.model_path)
        self.attackers = AttackerTracker(self.bus, [d.num for d in self.drones])
        self.vision, self.mode, self._tick = {}, {}, None
        for d in self.drones:
            if self.model.error:
                d.log(f"defend: {self.model.error} -> it will hold in place")
            if not self.use_vision:
                continue
            try:
                self.vision[d.num] = OwnVision(d)
                self.vision[d.num].start()
            except Exception as exc:  # noqa: BLE001  missing cv2/av/pupil/map: hold, don't crash the runner
                d.log(f"defend: no tag localization ({exc!r}) -> it will hold in place")

    def _picture(self, now):
        """Attackers + every one of our drones' state, computed once per tick."""
        if self._tick is None or now - self._tick[0] > 0.02:
            self.attackers.update(now)
            ours = {n: v.state(now) for n, v in self.vision.items()}
            self._tick = (now, self.attackers.estimates(now), ours)
        return self._tick[1], self._tick[2]

    def _hold(self, d, why, me=None, attackers=()):
        self.mode[d.num] = why
        self._publish(d, me, attackers)
        return HOLD

    def _publish(self, d, me, attackers):
        self.bus.publish(f"defend/{d.name}", {
            "mode": self.mode.get(d.num),
            "p": None if me is None else [round(float(v), 3) for v in me["xyz"]],
            "lost": None if me is None else me["lost"],
            "attackers": {f"drone{n}": {"p": [round(float(v), 3) for v in a["xyz"]],
                                        "coasting": a["coasting"]} for n, a in dict(attackers).items()}})

    def step(self, d, t):
        now = time.time()
        attackers, ours = self._picture(now)
        me = ours.get(d.num)
        if me is None:
            return self._hold(d, "no tag fix yet", None, attackers)
        if me["lost"]:
            return self._hold(d, "tag lost", me, attackers)
        others = [a["xyz"] for a in attackers.values()] + \
                 [s["xyz"] for n, s in ours.items() if n != d.num and s is not None and not s["lost"]]
        if any(np.linalg.norm(p - me["xyz"]) < STOP_DIST_M for p in others):
            return self._hold(d, "too close to a drone", me, attackers)
        if self.model.net is None:
            return self._hold(d, self.model.error, me, attackers)
        try:
            a, b, c, yaw = to_rc(self.model.act(observation(me, attackers, d)))
        except Exception as exc:  # noqa: BLE001  a failing model must not fly the drone
            return self._hold(d, f"model error {exc!r}", me, attackers)
        h = d.height_cm()
        if h is not None and h > MAX_CM:
            c = min(c, -15)
        elif h is not None and h < MIN_CM:
            c = max(c, 15)
        self.mode[d.num] = "model" if attackers else "model (no attackers seen)"
        self._publish(d, me, attackers)
        return (clip(a, MAX_RC), clip(b, MAX_RC), clip(c, MAX_RC_Z), clip(yaw, MAX_RC))

    def describe(self, d):
        me = self._tick[2].get(d.num) if self._tick else None
        where = "no fix" if me is None else "xyz " + ",".join(f"{v:+.2f}" for v in me["xyz"])
        n_att = len(self._tick[1]) if self._tick else 0
        return f"{self.name} {self.mode.get(d.num, 'starting')} | {where} | attackers {n_att}"


# ----------------------- offline self-test -----------------------
def _selftest():
    class Bus:                                      # the bus calls defend.py makes
        def __init__(self):
            self.topics = {}

        def publish(self, topic, data, t=None):
            self.topics[topic] = (data, time.time() if t is None else t)

        def latest(self, topic, max_age):
            if topic not in self.topics:
                return None
            data, t = self.topics[topic]
            age = time.time() - t
            return (data, age) if age <= max_age else None

    class Drone:
        def __init__(self, num):
            self.num, self.name, self.h = num, f"drone{num}", 120.0

        def height_cm(self):
            return self.h

        def log(self, msg):
            print(f"[{self.name}] {msg}")

    class Vision:                                   # stands in for OwnVision
        def __init__(self, xyz, lost=False):
            self.s = {"xyz": np.asarray(xyz, float), "vel": np.zeros(3), "yaw": 0.0,
                      "fix_age": 0.0, "lost": lost}

        def state(self, now):
            return self.s

    class FakeModel:
        net, error = True, None

        def act(self, obs):
            assert obs.shape == (11 + 8 * N_ATTACKERS,), obs.shape
            return np.array([0.5, -2.0, 0.1, 0.0])   # -2 must clip to -1

    real_time = time.time
    clock = [1000.0]
    time.time = lambda: clock[0]
    try:
        bus = Bus()
        p = DefendPolicy(model_path="/nonexistent", vision=False)
        p.bus = bus
        d6, d7 = Drone(6), Drone(7)
        p.setup([d6, d7])
        assert p.step(d6, 0) == HOLD and p.mode[6] == "no tag fix yet"
        p.vision = {6: Vision([0.0, 0.0, 2.0]), 7: Vision([1.5, 0.0, 2.0])}
        p._tick = None
        assert p.step(d6, 0) == HOLD and p.mode[6].startswith("no model")

        # attacker 4 flies +x at 0.5 m/s for 2 s, then loses its tag (fix_age grows)
        p.model = FakeModel()
        for i in range(31):
            clock[0] = 1000.0 + 0.1 * i
            lost = i > 20
            fix_t = 1000.0 + 0.1 * min(i, 20)
            bus.publish("formation/drone4", {"state": "APPROACH", "flying": True,
                                             "p": [-1.0 + 0.5 * (fix_t - 1000.0), 0.0, 3.0],
                                             "fix_age": round(clock[0] - fix_t, 3) if lost else 0.0})
            p._tick = None
            rc = p.step(d6, 0)
        est = p.attackers.estimates(clock[0])[4]
        assert est["coasting"] and abs(est["coast_s"] - 1.0) < 1e-6, est
        expect_x = 0.0 + 0.5 * STOP_TAU_S * (1 - math.exp(-1.0 / STOP_TAU_S))      # "stop" coast
        assert abs(est["xyz"][0] - expect_x) < 1e-6, (est, expect_x)
        assert rc == (20, -40, 3, 0), rc                                          # scaled + clipped
        assert bus.latest("defend/drone6", 1)[0]["attackers"]["drone4"]["coasting"] is True
        # our own drones are never tracked as attackers
        bus.publish("formation/drone7", {"state": "HOLDING", "flying": True, "p": [9, 9, 9], "fix_age": 0.0})
        p._tick = None
        p.step(d6, 0)
        assert 7 not in p.attackers.tracks

        # height band overrides the model
        d6.h = 250.0
        p._tick = None
        assert p.step(d6, 0)[2] <= -15
        d6.h = 120.0
        # too close to a drone: hold
        p.vision[6] = Vision([1.2, 0.0, 2.0])               # 0.3 m from teammate 7
        p._tick = None
        assert p.step(d6, 0) == HOLD and p.mode[6] == "too close to a drone"
        # own tag lost: hold
        p.vision[6] = Vision([0.0, 0.0, 2.0], lost=True)
        p._tick = None
        assert p.step(d6, 0) == HOLD and p.mode[6] == "tag lost"
        # model raising: hold
        p.vision[6] = Vision([0.0, 0.0, 2.0])

        class Broken(FakeModel):
            def act(self, obs):
                raise RuntimeError("boom")
        p.model = Broken()
        p._tick = None
        assert p.step(d6, 0) == HOLD and p.mode[6].startswith("model error")
        # attacker landed: no longer an attacker
        bus.publish("formation/drone4", {"state": "LANDED", "flying": False, "p": [0, 0, 3], "fix_age": 0.0})
        p._tick = None
        p.step(d6, 0)
        assert 4 not in p.attackers.estimates(clock[0])
        # coasting stops after MAX_COAST_S
        clock[0] += MAX_COAST_S + 1
        assert p.attackers.estimates(clock[0]) == {}
    finally:
        time.time = real_time
    print("SELFTEST PASSED")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
