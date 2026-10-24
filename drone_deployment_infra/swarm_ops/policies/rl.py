"""rl: fly a trained policy on its drones, fed by sensing.py (AprilTag positions + attacker tracks).

  python swarm_policies.py --run rl:5,6,7            # dry-run: positions + attackers printed
  python swarm_policies.py --run rl:5,6,7 --fly --duration 60

Settings come from policies/rl_config.yaml (swarm_policies builds policies without
arguments): the adapter file, the attackers' telemetry port and team, the spot model.

The adapter (copy policies/rl_adapter_template.py) is the trained policy's side:
  setup(drones)          once, before takeoff: load the model
  act(d, t, snap)        every tick for each of its drones -> (a, b, c, d) rc, or None = land it
`snap` is Sensing.snapshot(): every one of our drones' position and every attacker's
estimate (see sensing.py). This file only adds the safety around it: a drone without a
position, or an adapter that raises or returns junk, gets rc 0 0 0 0 (the Tello holds
in place on its own) instead of a command.
"""
import importlib.util
import os

import yaml

from .base import Policy

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "rl_config.yaml")
HOLD = (0, 0, 0, 0)


def load_adapter(path):
    path = path if os.path.isabs(path) else os.path.join(HERE, path)
    spec = importlib.util.spec_from_file_location("rl_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "act"):
        raise SystemExit(f"{path}: an rl adapter needs act(d, t, snap)")
    return module


class RLPolicy(Policy):
    name = "rl"

    def __init__(self, config=CONFIG):
        with open(config) as f:
            self.cfg = yaml.safe_load(f) or {}
        self.adapter = load_adapter(self.cfg.get("adapter", "rl_adapter_template.py"))
        self.held = {}               # drone num -> why it is being held (printed once per change)

    def setup(self, drones):
        super().setup(drones)
        import sensing                              # swarm_ops/sensing.py (numpy, cv2, av, ...)
        spot = self.cfg.get("spot_model")
        if spot and not os.path.isabs(spot):
            spot = os.path.join(HERE, spot)
        self.sensing = sensing.Sensing(self.drones, telemetry_port=self.cfg.get("telemetry_port", 9100),
                                       attacker_team=self.cfg.get("attacker_team", "attack"),
                                       spot_model=spot)
        self.sensing.start()                        # video comes on after each drone's preflight
        if hasattr(self.adapter, "setup"):
            self.adapter.setup(self.drones)

    def _hold(self, d, why):
        if self.held.get(d.num) != why:
            d.log(f"rl: holding ({why})")
        self.held[d.num] = why
        return HOLD

    def step(self, d, t):
        snap = self.sensing.snapshot()
        me = snap["defenders"][d.num]
        if me["xyz"] is None or me["lost"]:
            return self._hold(d, "no tag fix")
        try:
            rc = self.adapter.act(d, t, snap)
        except Exception as exc:  # noqa: BLE001  a crashing model must not fly the drone
            return self._hold(d, f"adapter error: {exc!r}")
        if rc is None:
            return None
        try:
            rc = tuple(int(round(max(-100, min(100, float(v))))) for v in rc)
        except (TypeError, ValueError):
            return self._hold(d, f"adapter returned {rc!r}")
        if len(rc) != 4:
            return self._hold(d, f"adapter returned {len(rc)} values")
        self.held.pop(d.num, None)
        return rc

    def describe(self, d):
        snap = self.sensing.snapshot()
        me = snap["defenders"][d.num]
        where = ("no fix" if me["xyz"] is None else
                 f"xyz {me['xyz'][0]:+.2f},{me['xyz'][1]:+.2f},{me['xyz'][2]:+.2f}"
                 + (" LOST" if me["lost"] else f" tags {me['tags']}"))
        att = snap["attackers"]
        coasting = sum(1 for e in att.values() if e and e["predicted"])
        feed = "no feed" if snap["feed_age"] is None else f"feed {snap['feed_age']:.1f}s"
        return f"rl {where} | attackers {len(att)} ({coasting} coasting), {feed}"
