"""Template adapter: connects a trained defender policy to `swarm_policies.py --run rl:<drones>`.

Copy this file, fill in setup() and act(), and set `adapter:` in rl_config.yaml. As shipped
it returns rc 0 0 0 0 (hold in place), so a dry-run or a short flight shows the sensing
without the model.

snap (sensing.Sensing.snapshot(), the same object for every drone in one tick):
  snap["t"]                    this laptop's time.time()
  snap["defenders"][num]       OUR drones, by swarm_policies drone number:
      xyz [x, y, z] m, vel [vx, vy, vz] m/s   Kalman estimate (positional_kalman_filter.py,
                                              the filter the simulator trains with), or None
      yaw   deg, 0 = facing the tag-10 wall, + = turned toward +x (last tag fix)
      fix_age s since the last tag fix,  lost  True after 0.4 s without one
      tags  mapped tag ids in the last fix
  snap["attackers"][name]      THEIR drones, by their telemetry name (None until first fix):
      xyz, vel                 room frame; while lost, coasted in a straight line
      predicted                True = coasting (no fresh fix for 0.5 s)
      coast_s, fix_age         s since the last real fix
      source                   "tag" (their telemetry) or "seen" (our camera spotted it)
  snap["feed_age"]             s since their last telemetry datagram (None = never)

Room frame: origin tag 10's centre, x right (facing the wall), y up, z out of the wall.
Height: y is relative to tag 10's centre (floor is about -1.22); d.height_cm() is the
Tello's own height sensor.

rc out: (a, b, c, d) = (left/right, forward/back, up/down, yaw), integers -100..100, in the
drone's own axes (forward = where its camera looks). Return None to land the drone.
"""
import numpy as np  # noqa: F401  (most adapters need it)

MODEL = None


def setup(drones):
    """Once, before takeoff. Load the model here (it runs on this laptop)."""
    global MODEL
    MODEL = None          # e.g. torch.jit.load(...) / PPO.load(...)


def act(d, t, snap):
    """rc for drone d (d.num, d.height_cm(), d.battery()), t = s since its takeoff."""
    me = snap["defenders"][d.num]
    attackers = [a for a in snap["attackers"].values() if a is not None]
    _ = me, attackers     # build the policy's observation from these, run MODEL, map its action
    return (0, 0, 0, 0)
