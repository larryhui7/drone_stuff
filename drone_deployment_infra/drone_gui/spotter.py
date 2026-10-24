"""Spot drones in the defenders' cameras, for drone_telemetry/track.py (drone_gui --spot).

One detector (ultralytics; by default the YOLO-World model with the baked "drone" vocabulary
from vision_and_mapping/teleop_detect.py --bake) runs in a background thread over the newest
frame of every drone on the spotting team, batched into one predict call. Each drone box is
turned into a camera-frame position with that drone's calibration (track.box_to_camera:
intrinsics + distortion, range from the Tello's width) and stored on the drone as
`seen = {"stamp", "dets": [{"cam": [x, y, z], "conf"}]}`; telemetry publishes it, and
track.py places it in the room with the drone's own pose.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from tello_unit import DD_ROOT

sys.path.insert(0, str(DD_ROOT / "drone_telemetry"))
from track import box_to_camera  # noqa: E402

DEFAULT_MODEL = DD_ROOT.parent / "vision_and_mapping" / "models" / "tello_world.pt"
DRONE_LABELS = {"drone", "quadcopter"}   # teleop_detect's vocabulary names for a Tello
IDLE_S = 0.02


class Spotter:
    def __init__(self, fleet, model_path: Path = DEFAULT_MODEL, conf: float = 0.3, imgsz: int = 640,
                 team: str = "defense") -> None:
        import torch
        from ultralytics import YOLO

        if not Path(model_path).exists():
            raise SystemExit(f"--spot: no model at {model_path}. Bake one with "
                             "`python vision_and_mapping/teleop_detect.py --bake`, or pass a Tello detector.")
        self.fleet, self.conf, self.imgsz, self.team = fleet, conf, imgsz, team
        self.model = YOLO(str(model_path))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        labels = set(self.model.names.values())
        if not labels & DRONE_LABELS:
            raise SystemExit(f"--spot: {Path(model_path).name} has no drone class ({sorted(labels)[:8]}...)")
        self.fps = 0.0                     # batches per second
        self.stop = threading.Event()
        print(f"[spot] {Path(model_path).name} on {self.device}, team {team}, conf {conf}", flush=True)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        done: dict[str, float] = {}        # drone name -> frame_t already processed
        while not self.stop.is_set():
            batch = []
            for slot, drone in enumerate(self.fleet.all(), 1):
                frame, stamp = drone.frame, drone.frame_t
                if (frame is not None and stamp > done.get(drone.name, 0.0)
                        and self.fleet.team(drone, slot) == self.team):
                    batch.append((drone, frame, stamp))
            if not batch:
                time.sleep(IDLE_S)
                continue
            t0 = time.monotonic()
            try:
                results = self.model.predict([f for _, f, _ in batch], imgsz=self.imgsz, conf=self.conf,
                                             device=self.device, verbose=False)
            except Exception as exc:  # noqa: BLE001  keep spotting through a bad frame
                print(f"[spot] detector error: {exc}", flush=True)
                time.sleep(0.5)
                continue
            for (drone, frame, stamp), result in zip(batch, results):
                done[drone.name] = stamp
                dets = []
                for xyxy, cls, p in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                        result.boxes.conf.tolist()):
                    if result.names[int(cls)] not in DRONE_LABELS:
                        continue
                    cam = box_to_camera(xyxy, drone.K, drone.dist, image_width=frame.shape[1])
                    if cam is not None:
                        dets.append({"cam": [round(float(v), 3) for v in cam], "conf": round(float(p), 2)})
                drone.seen = {"stamp": stamp, "dets": dets}
            self.fps = 0.8 * self.fps + 0.2 / max(time.monotonic() - t0, 1e-3)

    def close(self) -> None:
        self.stop.set()
