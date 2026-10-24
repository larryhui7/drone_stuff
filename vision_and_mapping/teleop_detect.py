"""Fly a Tello from the keyboard while its camera feed runs a light object detector.

This FLIES the drone. Clear the area, keep the battery above ~20%, and be ready to land.

    python -u vision_and_mapping/teleop_detect.py                 # default speed 40 (rc units)
    python -u vision_and_mapping/teleop_detect.py --speed 25 --conf 0.35
    python -u vision_and_mapping/teleop_detect.py --no-detect     # video + control only

Join the drone's TELLO-XXXXXX Wi-Fi first (as for read_tags.py --tello). Click the video
window so it has keyboard focus, then:

    t           take off               space    land
    w / s       forward / back         a / d    left / right
    q / e       up / down              z / c    turn left / right
    backspace   EMERGENCY motor stop (drops the drone; only for a runaway)
    esc         land and quit (closing the window does the same)

A key moves the drone while it is held (OpenCV sees key repeats, so motion continues
HOLD_S after the last repeat); with no key held the drone gets rc 0 0 0 0 and hovers.
Ctrl+C, an error or closing the window lands it. If the link drops in flight (no state
packet for STATE_STALE_S; FLYING.md's motor-interference drops) the keys are ignored and the
drone lands as soon as the link is back, like the flight engine's LAND_ON_LOSS.

Detection runs on its own thread so control never waits for it. The default model is
YOLO-World with a vocabulary baked in by --bake (person, drone, common room objects), so
Tello EDUs show up as "drone" without a custom-trained model. Without the baked file it
falls back to COCO yolo11n (people and objects, no drones). --model takes any ultralytics
detector, e.g. a yolo11n fine-tuned on Tello images.

    # once, to bake the vocabulary (needs CLIP: pip install git+https://github.com/ultralytics/CLIP.git)
    python -u vision_and_mapping/teleop_detect.py --bake
"""

from __future__ import annotations

import argparse
import queue
import socket
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent.parent / "drone_deployment_infra" / "cal_and_map"))

from tello_calibration.video.stream import VideoReceiver  # noqa: E402

MODELS = HERE.parent / "models"
BAKED = MODELS / "tello_world.pt"
FALLBACK = MODELS / "yolo11n.pt"
VOCAB = ["person", "drone", "quadcopter", "chair", "table", "laptop", "cell phone", "bottle",
         "cup", "backpack", "bag", "monitor", "keyboard", "book", "box", "ball", "door"]
DRONE_LABELS = {"drone", "quadcopter", "airplane", "kite"}  # shown as "drone"

TELLO = ("192.168.10.1", 8889)
RC_PERIOD_S = 0.05        # 20 Hz rc stream
HOLD_S = 0.6              # a key keeps acting this long after its last (repeat) event
STATE_STALE_S = 1.0       # no state packet for this long -> link counted as lost
LAND_ON_LOSS = True       # as core/tello_dual_video.py: a drop in flight lands on reconnect

# key -> (axis index in rc a b c d, sign); a = left/right, b = fwd/back, c = up/down, d = yaw
MOVES = {ord("a"): (0, -1), ord("d"): (0, 1), ord("w"): (1, 1), ord("s"): (1, -1),
         ord("q"): (2, 1), ord("e"): (2, -1), ord("z"): (3, -1), ord("c"): (3, 1)}
KEY_TAKEOFF, KEY_LAND, KEY_QUIT = ord("t"), ord(" "), 27
KEY_EMERGENCY = {8, 127}  # backspace / delete


class Tello:
    """Command socket (replies read on a thread) plus the 8890 state broadcast."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", 8889))
        self.sock.settimeout(0.5)
        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_sock.bind(("", 8890))
        self.state_sock.settimeout(0.5)
        self.replies: queue.Queue[str] = queue.Queue()
        self.state: dict[str, str] = {}
        self.state_t = 0.0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        for target in (self._read_replies, self._read_state):
            threading.Thread(target=target, daemon=True).start()

    def _read_replies(self) -> None:
        while not self.stop.is_set():
            try:
                data, _ = self.sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                return
            self.replies.put(data.decode(errors="replace").strip().lower())

    def _read_state(self) -> None:
        while not self.stop.is_set():
            try:
                data, _ = self.state_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                return
            fields = data.decode(errors="replace").strip().strip(";").split(";")
            self.state = dict(f.split(":", 1) for f in fields if ":" in f)
            self.state_t = time.monotonic()

    def ask(self, cmd: str, timeout: float = 5.0) -> str | None:
        """Send a command and wait for its reply (None on timeout)."""
        with self.lock:
            while not self.replies.empty():
                self.replies.get_nowait()
            self.sock.sendto(cmd.encode(), TELLO)
            try:
                return self.replies.get(timeout=timeout)
            except queue.Empty:
                return None

    def send(self, cmd: str) -> None:
        """Fire and forget (rc, emergency): never waits."""
        self.sock.sendto(cmd.encode(), TELLO)

    def ask_async(self, cmd: str, timeout: float, done) -> None:
        threading.Thread(target=lambda: done(cmd, self.ask(cmd, timeout)), daemon=True).start()

    def land_now(self) -> None:
        """Land on exit: retries without the lock, so it works even mid-takeoff."""
        self.send("rc 0 0 0 0")
        for _ in range(6):
            self.send("land")
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                try:
                    if self.replies.get(timeout=0.2) == "ok":
                        return
                except queue.Empty:
                    pass

    def close(self) -> None:
        self.stop.set()
        self.sock.close()
        self.state_sock.close()


class Detector:
    """Runs the model on the newest frame in a background thread; keeps the last boxes."""

    def __init__(self, model_path: Path | None, conf: float, imgsz: int) -> None:
        import torch
        from ultralytics import YOLO

        path = model_path or (BAKED if BAKED.exists() else FALLBACK)
        if path == FALLBACK:
            print(f"[teleop] {BAKED.name} not found (run --bake): using COCO {FALLBACK.name}, "
                  "which has no drone class", flush=True)
        MODELS.mkdir(exist_ok=True)
        self.model = YOLO(str(path))
        self.device = "mps" if torch.backends.mps.is_available() else \
            "cuda" if torch.cuda.is_available() else "cpu"
        self.conf, self.imgsz = conf, imgsz
        self.boxes: list[tuple[int, int, int, int, str, float]] = []
        self.fps = 0.0
        self.frame: np.ndarray | None = None
        self.new = threading.Event()
        self.stop = threading.Event()
        print(f"[teleop] detector {path.name} on {self.device}", flush=True)
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.new.set()

    def _run(self) -> None:
        while not self.stop.is_set():
            if not self.new.wait(0.2):
                continue
            self.new.clear()
            t0 = time.monotonic()
            result = self.model.predict(self.frame, imgsz=self.imgsz, conf=self.conf,
                                        device=self.device, verbose=False)[0]
            names = result.names
            self.boxes = [(*map(int, xyxy), names[int(c)], float(p)) for xyxy, c, p in
                          zip(result.boxes.xyxy.tolist(), result.boxes.cls, result.boxes.conf)]
            self.fps = 0.8 * self.fps + 0.2 / max(time.monotonic() - t0, 1e-3)


def draw(frame: np.ndarray, boxes, hud: list[str], warn: str | None) -> np.ndarray:
    out = frame.copy()
    for x1, y1, x2, y2, label, p in boxes:
        if label in DRONE_LABELS:
            label, color = "drone", (0, 0, 255)
        else:
            color = (0, 200, 0) if label == "person" else (255, 160, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{label} {p:.2f}", (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2)
    for i, line in enumerate(hud):
        cv2.putText(out, line, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if warn:
        cv2.putText(out, warn, (10, out.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
    return out


def bake(classes: list[str]) -> None:
    """Bake a YOLO-World vocabulary into a model file, so flying needs no CLIP."""
    from ultralytics import YOLO

    MODELS.mkdir(exist_ok=True)
    model = YOLO(str(MODELS / "yolov8s-worldv2.pt"))
    model.set_classes(classes)
    model.model.clip_model = None   # the text encoder is only needed to bake (~300 MB)
    model.save(str(BAKED))
    print(f"[teleop] baked {len(classes)} classes into {BAKED}", flush=True)


def fly(args: argparse.Namespace) -> None:
    tello = Tello()
    reply = tello.ask("command")
    if reply != "ok":
        raise SystemExit(f"no 'ok' to 'command' (got {reply!r}): is this computer on the "
                         "drone's TELLO-XXXXXX Wi-Fi?")
    print(f"[teleop] connected, battery {tello.ask('battery?')}%", flush=True)
    video = VideoReceiver("0.0.0.0")
    video.start()
    # Tello EDU: a lower bitrate loses fewer packets (green frames); keeps 960x720.
    print(f"[teleop] setbitrate {args.bitrate} -> {tello.ask(f'setbitrate {args.bitrate}')}", flush=True)
    print(f"[teleop] streamon -> {tello.ask('streamon')}", flush=True)
    detector = None if args.no_detect else Detector(args.model, args.conf, args.imgsz)

    status = {"msg": "on the ground: t to take off"}
    flight = {"flying": False, "lost": False, "landing": False, "taking_off": False}

    def done(cmd: str, result: str | None) -> None:
        status["msg"] = f"{cmd} -> {result}"
        print(f"[teleop] {cmd} -> {result}", flush=True)
        if result == "ok":
            flight["flying"] = cmd == "takeoff"
        if cmd == "takeoff":
            flight["taking_off"] = False
        if cmd == "land":
            flight["landing"] = False

    def land(reason: str) -> None:
        held.clear()
        flight["landing"] = True
        status["msg"] = reason
        tello.ask_async("land", 20.0, done)

    held = {}                 # key -> monotonic time of its last event
    last_rc, last_id, shown = 0.0, 0, False
    window = "tello teleop"
    try:
        while True:
            key = cv2.waitKey(5)
            now = time.monotonic()
            key = key & 0xFF if key != -1 else -1
            stale = now - tello.state_t > STATE_STALE_S
            if LAND_ON_LOSS and flight["flying"] and stale and not flight["lost"]:
                flight["lost"] = True
                held.clear()
                status["msg"] = "LINK LOST: will land on reconnect"
                print("[teleop] link lost in flight; landing on reconnect", flush=True)
            elif flight["lost"] and not stale:
                flight["lost"] = False
                print("[teleop] link back: landing", flush=True)
                land("link back: landing")
            locked = flight["lost"] or flight["landing"]   # no flying keys until landed
            if key in MOVES and not locked:
                held[key] = now
            elif key == KEY_TAKEOFF and not (locked or flight["flying"] or flight["taking_off"]):
                flight["taking_off"] = True         # key repeat must not resend takeoff
                status["msg"] = "taking off..."
                tello.ask_async("takeoff", 20.0, done)
            elif key == KEY_LAND:
                land("landing...")
            elif key in KEY_EMERGENCY:
                tello.send("emergency")
                flight.update(flying=False, lost=False, landing=False, taking_off=False)
                status["msg"] = "EMERGENCY: motors stopped"
                print("[teleop] EMERGENCY sent", flush=True)
            elif key == KEY_QUIT:
                break
            if shown and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

            rc = [0, 0, 0, 0]
            for k, t in list(held.items()):
                if now - t > HOLD_S:
                    del held[k]
                    continue
                axis, sign = MOVES[k]
                rc[axis] = sign * args.speed
            if now - last_rc >= RC_PERIOD_S:
                tello.send("rc {} {} {} {}".format(*rc))
                last_rc = now

            frame, frame_id, _ = video.get_latest()
            if frame is None or frame_id == last_id:
                continue
            last_id = frame_id
            if detector is not None:
                detector.submit(frame)
            st = tello.state
            hud = [f"bat {st.get('bat', '?')}%  h {st.get('h', '?')} cm  "
                   f"rc {rc[0]:+d} {rc[1]:+d} {rc[2]:+d} {rc[3]:+d}  speed {args.speed}",
                   status["msg"]]
            if detector is not None:
                hud.append(f"detect {detector.fps:4.1f} fps, {len(detector.boxes)} objects")
            warn = "LINK: no state from drone" if stale else (
                "LOW BATTERY: land" if st.get("bat", "100").isdigit() and int(st["bat"]) < 15
                else None)
            boxes = detector.boxes if detector is not None else []
            cv2.imshow(window, draw(frame, boxes, hud, warn))
            shown = True
    except KeyboardInterrupt:
        pass
    finally:
        print("[teleop] landing", flush=True)
        tello.land_now()
        if detector is not None:
            detector.stop.set()
        tello.send("streamoff")
        video.stop()
        tello.close()
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", type=int, default=40, help="rc magnitude per key, 10-100")
    ap.add_argument("--no-detect", action="store_true", help="video and control only")
    ap.add_argument("--bitrate", type=int, default=2, choices=range(6),
                    help="video Mbps, 0 = firmware auto (Tello EDU)")
    ap.add_argument("--model", type=Path, help=f"detector weights (default {BAKED.name}, "
                                               f"else {FALLBACK.name})")
    ap.add_argument("--conf", type=float, default=0.3, help="detection confidence threshold")
    ap.add_argument("--imgsz", type=int, default=480, help="detector input size, px")
    ap.add_argument("--bake", action="store_true", help=f"bake the vocabulary into {BAKED.name} and exit")
    ap.add_argument("--classes", nargs="+", default=VOCAB, help="with --bake: the vocabulary")
    args = ap.parse_args()
    if args.bake:
        bake(args.classes)
        return
    if not 10 <= args.speed <= 100:
        ap.error("--speed must be 10-100")
    fly(args)


if __name__ == "__main__":
    main()
