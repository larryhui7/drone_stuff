"""Read AprilTags from saved Tello frames and print where the drone is in the tag-10 frame.

Offline check of the localizer the flight engine uses: no drone, no network. It runs the
deployment pipeline unchanged (pupil-apriltags detector + quality gate in
cal_and_map/tello_calibration/tags/detect.py, room map config/map.yaml, core/positional.py,
and the Kalman filter in core/positional_kalman_filter.py), so what it prints is what the
laptop would publish for those frames.

    python -u vision_and_mapping/read_tags.py frame.png --drone tello_3
    python -u vision_and_mapping/read_tags.py flight.mp4 --drone tello_3 --fps 30 --filter
    python -u vision_and_mapping/read_tags.py frames_dir/ --drone tello_3 --all-tags
    python -u vision_and_mapping/read_tags.py --tello --drone tello_3 --record outputs/walk.mp4

--tello reads the live camera of the drone whose Wi-Fi (TELLO-XXXXXX) this computer is joined
to. It sends only `command`, `streamon`, `battery?` and `streamoff`, never a
flight command, so carry the drone by hand.

Positions are metres in the map frame: origin at tag 10's centre, x right, y up, z out of
the tag face (in front of the wall is +z).
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "drone_deployment_infra"
sys.path[:0] = [str(DEPLOY / "core"), str(DEPLOY / "cal_and_map")]

from positional import Positioner  # noqa: E402
from positional_kalman_filter import FilteredPositioner  # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def frames(source: Path, fps: float):
    """(time s, BGR frame, label) from an image, a directory of images, or a video."""
    if source.is_dir():
        for i, path in enumerate(sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXT)):
            yield i / fps, cv2.imread(str(path)), path.name
    elif source.suffix.lower() in IMAGE_EXT:
        yield 0.0, cv2.imread(str(source)), source.name
    else:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise SystemExit(f"cannot open {source}")
        rate = cap.get(cv2.CAP_PROP_FPS) or fps
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i / rate, frame, f"frame {i}"
            i += 1


class TelloCommands:
    """The read-only commands, over the SDK's UDP command port. No flight command exists here."""

    ADDR = ("192.168.10.1", 8889)
    ALLOWED = {"command", "streamon", "streamoff", "battery?"}

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", 8889))
        self.sock.settimeout(3.0)

    def send(self, cmd: str) -> str:
        assert cmd in self.ALLOWED, cmd
        for _ in range(3):
            self.sock.sendto(cmd.encode(), self.ADDR)
            try:
                return self.sock.recv(1024).decode(errors="replace").strip()
            except socket.timeout:
                continue
        raise SystemExit(f"no reply to {cmd!r}: is this computer on the drone's TELLO-XXXXXX Wi-Fi?")


def tello_frames(record: Path | None):
    """(time s, BGR frame, label) from the live Tello camera until Ctrl+C or q in the window.

    Video is decoded by cal_and_map's VideoReceiver, which skips the H.264 errors the Tello
    stream has before each keyframe (djitellopy's reader stops at the first one).
    """
    from tello_calibration.video.stream import VideoReceiver

    link = TelloCommands()
    print(f"[read_tags] command -> {link.send('command')}, battery {link.send('battery?')}%", flush=True)
    video = VideoReceiver("0.0.0.0")
    video.start()
    print(f"[read_tags] streamon -> {link.send('streamon')}", flush=True)
    writer = None
    last_id = 0
    t0 = last_new = last_ping = time.monotonic()
    try:
        while True:
            frame, frame_id, _ = video.get_latest()
            now = time.monotonic()
            if now - last_ping > 5.0:           # keep the SDK session alive, report battery
                print(f"[read_tags] battery {link.send('battery?')}%", flush=True)
                last_ping = now
            if frame is None or frame_id == last_id:
                if now - last_new > 3.0:
                    print(f"[read_tags] no new video for {now - last_new:.0f} s "
                          f"({video.bytes_received} bytes received, {video.decode_errors} decode errors)",
                          flush=True)
                    last_new = now
                cv2.waitKey(5)
                continue
            last_id, last_new = frame_id, now
            if record is not None:
                if writer is None:
                    record.parent.mkdir(parents=True, exist_ok=True)
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(record), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
                writer.write(frame)
            yield now - t0, frame, f"frame {frame_id}"
            cv2.imshow("read_tags", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if writer is not None:
            writer.release()
            print(f"[read_tags] recorded {record}", flush=True)
        try:
            link.send("streamoff")
        except SystemExit:
            pass
        video.stop()
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, nargs="?", help="image, directory of images, or video")
    ap.add_argument("--tello", action="store_true", help="read the live camera of the connected Tello")
    ap.add_argument("--record", type=Path, help="with --tello: also save the video here")
    ap.add_argument("--drone", default="tello_1", help="camera calibration config/camera/<drone>.yaml")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate for image directories")
    ap.add_argument("--filter", action="store_true", help="also print the Kalman estimate (FilteredPositioner)")
    ap.add_argument("--all-tags", action="store_true", help="print every detection, rejected ones too")
    args = ap.parse_args()
    if args.tello == (args.source is not None):
        ap.error("give a source file/directory or --tello, not both")

    loc = FilteredPositioner(args.drone) if args.filter else Positioner(args.drone)
    raw_loc = loc.positioner if args.filter else loc
    source = tello_frames(args.record) if args.tello else frames(args.source, args.fps)
    try:
        run(source, loc, raw_loc, args)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()                  # stops the stream and finishes the recording


def run(source, loc, raw_loc, args) -> None:
    seen = fixed = 0
    for t, frame, label in source:
        if frame is None:
            print(f"{label}: unreadable", flush=True)
            continue
        seen += 1
        if args.all_tags:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for obs in raw_loc.detector.detect(gray):
                state = "ok" if obs.accepted else "rejected: " + ", ".join(obs.reasons)
                dist = "?" if obs.distance_m is None else f"{obs.distance_m:.2f} m"
                mapped = "mapped" if obs.tag_id in raw_loc.tag_map.poses else "NOT in map"
                print(f"{label}:   tag {obs.tag_id} {dist} margin {obs.decision_margin:.0f} "
                      f"{mapped} [{state}]", flush=True)
        if args.filter:
            est = loc.locate(frame, t=t)
            raw = est["raw"]
            fixed += raw is not None
            print(f"{label}: raw {raw['xyz'] if raw else None} tags {raw['tags'] if raw else []} "
                  f"| kf {est['xyz']} vel {est['vel']} "
                  f"accepted {est['accepted']} lost {est['lost']}", flush=True)
        else:
            fix = loc.locate(frame)
            fixed += fix is not None
            print(f"{label}: " + (f"xyz {fix['xyz']} range {fix['range']} m tags {fix['tags']}"
                                  if fix else "no mapped tag"), flush=True)
    print(f"[read_tags] {fixed}/{seen} frames localized", flush=True)


if __name__ == "__main__":
    main()
