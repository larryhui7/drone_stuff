# vision_and_mapping

How the drones find out where they are: AprilTags seen by the Tello camera, turned into a
position in the room map. Dead reckoning between sightings comes from the Kalman filter.

## The AprilTag reader

The reader is **[pupil-apriltags](https://github.com/pupil-labs/apriltags)** (Python
bindings for the AprilRobotics C library, `tag36h11`). It is already the detector the flight
engine and the room map use, pinned in `drone_deployment_infra/cal_and_map/requirements.txt`
(`pupil-apriltags==1.0.4.post11`). We use it as is and don't add a second detector, so the sim,
the map and the flight all see the same fixes.

Pipeline (all existing code, read in place):

| Stage | File | What it does |
|---|---|---|
| detect + gate | `drone_deployment_infra/cal_and_map/tello_calibration/tags/detect.py` | pupil-apriltags detection, quality gate (decision margin, pixel size, blur, view angle, flip ambiguity), single-tag IPPE pose |
| tag config | `drone_deployment_infra/config/tags.yaml` | family, black-square sizes (tags 1-9: 132.08 mm, tag 10: 571.5 mm), detector + gate settings |
| room map | `drone_deployment_infra/config/map.yaml` | every tag's pose relative to tag 10 (built by `swarm_ops/map_tags.py` or `cal_and_map/scripts/05_map.py`) |
| camera | `drone_deployment_infra/config/camera/tello_<n>.yaml` | per-drone intrinsics; `default.yaml` if a drone isn't calibrated |
| position | `drone_deployment_infra/core/positional.py` (`Positioner`) | drone position in the tag-10 frame, averaged over every mapped tag in view |
| filter | `drone_deployment_infra/core/positional_kalman_filter.py` (`FilteredPositioner`) | constant-velocity Kalman filter: predicts between fixes, rejects outliers, freezes when lost. The sim loads this same file. |

Map frame: origin at tag 10's centre, x right, y up, z out of the tag face (in front of the
wall is +z), metres.

## Offline check: `read_tags.py`

Runs that pipeline on saved frames (no drone, no network) and prints the fix per frame:

```bash
pip install -r drone_deployment_infra/cal_and_map/requirements.txt   # or the repo venv
python -u vision_and_mapping/read_tags.py frame.png --drone tello_3 --all-tags
python -u vision_and_mapping/read_tags.py flight.mp4 --drone tello_3 --filter
python -u vision_and_mapping/read_tags.py frames_dir/ --drone tello_3 --fps 30
```

`--all-tags` also lists every detection with the gate's reasons for rejecting it, `--filter`
adds the Kalman estimate, velocity and lost flag. The last line, `[read_tags] N/M frames
localized`, is the summary.

## Live test on a real Tello (hand-held, no flight)

`--tello` reads the camera of the drone this computer's Wi-Fi is joined to (video decoded by
cal_and_map's `VideoReceiver`, which survives the H.264 errors before each keyframe). It
sends only `command`, `streamon`, `battery?`, `streamoff`: the motors never start. Works on
macOS; `cal_and_map/scripts/06_localize.py` does the same on the Linux desktop with the USB
Wi-Fi sticks (it needs `nmcli`).

1. Power the drone on and join its `TELLO-XXXXXX` network from the computer's Wi-Fi menu. On
   macOS, allow Python through the firewall if asked (video arrives on UDP 11111).
2. Run, with `--drone` set to the drone's number so its calibration
   `config/camera/tello_<n>.yaml` is used (falls back to `default.yaml`):
   ```bash
   python -u vision_and_mapping/read_tags.py --tello --drone tello_3 --filter --all-tags \
     --record outputs/walk_tello_3.mp4 | tee outputs/read_tags_live.log
   ```
   `q` in the video window or Ctrl+C stops it and saves the recording.
3. Checks, carrying the drone by hand:
   - **Distance:** point the camera straight at tag 10 from a taped spot (e.g. 2.00 m from
     the wall). `z` should read that distance within ~5 cm, `x` ~0 when centred.
   - **Axes:** step right: `x` grows; raise it: `y` grows; walk back: `z` grows.
   - **Hand-off:** pan from tag 10 to a side tag. The position should stay put (within ~10 cm)
     as the `tags` list changes. A jump means that tag's pose in `map.yaml` is off.
   - **Lost:** cover the camera for >0.4 s: `lost True`, then it recovers on the next fix.
4. Re-run the recording offline as often as you like:
   `python -u vision_and_mapping/read_tags.py outputs/walk_tello_3.mp4 --drone tello_3 --filter`.

Checked with a synthetic tag 10 rendered 2.5 m straight ahead (default camera intrinsics):
it reads back `tag 10 2.48 m`, `xyz [-0.037, -0.119, 2.475]`.

## Tag 10 height (to do)

The sim places the room with `env.tag10_height`, the height of tag 10's **centre** above the
floor (the map has no floor). Measure it with a tape and pass it to training and recording,
e.g. `env.tag10_height=1.22`. CLAUDE.md and the shipped `peer_qualified_84M` checkpoint
config use 1.22 m (~48 in, measured); older notes assume 1.2 m. Re-measure if the tag moves.

## Keyboard flight + object detection: `teleop_detect.py`

**This one flies.** Join the drone's Wi-Fi, run it, click the video window:
`t` take off, `space` land, `w/s` forward/back, `a/d` left/right, `q/e` up/down,
`z/c` turn, `esc` land and quit, `backspace` emergency motor stop. With no key held the
drone hovers; Ctrl+C, an error or closing the window lands it.

```bash
python -u vision_and_mapping/teleop_detect.py --speed 30
```

Detection (person, drone, common room objects) runs YOLO-World with a baked vocabulary,
`models/tello_world.pt` (26 MB, ~40 fps on Apple-silicon GPU, own thread). Re-bake with
different classes via `--bake --classes ...` (needs CLIP:
`pip install git+https://github.com/ultralytics/CLIP.git`). Weights in `models/` are
git-ignored.
