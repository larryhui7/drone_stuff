# drone_telemetry

Fleet telemetry for [drone_gui](../drone_gui/README.md): every drone's team
(attack/defense), pose, battery, height and velocity, published as UDP JSON at a
rate you set live in the GUI. A defender policy reads the `attack` entries, an
attacker policy reads the `defense` ones.

| File | What |
|---|---|
| `telemetry.py` | `Publisher` (background sender, live Hz/destinations), `load_roles()` |
| `roles.yaml` | team per grid slot / per drone name, default destinations, Hz, on/off |
| `listen.py` | example consumer, and the template for a policy that reads telemetry |

## Run

```bash
cd drone_deployment_infra/drone_gui
../../.venv/bin/python -u drone_gui.py              # publishes to 127.0.0.1:9100 at 10 Hz

cd ../drone_telemetry
../../.venv/bin/python -u listen.py                 # print everything
../../.venv/bin/python -u listen.py --team attack   # only the attackers
../../.venv/bin/python -u listen.py --raw           # raw JSON
```

In the GUI's top bar, the **Telemetry** box sets on/off, the rate (0.5-50 Hz) and the
destinations. Its status line shows the measured rate, datagram size and sequence number.
It turns amber when a datagram is bigger than 1400 B (see Size) and red when a send fails.
Changes made in the GUI are live only. Set the startup defaults in `roles.yaml`.

Only one process can bind a port. For several consumers, give each its own port, e.g.
`127.0.0.1:9100, 127.0.0.1:9101`. To feed another laptop, add its LAN address. Don't
use a `192.168.10.x` address: that subnet belongs to the Tellos' own WiFi.

## Coasting lost attackers: `track.py`

When an attacker loses its tag, drone_gui keeps publishing its last pose with a growing
`age`. `track.py` sits between drone_gui and the policy. It keeps each attacker's last
60 s of fixes in the tag-10 room frame (via `config/map.yaml`). Once a pose is older
than 0.5 s, it predicts the drone flying straight on at the velocity fitted to the last
second of fixes, so defenders can still intercept.

```bash
../../.venv/bin/python -u track.py --selftest                           # pass: SELFTEST PASSED
../../.venv/bin/python -u track.py --listen 9100 --forward 127.0.0.1:9101
```

Point the policy at `:9101` instead of `:9100`; it needs no code change. Each attacker
gains `"track"`: room-frame `xyz`, `vel`, `predicted`, `coast_s`, `fix_age`. While
coasting, its `pose` is the prediction in its last tag's frame, with `age` 0.0,
`"predicted": true` and the real age in `fix_age` (`--no-fill-pose` turns this off).
Limits: fixes from tags missing in `map.yaml` are skipped (the relay prints them), a
grounded drone (`fly` false) doesn't coast, coasting stops at the floor and gives up
after 60 s.

**Defender sightings.** Start the GUI with `drone_gui.py --spot`. One detector runs over
the defenders' cameras (`drone_gui/spotter.py`, default model
`vision_and_mapping/models/tello_world.pt` from `teleop_detect.py --bake`). Each drone box
becomes a camera-frame position using that drone's calibration: range = focal length ×
98 mm Tello width / box width, after removing lens distortion. The defender's entry carries
it as `"seen": {"age", "dets": [{"cam": [x, y, z], "conf"}]}` (camera frame: x right,
y down, z forward; m). `track.py` places each sighting in the room using the defender's own
pose. It drops sightings that land on another defender, and adds the rest to the nearest
attacker whose own tags no longer track it, giving `"source": "seen"` instead of a coast.
The camera is assumed level, since the pose carries only yaw, and a yaw flip on a small tag
puts the sighting off to the side. Boxes cut off by the image border, narrower than 12 px or
beyond 6 m are skipped.

## Teams

Each drone's team is resolved in this order, first match wins:

1. The **Team** selector on the drone's panel in the GUI (`attack`, `defense`, `none`;
   `auto` clears it).
2. `drones:` in `roles.yaml`, by drone name, e.g. `TELLO-2: defense`.
3. `slots:` in `roles.yaml`, by grid position. The default is slots 1-4 `attack` and
   5-8 `defense`.

Slots follow the GUI order (dongle number). Pin teams by name if dongles get swapped.

## Wire format (`v: 1`)

There is one datagram per tick, holding every drone, so a consumer always gets one
consistent snapshot. It is compact JSON (no spaces). Example:

```json
{"v":1,"src":"real","seq":1234,"t":1791660000.123,"hz":10.0,"drones":[
  {"name":"TELLO-2","slot":1,"team":"attack","link":true,"fly":false,"mode":"ground",
   "bat":87,"h":0.0,"tof":0.1,"vel":[0.0,0.0,0.0],
   "pose":{"tag":8,"xyz":[-0.56,0.89,-3.03],"yaw":-6.2,"age":0.04}}]}
```

| Field | Meaning |
|---|---|
| `v` | format version. Bumped only on breaking changes. New fields may appear in v1, so ignore unknown keys. |
| `src` | `real` or `sim` (from `drone_gui.py --sim N`) |
| `seq` | increments every datagram. A gap means datagrams were lost. |
| `t` | sender wall clock, Unix seconds |
| `hz` | configured rate |
| `name`, `slot`, `team` | drone name, GUI grid position (1 = first tile), `attack` / `defense` / `""` |
| `link` | the SDK link is alive (state packets arriving) |
| `fly` | airborne |
| `mode` | `ground`, `connecting`, `takeoff`, `hover`, `manual`, `auto`, `search`, `landing` |
| `bat` | battery % (`null` before the first state packet) |
| `h` | height above the takeoff point, m (Tello `h`) |
| `tof` | downward range sensor, m |
| `vel` | Tello `vgx, vgy, vgz` in m/s, in the Tello's own axes |
| `pose` | last camera pose from an AprilTag, or `null` if the drone has never seen one (see below) |
| `truth` | `sim` only: the true pose, in the same format as `pose`, for checking |
| `seen` | `--spot` only, spotting team only, present only when something was detected in the last 1 s: drones in this camera, `{"age", "dets": [{"cam": [x, y, z], "conf"}]}` (see Coasting) |

### `pose`

`pose` is the **camera position in the tag's frame**. The tag used is the drone's
target tag if visible, else the nearest tag.

- `tag`: the tag ID. Each tag is its own frame. There is no shared room frame yet.
- `xyz`, in metres, standing in front of the tag looking at it:
  - x is to the right.
  - y is down, so a drone below the tag has y > 0.
  - z points into the wall, so a drone in front of the tag has **z < 0**, and -z is the
    distance in front.
- `yaw`: degrees. 0 means the camera faces the tag head-on (looking along +z).
  Positive means the heading has turned toward +x.
- `age`: seconds since this pose was measured. The last pose is kept when the tag goes
  out of view and `age` keeps growing, so treat anything older than about 0.5 s as
  last-known (`listen.py` prints `STALE`).

**Known limitation:** a single small or distant tag has two mirror-image pose
solutions. Range and bearing are reliable: x, z, and the distance |xyz|. `y` and `yaw`
can flip, with x and y both changing sign and yaw jumping by 10-20°. In the sim this
happened on tags under about 40 px wide. For height, prefer `h` / `tof`. Drone_gui's
Auto steers on bearing and range for the same reason.

## Size

8 real drones come to about 1400 B per datagram. With `truth`, 8 sims come to about
2000 B. Larger datagrams get IP-fragmented. That is fine on localhost and wired links,
but over WiFi losing one fragment drops the whole datagram. The GUI flags this in amber.
If it matters, receive on the same machine or on a wired link.
