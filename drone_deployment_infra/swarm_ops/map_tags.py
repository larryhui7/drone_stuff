#!/usr/bin/env python3
"""Build map.yaml: every tag's pose relative to tag 10. NO FLYING (props off, hand-held).

Hold the drone so the camera sees tag 10 together with another tag; two tags in one
frame links them. Tags never seen with tag 10 can be linked through a tag that is
already mapped (chained). Each tag collects many views; outliers are dropped and the
rest are averaged.

  Window colors:  green = mapped (enough consistent views)   yellow = collecting
                  red = seen but rejected this frame (reason shown)   grey = not in list
  Keys: s = save map.yaml (keeps going), q = save and quit, r = reset the selected tag
        (the tag nearest the image center), Esc = quit WITHOUT saving

Tips: link each tag directly with tag 10 if you can; hold still for a second per view;
view each pair from a few angles (30-50 deg off-axis views are the most accurate; dead
straight-on views of a far tag are ambiguous and get skipped).

  python map_tags.py --drone 4 --tags-list 10,1,2,3,4,8,9
  python map_tags.py --drone 4 --tags-list 10,8,9          # writes config/map.yaml

Output format = same as before (x right, y up, z out of the tag face; meters), so
swarm_formation.py reads it directly. An existing map.yaml is backed up first.
"""
import argparse
import json
import math
import os
import shutil
import sys
import threading
import time

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from paths import CAMERA_DIR, DONGLES_PATH, MAP_PATH, TAGS_PATH, camera_file  # noqa: E402  (config/ lives in core)

import swarm_formation as sf          # same connection / camera / tag-pose code

TAG_ID = 10
MIN_MARGIN = 30          # stricter than flight: mapping can wait for clean views
MIN_SIDE_PX = 25
MAX_ERR_PX = 1.5
AMBIG_DEG = 3.0          # two near-equal poses this different -> skip (straight-on view)
MIN_SAMPLES = 15         # views needed before a tag counts as mapped
MAX_SPREAD_CM = 3.0      # ...and their (inlier) spread must be below this
OUTLIER_CM = 6.0
C_FLIP = np.diag([1.0, -1.0, -1.0])   # internal (y down, z into) <-> map (y up, z out)


def rot_mean(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rot_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class TagMapBuilder:
    """Observations of T_10k (internal frame). Direct (seen with tag 10) beat chained."""
    def __init__(self, ids):
        self.ids = ids
        self.obs = {k: [] for k in ids if k != TAG_ID}     # (R, t, direct)

    def add(self, k, R, t, direct):
        self.obs[k].append((R, t, direct))
        del self.obs[k][:-400]

    def estimate(self, k):
        """-> (R, t, n_inliers, spread_cm, n_direct) or None."""
        ob = self.obs.get(k) or []
        direct = [o for o in ob if o[2]]
        use = direct if len(direct) >= MIN_SAMPLES else ob
        if len(use) < 3:
            return None
        T = np.array([o[1] for o in use])
        med = np.median(T, axis=0)
        keep = np.linalg.norm(T - med, axis=1) < OUTLIER_CM / 100
        if keep.sum() < 3:
            return None
        Rs = [use[i][0] for i in range(len(use)) if keep[i]]
        Rm = rot_mean(Rs)
        rot_ok = [i for i, R in zip(np.nonzero(keep)[0], Rs) if rot_deg(R, Rm) < 5]
        if len(rot_ok) < 3:
            return None
        Tk = T[rot_ok]
        t = Tk.mean(axis=0)
        spread = float(np.sqrt(np.mean(np.sum((Tk - t) ** 2, axis=1)))) * 100
        return rot_mean([use[i][0] for i in rot_ok]), t, len(rot_ok), spread, len(direct)

    def mapped(self, k):
        if k == TAG_ID:
            return True
        e = self.estimate(k)
        return e is not None and e[2] >= MIN_SAMPLES and e[3] <= MAX_SPREAD_CM

    def reset(self, k):
        if k in self.obs:
            self.obs[k] = []


def unambiguous_pose(c, K, D, size):
    """Single-tag pose (internal frame, camera<-tag) or (None, reason)."""
    cands = sf.tag_candidates(c, K, D, size)
    if not cands:
        return None, "no pose"
    e0, R, t = cands[0]
    if e0 > MAX_ERR_PX:
        return None, f"fit {e0:.1f}px"
    near = [cc for cc in cands[1:] if cc[0] < 2.0 * e0 + 0.5]
    if near and rot_deg(near[0][1], R) > AMBIG_DEG:
        return None, "ambiguous (view at an angle)"
    return (R, t), None


def connect(d):
    if not sf.iface_ip(d.iface):
        sys.exit(f"{d.iface} has no 192.168.10.x address: run  nmcli con up {d.ssid}")
    ssid = sf.active_ssid(d.iface)
    if ssid != d.ssid:
        sys.exit(f"{d.iface} is on {ssid!r}, expected {d.ssid!r}")
    threading.Thread(target=d.state_loop, daemon=True).start()
    for _ in range(5):
        if d.send("command", 3) == "ok":
            break
        time.sleep(1)
    else:
        sys.exit("drone did not answer 'command'")
    threading.Thread(target=d.video_loop, daemon=True).start()
    for _ in range(3):
        if d.send("streamon", 5) == "ok":
            break
        time.sleep(1)
    t0 = time.time()
    while d.frames < 5 and time.time() - t0 < 15:
        time.sleep(0.1)
    if d.frames < 5:
        sys.exit("no video frames")
    b = d.send("battery?", 5)
    print(f"connected to {d.ssid}, battery {b}%")


def save(builder, ids, path, sizes):
    if os.path.exists(path):
        bak = path + time.strftime(".bak_%Y%m%d_%H%M%S")
        shutil.copy(path, bak)
        print(f"backed up old map -> {bak}")
    tags = {TAG_ID: {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0], "seen": 0, "samples": 0}}
    print(f"\n{'tag':>4} {'x right':>8} {'y up':>8} {'z out':>8}  {'dist':>6}  views  spread  link")
    for k in sorted(i for i in ids if i != TAG_ID):
        e = builder.estimate(k)
        if e is None or not builder.mapped(k):
            print(f"{k:>4}  NOT MAPPED (needs {MIN_SAMPLES} consistent views) -> left out")
            continue
        R, t, n, spread, ndir = e
        Rm, tm = C_FLIP @ R @ C_FLIP, C_FLIP @ t              # internal -> map frame
        tags[k] = {"R": Rm.tolist(), "t": tm.tolist(), "seen": len(builder.obs[k]), "samples": int(n)}
        print(f"{k:>4} {tm[0]:+8.3f} {tm[1]:+8.3f} {tm[2]:+8.3f}  {np.linalg.norm(tm):5.2f}m  "
              f"{n:5d}  {spread:4.1f}cm  {'direct' if ndir >= MIN_SAMPLES else 'chained'}")
    with open(path, "w") as f:
        yaml.safe_dump({"origin_tag": TAG_ID,
                        "frame": "origin tag: x right, y up, z out of the tag face; meters",
                        "tags": tags}, f, sort_keys=False)
    print(f"wrote {path}  ({len(tags)} tags incl. tag 10)\n"
          f"Check a couple of these with a tape measure (center to center) before flying.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drone", type=int, required=True, help="drone number in dongles.json, e.g. 4")
    ap.add_argument("--tags-list", required=True, help="tag ids to map, e.g. 10,1,2,3,4,8,9")
    ap.add_argument("--config", default=str(DONGLES_PATH))
    ap.add_argument("--tags", default=str(TAGS_PATH), help="tag sizes (sizes_mm)")
    ap.add_argument("--camera-dir", default=str(CAMERA_DIR))
    ap.add_argument("--out", default=str(MAP_PATH))
    args = ap.parse_args()

    ids = sorted({int(x) for x in args.tags_list.split(",") if x.strip()} | {TAG_ID})
    with open(args.tags) as f:
        sz = (yaml.safe_load(f) or {}).get("sizes_mm") or {}
    sizes = {int(k): float(v) / 1000 for k, v in sz.items() if v is not None}
    missing = [k for k in ids if k not in sizes]
    if missing:
        sys.exit(f"no size in {args.tags} for tag(s) {missing}: add them under sizes_mm (black square, mm)")
    with open(args.config) as f:
        data = json.load(f)
    data = data.get("dongles", []) if isinstance(data, dict) else data
    ent = next((e for e in data if int(e["number"]) == args.drone and e.get("ssid")), None)
    if ent is None:
        sys.exit(f"drone {args.drone} not in {args.config}")
    K, D = sf.load_camera(str(camera_file(args.drone, args.camera_dir)))
    d = sf.Drone(args.drone, ent["iface"], ent["ssid"], K, D, False, {}, {}, None)   # fly=False
    print(f"MAPPING tags {ids} with drone {args.drone} (no flying). Origin = tag {TAG_ID}.")
    connect(d)

    from pupil_apriltags import Detector
    det = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0, refine_edges=True,
                   decode_sharpening=0.25)
    builder = TagMapBuilder(ids)
    last_id, last_print = 0, 0.0
    cv2.namedWindow("map", cv2.WINDOW_NORMAL)
    try:
        while True:
            if d.frame is None or d.frame_id == last_id:
                if (cv2.waitKey(5) & 0xFF) == 27:
                    print("Esc -> quit without saving")
                    return
                continue
            last_id, img = d.frame_id, d.frame.copy()
            seen = {}                                   # id -> (R, t) camera<-tag, internal
            notes = {}
            for t_ in det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)):
                k, c = t_.tag_id, np.array(t_.corners, np.float64)
                if k not in ids:
                    cv2.polylines(img, [c.astype(np.int32).reshape(-1, 1, 2)], True, (128, 128, 128), 1)
                    continue
                side = min(np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4))
                if t_.decision_margin < MIN_MARGIN:
                    notes[k] = (c, f"weak {t_.decision_margin:.0f}")
                    continue
                if side < MIN_SIDE_PX:
                    notes[k] = (c, f"small {side:.0f}px")
                    continue
                pose, why = unambiguous_pose(c, K, D, sizes[k])
                if pose is None:
                    notes[k] = (c, why)
                    continue
                seen[k] = pose
                notes[k] = (c, None)
            # ---- link: directly with tag 10, else through an already-mapped tag ----
            if TAG_ID in seen:
                R10, t10 = seen[TAG_ID]
                for k, (Rk, tk) in seen.items():
                    if k != TAG_ID:
                        builder.add(k, R10.T @ Rk, R10.T @ (tk - t10), True)
            else:
                anchors = [m for m in seen if builder.mapped(m)]
                if anchors:
                    m = max(anchors, key=lambda a: builder.estimate(a)[2])
                    Rm0, tm0, *_ = builder.estimate(m)         # tag m in tag-10 frame
                    Rcm, tcm = seen[m]
                    for k, (Rk, tk) in seen.items():
                        if k in (m, TAG_ID):
                            continue
                        R_mk, t_mk = Rcm.T @ Rk, Rcm.T @ (tk - tcm)
                        builder.add(k, Rm0 @ R_mk, Rm0 @ t_mk + tm0, False)
            # ---- draw ----
            for k, (c, why) in notes.items():
                if why is not None:
                    col, label = (0, 0, 255), f"{k}: {why}"
                elif builder.mapped(k):
                    col, label = (0, 220, 0), f"{k} mapped"
                else:
                    e = builder.estimate(k) if k != TAG_ID else None
                    col = (0, 220, 220)
                    label = f"{k} " + (f"{e[2]}/{MIN_SAMPLES} {e[3]:.1f}cm" if e else "need tag 10 / mapped tag")
                cv2.polylines(img, [c.astype(np.int32).reshape(-1, 1, 2)], True, col, 3)
                x, y = c.min(axis=0)
                for cc, th in (((0, 0, 0), 4), (col, 2)):
                    cv2.putText(img, label, (int(x), max(int(y) - 8, 18)), 0, 0.6, cc, th)
            done = [k for k in ids if builder.mapped(k)]
            todo = [k for k in ids if not builder.mapped(k)]
            hdr = f"mapped {done}  still need {todo}   s=save q=save+quit Esc=quit"
            for cc, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
                cv2.putText(img, hdr, (10, 25), 0, 0.6, cc, th)
            cv2.imshow("map", img)
            now = time.time()
            if now - last_print > 2.0:
                last_print = now
                parts = []
                for k in ids:
                    if k == TAG_ID:
                        continue
                    e = builder.estimate(k)
                    parts.append(f"{k}:" + ("OK" if builder.mapped(k) else
                                            (f"{e[2]}v/{e[3]:.1f}cm" if e else "-")))
                print("  " + "  ".join(parts), flush=True)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                save(builder, ids, args.out, sizes)
            elif key == ord("q"):
                save(builder, ids, args.out, sizes)
                return
            elif key == 27:
                print("Esc -> quit without saving")
                return
            elif key == ord("r") and notes:
                h, w = img.shape[:2]
                k = min(notes, key=lambda i: np.linalg.norm(notes[i][0].mean(0) - [w / 2, h / 2]))
                builder.reset(k)
                print(f"reset tag {k}")
    except KeyboardInterrupt:
        print("\nCtrl+C -> saving")
        save(builder, ids, args.out, sizes)
    finally:
        d.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
