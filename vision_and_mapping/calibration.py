"""Calibrate a Tello camera from checkerboard images: K, distortion, skew, extrinsics.

The inner corners of a printed checkerboard are found in every image and paired with their
3D coordinates in the board (world) frame: origin at the first inner corner, x along the
rows, y down the columns, z = 0 on the board, metres. cv2.calibrateCamera estimates the
intrinsics and each image's extrinsics; a least-squares refinement then adds the skew term
(OpenCV's model has none). For validation the world points are back-projected onto every
image with the estimated parameters and compared with the detected corners.

    python -u vision_and_mapping/calibration.py frames_dir/ --pattern 9x6 --square-mm 25
    python -u vision_and_mapping/calibration.py board.mp4 --pattern 9x6 --square-mm 25 --every 15
    python -u vision_and_mapping/calibration.py --tello --save-dir outputs/checker_tello_3 \
        --pattern 9x6 --square-mm 25 --out drone_deployment_infra/config/camera/tello_3.yaml

--pattern is the number of INNER corners (columns x rows), not squares: a board of 10 x 7
squares is 9x6. --square-mm is one square's edge measured on the print.

Reported: focal length (fx, fy), principal point (cx, cy), image size, radial distortion
(k1, k2, k3), tangential distortion (p1, p2), skew, the matrix K, reprojection error per
image and overall, and each image's extrinsics (board pose in the camera frame).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
OUTLIER_FACTOR = 3.0       # drop images whose error is > 3x the median, as 03_calibrate.py does
CAPTURE_GAP_S = 1.0        # --tello: at least this long between saved views
MIN_MOVE_PX = 40.0         # --tello: board must move this much (mean corner shift) between saves


def load_frames(source: Path, every: int):
    """(label, BGR image) from an image, a directory of images, or every Nth frame of a video."""
    if source.is_dir():
        for path in sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXT):
            yield path.name, cv2.imread(str(path))
    elif source.suffix.lower() in IMAGE_EXT:
        yield source.name, cv2.imread(str(source))
    else:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise SystemExit(f"cannot open {source}")
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every == 0:
                yield f"frame_{i:05d}", frame
            i += 1


def find_corners(gray: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    """(N, 2) float32 sub-pixel inner corners in row-major board order, or None."""
    ok, corners = cv2.findChessboardCornersSB(
        gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY)
    if not ok:
        ok, corners = cv2.findChessboardCorners(
            gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            return None
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
    return corners.reshape(-1, 2).astype(np.float32)


def board_points(pattern: tuple[int, int], square_m: float) -> np.ndarray:
    """(N, 3) world coordinates of the inner corners, same order as find_corners."""
    cols, rows = pattern
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return grid


def project(world: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray,
            dist: np.ndarray) -> np.ndarray:
    """Pinhole + plumb_bob (k1, k2, p1, p2, k3) projection that honours K's skew K[0, 1].

    cv2.projectPoints ignores K[0, 1], so this is the same model written out with it.
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    cam = world @ R.T + np.asarray(tvec, np.float64).reshape(1, 3)
    x, y = cam[:, 0] / cam[:, 2], cam[:, 1] / cam[:, 2]
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = K[0, 0] * xd + K[0, 1] * yd + K[0, 2]
    v = K[1, 1] * yd + K[1, 2]
    return np.stack([u, v], axis=1)


def frame_errors(world, image_pts, rvecs, tvecs, K, dist) -> np.ndarray:
    """Per-image RMS distance (px) between detected corners and back-projected world points."""
    return np.array([np.sqrt(np.mean(np.sum((project(world, r, t, K, dist) - img) ** 2, axis=1)))
                     for img, r, t in zip(image_pts, rvecs, tvecs)])


def opencv_calibrate(world, image_pts, size, flags):
    obj = [world] * len(image_pts)
    img = [p.reshape(-1, 1, 2) for p in image_pts]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, None, None, flags=flags)
    return K, dist.reshape(-1)[:5], [r.reshape(3) for r in rvecs], [t.reshape(3) for t in tvecs]


def refine_with_skew(world, image_pts, K, dist, rvecs, tvecs, fix_k3: bool):
    """Joint least-squares refinement of all intrinsics + skew + every image's pose."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    n_img, n_pts = len(image_pts), len(world)
    n_intr = 10  # fx, fy, s, cx, cy, k1, k2, p1, p2, k3

    def unpack(p):
        fx, fy, s, cx, cy, *d = p[:n_intr]
        Kp = np.array([[fx, s, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        dp = np.array(d)
        if fix_k3:
            dp[4] = 0.0
        poses = p[n_intr:].reshape(n_img, 6)
        return Kp, dp, poses

    def residuals(p):
        Kp, dp, poses = unpack(p)
        return np.concatenate([(project(world, pose[:3], pose[3:], Kp, dp) - img).ravel()
                               for img, pose in zip(image_pts, poses)])

    p0 = np.concatenate([[K[0, 0], K[1, 1], 0.0, K[0, 2], K[1, 2]], dist,
                         np.concatenate([np.r_[r, t] for r, t in zip(rvecs, tvecs)])])
    sparsity = lil_matrix((2 * n_pts * n_img, p0.size), dtype=int)
    for i in range(n_img):
        rows = slice(2 * n_pts * i, 2 * n_pts * (i + 1))
        sparsity[rows, :n_intr] = 1
        sparsity[rows, n_intr + 6 * i:n_intr + 6 * (i + 1)] = 1
    result = least_squares(residuals, p0, jac_sparsity=sparsity, method="trf", x_scale="jac")
    Kp, dp, poses = unpack(result.x)
    return Kp, dp, [p[:3] for p in poses], [p[3:] for p in poses]


def calibrate(world, labels, image_pts, size, fix_k3: bool, skew: bool):
    """Calibrate, drop outlier images once, recalibrate; optionally add skew."""
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    K, dist, rvecs, tvecs = opencv_calibrate(world, image_pts, size, flags)
    errors = frame_errors(world, image_pts, rvecs, tvecs, K, dist)
    keep = errors <= OUTLIER_FACTOR * np.median(errors)
    dropped = [label for label, k in zip(labels, keep) if not k]
    if dropped:
        labels = [label for label, k in zip(labels, keep) if k]
        image_pts = [p for p, k in zip(image_pts, keep) if k]
        K, dist, rvecs, tvecs = opencv_calibrate(world, image_pts, size, flags)
    if skew:
        K, dist, rvecs, tvecs = refine_with_skew(world, image_pts, K, dist, rvecs, tvecs, fix_k3)
    return K, dist, rvecs, tvecs, labels, image_pts, dropped


def tello_capture(save_dir: Path, pattern: tuple[int, int], target: int):
    """Save views of the board from the live Tello camera (read_tags' read-only link).

    A view is saved when the full board is found, CAPTURE_GAP_S has passed and the board
    moved by MIN_MOVE_PX since the last save. Stops at `target` views, q or Ctrl+C.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from read_tags import tello_frames

    save_dir.mkdir(parents=True, exist_ok=True)
    saved, last_t, last_corners = 0, -1e9, None
    source = tello_frames(None)
    try:
        for _, frame, _ in source:
            now = time.monotonic()
            if now - last_t < CAPTURE_GAP_S:
                continue
            corners = find_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), pattern)
            if corners is None:
                continue
            if last_corners is not None and \
                    np.mean(np.linalg.norm(corners - last_corners, axis=1)) < MIN_MOVE_PX:
                continue
            cv2.imwrite(str(save_dir / f"checker_{saved:03d}.png"), frame)
            saved, last_t, last_corners = saved + 1, now, corners
            print(f"[calib] saved {saved}/{target} views", flush=True)
            if saved >= target:
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
    return saved


def report(K, dist, size, labels, errors, rvecs, tvecs, dropped, n_found, n_seen, skew):
    fx, fy, s, cx, cy = K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = dist
    print(f"[calib] board found in {n_found}/{n_seen} images; "
          f"{len(labels)} used, {len(dropped)} dropped as outliers"
          + (f" ({', '.join(dropped)})" if dropped else ""))
    print("\nIntrinsics")
    print(f"  image size            : {size[0]} x {size[1]} px (width x height)")
    print(f"  focal length          : fx = {fx:.4f} px, fy = {fy:.4f} px")
    print(f"  principal point       : cx = {cx:.4f} px, cy = {cy:.4f} px")
    print(f"  radial distortion     : k1 = {k1:.6f}, k2 = {k2:.6f}, k3 = {k3:.6f}")
    print(f"  tangential distortion : p1 = {p1:.6f}, p2 = {p2:.6f}")
    print(f"  skew                  : s = {s:.6f} px"
          + ("" if skew else " (fixed at 0; drop --no-skew to estimate it)")
          + f"  (angle between pixel axes {np.degrees(np.arctan2(fy, -s)):.4f} deg)")
    print("  K =")
    for row in K:
        print("    [" + "  ".join(f"{v:12.4f}" for v in row) + " ]")
    print("\nValidation: world points back-projected onto each image vs detected corners")
    for label, e in zip(labels, errors):
        print(f"  {label:<24} {e:.3f} px")
    total = np.sqrt(np.mean(errors ** 2))
    print(f"  mean per-image error  : {np.mean(errors):.3f} px, overall RMS {total:.3f} px")
    print("\nExtrinsics (board pose in camera frame: rotation vector rad, translation m)")
    for label, r, t in zip(labels, rvecs, tvecs):
        print(f"  {label:<24} r [{r[0]:+.4f} {r[1]:+.4f} {r[2]:+.4f}]  "
              f"t [{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}]  dist {np.linalg.norm(t):.3f} m")
    return total


def save_overlays(out_dir: Path, sources: dict, labels, image_pts, world, rvecs, tvecs, K, dist):
    """Detected corners (green) and back-projected world points (red) on each used image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, img_pts, r, t in zip(labels, image_pts, rvecs, tvecs):
        image = sources[label].copy()
        for (u, v) in img_pts:
            cv2.circle(image, (round(u), round(v)), 5, (0, 255, 0), 1)
        for (u, v) in project(world, r, t, K, dist):
            cv2.drawMarker(image, (round(u), round(v)), (0, 0, 255), cv2.MARKER_CROSS, 6, 1)
        cv2.imwrite(str(out_dir / f"{Path(label).stem}_reproj.png"), image)
    first = sources[labels[0]]
    cv2.imwrite(str(out_dir / "undistorted_sample.png"),
                cv2.hconcat([first, cv2.undistort(first, K, dist)]))


def camera_yaml(path: Path, drone: str, K, dist, size, rms, n_frames, fix_k3, sources) -> None:
    """Same layout as config/camera/<drone>.yaml written by cal_and_map's 03_calibrate.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "drone": drone,
        "image_width": int(size[0]),
        "image_height": int(size[1]),
        "model": "pinhole",
        "distortion_model": "plumb_bob (k1, k2, p1, p2, k3)",
        "fix_k3": fix_k3,
        "camera_matrix": K.tolist(),
        "dist_coeffs": [float(d) for d in dist],
        "skew": float(K[0, 1]),
        "rms_px": float(rms),
        "n_frames": n_frames,
        "target": "checkerboard",
        "date": dt.date.today().isoformat(),
        "source_sessions": [str(Path(s).resolve()) for s in sources],
    }
    path.write_text(yaml.safe_dump(data, default_flow_style=None, sort_keys=False))
    print(f"[calib] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, nargs="?", help="image, directory of images, or video")
    ap.add_argument("--pattern", required=True, help="inner corners, columns x rows, e.g. 9x6")
    ap.add_argument("--square-mm", type=float, required=True, help="measured square edge, mm")
    ap.add_argument("--every", type=int, default=10, help="video: use every Nth frame")
    ap.add_argument("--tello", action="store_true", help="capture views from the connected Tello first")
    ap.add_argument("--save-dir", type=Path, help="with --tello: where captured views go")
    ap.add_argument("--target", type=int, default=40, help="with --tello: views to capture")
    ap.add_argument("--fix-k3", action="store_true", help="hold k3 at 0 (steadier for mild lenses)")
    ap.add_argument("--no-skew", action="store_true", help="keep skew at 0 (OpenCV only, no scipy)")
    ap.add_argument("--min-images", type=int, default=10, help="need at least this many boards")
    ap.add_argument("--out", type=Path, help="write a camera yaml (config/camera/<drone>.yaml layout)")
    ap.add_argument("--drone", default="tello_1", help="drone name stored in --out")
    ap.add_argument("--overlays", type=Path, help="save reprojection overlays + an undistorted sample here")
    args = ap.parse_args()

    try:
        pattern = tuple(int(n) for n in args.pattern.lower().split("x"))
        assert len(pattern) == 2 and min(pattern) >= 2
    except (ValueError, AssertionError):
        ap.error("--pattern must be inner corners like 9x6")
    if args.tello:
        if args.save_dir is None or args.source is not None:
            ap.error("--tello needs --save-dir and no source")
        tello_capture(args.save_dir, pattern, args.target)
        args.source = args.save_dir
    elif args.source is None:
        ap.error("give a source file/directory or --tello")

    world = board_points(pattern, args.square_mm / 1000.0)
    labels, image_pts, sources, size, seen = [], [], {}, None, 0
    for label, frame in load_frames(args.source, args.every):
        if frame is None:
            print(f"{label}: unreadable", flush=True)
            continue
        seen += 1
        h, w = frame.shape[:2]
        if size is None:
            size = (w, h)
        elif size != (w, h):
            print(f"{label}: {w}x{h} differs from {size[0]}x{size[1]}, skipped", flush=True)
            continue
        corners = find_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), pattern)
        print(f"{label}: " + ("board found" if corners is not None else "no board"), flush=True)
        if corners is not None:
            labels.append(label)
            image_pts.append(corners)
            if args.overlays:
                sources[label] = frame
    n_found = len(labels)
    if n_found < args.min_images:
        raise SystemExit(f"[calib] board found in only {n_found}/{seen} images "
                         f"(need {args.min_images}); check --pattern (inner corners) and lighting")

    K, dist, rvecs, tvecs, labels, image_pts, dropped = calibrate(
        world, labels, image_pts, size, args.fix_k3, not args.no_skew)
    errors = frame_errors(world, image_pts, rvecs, tvecs, K, dist)
    rms = report(K, dist, size, labels, errors, rvecs, tvecs, dropped, n_found, seen,
                 not args.no_skew)
    if abs(K[0, 1]) > 1e-3 * K[0, 0]:
        print("[calib] note: OpenCV and pupil-apriltags ignore skew (K[0, 1]); "
              "re-run with --no-skew for a K they use exactly")
    if args.overlays:
        save_overlays(args.overlays, sources, labels, image_pts, world, rvecs, tvecs, K, dist)
        print(f"[calib] wrote overlays to {args.overlays}")
    if args.out:
        camera_yaml(args.out, args.drone, K, dist, size, rms, len(labels), args.fix_k3, [args.source])


if __name__ == "__main__":
    main()
