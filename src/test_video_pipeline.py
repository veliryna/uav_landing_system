"""
test_video_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Offline test for the precision landing pipeline.
Replaces CameraCapture and MAVLinkSender with an MP4 reader.
Testing modules apriltag detector, pose estimator, transformer, Kalman filter

Usage
─────
    python test_video_pipeline.py --video landing_tag.mp4 [--config config.yaml]
                                  [--out results.mp4] [--no-display] [--slow N]

Arguments
─────────
    --video      Path to your MP4 file (required)
    --config     Path to config.yaml (default: config.yaml in same directory)
    --out        Save annotated video to this path (optional)
    --no-display Skip the live OpenCV window (useful on headless machines)
    --slow N     Insert N ms delay between frames to watch in slow motion

Output
──────
    • Live annotated window showing:
        - Detected tag corners + ID
        - Pose axes (OpenCV convention, drawn on tag)
        - Raw body-FRD vector (red)
        - Kalman-smoothed body-FRD vector (green)
        - Distance and angles
    • Console summary per frame
    • matplotlib trajectory plot on exit (x, y, z over time + top-down path)
    • Optional saved annotated MP4

"""

import sys
import argparse
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── import pipeline modules (must be on PYTHONPATH or in same directory) ──────
try:
    from detect_apriltag import AprilTagDetector
    from pose_estimator import PoseEstimator
    from transformer import FrameTransformer
    from kalman_filter import KalmanFilter3D
except ImportError as e:
    sys.exit(
        f"[ERROR] Could not import pipeline module: {e}\n"
        "Make sure detect_apriltag.py, pose_estimator.py, transformer.py, "
        "and kalman_filter.py are in the same directory or on PYTHONPATH."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helpers
# ─────────────────────────────────────────────────────────────────────────────

FONT      = cv2.FONT_HERSHEY_SIMPLEX
GREEN     = (0, 220, 0)
RED       = (0, 60, 230)
YELLOW    = (0, 220, 220)
WHITE     = (255, 255, 255)
DARK_GREY = (40, 40, 40)
BLUE      = (230, 130, 0)


def draw_tag_corners(frame, detection):
    """Draw the four tag corners and connecting lines."""
    corners = detection.corners.astype(int)
    for i in range(4):
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[(i + 1) % 4])
        cv2.line(frame, pt1, pt2, YELLOW, 2)
        cv2.circle(frame, pt1, 5, YELLOW, -1)
    # tag centre
    cx, cy = detection.center.astype(int)
    cv2.drawMarker(frame, (cx, cy), YELLOW, cv2.MARKER_CROSS, 14, 2)
    cv2.putText(frame, f"id={detection.tag_id}  margin={detection.decision_margin:.0f}",
                (cx + 10, cy - 10), FONT, 0.55, YELLOW, 1, cv2.LINE_AA)


def draw_pose_axes(frame, rvec, tvec, camera_matrix, dist_coeffs, axis_len=0.1):
    """Project 3-D axes onto the tag plane so you can judge orientation."""
    axis_pts = np.float32([
        [0, 0, 0],
        [axis_len, 0, 0],   # X → red
        [0, axis_len, 0],   # Y → green
        [0, 0, -axis_len],  # Z → blue (out of tag toward camera)
    ])
    img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec,
                                   camera_matrix, dist_coeffs)
    img_pts = img_pts.reshape(-1, 2).astype(int)
    origin = tuple(img_pts[0])
    cv2.arrowedLine(frame, origin, tuple(img_pts[1]), (0, 0, 230), 2, tipLength=0.3)
    cv2.arrowedLine(frame, origin, tuple(img_pts[2]), (0, 200, 0), 2, tipLength=0.3)
    cv2.arrowedLine(frame, origin, tuple(img_pts[3]), (200, 100, 0), 2, tipLength=0.3)


def draw_hud(frame, body_raw, body_smooth, angle_x, angle_y,
             frame_idx, fps, detections_total):
    """Overlay telemetry panel in the top-left corner."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 340, 185
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), DARK_GREY, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def put(text, row, color=WHITE):
        cv2.putText(frame, text, (16, 30 + row * 26),
                    FONT, 0.52, color, 1, cv2.LINE_AA)

    put(f"Frame: {frame_idx:>5}   Detected: {detections_total}", 0, YELLOW)
    dist = float(np.linalg.norm(body_smooth))
    put(f"Dist  (smooth): {dist:>6.3f} m", 1, GREEN)
    put(f"Body  raw  FRD: ({body_raw[0]:+.3f}, {body_raw[1]:+.3f}, {body_raw[2]:+.3f})", 2, RED)
    put(f"Body  filt FRD: ({body_smooth[0]:+.3f}, {body_smooth[1]:+.3f}, {body_smooth[2]:+.3f})", 3, GREEN)
    put(f"angle_x (pitch): {np.degrees(angle_x):+.2f} deg", 4, WHITE)
    put(f"angle_y (roll) : {np.degrees(angle_y):+.2f} deg", 5, WHITE)
    put(f"Video FPS: {fps:.1f}", 6, DARK_GREY if False else (160, 160, 160))


def no_detection_banner(frame, streak):
    h, w = frame.shape[:2]
    cv2.putText(frame, f"NO TAG  (streak={streak})",
                (w // 2 - 120, 40), FONT, 0.9, RED, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectory(records, video_fps):
    """Four-panel matplotlib figure saved and shown after processing."""
    if not records:
        print("[WARN] No detections — nothing to plot.")
        return

    frames = np.array([r['frame'] for r in records])
    t      = frames / video_fps
    raw    = np.array([r['raw']    for r in records])
    smooth = np.array([r['smooth'] for r in records])
    ax_x   = np.degrees([r['ax'] for r in records])
    ax_y   = np.degrees([r['ay'] for r in records])
    dist   = np.linalg.norm(smooth, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Precision Landing Pipeline — Offline Test Results", fontsize=14)

    # ── X / Y / Z over time ────────────────────────────────────────────
    ax = axes[0, 0]
    for i, (label, color) in enumerate(zip(['X (fwd)', 'Y (right)', 'Z (down)'],
                                            ['tab:blue', 'tab:orange', 'tab:green'])):
        ax.plot(t, raw[:, i],    color=color, alpha=0.3, linewidth=1)
        ax.plot(t, smooth[:, i], color=color, linewidth=1.8, label=label)
    ax.set_title("Body-FRD position over time (Kalman smoothed = solid)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("meters")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Distance over time ─────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(t, dist, color='tab:red', linewidth=1.8)
    ax.set_title("Distance to target (smoothed)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("meters")
    ax.grid(True, alpha=0.3)

    # ── Angles over time ───────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(t, ax_x, label='angle_x (pitch)', color='tab:purple', linewidth=1.5)
    ax.plot(t, ax_y, label='angle_y (roll)',  color='tab:cyan',   linewidth=1.5)
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_title("ArduPilot angles (from smoothed body vec)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("degrees")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Top-down 2-D path (X forward vs Y right) ──────────────────────
    ax = axes[1, 1]
    sc = ax.scatter(smooth[:, 1], smooth[:, 0],
                    c=t, cmap='plasma', s=12, zorder=3)
    ax.plot(smooth[:, 1], smooth[:, 0], color='grey', linewidth=0.7, alpha=0.5)
    ax.scatter([smooth[0, 1]], [smooth[0, 0]],  color='green', s=60,
               zorder=5, label='start')
    ax.scatter([smooth[-1, 1]], [smooth[-1, 0]], color='red',   s=60,
               zorder=5, label='end')
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.set_title("Top-down path (Y-right vs X-forward)")
    ax.set_xlabel("Y right (m)")
    ax.set_ylabel("X forward (m)")
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label='time (s)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    plot_path = "pipeline_results.png"
    plt.savefig(plot_path, dpi=130)
    print(f"[INFO] Trajectory plot saved → {plot_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Offline video test for precision landing pipeline")
    parser.add_argument("--video",      required=True,          help="Path to MP4 video file")
    parser.add_argument("--config",     default="config.yaml",  help="Path to config.yaml")
    parser.add_argument("--out",        default=None,           help="Save annotated video to this path")
    parser.add_argument("--no-display", action="store_true",    help="Skip live OpenCV window")
    parser.add_argument("--slow",       type=int, default=1,    help="Delay between frames in ms (default=1)")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cal = cfg['camera_calibration']
    camera_matrix = np.array([
        [cal['fx'],      0, cal['cx']],
        [      0, cal['fy'], cal['cy']],
        [      0,       0,        1  ],
    ], dtype=np.float64)
    dist_coeffs = np.array(cal['dist_coeffs'], dtype=np.float64)

    # ── Pipeline modules (unmodified) ─────────────────────────────────────────
    detector  = AprilTagDetector(cfg['apriltag'])
    estimator = PoseEstimator(cfg['camera_calibration'], cfg['marker'])
    transform = FrameTransformer(cfg['mounting'])
    kalman    = KalmanFilter3D(cfg['filter'])

    # ── Video source ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {args.video}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {args.video}  {vid_w}x{vid_h}  {video_fps:.1f} fps  {total_frames} frames")

    # ── Optional output writer ─────────────────────────────────────────────────
    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.out, fourcc, video_fps, (vid_w, vid_h))
        print(f"[INFO] Saving annotated video → {args.out}")

    # ── Processing loop ───────────────────────────────────────────────────────
    records           = []   # for trajectory plot
    frame_idx         = 0
    detections_total  = 0
    no_detect_streak  = 0

    print("[INFO] Processing — press Q or ESC to quit early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = detector.detect(gray)
        best       = detector.best_detection(detections)

        if best is None:
            no_detect_streak += 1
            if no_detect_streak > 10:
                kalman.reset()
            no_detection_banner(frame, no_detect_streak)
        else:
            no_detect_streak = 0
            detections_total += 1

            rvec, tvec = estimator.estimate(best)

            if tvec is not None:
                body_raw    = transform.to_body_frame(tvec)
                body_smooth = kalman.update(body_raw)
                angle_x, angle_y = transform.body_to_angles(body_smooth)

                # ── Visual overlays ───────────────────────────────────────
                draw_tag_corners(frame, best)
                draw_pose_axes(frame, rvec, tvec.reshape(3, 1),
                               camera_matrix, dist_coeffs, axis_len=0.08)
                draw_hud(frame, body_raw, body_smooth,
                         angle_x, angle_y, frame_idx, video_fps, detections_total)

                # record for plot
                records.append({
                    'frame':  frame_idx,
                    'raw':    body_raw.copy(),
                    'smooth': body_smooth.copy(),
                    'ax':     angle_x,
                    'ay':     angle_y,
                })

                # console output every 15 frames
                if frame_idx % 15 == 0:
                    dist = np.linalg.norm(body_smooth)
                    print(f"  frame={frame_idx:>5}  dist={dist:.3f}m  "
                          f"body_smooth=({body_smooth[0]:+.3f}, "
                          f"{body_smooth[1]:+.3f}, {body_smooth[2]:+.3f})  "
                          f"angles=({np.degrees(angle_x):+.1f}°, "
                          f"{np.degrees(angle_y):+.1f}°)")

        # progress bar in terminal
        pct = frame_idx / max(total_frames, 1) * 100
        print(f"\r[{'█' * int(pct // 2):<50}] {pct:5.1f}%  "
              f"frame {frame_idx}/{total_frames}  "
              f"detections={detections_total}", end='', flush=True)

        if writer:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("Precision Landing — Pipeline Test", frame)
            key = cv2.waitKey(args.slow) & 0xFF
            if key in (ord('q'), 27):   # Q or ESC
                print("\n[INFO] Quit by user.")
                break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print()
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────────────
    detect_rate = detections_total / max(frame_idx, 1) * 100
    print(f"\n{'─'*60}")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Tag detections   : {detections_total}  ({detect_rate:.1f}%)")
    if records:
        dists = [np.linalg.norm(r['smooth']) for r in records]
        print(f"  Distance range   : {min(dists):.3f} m  →  {max(dists):.3f} m")
        axs = [np.degrees(r['ax']) for r in records]
        ays = [np.degrees(r['ay']) for r in records]
        print(f"  angle_x range    : {min(axs):+.2f}°  →  {max(axs):+.2f}°")
        print(f"  angle_y range    : {min(ays):+.2f}°  →  {max(ays):+.2f}°")
    print(f"{'─'*60}\n")

    if records:
        print("[INFO] Generating trajectory plot...")
        plot_trajectory(records, video_fps)
    else:
        print("[WARN] No detections in video — check tag family, tag size, and lighting.")
        print("       Config has placeholder calibration values (fx=fy=970).")
        print("       Run calibration.py with your phone camera before relying on distances.")


if __name__ == "__main__":
    main()