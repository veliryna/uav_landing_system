import time
import yaml
import logging
from logging.handlers import RotatingFileHandler

from camera_mock import CameraCapture
from detect_apriltag import AprilTagDetector
from pose_estimator import PoseEstimator
from transformer import FrameTransformer
from kalman_filter import KalmanFilter3D
from mavlink_signal_sender import MAVLinkSender

def setup_logging(cfg):
    handler = RotatingFileHandler(
        cfg['file'], maxBytes=cfg['max_bytes'], backupCount=3
    )
    logging.basicConfig(
        level=getattr(logging, cfg['level']),
        handlers=[handler, logging.StreamHandler()],
        format='%(asctime)s %(levelname)s %(message)s'
    )

def main():
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg['logging'])
    log = logging.getLogger(__name__)

    camera    = CameraCapture(cfg['camera'])
    detector  = AprilTagDetector(cfg['apriltag'])
    estimator = PoseEstimator(cfg['camera_calibration'], cfg['marker'])
    transform = FrameTransformer(cfg['mounting'])
    kalman        = KalmanFilter3D(cfg['filter'])
    sender    = MAVLinkSender(cfg['mavlink'])

    # open the MAVLink connection to the autopilot 
    sender.connect()
    # spawn the camera capture thread and begin capturing frames
    camera.start()
    # spawn the MAVLink output thread which sends LANDING_TARGET
    sender.start()

    log.info("Precision landing module running...")
    no_detect_streak = 0

    try:
        # retrieve a frame, detect tags, estimate pose, transform, filter, and send
        # main detection and processing cycle; runs until an exception breaks out of it
        while True:
            try:
                frame = camera.frame_queue.get(timeout=1.0)
            except Exception:
                log.warning("Frame timeout")
                continue

            detections = detector.detect(frame)
            best = detector.best_detection(detections)

            if best is None:
                no_detect_streak += 1
                if no_detect_streak > 10:
                    kalman.reset()
                    sender.clear()
                    if no_detect_streak % 30 == 0:
                        log.warning("Target not detected")
                continue

            no_detect_streak = 0
            rvec, tvec = estimator.estimate(best)
            if tvec is None:
                continue

            body_vec = transform.to_body_frame(tvec)
            body_smooth = kalman.update(body_vec)
            angle_x, angle_y = transform.body_to_angles(body_smooth)

            sender.update(body_smooth, angle_x, angle_y)
            log.debug(f"body={body_smooth} angles=({angle_x:.3f},{angle_y:.3f})")

    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        camera.stop()
        sender.stop()

if __name__ == '__main__':
    main()