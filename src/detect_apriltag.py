import time
import numpy as np
from pupil_apriltags import Detector

class AprilTagDetector:
    def __init__(self, cfg):
        self.detector = Detector(
            families=cfg['family'],
            nthreads=cfg['nthreads'],
            quad_decimate=cfg['quad_decimate'],
            quad_sigma=cfg['quad_sigma'],
            refine_edges=cfg['refine_edges'],
            decode_sharpening=cfg['decode_sharpening'],
        )

    def detect(self, gray_frame):
        """Returns list of Detection objects, empty if none found."""
        return self.detector.detect(gray_frame)

    def best_detection(self, detections):
        """
        Pick the detection with the highest decision margin (confidence).
        If you use multi-marker targets, aggregate them instead.
        """
        if not detections:
            return None
        return max(detections, key=lambda d: d.decision_margin)