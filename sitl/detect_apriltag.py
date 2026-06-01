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
    # return list of Detection objects, empty if none found
    def detect(self, gray_frame):
        return self.detector.detect(gray_frame)
    # select the most reliable detection 
    # correct method for single tag targets
    def best_detection(self, detections):
        if not detections:
            return None
        return max(detections, key=lambda d: d.decision_margin)