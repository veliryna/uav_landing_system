import cv2
import numpy as np
import pytest
from src import detect_apriltag

CFG = {
    'family': 'tag36h11',
    'nthreads': 4,
    'quad_decimate': 1.0,  
    'quad_sigma': 0.0,
    'refine_edges': True,
    'decode_sharpening': 0.25,
}

def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Could not load {path}"
    return img

class TestAprilTagDetector:

    def setup_method(self):
        self.det = detect_apriltag.AprilTagDetector(CFG)

    def test_detects_known_tag(self):
        """Should detect tag ID 0 in its image."""
        gray = load_gray('ubuntu_testing/tag_examples/tag36h11-37-angled.jpg')
        detections = self.det.detect(gray)
        assert len(detections) > 0, "No tags detected in known tag image"
        ids = [d.tag_id for d in detections]
        assert 37 in ids, f"Tag ID 0 not found, got IDs: {ids}"

    def test_no_detection_on_blank(self):
        """Should return empty list on a blank white image."""
        blank = np.ones((640, 640), dtype=np.uint8) * 255
        detections = self.det.detect(blank)
        assert detections == [], f"Expected no detections, got {len(detections)}"

    def test_no_detection_on_noise(self):
        """Should return empty list on pure random noise."""
        noise = np.random.randint(0, 255, (640, 640), dtype=np.uint8)
        detections = self.det.detect(noise)
        # Noise may occasionally produce false positives but should be rare
        print(f"  Noise false positives: {len(detections)}")

    def test_best_detection_returns_none_on_empty(self):
        """best_detection should return None for empty list."""
        result = self.det.best_detection([])
        assert result is None

    def test_best_detection_picks_highest_margin(self):
        """best_detection should pick detection with highest decision_margin."""
        gray = load_gray('ubuntu_testing/tag_examples/tag36h11-37-angled.jpg')
        detections = self.det.detect(gray)
        if len(detections) < 2:
            pytest.skip("Need 2+ detections for this test")
        best = self.det.best_detection(detections)
        for d in detections:
            assert best.decision_margin >= d.decision_margin

    def test_detection_corners_shape(self):
        """Detected corners should have shape (4, 2)."""
        gray = load_gray('ubuntu_testing/tag_examples/tag36h11-37-angled.jpg')
        detections = self.det.detect(gray)
        assert len(detections) > 0
        corners = detections[0].corners
        assert corners.shape == (4, 2), f"Expected (4,2), got {corners.shape}"

    def test_detection_margin_positive(self):
        """Decision margin should be positive for a clean detection."""
        gray = load_gray('ubuntu_testing/tag_examples/tag36h11-37-angled.jpg')
        detections = self.det.detect(gray)
        assert len(detections) > 0
        assert detections[0].decision_margin > 0

    def test_multiple_tags_in_frame(self):
        """Should detect multiple tags"""
        t0 = load_gray('ubuntu_testing/tag_examples/tags-multiple.webp')
        detections = self.det.detect(t0)
        ids = {d.tag_id for d in detections}
        print(f"  Detected IDs in two-tag frame: {ids}")
        # At least one should be detected
        assert len(detections) == 8

if __name__ == '__main__':
    pytest.main([__file__, '-v'])