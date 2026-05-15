import numpy as np
import cv2
import pytest
from src import pose_estimator
from unittest.mock import MagicMock

# Approximate laptop webcam intrinsics — close enough for unit testing
CAL_CFG = {
    'fx': 800.0, 'fy': 800.0,
    'cx': 320.0, 'cy': 240.0,
    'dist_coeffs': [0.0, 0.0, 0.0, 0.0, 0.0],
}
MARKER_CFG = {'tag_size_m': 0.21}

def make_synthetic_detection(rvec, tvec, estimator):
    """
    Project the known 3D tag corners through the camera model to produce
    synthetic 2D image points. Returns a mock Detection with those corners.
    """
    corners_2d, _ = cv2.projectPoints(
        estimator.obj_pts,
        rvec, tvec,
        estimator.camera_matrix,
        estimator.dist,
    )
    corners_2d = corners_2d.reshape(4, 2)
    mock_det = MagicMock()
    mock_det.corners = corners_2d.astype(np.float64)
    return mock_det

class TestPoseEstimator:

    def setup_method(self):
        self.est = pose_estimator.PoseEstimator(CAL_CFG, MARKER_CFG)

    def test_tag_directly_below_at_1m(self):
        """Tag directly below at 1m: tvec should be approx [0, 0, 1]."""
        rvec_gt = np.array([0.0, 0.0, 0.0])
        tvec_gt = np.array([0.0, 0.0, 1.0])
        det = make_synthetic_detection(rvec_gt, tvec_gt, self.est)
        rvec, tvec = self.est.estimate(det)
        assert tvec is not None, "Estimation failed"
        np.testing.assert_allclose(tvec, tvec_gt, atol=0.01,
            err_msg="Tag at [0,0,1] not recovered correctly")

    def test_tag_offset_right(self):
        """Tag 0.5m to the right and 2m below."""
        rvec_gt = np.array([0.0, 0.0, 0.0])
        tvec_gt = np.array([0.5, 0.0, 2.0])
        det = make_synthetic_detection(rvec_gt, tvec_gt, self.est)
        rvec, tvec = self.est.estimate(det)
        assert tvec is not None
        np.testing.assert_allclose(tvec, tvec_gt, atol=0.02)

    def test_tag_offset_forward(self):
        """Tag 0.3m forward (in camera z) and 3m away."""
        rvec_gt = np.array([0.0, 0.0, 0.0])
        tvec_gt = np.array([0.0, 0.3, 3.0])
        det = make_synthetic_detection(rvec_gt, tvec_gt, self.est)
        rvec, tvec = self.est.estimate(det)
        assert tvec is not None
        np.testing.assert_allclose(tvec, tvec_gt, atol=0.02)

    def test_tvec_is_1d(self):
        """Returned tvec should be flattened to shape (3,)."""
        rvec_gt = np.array([0.0, 0.0, 0.0])
        tvec_gt = np.array([0.0, 0.0, 2.0])
        det = make_synthetic_detection(rvec_gt, tvec_gt, self.est)
        rvec, tvec = self.est.estimate(det)
        assert tvec.shape == (3,), f"Expected shape (3,), got {tvec.shape}"

    def test_altitude_scales_correctly(self):
        """
        At 2m the z component should be 2x the value at 1m.
        Verifies the tag_size_m scale is correctly embedded.
        """
        rvec_gt = np.zeros(3)
        det_1m = make_synthetic_detection(rvec_gt, np.array([0,0,1.0]), self.est)
        det_2m = make_synthetic_detection(rvec_gt, np.array([0,0,2.0]), self.est)
        _, tvec_1m = self.est.estimate(det_1m)
        _, tvec_2m = self.est.estimate(det_2m)
        assert tvec_1m is not None and tvec_2m is not None
        ratio = tvec_2m[2] / tvec_1m[2]
        np.testing.assert_allclose(ratio, 2.0, atol=0.05,
            err_msg="Altitude scaling is wrong — check tag_size_m")

    def test_noisy_corners_still_converge(self):
        """With small pixel noise, estimate should still be close."""
        rvec_gt = np.zeros(3)
        tvec_gt = np.array([0.0, 0.0, 2.0])
        det = make_synthetic_detection(rvec_gt, tvec_gt, self.est)
        # Add 2-pixel Gaussian noise to corners
        det.corners = det.corners + np.random.normal(0, 2.0, det.corners.shape)
        rvec, tvec = self.est.estimate(det)
        assert tvec is not None
        np.testing.assert_allclose(tvec, tvec_gt, atol=0.15,
            err_msg="Too much error under 2px noise")

    def test_camera_matrix_shape(self):
        """Camera matrix should be 3x3."""
        assert self.est.camera_matrix.shape == (3, 3)

    def test_obj_pts_shape(self):
        """Object points should be (4, 3) — four 3D corners."""
        assert self.est.obj_pts.shape == (4, 3)

    def test_obj_pts_z_zero(self):
        """All object point z-values should be zero (flat tag)."""
        np.testing.assert_array_equal(self.est.obj_pts[:, 2], 0.0)

    def test_obj_pts_symmetric(self):
        """Object points should be symmetric around origin."""
        s = MARKER_CFG['tag_size_m'] / 2.0
        np.testing.assert_allclose(
            np.abs(self.est.obj_pts[:, :2]),
            s,
            atol=1e-9,
            err_msg="Object points are not symmetric"
        )

if __name__ == '__main__':
    pytest.main([__file__, '-v'])