import numpy as np
import cv2

class PoseEstimator:
    def __init__(self, cal_cfg, marker_cfg):
        self.fx = cal_cfg['fx']
        self.fy = cal_cfg['fy']
        self.cx = cal_cfg['cx']
        self.cy = cal_cfg['cy']
        self.dist = np.array(cal_cfg['dist_coeffs'], dtype=np.float64)

        s = marker_cfg['tag_size_m'] / 2.0
        # Tag corners in tag-local frame: TL, TR, BR, BL (counter-clockwise, z=0)
        self.obj_pts = np.array([
            [-s,  s, 0],
            [ s,  s, 0],
            [ s, -s, 0],
            [-s, -s, 0],
        ], dtype=np.float64)

        self.camera_matrix = np.array([
            [self.fx,    0, self.cx],
            [   0, self.fy, self.cy],
            [   0,    0,    1],
        ], dtype=np.float64)

    def estimate(self, detection):
        """
        Returns (rvec, tvec) in camera frame.
        tvec[0]=right, tvec[1]=down, tvec[2]=forward (OpenCV convention).
        """
        img_pts = detection.corners.astype(np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.obj_pts,
            img_pts,
            self.camera_matrix,
            self.dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,  # best for planar square targets
        )
        if not success:
            return None, None
        return rvec, tvec.flatten()