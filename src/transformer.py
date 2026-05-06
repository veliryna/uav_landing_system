import numpy as np

class FrameTransformer:
    def __init__(self, mount_cfg):
        yaw = np.radians(mount_cfg['yaw_deg'])
        # Rotation: camera optical frame → body FRD frame
        # OpenCV camera: x=right, y=down, z=forward
        # Body FRD:       x=forward, y=right, z=down
        # Apply yaw rotation around Z (down) for non-forward camera mounting
        R_cam_to_body = np.array([
            [ np.cos(yaw), np.sin(yaw), 0],
            [-np.sin(yaw), np.cos(yaw), 0],
            [           0,           0, 1],
        ]) @ np.array([
            [0, 0, 1],   # body x (fwd) = camera z (fwd)
            [1, 0, 0],   # body y (right) = camera x (right)
            [0, 1, 0],   # body z (down) = camera y (down)
        ])
        self.R = R_cam_to_body
        self.cam_offset = np.array([
            mount_cfg['pos_x'],
            mount_cfg['pos_y'],
            mount_cfg['pos_z'],
        ])

    def to_body_frame(self, tvec_camera):
        """
        tvec_camera: [x_right, y_down, z_fwd] in camera frame (meters)
        Returns: [x_fwd, y_right, z_down] offset from CoM to target in body frame
        """
        body = self.R @ tvec_camera
        # Subtract camera mounting offset to get offset from CoM
        body -= self.cam_offset
        return body  # [fwd, right, down]

    def body_to_angles(self, body_vec):
        """
        Convert body-frame offset to angle_x (pitch) and angle_y (roll) in radians.
        These are the angles ArduPilot uses if position_valid=0.
        """
        dist = np.linalg.norm(body_vec)
        if dist < 1e-6:
            return 0.0, 0.0
        angle_x = np.arctan2(body_vec[0], body_vec[2])  # pitch angle to target
        angle_y = np.arctan2(body_vec[1], body_vec[2])  # roll angle to target
        return angle_x, angle_y