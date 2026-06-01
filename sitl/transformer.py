import numpy as np
 
class FrameTransformer:
    def __init__(self, mount_cfg):
        yaw = np.radians(mount_cfg['yaw_deg'])

        # expr below constructs the rotation matrix that transforms a pose output vector 
        # from the OpenCV camera frame into the body FRD frame
        
        R_cam_to_body = np.array([          # yaw rotation around the down axis matrix
            [ np.cos(yaw), np.sin(yaw), 0],
            [-np.sin(yaw), np.cos(yaw), 0],
            [           0,           0, 1],
        ]) @ np.array([
            [ 0, -1, 0],   # body x = -camera y
            [ 1,  0, 0],   # body y =  camera x
            [ 0,  0, 1],   # body z =  camera z
        ])

        self.R = R_cam_to_body
        self.cam_offset = np.array([
            mount_cfg['pos_x'],
            mount_cfg['pos_y'],
            mount_cfg['pos_z'],
        ])
    # take the tag 3D position in the camera frame 
    # and return it in the body FRD frame centered at the vehicle's center of mass
    def to_body_frame(self, tvec_camera):
        # apply the rotation matrix to transform the camera-frame vector into the body frame. 
        body = self.R @ tvec_camera
        # minus camera mounting offset to get tag offset from CoM
        body -= self.cam_offset
        return body 

    # Convert body-frame offset to angle_x (pitch) and angle_y (roll) in radians.
    # ArduPilot uses pitch and roll if position_valid=0.
    def body_to_angles(self, body_vec):
        dist = np.linalg.norm(body_vec)
        if dist < 1e-6:
            return 0.0, 0.0
        angle_x = np.arctan2(body_vec[0], body_vec[2])
        angle_y = np.arctan2(body_vec[1], body_vec[2])
        return angle_x, angle_y