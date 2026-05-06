import numpy as np

class KalmanFilter3D:
    def __init__(self, cfg):
        q = cfg['process_noise']
        r = cfg['measurement_noise']
        # State: [x, y, z], simple constant-position model
        self.P = np.eye(3) * 1.0       # initial covariance
        self.Q = np.eye(3) * q         # process noise
        self.R = np.eye(3) * r         # measurement noise
        self.x = None                   # state estimate

    def update(self, measurement):
        """measurement: np.array([x, y, z])"""
        if self.x is None:
            self.x = measurement.copy()
            return self.x

        # Predict
        P_pred = self.P + self.Q

        # Update
        K = P_pred @ np.linalg.inv(P_pred + self.R)
        self.x = self.x + K @ (measurement - self.x)
        self.P = (np.eye(3) - K) @ P_pred

        return self.x

    def reset(self):
        self.x = None
        self.P = np.eye(3) * 1.0