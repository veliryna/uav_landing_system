import numpy as np

class KalmanFilter3D:
    def __init__(self, cfg):
        q = cfg['process_noise']
        r = cfg['measurement_noise']

        self.P = np.eye(3) * 1.0       # initial error covariance
        self.Q = np.eye(3) * q         # process noise
        self.R = np.eye(3) * r         # measurement noise
        self.x = None                  # state estimate

    # update filter's state estimate
    def update(self, measurement):
        # measurement: np.array([x, y, z]) body FRD meters
        if self.x is None:
            self.x = measurement.copy()
            return self.x

        # prediction step of the Kalman filter
        P_pred = self.P + self.Q

        # compute the Kalman gain matrix 
        K = P_pred @ np.linalg.inv(P_pred + self.R)
        # Update the state estimate
        self.x = self.x + K @ (measurement - self.x)
        # Updats the error covariance matrix
        self.P = (np.eye(3) - K) @ P_pred

        return self.x
    
    # use when the target is lost 
    def reset(self):
        self.x = None
        self.P = np.eye(3) * 1.0