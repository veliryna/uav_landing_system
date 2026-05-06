import time
import threading
import numpy as np
from pymavlink import mavutil

class MAVLinkSender:
    def __init__(self, cfg):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._latest = None          # (body_vec, timestamp_us)
        self._stop = threading.Event()

    def connect(self):
        self.conn = mavutil.mavlink_connection(
            self.cfg['port'],
            baud=self.cfg['baud'],
            source_system=self.cfg['system_id'],
            source_component=self.cfg['component_id'],
        )
        print("MAVLink: waiting for heartbeat...")
        self.conn.wait_heartbeat()
        print(f"MAVLink: heartbeat from system {self.conn.target_system}")

    def start(self):
        interval = 1.0 / self.cfg['output_hz']
        self._thread = threading.Thread(target=self._run, args=(interval,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def update(self, body_vec, angle_x, angle_y):
        """Call from detection thread with latest estimate."""
        with self._lock:
            self._latest = (body_vec.copy(), angle_x, angle_y, time.time())

    def clear(self):
        """Call when target is lost — stop sending stale data."""
        with self._lock:
            self._latest = None

    def _run(self, interval):
        while not self._stop.is_set():
            start = time.time()
            with self._lock:
                data = self._latest

            if data is not None:
                body_vec, angle_x, angle_y, ts = data
                age = time.time() - ts
                if age < 0.5:  # don't send data older than 500 ms
                    self._send(body_vec, angle_x, angle_y)

            elapsed = time.time() - start
            time.sleep(max(0, interval - elapsed))

    def _send(self, body_vec, angle_x, angle_y):
        distance = float(np.linalg.norm(body_vec))
        self.conn.mav.landing_target_send(
            int(time.time() * 1e6),   # time_usec
            0,                         # target_num
            mavutil.mavlink.MAV_FRAME_BODY_FRD,
            angle_x,                   # angle_x (pitch offset, radians)
            angle_y,                   # angle_y (roll offset, radians)
            distance,                  # distance (meters)
            0.0, 0.0,                  # size_x, size_y (unknown)
            float(body_vec[0]),        # x (fwd, meters)
            float(body_vec[1]),        # y (right, meters)
            float(body_vec[2]),        # z (down, meters)
            [1, 0, 0, 0],             # q (unused)
            mavutil.mavlink.LANDING_TARGET_TYPE_VISION_OTHER,
            1,                         # position_valid = 1 (we provide x,y,z)
        )