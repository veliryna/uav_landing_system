import queue
import threading
import numpy as np
from picamera2 import Picamera2

class CameraCapture:
    def __init__(self, cfg):
        self.cfg = cfg    # get camera params from config
        self.frame_queue = queue.Queue(maxsize=2)  # shared frame buffer 
        self._stop = threading.Event() # stop a background thread

    # method to begin camera capture
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    # method for camera thread to exit cleanly
    def stop(self):
        self._stop.set()
    # main camera loop, frames capture
    def _run(self):
        cam = Picamera2()
        w, h = self.cfg['resolution']
        # camera configuration, still_conf means full-resolution, unscaled capture
        config = cam.create_still_configuration( 
            main={"size": (w, h), "format": "YUV420"},
            controls={"ExposureTime": self.cfg['exposure_us'], "FrameRate": self.cfg['fps']}
        )
        cam.configure(config)
        cam.start()
        try:
            while not self._stop.is_set():
                # capture one frame from the camera and return it as a NumPy array
                frame = cam.capture_array()
                # get grayscale
                gray = frame[:h, :w].copy()
                # drop oldest frame if consumer is slow to never block capture
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(gray)
        finally:
            cam.stop()