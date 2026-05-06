import queue
import threading
import numpy as np
from picamera2 import Picamera2

class CameraCapture:
    def __init__(self, cfg):
        self.cfg = cfg
        self.frame_queue = queue.Queue(maxsize=2)  # drop old frames, never block
        self._stop = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        cam = Picamera2()
        w, h = self.cfg['resolution']
        config = cam.create_still_configuration(
            main={"size": (w, h), "format": "YUV420"},
            controls={"ExposureTime": self.cfg['exposure_us'], "FrameRate": self.cfg['fps']}
        )
        cam.configure(config)
        cam.start()
        try:
            while not self._stop.is_set():
                frame = cam.capture_array()
                # Convert YUV420 → grayscale (luma plane is free)
                gray = frame[:h, :w].copy()
                # Drop oldest if consumer is slow — never block capture
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(gray)
        finally:
            cam.stop()