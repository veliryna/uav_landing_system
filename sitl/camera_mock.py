# test replacement for camera-capture.py module
import queue
import threading
import time
import cv2

class CameraCapture:
    def __init__(self, cfg):
        self.frame_queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self.source = "landing_tag.jpg"   # or any test image or video data

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        img = cv2.imread(self.source, cv2.IMREAD_GRAYSCALE)
        while not self._stop.is_set():
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(img)   # same frame in the loop
            time.sleep(1.0 / 30)        # simulate 30 fps