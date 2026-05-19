import time, unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src import mavlink_signal_sender

CFG = {'port': 'tcp:localhost:5760', 'baud': 115200,
       'system_id': 1, 'component_id': 1, 'output_hz': 10}

class TestMAVLinkSender(unittest.TestCase):

    def _make_sender(self):
        sender = mavlink_signal_sender.MAVLinkSender(CFG)
        # Fully mock the connection so no socket is opened
        sender.conn = MagicMock()
        sender.conn.mav.landing_target_send = MagicMock()
        return sender

    def test_send_called_when_data_fresh(self):
        sender = self._make_sender()
        vec = np.array([0.5, 0.1, 2.0])
        sender.update(vec, 0.05, -0.03)
        sender.start()
        time.sleep(0.25)   # let the thread fire a few times
        sender.stop()
        self.assertTrue(sender.conn.mav.landing_target_send.called)

    def test_stale_data_not_sent(self):
        sender = self._make_sender()
        vec = np.array([0.5, 0.1, 2.0])
        sender.update(vec, 0.05, -0.03)
        # Artificially age the timestamp
        with sender._lock:
            body, ax, ay, _ = sender._latest
            sender._latest = (body, ax, ay, time.time() - 1.0)  # 1 s old
        sender.start()
        time.sleep(0.25)
        sender.stop()
        sender.conn.mav.landing_target_send.assert_not_called()

    def test_clear_stops_sending(self):
        sender = self._make_sender()
        sender.update(np.array([1, 0, 3]), 0.1, 0.1)
        sender.clear()
        sender.start()
        time.sleep(0.25)
        sender.stop()
        sender.conn.mav.landing_target_send.assert_not_called()

    def test_body_vec_not_mutated(self):
        """update() must copy the array — caller mutation must not affect sent data."""
        sender = self._make_sender()
        vec = np.array([1.0, 2.0, 3.0])
        sender.update(vec, 0, 0)
        vec[:] = 0          # mutate caller's array
        with sender._lock:
            stored_vec = sender._latest[0]
        np.testing.assert_array_equal(stored_vec, [1.0, 2.0, 3.0])

    def test_send_correct_distance(self):
        """distance must equal the L2 norm of body_vec."""
        sender = self._make_sender()
        vec = np.array([3.0, 4.0, 0.0])   # norm = 5.0
        sender._send(vec, 0.1, -0.1)
        args = sender.conn.mav.landing_target_send.call_args[0]
        self.assertAlmostEqual(args[5], 5.0)   # distance is 6th positional arg

if __name__ == '__main__':
    unittest.main()