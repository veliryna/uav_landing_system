import numpy as np
import pytest
from src import kalman_filter


CFG = {'process_noise': 0.01, 'measurement_noise': 0.1}

class TestKalmanFilter3D:

    def setup_method(self):
        self.kf = kalman_filter.KalmanFilter3D(CFG)

    def test_first_measurement_initializes_state(self):
        """First call should set state directly to measurement."""
        m = np.array([1.0, 2.0, 3.0])
        result = self.kf.update(m)
        np.testing.assert_array_equal(result, m)

    def test_returns_numpy_array(self):
        """update() should always return a numpy array."""
        result = self.kf.update(np.array([0.0, 0.0, 5.0]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_converges_to_constant_signal(self):
        """
        After many identical measurements, estimate should
        converge very close to that value.
        """
        true_pos = np.array([1.0, -2.0, 5.0])
        for _ in range(100):
            result = self.kf.update(true_pos)
        np.testing.assert_allclose(result, true_pos, atol=1e-3)

    def test_smooths_noisy_signal(self):
        """
        Filter output variance should be significantly lower
        than input noise variance.
        """
        true_pos = np.array([0.5, 0.5, 3.0])
        noise_std = 0.5
        raw_measurements = []
        filtered_outputs = []

        for _ in range(200):
            noisy = true_pos + np.random.normal(0, noise_std, 3)
            raw_measurements.append(noisy)
            filtered_outputs.append(self.kf.update(noisy))

        # Discard first 20 frames (filter settling)
        raw = np.array(raw_measurements[20:])
        filt = np.array(filtered_outputs[20:])

        raw_var = np.var(raw, axis=0)
        filt_var = np.var(filt, axis=0)

        for axis in range(3):
            assert filt_var[axis] < raw_var[axis], \
                f"Filter did not reduce variance on axis {axis}: " \
                f"raw={raw_var[axis]:.4f} filtered={filt_var[axis]:.4f}"

    def test_does_not_copy_input(self):
        """State should be a copy of measurement, not the same object."""
        m = np.array([1.0, 2.0, 3.0])
        self.kf.update(m)
        m[0] = 999.0  # mutate the original
        # Filter state should not have changed
        result = self.kf.update(np.array([1.0, 2.0, 3.0]))
        assert result[0] != 999.0, "Filter state was corrupted by input mutation"

    def test_reset_clears_state(self):
        """After reset, next measurement should initialize directly."""
        self.kf.update(np.array([10.0, 10.0, 10.0]))
        for _ in range(50):
            self.kf.update(np.array([10.0, 10.0, 10.0]))
        self.kf.reset()
        # After reset, new measurement far from old state
        new_m = np.array([0.0, 0.0, 1.0])
        result = self.kf.update(new_m)
        np.testing.assert_array_equal(result, new_m,
            err_msg="After reset, first measurement should be used directly")

    def test_covariance_shrinks_with_measurements(self):
        """P diagonal should decrease as measurements arrive."""
        initial_p_diag = np.diag(self.kf.P).copy()
        self.kf.update(np.array([1.0, 1.0, 1.0]))
        for _ in range(50):
            self.kf.update(np.array([1.0, 1.0, 1.0]))
        final_p_diag = np.diag(self.kf.P)
        for i in range(3):
            assert final_p_diag[i] < initial_p_diag[i], \
                f"Covariance did not shrink on axis {i}"

    def test_tracks_slow_drift(self):
        """Filter should track a slowly moving target."""
        pos = np.array([0.0, 0.0, 3.0])
        for i in range(100):
            pos[0] += 0.01  # slow forward drift
            result = self.kf.update(pos.copy())
        # After 100 steps, true position x = 1.0
        np.testing.assert_allclose(result[0], 1.0, atol=0.1,
            err_msg="Filter failed to track slow drift")

    def test_process_noise_affects_responsiveness(self):
        """
        Higher process noise → filter tracks step changes faster.
        """
        kf_slow = kalman_filter.KalmanFilter3D({'process_noise': 0.001, 'measurement_noise': 0.1})
        kf_fast = kalman_filter.KalmanFilter3D({'process_noise': 0.1,   'measurement_noise': 0.1})

        # Settle both filters at position 0
        for _ in range(50):
            kf_slow.update(np.array([0.0, 0.0, 3.0]))
            kf_fast.update(np.array([0.0, 0.0, 3.0]))

        # Step change to position 2.0
        new_pos = np.array([2.0, 0.0, 3.0])
        fast_result, slow_result = None, None
        for _ in range(10):
            slow_result = kf_slow.update(new_pos)
            fast_result = kf_fast.update(new_pos)

        # Fast filter should be closer to new position after 10 steps
        assert abs(fast_result[0] - 2.0) < abs(slow_result[0] - 2.0), \
            "Higher process noise should produce faster tracking"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])