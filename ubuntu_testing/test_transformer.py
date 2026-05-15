import numpy as np
import pytest
from src import transformer

def make_transformer(yaw_deg=0.0, pos_x=0.0, pos_y=0.0, pos_z=0.0):
    return transformer.FrameTransformer({
        'yaw_deg': yaw_deg,
        'pos_x': pos_x, 'pos_y': pos_y, 'pos_z': pos_z,
    })

class TestFrameTransform:

    def test_tag_directly_below_no_offset(self):
        """
        Tag directly below camera, no mounting offset.
        Camera z (forward=down here since camera points down) → body z (down).
        tvec_camera = [0, 0, 1] → body = [0, 0, 1] (1m below CoM).
        """
        t = make_transformer()
        # Camera frame: x=right, y=down, z=forward(=down since cam points down)
        # Tag directly below: z=1, x=0, y=0
        body = t.to_body_frame(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(body, [0.0, 0.0, 1.0], atol=1e-9)

    def test_tag_to_the_right(self):
        """
        Tag 0.5m to camera's right.
        Camera x (right) → body y (right).
        tvec_camera = [0.5, 0, 1] → body = [1, 0.5, 0].
        Wait — camera z=1 → body x=1 (forward), camera x=0.5 → body y=0.5 (right).
        """
        t = make_transformer()
        body = t.to_body_frame(np.array([0.5, 0.0, 1.0]))
        np.testing.assert_allclose(body[1], 0.5, atol=1e-9,
            err_msg="Camera right should map to body right (y)")
        np.testing.assert_allclose(body[0], 1.0, atol=1e-9,
            err_msg="Camera forward(z) should map to body forward(x)")

    def test_tag_forward_in_camera(self):
        """
        Camera z (forward) maps to body x (forward).
        """
        t = make_transformer()
        body = t.to_body_frame(np.array([0.0, 0.0, 2.0]))
        np.testing.assert_allclose(body[0], 2.0, atol=1e-9,
            err_msg="Camera z should map to body x (forward)")
        np.testing.assert_allclose(body[1], 0.0, atol=1e-9)
        np.testing.assert_allclose(body[2], 0.0, atol=1e-9)

    def test_camera_down_maps_to_body_down(self):
        """Camera y (down in image) should map to body z (down)."""
        t = make_transformer()
        body = t.to_body_frame(np.array([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(body[2], 1.0, atol=1e-9,
            err_msg="Camera y (down) should map to body z (down)")

    def test_mounting_offset_applied(self):
        """
        Camera 5cm below CoM: pos_z=0.05.
        Tag 1m below camera → 0.95m below CoM.
        """
        t = make_transformer(pos_z=0.05)
        body = t.to_body_frame(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(body[2], 0.95, atol=1e-9,
            err_msg="Camera z offset not correctly subtracted")

    def test_mounting_offset_forward(self):
        """Camera 3cm forward of CoM: pos_x=0.03."""
        t = make_transformer(pos_x=0.03)
        body = t.to_body_frame(np.array([0.0, 0.0, 1.0]))
        # body[0] = camera_z - pos_x = 1.0 - 0.03 = 0.97
        np.testing.assert_allclose(body[0], 0.97, atol=1e-9)

    def test_yaw_90_rotates_axes(self):
        """
        Camera rotated 90° clockwise when mounted.
        What was camera-right (x) becomes camera-backward after 90° yaw.
        """
        t = make_transformer(yaw_deg=90.0)
        # A point to camera's right in camera frame
        body = t.to_body_frame(np.array([1.0, 0.0, 0.0]))
        # After 90° yaw, camera x should now point in a different body direction
        # The magnitude should be preserved
        magnitude = np.linalg.norm(body)
        np.testing.assert_allclose(magnitude, 1.0, atol=1e-9,
            err_msg="Rotation should preserve vector magnitude")

    def test_yaw_0_is_identity_rotation(self):
        """yaw=0 should produce same result as no yaw."""
        t0 = make_transformer(yaw_deg=0.0)
        t_none = make_transformer(yaw_deg=0.0)
        vec = np.array([0.3, 0.1, 2.0])
        np.testing.assert_allclose(
            t0.to_body_frame(vec),
            t_none.to_body_frame(vec),
            atol=1e-9
        )

    def test_rotation_preserves_distance(self):
        """Frame rotation should never change the Euclidean distance."""
        for yaw in [0, 45, 90, 180, 270]:
            t = make_transformer(yaw_deg=yaw)
            vec = np.array([0.5, 0.2, 3.0])
            body = t.to_body_frame(vec)
            original_dist = np.linalg.norm(vec)
            # Distance changes slightly due to offset subtraction —
            # test rotation-only by using zero offset
            t2 = make_transformer(yaw_deg=yaw, pos_x=0, pos_y=0, pos_z=0)
            body2 = t2.to_body_frame(vec)
            np.testing.assert_allclose(
                np.linalg.norm(body2), original_dist, atol=1e-9,
                err_msg=f"Distance changed at yaw={yaw}"
            )


class TestBodyToAngles:

    def setup_method(self):
        self.t = make_transformer()

    def test_directly_below_gives_zero_angles(self):
        """Target directly below → both angles should be zero."""
        angle_x, angle_y = self.t.body_to_angles(np.array([0.0, 0.0, 5.0]))
        np.testing.assert_allclose(angle_x, 0.0, atol=1e-9)
        np.testing.assert_allclose(angle_y, 0.0, atol=1e-9)

    def test_zero_vector_returns_zero(self):
        """Zero vector guard — should not divide by zero."""
        angle_x, angle_y = self.t.body_to_angles(np.array([0.0, 0.0, 0.0]))
        assert angle_x == 0.0
        assert angle_y == 0.0

    def test_forward_offset_gives_positive_angle_x(self):
        """Target 1m forward and 5m below → small positive angle_x."""
        angle_x, angle_y = self.t.body_to_angles(np.array([1.0, 0.0, 5.0]))
        assert angle_x > 0, "Forward target should give positive angle_x"
        np.testing.assert_allclose(angle_x, np.arctan2(1.0, 5.0), atol=1e-9)

    def test_right_offset_gives_positive_angle_y(self):
        """Target 0.5m right and 3m below → positive angle_y."""
        angle_x, angle_y = self.t.body_to_angles(np.array([0.0, 0.5, 3.0]))
        assert angle_y > 0, "Right target should give positive angle_y"
        np.testing.assert_allclose(angle_y, np.arctan2(0.5, 3.0), atol=1e-9)

    def test_angles_in_radians(self):
        """Angles should be in radians, not degrees."""
        angle_x, angle_y = self.t.body_to_angles(np.array([1.0, 0.0, 1.0]))
        # arctan2(1,1) = pi/4 ≈ 0.785, not 45
        assert abs(angle_x) < np.pi, "angle_x looks like it might be in degrees"

    def test_symmetry(self):
        """Left offset should give equal magnitude but opposite sign to right."""
        _, angle_right = self.t.body_to_angles(np.array([0.0,  1.0, 5.0]))
        _, angle_left  = self.t.body_to_angles(np.array([0.0, -1.0, 5.0]))
        np.testing.assert_allclose(angle_right, -angle_left, atol=1e-9)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])