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
        Camera optical axis (z) points DOWN = body z.
        tvec_camera = [0, 0, 1] → body = [0, 0, 1].
        """
        t = make_transformer()
        body = t.to_body_frame(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(body, [0.0, 0.0, 1.0], atol=1e-9)

    def test_tag_to_the_right(self):
        """
        Tag 0.5m to camera's right (image x), 1m below (camera z).
        camera x → body y (right).
        camera z → body z (down).
        tvec = [0.5, 0, 1] → body = [0, 0.5, 1].
        """
        t = make_transformer()
        body = t.to_body_frame(np.array([0.5, 0.0, 1.0]))
        np.testing.assert_allclose(body[1], 0.5, atol=1e-9,
            err_msg="Camera x (right) should map to body y (right)")
        np.testing.assert_allclose(body[2], 1.0, atol=1e-9,
            err_msg="Camera z (optical/down) should map to body z (down)")
        np.testing.assert_allclose(body[0], 0.0, atol=1e-9,
            err_msg="No forward offset expected")

    def test_tag_in_camera_y_direction(self):
        """
        Tag displaced along camera y (image-down = toward vehicle tail).
        camera y → -body x (negative forward = backward).
        tvec = [0, 1, 0] → body = [-1, 0, 0].
        """
        t = make_transformer()
        body = t.to_body_frame(np.array([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(body[0], -1.0, atol=1e-9,
            err_msg="Camera y (image down = tail) should map to -body x")
        np.testing.assert_allclose(body[1],  0.0, atol=1e-9)
        np.testing.assert_allclose(body[2],  0.0, atol=1e-9)

    def test_tag_forward_of_vehicle(self):
        """
        Tag 1m forward of vehicle, 5m below.
        Forward in body = -camera y direction.
        Tag forward → appears near TOP of image → negative camera y.
        tvec = [0, -1, 5] → body = [1, 0, 5].
        """
        t = make_transformer()
        body = t.to_body_frame(np.array([0.0, -1.0, 5.0]))
        np.testing.assert_allclose(body[0],  1.0, atol=1e-9,
            err_msg="Negative camera y should map to positive body x (forward)")
        np.testing.assert_allclose(body[2],  5.0, atol=1e-9)

    def test_camera_optical_axis_maps_to_body_down(self):
        """
        Camera z (optical axis, pointing down) should map to body z (down).
        This is the fundamental property of a downward-facing camera.
        """
        t = make_transformer()
        body = t.to_body_frame(np.array([0.0, 0.0, 3.0]))
        np.testing.assert_allclose(body[2], 3.0, atol=1e-9,
            err_msg="Camera z (optical axis) should map to body z (down)")
        np.testing.assert_allclose(body[0], 0.0, atol=1e-9)
        np.testing.assert_allclose(body[1], 0.0, atol=1e-9)

    def test_camera_x_maps_to_body_right(self):
        """Camera x (image right) should map to body y (right)."""
        t = make_transformer()
        body = t.to_body_frame(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(body[1], 1.0, atol=1e-9,
            err_msg="Camera x (image right) should map to body y (right)")
        np.testing.assert_allclose(body[0], 0.0, atol=1e-9)
        np.testing.assert_allclose(body[2], 0.0, atol=1e-9)

    def test_mounting_offset_applied(self):
        """
        Camera 5cm below CoM: pos_z=0.05.
        Tag 1m below camera (camera z=1) → body z = 1.0 - 0.05 = 0.95.
        """
        t = make_transformer(pos_z=0.05)
        body = t.to_body_frame(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(body[2], 0.95, atol=1e-9,
            err_msg="Camera z offset not correctly subtracted from body z")

    def test_mounting_offset_forward(self):
        """
        Camera 3cm forward of CoM: pos_x=0.03.
        Tag directly below (camera z=1) → body z stays 1.
        The forward offset subtracts from body x, not body z.
        body = [0,0,1] - [0.03,0,0] = [-0.03, 0, 1].
        """
        t = make_transformer(pos_x=0.03)
        body = t.to_body_frame(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(body[0], -0.03, atol=1e-9,
            err_msg="Forward camera offset should subtract from body x")
        np.testing.assert_allclose(body[2],  1.0,  atol=1e-9)

    def test_yaw_90_rotates_axes(self):
        """Rotation should preserve vector magnitude."""
        t = make_transformer(yaw_deg=90.0)
        vec = np.array([1.0, 0.0, 0.0])
        body = t.to_body_frame(vec)
        np.testing.assert_allclose(np.linalg.norm(body), 1.0, atol=1e-9,
            err_msg="Rotation should preserve magnitude")

    def test_rotation_preserves_distance(self):
        """Frame rotation with zero offset should never change distance."""
        for yaw in [0, 45, 90, 180, 270]:
            t = make_transformer(yaw_deg=yaw)
            vec = np.array([0.5, 0.2, 3.0])
            body = t.to_body_frame(vec)
            np.testing.assert_allclose(
                np.linalg.norm(body), np.linalg.norm(vec), atol=1e-9,
                err_msg=f"Distance changed at yaw={yaw}"
            )

    def test_basis_vectors_are_orthogonal(self):
        """
        The three camera basis vectors should map to three
        orthogonal body vectors — rotation matrices preserve orthogonality.
        """
        t = make_transformer()
        bx = t.to_body_frame(np.array([1.0, 0.0, 0.0]))
        by = t.to_body_frame(np.array([0.0, 1.0, 0.0]))
        bz = t.to_body_frame(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(np.dot(bx, by), 0.0, atol=1e-9)
        np.testing.assert_allclose(np.dot(bx, bz), 0.0, atol=1e-9)
        np.testing.assert_allclose(np.dot(by, bz), 0.0, atol=1e-9)


class TestBodyToAngles:

    def setup_method(self):
        self.t = make_transformer()

    def test_directly_below_gives_zero_angles(self):
        angle_x, angle_y = self.t.body_to_angles(np.array([0.0, 0.0, 5.0]))
        np.testing.assert_allclose(angle_x, 0.0, atol=1e-9)
        np.testing.assert_allclose(angle_y, 0.0, atol=1e-9)

    def test_zero_vector_returns_zero(self):
        angle_x, angle_y = self.t.body_to_angles(np.array([0.0, 0.0, 0.0]))
        assert angle_x == 0.0 and angle_y == 0.0

    def test_forward_offset_gives_positive_angle_x(self):
        angle_x, angle_y = self.t.body_to_angles(np.array([1.0, 0.0, 5.0]))
        assert angle_x > 0
        np.testing.assert_allclose(angle_x, np.arctan2(1.0, 5.0), atol=1e-9)

    def test_right_offset_gives_positive_angle_y(self):
        angle_x, angle_y = self.t.body_to_angles(np.array([0.0, 0.5, 3.0]))
        assert angle_y > 0
        np.testing.assert_allclose(angle_y, np.arctan2(0.5, 3.0), atol=1e-9)

    def test_angles_in_radians(self):
        angle_x, _ = self.t.body_to_angles(np.array([1.0, 0.0, 1.0]))
        assert abs(angle_x) < np.pi

    def test_left_right_symmetry(self):
        _, angle_right = self.t.body_to_angles(np.array([0.0,  1.0, 5.0]))
        _, angle_left  = self.t.body_to_angles(np.array([0.0, -1.0, 5.0]))
        np.testing.assert_allclose(angle_right, -angle_left, atol=1e-9)

    def test_fore_aft_symmetry(self):
        angle_fwd, _ = self.t.body_to_angles(np.array([ 1.0, 0.0, 5.0]))
        angle_aft, _ = self.t.body_to_angles(np.array([-1.0, 0.0, 5.0]))
        np.testing.assert_allclose(angle_fwd, -angle_aft, atol=1e-9)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])