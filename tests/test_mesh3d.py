"""Exercise actual geometry and tracked grip math without a graphics device."""

from dataclasses import replace
import math
import unittest
import numpy as np

from air_violin.config import Calibration, Rect
from air_violin.mesh3d import (VIOLIN_GRIP, Pose, PoseTracker, make_bow_mesh,
                               make_marker_mesh, make_violin_mesh, model_matrix,
                               perspective_matrix)
from air_violin.types import Hand, Point, TrackingFrame


def posed_hand(label="Left", x=.30, confidence=.95, tilt=0., world=True):
    points = [Point(x, .50) for _ in range(21)]
    points[0] = Point(x, .62)
    points[2] = Point(x - .035, .49)
    points[5] = Point(x - .045, .47)
    points[9] = Point(x, .46)
    points[17] = Point(x + .045, .48)
    world_points = None
    if world:
        # Hand-centered metric data; no absolute position is encoded here.
        world_points = [Point(0., 0., 0.) for _ in range(21)]
        world_points[0] = Point(0., .045, tilt)
        world_points[5] = Point(-.023, -.030, -.25 * tilt)
        world_points[9] = Point(0., -.035, 0.)
        world_points[17] = Point(.023, -.025, .25 * tilt)
        world_points = tuple(world_points)
    return Hand(label, confidence, tuple(points), world_points)


class MeshGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.violin = make_violin_mesh()
        cls.bow = make_bow_mesh()
        cls.marker = make_marker_mesh()

    def test_models_have_finite_triangles_and_unit_surface_normals(self):
        for mesh in (self.violin, self.bow, self.marker):
            with self.subTest(vertices=len(mesh.vertices)):
                self.assertLess(len(mesh.vertices), 30_000)
                self.assertGreater(len(mesh.vertices), 100)
                self.assertEqual(mesh.vertices.dtype, np.float32)
                self.assertEqual(mesh.indices.dtype, np.uint32)
                self.assertEqual(len(mesh.indices) % 3, 0)
                self.assertTrue(np.isfinite(mesh.vertices).all())
                self.assertTrue(np.isfinite(mesh.normals).all())
                self.assertTrue(np.isfinite(mesh.colors).all())
                self.assertEqual(mesh.colors.shape, mesh.vertices.shape)
                self.assertEqual(mesh.normals.shape, mesh.vertices.shape)
                self.assertEqual(mesh.tags.shape, (len(mesh.vertices),))
                self.assertLess(int(mesh.indices.max()), len(mesh.vertices))
                self.assertTrue(((mesh.colors >= 0) & (mesh.colors <= 1)).all())
                np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1., atol=2e-6)
                triangles = mesh.vertices[mesh.indices].reshape(-1, 3, 3)
                cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                self.assertTrue((np.linalg.norm(cross, axis=1) > 1e-10).all())
                # Lighting normals must point in the triangle winding direction.
                self.assertTrue((np.sum(cross * mesh.normals[mesh.indices[::3]], axis=1) > 0).all())

    def test_violin_has_arched_front_back_ribs_and_neck(self):
        v = self.violin.vertices
        wood = v[self.violin.tags == -1]
        self.assertLess(float(v[:, 1].min()), -3.2)
        self.assertGreater(float(v[:, 1].max()), 1.4)
        self.assertLess(float(wood[:, 2].min()), -.18)
        self.assertGreater(float(wood[:, 2].max()), .28)
        # The back has its own outward normals, rather than a mirrored flat sprite.
        self.assertTrue((self.violin.normals[:, 2] < -.5).any())
        self.assertTrue((self.violin.normals[:, 2] > .5).any())
        self.assertTrue((np.abs(self.violin.normals[:, 2]) < .05).any())
        self.assertGreater(float(v[:, 0].max() - v[:, 0].min()), 1.3)

    def test_four_strings_are_distinct_raised_geometry_with_pitch_tags(self):
        centers = []
        for string in range(1, 5):
            selected = self.violin.vertices[self.violin.tags == string]
            self.assertGreater(len(selected), 20)
            self.assertLess(float(selected[:, 1].min()), -2.43)
            self.assertGreater(float(selected[:, 1].max()), 1.23)
            self.assertGreater(float(selected[:, 2].max()), .55)
            # Realistic string spacing widens from nut to bridge.
            centers.append(float(selected[selected[:, 1] < -2.439, 0].mean()))
        np.testing.assert_allclose(centers, (-.075, -.025, .025, .075), atol=1e-5)
        self.assertTrue(np.all(np.diff(centers) > .04))

    def test_full_size_proportions_and_smooth_body_lighting(self):
        # A 4/4 violin is approximately 590 mm long and 208 mm wide.
        span = np.ptp(self.violin.vertices, axis=0)
        self.assertGreater(float(span[1] / span[0]), 2.75)
        self.assertLess(float(span[1] / span[0]), 2.95)
        self.assertGreater(float(span[1]), 5.85)
        self.assertLess(float(span[1]), 5.98)
        # Smooth vertices on the same arched plate share their normals.
        front = (self.violin.tags == -1) & (self.violin.vertices[:, 2] > .15) & (self.violin.vertices[:, 1] > -.9)
        vertices, normals = self.violin.vertices[front], self.violin.normals[front]
        indices = np.flatnonzero(np.linalg.norm(vertices - (0., .66, .28), axis=1) < 1e-6)
        self.assertGreaterEqual(len(indices), 6)
        np.testing.assert_allclose(normals[indices], np.tile(normals[indices[0]], (len(indices), 1)), atol=1e-6)

    def test_bow_has_separate_wood_shaft_hair_and_thick_frog(self):
        self.assertGreater(float(self.bow.vertices[:, 0].max()), 5.5)
        wood = self.bow.vertices[self.bow.tags == -1]
        self.assertGreater(float(wood[:, 2].max()), .16)
        self.assertLess(float(self.bow.vertices[:, 2].min()), -.21)
        # White/ivory geometry includes real narrow hair below the wooden shaft.
        hair = self.bow.vertices[(self.bow.colors[:, 0] > .8) & (self.bow.vertices[:, 0] > 1)]
        self.assertTrue((hair[:, 2] < -.095).any())


class PoseMathTests(unittest.TestCase):
    def test_grip_attachment_projects_to_calibrated_pixels(self):
        pose = Pose(np.array((320., 190., 0.)), np.eye(3), 100., 1.)
        transform = model_matrix(pose, 880, 560, VIOLIN_GRIP)
        view = np.eye(4)
        view[2, 3] = -1100
        projected = perspective_matrix(880, 560) @ view @ transform @ np.r_[VIOLIN_GRIP, 1.]
        projected /= projected[3]
        pixels = ((projected[0] + 1) * 440, (1 - projected[1]) * 280)
        np.testing.assert_allclose(pixels, pose.position[:2], atol=1e-4)
        # Geometry +Y is downward on the camera display, exactly once.
        body = transform @ np.r_[np.asarray(VIOLIN_GRIP) + (0, 1, 0), 1.]
        grip = transform @ np.r_[VIOLIN_GRIP, 1.]
        self.assertLess(body[1], grip[1])

    def test_physical_hand_bindings_not_screen_position_or_mirror_guess(self):
        for mirrored in (True, False):
            cal = replace(Calibration(), mirrored=mirrored, left_label="Right", right_label="Left")
            physical_left = posed_hand("Right", x=.76)
            physical_right = posed_hand("Left", x=.23)
            frame = TrackingFrame(1., (physical_left, physical_right))
            violin = PoseTracker("violin").update(frame, cal, 1., (640, 480))
            bow = PoseTracker("bow").update(frame, cal, 1., (640, 480))
            for pose, hand in ((violin, physical_left), (bow, physical_right)):
                expected = ((hand.points[2].x + hand.points[5].x) * 320,
                            (hand.points[2].y + hand.points[5].y) * 240)
                np.testing.assert_allclose(pose.position[:2], expected)

    def test_letterboxed_camera_viewport_offsets_anchor_without_extra_mirroring(self):
        hand = posed_hand(x=.30)
        frame = TrackingFrame(1., (hand,))
        small = PoseTracker().update(frame, Calibration(), 1., (747, 560))
        placed = PoseTracker().update(frame, Calibration(), 1., (86, 64, 747, 560))
        np.testing.assert_allclose(placed.position, small.position + (86, 64, 0))
        np.testing.assert_allclose(placed.rotation, small.rotation)
        self.assertEqual(placed.scale, small.scale)

    def test_curling_fingertips_does_not_move_neck_grip_or_resize_instrument(self):
        hand = posed_hand()
        points = list(hand.points)
        for index in (8, 12, 16, 20):
            points[index] = Point(.2 + index / 200, .56, -.1)
        before = PoseTracker().update(TrackingFrame(1., (hand,)), Calibration(), 1., (640, 480))
        after = PoseTracker().update(TrackingFrame(1., (replace(hand, points=tuple(points)),)), Calibration(), 1., (640, 480))
        np.testing.assert_allclose(after.position, before.position)
        np.testing.assert_allclose(after.rotation, before.rotation)
        self.assertEqual(after.scale, before.scale)

    def test_bow_points_toward_left_grip_with_either_screen_layout(self):
        for left_x, right_x in ((.25, .75), (.75, .25)):
            frame = TrackingFrame(1., (posed_hand("Left", left_x), posed_hand("Right", right_x)))
            pose = PoseTracker("bow").update(frame, Calibration(), 1., (640, 480))
            screen_direction = np.array((pose.rotation[0, 0], -pose.rotation[1, 0]))
            self.assertGreater(screen_direction[0] * (left_x - right_x), 0)

    def test_violin_body_extends_toward_torso_in_both_physical_layouts(self):
        for left_x in (.25, .75):
            for mirrored in (True, False):
                cal = replace(Calibration(), mirrored=mirrored, left_label="Right", right_label="Left")
                frame = TrackingFrame(1., (posed_hand("Right", left_x),))
                pose = PoseTracker().update(frame, cal, 1., (640, 480))
                screen_axis = np.array((-pose.rotation[0, 1], pose.rotation[1, 1]))
                self.assertGreater(screen_axis[0] * (.5 - left_x), .15)
                self.assertGreater(abs(screen_axis[0]), abs(screen_axis[1]))
                self.assertGreater(screen_axis[1], .2)

    def test_palm_foreshortening_never_shrinks_violin_or_bow(self):
        left, right = posed_hand("Left", .25), posed_hand("Right", .75)
        narrowed = []
        for hand in (left, right):
            points = list(hand.points)
            points[17] = Point(points[5].x + .012, points[5].y + .005)
            narrowed.append(replace(hand, points=tuple(points)))
        for kind in ("violin", "bow"):
            before = PoseTracker(kind).update(TrackingFrame(1., (left, right)), Calibration(), 1., (640, 480))
            after = PoseTracker(kind).update(TrackingFrame(1., tuple(narrowed)), Calibration(), 1., (640, 480))
            self.assertEqual(before.scale, after.scale)
            larger = PoseTracker(kind).update(TrackingFrame(1., (left, right)), Calibration(), 1., (640, 480), size=1.3)
            self.assertAlmostEqual(larger.scale / before.scale, 1.3)

    def test_stale_future_duplicate_low_confidence_results_are_safe(self):
        hand = posed_hand()
        for frame in (TrackingFrame(0., (hand,)), TrackingFrame(2., (hand,)),
                      TrackingFrame(1., (replace(hand, confidence=.1),)),
                      TrackingFrame(1., (hand, hand))):
            self.assertIsNone(PoseTracker().update(frame, Calibration(), 1., (640, 480)))
        tracker = PoseTracker()
        first = tracker.update(TrackingFrame(1., (hand,)), Calibration(), 1., (640, 480))
        self.assertIs(tracker.update(TrackingFrame(1., (posed_hand(x=.9),)), Calibration(), 1.01, (640, 480)), first)
        self.assertIs(tracker.update(TrackingFrame(1.10), Calibration(), 1.10, (640, 480)), first)
        self.assertIsNone(tracker.update(TrackingFrame(1.20), Calibration(), 1.20, (640, 480)))

    def test_world_landmarks_add_tilt_without_changing_absolute_grip_position(self):
        cal = Calibration()
        flat_hand, tilted_hand = posed_hand(tilt=0.), posed_hand(tilt=.04)
        flat = PoseTracker().update(TrackingFrame(1., (flat_hand,)), cal, 1., (640, 480))
        tilted = PoseTracker().update(TrackingFrame(1., (tilted_hand,)), cal, 1., (640, 480))
        np.testing.assert_allclose(flat.position, tilted.position)
        self.assertGreater(float(np.linalg.norm(flat.rotation - tilted.rotation)), .15)
        np.testing.assert_allclose(tilted.rotation.T @ tilted.rotation, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.det(tilted.rotation)), 1., places=6)
        offset_world = tuple(Point(p.x + 2, p.y - 3, p.z + 4) for p in tilted_hand.world_points)
        offset = PoseTracker().update(TrackingFrame(1., (replace(tilted_hand, world_points=offset_world),)), cal, 1., (640, 480))
        np.testing.assert_allclose(offset.position, tilted.position)
        np.testing.assert_allclose(offset.rotation, tilted.rotation, atol=1e-6)

    def test_missing_or_corrupt_world_landmarks_use_flat_pose(self):
        base = posed_hand(world=False)
        expected = PoseTracker().update(TrackingFrame(1., (base,)), Calibration(), 1., (640, 480))
        corrupted = replace(base, world_points=tuple(Point(float("nan"), 0, 0) for _ in range(21)))
        actual = PoseTracker().update(TrackingFrame(1., (corrupted,)), Calibration(), 1., (640, 480))
        np.testing.assert_allclose(actual.rotation, expected.rotation)

    def test_rotation_smoothing_stays_proper_and_limits_sudden_flips(self):
        tracker = PoseTracker()
        hand = posed_hand()
        first = tracker.update(TrackingFrame(1., (hand,)), Calibration(), 1., (640, 480))
        points = list(hand.points)
        points[0], points[9] = points[9], points[0]
        flipped = tracker.update(TrackingFrame(1.033, (replace(hand, points=tuple(points)),)), Calibration(), 1.033, (640, 480))
        relative = first.rotation.T @ flipped.rotation
        angle = math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
        self.assertLessEqual(angle, .033 * 8 + 1e-6)
        np.testing.assert_allclose(flipped.rotation.T @ flipped.rotation, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.det(flipped.rotation)), 1., places=6)

    def test_invalid_pose_and_projection_are_rejected(self):
        for position, rotation, scale in (([0, 0], np.eye(3), 1), ([0, 0, 0], np.zeros((3, 3)), 1),
                                          ([0, 0, 0], np.eye(3), -1)):
            with self.assertRaises(ValueError):
                Pose(position, rotation, scale, 1.)
        for size in ((0, 480), (640, 0), (640, float("nan"))):
            with self.assertRaises(ValueError):
                perspective_matrix(*size)
        with self.assertRaises(ValueError):
            PoseTracker("unknown")


if __name__ == "__main__":
    unittest.main()
