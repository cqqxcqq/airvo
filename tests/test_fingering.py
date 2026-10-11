"""Synthetic articulation tests; these do not substitute for webcam trials."""

from dataclasses import replace
import math
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from air_violin.calibration import Calibrator, FINGER_STEPS, STEPS
from air_violin.config import Calibration, Controls, FingerCalibration, load_calibration, save_calibration, FINGER_FEATURE_VERSION
from air_violin.controller import Instrument
from air_violin.fingering import Fingering, BowStringSelector, finger_features
from air_violin.types import Hand, Point, TrackingFrame, mirror_point
from air_violin.vision import convert_result
from tests.helpers import hand, frame


def articulated(mask=0, label="Left", broken=None):
    world = [Point(0, 0, 0) for _ in range(21)]
    world[0] = Point(0, -0.06, 0)
    for index, x in enumerate((-0.030, -0.010, 0.010, 0.028)):
        base = 5+index*4
        world[base] = Point(x, 0, 0)
        angles = (0.35, 1.35, 2.45) if mask & (1 << index) else (0.15, 0.40, 0.65)
        previous = world[base]
        for offset, (length, angle) in enumerate(zip((0.030, 0.023, 0.018), angles), 1):
            point = Point(previous.x, previous.y+length*math.cos(angle), previous.z+length*math.sin(angle))
            world[base+offset] = previous = point
        if broken == index:
            world[base+2] = world[base+1]
    points = tuple(Point(0.30+p.x*2, 0.45-p.y*2, -p.z) for p in world)
    return Hand(label, 0.98, points, tuple(world))


def finger_calibration(**kwargs):
    return FingerCalibration(finger_features(articulated()),
                             finger_features(articulated(15)), **kwargs)


def hinged(mask=0, angle=0.22, label="Left"):
    """Lower each curved finger rigidly at MCP, with PIP/DIP angles unchanged."""
    source = articulated(0, label)
    world = list(source.world_points)
    for index in range(4):
        if not mask & (1 << index):
            continue
        base = 5+index*4
        origin = world[base]
        for joint in range(base+1, base+4):
            point = world[joint]
            y, z = point.y-origin.y, point.z-origin.z
            world[joint] = Point(point.x, origin.y+y*math.cos(angle)-z*math.sin(angle),
                                 origin.z+y*math.sin(angle)+z*math.cos(angle))
    return replace(source, world_points=tuple(world))


class GeometryTests(unittest.TestCase):
    def test_world_curl_changes_every_finger_feature(self):
        lifted = finger_features(articulated())
        pressed = finger_features(articulated(15))
        self.assertTrue(all(b-a > 0.06 for a, b in zip(lifted, pressed)))

    def test_rotation_scale_and_image_mirror_do_not_change_fingering(self):
        source = articulated(10)
        world = tuple(Point(-p.z*1.2, p.x*1.2, -p.y*1.2) for p in source.world_points)
        transformed = replace(source, world_points=world, points=tuple(mirror_point(p) for p in source.points))
        for a, b in zip(finger_features(source), finger_features(transformed)):
            self.assertAlmostEqual(a, b)

    def test_gentle_mcp_hinge_is_measurable_without_distal_curl(self):
        from air_violin.fingering import _angle, _distance
        source, down = hinged(), hinged(15)
        for base in (5, 9, 13, 17):
            for joint in (base+1, base+2):
                self.assertAlmostEqual(_angle(*source.world_points[joint-1:joint+2]),
                                       _angle(*down.world_points[joint-1:joint+2]))
            self.assertAlmostEqual(_distance(source.world_points[base], source.world_points[base+3]),
                                   _distance(down.world_points[base], down.world_points[base+3]))
        self.assertTrue(all(b-a >= .045 for a, b in zip(finger_features(source), finger_features(down))))

    def test_world_reflection_translation_preserve_gentle_press(self):
        source = hinged(9)
        reflected = replace(source, world_points=tuple(Point(-p.x+.4, p.y-.2, p.z+.1) for p in source.world_points))
        for a, b in zip(finger_features(source), finger_features(reflected)):
            self.assertAlmostEqual(a, b)

    def test_degenerate_palm_and_nonfinite_world_are_unknown(self):
        source = hinged()
        world = list(source.world_points)
        world[0] = world[9]
        self.assertEqual(finger_features(replace(source, world_points=tuple(world))), (None,)*4)
        world[0] = Point(float("nan"), 0, 0)
        self.assertEqual(finger_features(replace(source, world_points=tuple(world))), (None,)*4)

    def test_missing_world_and_collapsed_joint_are_unknown(self):
        self.assertEqual(finger_features(hand()), (None,)*4)
        result = finger_features(articulated(3, broken=2))
        self.assertIsNone(result[2])
        self.assertIsNotNone(result[0])

    def test_adapter_preserves_metric_world_and_capture_time(self):
        source = articulated()
        result = SimpleNamespace(hand_landmarks=[source.points], hand_world_landmarks=[source.world_points],
                                 handedness=[[SimpleNamespace(category_name="Left", score=0.98)]])
        converted = convert_result(result, 12, 20)
        self.assertEqual(converted.hands[0].world_points, source.world_points)
        self.assertEqual(converted.timestamp, 12)


class FingeringTests(unittest.TestCase):
    def test_all_sixteen_combinations_use_highest_finger_not_count(self):
        for mask in range(16):
            with self.subTest(mask=mask):
                fingering = Fingering(finger_calibration(dwell=0))
                expected = mask.bit_length()
                self.assertEqual(fingering.update(articulated(mask), 1), expected)

    def test_relaxed_curved_open_and_gentle_mcp_presses_play_all_combinations(self):
        calibration = FingerCalibration(finger_features(hinged()), finger_features(hinged(15)), dwell=0)
        for mask in range(16):
            with self.subTest(mask=mask):
                self.assertEqual(Fingering(calibration).update(hinged(mask), 1), mask.bit_length())

    def test_obsolete_feature_calibration_never_sounds_with_new_measurement(self):
        legacy = replace(finger_calibration(dwell=0), feature_version=1)
        fingers = Fingering(legacy)
        self.assertIsNone(fingers.update(articulated(15), 1))
        self.assertEqual(fingers.states, [-1]*4)

    def test_unknown_higher_finger_never_becomes_open(self):
        fingering = Fingering(finger_calibration(dwell=0))
        fingering.update(articulated(4), 1)
        self.assertEqual(fingering.update(articulated(4, broken=3), 1.03), 3)
        self.assertIsNone(fingering.expire(1.081))
        self.assertEqual(fingering.states[3], -1)

    def test_unknown_lower_finger_does_not_change_known_higher_stop(self):
        fingering = Fingering(finger_calibration(dwell=0))
        self.assertEqual(fingering.update(articulated(8, broken=0), 1), 4)

    def test_open_string_requires_all_four_confidently_lifted(self):
        fingering = Fingering(finger_calibration(dwell=0))
        self.assertIsNone(fingering.update(articulated(0, broken=0), 1))
        self.assertEqual(fingering.update(articulated(), 1.1), 0)

    def test_dwell_and_hysteresis_prevent_finger_flicker(self):
        fingering = Fingering(finger_calibration())
        self.assertIsNone(fingering.update(articulated(), 1))
        self.assertEqual(fingering.update(articulated(), 1.05), 0)
        self.assertEqual(fingering.update(articulated(4), 1.10), 0)
        self.assertEqual(fingering.update(articulated(4), 1.16), 3)
        # A calibrated intermediate value between thresholds retains the stop.
        source = articulated(4)
        cal = fingering.calibration
        midpoint = tuple((a+b)/2 for a, b in zip(cal.lifted, cal.pressed))
        from unittest.mock import patch
        with patch("air_violin.fingering.finger_features", return_value=midpoint):
            self.assertEqual(fingering.update(source, 1.20), 3)
            self.assertEqual(fingering.update(source, 1.26), 3)

    def test_old_frames_and_uncertainty_cannot_sustain(self):
        fingering = Fingering(finger_calibration(dwell=0))
        fingering.update(articulated(1), 2)
        self.assertEqual(fingering.update(articulated(8), 1), 1)
        fingering.uncertain(2.02)
        self.assertEqual(fingering.expire(2.079), 1)
        self.assertIsNone(fingering.expire(2.081))

    def test_confirmed_pose_survives_fifteen_fps_processing_age(self):
        fingering = Fingering(finger_calibration(dwell=0))
        fingering.update(articulated(1), 1)
        # A result can already be 84 ms old when displayed. Known geometry must
        # survive this; actual stale tracking is handled by Instrument instead.
        self.assertEqual(fingering.expire(1.084), 1)
        for timestamp in (1+index/15 for index in range(1, 10)):
            self.assertEqual(fingering.update(articulated(1), timestamp), 1)
            self.assertEqual(fingering.expire(timestamp+.084), 1)
        fingering.uncertain(timestamp+.01)
        self.assertIsNone(fingering.expire(timestamp+.081))


class FingerInstrumentTests(unittest.TestCase):
    def make(self, swapped=False):
        cal = replace(Calibration(version=2), fingers=finger_calibration(dwell=0))
        if swapped:
            cal = replace(cal, left_label="Right", right_label="Left")
        return Instrument(cal, Controls(), fingering_mode="fingers", string_override=1)

    def tracked(self, instrument, t, x, mask=0, left=True):
        hands = [hand(instrument.calibration.right_label, wrist=(x, 0.48))]
        if left:
            hands.append(articulated(mask, instrument.calibration.left_label))
        return TrackingFrame(t, tuple(hands))

    def test_pitch_changes_preserve_bow_gate_and_physical_binding(self):
        for swapped in (False, True):
            inst = self.make(swapped)
            inst.update(self.tracked(inst, 10, 0.60), 10)
            state = inst.update(self.tracked(inst, 10.04, 0.65), 10.04)
            self.assertTrue(state.audio.gate)
            self.assertEqual(state.audio.midi, 62)
            for mask, midi in ((1, 64), (3, 66), (7, 67), (15, 69), (7, 67), (3, 66), (1, 64), (0, 62)):
                for _ in range(3):
                    t = inst.last_frame+0.035
                    x = 0.65 + 0.04*math.sin((t-10)*15)
                    state = inst.update(self.tracked(inst, t, x, mask), t)
                self.assertEqual(state.audio.midi, midi)
                self.assertTrue(state.audio.gate)
            self.assertEqual(state.fingering_mode, "fingers")

    def test_gentle_knuckle_only_fingering_changes_d_string_pitch_during_bow(self):
        calibration = replace(Calibration(version=2), fingers=FingerCalibration(
            finger_features(hinged()), finger_features(hinged(15)), dwell=0))
        instrument = Instrument(calibration, Controls(), fingering_mode="fingers", string_override=1)
        timestamp = 1.0
        for mask, midi in ((0, 62), (1, 64), (3, 66), (7, 67), (15, 69), (7, 67), (3, 66), (1, 64), (0, 62)):
            for _ in range(4):
                timestamp += .04
                wrist_x = .72+.10*math.sin(timestamp*8)
                sample = TrackingFrame(timestamp, (hinged(mask), hand("Right", wrist=(wrist_x, .48))))
                state = instrument.update(sample, timestamp)
            self.assertEqual(state.audio.midi, midi)
            self.assertTrue(state.audio.gate)

    def test_tracking_loss_releases_within_eighty_ms_then_recovers(self):
        inst = self.make()
        inst.update(self.tracked(inst, 10, 0.60), 10)
        inst.update(self.tracked(inst, 10.04, 0.65), 10.04)
        state = inst.update(self.tracked(inst, 10.10, 0.70, left=False), 10.10)
        self.assertTrue(state.audio.gate)
        state = inst.update(self.tracked(inst, 10.13, 0.73, left=False), 10.13)
        self.assertFalse(state.audio.gate)
        self.assertIsNone(state.finger)  # Clear note preview while the release retains synth pitch.
        state = inst.update(self.tracked(inst, 10.17, 0.77, mask=3), 10.17)
        self.assertTrue(state.audio.gate)
        self.assertEqual(state.audio.midi, 66)

    def test_uncalibrated_fingers_cannot_sound(self):
        inst = Instrument(Calibration(), Controls(), fingering_mode="fingers")
        inst.update(self.tracked(inst, 1, 0.6), 1)
        self.assertFalse(inst.update(self.tracked(inst, 1.04, 0.65), 1.04).audio.gate)

    def test_eighty_four_ms_old_confirmed_pose_still_bows_then_stale_releases(self):
        inst = self.make()
        inst.update(self.tracked(inst, 10, .60, mask=1), 10)
        state = inst.update(self.tracked(inst, 10.04, .65, mask=1), 10.04)
        self.assertTrue(state.audio.gate)
        delayed = inst.update(None, 10.124)
        self.assertTrue(delayed.audio.gate)
        self.assertEqual(delayed.audio.midi, 64)
        self.assertFalse(inst.update(None, 10.221).audio.gate)

    def test_string_bands_use_dwell_and_keyboard_override(self):
        inst = self.make()
        inst.set_string_override(None)
        selector = inst.strings
        self.assertEqual(selector.update(Point(.7, .77), 1), 1)
        self.assertEqual(selector.update(Point(.7, .77), 1.06), 3)
        # Tiny movement over the A/E boundary keeps E selected.
        boundary = inst.calibration.bow.y0 + .74*(inst.calibration.bow.y1-inst.calibration.bow.y0)
        self.assertEqual(selector.update(Point(.7, boundary), 1.10), 3)
        inst.set_string_override(0)
        state = inst.update(self.tracked(inst, 2, .65, mask=1), 2)
        self.assertEqual(state.audio.midi, 57)


class FingerCalibrationTests(unittest.TestCase):
    def feed(self, wizard, h, start):
        for index in range(26):
            t = start+index*.04
            wizard.observe(TrackingFrame(t, (h,)), t)
        wizard.advance(start+1.0)

    def verify(self, wizard, start=10, label="Left"):
        for count, mask in enumerate((15, 7, 3, 1, 0)):
            for index in range(15):
                t = start+count*.6+index*.04
                wizard.observe(TrackingFrame(t, (articulated(mask, label),)), t)

    def test_old_config_and_new_config_round_trip(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/"calibration.json"
            old = Calibration()
            save_calibration(path, old)
            self.assertEqual(load_calibration(path, True), old)
            new = replace(old, version=2, fingers=finger_calibration())
            save_calibration(path, new)
            self.assertEqual(load_calibration(path, True), new)

    def test_legacy_feature_data_requires_recalibration_but_keeps_hand_binding(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/"calibration.json"
            old = replace(Calibration(version=2), left_label="Right", right_label="Left", fingers=finger_calibration())
            save_calibration(path, old)
            data = json.loads(path.read_text())
            del data["fingers"]["feature_version"]
            path.write_text(json.dumps(data))
            loaded = load_calibration(path, True)
            self.assertEqual(loaded.fingers.feature_version, 1)
            self.assertEqual(loaded.left_label, "Right")
            self.assertEqual(loaded.bow, old.bow)

    def test_future_calibration_frames_cannot_advance(self):
        wizard = Calibrator(True, mode="fingers")
        for t in (10, 10.05, 10.10):
            wizard.observe(TrackingFrame(t, (articulated(),)), 1)
        wizard.advance(1)
        self.assertEqual(wizard.step, 0)
        self.assertEqual(wizard.recent, [])
        # Defence in depth if a malformed producer bypasses observe().
        wizard.recent = [TrackingFrame(t, (articulated(),)) for t in (10, 10.05, 10.10)]
        wizard.advance(1)
        self.assertEqual(wizard.step, 0)

    def test_finger_wizard_omits_pointing_corners_and_preserves_world(self):
        wizard = Calibrator(True, mode="fingers")
        self.assertEqual(wizard.steps, FINGER_STEPS)
        self.assertEqual(Calibrator(True).steps, STEPS)
        self.feed(wizard, articulated(0, "Right"), 1)
        self.feed(wizard, hand("Left"), 2)
        self.feed(wizard, articulated(0, "Right"), 3)
        for step, mask in enumerate((1, 3, 7, 15), 4):
            self.feed(wizard, articulated(mask, "Right"), step)
        self.feed(wizard, articulated(15, "Right"), 8)
        self.assertTrue(wizard.verification_active)
        self.assertEqual(wizard.step, 7)
        self.verify(wizard, 10, "Right")
        self.assertEqual(wizard.step, 8)
        self.assertIsNone(wizard.value.fingers)  # Nothing replaces saved data until full success.
        self.assertEqual(wizard.value.left_label, "Right")
        self.feed(wizard, hand("Left", wrist=(.95, .8)), 14)
        self.feed(wizard, hand("Left", wrist=(.55, .2)), 16)
        self.assertEqual(wizard.step, 10)
        self.feed(wizard, hand("Left", pinch=1.5), 18)
        self.feed(wizard, hand("Left", pinch=.2), 20)
        self.feed(wizard, hand("Left"), 22)
        for i in range(1, 100):
            t = 23+i/30
            wizard.observe(TrackingFrame(t, (hand("Left", wrist=(.75+.14*math.sin(i/10), .5)),)), t)
        self.assertTrue(wizard.complete)
        self.assertEqual(wizard.value.fingers.feature_version, FINGER_FEATURE_VERSION)

    def test_indistinguishable_finger_poses_and_failed_release_rejected(self):
        wizard = Calibrator(True, mode="fingers")
        self.feed(wizard, articulated(), 1)
        self.feed(wizard, hand("Right"), 2)
        self.feed(wizard, articulated(), 3)
        self.feed(wizard, articulated(), 4)
        self.assertEqual(wizard.step, 3)
        self.assertIn("too similar", wizard.message)
        for step, mask in enumerate((1, 3, 7, 15), 5):
            self.feed(wizard, articulated(mask), step)
        self.feed(wizard, articulated(0), 10)
        self.assertEqual(wizard.step, 7)
        self.assertIn("all four fingers gently down", wizard.message)

    def test_three_frames_are_not_enough_to_calibrate_fingers(self):
        wizard = Calibrator(True, mode="fingers")
        wizard.step = 2
        for timestamp in (1, 1.04, 1.08):
            wizard.observe(TrackingFrame(timestamp, (hinged(),)), timestamp)
        wizard.advance(1.08)
        self.assertEqual(wizard.step, 2)
        self.assertIn("one second", wizard.message)

    def test_noisy_pose_window_cannot_be_saved(self):
        wizard = Calibrator(True, mode="fingers")
        wizard.step = 2
        for index in range(26):
            t = 1+index*.04
            pose = hinged(1, angle=.35) if index % 2 else hinged()
            wizard.observe(TrackingFrame(t, (pose,)), t)
        wizard.advance(2)
        self.assertEqual(wizard.step, 2)
        self.assertIsNone(wizard.lifted)
        self.assertIn("moving or noisy", wizard.message)

    def test_latest_missing_or_degenerate_metric_pose_cannot_save_old_geometry(self):
        for invalid in (replace(hinged(), world_points=None), articulated(broken=2)):
            with self.subTest(invalid_world=invalid.world_points is None):
                wizard = Calibrator(True, mode="fingers")
                wizard.step = 2
                for index in range(25):
                    timestamp = 1+index*.04
                    wizard.observe(TrackingFrame(timestamp, (hinged(),)), timestamp)
                wizard.observe(TrackingFrame(2, (invalid,)), 2)
                wizard.advance(2)
                self.assertEqual(wizard.step, 2)
                self.assertIsNone(wizard.lifted)
                self.assertIn("visible now", wizard.message)
                # One recovered frame cannot resurrect the old pose window.
                wizard.observe(TrackingFrame(2.04, (hinged(),)), 2.04)
                wizard.advance(2.04)
                self.assertEqual(wizard.step, 2)
                self.assertIn("one second", wizard.message)

    def test_wrong_cumulative_pose_cannot_advance(self):
        wizard = Calibrator(True, mode="fingers")
        wizard.step = 2
        self.feed(wizard, articulated(), 1)
        self.feed(wizard, articulated(15), 3)
        self.assertEqual(wizard.step, 3)
        self.assertIn("lifted", wizard.message)
        self.feed(wizard, articulated(1), 5)
        self.feed(wizard, articulated(2), 7)
        self.assertEqual(wizard.step, 4)
        self.assertIn("Keep INDEX down", wizard.message)

    def test_verification_cannot_pass_when_only_open_pose_is_detected(self):
        wizard = Calibrator(True, mode="fingers")
        wizard.step = 2
        self.feed(wizard, articulated(), 1)
        for step, mask in enumerate((1, 3, 7, 15), 3):
            self.feed(wizard, articulated(mask), step*2)
        self.feed(wizard, articulated(15), 16)
        for index in range(45):
            t = 18+index*.04
            wizard.observe(TrackingFrame(t, (articulated(),)), t)
        self.assertEqual(wizard.step, 7)
        self.assertEqual(wizard.verification_progress, 0)
        self.assertIn("target: 4", wizard.verification_note)

    def test_incomplete_recalibration_preserves_previous_finger_data(self):
        previous = replace(Calibration(version=2), fingers=finger_calibration())
        wizard = Calibrator(True, mode="fingers", initial=previous)
        wizard.step = 2
        self.feed(wizard, hinged(), 1)
        self.feed(wizard, hinged(1), 3)
        self.assertIs(wizard.value.fingers, previous.fingers)


class DeliveryTimingTests(unittest.TestCase):
    def test_delayed_thirty_fps_bow_survives_eight_ms_heartbeats(self):
        inst = Instrument(Calibration(), Controls())
        delay = .084
        for index in range(12):
            captured = 10+index/30
            delivered = captured+delay
            state = inst.update(frame(captured, .60+index*.020), delivered)
            if index:
                self.assertTrue(state.audio.gate)
                for tick in (.008, .016, .024):
                    self.assertTrue(inst.update(None, delivered+tick).audio.gate)

    def test_delivered_stopped_result_releases_despite_delay_allowance(self):
        inst = Instrument(Calibration(), Controls())
        inst.update(frame(10, .60), 10.084)
        self.assertTrue(inst.update(frame(10.04, .65), 10.124).audio.gate)
        # The source has genuinely stopped longer than reversal grace; receipt
        # must release immediately instead of replaying old motion grace.
        state = inst.update(frame(10.14, .65), 10.224)
        self.assertFalse(state.audio.gate)

    def test_frozen_result_still_expires_by_acquisition_watchdog(self):
        inst = Instrument(Calibration(), Controls())
        inst.update(frame(10, .60), 10.12)
        state = inst.update(frame(10.04, .65), 10.16)
        self.assertTrue(state.audio.gate)
        self.assertTrue(inst.update(None, 10.219).audio.gate)
        self.assertFalse(inst.update(None, 10.221).audio.gate)

    def test_zero_delay_retains_original_eighty_five_ms_grace(self):
        inst = Instrument(Calibration(), Controls())
        inst.update(frame(10, .60), 10)
        inst.update(frame(10.04, .65), 10.04)
        self.assertTrue(inst.update(None, 10.124).audio.gate)
        self.assertFalse(inst.update(None, 10.126).audio.gate)


if __name__ == "__main__":
    unittest.main()
