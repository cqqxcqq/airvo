from dataclasses import replace
import math
import tempfile
from pathlib import Path
import unittest
from air_violin.calibration import Calibrator
from air_violin.config import Calibration, Controls, Rect, load_calibration, save_calibration
from air_violin.controller import Instrument
from air_violin.demo import KeyboardInstrument
from air_violin.gestures import BowInterpreter
from air_violin.mapping import EASY_OFFSETS, OPEN_MIDI, Fingerboard, midi_frequency, note_name
from air_violin.types import Point, PlayState, TrackingFrame, mirror_point
from tests.helpers import hand, frame


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.cal = Calibration()
        self.controls = Controls(position_tau=0.005)
        self.board = Fingerboard(self.cal.board, self.controls)

    def point(self, s, z):
        r = self.cal.board
        return Point(r.x0 + (z + 0.5) / 5 * (r.x1 - r.x0),
                     r.y0 + (s + 0.5) / 4 * (r.y1 - r.y0))

    def test_all_twenty_note_zones(self):
        for s in range(4):
            for z in range(5):
                with self.subTest(string=s, zone=z):
                    self.board.reset()
                    self.assertEqual(self.board.update(self.point(s, z), 1), (s, z))
                    self.assertEqual(self.board.midi, OPEN_MIDI[s] + EASY_OFFSETS[z])

    def test_equal_temperament_and_names(self):
        self.assertEqual(midi_frequency(69), 440)
        self.assertAlmostEqual(midi_frequency(55), 195.99771799)
        self.assertAlmostEqual(midi_frequency(81), 880)
        self.assertEqual(note_name(55), "G3")
        self.assertEqual(note_name(76), "E5")

    def test_mirror_is_involution_preserving_y_and_depth(self):
        point = Point(0.2, 0.4, -0.1)
        mirrored = mirror_point(point)
        self.assertAlmostEqual(mirrored.x, 0.8)
        self.assertEqual((mirrored.y, mirrored.z), (point.y, point.z))
        self.assertAlmostEqual(mirror_point(mirrored).x, point.x)

    def test_mirrored_board_selects_same_physical_zone_when_recalibrated(self):
        original = Rect(0.05, 0.2, 0.45, 0.8)
        flipped = Rect(0.55, 0.2, 0.95, 0.8)
        # Horizontal ordering is explicitly screen-left -> screen-right.
        a = Fingerboard(original, self.controls)
        b = Fingerboard(flipped, self.controls)
        point = Point(0.09, 0.275)
        self.assertEqual(a.update(point, 1), (0, 0))
        self.assertEqual(b.update(mirror_point(point), 1), (0, 4))

    def test_boundary_jitter_stays_in_current_zone(self):
        self.board.update(self.point(1, 0), 1)
        r = self.cal.board
        for i, u in enumerate((0.205, 0.195, 0.21, 0.219, 0.198)):
            p = Point(r.x0 + u * (r.x1 - r.x0), self.point(1, 0).y)
            self.assertEqual(self.board.update(p, 1.1 + i * 0.1), (1, 0))

    def test_deliberate_change_requires_dwell(self):
        self.board.update(self.point(1, 0), 1)
        self.assertEqual(self.board.update(self.point(1, 1), 1.1), (1, 0))
        self.assertEqual(self.board.update(self.point(1, 1), 1.13), (1, 0))
        self.assertEqual(self.board.update(self.point(1, 1), 1.17), (1, 1))

    def test_string_hysteresis_and_dwell(self):
        p = self.point(0, 2)
        self.board.update(p, 1)
        r = self.cal.board
        jitter = Point(p.x, r.y0 + 0.265 * (r.y1 - r.y0))
        self.assertEqual(self.board.update(jitter, 1.1), (0, 2))
        self.board.update(self.point(1, 2), 1.2)
        self.assertEqual(self.board.update(self.point(1, 2), 1.3), (1, 2))

    def test_outside_region_clears_note(self):
        self.board.update(self.point(2, 2), 1)
        self.assertIsNone(self.board.update(Point(0, 0), 1.1))
        self.assertIsNone(self.board.midi)

    def test_smoothing_uses_time_not_frame_count(self):
        a = Fingerboard(self.cal.board, Controls())
        b = Fingerboard(self.cal.board, Controls())
        start, end = self.point(1, 0), self.point(1, 3)
        a.update(start, 1); b.update(start, 1)
        a.update(end, 1.1)
        for i in range(1, 6):
            b.update(end, 1 + i * 0.02)
        self.assertAlmostEqual(a.point.x, b.point.x)

    def test_out_of_order_points_ignored(self):
        self.board.update(self.point(0, 0), 2)
        self.board.update(self.point(3, 4), 1)
        self.assertEqual(self.board.selected, (0, 0))


class BowTests(unittest.TestCase):
    def setUp(self):
        self.bow = BowInterpreter(Calibration(), Controls())

    def test_velocity_uses_actual_elapsed_time(self):
        self.bow.update(hand("Right", wrist=(0.6, 0.5)), 1)
        state = self.bow.update(hand("Right", wrist=(0.65, 0.5)), 1.1)
        self.assertAlmostEqual(state.raw_velocity, 0.5)
        self.assertTrue(state.moving)
        self.assertEqual(state.direction, 1)

    def test_stationary_hand_never_starts_sound(self):
        self.bow.update(hand("Right"), 1)
        self.assertFalse(self.bow.update(hand("Right"), 1.05).moving)

    def test_small_alternating_wrist_jitter_does_not_start_sound(self):
        self.bow.update(hand("Right", wrist=(0.7, 0.5)), 1)
        for i in range(1, 30):
            x = 0.7 + (0.001 if i % 2 else -0.001)
            state = self.bow.update(hand("Right", wrist=(x, 0.5)), 1 + i * 0.03)
            self.assertFalse(state.moving)

    def test_pinching_has_separate_engage_release_thresholds(self):
        self.bow.set_mode("precision")
        self.assertFalse(self.bow.update(hand("Right", pinch=1), 1).engaged)
        self.assertTrue(self.bow.update(hand("Right", pinch=0.3), 1.03).engaged)
        self.assertTrue(self.bow.update(hand("Right", pinch=0.45), 1.06).engaged)
        self.assertFalse(self.bow.update(hand("Right", pinch=0.6), 1.09).engaged)

    def test_short_reversal_pause_sustains_but_stop_releases(self):
        self.bow.update(hand("Right", wrist=(0.6, 0.5)), 1)
        self.bow.update(hand("Right", wrist=(0.65, 0.5)), 1.04)
        self.assertTrue(self.bow.update(hand("Right", wrist=(0.65, 0.5)), 1.08).moving)
        reverse = self.bow.update(hand("Right", wrist=(0.61, 0.5)), 1.12)
        self.assertTrue(reverse.moving)
        self.assertEqual(reverse.direction, -1)
        self.assertFalse(self.bow.update(hand("Right", wrist=(0.61, 0.5)), 1.22).moving)

    def test_region_exit_disengages_immediately(self):
        self.bow.update(hand("Right", wrist=(0.8, 0.5)), 1)
        self.bow.update(hand("Right", wrist=(0.84, 0.5)), 1.04)
        self.assertFalse(self.bow.update(hand("Right", wrist=(0.97, 0.5)), 1.08).moving)

    def test_tracking_gap_reacquires_without_velocity_spike(self):
        self.bow.update(hand("Right", wrist=(0.6, 0.5)), 1)
        state = self.bow.update(hand("Right", wrist=(0.85, 0.5)), 1.5)
        self.assertEqual(state.raw_velocity, 0)
        self.assertFalse(state.moving)

    def test_teleport_and_old_frames_cannot_trigger_bow(self):
        self.bow.update(hand("Right", wrist=(0.6, 0.5)), 1)
        self.assertFalse(self.bow.update(hand("Right", wrist=(0.9, 0.5)), 1.01).moving)
        self.assertFalse(self.bow.update(hand("Right", wrist=(0.8, 0.5)), 0.9).moving)


class StateTests(unittest.TestCase):
    def setUp(self):
        self.instrument = Instrument(Calibration(), Controls())

    def start_bow(self):
        self.instrument.update(frame(10, 0.65), 10)
        return self.instrument.update(frame(10.04, 0.70), 10.04)

    def test_idle_ready_bowing_releasing_ready(self):
        self.assertEqual(self.instrument.update(None, 9).phase, PlayState.IDLE)
        self.assertEqual(self.instrument.update(frame(10, 0.65), 10).phase, PlayState.READY)
        self.assertEqual(self.instrument.update(frame(10.04, 0.7), 10.04).phase, PlayState.BOWING)
        self.assertEqual(self.instrument.update(frame(10.14, 0.7), 10.14).phase, PlayState.RELEASING)
        self.instrument.update(frame(10.30, 0.7), 10.30)
        self.assertEqual(self.instrument.update(frame(10.41, 0.7), 10.41).phase, PlayState.READY)

    def test_tracking_loss_releases_then_recovers(self):
        self.assertTrue(self.start_bow().audio.gate)
        lost = self.instrument.update(frame(10.3, left=False, right=False), 10.3)
        self.assertFalse(lost.audio.gate)
        self.assertFalse(lost.left_fresh)
        self.assertEqual(lost.phase, PlayState.RELEASING)
        self.assertEqual(self.instrument.update(None, 10.6).phase, PlayState.IDLE)
        self.assertFalse(self.instrument.update(frame(10.7, 0.7), 10.7).audio.gate)
        self.assertTrue(self.instrument.update(frame(10.74, 0.75), 10.74).audio.gate)

    def test_one_missing_hand_cannot_sustain_indefinitely(self):
        self.start_bow()
        state = self.instrument.update(frame(10.3, 0.74, left=False), 10.3)
        self.assertFalse(state.audio.gate)

    def test_note_changes_while_bowing_keep_gate_open(self):
        self.start_bow()
        old = self.instrument.board.midi
        for t, x in ((10.08, 0.74), (10.12, 0.78), (10.16, 0.82), (10.20, 0.86)):
            state = self.instrument.update(frame(t, x, finger=(0.44, 0.43)), t)
            self.assertTrue(state.audio.gate)
        self.assertNotEqual(state.audio.midi, old)

    def test_stale_future_and_repeated_frames_rejected(self):
        self.assertEqual(self.instrument.update(frame(1), 10).phase, PlayState.IDLE)
        self.assertEqual(self.instrument.update(frame(20), 10).phase, PlayState.IDLE)
        self.start_bow()
        self.assertFalse(self.instrument.update(frame(10.04, 0.85), 10.4).audio.gate)

    def test_duplicate_handedness_low_confidence_and_nan_rejected(self):
        duplicate = TrackingFrame(1, (hand("Left"), hand("Left")))
        self.assertIsNone(duplicate.hand("Left"))
        self.assertIsNone(TrackingFrame(1, (hand(confidence=0.2),)).hand("Left"))
        invalid = hand(finger=(math.nan, 0.5))
        self.assertIsNone(TrackingFrame(1, (invalid,)).hand("Left"))

    def test_physical_hand_binding_can_be_swapped(self):
        cal = replace(Calibration(), left_label="Right", right_label="Left")
        inst = Instrument(cal, Controls())
        tracked = TrackingFrame(1, (hand("Right", finger=(0.12, 0.3)), hand("Left")))
        inst.update(tracked, 1)
        self.assertEqual(inst.board.midi, 55)


class CalibrationTests(unittest.TestCase):
    def test_persistence_and_mirror_change(self):
        cal = replace(Calibration(), left_label="Right", right_label="Left")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "calibration.json"
            save_calibration(path, cal)
            self.assertEqual(load_calibration(path, True), cal)
            self.assertIsNone(load_calibration(path, False))

    def test_invalid_boundaries_and_thresholds(self):
        for args in ((0.8, 0.2, 0.2, 0.8), (0, 0, 0.02, 0.9), (0, 0, math.nan, 0.9)):
            with self.assertRaises(ValueError):
                Rect(*args)
        with self.assertRaises(ValueError):
            replace(Calibration(), pinch_on=1, pinch_off=0.5)
        with self.assertRaises(ValueError):
            replace(Calibration(), bow_full_speed=math.nan)

    def feed(self, wizard, h, start=1):
        for i in range(6):
            wizard.observe(TrackingFrame(start + i * 0.04, (h,)), start + i * 0.04)
        wizard.advance(start + 0.20)

    def test_wizard_resolves_physical_identity_not_screen_side(self):
        wizard = Calibrator(True)
        self.feed(wizard, hand("Right", wrist=(0.8, 0.5)))
        self.assertEqual(wizard.value.left_label, "Right")
        self.feed(wizard, hand("Left", wrist=(0.2, 0.5)), 2)
        self.assertEqual(wizard.step, 2)
        self.assertEqual(wizard.value.right_label, "Left")

    def test_stable_low_frame_rate_samples_can_calibrate(self):
        wizard = Calibrator(True)
        for t in (1, 1.1, 1.2):
            wizard.observe(TrackingFrame(t, (hand("Left"),)), t)
        wizard.advance(1.2)
        self.assertEqual(wizard.step, 1)

    def test_wizard_rejects_same_hand_and_stale_samples(self):
        wizard = Calibrator(True)
        self.feed(wizard, hand("Right"))
        self.feed(wizard, hand("Right"), 2)
        self.assertEqual(wizard.step, 1)
        wizard.advance(5)
        self.assertEqual(wizard.step, 1)

    def test_complete_wizard_calibrates_regions_pinches_and_speed(self):
        wizard = Calibrator(True)
        self.feed(wizard, hand("Left"), 1)
        self.feed(wizard, hand("Right"), 2)
        self.feed(wizard, hand("Left", finger=(0.05, 0.2)), 3)
        self.feed(wizard, hand("Left", finger=(0.45, 0.8)), 4)
        self.feed(wizard, hand("Right", wrist=(0.55, 0.2)), 5)
        self.feed(wizard, hand("Right", wrist=(0.95, 0.8)), 6)
        self.feed(wizard, hand("Right", pinch=1.5), 7)
        self.feed(wizard, hand("Right", pinch=0.2), 8)
        self.feed(wizard, hand("Right"), 9)
        self.assertIsNotNone(wizard.started)
        for i in range(1, 100):
            t = 9.2 + i / 30
            x = 0.75 + 0.14 * math.sin(i / 10)
            wizard.observe(TrackingFrame(t, (hand("Right", wrist=(x, 0.5)),)), t)
        self.assertTrue(wizard.complete)
        self.assertEqual(wizard.value.board, Rect(0.05, 0.2, 0.45, 0.8))
        self.assertLess(wizard.value.pinch_on, wizard.value.pinch_off)
        self.assertGreater(wizard.value.bow_full_speed, 0.15)

    def test_disappeared_hand_cannot_mark_a_corner(self):
        wizard = Calibrator(True)
        for i in range(6):
            wizard.observe(TrackingFrame(1 + i * .04, (hand(),)), 1 + i * .04)
        wizard.observe(TrackingFrame(1.22), 1.22)
        wizard.advance(1.22)
        self.assertEqual(wizard.step, 0)


class DemoTests(unittest.TestCase):
    def test_keyboard_note_string_and_safe_stop(self):
        demo = KeyboardInstrument()
        demo.key(ord("3"), 1)
        demo.key(ord("]"), 1)
        demo.key(ord(" "), 1)
        self.assertEqual(demo.update(1, 0.5).audio.midi, 73)
        self.assertTrue(demo.update(1, 0.5).audio.gate)
        demo.key(ord("0"), 1.1)
        self.assertFalse(demo.update(1.1, 0.5).audio.gate)

    def test_short_stroke_expires_without_key_release(self):
        demo = KeyboardInstrument()
        demo.key(13, 1)
        self.assertTrue(demo.update(1.1, 0.5).audio.gate)
        self.assertFalse(demo.update(1.8, 0.5).audio.gate)


if __name__ == "__main__":
    unittest.main()
