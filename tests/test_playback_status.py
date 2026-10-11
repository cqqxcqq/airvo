"""Playback guidance follows actual control output, not decorative AR anchors."""

from dataclasses import replace
import unittest

import cv2

from air_violin.config import Calibration, Controls
from air_violin.controller import Instrument
from air_violin.demo import KeyboardInstrument
from air_violin.render import playback_status
from air_violin.types import TrackingFrame
from tests.helpers import hand
from tests.test_fingering import articulated, finger_calibration


class PlaybackStatusTests(unittest.TestCase):
    def setUp(self):
        self.calibration = replace(Calibration(version=2),
                                   fingers=finger_calibration(dwell=0))

    def make(self, mode="assisted"):
        return Instrument(self.calibration, Controls(), mode,
                          fingering_mode="fingers", string_override=1)

    def tracked(self, timestamp, *, pinch=1.0, wrist=(.7, .5),
                left=True, right=True, invalid_left=False):
        hands = []
        if left:
            left_hand = articulated()
            hands.append(replace(left_hand, world_points=None) if invalid_left else left_hand)
        if right:
            hands.append(hand("Right", wrist=wrist, pinch=pinch))
        return TrackingFrame(timestamp, tuple(hands))

    def status(self, state, mode="assisted", demo=False):
        return playback_status(state, self.calibration, mode, demo)

    def test_assisted_pinch_blocks_sound_and_reports_open_hand(self):
        instrument = self.make()
        instrument.update(self.tracked(10, pinch=.2), 10)
        self.assertFalse(instrument.toggle_assisted(10))
        state = instrument.update(None, 10)
        self.assertEqual(state.audio.midi, 62)
        self.assertFalse(state.audio.gate)
        self.assertEqual(self.status(state), "Separate right thumb/index")

    def test_outside_start_area_reports_region_even_with_visible_bow_hand(self):
        instrument = self.make()
        state = instrument.update(self.tracked(10, wrist=(.4, .5)), 10)
        self.assertTrue(state.right_fresh)
        self.assertFalse(instrument.toggle_assisted(10))
        self.assertEqual(self.status(state), "Move right wrist into bow area")

    def test_detector_teleport_does_not_report_an_unpinched_hand_as_pinched(self):
        instrument = self.make()
        instrument.update(self.tracked(10, wrist=(.55, .5)), 10)
        state = instrument.update(self.tracked(10.04, wrist=(.9, .5)), 10.04)
        self.assertTrue(state.right_fresh)
        self.assertFalse(state.engaged)
        self.assertGreaterEqual(state.pinch, self.calibration.pinch_off)
        self.assertEqual(self.status(state), "Try a fresh right-hand stroke")

    def test_ready_and_sustaining_report_the_actions_that_are_available(self):
        instrument = self.make()
        state = instrument.update(self.tracked(10), 10)
        self.assertEqual(self.status(state), "Short stroke / Space starts")
        self.assertTrue(instrument.toggle_assisted(10))
        state = instrument.update(None, 10)
        self.assertTrue(state.audio.gate)
        self.assertEqual(self.status(state), "Sustaining: pinch / Space stops")
        # Resting outside the start box is allowed once a phrase is latched.
        state = instrument.update(self.tracked(10.04, wrist=(.7, .95)), 10.04)
        self.assertTrue(state.audio.gate)
        self.assertEqual(self.status(state), "Sustaining: pinch / Space stops")

    def test_unknown_fingers_do_not_masquerade_as_retained_note(self):
        instrument = self.make()
        instrument.update(self.tracked(10), 10)
        self.assertTrue(instrument.toggle_assisted(10))
        state = instrument.update(self.tracked(10.09, invalid_left=True), 10.09)
        self.assertEqual(state.audio.midi, 62)  # The release voice keeps its pitch.
        self.assertIsNone(state.finger)
        self.assertFalse(state.audio.gate)
        self.assertEqual(self.status(state), "Left note unclear; show fingers")

    def test_missing_or_stale_hands_report_the_physical_hand_to_show(self):
        cases = ((False, False, "Show both hands to the camera"),
                 (False, True, "Show your left hand to the camera"),
                 (True, False, "Show your right hand to the camera"))
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                state = self.make().update(self.tracked(10, left=left, right=right), 10)
                self.assertEqual(self.status(state), expected)
        instrument = self.make()
        instrument.update(self.tracked(10), 10)
        self.assertEqual(self.status(instrument.update(None, 10.19)),
                         "Show both hands to the camera")

    def test_precision_has_opposite_pinch_action_and_auto_needs_movement(self):
        precision = self.make("precision")
        state = precision.update(self.tracked(10), 10)
        self.assertEqual(self.status(state, "precision"), "Pinch right thumb/index to bow")
        state = precision.update(self.tracked(10.04, pinch=.2), 10.04)
        self.assertEqual(self.status(state, "precision"), "Sweep your right hand to bow")
        state = precision.update(self.tracked(10.08, pinch=.2, wrist=(.73, .5)), 10.08)
        self.assertTrue(state.audio.gate)
        self.assertEqual(self.status(state, "precision"), "Bowing")
        auto = self.make("auto")
        self.assertEqual(self.status(auto.update(self.tracked(10), 10), "auto"),
                         "Sweep your right hand to bow")

    def test_keyboard_guidance_is_independent_of_camera_and_finger_freshness(self):
        keyboard = KeyboardInstrument()
        state = keyboard.update(10, .65)
        self.assertEqual(self.status(state, demo=True), "Space sustains / Enter bows")
        keyboard.key(ord(" "), 10)
        state = keyboard.update(10, .65)
        self.assertTrue(state.audio.gate)
        self.assertEqual(self.status(state, demo=True), "Keyboard sounding / Space stops")

    def test_guidance_fits_the_existing_performance_strip(self):
        messages = ("Keyboard sounding / Space stops", "Space sustains / Enter bows",
                    "Show both hands to the camera", "Show your left hand to the camera",
                    "Show your right hand to the camera", "Left note unclear; show fingers",
                    "Point left index inside note area", "Sustaining: pinch / Space stops",
                    "Bowing", "Move right wrist into bow area", "Separate right thumb/index",
                    "Pinch right thumb/index to bow", "Try a fresh right-hand stroke",
                    "Short stroke / Space starts", "Sweep your right hand to bow")
        for message in messages:
            with self.subTest(message=message):
                width = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, .41, 1)[0][0]
                self.assertLess(width, 435-124)

    def test_inactive_bow_meter_does_not_show_a_colored_line(self):
        from air_violin.audio import AudioEngine, StateBus
        from air_violin.render import Renderer
        state = self.make().update(self.tracked(10), 10)
        canvas = Renderer(quality='low', reduced_motion=True).draw(
            None, state, self.calibration, None, None, False, 'assisted', AudioEngine(StateBus()))
        self.assertEqual(tuple(canvas[585, 480]), (60, 57, 52))


if __name__ == "__main__":
    unittest.main()
