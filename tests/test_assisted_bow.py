"""Low-effort sustained phrases without weakening acquisition safety."""

import time
from types import SimpleNamespace
import unittest
from air_violin.audio import StateBus
from air_violin.capture import LatestValue
from air_violin.config import Calibration, Controls
from air_violin.controller import Instrument
from air_violin.gestures import BowInterpreter
from air_violin.runtime import ControlWorker
from air_violin.types import AudioState, PlayState
from tests.helpers import frame, hand


class AssistedBowTests(unittest.TestCase):
    def make(self):
        return BowInterpreter(Calibration(), Controls(), "assisted")

    def start(self, bow):
        bow.update(hand("Right", wrist=(.65, .5)), 1)
        return bow.update(hand("Right", wrist=(.68, .5)), 1.08)

    def test_small_deliberate_stroke_sustains_with_fresh_stationary_frames(self):
        bow = self.make()
        self.assertTrue(self.start(bow).sustaining)
        for index in range(1, 180):
            state = bow.update(hand("Right", wrist=(.68, .5)), 1.08+index/30)
            self.assertTrue(state.sustaining)
            self.assertGreaterEqual(state.intensity, .42)
        self.assertFalse(state.moving)  # Movement and musical sustain are separate.

    def test_stationary_and_alternating_jitter_do_not_latch(self):
        bow = self.make()
        bow.update(hand("Right", wrist=(.7, .5)), 1)
        for index in range(1, 100):
            x = .7 + (.002 if index % 2 else -.002)
            self.assertFalse(bow.update(hand("Right", wrist=(x, .5)), 1+index/30).sustaining)

    def test_pinch_stops_with_hysteresis_and_unpinch_needs_new_stroke(self):
        bow = self.make()
        self.start(bow)
        self.assertFalse(bow.update(hand("Right", wrist=(.68, .5), pinch=.30), 1.12).sustaining)
        self.assertFalse(bow.update(hand("Right", wrist=(.69, .5), pinch=.45), 1.16).engaged)
        self.assertFalse(bow.update(hand("Right", wrist=(.69, .5), pinch=.60), 1.20).sustaining)
        self.assertFalse(bow.update(hand("Right", wrist=(.69, .5)), 1.24).sustaining)
        self.assertTrue(bow.update(hand("Right", wrist=(.72, .5)), 1.30).sustaining)

    def test_resting_outside_bow_box_but_visible_preserves_phrase(self):
        bow = self.make()
        self.start(bow)
        for index, y in enumerate((.6, .7, .8, .9, .95), 1):
            state = bow.update(hand("Right", wrist=(.68, y)), 1.08+index*.04)
            self.assertTrue(state.sustaining)
            self.assertTrue(state.engaged)

    def test_out_of_camera_clears_phrase_then_reentry_cannot_resume(self):
        bow = self.make()
        self.start(bow)
        self.assertFalse(bow.update(hand("Right", wrist=(.68, 1.05)), 1.12).sustaining)
        self.assertFalse(bow.update(hand("Right", wrist=(.68, .5)), 1.16).sustaining)

    def test_tracking_gap_teleport_and_out_of_order_clear_phrase(self):
        for timestamp, wrist in ((1.5, (.68, .5)), (1.10, (.99, .5)), (1.04, (.68, .5))):
            with self.subTest(timestamp=timestamp, wrist=wrist):
                bow = self.make()
                self.start(bow)
                self.assertFalse(bow.update(hand("Right", wrist=wrist), timestamp).sustaining)

    def test_keyboard_start_requires_fresh_unpinched_hand_in_start_box(self):
        bow = self.make()
        self.assertFalse(bow.toggle_assisted(1))
        bow.update(hand("Right", wrist=(.7, .5)), 1)
        self.assertFalse(bow.toggle_assisted(1.2))
        bow.update(hand("Right", wrist=(.7, .5), pinch=.2), 1.22)
        self.assertFalse(bow.toggle_assisted(1.23))
        bow.update(hand("Right", wrist=(.7, .5), pinch=1), 1.26)
        self.assertTrue(bow.toggle_assisted(1.27))
        self.assertFalse(bow.toggle_assisted(1.28))
        bow.update(hand("Right", wrist=(.7, .95)), 1.30)
        self.assertFalse(bow.toggle_assisted(1.31))

    def test_stop_while_resting_cannot_restart_outside_start_box(self):
        bow = self.make()
        self.start(bow)
        bow.update(hand("Right", wrist=(.68, .95)), 1.12)
        self.assertFalse(bow.toggle_assisted(1.13))
        self.assertFalse(bow.toggle_assisted(1.14))

    def test_mode_switch_drops_latch_and_keeps_physical_modes(self):
        bow = self.make()
        self.start(bow)
        bow.set_mode("auto")
        self.assertFalse(bow.state.sustaining)
        self.assertFalse(bow.toggle_assisted(1.09))
        self.assertFalse(bow.update(hand("Right", wrist=(.68, .5)), 1.12).moving)


class AssistedInstrumentTests(unittest.TestCase):
    def make(self):
        return Instrument(Calibration(), Controls(), "assisted")

    def start(self, instrument):
        instrument.update(frame(10, .65), 10)
        return instrument.update(frame(10.08, .68), 10.08)

    def test_resting_phrase_and_pitch_transitions_keep_gate_open(self):
        instrument = self.make()
        original = self.start(instrument)
        self.assertTrue(original.audio.gate)
        for index in range(1, 90):
            t = 10.08+index/30
            state = instrument.update(frame(t, .68, finger=(.44, .43)), t)
            self.assertTrue(state.audio.gate)
            self.assertTrue(state.bow_sustaining)
            self.assertEqual(state.phase, PlayState.BOWING)
        self.assertNotEqual(state.audio.midi, original.audio.midi)

    def test_four_finger_melody_changes_pitch_without_more_bow_motion(self):
        from dataclasses import replace
        from air_violin.types import TrackingFrame
        from tests.test_fingering import articulated, finger_calibration
        calibration = replace(Calibration(version=2), fingers=finger_calibration(dwell=0))
        instrument = Instrument(calibration, Controls(), "assisted",
                                fingering_mode="fingers", string_override=1)
        def tracked(t, mask, x=.68):
            return TrackingFrame(t, (articulated(mask), hand("Right", wrist=(x, .5))))
        instrument.update(tracked(10, 0, .65), 10)
        self.assertTrue(instrument.update(tracked(10.08, 0), 10.08).audio.gate)
        for index, (mask, expected) in enumerate(((1, 64), (3, 66), (7, 67), (15, 69),
                                                  (7, 67), (3, 66), (1, 64), (0, 62)), 1):
            t = 10.08+index*.06
            state = instrument.update(tracked(t, mask), t)
            self.assertEqual(state.audio.midi, expected)
            self.assertTrue(state.audio.gate)
            self.assertTrue(state.bow_sustaining)

    def test_auto_string_selection_freezes_at_rest_and_changes_with_new_stroke_dwell(self):
        from dataclasses import replace
        from air_violin.types import TrackingFrame
        from tests.test_fingering import articulated, finger_calibration
        calibration = replace(Calibration(version=2), fingers=finger_calibration(dwell=0))
        instrument = Instrument(calibration, Controls(), "assisted", fingering_mode="fingers")
        def tracked(t, x, y):
            return TrackingFrame(t, (articulated(), hand("Right", wrist=(x, y))))
        instrument.update(tracked(10, .65, .45), 10)
        self.assertTrue(instrument.update(tracked(10.08, .68, .45), 10.08).audio.gate)
        # Even the reversal-grace tail must not mistake lowering for a stroke.
        for timestamp in (10.12, 10.16, 10.20):
            state = instrument.update(tracked(timestamp, .68, .74), timestamp)
            self.assertEqual(state.string, 1)
            self.assertEqual(state.audio.midi, 62)
            self.assertTrue(state.audio.gate)
        for timestamp, x in ((10.24, .70), (10.28, .72)):
            state = instrument.update(tracked(timestamp, x, .74), timestamp)
            self.assertEqual(state.string, 1)  # 40 ms is shorter than note dwell.
        state = instrument.update(tracked(10.32, .74, .74), 10.32)
        self.assertEqual(state.string, 3)
        self.assertEqual(state.audio.midi, 76)
        self.assertTrue(state.audio.gate)

    def test_unknown_fingering_releases_then_requires_new_phrase(self):
        from dataclasses import replace
        from air_violin.types import TrackingFrame
        from tests.test_fingering import articulated, finger_calibration
        calibration = replace(Calibration(version=2), fingers=finger_calibration(dwell=0))
        instrument = Instrument(calibration, Controls(), "assisted",
                                fingering_mode="fingers", string_override=1)
        def tracked(t, left=True):
            hands = [hand("Right", wrist=(.68, .5))]
            if left:
                hands.append(articulated())
            return TrackingFrame(t, tuple(hands))
        instrument.update(tracked(10), 10)
        self.assertTrue(instrument.toggle_assisted(10))
        self.assertTrue(instrument.update(None, 10).audio.gate)
        self.assertTrue(instrument.update(tracked(10.04, False), 10.04).audio.gate)
        self.assertFalse(instrument.update(tracked(10.09, False), 10.09).audio.gate)
        self.assertFalse(instrument.bow.state.sustaining)
        self.assertFalse(instrument.update(tracked(10.12), 10.12).audio.gate)

    def test_subthreshold_wrist_motion_cannot_produce_short_bursts(self):
        instrument = self.make()
        instrument.update(frame(10, .7), 10)
        for index in range(1, 30):
            t = 10+index/30
            x = .7+(.002 if index % 2 else -.002)
            self.assertFalse(instrument.update(frame(t, x), t).audio.gate)

    def test_acquisition_watchdog_expires_without_new_frames(self):
        instrument = self.make()
        self.start(instrument)
        self.assertTrue(instrument.update(None, 10.25).audio.gate)
        self.assertFalse(instrument.update(None, 10.261).audio.gate)
        # Both hands reacquired at rest: a new explicit start is necessary.
        self.assertFalse(instrument.update(frame(10.30, .68), 10.30).audio.gate)

    def test_one_lost_hand_releases_and_reacquisition_does_not_resume(self):
        for missing in ("left", "right"):
            with self.subTest(missing=missing):
                instrument = self.make()
                self.start(instrument)
                state = instrument.update(frame(10.30, .68, **{missing: False}), 10.30)
                self.assertFalse(state.audio.gate)
                self.assertFalse(instrument.update(frame(10.34, .68), 10.34).audio.gate)

    def test_old_frame_cannot_continue_assisted_latch(self):
        instrument = self.make()
        self.start(instrument)
        self.assertFalse(instrument.update(frame(10.04, .69), 10.10).audio.gate)

    def test_delayed_fresh_capture_sustains_without_motion_grace_dependency(self):
        instrument = self.make()
        instrument.update(frame(10, .65), 10.084)
        self.assertTrue(instrument.update(frame(10.08, .68), 10.164).audio.gate)
        for index in range(1, 20):
            captured = 10.08+index/30
            delivered = captured+.084
            self.assertTrue(instrument.update(frame(captured, .68), delivered).audio.gate)
            self.assertTrue(instrument.update(None, delivered+.024).audio.gate)


class AssistedOwnershipTests(unittest.TestCase):
    def test_camera_keyboard_toggle_publishes_immediately_and_respects_ownership(self):
        mailbox, bus = LatestValue(), StateBus()
        worker = ControlWorker(SimpleNamespace(latest=mailbox), bus, Calibration(),
                               Controls(), "assisted", .65)
        try:
            # A disabled camera cannot overwrite keyboard ownership.
            demo = AudioState(time.perf_counter(), 69, .5, True)
            bus.publish(demo)
            self.assertFalse(worker.toggle_assisted())
            self.assertEqual(bus.read_nowait(AudioState(0)), demo)
            worker.configure(Calibration(), "assisted", .65, True)
            now = time.perf_counter()
            mailbox.put(frame(now, .7))
            self.assertTrue(worker.toggle_assisted())
            self.assertTrue(bus.read_nowait(AudioState(0)).gate)
            worker.stop_assisted()
            self.assertFalse(bus.read_nowait(AudioState(0)).gate)
            self.assertTrue(worker.toggle_assisted())
            worker.configure(Calibration(), "auto", .65, True)
            self.assertFalse(worker.instrument.bow.state.sustaining)
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
