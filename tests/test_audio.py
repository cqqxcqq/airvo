from types import SimpleNamespace
import unittest
import numpy as np
from air_violin.audio import AudioEngine, StateBus, Synthesizer, exponential_curve
from air_violin.types import AudioState


class AudioTests(unittest.TestCase):
    def test_attack_and_release_envelopes(self):
        attack = exponential_curve(0, 1, 4800, 0.015, 48000)
        self.assertTrue(np.all(np.diff(attack) > 0))
        self.assertLess(attack[0], 0.002)
        self.assertGreater(attack[-1], 0.998)
        release = exponential_curve(1, 0, 48000, 0.065, 48000)
        self.assertTrue(np.all(np.diff(release) < 0))
        self.assertLess(release[-1], 1e-6)

    def test_sustained_note_has_correct_fundamental(self):
        synth = Synthesizer()
        state = AudioState(1, 69, 0.8, True, 0.65)
        synth.render(4800, state, 1)
        audio = synth.render(48000, state, 1)
        spectrum = abs(np.fft.rfft(audio * np.hanning(len(audio))))
        fundamental = np.fft.rfftfreq(len(audio), 1 / 48000)[np.argmax(spectrum)]
        self.assertAlmostEqual(fundamental, 440, delta=1)
        # More than a single sine: a distinct second harmonic is present.
        self.assertGreater(spectrum[880], 0.2 * spectrum[440])

    def test_maximum_gain_is_finite_and_does_not_clip(self):
        synth = Synthesizer()
        audio = synth.render(48000, AudioState(1, 76, 1, True, 1), 1)
        self.assertEqual(audio.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(audio)))
        self.assertLess(float(abs(audio).max()), 0.67)

    def test_pitch_transition_does_not_reset_phase_or_envelope(self):
        synth = Synthesizer()
        before = synth.render(4800, AudioState(1, 69, 0.8, True), 1)
        phase_before, amplitude_before = synth.phase, synth.amplitude
        after = synth.render(256, AudioState(1.1, 76, 0.8, True), 1.1)
        self.assertNotEqual(synth.phase, phase_before)
        self.assertAlmostEqual(synth.amplitude, amplitude_before, delta=0.001)
        self.assertLess(abs(float(after[0] - before[-1])), 0.08)
        self.assertGreater(synth.frequency, 440)
        self.assertLess(synth.frequency, 659.3)

    def test_stale_state_releases_even_when_video_freezes(self):
        synth = Synthesizer()
        state = AudioState(1, 69, 1, True)
        synth.render(4800, state, 1)
        released = synth.render(48000, state, 1.3)
        self.assertLess(float(abs(released[-1000:]).max()), 1e-6)

    def test_gate_off_and_future_timestamp_are_silent(self):
        synth = Synthesizer()
        self.assertEqual(float(abs(synth.render(256, AudioState(1, 69, 1, False), 1)).max()), 0)
        self.assertEqual(float(abs(synth.render(256, AudioState(2, 69, 1, True), 1)).max()), 0)

    def test_waveform_is_continuous_across_callback_sizes(self):
        a, b = Synthesizer(), Synthesizer()
        state = AudioState(1, 62, 1, True)
        whole = a.render(1024, state, 1)
        pieces = np.concatenate([b.render(256, state, 1) for _ in range(4)])
        np.testing.assert_allclose(whole, pieces, atol=1e-6)

    def test_audio_read_does_not_wait_for_writer_lock(self):
        bus = StateBus()
        fallback = AudioState(1, 62)
        with bus._lock:
            self.assertEqual(bus.read_nowait(fallback), fallback)
        published = AudioState(2, 69)
        bus.publish(published)
        self.assertEqual(bus.read_nowait(fallback), published)

    def test_callback_writes_every_sample_and_reports_underrun(self):
        bus = StateBus()
        engine = AudioEngine(bus)
        output = np.full((256, 1), np.nan, dtype=np.float32)
        engine._callback(output, 256, None, SimpleNamespace(output_underflow=True))
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertEqual(engine.health.underruns, 1)
        self.assertEqual(engine.health.callbacks, 1)

    def test_callback_failure_is_silence_and_visible_error(self):
        engine = AudioEngine(StateBus())
        def broken(*args):
            raise ValueError("test fault")
        engine.synth.render = broken
        output = np.ones((256, 1), dtype=np.float32)
        engine._callback(output, 256, None, SimpleNamespace(output_underflow=False))
        self.assertTrue(np.all(output == 0))
        self.assertIn("test fault", engine.health.error)

    def test_low_sample_rate_omits_aliasing_harmonics(self):
        synth = Synthesizer(16000)
        signal = synth.render(16000, AudioState(1, 83, 1, True), 1)
        self.assertTrue(np.isfinite(signal).all())
        spectrum = abs(np.fft.rfft(signal * np.hanning(len(signal))))
        self.assertGreater(spectrum[988], spectrum[3952])


if __name__ == "__main__":
    unittest.main()
