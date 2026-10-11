# Milestone 1 validation — 11 October 2026

Development instance: Linux x86_64, glibc 2.41, CPython 3.12.14. This document
reports current-instance evidence, not acceptance on a user's laptop.

| Check | Outcome |
| --- | --- |
| Pinned dependency installation and `pip check` | Passed; no broken requirements |
| Windows x64 / CPython 3.12 wheel resolution, binary-only | Passed for all 18 pinned packages; no Windows execution claim |
| Automated unittest suite | **58 passed**, zero skipped or expected failures |
| Full `bash tools/cloud_setup.sh` | Passed: dependency imports, tests, verified model download, synthetic validation, real Tasks inference |
| Local PortAudio provisioning | Passed, using signed Debian metadata and package checksums; backend reports **zero devices** |
| Model provenance/integrity | Official version-1 object size and MD5 matched Google's object metadata; pinned SHA-256 enforced on download and launch |
| Actual Tasks VIDEO model initialization and blank frames | Passed on CPU; both blank frames returned no hands |
| Official `right_hands.jpg` fixture | Two hands, 21 landmarks each, original and mirrored, with separate fresh trackers |
| Gesture-controller synthetic Ode to Joy | Passed; all five expected pitches selected and final release verified |
| Synthesized PCM | Finite, mono 48 kHz; peak 0.23834, below clipping |
| Desktop loop/input/failure/cleanup | Passed with substituted window transport; physical desktop interaction **unrun** |
| Webcam, physical left/right calibration, audio output | **Unrun:** no camera, output device, or desktop display attached |
| Physical melody acceptance and end-to-end latency | **Unrun:** requires laptop manual tests in README |

Latest full validator measured synthesis-only median block rendering around
0.083 ms and p95 around 0.159 ms for 256 samples, against a 5.333 ms block budget.
These timings fluctuate with load and exclude camera, selection dwell, audio
device buffering, and physical note onset. They are not a sub-100 ms claim.

The two-hand fixture is used by Google's official
[Hand Landmarker tests](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/test/vision/hand_landmarker_test.py)
and was fetched from
`https://storage.googleapis.com/mediapipe-assets/right_hands.jpg` for a transient
integration check. It is not shipped as an application asset. Its two hands
are both classified Right before mirroring, and Left after mirroring. It checks
the raw result adapter and mirror behavior, not recognition of a performer's
two different physical hands. Calibration resolves physical identity separately.

An exploratory test flipped an already-running VIDEO stream and received no
hands on that next frame. Reinitializing the tracker for each orientation passed
both initial and follow-up frames. This is why mirror mode remains fixed per
application session; changing it requires a restart and fresh calibration.

MediaPipe emitted native warnings about feedback tensors and normalized ROI
projection, but inference completed and fixture assertions passed. These are
not suppressed. Revisit with actual camera evidence if landmarks are distorted
or recognition fails on a particular device.

Diagnostics in ignored `artifacts/` include `validation.json`,
`synthetic-ode.wav`, and `foundation-preview.png`. The WAV is a synthetic
validation rendering, not a recording of a user's performance.

Before Milestone 2, verify physical melody control, comfortable region reach,
stationary-wrist silence, pinch reliability, device behavior, and responsiveness
using README's exact manual acceptance tests. No hardware acceptance is claimed.

## Calibration correction

A user screenshot showed swapped raw handedness captions and a rejected second
bow corner. The overlay now resolves physical LEFT/notes and RIGHT/bow from
the same learned binding used by the controls. Before identity calibration it
says "Hand detected". No global label swap or screen-side identity assumption
was introduced.

Both fingerboard and bow corners are now sorted into normalized rectangles,
accepting either diagonal order; minimum-size and in-frame checks remain.
The second-corner step shows a live rectangle. Added regressions cover all four
diagonal corner orders, both mirror settings, swapped model labels, completing
the wizard and then playing with those bindings, tiny/out-of-frame rejection,
and physical versus unassigned captions. Laptop confirmation of the corrected
wizard remains pending.
