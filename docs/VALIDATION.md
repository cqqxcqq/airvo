# Milestone 2 validation — 11 October 2026

Development instance: Linux x86_64, glibc 2.41, CPython 3.12.14. The current
automated, model-integration, and illustrative-stage checks passed. Independent
follow-up review accepted the two identified defect fixes at that review scope;
laptop acceptance of Milestone 2 remains pending. The user has reported that the
corrected foundation calibration and hand controls work; that report does not
establish acceptance of the new stage.

| Check | Current outcome |
| --- | --- |
| Automated unittest suite | **80 passed**, zero skipped or expected failures |
| Full `bash tools/cloud_setup.sh` | Passed: exact dependencies, `pip check`, imports, 80 tests, pinned hand/person model downloads, real Tasks inference, and synthetic validators |
| Hand model integration | Passed: actual Tasks VIDEO CPU inference on two blank frames; this checks API/model integration, not recognition accuracy |
| Gesture-controller synthetic Ode to Joy | Passed: all five expected pitches selected and final release verified |
| Synthesized PCM | Finite, mono 48 kHz; peak **0.209409**, below digital clipping |
| Person model provenance/integrity | Official landscape version-1 object size and MD5 matched Google's metadata; pinned SHA-256 enforced on download and launch |
| Person model integration | Passed: actual Tasks VIDEO CPU inference on the official transient `segmentation_input_rotation0.jpg` fixture; one finite `selfie` person-confidence output |
| Stage presentation | Six headless captures inspected: all four scenes, guides, and calibration; synthetic hand positions explicitly labelled |
| Failure handling and control independence | Passed in automated tests: stale-mask fallback, failed-controller keyboard retention, source ownership, control updates without rendering, and cleanup |
| Webcam, physical bow attachment, audio output, desktop input | **Unrun in this instance:** no webcam, output device, or display attached |
| Fast-finger mask edges, actual sound quality, physical musical responsiveness | **Unverified:** requires the README laptop acceptance checks |

## Current measurements and their scope

`artifacts/validation.json` records synthesis-only rendering of 256 samples:
median **0.5395885 ms**, p95 **0.786776 ms**, against the **5.333333 ms** block
budget at 48 kHz. The exported `synthetic-ode.wav` is a synthetic controller
validation, not a recording of a user's performance. These timings exclude
camera acquisition, note-selection dwell, device buffering, and physical onset;
they do not measure end-to-end latency or perceived audio quality.

`artifacts/visual-review/studio-validation.json` records eight CPU person-model
inferences on the same still fixture: median **4.069451 ms**, p95 **4.942986 ms**.
Twenty stage render calls across the four scenes measured median
**8.3088625 ms**, p95 **15.79778035 ms**. These are current-instance measurements,
not sustained webcam throughput or musical-latency guarantees. The stage
captures use reduced motion, stationary synthetic hand landmarks, and refreshed
timestamps on the still-image mask; they do not demonstrate moving-hand edges
or concurrent live camera, rendering, and audio performance.

The six captures in `artifacts/visual-review/` are `studio-studio.png`,
`studio-nebula.png`, `studio-midnight-rain.png`, `studio-forest-of-light.png`,
`studio-guides.png`, and `studio-calibration.png`. The additional
`fixture-person-mask.png` is a model diagnostic. Google's fixture was downloaded
transiently for integration validation, not shipped as an application asset.
The model retains some objects near the people in this fixture, so successful
inference is not evidence of perfect person isolation. See
[research and model limits](RESEARCH.md) for source provenance and constraints.

## Source, presentation, and documentation consistency

Compared with the incumbent foundation renderer, the current stage retains the
dark restrained desktop shell, four string colors, physical-hand calibration,
and keyboard controls. It expands the camera/stage area, moves note and bow
feedback into a compact lower strip, and makes guides and diagnostics optional.
The inspected captures show the approved procedural backgrounds, recognizable
violin and bow, selected-note feedback, guide view, and unclipped calibration
view. Calibration keeps the original camera image and separate instructions;
person replacement and the decorative instrument are disabled there.

The bow anchor selects the calibrated physical right-hand label, not screen
right. Its grip and orientation follow palm landmarks with short smoothing and
tracking-expiry fading. Violin placement and bow direction derive from the
actual calibrated playing regions, including reversed screen positions. The
existing four-string/five-zone mapping still determines pitch independently of
the decorative geometry. Source and [README](../README.md) agree on the stage
shortcuts, both-model update instructions, fallback behavior, and the simplified
2D instrument limits. [RESEARCH.md](RESEARCH.md) explains why the original fixed
tone and gain-led wrist control felt like a note player, and why the new
speed-sensitive harmonics and reversal accents remain a bowed-string-inspired
synth rather than an acoustic physical model.

The musical controller now has an independent **8 ms polling heartbeat** and a
short publication lock for camera/keyboard ownership. That interval is a
scheduling choice, not measured note-onset latency. Person inference runs on a
separate worker with requested quality limits of LOW 256 px/8 Hz, MEDIUM
384 px/12 Hz, and HIGH 512 px/15 Hz. Slow inference further limits its duty cycle;
tracking pressure or underruns request LOW. Masks older than **250 ms** return
to camera video. Segmentation does not feed the hand detector or audio callback.
Fresh hand-landmark paths conservatively preserve fingers but can retain nearby
background. Effects and scene caches are bounded, reduced motion disables
moving effects, all four scenes are original procedural graphics, and the
extension adds no Python dependencies.

Independent finish review identified two P2 defects: a bow that always pointed
screen left, and a keyboard toggle that could return to a failed control worker.
Both were corrected with regressions. The bow now points toward the violin in
either calibrated layout; failed camera controls keep keyboard mode available
until restart. The current 80-test result includes these regressions. The
independent follow-up verdict was **"ship at this review scope"**, limited to
these two fixes; laptop acceptance of Milestone 2 remains pending.

Complete the README's foundation and visual-studio acceptance checks on the
user's laptop, including both mirror settings, physical right-hand attachment,
fast fingers, background-on/off responsiveness, sound quality, and failure
fallbacks. The available cloud evidence does not resolve those hardware checks.

# Historical Milestone 1 validation — 11 October 2026

The following is the earlier foundation checkpoint. Its **58-test count**,
audio measurements, and then-pending calibration confirmation are historical;
use the Milestone 2 section above for current evidence and remaining checks.

Development instance: Linux x86_64, glibc 2.41, CPython 3.12.14. This document
reported instance evidence, not acceptance on a user's laptop.

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

The Milestone 1 validator measured synthesis-only median block rendering around
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

## Calibration correction — historical Milestone 1 status

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
and physical versus unassigned captions. At that checkpoint, laptop confirmation
of the corrected wizard remained pending.
