# Version 0.3.1 correction validation — 11 October 2026

The user's actual screenshot showed a severed neck, unstable proportions, and
reported both short sound bursts and pitch stuck on D4. This revision addresses
the causes; it does not treat cloud screenshots as proof of laptop playability.

| Check | Outcome |
| --- | --- |
| Full unittest suite | **170 passed**, zero skips; 3.294 seconds in the full setup run |
| Complete `bash tools/cloud_setup.sh` | Passed: binary dependencies, imports, `pip check`, models, tests, audio/Tasks/stage/EGL validators |
| Finger articulation | Pure MCP-only synthetic presses produce D4→E4→F#4→G4→A4 and back without changing bow gating |
| Calibration | Stable/noise sampling, cumulative pose checks, 4→3→2→1→0 verification, obsolete-feature migration, and incomplete-save preservation pass |
| Calibration occlusion | Latest missing/degenerate metric landmarks cannot be replaced with old good samples; recovery needs a fresh valid pose window |
| Assisted sustain | Genuine small stroke starts, fresh stationary frames sustain, notes change, pinch/Space/0 stop, tracking/teleport expiry prevents unintended restart |
| Automatic strings | Stationary assisted sustain freezes the string; deliberate strokes change it after dwell |
| Original behavior | Auto/Precision, pointing, mirror/hand binding, state machine, persistent synth, source ownership, fallback and cleanup checks pass |
| 3D | 17 mesh/pose tests pass; fuller mesh has 28,248 vertices, fixed stage scale, mirrored torso-directed posture, no default palm cutout |
| Real EGL render | Mesh depth/shading, continuous neck-under-palm pixel probe, texture orientation, stale pose removal, HUD scissor and actual application integration pass |
| Visual review | Front/side/back model, all four scenes, error/verification calibration and Assisted tutorial captures inspected; synthetic poses explicitly labelled |
| Physical webcam/audio/Windows | **Unrun here**; live finger discrimination, comfort, graphics performance, and sound quality still need laptop acceptance |

New regressions use synthetic landmarks, not recorded performances. Independent
review identified the calibration occlusion defect above; it was corrected and
tested. No physically accurate friction, force, bow/string contact, or palm depth
is claimed. The additive timbre has not become an acoustic violin model.

Fresh full-setup artifacts report synthesis block median **0.509834 ms**, p95
**0.651542 ms**, 256 samples/48 kHz (budget **5.333333 ms**), finite peak
**0.209409**. Real Tasks inference on blank frames passed for both models.
Person inference median **4.077162 ms**, p95 **5.264065 ms**; CPU stage median
**6.820008 ms**, p95 **68.746325 ms**, including cold scene construction.

The final expanded graphics validator wrote `artifacts/3d-review/report.json`:
pyglet 2.1.17, EGL OpenGL 4.5 Mesa llvmpipe, GPU-completion presentation median
**5.376111 ms**, p95 **6.074247 ms**. The earlier full-setup graphics run measured
**10.985384 / 12.797334 ms**; timings vary with concurrent work. These component
measurements exclude camera, gesture dwell, CPU HUD work and device buffering.
They do not establish laptop 30 FPS or sub-100 ms onset.

On the laptop, update and complete the new calibration. Pass its five-note
release check; then start Assisted with a short stroke, rest the right hand in
view, and play the README Ode phrase three times. Verify pinch/Space/0 stops,
cover-and-return safety, Auto/Precision, whole neck under the palm, stable size,
and backgrounds on/off over a ten-minute session. The full manual checklist is
in [README](../README.md).

---

# Historical version 0.3.0 validation — 11 October 2026

Development instance: Linux x86_64, glibc 2.41, CPython 3.12.14. The current
revision passed automated, real-model API, and actual headless OpenGL checks.
The user previously reported working foundation hand controls; that report does
not establish acceptance of the new finger classifier or 3D hand attachment.
The older milestone records below are retained as historical evidence.

| Check | Current outcome |
| --- | --- |
| Full unittest suite | **132 passed**, zero skips or expected failures; 2.714 seconds in the complete setup run |
| Full `bash tools/cloud_setup.sh` | Passed: exact binary pins, dependency imports, `pip check`, all tests, both verified local models, hand/person Tasks inference, synthetic audio/stage checks, and native EGL graphics validator |
| Windows x64 / Python 3.12 packages | Prior binary-wheel resolution covers the existing pins; the new pyglet 2.1.17 universal wheel also resolved for that target. No Windows execution claim |
| Finger control | Synthetic checks passed for all 16 finger combinations, highest-stop semantics, unknown fingers, dwell/hysteresis, rotation/scale/mirror behavior, calibration, note transitions, and tracking expiry |
| Official hand fixture | Actual installed Tasks inference on transient `right_hands.jpg` returned two hands, each with 21 world landmarks and four finite curl features; this validates the adapter/math, not live stopping-gesture recognition |
| 3D geometry | 14 mesh/pose math tests passed; original violin 6,012 triangles, bow 1,280, marker 120; no external model assets |
| Actual graphics pipeline | Passed with pyglet 2.1.17, headless EGL, **OpenGL 4.5 Mesa llvmpipe**: texture orientation, vertex/index mesh drawing, shading/depth, tracking expiry, HUD clipping, Enter-key translation, and real application HUD integration |
| Stage review | Four richer scene captures and finger-calibration captures inspected; independent review found no release-blocking presentation issues. All use labelled synthetic poses |
| Startup and fallback | Tests passed with substituted devices/window transport: asynchronous loading, failure reporting, cancelled/late resource cleanup, camera/keyboard source ownership, renderer fallback, and control shortcuts |
| Windows launchers | Static checks: CRLF, quoted project paths, existing-venv reuse, argument forwarding, and actionable errors. **Not executed under Windows cmd** |
| Physical webcam, audio, Windows graphics, hand grip | **Unrun in this instance**; no physical webcam/audio output or interactive desktop attached |
| Real finger discrimination, occlusion, sound quality, musical latency | **Unverified**; requires laptop acceptance below |

## Fresh measurements and their scope

The full setup run wrote `artifacts/validation.json`:

- Synthesis of 256-sample blocks at 48 kHz: median **0.524165 ms**, p95
  **0.783453 ms**, against a **5.333333 ms** block budget.
- Synthetic mono PCM was finite, with peak **0.209409**, below digital clipping.
- The synthetic Ode to Joy regression selected all five expected pitches and
  released at the end. It exercises the existing pointing-controller path,
  protecting that fallback; finger-playing behavior is covered separately by
  the new synthetic tests.
- Real Hand Landmarker Tasks VIDEO CPU inference on blank frames passed.

`artifacts/studio-validation.json` records actual person-model CPU inference on
**blank frames**: median **4.624276 ms**, p95 **6.745288 ms**. Twenty CPU stage
render calls across all four scenes measured median **6.751604 ms**, p95
**71.710450 ms**. Four calls include cold procedural-artwork construction, so
this report keeps the observed p95 rather than substituting cached-only timing.
These checks establish model integration and isolated rendering behavior, not
person cutout quality or sustained concurrent webcam throughput.

`artifacts/3d-review/report.json` records native texture/mesh presentation with
GPU completion explicitly waited for: median **6.178971 ms**, p95 **11.480858 ms**
in the final verification run alongside the automated suite.
The renderer is Mesa **software** llvmpipe, not the user's laptop GPU. Timings
exclude CPU scene/HUD construction, webcam acquisition, inference, gesture dwell,
and audio/device latency. The validator additionally exercised the real
`Renderer.draw` HUD and captured all four presets plus finger calibration.
That integration check is separate from the isolated presentation benchmark.

No measurement above establishes a sub-100 ms note onset, 30 FPS on a laptop,
or perceived acoustic quality. The exported WAV is a synthetic regression
artifact, not a recording of a performance. Screenshot timestamps and landmarks
are controlled inputs, not evidence of a live camera session.

## Correctness fixes covered by regressions

A confirmed finger pose no longer expires merely because its acquisition time
is more than 80 ms old. The **80 ms limit applies to observed uncertainty**;
confirmed geometry remains governed by the normal whole-hand freshness watchdog.
A regression uses 84 ms-old results and 15 FPS processing to protect this
separation. Unknown higher fingers cannot become an open string or silently
allow an unreliable lower note.

The independent 8 ms controller heartbeat now accounts for bounded inference
delivery age when bridging bow reversals. Previously, 84 ms delivery delay could
consume nearly all the reversal grace before the result arrived and produce
flutter. Tests cover delayed 30 FPS motion, an actually stopped result, zero-delay
behavior, and frozen-result expiry. This allowance does not refresh acquisition
age or permit indefinitely stale sustain.

String-lock shortcuts update the existing musical controller rather than
restarting it. A regression changes the string during continuous bowing without
a new camera frame and verifies preserved bow gating and correct new pitch.
Calibration migration, physical-role binding, source switching, missing models,
failed devices, renderer fallback, and late resource cleanup also have checks.

## Laptop acceptance still required

Use [README](../README.md) for setup, shortcuts, and the complete checklist:

1. Update with Setup-Air-Violin.cmd, then launch Play-Air-Violin.cmd. Confirm
   visible startup progress, clean close/relaunch, and actionable failure errors.
2. Complete the 13-step finger calibration with physical hands identified
   separately. Lock D using `--string D`; play the finger-number Ode phrase
   three times without screen pointing. Check each held pose for unintended
   changes, and lift the highest stopped finger to expose the next lower one.
3. Test four wrist-height string bands, Auto/Precision bow, stops, reversals,
   both mirror settings after recalibration, and lost-hand safe release/recovery.
4. Move and modestly rotate both grips. The neck and bow frog should follow the
   correct physical hands without distracting flips. Coarse palm occlusion is
   not proof of exact depth or virtual contact.
5. Compare 3D/backgrounds off and on while watching tracking age and underruns.
   Run ten minutes; compare LOW, 2D, and original camera if responsiveness worsens.
   Check motion reduction, fullscreen, custom images, and missing-resource paths.

MediaPipe has no dependable per-finger confidence. Degenerate/out-of-calibration
geometry is rejected, but plausible hidden-joint predictions can survive those
heuristics. Comfortable gesture discrimination, fast-finger segmentation,
Windows graphics/device behavior, and actual melody acceptance cannot be settled
by the cloud tests. No physically accurate string-contact or pressure measurement
is claimed.

---

# Historical Milestone 2 validation — 11 October 2026

This is the earlier 80-test checkpoint. References to "current" below describe
that checkpoint; version 0.3.0 above supersedes its feature and test status.

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
use the version 0.3.0 section above for current evidence and remaining checks.

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
