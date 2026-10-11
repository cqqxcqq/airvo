# Gesture and rendering decisions

## Why this revision changes the interaction

The original index-tip grid fulfilled the beginner mapping specification, but
felt like selecting notes on a screen. Its 2D violin improved presentation
without changing that interaction. Version 0.3.0 replaces pointing as the default
with four calibrated stopping gestures, retains pointing as a fallback, and
attaches an actual 3D mesh to the left grip. The camera still measures poses,
not contact or force.

The user's 0.3.0 laptop screenshot exposed shortcomings that synthetic rendering
alone had not settled: palm-based occlusion erased the neck, hand-size scaling
produced a toy-like instrument, and the finger feature ignored base-knuckle
movement. Continuous waving also made phrases tiring. Version 0.3.1 corrects
those causes. Its assisted sustain is explicitly a convenience mode.

The persistent audio stream, latest-frame tracking, independent control worker,
hand-identity calibration, and stale-result safety remain in place. These are
useful foundations; rendering changes do not require restarting the instrument
or moving inference into the audio callback.

## Primary sources and actual API checks

1. Google, [MediaPipe Hand Landmarker Python guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
   and [tagged 0.10.32 implementation](https://github.com/google-ai-edge/mediapipe/blob/v0.10.32/mediapipe/tasks/python/vision/hand_landmarker.py).
   The installed result contains handedness, normalized image landmarks, and
   world landmarks. World coordinates are predicted meters centered separately
   on each hand, not a shared scene coordinate system. Image coordinates use
   normalized x/y and wrist-relative z; mixing them as equally scaled Euclidean
   coordinates would give misleading joint angles.
2. Google, [Hand Tracking model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20%28Lite_Full%29%20with%20Fairness%20Oct%202021.pdf).
   Recognition remains sensitive to pose, visibility, and training-domain
   limitations. In actual installed-model fixture checks, landmark `visibility`
   and `presence` were unset. Handedness `Category.score` is left/right
   classification confidence, not fingertip certainty. No per-finger confidence
   is invented by this app.
3. [pyglet installation](https://docs.pyglet.org/en/latest/programming_guide/installation.html)
   and [OpenGL guide](https://docs.pyglet.org/en/latest/programming_guide/gl.html).
   Pyglet supplies native window/context handling. The pinned 2.1.17 universal
   wheel was checked, and development created a real headless Linux EGL context.
   The app requests OpenGL 3.3. Windows graphics-driver support and physical
   hand anchoring still need laptop acceptance; Linux software rendering is not
   evidence of Windows real-time performance.
4. Julius O. Smith III, *Physical Audio Signal Processing*,
   [Bowed Strings](https://www.dsprelated.com/freebooks/pasp/Bowed_Strings.html).
   Bow velocity, force, contact point, nonlinear friction, string propagation,
   and body response all affect real tone. Our harmonic source with controlled
   brightness, noise, and envelopes is a lightweight approximation, not the
   book's physical waveguide model.
5. Google, [MediaPipe Tasks Image Segmenter overview](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter)
   and [Image Segmenter test source](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/test/vision/image_segmenter_test.py).
   The landscape selfie model has 144 × 256 input. Actual 0.10.32 inference
   produced one confidence output labelled `selfie`; assuming two masks would
   break integration. Google's segmentation fixture was used only for inference
   validation, not as a shipped background.

These sources establish API behavior and engineering constraints. They do not
establish the app's measured musical latency or recognition accuracy on the
user's laptop. Exact validation evidence belongs in [VALIDATION.md](VALIDATION.md).

## Four-finger control

For index/middle/ring/little, the controller constructs a local palm plane from
wrist and knuckle landmarks. It measures base-knuckle (MCP) flexion with
`atan2(abs(normal component), along-palm component)` of the first finger bone.
The feature combines 65% MCP angle, 20% distal bending, 10% normalized tip height,
and 5% shortening, using metric world landmarks. These are empirical feature
weights, not biomechanical measurements. A finger can lower as a whole while
its smaller joints keep the same curve; the old distal-only feature missed
that gesture. Rotation and reflection of the entire hand preserve the feature.
Invalid bone geometry or missing world landmarks is unknown.

Calibration records comfortably curved lifted fingers, then cumulative
stopping poses: index; index+middle; index+middle+ring; all four. Each pose uses
about one second of valid samples, medians, and measured variability. A guided
4→3→2→1→0 check verifies each release and rejects calibration that only detects
the open string. Feature-version metadata prevents reuse of old endpoints with
new math; prior hand identity and the saved file are preserved until success.
Per-finger endpoints normalize articulation into an activation
score. Separate engage/release thresholds, 25 ms exponential smoothing, and a
short dwell suppress jitter. These are configurable engineering values, not a
claim of measured onset latency.

The highest stopped finger determines pitch. Lower fingers can remain stopped
or uncertain without altering a confirmed higher finger. Every finger above
that finger must be confidently lifted; an unknown higher finger could still
be stopping the imaginary string. With no stopped fingers, all four must be
confidently lifted for the open string. Unresolved fingering expires after a
short timeout rather than silently becoming open.

This is intentionally quantized: open strings MIDI 55/62/69/76 and offsets
0/2/4/5/7. Those intervals are not a universal violin scale or continuous
intonation. The webcam estimates a pressed-like gesture, not actual pressure.
Sideways fingers, severe overlap, and a closed fist remain difficult. Plausible
model hallucinations cannot always be detected through geometry checks.

Four right-wrist height bands select G/D/A/E while horizontal motion controls
bow speed. Hysteresis and dwell keep ordinary strokes from crossing strings
accidentally. Explicit keyboard locking permits learning one string first.
Assisted defaults to locked D; in explicitly requested height-selection mode,
stationary sustain holds the string and horizontal strokes can change it.
This is a deliberate string-selection control, not a reconstructed collision
between the 3D bow and strings.

## Real 3D with bounded assumptions

`mesh3d.py` builds original procedural violin and bow meshes from numerical
geometry. They have depth, surface normals, materials/colors, and perspective
shading. No external model file, image, or copyrighted asset is silently
required. The geometry is distributed as source under the project's license.

`stage_window.py` displays the camera/stage and controls as textures and draws
the mesh through a native OpenGL context. Normal rendering does not read the GPU
framebuffer back to the CPU. Camera/inference/control/audio workers keep their
existing roles. A missing or unsupported graphics context has a 2D fallback.

The left thumb/index web anchors the neck grip. The body extends diagonally
toward a stage-center torso estimate; scale depends on stage dimensions and a
user size setting, not palm width or finger spread. The original model uses
full-size violin-inspired proportions, smooth arched plates, and cross-section
grids that avoid folds across the concave C-bouts. Hand-relative geometry supplies
bounded tilt. No body-pose model or measured torso anchor is claimed. The right
palm/grip anchors the bow frog. Overall depth and scale are estimates; each
hand's world coordinates have an independent origin, so subtracting them cannot
measure hand separation or true bow/string contact.

Palm-hull cutouts are disabled by default: the user's screenshot showed a palm
polygon erasing the middle of the neck. It cannot establish which pixels are
actually in front of the mesh. The violin may draw over the hand until a useful
depth/occlusion method is implemented. This is not precise fingertip occlusion
or contact simulation.
Tracking loss removes the stale visual and releases sound through the normal
safety path. Musical selection remains independent of decorative pose.

## Timbre and background decisions

The sound uses up to 24 band-limited harmonics, colored by two broad **designed**
body response bands. These frequencies are engineering choices, not measurements
of a particular violin. Bow speed opens harmonic rolloff and changes gain with
smoothing. Stroke starts and nonzero reversals add restrained short accents;
oscillator phase and the main envelope remain continuous through note changes.
Friction noise is quiet and filtered. No sampled violin, physical friction
solver, reverb, or vibrato is claimed.

Assisted Bow latches a phrase after a small genuine horizontal excursion,
roughly 1.2–1.5% of frame width for ordinary calibration, or explicit Space.
It sustains at a modest intensity floor while both hands remain freshly tracked.
Pinch, Space, 0, stale tracking, or a detector teleport stops it. The control
worker updates independently of graphics; latching never refreshes old capture
timestamps. Auto and Precision retain stroke-dependent sound. A stationary
real bow does not produce this continuous tone: assisted sustain trades that
physical constraint for comfort and learnability.

### Acoustic, physical-model, and gesture instruments

| Instrument | How it produces tone | What this application supplies |
| --- | --- | --- |
| Acoustic violin | Bow hair sticks/slips on a string; tension, contact point, bow force, and body resonance determine vibration | No physical string, contact, or pressure sensor |
| Physical-model simulation | Numerically approximates friction, wave propagation, and body response | Not implemented; a webcam cannot directly measure bow force |
| This gesture synthesizer | Finger pitch drives continuous harmonics, noise, and envelopes; gestures control expression | Playable melodic controls and violin-inspired tone, with simplified articulation |

A sample-based instrument can sound more convincing using licensed recordings
without becoming a physical simulation. An attractive 3D model also does not
make the audio model physically accurate. Acoustic realism remains a separate
development task; current acceptance focuses on controllable melodies and
comfortable performance.

All four backgrounds are original procedural scenes: acoustic Studio with
spotlights/floor perspective; layered Nebula clouds/stars; Midnight Rain city,
lamps/reflections; Forest of Light canopy/clearing/fireflies. Cached artwork and
bounded dynamic effects preserve reduced-motion and LOW/MEDIUM/HIGH controls.
Custom JPG/PNG images use aspect-correct cover fitting.

The optional person segmenter runs separately with requested maxima LOW
256 px/8 Hz, MEDIUM 384 px/12 Hz, HIGH 512 px/15 Hz. Slow inference further
limits duty cycle. Stale masks return to original camera video; fresh landmark
paths conservatively preserve fingers, sometimes retaining room background.
Segmentation is not an input to hand recognition. No claim of accurate scene
depth follows from a person mask.

The musical control heartbeat is independent of graphics, and audio has its own
stale-state watchdog. Threads reduce coupling but cannot guarantee responsiveness
under CPU exhaustion or Python scheduling. Compare graphics/backgrounds on/off
on actual hardware before increasing quality.

## Model integrity and redistribution

The official SelfieSegmenter landscape version-1 model is pinned to SHA-256
`490e9ea734313e0de10fa0cd9e3c6133e36ea4db2b7a49bde9ef019f72796b8e`.
Official size 250177 and MD5 matched Google object metadata, generation
1683436452064647, before establishing the pin. The hand model also has a pinned
SHA-256 and validated task-bundle structure. This is integrity verification
over HTTPS, not a publisher signature.

Models download separately to ignored `models/` files with checksum sidecars.
Runtime does not download anything. Graphics and 3D geometry are original code;
no internet background or model asset is redistributed.
