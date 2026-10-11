# Gesture and rendering decisions

## Why this revision changes the interaction

The original index-tip grid fulfilled the beginner mapping specification, but
felt like selecting notes on a screen. Its 2D violin improved presentation
without changing that interaction. Version 0.3.0 replaces pointing as the default
with four calibrated stopping gestures, retains pointing as a fallback, and
attaches an actual 3D mesh to the left grip. The camera still measures poses,
not contact or force.

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

For index/middle/ring/little, the controller combines PIP/DIP joint bending
with tip-to-base shortening, using the hand-relative metric landmarks. The
weighted measurement is 75% bend and 25% shortening. Invalid bone geometry or
missing world landmarks is unknown, not an arbitrary fallback note.

Calibration records comfortably curved lifted fingers, then cumulative
stopping poses: index; index+middle; index+middle+ring; all four. A final all-up
pose verifies release. Per-finger endpoints normalize curl into an activation
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

The left thumb/index web anchors the neck grip. Palm landmarks supply a stable
orientation, with bounded tilt from hand-relative world geometry. The right
palm/grip anchors the bow frog. Overall depth and scale are estimates; each
hand's world coordinates have an independent origin, so subtracting them cannot
measure hand separation or true bow/string contact.

A small camera-texture palm-hull cutout gives coarse image-space hand occlusion.
It is not measured depth, precise fingertip occlusion, or contact simulation.
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
