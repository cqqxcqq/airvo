# Why the first version felt like a note player

The criticism is fair. The original grid selected a quantized pitch, wrist
speed mostly controlled gain, and the same 12-partial sound sustained each
note. It met the beginner mapping specification, but omitted much of the
interaction that gives a bowed instrument its character. Drawing a bow alone
would not address that.

## Sources consulted before this update

1. Julius O. Smith III, *Physical Audio Signal Processing*,
   [Bowed Strings](https://www.dsprelated.com/freebooks/pasp/Bowed_Strings.html).
   Bow velocity, force, contact position, nonlinear friction, and the instrument
   body all matter. The text describes velocity-wave digital waveguides,
   bow-string scattering, and Helmholtz motion. Its discussion of periodic
   excitation and efficient body coloration supports a simplified sustained
   harmonic source, but does **not** make additive synthesis a physical model.
2. Google, [MediaPipe Tasks Image Segmenter overview](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter)
   and the linked Selfie Segmentation model card. The landscape model has
   144 × 256 input and is intended for portrait/background segmentation. Actual
   Python 0.10.32 testing confirmed one confidence output labelled `selfie`.
   Assuming two output masks would break this model integration.
3. Google's [Image Segmenter test source](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/test/vision/image_segmenter_test.py)
   identifies `segmentation_input_rotation0.jpg`. That official fixture was
   temporarily downloaded to an ignored research directory for actual CPU
   inference and mask inspection. It is not an application background or a
   redistributed asset.

All links were fetched over verified HTTPS during development. No source was
used as evidence of measured responsiveness on the user's laptop.

## Concrete changes and their limits

The sound now has up to 24 band-limited harmonics, gently colored by two broad
**designed** body response bands. Their frequencies are engineering choices,
not measurements of a particular violin. Increasing bow speed opens harmonic
rolloff with a 40 ms smoothing time constant, so speed changes tone color as
well as loudness. Stroke starts and nonzero direction reversals introduce short,
small friction/brightness accents. Oscillator phase and the main envelope
continue through reversals and note changes: every camera frame does not
restart a sound. Noise is deliberately quiet and less bass-heavy.

The decorative instrument has a violin body, f holes, four strings, fingerboard,
active string, note marker, and a bow grip derived from the calibrated physical
right hand. This connects gesture feedback to the musical metaphor. Its position
is calibrated in 2D, not estimated from a full body pose or three-dimensional
string collision. Hiding the teaching grid does not change its logical mapping.

The beginner five-zone mapping remains intentionally simplified. There is no
physical finger pressure, real bow contact, contact-point timbre control,
continuous fingerboard intonation, trained gesture coach, or acoustic violin
simulation. A webcam cannot recover bow force from bare hands. Adding a hidden
"pressure" heuristic would imply a capability we cannot measure and would risk
making the working controls less predictable.

A genuine physical model or measured, legally licensed violin samples remain
possible later improvements, after laptop acceptance of this stage. No samples,
reverb, vibrato, advanced chromatic mode, scoring, or recording were silently
added as incomplete later milestones.

## Background and performance choices

SelfieSegmenter landscape version 1 is verified by pinned SHA-256
`490e9ea734313e0de10fa0cd9e3c6133e36ea4db2b7a49bde9ef019f72796b8e`.
Official object size 250177 and MD5 matched Google's object metadata
(generation 1683436452064647) before establishing the pin. This is integrity
verification over HTTPS, not a publisher signature. Models download separately;
all four stage backgrounds are original procedural graphics.

CPU inference runs in a separate rate-limited worker: LOW 256 px / 8 Hz,
MEDIUM 384 px / 12 Hz, HIGH 512 px / 15 Hz. These are requested maxima, not
promised throughput. Slow inference further limits its duty cycle. Tracking
results older than 140 ms or a new audio underrun temporarily request LOW.
A mask older than 250 ms is discarded in favor of the original camera video.
Fresh hand landmark paths conservatively preserve fingers; this may retain
small pieces of room background around a hand. Fast motion, dim lighting,
unusual backgrounds, multiple people, and objects near the body remain real
model limitations. Segmentation does not feed the hand detector.

The musical controller has an independent 8 ms polling heartbeat. This is a
scheduling interval, not measured musical latency. Source changes use a short
publication lock so the camera worker cannot overwrite keyboard audio state.
Audio has its own stale-state watchdog. Separate threads reduce coupling but
cannot guarantee responsiveness under CPU exhaustion or Python scheduling.
