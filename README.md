# Air Violin Studio — playable visual studio

A local Python desktop instrument controlled by two bare hands. This delivery is
**Milestones 1 and 2**: calibrated Easy Mode hand controls, Auto/Precision Bow,
continuous bowed-string-inspired synthesis, a stylized violin, a bow attached to
your physical right hand, person background replacement, four stage scenes,
bounded musical effects, and a keyboard demo. Bow speed changes tone color and
loudness; stroke reversals add gentle articulation. This is a simplified gesture
instrument, not a physically accurate acoustic violin or real string-contact
simulator. See [research and design limits](docs/RESEARCH.md).

Scoring, live recording, looper, accompaniment, advanced chromatic mode, body
pose anchoring, and an AI coach remain future milestones. No cloud service is
used during playing. No camera frames are saved. Diagnostic tools export
synthetic examples or a local image explicitly supplied for validation.

## Install on Windows

Use **64-bit CPython 3.12**. The exact dependency set was installed and tested on
Linux CPython 3.12.14; pip also resolved Windows x64 wheels for every pinned
package. That confirms package availability, not Windows hardware behavior.
Python 3.11 is not supported by this lock (some transitive packages need 3.12).
Do not substitute the newest MediaPipe or install a second OpenCV distribution.

From this project directory, in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
.\.venv\Scripts\python.exe tools\fetch_model.py --all
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m air_violin
```

Activation is optional when using the full executable path. If you prefer it,
run `.\.venv\Scripts\Activate.ps1`; use the explicit executable if your local
PowerShell execution policy disallows activation. No policy change is required.

The downloader uses Google's versioned official
[Hand Landmarker task bundle](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task).
See the [Tasks guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
and [model overview](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).
It checks a pinned SHA-256 over verified HTTPS, validates the bundle, and writes
a SHA-256 sidecar. The pin was established after checking the official object's
MD5 and size against Google's object metadata API (generation 1682480004222387).
Both runtime and download verification enforce the pin, so modifying the
sidecar cannot make a different model pass. It is not a publisher signature.
The versioned URL fixes the model revision.
The downloaded model is ignored by Git and is not redistributed here.

Model files land in `models/hand_landmarker.task` and
`models/hand_landmarker.task.sha256`. Keep both for offline launch. To place them
elsewhere, use `tools/fetch_model.py --output PATH`, then launch with `--model PATH`.
Re-running the downloader uses the verified local file without network access.
Use `--force` to replace a damaged model; certificate or checksum failures never
fall back to an insecure download. Normal application launch never downloads.

`--all` also downloads the official version-1 [SelfieSegmenter landscape model](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite)
to `models/selfie_segmenter_landscape.tflite`, with its SHA-256 sidecar. It uses
the installed Tasks `ImageSegmenter` API; no additional Python dependencies are
needed. The pinned digest is documented in `docs/RESEARCH.md`. Keep both models
and both sidecars for offline use. Background failure falls back to camera
video; it does not disable hand controls. The hand model remains sufficient for
the original instrument and `--no-background`.

### Linux and other platforms

On Debian/Ubuntu with Python 3.12 installed:

```bash
sudo apt-get update
sudo apt-get install libportaudio2 libgl1 libglib2.0-0
python3.12 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
.venv/bin/python tools/fetch_model.py --all
.venv/bin/python -m air_violin
```

A desktop display and webcam/audio permissions are needed. A headless SSH/cloud
session can run the tests and synthetic validator, but cannot provide the
interactive desktop experience. For this Codex cloud's Debian 13 x64 instance,
`bash tools/cloud_setup.sh` installs PortAudio locally from signed Debian package
metadata and sets writable caches. `--skip-model` is an explicitly partial
prerequisite diagnostic while model-download network access is unavailable.
The model requires HTTPS access to `storage.googleapis.com`, dependencies need
`pypi.org` / `files.pythonhosted.org`, and the cloud helper needs `deb.debian.org`.
No credentials are needed.

The MediaPipe 0.10.32 Linux wheel requires glibc 2.28 or newer. macOS and Windows
ARM hardware have not been validated; use pip's wheel availability as a first
check rather than assuming these platform pins work there. The modules avoid
OS-specific application APIs; camera and audio backends still vary by platform.

## First launch and calibration

Use even front lighting, keep both hands visible with palms mostly facing the
camera, and leave room to move. Default capture requests 640 × 480 at 30 FPS;
camera drivers may negotiate another size. Detection and display use the same
mirrored pixels. The interface preserves camera aspect ratio using letterboxing.

The first webcam launch guides you through nine steps. Press **Space** to confirm
each position after holding still briefly:

1. Raise **only your physical left hand**. This binds the model's raw label to
   the note hand. It does not infer identity from screen position.
2. Raise **only your physical right hand** to verify the opposite raw label.
3. Mark any corner of the fingerboard with your **physical left index fingertip**.
4. Mark the diagonally opposite corner with the same finger. Either corner order
   is accepted. Use at least 15% image width and 24% image height;
   a larger comfortable board makes notes easier to hit.
5. Mark any corner of the bow box with your **physical right wrist**.
6. Mark its diagonally opposite corner with the same wrist. Give the wrist room
   for horizontal strokes. Either corner order is accepted.
7. Hold right thumb and index comfortably apart.
8. Touch the two fingertips to calibrate Precision Bow's pinch thresholds.
9. Press Space and sweep your right wrist horizontally for three seconds at a
   comfortable speed. The 80th percentile tracked speed sets full intensity.

The camera remains visible throughout; landmarks, calibrated physical hand
labels, and a live preview between opposite corners are shown. Before the first
hand is identified, the overlay says "Hand detected". Model classifications are
not displayed as physical sides: raising your physical left hand in step 1 binds
it to LEFT/notes, even if the underlying mirrored-input model calls it Right.
Each point uses a short median window. Ambiguous labels,
stale results, unstable samples, tiny/out-of-frame rectangles, and insufficient
sweep motion are rejected with an explanation. **C** restarts calibration.

Calibration is stored as JSON in `%LOCALAPPDATA%\.air-violin-studio` on Windows
or `~/.air-violin-studio` elsewhere. `--config PATH` overrides this location.
No video is stored. Changing `--no-mirror` invalidates the saved calibration and
asks you to calibrate again, including physical hand identity. If the file is
invalid it is kept until a successful new calibration replaces it. A save
failure keeps the in-memory calibration working and reports that it was not
persisted.

## Playing

The physical **left index tip** selects the note, whether or not you are bowing.
Four horizontal bands run top to bottom in **G, D, A, E** order. Five columns run
left to right with offsets **0, 2, 4, 5, 7** semitones:

| String | Open MIDI | Five Easy Mode notes |
| --- | ---: | --- |
| G | 55 | G3, A3, B3, C4, D4 |
| D | 62 | D4, E4, F#4, G4, A4 |
| A | 69 | A4, B4, C#5, D5, E5 |
| E | 76 | E5, F#5, G#5, A5, B5 |

The selected cell and note light up before sound starts. Coordinates are
normalized to the actual camera image. The logical fingerboard is independent
of the decorative violin overlay.

**Auto Bow:** move your physical right wrist horizontally inside its calibrated
box. Speed controls intensity. **Precision Bow:** do the same while pinching
thumb and index. Press **B** to switch. The new mode reacquires motion safely.

Reverse direction smoothly. A short grace period bridges the slow point of a
reversal; genuine stops release naturally. Leaving the bow box or unpinching
releases immediately when detected. Losing either hand releases within the
configured timeout plus the audio envelope. Bringing it back requires fresh
tracking and a new bow movement; an old frame cannot keep sound alive.

Start with Ode to Joy transposed into D major, all on the D string:

```text
F# F# G A | A G F# E | D D E F# | F# E E
 3  3 4 5 | 5 4  3 2 | 1 1 2  3 |  3 2 2    (zone numbers)
```

Preview one zone at a time and use a separate gentle stroke for repeated notes.
Moving pitch during a sustained bow glides briefly to the new note; it does not
reopen the stream or restart the oscillator.

## Keyboard demo and shortcuts

First test audio independently of vision:

```powershell
.\.venv\Scripts\python.exe -m air_violin --demo
```

The demo starts on the D string. Focus the application window. **1–5** choose
notes, **[ / ]** choose strings, **Space** toggles sustained bowing, **Enter**
plays a 0.7-second stroke, and **0** stops. This uses the same synth and audio
stream as gesture playing. It deliberately uses toggles/short strokes because
OpenCV does not expose portable key-up events. Holding a key is unnecessary.

| Key | Action |
| --- | --- |
| H | Show/hide the short playing tutorial |
| K | Switch between camera and keyboard sources; stop demo bow |
| B | Auto / Precision Bow |
| G | Show/hide the fingerboard and bow-region guides |
| S | Cycle Studio, Nebula, Midnight Rain, Forest of Light, and an optional custom image |
| V | Background replacement on/off |
| P | Cycle LOW, MEDIUM, HIGH visual quality |
| M | Toggle reduced motion; removes trails, particles, rain, and scene motion |
| D | Show/hide diagnostics and hand landmarks |
| F | Fullscreen/windowed stage |
| C | Restart calibration in camera mode |
| + / - | Raise/lower volume |
| R | Retry failed audio-device initialization |
| Q / Esc | Release resources and quit |
| 1–5, [ / ], Space, Enter, 0 | Demo notes, string, sustain, stroke, stop |

`--demo --silent` tests display/input without requesting camera or audio.
`--list-audio-devices` lists available PortAudio devices. Select an output with
`--device 3` or `--device "Speakers"`. Use the actual index/name on your machine.

## Stage backgrounds and the hand-attached bow

For an existing installation, close the app, open PowerShell in the folder that
contains this README and `requirements.txt`, then run:

```powershell
git pull --ff-only
.\.venv\Scripts\python.exe tools\fetch_model.py --all
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m air_violin
```

Your existing physical-hand calibration is reused. Do not recreate or activate
an already-running virtual environment. The dependency pins are unchanged.

The default Studio background replaces the room while keeping the person.
Press **S** for other scenes, **V** to compare original camera video, **G** to
hide the teaching grid, **F** for fullscreen, **P** to lower quality, and **M**
for reduced motion. Press **H** for all controls. The bow uses your calibrated
physical right hand, not whichever hand appears on screen right. It follows
palm position/orientation, with short smoothing and stale-result fading. The
violin remains decorative and independent of the calibrated note regions;
its neck position does not introduce a new pitch-mapping rule.

```powershell
.\.venv\Scripts\python.exe -m air_violin --scene nebula --quality low
.\.venv\Scripts\python.exe -m air_violin --background "C:\Pictures\stage.png"
.\.venv\Scripts\python.exe -m air_violin --background-blur --reduced-motion
.\.venv\Scripts\python.exe -m air_violin --no-background --effects 0
```

All four presets are original procedural scenes, with no internet image assets
or extra license requirements. Custom images use center cropping to cover the
camera area without stretching. Background blur softens the chosen background,
not your face. Segmentation is disabled during calibration; the original view
helps you mark reachable corners. It runs at reduced resolution/rate on a
separate worker, and old masks return to camera video rather than holding an
old cutout. **D** shows processing age, segmentation time and audio health.

Person masks are imperfect, especially for fast fingers and poor lighting.
Narrow fresh landmark paths preserve fingers but may retain a little room
background around them. Bow/violin are drawn after composition to remain
visible; they are stylized 2D overlays and do not reconstruct finger occlusion
or real bow-string collision. Reduce quality or disable replacement if your
laptop's tracking becomes less reliable. No frame is saved by normal playing.

## Architecture and the useful math

```text
camera thread -> latest frame -> hand Tasks VIDEO worker -> latest hands
       |                                                   |
       |                              independent control heartbeat (runtime.py)
       |                                                   |
       |                                      immutable AudioState mailbox
       |                                                   |
       |                                           PortAudio callback
       |                                            persistent synth
       |
       +-> rate-limited person Tasks VIDEO worker -> timestamped mask
                                                          |
       +-> OpenCV stage: scene -> person -> violin/bow/effects -> controls
```

`capture.py` keeps only the newest frame. `vision.py` runs the current MediaPipe
Tasks **VIDEO** API on its own worker with two hands and strictly increasing
millisecond timestamps. It does not mix in `mp.solutions` examples. VIDEO is
synchronous inside that worker: it avoids LIVE_STREAM callback ordering issues,
while still allowing rendering/audio to proceed independently. Frame skipping
is intentional; results retain acquisition time, not completion time.

`calibration.py` binds physical hands and stores regions. `mapping.py` selects
notes. `gestures.py` interprets wrist motion and pinch. `controller.py` implements
IDLE → READY → BOWING → RELEASING, returning to READY when a selectable note
remains or IDLE after tracking loss. `types.py` holds immutable messages.
`runtime.py` updates camera-driven musical state independently of rendering.
`audio.py` owns synthesis and persistent device output. `segmentation.py` runs
optional person inference and composition. `scenes.py` owns original procedural
backgrounds and custom-image cover fitting. `instrument_visual.py` draws the
violin and anchors the bow; `effects.py` caps event-driven trails/particles.
`render.py` draws the stage, teaching guides, calibration, and diagnostics.
`app.py` handles lifecycle and shortcuts; `demo.py` is a substitute input source.
Body pose, gameplay, and recording remain deferred.

The main calculations are:

- **Landmarks:** normalized `x,y` are image fractions, with y increasing down the
  image. Index tip is landmark 8, wrist 0, thumb tip 4. We do not use MediaPipe z
  as a reliable 3D instrument coordinate. Normalized x differences measure screen
  widths, not meters. Calibration is tied to the mirror setting.
- **Pitch:** `frequency = 440 × 2^((MIDI − 69)/12)`. Semitone spacing follows
  equal temperament. Vertical and horizontal positions are converted to four
  string cells and five pitch cells within the calibrated rectangle.
- **Smoothing:** `alpha = 1 − exp(−dt / tau)` and
  `filtered += alpha × (raw − filtered)`. Using elapsed seconds gives similar
  behavior at different frame rates. Default position tau is 35 ms.
- **Stable notes:** the current cell expands by 12% of one zone; the next cell
  must remain selected for 50 ms before committing. A small boundary jitter
  therefore cannot repeatedly change pitch. Leaving the board clears selection.
- **Bow:** `vx = (x_now − x_previous)/dt`. Signed velocity is smoothed with a
  40 ms time constant; intensity also smooths speed magnitude, so opposite
  directions do not cancel loudness. Both raw and smoothed signed motion must
  exceed the 0.045 widths/s deadzone. An 85 ms grace period bridges reversals.
  Large detector teleports and time gaps reset velocity instead of sounding.
- **Pinch:** tip distance divided by palm width (landmarks 5–17) helps normalize
  camera distance. Engagement is at a lower ratio than disengagement. The wizard
  derives the thresholds from your own open and pinched hand.
- **Audio:** one phase-continuous oscillator sums up to 24 normalized harmonics
  with designed body coloration and subtle friction noise. Bow intensity changes
  harmonic rolloff, not only gain. Stroke starts/reversals add short accents
  without resetting the main envelope or oscillator. Harmonics above a
  conservative Nyquist boundary are omitted. Frequency glides with a 22 ms
  time constant; gain attacks at 15 ms
  and releases at 65 ms. These are exponential time constants, not fixed full
  fade lengths. Normalization leaves peak headroom below digital clipping.
- **Concurrency:** the audio callback reads an immutable state with a
  nonblocking lock attempt. If the writer briefly holds the mailbox lock, it
  uses its previous snapshot. A separate 180 ms audio watchdog releases stale
  state even if rendering stops. There is no camera work, disk I/O, or device
  discovery in the callback. NumPy callback allocations remain a practical
  Python limitation, so performance is measured rather than guaranteed.

Press **D** for UI FPS, processed vision FPS, inference time, result age,
PortAudio-reported device latency, callback work, and underrun count. These
are separate measurements, **not an end-to-end note-onset latency claim**.
Capture timestamps can include driver buffering; measuring physical onset
requires a separate hardware experiment.

CLI tuning examples:

```bash
python -m air_violin --confidence 0.75 --smoothing-ms 45 --note-hold-ms 70
python -m air_violin --bow-deadzone 0.08 --bow-speed 0.6
python -m air_violin --block-size 512 --sample-rate 44100
python -m air_violin --camera 1 --width 1280 --height 720
```

Larger smoothing/dwell improves stability but adds selection delay. Raise
`--bow-deadzone` if stationary wrist jitter produces sound. A lower full-speed
value makes relaxed sweeps louder. A larger audio buffer may fix underruns but
adds latency. Start with defaults and change one thing at a time.

## Automated checks and exact manual acceptance tests

```bash
python -m unittest discover -s tests -v
python tools/validate.py
python tools/validate.py --require-model
python tools/validate_studio.py --require-segmenter
python -m pip check
```

The unittest suite uses synthetic landmarks to check every note zone,
coordinate mirroring, physical hand binding, smoothing, hysteresis, elapsed-time
velocity, pinch engagement, reversals, stops, note transitions, tracking-loss
recovery, stale/out-of-order results, calibration, audio envelopes, sustained
fundamental/harmonics, clipping bounds, block continuity, watchdog release,
callback errors, download integrity, and aspect-ratio rendering. It does not
need camera, audio hardware, or a model download. No tests are skipped silently.

The validator sends a synthetic Ode to Joy phrase through the actual gesture
controller and synthesizer, asserts all five pitches and final release, and
exports `artifacts/synthetic-ode.wav`, a desktop image, and `validation.json`.
This is **not live performance recording**. `--require-model` additionally
initializes the real Tasks model and runs blank-frame inference. Missing models
make this required check fail; a blank frame checks the API/model integration,
not actual hand-recognition accuracy.

On your laptop, repeat these foundation checks after the update:

1. **Audio alone:** launch `--demo`, play the D-string phrase above with keys
   1–5 and Enter. Hear distinct, stable notes and natural endings. Toggle Space,
   change notes, then press 0. No beep-only sound, stuck tone, clicks, or clipping.
2. **Hand identity:** launch camera mode, calibrate physical left/right in
   isolation, then move each separately. Left index must change note preview;
   right wrist must control bow. Repeat with `--no-mirror` and recalibrate.
3. **Zone stability:** visit all 20 centers without bowing and confirm the table.
   Hold near a boundary with minor jitter: note stays stable. Move deliberately
   to the next center: it changes after a short dwell.
4. **Bowing:** in Auto mode a stationary right hand is silent. Sweep slowly and
   quickly for different intensities; reverse smoothly, then stop. In Precision
   mode an open hand is silent, pinching engages, and unpinching releases.
5. **Melody:** play the D-string Ode phrase with your hands three times at a
   relaxed pace. At least two takes should be recognizable with no unintended
   string changes. You reported that hand control now works; repeat this check after the update.
6. **Loss/recovery:** cover either hand, leave camera view, then return. Sound
   must fade promptly, and recovery must not produce a velocity spike. Unplug
   the webcam: the app must stay usable through K/demo and show the failure.
7. **Audio device errors:** select an invalid device index. The app must show a
   muted warning while controls remain usable. Restart with the valid device,
   or use R after correcting a temporary device failure.
8. **Ten-minute session:** inspect FPS/result age and audio underruns while
   playing. Close the window and reopen it: camera/audio must be released and
   calibration restored. Check 640×480 first, then an alternate aspect ratio.

Aim for roughly 30 processed FPS and perceptually quick notes on your hardware,
but do not assume a sub-100 ms onset: dwell, smoothing, detection, camera buffers,
and the audio envelope all contribute. Compare the keyboard demo to gesture
mode to isolate audio from vision delay. If meaningful hand results routinely
arrive older than the timeout, fix throughput instead of extending sustain
indefinitely.

## Troubleshooting

| Symptom | Focused correction |
| --- | --- |
| Missing model / checksum sidecar | Run `python tools/fetch_model.py --all`. Allow the official storage domain. Keep both files. Use `--force` for corruption. |
| Model download returns 403 | Network policy/proxy is blocking the official domain; update settings. No key or account is needed. |
| TLS error | Correct your system clock/trust store or configure your organization's trusted CA. Do not disable TLS verification. |
| No webcam | Close other camera apps, enable OS desktop-camera permission, try `--camera 1`. On Windows try `--backend dshow`, then `msmf`. These are runtime alternatives, not tested hardware fixes. |
| No desktop display | Run tests/`tools/validate.py` in cloud; run the interactive app locally. |
| Missing PortAudio | Windows wheel bundles it; on Debian/Ubuntu install `libportaudio2`. Cloud helper provides a trusted local copy. |
| No sound / bad output device | `--list-audio-devices`, select `--device`, check OS volume/output. Muted status is explicit. |
| Unsupported sample rate | Try `--sample-rate 44100`, using the device's supported format. |
| Background unavailable / cutout flicker | Run `python tools/fetch_model.py --all`; use even lighting. Try P/LOW or V to keep original camera video. Old masks are deliberately discarded. |
| Custom image fails | Confirm the full JPG/PNG path and that the file opens locally. The app reports the error and uses Studio. |
| Bow follows the wrong hand | Press C and identify physical left/right hands in isolation. The bow uses the same saved binding as sound controls. |
| Pops / underruns | Use `--block-size 512`, close heavy applications, compare `--demo`, inspect callback/underrun metrics. |
| Wrong hand / unstable labels | Recalibrate C with only the requested physical hand visible. Improve lighting and palm angle. Duplicate/low-confidence labels are rejected. |
| Note flicker | Use a larger board, hold closer to zone centers, increase dwell/smoothing slightly. |
| Sound from a still wrist | Increase `--bow-deadzone`; check that the palm and wrist remain clearly visible. |
| Tracking too slow | Start at 640×480, improve lighting, reduce other CPU use. Frames are already processed on a latest-only worker. |
| Calibration won't save | Supply a writable `--config PATH`; current-session calibration remains usable. |
| Two OpenCV packages installed | Create a fresh venv and install this requirements file; do not mix headless/regular/contrib wheels. |

## Project map and next milestones

```text
air_violin/       types, config, capture, vision, calibration, mapping,
                  gestures, controller, runtime, audio, demo, render, app,
                  segmentation, scenes, instrument_visual, effects
tools/            fetch_model.py, validate.py, validate_studio.py, cloud_setup.sh
tests/            control/calibration/audio/integration/studio tests
docs/             VALIDATION.md, RESEARCH.md
models/           downloaded bundle + integrity sidecar (ignored)
artifacts/        synthetic diagnostics (ignored)
requirements.txt  exact direct and transitive pins
LICENSE           existing GPLv3 project license
```

Milestones 1–2 are implemented. You reported that the corrected hand controls
work; verify the new visuals on your laptop before expanding. Milestone 3 adds
advanced chromatic mapping, rule-based practice, rhythm scoring, and WAV
recording. Milestones 4–5 add creative tools and Vision Lab exports. A more
acoustic sound needs a separately evaluated physical model or licensed samples.

## Visual studio acceptance checks

1. Close the old app, update and download both models using the commands above.
   Launch without `--demo`. Existing calibration should load. Raise each physical
   hand separately: only right should carry the bow; left should select notes.
2. Play the same D-string phrase with V/background off, then on. Compare D's hand
   result age, processed FPS, and underruns. Sound should stay responsive; no new
   stuck notes or repeated audio resets. If worse, use P/LOW and compare again.
3. Bow slowly then faster on one note: hear a change in brightness and loudness.
   Reverse several times: retain the sustained pitch with a gentle stroke accent.
   Change left-hand zones while bowing: notes stay connected.
4. Check S through all four scenes, custom JPG/PNG cover fitting, V off/on, G
   guide visibility, F fullscreen, and M reduced motion. Confirm the face and
   fast fingers remain reasonably intact; visible mask failure means use V/off.
5. Cover your right hand: the bow should disappear within tracking timeout and
   sound release. Cover the left: sound release. Returning should require fresh
   tracking and motion. Repeat `--no-mirror` after recalibrating, including bow
   pointing toward the instrument with physically reversed screen positions.
6. Launch with `--segmenter-model models/missing.tflite`: a clear warning and
   original camera video should remain playable. Launch with a missing custom
   image: Studio fallback. K/demo and C/calibration must still work. Do not call
   these webcam checks passed on the basis of the synthetic validator.
