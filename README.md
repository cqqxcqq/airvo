# Air Violin Studio — finger-controlled visual studio

A local Python desktop instrument played with two bare hands. Version **0.3.2**
corrects knuckle-based finger detection, adds relaxed assisted sustain, and
rebuilds the 3D violin with stable size, fuller proportions, and a continuous neck. The audio
stream, hand tracking, and optional person segmentation run independently of
stage rendering. No cloud connection is required after installation.

The camera status now explains why a note cannot start: missing hands, unknown
fingering, wrist outside the bow area, or a pinch stopping Assisted. Pressing
Space in camera Assisted also prints the status and gesture details in the
terminal. These messages replace the misleading generic start instruction.

This is a simplified gesture instrument with bowed-string-inspired synthesis.
The camera estimates finger poses; it does **not** measure finger pressure,
real string contact, or acoustic violin physics. Scoring, practice coach,
recording, looper, accompaniment, and advanced chromatic mode remain future
milestones. See [research and limits](docs/RESEARCH.md) and
[validation evidence](docs/VALIDATION.md).

## Install once on Windows

Use **64-bit CPython 3.12**. Open the project folder containing this README,
`requirements.txt`, and the two launchers. On the laptop path reported during
setup, this is the **inner** `airvo` folder:

```powershell
cd "C:\Users\nonos\OneDrive\Desktop\essay\vcpj\airvo\airvo"
.\Setup-Air-Violin.cmd
```

Alternatively, double-click **Setup-Air-Violin.cmd**. It creates `.venv` only
when missing, installs the exact binary dependency pins, downloads/verifies both
Google models, and runs the automated tests. Read any error before proceeding.
Do not recreate a virtual environment while its Python executable is running.
Activation and PowerShell execution-policy changes are unnecessary.

To update an existing installation, close the app first:

```powershell
git pull --ff-only
.\Setup-Air-Violin.cmd
```

Your saved physical left/right binding is retained, but **finger mode needs one
new calibration** for the corrected knuckle measurement. Older finger settings
are not reused with the new math. The saved file is replaced only after all
calibration steps succeed. There are no new dependencies in 0.3.1.

The equivalent manual setup is:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
.\.venv\Scripts\python.exe tools\fetch_model.py --all
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Skip the first command if the environment already exists. Python 3.11 is not
supported by these dependency pins. Do not install another OpenCV distribution
alongside `opencv-contrib-python`.

## Play whenever you want

Double-click **Play-Air-Violin.cmd**, or run:

```powershell
.\Play-Air-Violin.cmd
```

The launcher uses its own folder, so spaces in the path and starting from
another directory do not change the project location. It does not reinstall
packages or download models. It pauses on failure so you can read the error.
Extra arguments are passed to the app:

```powershell
.\Play-Air-Violin.cmd --scene nebula
.\Play-Air-Violin.cmd --instrument-size 1.2
.\Play-Air-Violin.cmd --bow-mode auto --string auto
.\Play-Air-Violin.cmd --demo
.\Play-Air-Violin.cmd --backend dshow
.\Play-Air-Violin.cmd --renderer 2d --quality low
```

The direct launch command remains:

```powershell
.\.venv\Scripts\python.exe -m air_violin
```

Default controls use **finger mode, Assisted Bow, a fixed D string**, and the automatic renderer choice. Use
`--fingering pointing` for the previous index-tip grid, or `--renderer 2d` for
the lightweight visual fallback. Device progress appears while the camera,
audio, and models initialize. A blocked camera driver cannot be forcibly
unblocked by Python; close competing camera apps and try another backend.
Use **[ / ]** to select another string. `--string auto` enables bow-height string
selection; `--bow-mode auto` returns to continuous physical strokes.
Windows defaults to the DirectShow camera backend; `--backend msmf` or
`--backend auto` are alternatives if that driver fails.

## Calibration and hand posture

Use even front lighting. Hold your hands far enough apart to remain visible,
with the left palm/fingers mostly facing the camera. The intended pose is an
**imaginary violin neck grip**, with the thumb supporting it and fingers gently
curved. A tightly closed fist or fingers hidden behind one another is unsuitable
for reliable webcam recognition.

Finger mode has 13 guided steps. Hold each requested finger pose for about one second, then press
**Space** when the preview is ready:

1. Raise only your **physical left hand** to bind its role.
2. Raise only your **physical right hand** to verify its role.
3. Left: all four fingers comfortably lifted, with slight natural curves.
4. Left: index down; higher fingers lifted.
5. Left: index and middle down; ring/little lifted.
6. Left: index, middle, and ring down; little lifted.
7. Left: all four fingers down, using gentle curls rather than a fist.
8. Begin with all four down; press Space to start the check. Lift little, ring,
   middle, then index in order, pausing briefly at each note. The check must
   detect **4 → 3 → 2 → 1 → 0** before advancing automatically. No bow is needed.
9. Right: place the wrist at a comfortable corner of the bowing region.
10. Right: place it at the opposite corner. Either corner order is accepted;
    allow at least 24% of image height for the four string bands.
11. Right: thumb and index comfortably apart.
12. Right: touch thumb/index tips for Precision Bow calibration.
13. Press Space, then sweep the right wrist horizontally for three seconds at
    a comfortable speed to calibrate bow intensity.

Lower the finger gently at its **base knuckle**, keeping your grip stable; a deep
curl or closed fist is unnecessary. Calibration checks stable samples, pose
separation, and the actual note transitions. If a pose is too similar or noisy,
it explains which pose to retry. **C** recalibrates at any time. Both display and inference use the
same mirrored pixels; physical hand roles come from calibration, not screen
position or an assumed MediaPipe label. Changing `--no-mirror` requires new
calibration.

Calibration is saved in `%LOCALAPPDATA%\.air-violin-studio` on Windows or
`~/.air-violin-studio` elsewhere. `--config PATH` changes the location. Settings
are numerical JSON; normal playing saves no camera frames. Pointing mode keeps
its previous nine-step calibration, including opposite fingerboard corners.

## Finger playing

Your **highest numbered finger down** selects the note. Lower fingers may stay
down, as in first-position fingering. Raising the ring finger reveals the middle
finger's note if the middle remains down. This is not a count of curled fingers.

| Left-hand stopping gesture | G string | D string | A string | E string |
| --- | --- | --- | --- | --- |
| All four lifted: open string | G3 | D4 | A4 | E5 |
| Index down: finger 1 | A3 | E4 | B4 | F#5 |
| Middle down: finger 2 | B3 | F#4 | C#5 | G#5 |
| Ring down: finger 3 | C4 | G4 | D5 | A5 |
| Little down: finger 4 | D4 | A4 | E5 | B5 |

These Easy offsets are **0, 2, 4, 5, 7 semitones** from each open string. They
are a simplified quantized preset, not every violin scale or continuous
intonation. The preview changes before bowing; moving the left hand around the
screen is unnecessary in finger mode.

**Assisted Bow (default):** keep the right thumb/index apart. A short horizontal
stroke inside the calibrated bowing area starts a sustained phrase. Alternatively,
press **Space** with both hands tracked and the right wrist in the area. Rest
the right hand anywhere inside the camera view; change notes with the left
fingers. **Pinch right thumb/index, Space, or 0** stops the phrase. Losing valid
tracking also releases sound; returning to view requires a new stroke or Space.
This is an ergonomic sustain aid, not a model of a real stationary bow.

**Auto Bow:** sound follows horizontal strokes inside the region. **Precision
Bow:** strokes sound only while right thumb/index are pinched. **B** cycles
Assisted → Auto → Precision. In the stroke modes, stopping, leaving the region,
or releasing Precision's pinch releases sound naturally. Brief reversals are
bridged. All modes change pitch without reopening the audio device.

D is locked by default in Assisted so lowering your right arm cannot change the
string. **[ / ]** changes the locked string; **\** or `--string auto` enables
right-wrist height bands **G, D, A, E**, top to bottom. In Assisted, deliberate
horizontal strokes can change the height-selected string; stationary sustain
holds it. The four finger indicators show measured press percentages: roughly
0% lifted, 100% down, and `--` when the feature is unknown.

Try this D-string Ode to Joy phrase, using finger numbers:

```text
F# F# G A | A G F# E | D D E F# | F# E E
 2  2 3 4 | 4 3  2 1 | 0 0 1  2 |  2 1 1
```

Assisted connects held notes into a phrase. To articulate repeated notes,
stop/start with Space or pinch/stroke, or choose Auto for separate strokes.
Simply holding the same fingering cannot create separate rhythmic notes.
In pointing mode, the previous
four-row/five-column grid remains available; its columns are numbered 1–5.

## Stage, 3D instrument, and controls

The four original backgrounds now depict a warm acoustic **Studio**, layered
**Nebula**, reflective **Midnight Rain** city, and **Forest of Light** clearing.
They use procedural artwork, with no external image licenses or online assets.
Custom JPG/PNG backgrounds are center-cropped to cover without stretching:

```powershell
.\Play-Air-Violin.cmd --background "C:\Pictures\stage.png"
.\Play-Air-Violin.cmd --background-blur --reduced-motion
.\Play-Air-Violin.cmd --no-background --effects 0
```

The 3D violin is an original mesh with depth and shading, anchored near the
left thumb/index grip, with the body extending diagonally toward the torso.
Size is relative to the stage rather than palm width, so curling fingers does
not shrink the violin. **Comma / period** adjusts size; `--instrument-size 1.2`
sets the launch size (0.65–1.75). The bow follows the right grip and aims toward
the violin body. Hand-relative geometry helps estimate depth/tilt. This does not
reconstruct shared 3D bow-string contact or exact finger occlusion. The logical
musical controls remain independent of the decorative instrument.
The old palm-hull cutout is disabled because it could erase the middle of the
neck. The mesh can draw in front of a hand; that is preferable to pretending a
coarse palm polygon supplies accurate depth.
The native stage uses **pyglet 2.1.17** and requests **OpenGL 3.3**. Automatic
mode falls back to 2D when the graphics context is unavailable; Windows driver
support still needs testing on your laptop.

| Key | Action |
| --- | --- |
| H | Show/hide the playing tutorial |
| K | Camera / keyboard source; stop demo bow |
| J | Finger / pointing controls |
| [ / ], \ | Camera: lock string down/up; restore automatic string selection |
| B | Assisted / Auto / Precision Bow; switching stops the previous phrase |
| Space / 0 | Camera Assisted: sustain toggle / stop; start needs fresh tracked hands |
| , / . | Decrease/increase 3D violin size |
| C | Recalibrate in camera mode |
| G | Show/hide teaching and bow-region guides |
| S | Cycle Studio, Nebula, Midnight Rain, Forest of Light, custom image |
| V | Background replacement on/off |
| P | LOW / MEDIUM / HIGH quality |
| M | Reduced motion: remove trails, particles, rain, scene motion |
| D | Diagnostics and hand landmarks |
| F | Fullscreen/windowed |
| + / - | Volume |
| R | Retry failed audio initialization |
| Q / Esc | Release resources and quit |
| 1–5, [ / ], Space, Enter, 0 | Keyboard demo: notes, strings, sustain toggle, short stroke, stop |

For an audio-only check, launch `--demo`. It starts on D: keys **1–5** choose the
five notes, **Space** toggles sustained bowing, **Enter** plays a short stroke,
and **0** stops. `--demo --silent` tests display/input without camera or audio.

Use `--list-audio-devices` to list outputs, then `--device 3` or
`--device "Speakers"` with a real index/name on your machine. Larger
`--block-size 512` may help underruns at the cost of latency.

## Dependencies, models, and offline operation

`requirements.txt` pins direct and transitive packages. The previous dependency
set was exercised on Linux CPython 3.12.14 and resolved to Windows x64 wheels;
the complete updated setup also passed on Linux, and the new renderer wheel
resolved for the Windows target. See the validation report for actual graphics
checks. Package availability does not establish Windows camera, audio, or
graphics behavior.

The downloader uses official, versioned Google model URLs over verified HTTPS:

- [Hand Landmarker](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
- [SelfieSegmenter landscape](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite)

It validates pinned SHA-256 digests and model structure. Keep each downloaded
model and its `.sha256` sidecar in `models/` for offline launch. Re-running setup
reuses verified local models. `tools/fetch_model.py --all --force` replaces a
corrupt model or missing sidecar. Certificate/checksum failures never disable
verification. Models are ignored by Git and downloaded separately.

The optional person segmenter runs at bounded resolution/rate on its own
worker. Missing or stale masks return to ordinary camera video. Narrow fresh
landmark paths preserve fingers, sometimes retaining small room-background
fragments. Poor light, motion, or overlapping fingers remain limitations.
Background replacement is disabled during calibration.

On Debian/Ubuntu with Python 3.12:

```bash
sudo apt-get update
sudo apt-get install libportaudio2 libgl1 libglib2.0-0
python3.12 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
.venv/bin/python tools/fetch_model.py --all
.venv/bin/python -m air_violin
```

A local desktop display and webcam/audio permissions are required. A headless
cloud can run tests and validators, but cannot establish interactive hardware
acceptance. `bash tools/cloud_setup.sh` prepares this Debian cloud with local
PortAudio and writable caches. Linux MediaPipe needs glibc 2.28 or newer;
macOS and Windows ARM remain unverified.

## How the controls work

```text
latest camera frame -> MediaPipe hand worker -> image + hand-relative 3D landmarks
                                              |
                         calibrated fingering + wrist/pinch controller
                                              |
                                  immutable timestamped AudioState
                                              |
                                 persistent PortAudio synth callback

latest camera frame -> person segmenter -> scene/person composition -> stage renderer
```

- **Finger geometry:** base-knuckle flexion, tip height above the palm plane,
  distal bends, and shortening are computed in MediaPipe's predicted metric
  hand coordinates. Palm translation and camera
  image location do not determine the note. Calibrated lifted/down endpoints
  turn those measurements into activation scores.
- **Stability:** scores smooth over elapsed time; separate down/up thresholds and
  a short dwell prevent flicker. Missing/degenerate geometry becomes unknown.
  A higher unknown finger blocks a reliable lower note; uncertainty never means
  an open string. A short timeout releases unresolved fingering.
- **Pitch:** `frequency = 440 * 2**((MIDI - 69)/12)` using open MIDI notes
  `55, 62, 69, 76`. The same five Easy offsets remain available in both controls.
- **Bow:** `vx = (x_now - x_previous)/dt`, in image widths per second. Smoothed
  speed controls intensity; a deadzone rejects jitter. Pinch distance is divided
  by palm width, with separate engage/disengage thresholds. Assisted latches
  sustain after a genuine small excursion; a stationary real bow would be silent.
- **Sound:** up to 24 band-limited harmonics, designed body coloration, restrained
  friction noise, smooth intensity and phase-continuous pitch transitions.
  Audio attack/release use exponential envelopes. No inference, disk access, or
  graphics work occurs inside the callback.
- **Freshness:** results retain acquisition timestamps, and the control worker
  has an independent heartbeat. Audio has a stale-state watchdog. Old frames
  cannot sustain a note indefinitely, even if rendering slows.

Press **D** to inspect FPS, inference time, result age, and audio health. Device
latency and callback duration are not an end-to-end note-onset measurement.
MediaPipe supplies no reliable per-finger confidence; geometric rejection is
an application heuristic, not guaranteed detection of hidden fingertips.

## Tests and laptop acceptance

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tools\validate.py --require-model
.\.venv\Scripts\python.exe tools\validate_studio.py --require-segmenter
.\.venv\Scripts\python.exe -m pip check
```

Tests use synthetic landmarks and audio blocks; they do not require a webcam.
The validator's WAV is a synthetic regression artifact, not a recording of
live playing. See [VALIDATION.md](docs/VALIDATION.md) for actual results and
which checks used real model inference.

The Linux EGL/Mesa graphics validator also exercises the actual texture, mesh,
and integrated HUD pipeline without a camera or audio device:

```bash
.venv/bin/python tools/validate_3d.py
```

This headless graphics check does not test the Windows window/driver path.

Before calling this playable on your laptop:

1. Play the Ode phrase in keyboard demo. Confirm stable tones, smooth endings,
   no stuck sound, and clean shutdown/relaunch.
2. Calibrate with only the requested physical hand visible. Left controls pitch;
   right carries the bow and controls strokes. Repeat after mirror changes.
3. Lock D. Check open, index, middle, ring, little individually and in cumulative
   combinations. Lift the highest finger to expose the next lower one. Play the
   phrase three times without using screen pointing.
4. In Assisted, start with one short stroke, then rest the right hand in view
   for 20 seconds while changing all five notes. Repeat with Space. Check pinch,
   Space, and 0 stop reliably; returning a covered hand must not resume sound.
5. Restore automatic string selection. Bow horizontally at each wrist-height
   band without accidental crossings. Resting Assisted must hold its string.
   Compare Auto and Precision stops/reversals; hold left poses without flicker.
6. Move and gently rotate the left grip: violin neck stays near the grip. Move
   the right grip: bow follows toward the body. Curling must not shrink the mesh
   or remove a section of its neck. Check comma/period size controls. Cover a
   hand: its tracked visual must disappear.
7. Compare 3D/backgrounds on and off with D diagnostics visible. Tracking age and
   underruns must remain acceptable. If worse, use LOW, 2D, or original camera.
8. Try S/V/G/F/M controls, a custom image, missing model/image paths, and a
   ten-minute session. Close/reopen to confirm camera and audio are released.

The cloud has no physical webcam, audio output, or interactive desktop. Actual
finger recognition, comfortable posture, Windows driver behavior, and musical
latency remain manual acceptance work. No guaranteed 30 FPS or sub-100 ms onset
is claimed.

## Troubleshooting

If **keyboard demo has sound but camera mode does not**, the output device and
synthesizer are working. In camera mode, press Space and read the message below
the note. Assisted needs a recognized left-hand note, fresh tracking of both
hands, an unpinched right thumb/index, and the wrist inside the bow area to
start. A 3D violin/bow alone does not prove those musical conditions are met.
Pinch is **stop** in Assisted and **engage** in Precision; check the displayed
mode. D shows tracking age plus the audio callback's last request gate, age,
signal peak and callback count. Space prints numerical gesture/audio diagnostics.
If the BOW bar is filled but sound is absent, these values distinguish a late
request rejected by the audio watchdog from a gesture that never starts.
Share the status or screenshot to identify the blocked step; do not reinstall
audio dependencies or delete calibration merely because camera playback is silent.

| Symptom | Focused correction |
| --- | --- |
| `requirements.txt` / module not found | Use the inner project folder or the launchers; they select their own directory. |
| Venv creation permission denied | Close running Python/app processes. Do not recreate an existing active venv. |
| Missing packages after update | Close the app and run Setup-Air-Violin.cmd. |
| Camera takes a long time / unavailable | Read the loading stage. Close other camera apps, permit desktop-camera access, try `--backend dshow`, then `msmf`, or `--camera 1`. |
| 3D cannot initialize | Use `--renderer 2d`, update the graphics driver, and retain the error for diagnosis. |
| Fingers stay on D4 / cannot calibrate | C: redo the new knuckle-based calibration. Gently lower at the base joint, keep the palm visible, and pass the 4→3→2→1→0 check. Inspect the four press percentages. J restores pointing. |
| Wrong physical hand | C: identify hands separately. Do not infer physical role from screen side. |
| Accidental string changes | Lock with [ / ]; leave enough vertical bow room and stay away from band borders. |
| Sound continues while wrist rests | Expected in Assisted. Pinch, Space or 0 stops. B selects Auto for stroke-dependent sound. In Auto, improve landmark visibility or increase `--bow-deadzone`. |
| Bow sound stops too quickly / tiring | Use Assisted (default). Start with a short stroke or Space and keep both hands visible; rest the right arm. Tracking loss still releases safely. |
| Violin too small / neck disappears | Update to 0.3.1. Comma/period or `--instrument-size 1.2` adjusts size. The destructive palm cutout is disabled. |
| No sound / underruns | Check OS output/volume; list devices and select `--device`; try `--block-size 512` and compare demo. |
| Missing model / sidecar | Run setup; for corruption or a missing sidecar run `tools/fetch_model.py --all --force` with the venv Python. |
| Background cutout flickers | Use front lighting, P/LOW, or V/original camera; fast fingers remain a model limitation. |
| Custom image fails | Verify the local JPG/PNG path. An error is shown and Studio remains available. |
| Calibration cannot save | Use a writable `--config PATH`; session calibration can still work. |
| TLS/download failure | Correct clock/trust or allow `storage.googleapis.com`; never disable certificate checking. |
| Two OpenCV packages | Use a clean venv with this requirements file; do not mix headless/regular/contrib packages. |

## Project map

```text
air_violin/           capture, vision, calibration, fingering, mapping, gestures,
                      controller, runtime, audio, demo, render, app, startup,
                      segmentation, scenes, instrument_visual, effects,
                      mesh3d, stage_window
tools/                model downloader and synthetic/inference validators
tests/                control, fingering, calibration, audio, studio tests
docs/                 RESEARCH.md, VALIDATION.md
models/               downloaded models + verification sidecars (ignored)
artifacts/            diagnostic exports (ignored)
Play-Air-Violin.cmd   everyday Windows launcher
Setup-Air-Violin.cmd  installation/update launcher
requirements.txt     exact dependency pins
LICENSE              GPLv3 project license
```
