"""Headless studio checks and illustrative captures, never hardware acceptance.

--require-segmenter exercises actual Tasks inference. --image may provide a
local portrait for mask inspection. No camera is opened and no image downloaded.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from air_violin.audio import AudioEngine, StateBus
from air_violin.capture import CameraFrame
from air_violin.config import Calibration, Controls
from air_violin.controller import Instrument
from air_violin.models import DEFAULT_SEGMENTER
from air_violin.render import Renderer
from air_violin.scenes import SCENES
from air_violin.segmentation import PersonMask, PersonSegmenter, composite_person
from air_violin.types import Hand, Point, TrackingFrame


def illustrated_hands(timestamp):
    # Fully labelled synthetic hand poses to exercise the actual bow anchor.
    left = [Point(.28, .46) for _ in range(21)]
    left[0], left[8] = Point(.28, .65), Point(.28, .43)
    right = [Point(.77, .45) for _ in range(21)]
    right[0], right[9] = Point(.79, .55), Point(.78, .42)
    right[2], right[5], right[17] = Point(.74, .49), Point(.77, .44), Point(.85, .47)
    right[4], right[8] = Point(.72, .43), Point(.75, .39)
    return TrackingFrame(timestamp, (Hand('Left', .98, tuple(left)), Hand('Right', .98, tuple(right))))


def validate(output, image_path=None, require_segmenter=False):
    output.mkdir(parents=True, exist_ok=True)
    image = None
    if image_path:
        image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot decode validation image: {image_path}')
    blank = np.zeros((480, 640, 3), np.uint8)
    inference_times = []
    result_summary = 'UNRUN: person model not requested'
    mask = None
    if require_segmenter:
        task = PersonSegmenter(DEFAULT_SEGMENTER)
        try:
            source = image if image is not None else blank
            for i in range(8):
                timestamp = time.perf_counter()
                mask = task.detect(CameraFrame(i, timestamp, source))
                inference_times.append(mask.inference_ms)
                assert mask.alpha.ndim == 2 and np.isfinite(mask.alpha).all()
                assert 0 <= mask.alpha.min() <= mask.alpha.max() <= 1
            if image is not None:
                # Explicit fixture, not webcam frames or an app recording feature.
                cv2.imwrite(str(output / 'fixture-person-mask.png'), (mask.alpha * 255).astype(np.uint8))
            result_summary = 'PASS: real Tasks VIDEO CPU confidence-mask inference on ' + ('local image' if image is not None else 'blank frames')
        finally:
            task.close()
    now = time.perf_counter()
    hands = illustrated_hands(now)
    instrument = Instrument(Calibration(), Controls())
    instrument.update(illustrated_hands(now - .05), now - .05)
    state = instrument.update(hands, now)
    # The held synthetic wrist is stationary: preview is honest and silent.
    audio = AudioEngine(StateBus())
    timings = []
    for preset in SCENES:
        renderer = Renderer(scene=preset, reduced_motion=True)
        renderer.guides = False
        for i in range(5):
            now = time.perf_counter()
            fresh_hands = replace(hands, timestamp=now)
            fresh_mask = replace(mask, timestamp=now) if mask is not None else None
            start = time.perf_counter()
            stage = renderer.draw(image, state, Calibration(), fresh_hands, None, False, 'auto', audio,
                                  'Illustrative capture: synthetic hands. No webcam or speaker acceptance.',
                                  mask=fresh_mask, camera_timestamp=now)
            timings.append((time.perf_counter() - start) * 1000)
        assert stage.shape == (710, 920, 3) and stage.dtype == np.uint8
        if not cv2.imwrite(str(output / f'studio-{preset}.png'), stage):
            raise OSError('Could not write stage capture')
    renderer = Renderer(reduced_motion=True)
    stage = renderer.draw(image, state, Calibration(), replace(hands, timestamp=time.perf_counter()),
                          None, False, 'auto', audio, 'Illustrative capture: synthetic landmarks / guide view.',
                          mask=replace(mask, timestamp=time.perf_counter()) if mask else None,
                          camera_timestamp=time.perf_counter())
    cv2.imwrite(str(output / 'studio-guides.png'), stage)
    from air_violin.calibration import Calibrator
    wizard = Calibrator(True, .65)
    wizard.step = 5
    wizard.corner = Point(.55, .20)
    stage = Renderer().draw(image, state, Calibration(), replace(hands, timestamp=time.perf_counter()),
                            wizard, False, 'auto', audio, 'Illustrative calibration capture: synthetic landmarks.')
    cv2.imwrite(str(output / 'studio-calibration.png'), stage)
    # Validate real composition without mutating model output or inverting alpha.
    if mask is not None and image is not None:
        scene = np.full_like(image, (30, 25, 20))
        composed, used = composite_person(image, scene, replace(mask, timestamp=1), 1)
        assert used and np.mean(abs(composed.astype(float) - image)) > 1
    report = {
        'segmentation': result_summary,
        'segmentation_ms_median': float(np.median(inference_times)) if inference_times else None,
        'segmentation_ms_p95': float(np.percentile(inference_times, 95)) if inference_times else None,
        'render_ms_median': float(np.median(timings)),
        'render_ms_p95': float(np.percentile(timings, 95)),
        'captures': 'four procedural scenes, guides, calibration; synthetic hand positions labelled',
        'hardware': 'UNRUN: actual hands, webcam, desktop, speakers, perceived latency and fast-finger mask quality',
    }
    (output / 'studio-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    parser.add_argument('--image', type=Path)
    parser.add_argument('--require-segmenter', action='store_true')
    args = parser.parse_args()
    try:
        validate(args.output, args.image, args.require_segmenter)
    except (OSError, ValueError, AssertionError) as exc:
        print(f'Studio validation failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
