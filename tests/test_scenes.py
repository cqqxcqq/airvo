"""Background contracts: bounded caches, reproducibility, and safe composition."""
import unittest

import numpy as np

from air_violin.scenes import SCENES, SceneEngine


class SceneTests(unittest.TestCase):
    def test_cached_artwork_is_not_mutated_by_animation_or_callers(self):
        scene = SceneEngine('nebula')
        original = scene._base(160, 120).copy()
        frame = scene.render(160, 120, 2, quality='high')
        frame[:] = 255
        np.testing.assert_array_equal(scene._base(160, 120), original)
        self.assertFalse(scene._base(160, 120).flags.writeable)

    def test_cache_remains_bounded_across_resolutions(self):
        scene = SceneEngine()
        for width in range(80, 100):
            scene.render(width, 60, 0)
            self.assertLessEqual(len(scene.cache), 8)

    def test_quality_and_motion_controls_have_distinct_effects(self):
        for preset in ('nebula', 'midnight-rain', 'forest-of-light'):
            scene = SceneEngine(preset)
            a = scene.render(320, 240, 1, 'high')
            b = scene.render(320, 240, 10, 'high')
            self.assertFalse(np.array_equal(a, b))
            still_a = scene.render(320, 240, 1, 'high', True)
            still_b = scene.render(320, 240, 10, 'high', True)
            np.testing.assert_array_equal(still_a, still_b)
            np.testing.assert_array_equal(scene.render(320, 240, 1, intensity=0),
                                          scene._base(320, 240))

    def test_artwork_is_reproducible_and_presets_are_distinct(self):
        images = []
        for preset in SCENES:
            a = SceneEngine(preset).render(160, 120, 1, reduced_motion=True)
            b = SceneEngine(preset).render(160, 120, 1, reduced_motion=True)
            np.testing.assert_array_equal(a, b)
            images.append(a)
        for i, a in enumerate(images):
            for b in images[i + 1:]:
                self.assertGreater(float(np.abs(a.astype(float) - b).mean()), 4)

    def test_invalid_presets_and_dimensions_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, 'Unknown scene'):
            SceneEngine('not-a-scene')
        with self.assertRaisesRegex(ValueError, 'positive'):
            SceneEngine().render(0, 80, 1)

    def test_blur_retains_geometry_and_cache_key(self):
        scene = SceneEngine('forest-of-light')
        sharp = scene.render(160, 120, 0, intensity=0)
        scene.blur = True
        soft = scene.render(160, 120, 0, intensity=0)
        self.assertEqual(sharp.shape, soft.shape)
        self.assertGreater(float(np.abs(sharp.astype(float) - soft).mean()), 0)
        self.assertEqual(len(scene.cache), 2)


if __name__ == '__main__':
    unittest.main()
