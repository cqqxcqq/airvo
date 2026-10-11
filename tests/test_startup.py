import threading
import unittest
from unittest.mock import Mock
from air_violin.startup import ResourceLoader


class StartupTests(unittest.TestCase):
    def test_slow_device_does_not_block_construction_and_late_result_closes(self):
        entered, release = threading.Event(), threading.Event()
        resource = Mock()
        def factory():
            entered.set()
            release.wait(2)
            return resource
        loader = ResourceLoader('camera', factory)
        self.assertTrue(entered.wait(1))
        self.assertFalse(loader.done.is_set())
        self.assertIsNone(loader.take())
        loader.close()
        release.set()
        self.assertTrue(loader.done.wait(1))
        resource.close.assert_called_once()
        self.assertIsNone(loader.take())

    def test_taking_resource_transfers_cleanup_ownership(self):
        resource = Mock()
        loader = ResourceLoader('camera', lambda: resource)
        self.assertTrue(loader.done.wait(1))
        self.assertIs(loader.take(), resource)
        self.assertIsNone(loader.take())
        loader.close()
        resource.close.assert_not_called()

    def test_device_failure_is_reported_without_throwing_on_ui_thread(self):
        def factory():
            raise RuntimeError('camera permission denied')
        loader = ResourceLoader('camera', factory)
        self.assertTrue(loader.done.wait(1))
        self.assertIn('camera permission denied', loader.error)
        self.assertIsNone(loader.take())
        self.assertGreaterEqual(loader.elapsed, 0)
        loader.close()

    def test_cancelled_unclaimed_resource_closes(self):
        resource = Mock()
        loader = ResourceLoader('audio', lambda: resource)
        self.assertTrue(loader.done.wait(1))
        loader.close()
        loader.close()
        resource.close.assert_called_once()
