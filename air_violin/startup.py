"""Load blocking devices without freezing the window; close late results safely."""

from contextlib import suppress
import threading
import time


class ResourceLoader:
    """One resource, with an explicit transfer of ownership to the application.

    Camera drivers can block inside VideoCapture. Cancellation cannot interrupt
    a driver, but a resource returned after the window closes is still released.
    Neither the worker nor its finalizer makes graphics calls.
    """

    def __init__(self, name, factory):
        self.name, self.factory = name, factory
        self.started = time.perf_counter()
        self.finished = None
        self.error = ""
        self._lock = threading.Lock()
        self._resource = None
        self._cancelled = False
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"load-{name}", daemon=True)
        self.thread.start()

    def _run(self):
        resource = None
        try:
            resource = self.factory()
            with self._lock:
                if not self._cancelled:
                    self._resource, resource = resource, None
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if resource is not None:
                with suppress(Exception):
                    resource.close()
            self.finished = time.perf_counter()
            self.done.set()

    @property
    def elapsed(self):
        return (self.finished or time.perf_counter()) - self.started

    def take(self):
        with self._lock:
            resource, self._resource = self._resource, None
            return resource

    def close(self):
        with self._lock:
            self._cancelled = True
            resource, self._resource = self._resource, None
        if resource is not None:
            with suppress(Exception):
                resource.close()
        self.thread.join(timeout=.25)
