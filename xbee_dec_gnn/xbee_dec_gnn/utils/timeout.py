import threading
from typing import Optional


class TimeoutErrorWithStatus(TimeoutError):
    """TimeoutError with status attribute for tracking failure type."""
    def __init__(self, message=None, status=None):
        if message is None:
            message = "Operation timed out."
        super().__init__(message)
        self.status = status


def run_with_timeout(timeout_seconds: float, func, *args, **kwargs):
    """Run a blocking callable in a daemon thread and enforce an outer timeout.

    Raises:
        TimeoutErrorWithStatus: If timeout expires (status="external_timeout").
    """

    outcome: dict[str, Optional[Exception]] = {"exc": None}

    def _run():
        try:
            func(*args, **kwargs)
        except Exception as exc:  # Preserve the original transport error for caller handling.
            outcome["exc"] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=timeout_seconds)

    if worker.is_alive():
        raise TimeoutErrorWithStatus(status="watchdog")

    if outcome["exc"] is not None:
        raise outcome["exc"]


if __name__ == "__main__":
    import time

    def test_func(duration):
        print(f"Starting test_func with duration {duration} seconds...")
        time.sleep(duration)
        print("test_func completed.")

    try:
        run_with_timeout(2, test_func, 1)  # Should complete successfully.
    except TimeoutErrorWithStatus as exc:
        print(f"TimeoutErrorWithStatus caught: {exc}, status={exc.status}")

    try:
        run_with_timeout(2, test_func, 3)  # Should raise TimeoutErrorWithStatus.
    except TimeoutErrorWithStatus as exc:
        print(f"TimeoutErrorWithStatus caught: {exc}, status={exc.status}")