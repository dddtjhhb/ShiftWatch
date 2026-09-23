"""Worker process: N execution slots, one heartbeat/reaper thread, graceful shutdown.

Each slot loops claim → call provider → complete/fail. Every thread owns its own DB
connection (psycopg connections must not be shared across threads concurrently).
"""
import logging
import os
import signal
import socket
import threading
import time
import uuid

import psycopg

from . import queue
from .config import Settings
from .db import connect
from .providers import build_provider, classify_exception

log = logging.getLogger("shiftwatch.worker")


class Worker:
    def __init__(self, settings: Settings, slots: int = 4, worker_id: str | None = None,
                 poll_seconds: float = 0.5, reap_every_seconds: float = 2.0):
        self.settings = settings
        self.slots = slots
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
        self.poll_seconds = poll_seconds
        self.reap_every_seconds = reap_every_seconds
        self.stop_event = threading.Event()
        self._active: dict[uuid.UUID, queue.Lease] = {}
        self._lost: set[uuid.UUID] = set()
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self.stats = {"completed": 0, "failed_attempts": 0, "stale": 0, "reaped": 0}

    # ----------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._threads = [
            threading.Thread(target=self._slot_loop, name=f"slot-{i}", daemon=True)
            for i in range(self.slots)
        ]
        self._threads.append(
            threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True)
        )
        for thread in self._threads:
            thread.start()
        log.info("worker %s started with %d slots", self.worker_id, self.slots)

    def stop(self, timeout: float | None = None) -> None:
        """Stop claiming; in-flight tasks finish and commit before threads exit."""
        self.stop_event.set()
        for thread in self._threads:
            thread.join(timeout)

    def run_forever(self) -> None:
        self.start()
        signal.signal(signal.SIGTERM, lambda *_: self.stop_event.set())
        signal.signal(signal.SIGINT, lambda *_: self.stop_event.set())
        while not self.stop_event.is_set():
            time.sleep(0.2)
        self.stop()

    # --------------------------------------------------------------------- slots
    def _slot_loop(self) -> None:
        conn = None
        while not self.stop_event.is_set():
            try:
                if conn is None or conn.closed:
                    conn = connect(self.settings.database_url)
                lease = queue.claim_task(
                    conn, self.worker_id, self.settings.lease_seconds,
                    self.settings.default_provider_concurrency, self.settings.scheduler,
                )
                if lease is None:
                    self.stop_event.wait(self.poll_seconds)
                    continue
                self._execute(conn, lease)
            except psycopg.OperationalError:
                log.exception("database connection problem; reconnecting")
                conn = None
                self.stop_event.wait(1.0)
            except Exception:  # noqa: BLE001 - a slot must survive unexpected errors
                log.exception("unexpected slot error")
                self.stop_event.wait(1.0)
        if conn is not None:
            conn.close()

    def _execute(self, conn, lease: queue.Lease) -> None:
        with self._lock:
            self._active[lease.token] = lease
        try:
            try:
                provider = build_provider(lease.provider, lease.params)
                output = provider.generate(lease.prompt, lease.case_payload, lease.condition)
            except Exception as error:  # noqa: BLE001
                failure = classify_exception(error)
                status = queue.fail_task(
                    conn, lease, failure.kind, str(failure), failure.retryable,
                    self.settings.retry_base_seconds, self.settings.retry_max_seconds,
                )
                with self._lock:
                    self.stats["failed_attempts"] += 1
                    if status is None:
                        self.stats["stale"] += 1
                return
            if queue.complete_task(conn, lease, output.as_record()):
                with self._lock:
                    self.stats["completed"] += 1
            else:
                log.warning("task %s: lease lost before commit; result discarded", lease.task_id)
                with self._lock:
                    self.stats["stale"] += 1
        finally:
            with self._lock:
                self._active.pop(lease.token, None)
                self._lost.discard(lease.token)

    # ------------------------------------------------------- heartbeat + reaper
    def _heartbeat_loop(self) -> None:
        interval = max(0.2, self.settings.lease_seconds / 3)
        last_reap = 0.0
        conn = None
        # Keep heartbeating during graceful shutdown until in-flight tasks finish.
        while not self.stop_event.is_set() or self._active:
            try:
                if conn is None or conn.closed:
                    conn = connect(self.settings.database_url)
                with self._lock:
                    leases = list(self._active.values())
                lost = queue.heartbeat(conn, leases, self.settings.lease_seconds)
                if lost:
                    with self._lock:
                        self._lost |= lost
                    log.warning("lost %d lease(s); their results will be discarded", len(lost))
                now = time.monotonic()
                if now - last_reap >= self.reap_every_seconds:
                    reaped = queue.reap_expired_leases(conn)
                    last_reap = now
                    if reaped:
                        self.stats["reaped"] += reaped
                        log.info("requeued %d task(s) with expired leases", reaped)
            except psycopg.OperationalError:
                log.exception("heartbeat connection problem")
                conn = None
            except Exception:  # noqa: BLE001
                log.exception("heartbeat error")
            time.sleep(min(interval, self.reap_every_seconds))
        if conn is not None:
            conn.close()


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ShiftWatch evaluation worker")
    parser.add_argument("--slots", type=int, default=int(os.environ.get("SHIFTWATCH_WORKER_SLOTS", "4")))
    parser.add_argument("--worker-id")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    Worker(Settings.from_env(), slots=args.slots, worker_id=args.worker_id).run_forever()


if __name__ == "__main__":
    main()
