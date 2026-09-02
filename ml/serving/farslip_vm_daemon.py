"""Pub/Sub daemon for the Compute Engine VM `agrosat-farslip-trainer` (US-022-c P1 fix).

Replaces the Vertex AI Custom Job pattern (finicky scheduling in us-central1
saturated with L4) with a persistent VM with on-demand L4 + Pub/Sub
event-driven + idle auto-shutdown 5 min.

Flow:
  1) Subscribes to `agrosat-farslip-jobs` (subscription `farslip-vm-sub`).
  2) For each message: runs the payload's shell command (uses `subprocess.run`),
     logs stdout/stderr to Cloud Logging via structlog.
  3) Acknowledges the message (ack) only after the exit code (success or fail).
  4) When the queue has gone >= IDLE_SHUTDOWN_SECONDS without new messages AND no
     process is running, it runs `shutdown -h now` to auto-power-off the VM.

Expected Pub/Sub payload (JSON UTF-8):
  {
    "command": "make farslip-train ...",   # shell command to run (workdir /app)
    "label": "smoke-farslip-2026-05-24",   # optional log identifier
    "timeout_seconds": 21600               # individual cap; default 28800 (8h)
  }

Environment variables (injected by systemd / cloud-init):
  PROJECT_ID                    GCP project id (default: agrosat-copilot)
  SUBSCRIPTION_ID               Pub/Sub subscription id (default: farslip-vm-sub)
  IDLE_SHUTDOWN_SECONDS         seconds of empty queue before shutdown (default: 300)
  WORKDIR                       directory where commands run (default: /app)
  LOG_LEVEL                     INFO|DEBUG|WARNING (default: INFO)

Output:
  - Cloud Logging via structlog (JSON).
  - stdout/stderr of each command included in the structured log.
  - Exit code 0 on SIGTERM (cloud-init shutdown). Never a voluntary exit 1.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any

try:
    import structlog
    from google.cloud import pubsub_v1
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        f"FATAL: dependencias missing ({exc}). Run pip install google-cloud-pubsub structlog\n"
    )
    sys.exit(1)


_PROJECT_ID = os.environ.get("PROJECT_ID", "agrosat-copilot")
_SUBSCRIPTION_ID = os.environ.get("SUBSCRIPTION_ID", "farslip-vm-sub")
_IDLE_SHUTDOWN_SECONDS = int(os.environ.get("IDLE_SHUTDOWN_SECONDS", "300"))
_WORKDIR = os.environ.get("WORKDIR", "/app")
_DEFAULT_TIMEOUT = int(os.environ.get("DEFAULT_JOB_TIMEOUT_SECONDS", "28800"))

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(__import__("logging"), os.environ.get("LOG_LEVEL", "INFO"))
    ),
)
log = structlog.get_logger(__name__)


@dataclass
class DaemonState:
    """Shared state between the Pub/Sub thread and the idle watchdog."""

    last_message_at: float
    job_running: bool
    shutdown_requested: Event
    lock: Lock


def _trigger_shutdown(reason: str) -> None:
    """Run shutdown -h now with the recorded reason (idempotent)."""
    log.info("auto_shutdown_triggered", reason=reason)
    try:
        # Daemon de control de VM: comando fijo de apagado (no input de usuario).
        cmd = ["sudo", "shutdown", "-h", "+1", reason]
        subprocess.run(cmd, check=False, timeout=10)  # noqa: S603
    # Best-effort shutdown: swallow everything, never re-raise (a failure here
    # must not kill the daemon loop). BLE001 does not fire because the handler
    # logs with exc_info, so no noqa is needed.
    except Exception as exc:
        log.error("shutdown_failed", exc_info=str(exc))


def _run_command(payload: dict[str, Any]) -> int:
    """Run the payload's command in WORKDIR; returns the exit code."""
    command = payload.get("command")
    if not command:
        log.warning("payload_missing_command", payload_keys=list(payload.keys()))
        return 2
    label = payload.get("label", "unlabeled")
    timeout = int(payload.get("timeout_seconds", _DEFAULT_TIMEOUT))
    log.info("job_start", label=label, command=command, timeout=timeout, workdir=_WORKDIR)
    start = time.monotonic()
    try:
        result = subprocess.run(  # noqa: S602 - daemon de jobs: shell intencional, payload de canal Pub/Sub propio
            command,
            shell=True,
            cwd=_WORKDIR,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed = time.monotonic() - start
        log.info(
            "job_end",
            label=label,
            exit_code=result.returncode,
            elapsed_seconds=round(elapsed, 1),
            stdout_tail=result.stdout[-2000:] if result.stdout else "",
            stderr_tail=result.stderr[-2000:] if result.stderr else "",
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        log.error("job_timeout", label=label, elapsed_seconds=round(elapsed, 1), timeout=timeout)
        return 124


def _make_callback(state: DaemonState):
    def callback(message: pubsub_v1.subscriber.message.Message) -> None:
        try:
            payload = json.loads(message.data.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - el callback nunca debe tumbar el subscriber
            log.error("payload_decode_failed", error=str(exc), raw=message.data[:200])
            message.ack()
            return
        with state.lock:
            state.job_running = True
            state.last_message_at = time.monotonic()
        try:
            exit_code = _run_command(payload)
            log.info("ack_message", message_id=message.message_id, exit_code=exit_code)
        finally:
            with state.lock:
                state.job_running = False
                state.last_message_at = time.monotonic()
            message.ack()

    return callback


def _watchdog_loop(state: DaemonState) -> None:
    """Watch idle time and trigger shutdown when IDLE_SHUTDOWN_SECONDS is exceeded."""
    while not state.shutdown_requested.is_set():
        time.sleep(30)
        with state.lock:
            idle = time.monotonic() - state.last_message_at
            running = state.job_running
        if not running and idle >= _IDLE_SHUTDOWN_SECONDS:
            _trigger_shutdown(f"idle {int(idle)}s >= threshold {_IDLE_SHUTDOWN_SECONDS}s")
            state.shutdown_requested.set()
            return
        log.debug("watchdog_tick", idle_seconds=round(idle, 1), job_running=running)


def main() -> int:
    """Entrypoint: starts subscriber + watchdog. Blocks until SIGTERM."""
    state = DaemonState(
        last_message_at=time.monotonic(),
        job_running=False,
        shutdown_requested=Event(),
        lock=Lock(),
    )

    def _handle_sigterm(_signum, _frame):
        log.info("sigterm_received")
        state.shutdown_requested.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(_PROJECT_ID, _SUBSCRIPTION_ID)
    log.info(
        "daemon_start",
        project=_PROJECT_ID,
        subscription=_SUBSCRIPTION_ID,
        idle_shutdown_seconds=_IDLE_SHUTDOWN_SECONDS,
        workdir=_WORKDIR,
    )

    from threading import Thread

    watchdog = Thread(target=_watchdog_loop, args=(state,), daemon=True)
    watchdog.start()

    future = subscriber.subscribe(subscription_path, callback=_make_callback(state))
    try:
        future.result(timeout=None)
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001 - punto de entrada del daemon: log + exit code, no propagar
        log.error("subscriber_crashed", error=str(exc))
        return 1
    finally:
        future.cancel()
        subscriber.close()
        log.info("daemon_exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
