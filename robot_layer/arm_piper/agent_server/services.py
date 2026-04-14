"""Service Manager — manages backend processes for the hardware server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable

from config import ServiceDefinition, ServiceManagerConfig

logger = logging.getLogger(__name__)


@dataclass
class ServiceState:
    """Runtime state for a managed service."""
    definition: ServiceDefinition
    process: subprocess.Popen | None = None
    start_time: float | None = None
    logs: deque = field(default_factory=lambda: deque(maxlen=100))

    def is_running(self) -> bool:
        """Check if the service process is still running."""
        if self.process is None:
            return False
        ret = self.process.poll()
        if ret is not None:
            self.logs.append(f"[exited with code {ret}]")
            self.process = None
            self.start_time = None
            return False
        return True


class ServiceManager:
    """Manages backend services (ROS nodes, drivers, etc.)."""

    def __init__(
        self,
        config: ServiceManagerConfig,
        services: dict[str, ServiceDefinition],
        dry_run: bool = False,
        on_event: Callable[[dict], None] | None = None,
    ) -> None:
        self._config = config
        self._dry_run = dry_run
        self._on_event = on_event
        self._lock = Lock()
        self._health_task: asyncio.Task | None = None
        self._running = False

        # Initialize service states
        self._services: dict[str, ServiceState] = {}
        for key, defn in services.items():
            state = ServiceState(
                definition=defn,
                logs=deque(maxlen=config.log_max_lines),
            )
            self._services[key] = state

        # PID file path
        self._pid_file = Path(config.pid_file)

    @property
    def service_keys(self) -> list[str]:
        """Return list of managed service keys."""
        return list(self._services.keys())

    async def start(self) -> None:
        """Start the service manager (cleanup orphans, optionally auto-start services)."""
        logger.info("Starting service manager (dry_run=%s)", self._dry_run)
        self._restore_or_cleanup()
        self._running = True

        # Start health check loop
        self._health_task = asyncio.create_task(self._health_check_loop())

        # Auto-start services if configured
        if self._config.auto_start:
            logger.info("Auto-starting backend services")
            for key in self._services:
                await self.start_service(key)
                # Wait for service to initialize before starting next one
                await asyncio.sleep(3.0)

    async def stop(self) -> None:
        """Stop the service manager and all managed services."""
        logger.info("Stopping service manager")
        self._running = False

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Stop all running services
        for key in self._services:
            await self.stop_service(key)

    async def start_service(self, key: str) -> dict[str, Any]:
        """Start a service by key."""
        if key not in self._services:
            return {"ok": False, "error": f"Unknown service: {key}"}

        with self._lock:
            state = self._services[key]
            if state.is_running():
                return {"ok": False, "error": "Service already running", "key": key}

            defn = state.definition

            # Check dependencies before starting
            if defn.depends_on and not self._dry_run:
                missing = self._check_dependencies(key)
                if missing:
                    return {
                        "ok": False,
                        "error": f"Dependencies not running: {missing}",
                        "key": key,
                    }

            if self._dry_run:
                logger.info("[DRY-RUN] Would start service %s: %s", key, defn.cmd)
                state.logs.append(f"[dry-run] would start: {defn.cmd}")
                # Simulate running state for dry-run
                state.start_time = time.time()
                self._emit_event("service_started", key, dry_run=True)
                return {"ok": True, "message": f"[dry-run] started {key}", "key": key}

            # Build full command with shell prefix
            full_cmd = defn.shell_prefix + defn.cmd

            try:
                proc = subprocess.Popen(
                    ["bash", "-c", full_cmd],
                    cwd=defn.cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                )
                state.process = proc
                state.start_time = time.time()
                state.logs.append(f"[started pid={proc.pid}]")

                # Start log reader thread
                Thread(
                    target=self._log_reader,
                    args=(key, proc.stdout),
                    daemon=True,
                ).start()

                self._save_pids()
                self._emit_event("service_started", key, pid=proc.pid)

                logger.info("Started service %s (pid=%d)", key, proc.pid)
                return {"ok": True, "message": f"started pid={proc.pid}", "key": key, "pid": proc.pid}

            except Exception as e:
                logger.exception("Failed to start service %s", key)
                state.logs.append(f"[failed to start: {e}]")
                return {"ok": False, "error": str(e), "key": key}

    async def stop_service(self, key: str) -> dict[str, Any]:
        """Stop a service by key."""
        if key not in self._services:
            return {"ok": False, "error": f"Unknown service: {key}"}

        with self._lock:
            state = self._services[key]
            defn = state.definition

            if self._dry_run:
                logger.info("[DRY-RUN] Would stop service %s", key)
                state.logs.append("[dry-run] would stop")
                state.start_time = None
                self._emit_event("service_stopped", key, dry_run=True)
                return {"ok": True, "message": f"[dry-run] stopped {key}", "key": key}

            stopped_tracked = False
            proc = state.process

            # Try to stop tracked process first
            if proc is not None and proc.poll() is None:
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(pgid, signal.SIGKILL)
                        proc.wait(timeout=3)
                    stopped_tracked = True
                except (ProcessLookupError, PermissionError):
                    pass

            # Always also kill by pattern to catch orphans
            self._kill_by_pattern(defn.kill_patterns)

            state.process = None
            state.start_time = None
            self._save_pids()
            state.logs.append("[stopped]")
            self._emit_event("service_stopped", key)

            logger.info("Stopped service %s", key)
            return {"ok": True, "message": "stopped", "key": key}

    async def restart_service(self, key: str) -> dict[str, Any]:
        """Restart a service by key."""
        stop_result = await self.stop_service(key)
        if not stop_result.get("ok", False) and "already" not in stop_result.get("error", ""):
            return stop_result

        # Brief pause to ensure cleanup
        await asyncio.sleep(0.5)

        return await self.start_service(key)

    def get_status(self, key: str | None = None) -> list[dict] | dict:
        """Get status of one or all services."""
        with self._lock:
            if key is not None:
                if key not in self._services:
                    return {"error": f"Unknown service: {key}"}
                return self._service_status(key)

            return [self._service_status(k) for k in self._services]

    def _service_status(self, key: str) -> dict:
        """Get status dict for a single service (called with lock held)."""
        state = self._services[key]
        running = state.is_running() if not self._dry_run else state.start_time is not None
        pid = state.process.pid if state.process and running else None
        uptime = None
        if running and state.start_time:
            uptime = int(time.time() - state.start_time)

        return {
            "key": key,
            "name": state.definition.name,
            "running": running,
            "pid": pid,
            "uptime": uptime,
            "dry_run": self._dry_run,
        }

    def get_logs(self, key: str, lines: int = 50) -> dict:
        """Get recent log output for a service."""
        if key not in self._services:
            return {"error": f"Unknown service: {key}"}

        with self._lock:
            state = self._services[key]
            log_list = list(state.logs)
            # Return last N lines
            return {
                "key": key,
                "lines": log_list[-lines:] if lines < len(log_list) else log_list,
            }

    def _log_reader(self, key: str, stream) -> None:
        """Background thread that drains stdout into the log deque."""
        try:
            for line in iter(stream.readline, b""):
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                with self._lock:
                    self._services[key].logs.append(text)
        except Exception:
            pass

    def _save_pids(self) -> None:
        """Persist tracked PIDs to disk for crash recovery."""
        data = {}
        for key, state in self._services.items():
            if state.process is not None and state.process.poll() is None:
                data[key] = state.process.pid
        try:
            with open(self._pid_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning("Failed to save PIDs: %s", e)

    def _restore_or_cleanup(self) -> None:
        """On startup, kill orphaned processes from a *previous server run* only.

        Only kills processes whose PIDs were saved in the PID file.  Does NOT
        blindly ``pkill`` by pattern, because that would murder externally
        managed processes (e.g. a roscore / camera driver the user started
        manually before the server).
        """
        if self._pid_file.exists():
            try:
                with open(self._pid_file) as f:
                    old_pids = json.load(f)
                for key, pid in old_pids.items():
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                        logger.info("Killed orphaned process group for %s (pid=%d)", key, pid)
                    except (ProcessLookupError, PermissionError):
                        pass
            except Exception as e:
                logger.warning("Failed to restore PIDs: %s", e)
            try:
                self._pid_file.unlink()
            except Exception:
                pass

    def _kill_by_pattern(self, patterns: list[str]) -> None:
        """Kill any process matching the given command-line patterns."""
        for pat in patterns:
            subprocess.run(["pkill", "-f", pat], capture_output=True)
        # Follow up with SIGKILL after delay for stubborn processes
        # Give enough time for cleanup handlers to complete (e.g., network calls)
        time.sleep(3)
        for pat in patterns:
            subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)

    def _check_dependencies(self, key: str) -> list[str]:
        """Return list of missing dependencies for a service."""
        state = self._services[key]
        missing = []
        for dep_key in state.definition.depends_on:
            if dep_key in self._services:
                dep_state = self._services[dep_key]
                if not dep_state.is_running():
                    missing.append(dep_key)
        return missing

    async def _health_check_loop(self) -> None:
        """Async loop that monitors service health."""
        while self._running:
            await asyncio.sleep(self._config.health_check_interval_s)

            services_to_stop = []

            with self._lock:
                for key, state in self._services.items():
                    if self._dry_run:
                        continue

                    # Check if a previously running service has crashed
                    was_running = state.process is not None
                    is_running = state.is_running()

                    if was_running and not is_running:
                        logger.warning("Service %s has crashed", key)
                        self._emit_event("service_crashed", key)

                        # Auto-restart if configured
                        if state.definition.auto_restart:
                            logger.info("Auto-restarting service %s", key)
                            # Schedule restart outside the lock
                            asyncio.create_task(self._auto_restart(key))

                    # Check dependencies - stop service if dependencies are down
                    if is_running and state.definition.depends_on:
                        missing = self._check_dependencies(key)
                        if missing:
                            logger.warning(
                                "Service %s dependencies not running: %s - stopping",
                                key, missing
                            )
                            state.logs.append(f"[stopping: dependencies down: {missing}]")
                            services_to_stop.append(key)

            # Stop services outside the lock
            for key in services_to_stop:
                await self.stop_service(key)

    async def _auto_restart(self, key: str) -> None:
        """Auto-restart a crashed service."""
        await asyncio.sleep(1)  # Brief delay before restart
        result = await self.start_service(key)
        if result.get("ok"):
            logger.info("Auto-restarted service %s", key)
        else:
            logger.error("Failed to auto-restart service %s: %s", key, result.get("error"))

    def _emit_event(self, event_type: str, key: str, **kwargs) -> None:
        """Emit a service event via the callback."""
        if self._on_event:
            event = {
                "type": event_type,
                "service": key,
                "timestamp": time.time(),
                **kwargs,
            }
            try:
                self._on_event(event)
            except Exception as e:
                logger.warning("Failed to emit event: %s", e)
