"""Code execution service for running submitted code in isolated subprocess."""

from __future__ import annotations

import ast
import asyncio
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import List, Optional, Tuple

from config import TIMING

logger = logging.getLogger(__name__)

# Directory for code execution logs and saved scripts
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_CODE_DIR = _LOG_DIR / "code_executions"

# Dedicated logger for code execution output (stdout/stderr)
_exec_logger: Optional[logging.Logger] = None


def _get_exec_logger() -> logging.Logger:
    """Get or create the dedicated code execution output logger."""
    global _exec_logger
    if _exec_logger is not None:
        return _exec_logger

    _CODE_DIR.mkdir(parents=True, exist_ok=True)

    _exec_logger = logging.getLogger("code_execution_output")
    _exec_logger.setLevel(logging.INFO)
    _exec_logger.propagate = False  # Don't send to root logger / agent_server.log

    fh = TimedRotatingFileHandler(
        str(_LOG_DIR / "code_execution.log"),
        when="midnight",
        backupCount=30,
    )
    fh.suffix = "%Y-%m-%d"
    fh.setFormatter(logging.Formatter("%(message)s"))
    _exec_logger.addHandler(fh)

    return _exec_logger


class ExecutionStatus(str, Enum):
    """Status of code execution."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    STOPPED = "stopped"


@dataclass
class ExecutionResult:
    """Result of code execution."""
    status: ExecutionStatus
    execution_id: str
    exit_code: Optional[int]
    stdout: str
    stderr: str
    duration: float
    error: str = ""
    holder: str = ""
    client_host: str = ""
    stop_reason: str = ""
    started_at: float = 0.0
    code: str = ""
    api_key_name: str = ""


@dataclass
class CodeValidationResult:
    """Result of code validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)

    def format_errors(self) -> str:
        """Format errors as a readable string for client feedback."""
        if not self.errors:
            return ""
        header = "Code validation failed. The following issues were found:\n"
        items = "\n".join(f"  - {err}" for err in self.errors)
        footer = "\n\nPlease fix these issues and resubmit."
        return header + items + footer


class CodeValidator:
    """Basic static analysis to catch unintentional dangerous code.

    This is NOT a security sandbox - it catches common mistakes from trusted
    lab agents who may accidentally include dangerous operations.

    Blocked categories:
    - Shell command execution (subprocess, os.system)
    - File deletion (os.remove, shutil.rmtree)
    - Network access (socket, urllib, requests)
    - Dynamic code execution (eval, exec, pickle)
    - Process control (fork, kill, multiprocessing)
    """

    # Imports that agents almost certainly don't need
    BLOCKED_IMPORTS = {
        "subprocess",       # Shell commands
        "shutil",           # rmtree, etc.
        "pickle",           # Code execution via deserialization
        "marshal",          # Same
        "socket",           # Raw network access
        "urllib",           # Network requests
        "requests",         # Network requests
        "httpx",            # Network requests
        "aiohttp",          # Async network requests
        "http",             # HTTP client/server
        "ftplib",           # FTP
        "smtplib",          # Email
        "telnetlib",        # Telnet
        "ctypes",           # C interop, memory access
        "multiprocessing",  # Process spawning
        "pty",              # Pseudo-terminal (shell access)
        "pdb",              # Debugger (can execute arbitrary code)
    }

    # Dangerous function calls: (module, function) or (None, function) for builtins
    BLOCKED_CALLS = {
        # os module dangers - shell/process execution
        ("os", "system"),
        ("os", "popen"),
        ("os", "popen2"),
        ("os", "popen3"),
        ("os", "popen4"),
        ("os", "spawn"),
        ("os", "spawnl"),
        ("os", "spawnle"),
        ("os", "spawnlp"),
        ("os", "spawnlpe"),
        ("os", "spawnv"),
        ("os", "spawnve"),
        ("os", "spawnvp"),
        ("os", "spawnvpe"),
        ("os", "execl"),
        ("os", "execle"),
        ("os", "execlp"),
        ("os", "execlpe"),
        ("os", "execv"),
        ("os", "execve"),
        ("os", "execvp"),
        ("os", "execvpe"),
        ("os", "fork"),
        ("os", "forkpty"),
        ("os", "kill"),
        ("os", "killpg"),
        # os module dangers - file deletion
        ("os", "remove"),
        ("os", "unlink"),
        ("os", "rmdir"),
        ("os", "removedirs"),
        # Builtins - dynamic code execution
        (None, "eval"),
        (None, "exec"),
        (None, "compile"),
        (None, "__import__"),
        # Builtins - file operations (open with write is checked separately)
        (None, "input"),  # Can hang waiting for input
    }

    # Human-readable descriptions for blocked items
    BLOCK_REASONS = {
        "subprocess": "shell command execution",
        "shutil": "file/directory operations (including deletion)",
        "pickle": "code execution via deserialization",
        "marshal": "code execution via deserialization",
        "socket": "raw network access",
        "urllib": "network requests",
        "requests": "network requests",
        "httpx": "network requests",
        "aiohttp": "async network requests",
        "http": "HTTP client/server",
        "ftplib": "FTP access",
        "smtplib": "email sending",
        "telnetlib": "telnet access",
        "ctypes": "C interop and memory access",
        "multiprocessing": "process spawning",
        "pty": "pseudo-terminal (shell access)",
        "pdb": "debugger (can execute arbitrary code)",
        "os.system": "shell command execution",
        "os.popen": "shell command execution",
        "os.fork": "process spawning",
        "os.kill": "process termination",
        "os.remove": "file deletion",
        "os.unlink": "file deletion",
        "os.rmdir": "directory deletion",
        "eval": "dynamic code execution",
        "exec": "dynamic code execution",
        "compile": "dynamic code compilation",
        "__import__": "dynamic module importing",
        "input": "blocking user input (will hang)",
    }

    def validate(self, code: str) -> CodeValidationResult:
        """Validate code for obvious dangerous patterns.

        Args:
            code: Python source code

        Returns:
            CodeValidationResult with valid=True/False and error messages
        """
        errors = []

        # Parse the code (also catches syntax errors)
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return CodeValidationResult(
                valid=False,
                errors=[f"Syntax error at line {e.lineno}: {e.msg}"]
            )

        # Walk the AST looking for dangerous patterns
        for node in ast.walk(tree):
            # Check imports: import x, import x.y
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in self.BLOCKED_IMPORTS:
                        reason = self.BLOCK_REASONS.get(module, "security risk")
                        errors.append(
                            f"Line {node.lineno}: 'import {alias.name}' is not allowed ({reason})"
                        )

            # Check imports: from x import y
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module in self.BLOCKED_IMPORTS:
                        reason = self.BLOCK_REASONS.get(module, "security risk")
                        errors.append(
                            f"Line {node.lineno}: 'from {node.module} import ...' is not allowed ({reason})"
                        )

            # Check function calls
            elif isinstance(node, ast.Call):
                call_info = self._get_call_info(node)
                if call_info:
                    module, func = call_info

                    # Check module.function() calls (e.g., os.system)
                    if module is not None and (module, func) in self.BLOCKED_CALLS:
                        key = f"{module}.{func}"
                        reason = self.BLOCK_REASONS.get(key, "security risk")
                        errors.append(
                            f"Line {node.lineno}: '{module}.{func}()' is not allowed ({reason})"
                        )
                    # Check builtin function calls (e.g., eval, exec)
                    elif module is None and (None, func) in self.BLOCKED_CALLS:
                        reason = self.BLOCK_REASONS.get(func, "security risk")
                        errors.append(
                            f"Line {node.lineno}: '{func}()' is not allowed ({reason})"
                        )

        return CodeValidationResult(
            valid=len(errors) == 0,
            errors=errors
        )

    def _get_call_info(self, node: ast.Call) -> Optional[Tuple[Optional[str], str]]:
        """Extract (module, function) from a Call node.

        Returns:
            Tuple of (module_name, function_name) or (None, function_name) for builtins,
            or None if cannot determine.
        """
        if isinstance(node.func, ast.Attribute):
            # e.g., os.system() -> ("os", "system")
            if isinstance(node.func.value, ast.Name):
                return (node.func.value.id, node.func.attr)
        elif isinstance(node.func, ast.Name):
            # e.g., eval() -> (None, "eval")
            return (None, node.func.id)
        return None


# Module-level validator instance
_validator = CodeValidator()


class CodeExecutor:
    """Manages subprocess execution of submitted code.

    Runs code in an isolated subprocess with access to robot_sdk.
    Enforces timeout from lease system.
    """

    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._execution_id: Optional[str] = None
        self._start_time: Optional[float] = None
        self._last_result: Optional[ExecutionResult] = None
        self._history: List[ExecutionResult] = []  # Last N results
        self._current_code: Optional[str] = None  # User code (without wrapper)
        # Incremental output capture (thread-safe)
        self._stdout_lines: List[str] = []
        self._stderr_lines: List[str] = []
        self._output_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Check if code is currently executing."""
        return self._process is not None and self._process.poll() is None

    def validate_code(self, code: str) -> CodeValidationResult:
        """Validate code before execution.

        Checks for dangerous patterns that trusted lab agents might
        accidentally include. Not a security sandbox.

        Args:
            code: Python source code to validate

        Returns:
            CodeValidationResult with valid=True/False and error messages
        """
        return _validator.validate(code)

    @property
    def status(self) -> ExecutionStatus:
        """Get current execution status."""
        if self._process is None:
            return ExecutionStatus.IDLE
        if self._process.poll() is None:
            return ExecutionStatus.RUNNING
        if self._last_result:
            return self._last_result.status
        return ExecutionStatus.IDLE

    def _read_stream(self, stream, target: str) -> None:
        """Read from a stream line-by-line and accumulate into target list.

        Runs in a background thread. Reads until EOF.

        Args:
            stream: File-like object (process.stdout or process.stderr)
            target: "stdout" or "stderr" to select accumulator
        """
        try:
            for line in stream:
                with self._output_lock:
                    if target == "stdout":
                        self._stdout_lines.append(line)
                    else:
                        self._stderr_lines.append(line)
        except (ValueError, OSError):
            # Stream closed
            pass

    def get_current_output(self) -> Tuple[str, str]:
        """Get accumulated output so far (works during and after execution).

        Returns:
            Tuple of (stdout, stderr) strings
        """
        with self._output_lock:
            return "".join(self._stdout_lines), "".join(self._stderr_lines)

    def _log_execution_output(self, result: ExecutionResult) -> None:
        """Write execution stdout/stderr to the dedicated code_execution.log."""
        log = _get_exec_logger()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = (
            f"{'=' * 72}\n"
            f"[{timestamp}] Execution {result.execution_id} "
            f"| status={result.status.value} exit_code={result.exit_code} "
            f"duration={result.duration:.2f}s"
            f"{f' holder={result.holder}' if result.holder else ''}\n"
            f"{'=' * 72}"
        )
        log.info(header)
        if result.stdout:
            log.info(f"--- stdout ---\n{result.stdout.rstrip()}")
        if result.stderr:
            log.info(f"--- stderr ---\n{result.stderr.rstrip()}")
        if not result.stdout and not result.stderr:
            log.info("(no output)")
        log.info("")

    async def execute(
        self,
        code: str,
        execution_id: str,
        timeout: float = TIMING.code_execution_timeout_s,
        lease_id: Optional[str] = None,
        server_url: str = "http://localhost:8080",
        holder: str = "",
        client_host: str = "",
        api_key_name: str = "",
    ) -> ExecutionResult:
        """Execute code in subprocess.

        Args:
            code: Python code to execute
            execution_id: Unique ID for this execution
            timeout: Maximum execution time in seconds
            lease_id: Lease ID for rewind authorization (optional)
            server_url: Agent server URL for rewind API (default: http://localhost:8080)
            holder: Lease holder name for history tracking

        Returns:
            ExecutionResult with status, stdout, stderr, etc.

        Raises:
            RuntimeError: If code is already running
        """
        if self.is_running:
            raise RuntimeError("Code is already running. Stop it first.")

        self._execution_id = execution_id
        self._start_time = time.time()
        self._lease_id = lease_id
        self._server_url = server_url
        self._holder = holder
        self._client_host = client_host
        self._api_key_name = api_key_name
        self._current_code = code

        # Save code to logs/code_executions/ directory
        temp_file = self._create_temp_file(code)

        logger.info(f"Executing code (ID: {execution_id}): {temp_file}")

        # Clean up old code files (keep last 50)
        self.cleanup_old_code_files()

        # Reset output accumulators
        with self._output_lock:
            self._stdout_lines.clear()
            self._stderr_lines.clear()

        try:
            # Start subprocess
            self._process = subprocess.Popen(
                ["/usr/bin/python3", "-u", str(temp_file)],  # System Python 3.8 (ROS compatible)
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(__file__),  # Set working directory to agent server root
                env=self._get_env(),
            )

            # Save local reference: stop() may set self._process = None
            # while we're awaiting, so we need a stable reference.
            process = self._process

            # Start reader threads for incremental output capture
            stdout_thread = threading.Thread(
                target=self._read_stream, args=(process.stdout, "stdout"), daemon=True
            )
            stderr_thread = threading.Thread(
                target=self._read_stream, args=(process.stderr, "stderr"), daemon=True
            )
            stdout_thread.start()
            stderr_thread.start()

            # Wait for completion or timeout (non-blocking to event loop)
            try:
                loop = asyncio.get_event_loop()
                exit_code = await loop.run_in_executor(None, process.wait, timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                await loop.run_in_executor(None, process.wait)
                # Wait for readers to finish consuming remaining output
                stdout_thread.join(timeout=2.0)
                stderr_thread.join(timeout=2.0)
                stdout, stderr = self.get_current_output()
                duration = time.time() - self._start_time

                result = ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    execution_id=execution_id,
                    exit_code=None,
                    stdout=stdout,
                    stderr=stderr,
                    duration=duration,
                    error=f"Execution timed out after {timeout}s",
                    stop_reason="timeout",
                    started_at=self._start_time,
                    code=code,
                )
            else:
                # Process exited normally - wait for readers to finish
                stdout_thread.join(timeout=5.0)
                stderr_thread.join(timeout=5.0)
                stdout, stderr = self.get_current_output()
                duration = time.time() - self._start_time

                if exit_code == 0:
                    status = ExecutionStatus.COMPLETED
                    error = ""
                else:
                    status = ExecutionStatus.FAILED
                    error = f"Process exited with code {exit_code}"

                result = ExecutionResult(
                    status=status,
                    execution_id=execution_id,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    duration=duration,
                    error=error,
                    started_at=self._start_time,
                    code=code,
                )

        except Exception as e:
            duration = time.time() - self._start_time if self._start_time else 0

            result = ExecutionResult(
                status=ExecutionStatus.FAILED,
                execution_id=execution_id,
                exit_code=None,
                stdout="",
                stderr=str(e),
                duration=duration,
                error=f"Failed to execute code: {e}",
                started_at=self._start_time or 0.0,
                code=code,
            )

        finally:
            self._process = None
            # If stop() already recorded a STOPPED result for this execution,
            # keep that result instead of overwriting with a misleading status.
            if (self._last_result
                    and self._last_result.execution_id == execution_id
                    and self._last_result.status == ExecutionStatus.STOPPED):
                result = self._last_result
            else:
                result.holder = self._holder
                result.client_host = self._client_host
                result.api_key_name = self._api_key_name
                self._last_result = result
            self._history.append(result)
            if len(self._history) > 10:
                self._history = self._history[-10:]

        logger.info(
            f"Execution {execution_id} finished: {result.status} "
            f"(duration: {result.duration:.2f}s, exit_code: {result.exit_code})"
        )
        self._log_execution_output(result)

        return result

    def stop(self, reason: str = "manual") -> bool:
        """Stop currently running code.

        Sends SIGTERM for graceful shutdown, then SIGKILL if needed.

        Args:
            reason: Why the execution was stopped. Common values:
                - "manual": User explicitly stopped via /code/stop
                - "arm_error": Arm error detected
                - "idle_timeout": Lease expired due to idle timeout
                - "max_duration": Lease expired due to max duration
                - "queue_cleared": Lease revoked via queue clear

        Returns:
            True if code was stopped, False if nothing was running
        """
        if not self.is_running:
            return False

        logger.info(f"Stopping execution {self._execution_id} (reason: {reason})")

        # Try graceful shutdown first
        self._process.terminate()

        try:
            # Wait up to 2 seconds for graceful shutdown
            self._process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            # Force kill if still running
            logger.warning(f"Graceful shutdown failed, force killing {self._execution_id}")
            self._process.kill()
            self._process.wait()

        duration = time.time() - self._start_time if self._start_time else 0

        # Give reader threads a moment to flush, then grab accumulated output
        time.sleep(0.1)
        stdout, stderr = self.get_current_output()

        # Human-readable error messages
        reason_messages = {
            "manual": "Stopped by user",
            "arm_error": "Stopped: arm error detected",
            "idle_timeout": "Stopped: lease expired (idle timeout — no commands sent)",
            "max_duration": "Stopped: lease expired (max duration reached)",
            "queue_cleared": "Stopped: lease revoked (queue cleared)",
        }
        error_msg = reason_messages.get(reason, f"Stopped: {reason}")

        self._last_result = ExecutionResult(
            status=ExecutionStatus.STOPPED,
            execution_id=self._execution_id or "unknown",
            exit_code=self._process.returncode,
            stdout=stdout,
            stderr=stderr,
            duration=duration,
            error=error_msg,
            stop_reason=reason,
            started_at=self._start_time or 0.0,
            code=self._current_code or "",
            api_key_name=self._api_key_name,
        )

        self._process = None
        self._log_execution_output(self._last_result)
        return True

    @property
    def current_code(self) -> Optional[str]:
        """Get the user code from the current or last execution (without wrapper)."""
        return self._current_code

    def get_last_result(self) -> Optional[ExecutionResult]:
        """Get result from last execution."""
        return self._last_result

    def get_history(self, count: int = 3) -> List[ExecutionResult]:
        """Get last N execution results (newest first)."""
        return list(reversed(self._history[-count:]))

    def cleanup_old_code_files(self, keep: int = 50) -> None:
        """Remove old code files, keeping the most recent ones.

        Args:
            keep: Number of most recent files to keep (default 50).
        """
        try:
            files = sorted(_CODE_DIR.glob("*.py"), key=lambda f: f.stat().st_mtime)
            for old_file in files[:-keep] if len(files) > keep else []:
                old_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to clean up old code files: {e}")

    def _create_temp_file(self, code: str) -> Path:
        """Create temporary Python file with code + SDK initialization.

        Args:
            code: User-submitted Python code

        Returns:
            Path to temporary file
        """
        # Wrapper code: minimal bootstrap, hardware drivers are started externally
        agent_server_dir = os.path.dirname(os.path.abspath(__file__))
        robot_sdk_dir = os.path.join(agent_server_dir, "robot_sdk")

        wrapper = f'''#!/usr/bin/env python3
"""Auto-generated code execution wrapper.

Architecture:
  - Runs under /usr/bin/python3 (ROS Noetic + MoveIt)
  - env: PiperRobotEnv — arm/gripper control via MoveIt, cameras via ROS
  - yolo: YoloSDK — YOLO HTTP API + ROS depth/TF for 3D projection
  - grasp: GraspSDK — Grasp HTTP API (YOLO + AnyGrasp)
  - memory: MemorySDK — Spatial Memory Hub HTTP API for object upsert/query
  - Grasp exec pose: translation_base_retreat + quaternion_base only
"""

import sys
sys.path.insert(0, "{robot_sdk_dir}")

from piper_sdk import PiperRobotEnv
from yolo_sdk import YoloSDK
from grasp_sdk import GraspSDK
from memory_sdk import MemorySDK

# Robot control (ROS/MoveIt)
env = PiperRobotEnv()

# Perception (HTTP API + ROS)
yolo = YoloSDK()
yolo.start()
grasp = GraspSDK()
grasp.start()

# Spatial Memory Hub (HTTP API)
memory = MemorySDK()

# ============================================================================
# USER CODE STARTS HERE
# ============================================================================

{code}

# ============================================================================
# USER CODE ENDS HERE
# ============================================================================

yolo.stop()
grasp.stop()
'''

        # Save to logs/code_executions/ with timestamped name
        _CODE_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exec_id = getattr(self, "_execution_id", None) or "unknown"
        filename = f"{timestamp}_{exec_id}.py"
        path = _CODE_DIR / filename
        path.write_text(wrapper, encoding="utf-8")

        return path

    def _get_env(self) -> dict:
        """Get environment variables for subprocess.

        Returns current environment with Python path pointing to robot_sdk/.
        """
        env = os.environ.copy()

        agent_server_dir = os.path.dirname(os.path.abspath(__file__))
        robot_sdk_dir = os.path.join(agent_server_dir, "robot_sdk")

        python_path = env.get("PYTHONPATH", "")
        paths = [robot_sdk_dir]
        if python_path:
            paths.append(python_path)
        env["PYTHONPATH"] = ":".join(paths)

        env["PYTHONUNBUFFERED"] = "1"

        return env
