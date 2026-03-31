"""API routes for code execution."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Query, Request
from pydantic import BaseModel, Field

from fastapi.responses import JSONResponse, Response

from code_executor import CodeExecutor, CodeValidationResult, ExecutionResult, ExecutionStatus
from config import TIMING
from execution_recorder import ExecutionRecorder
from lease import LeaseManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/code", tags=["code"])


# Request/Response models
class CodeExecuteRequest(BaseModel):
    """Request to execute code."""
    code: str = Field(..., description="Python code to execute")
    timeout: Optional[float] = Field(None, description="Optional timeout in seconds (default: from lease)")


class CodeExecuteResponse(BaseModel):
    """Response from code execution request."""
    success: bool
    execution_id: str
    message: str = ""
    validation_errors: Optional[list[str]] = None


class CodeStatusResponse(BaseModel):
    """Response with execution status."""
    execution_id: Optional[str]
    status: ExecutionStatus
    is_running: bool
    stdout: str = ""
    stderr: str = ""
    stdout_offset: int = 0
    stderr_offset: int = 0
    duration: float = 0.0
    code: Optional[str] = None
    error: str = ""
    stop_reason: str = ""


class CodeResultResponse(BaseModel):
    """Response with execution result."""
    success: bool
    result: Optional[ExecutionResult]
    error: str = ""


class CodeStopResponse(BaseModel):
    """Response from stop request."""
    success: bool
    message: str


class CodeValidateRequest(BaseModel):
    """Request to validate code without executing."""
    code: str = Field(..., description="Python code to validate")


class CodeValidateResponse(BaseModel):
    """Response from code validation."""
    valid: bool
    errors: list[str] = []
    message: str = ""


# Module-level singletons (shared across routes)
_executor: Optional[CodeExecutor] = None
_recorder: Optional[ExecutionRecorder] = None


def get_executor() -> CodeExecutor:
    """Get or create code executor instance."""
    global _executor
    if _executor is None:
        _executor = CodeExecutor()
    return _executor


def get_recorder() -> ExecutionRecorder:
    """Get or create execution recorder instance."""
    global _recorder
    if _recorder is None:
        _recorder = ExecutionRecorder()
    return _recorder


def init_code_routes(lease_manager: LeaseManager, camera_backend=None, state_agg=None):
    """Initialize code routes with dependencies."""

    @router.post("/execute", response_model=CodeExecuteResponse)
    async def execute_code(
        request: Request,
        body: CodeExecuteRequest,
        x_lease_id: Optional[str] = Header(None),
    ):
        """Execute submitted code in subprocess.

        Requires valid lease. Code runs with access to robot_sdk (robot, yolo).

        Returns immediately with execution_id. Use /code/status to check progress.
        """
        # Verify lease
        if not x_lease_id:
            raise HTTPException(status_code=401, detail="Missing X-Lease-Id header")

        if not lease_manager.validate_lease(x_lease_id):
            raise HTTPException(status_code=403, detail="Invalid or expired lease")

        lease_manager.record_command()

        # Check if code is already running
        executor = get_executor()
        if executor.is_running:
            raise HTTPException(
                status_code=409,
                detail="Code is already running. Stop it first with POST /code/stop"
            )

        # Validate code before execution (catches dangerous patterns)
        validation = executor.validate_code(body.code)
        if not validation.valid:
            logger.warning(f"Code validation failed for lease {x_lease_id}: {validation.errors}")
            return CodeExecuteResponse(
                success=False,
                execution_id="",
                message=validation.format_errors(),
                validation_errors=validation.errors,
            )

        # Generate execution ID
        execution_id = str(uuid.uuid4())[:8]

        # Use timeout from request or central default
        timeout = body.timeout if body.timeout is not None else TIMING.code_execution_timeout_s

        # Get holder name from lease, client IP, and API key name
        lease_info = lease_manager.current_lease
        holder = lease_info.holder if lease_info else ""
        client_host = request.client.host if request.client else ""
        api_key_name = getattr(request.state, "auth_user", "")

        logger.info(f"Executing code (ID: {execution_id}) for lease {x_lease_id} holder={holder} from={client_host}")

        # Execute code in background task (non-blocking)
        import asyncio

        recorder = get_recorder()

        async def run_code():
            recorder.start(execution_id, camera_backend, state_agg)
            try:
                result = await executor.execute(
                    code=body.code,
                    execution_id=execution_id,
                    timeout=timeout,
                    lease_id=x_lease_id,  # Pass lease for rewind API
                    holder=holder,
                    client_host=client_host,
                    api_key_name=api_key_name,
                )
                logger.info(f"Code execution {execution_id} finished: {result.status}")
            except Exception as e:
                logger.error(f"Code execution {execution_id} failed: {e}", exc_info=True)
            finally:
                recorder.stop()
                recorder.cleanup_old_recordings()

        # Start execution as background task
        task = asyncio.create_task(run_code())
        request.app.state.background_tasks.add(task)

        return CodeExecuteResponse(
            success=True,
            execution_id=execution_id,
            message=f"Code execution started (ID: {execution_id})",
        )

    @router.post("/validate", response_model=CodeValidateResponse)
    async def validate_code(body: CodeValidateRequest):
        """Validate code without executing it.

        Checks for dangerous patterns (shell commands, network access, file deletion, etc.)
        that trusted lab agents might accidentally include.

        No lease required. Use this to pre-check code before submitting to /execute.
        """
        executor = get_executor()
        validation = executor.validate_code(body.code)

        if validation.valid:
            return CodeValidateResponse(
                valid=True,
                errors=[],
                message="Code validation passed",
            )
        else:
            return CodeValidateResponse(
                valid=False,
                errors=validation.errors,
                message=validation.format_errors(),
            )

    @router.post("/stop", response_model=CodeStopResponse)
    async def stop_code(x_lease_id: Optional[str] = Header(None)):
        """Stop currently running code.

        Requires valid lease. Sends SIGTERM for graceful shutdown.
        """
        # Verify lease
        if not x_lease_id:
            raise HTTPException(status_code=401, detail="Missing X-Lease-Id header")

        if not lease_manager.validate_lease(x_lease_id):
            raise HTTPException(status_code=403, detail="Invalid or expired lease")

        lease_manager.record_command()

        executor = get_executor()
        if not executor.is_running:
            return CodeStopResponse(
                success=False,
                message="No code is currently running",
            )

        logger.info(f"Stopping code execution for lease {x_lease_id}")
        stopped = executor.stop(reason="manual")

        if stopped:
            return CodeStopResponse(
                success=True,
                message="Code execution stopped",
            )
        else:
            return CodeStopResponse(
                success=False,
                message="Failed to stop code execution",
            )

    @router.get("/status", response_model=CodeStatusResponse)
    async def get_status(
        stdout_offset: int = Query(0, ge=0, description="Character offset into stdout; only output after this position is returned"),
        stderr_offset: int = Query(0, ge=0, description="Character offset into stderr; only output after this position is returned"),
    ):
        """Get current execution status.

        Returns execution ID, status, whether code is running, and live output.
        Pass stdout_offset/stderr_offset from the previous response to receive
        only new output since the last poll.
        No lease required (read-only).
        """
        executor = get_executor()
        stdout, stderr = "", ""
        duration = 0.0
        error, stop_reason = "", ""
        if executor.is_running:
            stdout, stderr = executor.get_current_output()
            if executor._start_time:
                duration = time.time() - executor._start_time
        else:
            # Pull final output and stop info from last result
            last = executor.get_last_result()
            if last:
                stdout = last.stdout
                stderr = last.stderr
                duration = last.duration
                error = last.error
                stop_reason = last.stop_reason
        # Slice to return only new output since the requested offsets
        full_stdout_len = len(stdout)
        full_stderr_len = len(stderr)
        stdout = stdout[stdout_offset:]
        stderr = stderr[stderr_offset:]
        return CodeStatusResponse(
            execution_id=executor._execution_id,
            status=executor.status,
            is_running=executor.is_running,
            stdout=stdout,
            stderr=stderr,
            stdout_offset=full_stdout_len,
            stderr_offset=full_stderr_len,
            duration=duration,
            code=executor.current_code,
            error=error,
            stop_reason=stop_reason,
        )

    @router.get("/result", response_model=CodeResultResponse)
    async def get_result():
        """Get result from last execution.

        Returns stdout, stderr, exit code, duration, etc.
        No lease required (read-only).
        """
        executor = get_executor()
        result = executor.get_last_result()

        if result is None:
            return CodeResultResponse(
                success=False,
                result=None,
                error="No execution result available",
            )

        return CodeResultResponse(
            success=True,
            result=result,
            error="",
        )

    @router.get("/history")
    async def get_history(count: int = 3):
        """Get last N execution results (newest first).

        No lease required (read-only).
        """
        executor = get_executor()
        history = executor.get_history(count)
        return {"history": history}

    @router.get("/recordings/{execution_id}")
    async def get_recording(execution_id: str):
        """Get recording metadata with frames matched to nearest state.

        Each entry in the ``timeline`` array pairs a frame with the
        closest state sample by timestamp.  No lease required (read-only).
        """
        import bisect
        from execution_recorder import _CODE_DIR

        recorder = get_recorder()
        metadata = recorder.get_recording(execution_id)
        if metadata is None:
            return JSONResponse(
                {"error": f"No recording found for execution {execution_id}"},
                status_code=404,
            )

        # Load state log timestamps for matching
        state_log_path = _CODE_DIR / execution_id / "state_log.jsonl"
        state_times: list[float] = []
        state_lines: list[dict] = []
        if state_log_path.exists():
            for line in state_log_path.read_text().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                state_times.append(entry.get("timestamp", 0.0))
                state_lines.append(entry)

        # Build timeline: match each frame to nearest state
        timeline = []
        timestamps = metadata.get("timestamps", [])
        frames = metadata.get("frames", [])
        for i, (frame, t) in enumerate(zip(frames, timestamps)):
            matched_state = None
            if state_times:
                idx = bisect.bisect_left(state_times, t)
                # Pick whichever neighbor is closer
                best = None
                if idx < len(state_times):
                    best = idx
                if idx > 0 and (best is None or abs(state_times[idx - 1] - t) <= abs(state_times[best] - t)):
                    best = idx - 1
                if best is not None:
                    matched_state = state_lines[best]
            timeline.append({
                "frame": frame,
                "timestamp": t,
                "state": matched_state,
            })

        return {
            "execution_id": metadata.get("execution_id"),
            "started_at": metadata.get("started_at"),
            "stopped_at": metadata.get("stopped_at"),
            "duration": metadata.get("duration"),
            "cameras": metadata.get("cameras", []),
            "frame_count": metadata.get("frame_count", 0),
            "state_samples": metadata.get("state_samples", 0),
            "timeline": timeline,
        }

    @router.get("/recordings/{execution_id}/frames/{filename}")
    async def get_recording_frame(execution_id: str, filename: str):
        """Serve a recorded JPEG frame.

        No lease required (read-only).
        """
        from pathlib import Path

        # Sanitize filename to prevent path traversal
        safe_name = Path(filename).name
        if safe_name != filename or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        from execution_recorder import _CODE_DIR
        filepath = _CODE_DIR / execution_id / safe_name
        if not filepath.exists() or not filepath.suffix == ".jpg":
            raise HTTPException(status_code=404, detail="Frame not found")

        return Response(
            content=filepath.read_bytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @router.get("/recordings")
    async def list_recordings():
        """List all execution IDs that have recordings (newest first).

        No lease required (read-only).
        """
        recorder = get_recorder()
        return {"recordings": recorder.list_recordings()}

    return router
