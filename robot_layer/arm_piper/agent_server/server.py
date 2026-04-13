"""Hardware server — FastAPI app wiring everything together."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

# Add project root to path
_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SERVER_DIR)
for _p in [_PROJECT_ROOT, _SERVER_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from auth import APIKeyMiddleware, KeyStore
from config import LeaseConfig, ServerConfig, ServiceManagerConfig, default_services
from lease import LeaseManager
from display_state import DisplayBroadcaster
from services import ServiceManager
from state import StateAggregator

from logging_config import setup_logging

logger = setup_logging("agent_server")


def build_app(cfg: ServerConfig, service_mgr: ServiceManager | None = None) -> FastAPI:
    app = FastAPI(title="Piper Robot Agent Server")

    # -- API key auth --------------------------------------------------------
    keys_path = os.path.join(_SERVER_DIR, "api_keys.json")
    key_store = KeyStore(keys_path)
    app.add_middleware(APIKeyMiddleware, key_store=key_store)
    app.state.key_store = key_store

    app.state.background_tasks = set()

    @app.get("/", include_in_schema=False)
    async def root():
        if cfg.dashboard:
            return RedirectResponse(url="/services/dashboard")
        return {"status": "ok", "message": "Piper Robot Agent Server", "docs": "/docs"}

    # -- 状态聚合器 (PiperRobotEnv 在代码执行子进程中初始化, 这里用 None 表示轻量监控) ------
    # StateAggregator 在 server 进程中以 env=None 启动，仅用于 /state API 端点展示轮廓。
    # 实际机器人控制在 code_executor 的子进程中通过 PiperRobotEnv 完成。
    state_agg = StateAggregator(env=None, poll_hz=cfg.base.poll_hz)

    display = DisplayBroadcaster()

    lease_mgr = LeaseManager(
        cfg.lease,
        last_moved_at_fn=state_agg.last_moved_at,
    )

    # -- routes --------------------------------------------------------------
    from routes.lease_routes import create_router as lease_router
    from routes.state_routes import create_router as state_router
    from routes.ws import create_router as ws_router
    from routes.code_routes import init_code_routes
    from routes.sdk_docs import router as sdk_docs_router
    from routes.system_guide import router as system_guide_router
    from routes.yolo_routes import router as yolo_router
    from routes.display_routes import create_router as display_router

    app.include_router(state_router(state_agg, None, lease_mgr, None, None, None, None, None))
    app.include_router(lease_router(lease_mgr))
    app.include_router(ws_router(state_agg, cfg, None, key_store=key_store))
    app.include_router(init_code_routes(lease_mgr, None, state_agg))
    app.include_router(sdk_docs_router)
    app.include_router(system_guide_router)
    app.include_router(yolo_router)
    app.include_router(display_router(display, key_store=key_store))

    # Service manager routes (includes dashboard)
    if cfg.dashboard and service_mgr is not None:
        from routes.service_routes import create_router as service_router
        app.include_router(service_router(service_mgr))

    # -- lifecycle -----------------------------------------------------------
    @app.on_event("startup")
    async def startup():
        logger.info("Starting Piper Agent Server (dry_run=%s)", cfg.dry_run)

        if service_mgr is not None:
            await service_mgr.start()

        await state_agg.start()
        await lease_mgr.start()

        # Display status polling (1 Hz)
        async def _display_status_loop():
            from routes.code_routes import get_executor
            prev_running = False
            while True:
                try:
                    executor = get_executor()
                    is_running = executor.is_running
                    lease_status = lease_mgr.status()
                    queue_length = lease_status.get("queue_length", 0)
                    holder = lease_status.get("holder", "") or ""

                    status = "executing" if is_running else "idle"
                    display.update_robot_status(status, queue_length, holder)

                    if prev_running and not is_running:
                        display.on_execution_ended()
                    prev_running = is_running
                except Exception:
                    pass
                await asyncio.sleep(1.0)

        task = asyncio.create_task(_display_status_loop())
        app.state.background_tasks.add(task)
        task.add_done_callback(app.state.background_tasks.discard)

        if key_store.enabled:
            logger.info("API key auth ENABLED (%d keys loaded)", len(key_store._keys))
        else:
            logger.info("API key auth DISABLED (no keys configured)")
        logger.info("Piper Agent Server ready on %s:%d", cfg.host, cfg.port)

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("Shutting down Piper Agent Server")

        try:
            from routes.code_routes import get_executor
            executor = get_executor()
            if executor.is_running:
                executor.stop()
            executor.cleanup_old_code_files()
        except Exception as e:
            logger.warning("Failed to cleanup code executor: %s", e)

        await lease_mgr.stop()
        await state_agg.stop()

        if service_mgr is not None:
            await service_mgr.stop()

    return app


def main():
    parser = argparse.ArgumentParser(description="Piper Robot Agent Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--dry-run", action="store_true", help="Dry-run mode (no real robot)")
    parser.add_argument(
        "--auto-start-services",
        action="store_true",
        help="Auto-start backend services on startup",
    )
    parser.add_argument(
        "--no-service-manager",
        action="store_true",
        help="Disable service management entirely",
    )
    parser.add_argument(
        "--no-reset-on-release",
        action="store_true",
        help="Disable auto-reset when lease ends",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable the web dashboard",
    )
    args = parser.parse_args()

    svc_mgr_cfg = ServiceManagerConfig(
        enabled=not args.no_service_manager,
        auto_start=args.auto_start_services,
    )
    lease_cfg = LeaseConfig()
    if args.no_reset_on_release:
        lease_cfg.reset_on_release = False

    cfg = ServerConfig(
        host=args.host,
        port=args.port,
        dry_run=args.dry_run,
        service_manager=svc_mgr_cfg,
        lease=lease_cfg,
        dashboard=not args.no_dashboard,
    )

    service_mgr = None
    if cfg.service_manager.enabled:
        service_mgr = ServiceManager(
            config=cfg.service_manager,
            services=default_services(),
            dry_run=cfg.dry_run,
        )

    app = build_app(cfg, service_mgr=service_mgr)
    uvicorn.run(app, host=cfg.host, port=cfg.port, access_log=False)


if __name__ == "__main__":
    main()
