"""FastAPI entry point for timeline and roster preview."""
from contextlib import asynccontextmanager
import os
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api import sleep_router, timeline_router
from app.api.roster_routes import router as roster_router
from app.api.baseline_routes import router as baseline_router
from app.api.reactive_routes import router as reactive_router
from app.services.timeline_service import TimelineService


def create_app(timeline_db_path: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.timeline_service = TimelineService(db_path=timeline_db_path)
        with app.state.timeline_service.store.transaction() as conn:
            conn.execute("UPDATE reactive_jobs SET status='failed', error='服务重启，任务已中断；可重新检查' WHERE status IN ('reviewing', 'queued', 'running')")
        try:
            yield
        finally:
            app.state.timeline_service.close()
    app = FastAPI(title="Rooster Backend", version="0.1.0", lifespan=lifespan)
    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        field = '.'.join(str(x) for x in errors[0]['loc']) if errors else None
        return JSONResponse(status_code=400, content={'error_code': 'validation_error', 'message': 'Request body failed validation', 'field': field, 'request_id': str(uuid.uuid4())})

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if isinstance(exc.detail, dict) and 'error_code' in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(status_code=exc.status_code, content={'error_code': 'validation_error', 'message': str(exc.detail), 'field': None, 'request_id': str(uuid.uuid4())})
    origins = os.environ.get("ROOSTER_CORS_ORIGINS", "").strip()
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in origins.split(",") if x.strip()], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
    app.include_router(timeline_router)
    app.include_router(sleep_router)
    app.include_router(roster_router)
    app.include_router(baseline_router)
    app.include_router(reactive_router)
    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}
    return app

app = create_app()
