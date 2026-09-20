"""FastAPI application for the local tau3 visualization console."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tau3.utils.utils import DATA_DIR
from tau3.web_console.jobs import JobManager
from tau3.web_console.models import (
    ArtifactRef,
    ChangeSetCreate,
    HumanAttribution,
    JobCreate,
)
from tau3.web_console.store import ArtifactStore


def _api_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=f"Unknown {exc.args[0]}")
    return HTTPException(status_code=409, detail=str(exc))


def _static_directory() -> Path | None:
    package_static = Path(__file__).parent / "static"
    source_static = Path(__file__).resolve().parents[3] / "web" / "visualizer" / "dist"
    for directory in (package_static, source_static):
        if (directory / "index.html").exists():
            return directory
    return None


def create_app(data_dir: Path | None = None, refresh_seconds: float = 5) -> FastAPI:
    """Create the same-origin API and static frontend application."""

    store = ArtifactStore(
        Path(data_dir or DATA_DIR),
        refresh_seconds=refresh_seconds,
        background_refresh=True,
    )
    jobs = JobManager(store)
    app = FastAPI(title="tau3 本地可视化与人工校准平台", version="1")
    app.state.store = store
    app.state.jobs = jobs

    @app.get("/api/web/v1/health")
    def health():
        return {
            "status": "ok",
            "data_dir": str(store.data_dir),
            "index": store.index_status(),
        }

    @app.get("/api/web/v1/overview")
    def overview():
        return store.overview() | {"external_job": jobs.external_status()}

    @app.get("/api/web/v1/runs")
    def runs(
        q: str = "",
        page: int = 1,
        page_size: int = Query(25, le=100),
        sort: str = "",
        order: str = Query("asc", pattern="^(asc|desc)$"),
    ):
        return store.list_runs(q, page, page_size, sort, order)

    @app.get("/api/web/v1/synthetic-trajectories")
    def synthetic_trajectories(
        q: str = "",
        page: int = 1,
        page_size: int = Query(25, le=100),
        sort: str = "",
        order: str = Query("asc", pattern="^(asc|desc)$"),
        status: str = Query("", pattern="^(|success|failed|error|unknown|training)$"),
        bundle_id: str = "",
        bundle_task: str = "",
        candidate_trial: str = "",
        termination: str = "",
        model: str = "",
        review: str = Query("", pattern="^(|reviewed|unreviewed)$"),
        package_prefix: str = "",
        task_prefix: str = "",
        dataset_variant: str = Query("", pattern="^(|full|focus|balanced)$"),
    ):
        return store.list_synthetic_trajectories(
            q,
            page,
            page_size,
            sort,
            order,
            status,
            bundle_id,
            bundle_task,
            candidate_trial,
            termination,
            model,
            review,
            package_prefix,
            task_prefix,
            dataset_variant,
        )

    @app.get("/api/web/v1/synthesis-samples/rounds")
    def synthesis_sample_rounds(
        q: str = "",
        page: int = 1,
        page_size: int = Query(50, le=100),
    ):
        return store.list_synthesis_rounds(q, page, page_size)

    @app.get("/api/web/v1/synthesis-samples/rounds/{round_id}/tasks")
    def synthesis_sample_tasks(
        round_id: str,
        q: str = "",
        stage: str = Query("", pattern="^(|pilot|small|validation)$"),
        family: str = "",
        page: int = 1,
        page_size: int = Query(100, le=100),
    ):
        try:
            return store.list_synthesis_tasks(
                round_id, q, stage, family, page, page_size
            )
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/synthesis-samples/rounds/{round_id}/tasks/{task_id}")
    def synthesis_sample_task(round_id: str, task_id: str):
        try:
            return store.synthesis_task_detail(round_id, task_id)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/synthesis-samples/rounds/{round_id}/rows/{row_id}")
    def synthesis_sample_row(round_id: str, row_id: str):
        try:
            return store.synthesis_sample_detail(round_id, row_id)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/classic-view/runs")
    def classic_runs():
        return {"items": store.list_classic_runs()}

    @app.get("/api/web/v1/classic-view/runs/{run_id}/simulations")
    def classic_simulations(
        run_id: str,
        mode: str = Query("all", pattern="^(all|failed|all_failed)$"),
        q: str = "",
        page: int = 1,
        page_size: int = Query(50, le=100),
    ):
        try:
            return store.list_classic_simulations(run_id, mode, q, page, page_size)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/classic-view/runs/{run_id}/tasks")
    def classic_tasks(run_id: str):
        try:
            return {"items": store.list_classic_tasks(run_id)}
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/runs/{run_id}/simulations")
    def run_simulations(
        run_id: str,
        q: str = "",
        failed: bool = False,
        page: int = 1,
        page_size: int = Query(25, le=100),
        sort: str = "",
        order: str = Query("asc", pattern="^(asc|desc)$"),
    ):
        try:
            return store.list_run_simulations(
                run_id, q, failed, page, page_size, sort, order
            )
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/runs/{run_id}/simulations/{simulation_id}")
    def simulation(run_id: str, simulation_id: str):
        try:
            return store.simulation_detail(run_id, simulation_id)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/bundles")
    def bundles(
        q: str = "",
        page: int = 1,
        page_size: int = Query(25, le=100),
        sort: str = "",
        order: str = Query("asc", pattern="^(asc|desc)$"),
        status: str = "",
    ):
        return store.list_bundles(q, page, page_size, sort, order, status)

    @app.get("/api/web/v1/synthetic-packages")
    def synthetic_packages(
        q: str = "",
        page: int = 1,
        page_size: int = Query(25, le=100),
        sort: str = "",
        order: str = Query("asc", pattern="^(asc|desc)$"),
        status: str = "",
        format: str = "",
    ):
        return store.list_synthetic_packages(
            q, page, page_size, sort, order, status, format
        )

    @app.get("/api/web/v1/synthetic-packages/{package_id}/tasks")
    def synthetic_package_tasks(
        package_id: str,
        q: str = "",
        accepted: bool | None = None,
        page: int = 1,
        page_size: int = Query(25, le=100),
        sort: str = "",
        order: str = Query("asc", pattern="^(asc|desc)$"),
    ):
        try:
            return store.list_synthetic_package_tasks(
                package_id, q, page, page_size, sort, order, accepted
            )
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/synthetic-packages/{package_id}/tasks/{task_id}")
    def synthetic_package_task(package_id: str, task_id: str):
        try:
            return store.synthetic_package_task_detail(package_id, task_id)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/bundles/{bundle_id}/tasks")
    def tasks(
        bundle_id: str,
        q: str = "",
        accepted: bool | None = None,
        page: int = 1,
        page_size: int = Query(25, le=100),
        sort: str = "",
        order: str = Query("asc", pattern="^(asc|desc)$"),
    ):
        try:
            return store.list_tasks(
                bundle_id, q, accepted, page, page_size, sort, order
            )
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/bundles/{bundle_id}/tasks/{task_id}")
    def task(bundle_id: str, task_id: str):
        try:
            return store.task_detail(bundle_id, task_id)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/trajectories/{trajectory_id}")
    def trajectory(trajectory_id: str):
        try:
            return store.trajectory_detail(trajectory_id)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.post("/api/web/v1/fixed-diagnoses/resolve")
    def fixed_diagnosis(payload: ArtifactRef):
        """Resolve a logical trajectory to its task-scoped diagnosis file."""

        try:
            return store.fixed_diagnosis(payload)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/documents")
    def documents(
        q: str = "",
        page: int = 1,
        page_size: int = Query(25, le=100),
        sort: str = "",
        order: str = Query("asc", pattern="^(asc|desc)$"),
    ):
        return store.list_documents(q, page, page_size, sort, order)

    @app.get("/api/web/v1/documents/{document_id}")
    def document(document_id: str):
        try:
            return store.document_detail(document_id)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/seed-db/tables")
    def db_tables(
        q: str = "",
        page: int = 1,
        page_size: int = Query(25, le=100),
        sort: str = "",
        order: str = Query("asc", pattern="^(asc|desc)$"),
    ):
        return store.list_db_tables(q, page, page_size, sort, order)

    @app.get("/api/web/v1/seed-db/tables/{table}")
    def db_table(table: str):
        try:
            return store.db_table_detail(table)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/seed-db/tables/{table}/records")
    def db_records(
        table: str,
        q: str = "",
        page: int = 1,
        page_size: int = Query(25, le=100),
        sort: str = "",
        order: str = Query("asc", pattern="^(asc|desc)$"),
    ):
        try:
            return store.list_db_records(table, q, page, page_size, sort, order)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/seed-db/tables/{table}/records/{record_id}")
    def db_record(table: str, record_id: str):
        try:
            return store.db_record_detail(table, record_id)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/change-sets")
    def change_sets():
        return {"items": store.list_change_sets()}

    @app.post("/api/web/v1/change-sets", status_code=201)
    def create_change_set(payload: ChangeSetCreate):
        try:
            return store.create_change_set(payload)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.post("/api/web/v1/change-sets/{change_id}/validate")
    def validate_change_set(change_id: str):
        try:
            return store.validate_change_set(change_id)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.post("/api/web/v1/change-sets/{change_id}/apply")
    def apply_change_set(change_id: str):
        external = jobs.external_status()
        if external and external.get("process_alive"):
            raise HTTPException(
                status_code=423,
                detail="An external synthesis process is active; calibration writes are locked",
            )
        if any(job.status in {"queued", "running"} for job in jobs.list()):
            raise HTTPException(status_code=423, detail="A web-console job is active")
        try:
            return store.apply_change_set(change_id)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/annotations")
    def annotations():
        return {"items": store.list_attributions()}

    @app.post("/api/web/v1/annotations", status_code=201)
    def create_annotation(payload: HumanAttribution):
        try:
            return store.add_attribution(payload)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/jobs")
    def list_jobs(
        artifact_kind: str = "",
        artifact_id: str = "",
        run_id: str = "",
        bundle_id: str = "",
    ):
        records = (
            jobs.list_for_artifact(
                artifact_kind,
                artifact_id,
                run_id=run_id,
                bundle_id=bundle_id,
            )
            if artifact_kind and artifact_id
            else jobs.list()
        )
        return {
            "items": [job.model_dump(mode="json") for job in records],
            "external": jobs.external_status(),
        }

    @app.post("/api/web/v1/jobs", status_code=202)
    def create_job(payload: JobCreate):
        try:
            return jobs.create(payload)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return jobs.get(job_id)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/jobs/{job_id}/log")
    def get_job_log(job_id: str, tail: int = Query(20000, le=200000)):
        try:
            return {"log": jobs.log(job_id, tail)}
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.get("/api/web/v1/jobs/{job_id}/result")
    def get_job_result(job_id: str):
        try:
            return jobs.result(job_id)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.post("/api/web/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        try:
            return jobs.cancel(job_id)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.api_route(
        "/api/{api_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    def unknown_api(api_path: str):
        return JSONResponse(
            status_code=404,
            content={"detail": f"Unknown API endpoint: /api/{api_path}"},
        )

    static = _static_directory()
    if static is not None:
        assets = static / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def frontend(full_path: str):
            candidate = (static / full_path).resolve()
            if (
                full_path
                and candidate.is_relative_to(static.resolve())
                and candidate.is_file()
            ):
                return FileResponse(candidate)
            return FileResponse(static / "index.html")

    else:

        @app.get("/", include_in_schema=False)
        def frontend_missing():
            return JSONResponse(
                {
                    "message": "Frontend is not built. Run npm install && npm run build in web/visualizer.",
                    "docs": "/docs",
                }
            )

    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8001,
    refresh_seconds: float = 5,
    open_browser: bool = False,
) -> None:
    """Run the local console using uvicorn."""

    import threading
    import webbrowser

    import uvicorn

    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"tau3 web console: {url}", flush=True)
    uvicorn.run(create_app(refresh_seconds=refresh_seconds), host=host, port=port)
