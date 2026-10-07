"""
PromptShield X — FastAPI entrypoint.

This is a skeleton: it wires up the app, health check, and route stubs so the
project boots. Fill in each module under app/modules/ per the README's build
order, then wire it into app/api/routes.py.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import routes
from app.api import gateway_routes

app = FastAPI(
    title="PromptShield X",
    description="Multi-layer AI firewall for LLMs and RAG pipelines.",
    version="0.1.0",
)

from fastapi.responses import RedirectResponse

app.include_router(routes.router)
app.include_router(gateway_routes.router)

# Serve the audit dashboard's static assets (Chart.js, css, js)
app.mount("/dashboard/static", StaticFiles(directory="dashboard/static"), name="dashboard-static")


@app.get("/")
def root_redirect():
    return RedirectResponse(url="/app")


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "promptshield-x"}
