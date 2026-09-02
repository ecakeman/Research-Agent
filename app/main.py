from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="Research Agent", version="0.1.0")
app.include_router(router)


@app.get("/healthz")
def healthz():
    return {"ok": True}
