"""Phase 8: the student as a small HTTP service.

POST /extract {"narrative": "...", "complaint_id": optional, "received": optional ISO date}
  -> {"label": {...} | null, "valid": bool, "notes": [...], "latency_ms": int}
GET /health -> {"status": "ok", "model": "..."}

Every request is appended to a JSONL log (input length, latency, output, validity): this log is
what phase 9's drift monitor reads. Invalid model output is retried once, then returned as
label=null with the reason, never as a half-valid object.

Run:  MODEL_DIR=models/pilot/merged uv run --group ml uvicorn serve.app:app --port 8000
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
LOG = Path(os.environ.get("REQUEST_LOG", ROOT / "data" / "logs" / "requests.jsonl"))


class Request(BaseModel):
    narrative: str
    complaint_id: str | None = None
    received: str | None = None  # date the complaint arrived; replay sets it, live traffic may not


def create_app(extractor=None, log_path=LOG):
    """extractor: object with .extract(text) -> (label, notes, raw, seconds) and .name.
    Loaded from MODEL_DIR on first use when not given (tests pass a fake)."""
    app = FastAPI(title="complaint-extraction")
    state = {"extractor": extractor}
    lock = threading.Lock()  # one generation at a time: the model is not thread-safe on CPU
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def get_extractor():
        if state["extractor"] is None:
            import sys
            sys.path.insert(0, str(ROOT / "train"))
            from predict import Student
            state["extractor"] = Student(os.environ["MODEL_DIR"], int8=os.environ.get("INT8") == "1")
        return state["extractor"]

    @app.get("/health")
    def health():
        ex = state["extractor"]
        return {"status": "ok", "model": ex.name if ex else os.environ.get("MODEL_DIR", "not loaded")}

    @app.post("/extract")
    def extract(req: Request):
        t = time.time()
        with lock:
            ex = get_extractor()
            label, notes, raw, _ = ex.extract(req.narrative)
            attempts = 1
            if label is None:  # one retry; greedy decoding usually repeats, but it is cheap insurance
                label, notes, raw, _ = ex.extract(req.narrative)
                attempts = 2
        ms = int((time.time() - t) * 1000)
        rec = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "complaint_id": req.complaint_id, "received": req.received,
               "input_words": len(req.narrative.split()), "latency_ms": ms, "attempts": attempts,
               "valid": label is not None, "label": label, "notes": notes, "model": ex.name}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"label": label, "valid": label is not None, "notes": notes, "latency_ms": ms}

    return app


app = create_app()
