"""
AxiLattice FastAPI Backend
──────────────────────────
Endpoints:
  POST /upload              → ingest CSV/Excel/Parquet, build cube, return schema
  POST /query               → NLU + cube lookup → return card data
  GET  /schema              → current schema + cube stats
  GET  /suggest             → contextual query suggestions for current schema
  POST /dashboard           → save a named dashboard
  GET  /dashboard/{id}      → load a dashboard
  GET  /periods/{grain}     → available period keys for a grain
  GET  /health              → status check

State: one active dataset per server (session-less by design for Phase 1).
       Extend to per-user sessions by keying cube by session token.
"""

import os
import io
import json
import uuid
import time
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from profiler import DataProfiler
from cube import CubeEngine
from nlu import parse_intent

# ── App init ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AxiLattice Engine",
    description="Pre-computed insight engine: Cube + Voice + NLU",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state (single-tenant Phase 1) ─────────────────────────────────────
_STATE: Dict[str, Any] = {
    "cube":         None,
    "profiler":     None,
    "schema_ctx":   None,
    "df":           None,
    "build_status": "idle",   # idle | building | ready | error
    "build_error":  None,
    "dashboards":   {},        # {id: dashboard_dict}
    "query_history": [],       # last 20 queries for context
}


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    text:       str
    session_id: Optional[str] = "default"

class DashboardSaveRequest(BaseModel):
    name:   str
    layout: str = "grid"
    cards:  List[Dict]  = []

class CardData(BaseModel):
    id:         str
    title:      str
    insight_type: str
    measure:    str
    dimension:  Optional[str]
    grain:      str
    chart_type: str
    chart_data: List[Dict]
    kpi:        Optional[float]
    delta:      Optional[float]
    period:     Optional[str]
    summary:    str


# ── Upload & cube build ───────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accept CSV, Excel, or Parquet. Profile data, build cube in background.
    Returns immediately with schema; cube builds async.
    """
    content = await file.read()
    fname   = file.filename or "data.csv"

    try:
        if fname.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(content))
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        else:
            # CSV — try multiple encodings
            df = None
            for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(io.BytesIO(content), encoding=enc, index_col=False)
                    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            if df is None:
                raise ValueError("Could not decode file with any supported encoding")

        if df.empty:
            raise ValueError("File contains no data")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File read error: {str(e)}")

    # Profile synchronously (fast)
    profiler    = DataProfiler(df)
    schema_ctx  = profiler.result()
    df_parsed   = profiler.parsed_df()

    _STATE["profiler"]     = profiler
    _STATE["schema_ctx"]   = schema_ctx
    _STATE["df"]           = df_parsed
    _STATE["build_status"] = "building"
    _STATE["build_error"]  = None

    # Build cube in background
    background_tasks.add_task(_build_cube_bg, df_parsed, schema_ctx)

    return {
        "status":    "building",
        "schema":    schema_ctx,
        "file_name": fname,
        "rows":      len(df),
        "cols":      len(df.columns),
    }


def _build_cube_bg(df: pd.DataFrame, schema_ctx: dict):
    """Background task: build the DuckDB cube."""
    try:
        db_path = os.environ.get("DUCKDB_PATH", ":memory:")
        cube    = CubeEngine(db_path=db_path)
        stats   = cube.build(df, schema_ctx)
        _STATE["cube"]         = cube
        _STATE["build_status"] = "ready"
        _STATE["build_stats"]  = stats
    except Exception as e:
        _STATE["build_status"] = "error"
        _STATE["build_error"]  = str(e)


# ── Schema & status ───────────────────────────────────────────────────────────

@app.get("/schema")
def get_schema():
    if not _STATE["schema_ctx"]:
        raise HTTPException(status_code=404, detail="No data loaded")
    return {
        "build_status": _STATE["build_status"],
        "schema":       _STATE["schema_ctx"],
        "cube_stats":   _STATE["cube"].stats() if _STATE["cube"] else None,
    }


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "build_status": _STATE["build_status"],
        "has_data":     _STATE["df"] is not None,
    }


# ── Query ─────────────────────────────────────────────────────────────────────

@app.post("/query")
async def query(req: QueryRequest):
    """
    Main query endpoint.
    1. Parse intent with Claude NLU
    2. Execute against cube
    3. Return card-ready response
    """
    if not _STATE["cube"] or _STATE["build_status"] != "ready":
        raise HTTPException(status_code=503, detail="Cube not ready. Check /health.")

    cube       = _STATE["cube"]
    schema_ctx = _STATE["schema_ctx"]

    # Build conversation history for context
    hist = _STATE["query_history"][-4:] if _STATE["query_history"] else []

    # NLU
    intent = await parse_intent(
        text       = req.text,
        schema_ctx = schema_ctx,
        history    = hist,
        api_key    = os.environ.get("ANTHROPIC_API_KEY"),
    )

    # Log to history
    _STATE["query_history"].append({"role": "user", "content": req.text})
    _STATE["query_history"] = _STATE["query_history"][-20:]

    # Execute against cube
    card = await _resolve_to_card(intent, cube, schema_ctx)

    return card


async def _resolve_to_card(intent: dict, cube: CubeEngine, schema_ctx: dict) -> dict:
    """Map intent → cube query → card payload."""
    itype    = intent["insight_type"]
    measure  = intent["measure"]
    dim      = intent.get("dimension")
    dim2     = intent.get("dimension2")
    grain    = intent.get("grain", "month")
    k        = intent.get("k", 5)
    period   = intent.get("period_key")  # None → latest

    chart_data = []
    kpi        = None
    delta      = None
    period_key = period
    chart_type = "bar"

    if itype == "trend":
        n = {"day": 30, "week": 16, "month": 12, "quarter": 8, "year": 5}.get(grain, 12)
        rows = cube.query_trend(measure, grain, n)
        chart_data = [{"period": r["period"], "value": r["value"]} for r in rows]
        chart_type = "area"
        if rows:
            kpi   = rows[-1]["value"]
            delta = rows[-1].get("delta")
            period_key = rows[-1]["period"]

    elif itype == "total":
        result = cube.query_total(measure, grain, period)
        kpi        = result.get("value")
        delta      = result.get("delta")
        period_key = result.get("period")
        # Also get trend for sparkline
        rows = cube.query_trend(measure, grain, 12)
        chart_data = [{"period": r["period"], "value": r["value"]} for r in rows]
        chart_type = "area"

    elif itype == "topk":
        if not dim:
            # No dimension → fall back to trend
            itype = "trend"
            rows = cube.query_trend(measure, grain, 10)
            chart_data = [{"period": r["period"], "value": r["value"]} for r in rows]
            chart_type = "area"
            if rows:
                kpi   = rows[-1]["value"]
                delta = rows[-1].get("delta")
                period_key = rows[-1]["period"]
        else:
            rows = cube.query_topk(dim, measure, grain, k, period)
            chart_data = [{"label": r["label"], "value": r["value"]} for r in rows]
            chart_type = "bar"
            if rows:
                kpi   = rows[0]["value"]
                delta = rows[0].get("delta")

    elif itype == "cross":
        if dim and dim2:
            rows = cube.query_cross(dim, dim2, measure, grain, period)
            # Reshape cross data into grouped bars: d1 values become groups,
            # d2 values become the label axis — frontend bar chart can render this
            chart_data = [{"label": f"{r['d1']} × {r['d2']}", "value": r["value"]} for r in rows[:20]]
            chart_type = "bar"
            if chart_data:
                kpi = sum(r["value"] for r in chart_data)
        else:
            # Fallback if dims not resolved
            rows = cube.query_trend(measure, grain, 12)
            chart_data = [{"period": r["period"], "value": r["value"]} for r in rows]
            chart_type = "area"

    elif itype == "breakdown":
        if dim:
            rows = cube.query_breakdown(dim, measure, grain, period)
            n_vals = len(rows)
            chart_type = "pie" if n_vals <= 5 else "bar"
            chart_data = [{"label": r["label"], "value": r["value"]} for r in rows]
            if rows:
                kpi   = sum(r["value"] for r in rows)
                delta = rows[0].get("delta")
        else:
            # No dimension → total
            result = cube.query_total(measure, grain, period)
            kpi        = result.get("value")
            delta      = result.get("delta")
            period_key = result.get("period")
            chart_data = []
            chart_type = "kpi"

    # Generate natural language summary
    summary = _generate_summary(itype, measure, dim, grain, chart_data, kpi, delta)

    return {
        "id":           str(uuid.uuid4())[:8],
        "title":        intent.get("title", req_text_fallback(intent)),
        "insight_type": itype,
        "measure":      measure,
        "dimension":    dim,
        "grain":        grain,
        "chart_type":   chart_type,
        "chart_data":   chart_data,
        "kpi":          kpi,
        "delta":        delta,
        "period":       period_key,
        "summary":      summary,
    }


def req_text_fallback(intent: dict) -> str:
    m = intent.get("measure", "value")
    d = intent.get("dimension", "")
    g = intent.get("grain", "month")
    t = intent.get("insight_type", "")
    return f"{t.title()} of {m}{' by '+d if d else ''} ({g})"


def _generate_summary(itype, measure, dimension, grain, chart_data, kpi, delta) -> str:
    if not chart_data and kpi is None:
        return "No data available for this query."

    delta_str = ""
    if delta is not None:
        pct = abs(delta * 100)
        dir_word = "up" if delta > 0 else "down"
        delta_str = f" ({dir_word} {pct:.1f}% vs prior {grain})"

    # Trend: show growth over period + latest value
    if itype == "trend" and chart_data:
        first = chart_data[0]["value"] if chart_data[0].get("value") else 1
        last  = chart_data[-1]["value"] if chart_data[-1].get("value") else 0
        chg   = ((last - first) / first * 100) if first else 0
        word  = "grew" if chg >= 0 else "declined"
        return f"{measure} {word} {abs(chg):.1f}% over the period. Latest: {_fmt(last)}{delta_str}."

    # Total: single number with delta
    if itype == "total" and kpi is not None:
        return f"Total {measure}: {_fmt(kpi)}{delta_str}."

    # Breakdown: who leads, who lags
    if itype == "breakdown" and chart_data:
        top = chart_data[0]
        tot = sum(r["value"] for r in chart_data if r.get("value"))
        pct = (top["value"] / tot * 100) if tot else 0
        tail = (f" {chart_data[-1]['label']} is the lowest at {_fmt(chart_data[-1]['value'])}."
                if len(chart_data) > 1 else "")
        return f"{top['label']} leads with {_fmt(top['value'])} ({pct:.0f}% of total).{tail}"

    # Top-K
    if itype == "topk" and chart_data:
        top = chart_data[0]
        return f"#1 is {top['label']} at {_fmt(top['value'])}{delta_str}."

    # Cross
    if itype == "cross" and chart_data:
        top = chart_data[0]
        return f"Highest intersection: {top['label']} at {_fmt(top['value'])}."

    return f"{measure} insight computed successfully."


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"{v/1_000:.1f}K"
    return f"{v:.2f}"


# ── Suggestions ───────────────────────────────────────────────────────────────

@app.get("/suggest")
def suggest():
    """Return contextual query suggestions based on current schema."""
    if not _STATE["schema_ctx"]:
        return {"suggestions": []}

    sc  = _STATE["schema_ctx"]
    m   = sc.get("measures", [{}])[0].get("col", "revenue") if sc.get("measures") else "revenue"
    dims = [d["col"] for d in sc.get("dims", [])]
    d0  = dims[0] if dims else None
    d1  = dims[1] if len(dims) > 1 else None

    suggestions = [
        f"Total {m} this month",
        f"{m} trend last year",
        f"Top 5 {d0} by {m}" if d0 else f"Overall {m}",
        f"{m} by {d0}" if d0 else f"{m} breakdown",
        f"{m} by {d1} this quarter" if d1 else f"{m} quarterly trend",
        f"Compare {m} by {d0} vs last quarter" if d0 else f"Year over year {m}",
    ]
    return {"suggestions": [s for s in suggestions if s]}


# ── Periods ───────────────────────────────────────────────────────────────────

@app.get("/periods/{grain}")
def get_periods(grain: str):
    if not _STATE["cube"]:
        raise HTTPException(status_code=404, detail="No cube loaded")
    periods = _STATE["cube"].query_periods(grain)
    return {"grain": grain, "periods": periods}


# ── Dashboard CRUD ────────────────────────────────────────────────────────────

@app.post("/dashboard")
def save_dashboard(req: DashboardSaveRequest):
    dash_id = str(uuid.uuid4())[:8]
    _STATE["dashboards"][dash_id] = {
        "id":      dash_id,
        "name":    req.name,
        "layout":  req.layout,
        "cards":   req.cards,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return {"id": dash_id, "url": f"/dashboard/{dash_id}"}


@app.get("/dashboard/{dash_id}")
def load_dashboard(dash_id: str):
    d = _STATE["dashboards"].get(dash_id)
    if not d:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return d


@app.get("/dashboard")
def list_dashboards():
    return {"dashboards": list(_STATE["dashboards"].values())}
