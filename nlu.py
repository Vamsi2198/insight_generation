"""
AxiLattice NLU — Claude-powered intent parser
──────────────────────────────────────────────
Converts free-form natural language analytical queries into
structured intent dicts that CubeEngine can execute directly.

Key improvement over regex classifiers:
  - Handles paraphrases: "how's revenue doing" → trend
  - Handles column name variance: "top line" → revenue (with schema context)
  - Handles compound queries: "revenue by region vs last quarter"
  - Maintains 5-message conversation context for follow-ups
"""

import os
import json
import re
import httpx
from typing import Optional, List, Dict

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL             = "claude-sonnet-4-20250514"
MAX_TOKENS        = 400


def _build_system_prompt(schema_ctx: dict) -> str:
    dims     = [d["col"] for d in schema_ctx.get("dims", [])]
    measures = [m["col"] for m in schema_ctx.get("measures", [])]
    time_col = schema_ctx.get("time_col", "")
    excl     = [d["col"] for d in schema_ctx.get("excluded_dims", [])]

    return f"""You are an analytical query intent parser for a pre-computed BI cube engine.

SCHEMA:
- Measures (numeric): {measures}
- Cube dimensions (low-cardinality): {dims}
- High-cardinality dimensions (SQL-only): {excl}
- Time column: {time_col or "none"}

Parse the user's query into EXACTLY this JSON structure. No other text.

{{
  "insight_type": "breakdown" | "trend" | "topk" | "total" | "cross" | "distribution" | "correlation" | "anomaly",
  "measure": "<one of the measure column names>",
  "dimension": "<one of the cube dimension column names or null>",
  "dimension2": "<second dimension for cross-type or null>",
  "grain": "day" | "week" | "month" | "quarter" | "year",
  "k": <integer for topk, default 5>,
  "period_key": "<specific period string like '2024-01' or null for latest>",
  "title": "<concise human-readable title for this insight card, max 8 words>"
}}

RULES:
1. Default measure: {measures[0] if measures else "revenue"}
2. Default grain: month
3. Default k: 5
4. For "trend" and "total": dimension = null
5. For "cross": both dimension and dimension2 must be set
6. Grain mapping: "daily"→day, "weekly"→week, "monthly/this month/last month"→month,
   "quarterly/this quarter/last quarter/QoQ"→quarter, "yearly/annual/YoY"→year
7. If user says "top N", set insight_type=topk and k=N
8. If user says "breakdown/split/by X", set insight_type=breakdown
9. If user says "trend/over time/trajectory/direction", set insight_type=trend
10. If user says "total/overall/grand total", set insight_type=total
11. Map informal names to column names: "top line"→first measure, "GMV"→revenue if revenue exists
12. Dimension must be one from the cube dimensions list above, or null
"""


async def parse_intent(
    text: str,
    schema_ctx: dict,
    history: Optional[List[Dict]] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    Parse a natural language query into a structured intent dict.
    
    Args:
        text:       Raw user query
        schema_ctx: Profiler result dict (has dims, measures, time_col)
        history:    Last N message pairs for context (list of {role, content})
        api_key:    Anthropic API key (falls back to env var)
    
    Returns:
        Intent dict with insight_type, measure, dimension, grain, etc.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return _fallback_parse(text, schema_ctx)

    system = _build_system_prompt(schema_ctx)
    messages = []

    # Include up to 4 prior turns for context
    if history:
        for h in history[-4:]:
            messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": text})

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key":         key,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":      MODEL,
                    "max_tokens": MAX_TOKENS,
                    "system":     system,
                    "messages":   messages,
                },
            )
            data = resp.json()
    except Exception as e:
        return _fallback_parse(text, schema_ctx)

    raw = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            raw += block["text"]

    raw = raw.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        intent = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                intent = json.loads(m.group())
            except Exception:
                intent = {}
        else:
            intent = {}

    return _validate_and_fill(intent, schema_ctx, text)


def _validate_and_fill(intent: dict, schema_ctx: dict, raw_text: str) -> dict:
    """Ensure all required fields are present and valid."""
    dims     = [d["col"] for d in schema_ctx.get("dims", [])]
    measures = [m["col"] for m in schema_ctx.get("measures", [])]

    # Defaults
    defaults = {
        "insight_type": "breakdown" if dims else "total",
        "measure":      measures[0] if measures else "value",
        "dimension":    dims[0] if dims else None,
        "dimension2":   None,
        "grain":        "month",
        "k":            5,
        "period_key":   None,
        "title":        raw_text[:60],
    }

    for k, v in defaults.items():
        if k not in intent or intent[k] is None:
            intent[k] = v

    # Validate measure exists
    if intent["measure"] not in measures and measures:
        intent["measure"] = measures[0]

    # Validate dimension exists in cube dims
    if intent["dimension"] and intent["dimension"] not in dims:
        intent["dimension"] = dims[0] if dims else None

    # Validate grain
    if intent["grain"] not in ("day", "week", "month", "quarter", "year"):
        intent["grain"] = "month"

    # Ensure topk has dimension
    if intent["insight_type"] == "topk" and not intent["dimension"] and dims:
        intent["dimension"] = dims[0]

    # Ensure cross has two dimensions
    if intent["insight_type"] == "cross":
        if not intent["dimension"] and dims:
            intent["dimension"] = dims[0]
        if not intent["dimension2"] and len(dims) > 1:
            intent["dimension2"] = dims[1]
        if intent["dimension"] == intent["dimension2"]:
            intent["dimension2"] = dims[1] if len(dims) > 1 else None

    return intent


def _fallback_parse(text: str, schema_ctx: dict) -> dict:
    """
    Regex-based fallback when Claude API is unavailable.
    Better than nothing, worse than Claude.
    """
    dims     = [d["col"] for d in schema_ctx.get("dims", [])]
    measures = [m["col"] for m in schema_ctx.get("measures", [])]
    ql       = text.lower()

    # Insight type
    if any(w in ql for w in ["trend", "over time", "trajectory", "direction", "history"]):
        itype = "trend"
    elif any(w in ql for w in ["top", "best", "worst", "rank", "highest", "lowest"]):
        itype = "topk"
    elif any(w in ql for w in ["total", "overall", "grand", "aggregate", "sum"]):
        itype = "total"
    elif any(w in ql for w in ["vs", "cross", "heatmap", "matrix"]):
        itype = "cross"
    else:
        itype = "breakdown"

    # Measure
    measure = measures[0] if measures else "value"
    for m in measures:
        if m.lower().replace("_", " ") in ql or m.lower() in ql:
            measure = m
            break

    # Dimension
    dimension = dims[0] if dims else None
    for d in dims:
        if d.lower().replace("_", " ") in ql or d.lower() in ql:
            dimension = d
            break

    # Grain
    grain = "month"
    grain_map = {
        "day": ["daily", "day", "yesterday", "today"],
        "week": ["week", "weekly"],
        "month": ["month", "monthly"],
        "quarter": ["quarter", "quarterly", "qoq"],
        "year": ["year", "yearly", "annual", "yoy"],
    }
    for g, words in grain_map.items():
        if any(w in ql for w in words):
            grain = g
            break

    # k for topk
    k = 5
    m = re.search(r"\btop\s+(\d+)\b", ql)
    if m:
        k = int(m.group(1))

    return {
        "insight_type": itype,
        "measure":      measure,
        "dimension":    dimension,
        "dimension2":   dims[1] if len(dims) > 1 else None,
        "grain":        grain,
        "k":            k,
        "period_key":   None,
        "title":        text[:60],
    }
