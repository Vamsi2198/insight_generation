"""
AxiLattice Data Profiler
─────────────────────────
Auto-discovers:
  - Temporal columns (datetime detection via format probing)
  - Numeric measures (with summary stats)
  - Categorical dimensions (low-cardinality: ≤ CARDINALITY_CUTOFF)
  - High-cardinality dimensions (excluded from cube, available via SQL fallback)
  - Identifier columns (high uniqueness ratio, excluded entirely)
  - Text columns (long strings, excluded)

Key fix vs v3: Cardinality check is PER DIMENSION, not on cross-product.
Returns a clean profiler_result dict that CubeEngine.build() consumes.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

CARDINALITY_CUTOFF = 50       # ≤ this → cube-worthy dimension
ID_RATIO_THRESHOLD = 0.85     # uniqueness ratio above this → identifier
TEXT_AVG_LEN_THRESHOLD = 60   # avg string length above this → text column
MIN_NUMERIC_UNIQUE = 10       # fewer than this unique values in a numeric → treat as categorical code


class ColType(Enum):
    TEMPORAL    = "temporal"
    MEASURE     = "measure"
    DIMENSION   = "dimension"      # low-cardinality categorical
    DIM_HICARDINAL = "dim_high_card"  # categorical but too many values for cube
    IDENTIFIER  = "identifier"
    TEXT        = "text"
    BOOLEAN     = "boolean"
    UNKNOWN     = "unknown"


@dataclass
class ColProfile:
    name:       str
    dtype:      str
    col_type:   ColType
    cardinality: int
    null_pct:   float
    sample_values: List  = field(default_factory=list)
    stats:      Dict     = field(default_factory=dict)
    warnings:   List[str] = field(default_factory=list)


class DataProfiler:

    DATETIME_FORMATS = [
        "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S", "%m/%d/%Y", "%b %Y", "%B %Y",
        "%Y-%m-%dT%H:%M:%S", "%d %b %Y", "%d %B %Y",
    ]

    def __init__(self, df: pd.DataFrame, cardinality_cutoff: int = CARDINALITY_CUTOFF):
        self.df = df.copy()
        self.cutoff = cardinality_cutoff
        self.profiles: Dict[str, ColProfile] = {}
        self.time_col:     Optional[str] = None
        self.measures:     List[Dict]    = []
        self.dims:         List[Dict]    = []   # low-card categoricals
        self.excluded_dims: List[Dict]   = []   # high-card categoricals
        self.id_cols:      List[str]     = []
        self._run()

    # ── Core profiling pass ───────────────────────────────────────────────────

    def _run(self):
        for col in self.df.columns:
            profile = self._profile_column(col)
            self.profiles[col] = profile

            t = profile.col_type
            if t == ColType.TEMPORAL and self.time_col is None:
                self.time_col = col
                # Ensure column is parsed as datetime in the dataframe
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

            elif t == ColType.MEASURE:
                self.measures.append({
                    "col":  col,
                    "name": col,
                    "stats": profile.stats,
                })

            elif t in (ColType.DIMENSION, ColType.BOOLEAN):
                self.dims.append({
                    "col":         col,
                    "name":        col,
                    "cardinality": profile.cardinality,
                    "values":      profile.sample_values,
                })

            elif t == ColType.DIM_HICARDINAL:
                self.excluded_dims.append({
                    "col":         col,
                    "name":        col,
                    "cardinality": profile.cardinality,
                    "reason":      f"cardinality={profile.cardinality} > cutoff={self.cutoff}",
                })

            elif t == ColType.IDENTIFIER:
                self.id_cols.append(col)

    def _profile_column(self, col: str) -> ColProfile:
        s         = self.df[col]
        dtype_str = str(s.dtype)
        n         = len(s)
        null_pct  = float(s.isnull().mean() * 100)
        uniq      = int(s.nunique(dropna=True))
        ratio     = uniq / n if n > 0 else 0

        col_type, stats, warnings = self._infer_type(s, dtype_str, uniq, ratio, col)

        sample = list(s.dropna().unique()[:8])
        try:
            sample = [str(v) for v in sample]
        except Exception:
            sample = []

        return ColProfile(
            name=col, dtype=dtype_str, col_type=col_type,
            cardinality=uniq, null_pct=null_pct,
            sample_values=sample, stats=stats, warnings=warnings,
        )

    def _infer_type(self, s, dtype_str: str, uniq: int,
                    ratio: float, col_name: str) -> Tuple[ColType, Dict, List[str]]:
        stats:    Dict      = {}
        warnings: List[str] = []

        # ── 1. Datetime detection ─────────────────────────────────────────
        if pd.api.types.is_datetime64_any_dtype(s):
            return ColType.TEMPORAL, {"parsed": True}, []

        if dtype_str == "object":
            if self._looks_like_datetime(s):
                return ColType.TEMPORAL, {"parsed": False, "needs_parse": True}, []

        # ── 2. Year column heuristic ──────────────────────────────────────
        if re.search(r"\byear\b", col_name, re.I) and dtype_str in ("int64", "float64"):
            s_clean = s.dropna()
            if len(s_clean) and (1900 < float(s_clean.min()) < 2100):
                return ColType.TEMPORAL, {"part": "year"}, ["Year-only column — no sub-year granularity"]

        # ── 3. Boolean ───────────────────────────────────────────────────
        if uniq == 2:
            return ColType.BOOLEAN, {}, []

        # ── 4. Identifier detection (very high uniqueness) ────────────────
        if ratio >= ID_RATIO_THRESHOLD and dtype_str in ("int64", "object", "float64"):
            if uniq > self.cutoff:
                return ColType.IDENTIFIER, {"ratio": round(ratio, 3)}, []

        # ── 5. Numeric ───────────────────────────────────────────────────
        if dtype_str in ("int64", "float64", "int32", "float32"):
            if uniq <= MIN_NUMERIC_UNIQUE and ratio < 0.05:
                # Looks like a coded categorical (e.g., status=1/2/3)
                return ColType.DIMENSION, {"coded": True}, ["Numeric codes treated as category"]
            mu  = float(s.mean()) if s.notna().any() else 0.0
            std = float(s.std())  if s.notna().any() else 0.0
            if std == 0:
                warnings.append("Zero variance — might be a constant")
                return ColType.IDENTIFIER, {"constant": True}, warnings
            stats = {
                "mean":   round(mu, 4),
                "std":    round(std, 4),
                "min":    float(s.min()),
                "max":    float(s.max()),
                "median": float(s.median()),
            }
            return ColType.MEASURE, stats, warnings

        # ── 6. Object / string columns ────────────────────────────────────
        if dtype_str == "object":
            avg_len = float(s.dropna().astype(str).str.len().mean()) if s.notna().any() else 0
            if avg_len > TEXT_AVG_LEN_THRESHOLD:
                return ColType.TEXT, {"avg_len": round(avg_len, 1)}, []
            if uniq <= self.cutoff:
                return ColType.DIMENSION, {"cats": uniq}, []
            # Over cutoff — still a dimension, just excluded from cube
            return ColType.DIM_HICARDINAL, {"cats": uniq}, []

        return ColType.UNKNOWN, {}, []

    def _looks_like_datetime(self, s: pd.Series) -> bool:
        sample = s.dropna().head(50)
        if len(sample) == 0:
            return False
        for fmt in self.DATETIME_FORMATS:
            try:
                parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    return True
            except Exception:
                pass
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.8:
                return True
        except Exception:
            pass
        return False

    # ── Public API ────────────────────────────────────────────────────────────

    def result(self) -> dict:
        """Return the profiler result dict that CubeEngine.build() expects."""
        return {
            "dims":          self.dims,
            "measures":      self.measures,
            "excluded_dims": self.excluded_dims,
            "time_col":      self.time_col,
            "id_cols":       self.id_cols,
            "row_count":     len(self.df),
            "col_count":     len(self.df.columns),
            "schema":        {
                col: {
                    "type":        p.col_type.value,
                    "cardinality": p.cardinality,
                    "null_pct":    round(p.null_pct, 1),
                    "sample":      p.sample_values[:5],
                    "stats":       p.stats,
                    "warnings":    p.warnings,
                }
                for col, p in self.profiles.items()
            },
        }

    def parsed_df(self) -> pd.DataFrame:
        """Return df with time column parsed as datetime."""
        return self.df
