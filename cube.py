"""
AxiLattice Cube Engine v1
─────────────────────────
Architecture:
  - DuckDB for persistence + SQL fallback on high-cardinality dims
  - Per-dimension cardinality cutoff (default 50) — not cross-product cap
  - Time as a first-class grain hierarchy: day→week→month→quarter→year
  - Pre-computed period-over-period deltas baked into every cell
  - Incremental append support (additive measures only: sum, count)
  - Rank within period baked in at build time

Key design choices vs axilattice_pro_v3:
  1. Time stored at grain level (5 separate passes), not as a raw categorical
  2. Cardinality check per-dim before inclusion, not on cross-product
  3. Deltas (MoM, QoQ, YoY) pre-computed as LAG window in DuckDB SQL
  4. Cube persists to .duckdb file — survives process restart
"""

import duckdb
import pandas as pd
import numpy as np
from itertools import combinations
from typing import Dict, List, Optional, Tuple
import json
import os
import time

# ── Constants ────────────────────────────────────────────────────────────────
CARDINALITY_CUTOFF = 50        # per-dimension max distinct values for cube inclusion
MAX_DIM_CROSS      = 2         # max simultaneous dimensions in a cross-cut
TIME_GRAINS        = ["day", "week", "month", "quarter", "year"]
CUBE_TABLE         = "axl_cube"
META_TABLE         = "axl_meta"


def _grain_expr(col: str, grain: str) -> str:
    """
    DuckDB SQL expression to extract period key from a date/timestamp column.
    DuckDB strftime signature: strftime(format, timestamp) — format FIRST.
    """
    if grain == "day":
        return f"strftime('%Y-%m-%d', {col})"
    elif grain == "week":
        return f"strftime('%G-W%V', {col})"          # ISO week
    elif grain == "month":
        return f"strftime('%Y-%m', {col})"
    elif grain == "quarter":
        return f"(strftime('%Y', {col}) || '-Q' || CAST(CEIL(CAST(strftime('%m', {col}) AS INT) / 3.0) AS INT))"
    elif grain == "year":
        return f"strftime('%Y', {col})"
    else:
        raise ValueError(f"Unknown grain: {grain}")


class CubeEngine:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path  = db_path
        self.conn     = duckdb.connect(db_path)
        self.dims: List[Dict]     = []   # [{name, col, cardinality, values}]
        self.measures: List[Dict] = []   # [{name, col}]
        self.time_col: Optional[str] = None
        self.excluded_dims: List[Dict] = []  # high-cardinality dims, queryable via SQL
        self._build_time: float = 0.0
        self._ensure_tables()

    # ── Schema ───────────────────────────────────────────────────────────────

    def _ensure_tables(self):
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {CUBE_TABLE} (
                grain       VARCHAR NOT NULL,
                period_key  VARCHAR NOT NULL,
                dim_combo   VARCHAR NOT NULL,
                dim_json    VARCHAR NOT NULL,
                measure     VARCHAR NOT NULL,
                val_sum     DOUBLE,
                val_count   BIGINT,
                val_min     DOUBLE,
                val_max     DOUBLE,
                val_mean    DOUBLE,
                val_stddev  DOUBLE,
                PRIMARY KEY (grain, period_key, dim_combo, dim_json, measure)
            )
        """)
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {META_TABLE} (
                key   VARCHAR PRIMARY KEY,
                value VARCHAR
            )
        """)

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self, df: pd.DataFrame, profiler_result: dict) -> dict:
        """
        Build the full pre-computed cube from a DataFrame.
        Returns stats dict.
        """
        t0 = time.time()
        self.dims         = profiler_result["dims"]
        self.measures     = profiler_result["measures"]
        self.time_col     = profiler_result.get("time_col")
        self.excluded_dims = profiler_result.get("excluded_dims", [])

        # Drop any existing cube rows (fresh build)
        self.conn.execute(f"DELETE FROM {CUBE_TABLE}")

        # Register source data in DuckDB
        self.conn.register("_src", df)

        # Parse time column to DATE if present
        if self.time_col:
            self.conn.execute(f"""
                CREATE OR REPLACE TABLE _src_parsed AS
                SELECT *, TRY_CAST("{self.time_col}" AS DATE) AS _date_parsed
                FROM _src
            """)
        else:
            self.conn.execute("CREATE OR REPLACE TABLE _src_parsed AS SELECT * FROM _src")

        cube_dims = [d for d in self.dims if d["cardinality"] <= CARDINALITY_CUTOFF]
        dim_names = [d["col"] for d in cube_dims]

        rows_inserted = 0

        # ── Pass 1: aggregate for each time grain ──────────────────────────
        for grain in TIME_GRAINS:
            if self.time_col:
                period_expr = _grain_expr("_date_parsed", grain)
            else:
                period_expr = "'__all__'"

            # Totals (no dim grouping)
            rows_inserted += self._agg_and_insert(
                grain=grain,
                period_expr=period_expr,
                group_cols=[],
                dim_combo="__total__",
            )

            # Single-dimension cuboids
            for dcol in dim_names:
                rows_inserted += self._agg_and_insert(
                    grain=grain,
                    period_expr=period_expr,
                    group_cols=[dcol],
                    dim_combo=dcol,
                )

            # Cross-dimensional cuboids (up to MAX_DIM_CROSS)
            for r in range(2, MAX_DIM_CROSS + 1):
                for combo in combinations(dim_names, r):
                    combo_key = "|".join(sorted(combo))
                    rows_inserted += self._agg_and_insert(
                        grain=grain,
                        period_expr=period_expr,
                        group_cols=list(combo),
                        dim_combo=combo_key,
                    )

        # ── Pass 2: compute period-over-period deltas ──────────────────────
        self._compute_deltas()

        self._build_time = time.time() - t0

        # Save meta
        meta = {
            "dims":          json.dumps([d["col"] for d in self.dims]),
            "measures":      json.dumps([m["col"] for m in self.measures]),
            "time_col":      self.time_col or "",
            "excluded_dims": json.dumps([d["col"] for d in self.excluded_dims]),
            "rows_inserted": str(rows_inserted),
            "build_seconds": f"{self._build_time:.2f}",
        }
        for k, v in meta.items():
            self.conn.execute(f"""
                INSERT INTO {META_TABLE} VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, [k, v])

        return {
            "cube_cells":    rows_inserted,
            "dims_cubed":    len(cube_dims),
            "dims_excluded": len(self.excluded_dims),
            "grains":        len(TIME_GRAINS),
            "build_seconds": round(self._build_time, 2),
        }

    def _agg_and_insert(self, grain: str, period_expr: str,
                         group_cols: List[str], dim_combo: str) -> int:
        """Run one group-by aggregation and insert into cube table."""
        measure_aggs = ", ".join([
            f"SUM(\"{m['col']}\") AS {m['col']}_sum, "
            f"COUNT(\"{m['col']}\") AS {m['col']}_cnt, "
            f"MIN(\"{m['col']}\") AS {m['col']}_min, "
            f"MAX(\"{m['col']}\") AS {m['col']}_max, "
            f"AVG(\"{m['col']}\") AS {m['col']}_avg, "
            f"STDDEV(\"{m['col']}\") AS {m['col']}_std"
            for m in self.measures
        ])

        if group_cols:
            quoted = [f'"{c}"' for c in group_cols]
            group_expr = ", ".join(quoted) + ", "
            group_by   = "GROUP BY " + ", ".join(quoted) + ", period_key"
            # Use json_object() — safe against quotes/apostrophes/commas in values
            dim_json_expr = "json_object(" + ", ".join([
                f"'{c}', CAST(\"{c}\" AS VARCHAR)"
                for c in group_cols
            ]) + ")"
        else:
            group_expr    = ""
            group_by      = "GROUP BY period_key"
            dim_json_expr = "'{\"__total__\": true}'"

        sql = f"""
            SELECT
                '{grain}'   AS grain,
                {period_expr} AS period_key,
                '{dim_combo}' AS dim_combo,
                {dim_json_expr} AS dim_json,
                {group_expr}
                {measure_aggs}
            FROM _src_parsed
            WHERE {period_expr} IS NOT NULL
            {group_by}
        """

        try:
            result_df = self.conn.execute(sql).df()
        except Exception as e:
            return 0

        if result_df.empty:
            return 0

        rows = 0
        for _, row in result_df.iterrows():
            period_key = str(row["period_key"])
            dim_json   = str(row["dim_json"])
            for m in self.measures:
                col = m["col"]
                try:
                    self.conn.execute(f"""
                        INSERT OR REPLACE INTO {CUBE_TABLE}
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        grain, period_key, dim_combo, dim_json, col,
                        float(row.get(f"{col}_sum", 0) or 0),
                        int(row.get(f"{col}_cnt", 0) or 0),
                        float(row.get(f"{col}_min", 0) or 0),
                        float(row.get(f"{col}_max", 0) or 0),
                        float(row.get(f"{col}_avg", 0) or 0),
                        float(row.get(f"{col}_std", 0) or 0),
                    ])
                    rows += 1
                except Exception:
                    pass
        return rows

    def _compute_deltas(self):
        """
        Compute period-over-period deltas by adding a deltas table.
        This gives MoM, QoQ, YoY without re-querying raw data.
        """
        self.conn.execute("""
            CREATE OR REPLACE TABLE axl_deltas AS
            SELECT
                grain, period_key, dim_combo, dim_json, measure,
                val_sum,
                LAG(val_sum) OVER w AS prior_sum,
                CASE
                    WHEN LAG(val_sum) OVER w IS NULL OR LAG(val_sum) OVER w = 0 THEN NULL
                    ELSE (val_sum - LAG(val_sum) OVER w) / LAG(val_sum) OVER w
                END AS delta_pct,
                RANK() OVER (
                    PARTITION BY grain, period_key, dim_combo, measure
                    ORDER BY val_sum DESC
                ) AS rank_in_period
            FROM axl_cube
            WINDOW w AS (
                PARTITION BY grain, dim_combo, dim_json, measure
                ORDER BY period_key
            )
        """)

    # ── Query API ────────────────────────────────────────────────────────────

    def query_breakdown(self, dimension: str, measure: str,
                        grain: str, period_key: Optional[str] = None) -> List[dict]:
        """Return measure values broken down by one dimension for a period."""
        pk = period_key or self._latest_period(grain)
        if not pk:
            return []
        rows = self.conn.execute(f"""
            SELECT
                JSON_EXTRACT_STRING(d.dim_json, '$.{dimension}') AS label,
                d.val_sum AS value,
                dl.delta_pct,
                dl.rank_in_period
            FROM {CUBE_TABLE} d
            LEFT JOIN axl_deltas dl USING (grain, period_key, dim_combo, dim_json, measure)
            WHERE d.grain     = ?
              AND d.period_key = ?
              AND d.dim_combo = ?
              AND d.measure   = ?
            ORDER BY d.val_sum DESC
        """, [grain, pk, dimension, measure]).fetchall()

        return [{"label": r[0], "value": r[1], "delta": r[2], "rank": r[3]}
                for r in rows if r[0] is not None]

    def query_trend(self, measure: str, grain: str,
                    n_periods: int = 12) -> List[dict]:
        """Return total measure over last n periods."""
        rows = self.conn.execute(f"""
            SELECT d.period_key, d.val_sum AS value, dl.delta_pct
            FROM {CUBE_TABLE} d
            LEFT JOIN axl_deltas dl USING (grain, period_key, dim_combo, dim_json, measure)
            WHERE d.grain     = ?
              AND d.dim_combo = '__total__'
              AND d.measure   = ?
            ORDER BY d.period_key DESC
            LIMIT ?
        """, [grain, measure, n_periods]).fetchall()

        return [{"period": r[0], "value": r[1], "delta": r[2]}
                for r in reversed(rows)]

    def query_topk(self, dimension: str, measure: str,
                   grain: str, k: int = 5,
                   period_key: Optional[str] = None) -> List[dict]:
        """Top-K values of a dimension for a measure in a period."""
        pk = period_key or self._latest_period(grain)
        if not pk:
            return []
        rows = self.conn.execute(f"""
            SELECT
                JSON_EXTRACT_STRING(dim_json, '$.{dimension}') AS label,
                val_sum AS value
            FROM {CUBE_TABLE}
            WHERE grain     = ?
              AND period_key = ?
              AND dim_combo = ?
              AND measure   = ?
            ORDER BY val_sum DESC
            LIMIT ?
        """, [grain, pk, dimension, measure, k]).fetchall()
        return [{"label": r[0], "value": r[1]} for r in rows if r[0]]

    def query_total(self, measure: str, grain: str,
                    period_key: Optional[str] = None) -> dict:
        """Grand total for a measure in a period."""
        pk = period_key or self._latest_period(grain)
        if not pk:
            return {}
        row = self.conn.execute(f"""
            SELECT d.val_sum, dl.delta_pct
            FROM {CUBE_TABLE} d
            LEFT JOIN axl_deltas dl USING (grain, period_key, dim_combo, dim_json, measure)
            WHERE d.grain     = ?
              AND d.period_key = ?
              AND d.dim_combo = '__total__'
              AND d.measure   = ?
        """, [grain, pk, measure]).fetchone()
        if not row:
            return {}
        return {"value": row[0], "delta": row[1], "period": pk}

    def query_cross(self, dim1: str, dim2: str, measure: str,
                    grain: str, period_key: Optional[str] = None) -> List[dict]:
        """Cross-dimensional breakdown (heatmap data)."""
        pk = period_key or self._latest_period(grain)
        if not pk:
            return []
        combo_key = "|".join(sorted([dim1, dim2]))
        rows = self.conn.execute(f"""
            SELECT
                JSON_EXTRACT_STRING(dim_json, '$.{dim1}') AS d1,
                JSON_EXTRACT_STRING(dim_json, '$.{dim2}') AS d2,
                val_sum AS value
            FROM {CUBE_TABLE}
            WHERE grain     = ?
              AND period_key = ?
              AND dim_combo = ?
              AND measure   = ?
            ORDER BY val_sum DESC
        """, [grain, pk, combo_key, measure]).fetchall()
        return [{"d1": r[0], "d2": r[1], "value": r[2]}
                for r in rows if r[0] and r[1]]

    def query_periods(self, grain: str) -> List[str]:
        """All available period keys for a grain."""
        rows = self.conn.execute(f"""
            SELECT DISTINCT period_key FROM {CUBE_TABLE}
            WHERE grain = ? AND dim_combo = '__total__'
            ORDER BY period_key
        """, [grain]).fetchall()
        return [r[0] for r in rows]

    def _latest_period(self, grain: str) -> Optional[str]:
        row = self.conn.execute(f"""
            SELECT MAX(period_key) FROM {CUBE_TABLE}
            WHERE grain = ? AND dim_combo = '__total__'
        """, [grain]).fetchone()
        return row[0] if row else None

    def available_dims(self) -> List[str]:
        return [d["col"] for d in self.dims if d["cardinality"] <= CARDINALITY_CUTOFF]

    def available_measures(self) -> List[str]:
        return [m["col"] for m in self.measures]

    def stats(self) -> dict:
        row = self.conn.execute(f"SELECT COUNT(*) FROM {CUBE_TABLE}").fetchone()
        return {
            "cube_cells":    row[0] if row else 0,
            "dims":          len(self.dims),
            "measures":      len(self.measures),
            "time_col":      self.time_col,
            "excluded_dims": [d["col"] for d in self.excluded_dims],
            "build_seconds": round(self._build_time, 2),
        }

    # ── Incremental append ───────────────────────────────────────────────────

    def append(self, df_new: pd.DataFrame):
        """
        Incrementally update the cube with new rows.
        Only works for additive measures (sum, count).
        Recomputes deltas after append.
        """
        self.conn.register("_new", df_new)
        if self.time_col:
            self.conn.execute(f"""
                CREATE OR REPLACE TABLE _src_parsed AS
                SELECT *, TRY_CAST("{self.time_col}" AS DATE) AS _date_parsed FROM _new
            """)
        else:
            self.conn.execute("CREATE OR REPLACE TABLE _src_parsed AS SELECT * FROM _new")

        cube_dims = [d for d in self.dims if d["cardinality"] <= CARDINALITY_CUTOFF]
        dim_names = [d["col"] for d in cube_dims]

        for grain in TIME_GRAINS:
            period_expr = _grain_expr("_date_parsed", grain) if self.time_col else "'__all__'"
            self._agg_and_insert(grain, period_expr, [], "__total__")
            for dcol in dim_names:
                self._agg_and_insert(grain, period_expr, [dcol], dcol)

        self._compute_deltas()
