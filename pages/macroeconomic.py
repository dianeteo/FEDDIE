import os
import dash
import sqlite3
import requests
import math

import pandas as pd
import numpy as np
import plotly.graph_objs as go

from datetime import datetime
from typing import Iterable, Tuple
from dash import html, dcc, Input, Output, callback
import dash_mantine_components as dmc
from dash_iconify import DashIconify

dash.register_page(__name__, path="/macroeconomic", name="Macroeconomic", order=2)

FRED_API_KEY = os.environ.get("FRED_API_KEY")

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

layout = html.Div(
    id="macro-container",
    children=[
        html.Div(
            className="macro-hero",
            children=[
                dmc.Title("Macroeconomic Dashboard", order=2),
                html.Div(
                    className="macro-actions",
                    children=[
                        dmc.Button(
                            "Get latest data",
                            id="btn-get-latest",
                            variant="light",
                            color="#062840",
                            leftSection=DashIconify(icon="mdi:download"),
                        ),
                        dmc.Badge("Idle", id="macro-status-badge", color="gray", variant="filled"),
                    ],
                ),
            ],
        ),
        html.Div(
            className="status-row",
            children=[
                dmc.Text("Click the button to fetch/refresh FRED data.", id="macro-status-text"),
            ],
        ),

        # KPI ROW
        html.Div(
            id="macro-kpis",
            children=[
                # Unemployment
                dmc.Paper(
                    className="kpi-card",
                    withBorder=True,
                    shadow="xs",
                    children=[
                        dmc.Text("Unemployment", className="kpi-title"),
                        dmc.Text(id="kpi-unemp", className="kpi-value"),
                        dmc.Text(id="kpi-unemp-date", className="kpi-date"),
                    ],
                ),
                # GDP
                dmc.Paper(
                    className="kpi-card",
                    withBorder=True,
                    shadow="xs",
                    children=[
                        dmc.Text("GDP", className="kpi-title"),
                        dmc.Text(id="kpi-gdp", className="kpi-value"),
                        dmc.Text(id="kpi-gdp-date", className="kpi-date"),
                    ],
                ),
                # Fed Funds (Interest)
                dmc.Paper(
                    className="kpi-card",
                    withBorder=True,
                    shadow="xs",
                    children=[
                        dmc.Text("Interest (Fed Funds)", className="kpi-title"),
                        dmc.Text(id="kpi-rate", className="kpi-value"),
                        dmc.Text(id="kpi-rate-date", className="kpi-date"),
                    ],
                ),
                # Inflation (CPI level for now)
                dmc.Paper(
                    className="kpi-card",
                    withBorder=True,
                    shadow="xs",
                    children=[
                        dmc.Text("Inflation (CPI index)", className="kpi-title"),
                        dmc.Text(id="kpi-cpi", className="kpi-value"),
                        dmc.Text(id="kpi-cpi-date", className="kpi-date"),
                    ],
                ),
                # Housing (Starts)
                dmc.Paper(
                    className="kpi-card",
                    withBorder=True,
                    shadow="xs",
                    children=[
                        dmc.Text("Housing Starts", className="kpi-title"),
                        dmc.Text(id="kpi-housing", className="kpi-value"),
                        dmc.Text(id="kpi-housing-date", className="kpi-date"),
                    ],
                ),
                # S&P 500
                dmc.Paper(
                    className="kpi-card",
                    withBorder=True,
                    shadow="xs",
                    children=[
                        dmc.Text("S&P 500", className="kpi-title"),
                        dmc.Text(id="kpi-sp500", className="kpi-value"),
                        dmc.Text(id="kpi-sp500-date", className="kpi-date"),
                    ],
                ),
            ],
        ),

        dcc.Store(id="macro-run-id"),
        dcc.Store(id="macro-last-updated"),
    ],
)

# ---- DISPLAY ROW (controls + chart) -----------------------------------------
display = html.Div(
    id="macro-display",
    children=[
        # controls
        html.Div(
            id="display-controls",
            children=[
                dmc.Select(
                    id="disp-indicator",
                    label="Indicator",
                    value="cpi_monthly",
                    data=[
                        {"value": "unemployment_monthly", "label": "Unemployment (UNRATE)"},
                        {"value": "gdp_quarterly",        "label": "GDP (Real, Ch.2017 $)"},
                        {"value": "fedfunds_daily",       "label": "Fed Funds (DFF)"},
                        {"value": "cpi_monthly",          "label": "Inflation (CPIAUCSL)"},
                        {"value": "housing_monthly",      "label": "Housing Starts (HOUST)"},
                        {"value": "sp500_daily",          "label": "S&P 500 (SP500)"},
                    ],
                    style={"minWidth": 260},
                ),
                dmc.SegmentedControl(
                    id="disp-range",
                    value="5Y",
                    data=[{"value": v, "label": v} for v in ["1Y", "5Y", "10Y", "MAX"]],
                ),
                dmc.Select(
                    id="disp-frequency",
                    label="Frequency",
                    value="native",
                    data=[
                        {"value": "native", "label": "Native"},
                        {"value": "d", "label": "Daily"},
                        {"value": "m", "label": "Monthly"},
                        {"value": "q", "label": "Quarterly"},
                    ],
                    style={"minWidth": 160},
                ),
                dmc.Select(
                    id="disp-units",
                    label="Units",
                    value="level",
                    data=[
                        {"value": "level", "label": "Level"},
                        {"value": "yoy",   "label": "YoY %"},
                        {"value": "mom",   "label": "MoM %"},
                        {"value": "ann_mom", "label": "MoM % (annualized)"},
                    ],
                    style={"minWidth": 200},
                )
            ],
            className="display-controls-row",
        ),
        html.Div(
            id="display-badges",
            children=[
                dmc.Badge(id="disp-varname", color="gray", variant="light"),
                dmc.Badge(id="disp-source", color="gray", variant="light"),
            ],
            className="display-badges-row",
        ),
        # chart
        dcc.Graph(id="disp-graph", config={"displayModeBar": True}),
    ],
)

# add this just after your KPI row in `layout.children`
layout.children.append(display)

# --- Map each series to a table name and the FRED series id ---
SERIES_MAP = {
    "gdp_quarterly":      {"table": "gdp_quarterly",      "fred_id": "GDP"},       # quarterly
    "unemployment_monthly":{"table": "unemployment_monthly","fred_id": "UNRATE"},   # monthly
    "inflation_monthly":  {"table": "cpi_monthly",         "fred_id": "CPIAUCSL"}, # monthly CPI, SA
    "housing_monthly":    {"table": "housing_monthly",     "fred_id": "HOUST"},    # monthly housing starts
    "sp500_daily":        {"table": "sp500_daily",         "fred_id": "SP500"},    # daily close
    "fedfunds_daily":     {"table": "fedfunds_daily",      "fred_id": "DFF"},      # daily eff fed funds
}

# --- DDL per table (simple: date PK, value REAL) ---

CREATE_SQL = {
    "gdp_quarterly":
        "CREATE TABLE IF NOT EXISTS gdp_quarterly (date TEXT PRIMARY KEY, value REAL)",
    "unemployment_monthly":
        "CREATE TABLE IF NOT EXISTS unemployment_monthly (date TEXT PRIMARY KEY, value REAL)",
    "cpi_monthly":
        "CREATE TABLE IF NOT EXISTS cpi_monthly (date TEXT PRIMARY KEY, value REAL)",
    "housing_monthly":
        "CREATE TABLE IF NOT EXISTS housing_monthly (date TEXT PRIMARY KEY, value REAL)",
    "sp500_daily":
        "CREATE TABLE IF NOT EXISTS sp500_daily (date TEXT PRIMARY KEY, value REAL)",
    "fedfunds_daily":
        "CREATE TABLE IF NOT EXISTS fedfunds_daily (date TEXT PRIMARY KEY, value REAL)",
}

def fetch_fred_series(series_id: str, start_date: str = None, end_date: str = None, frequency: str = None):
    """
    Fetch observations from FRED for a given series_id.
    
    Args:
        series_id (str): FRED series ID (e.g., 'UNRATE').
        start_date (str): Optional start date, 'YYYY-MM-DD'.
        end_date (str): Optional end date, 'YYYY-MM-DD'.
        frequency (str): Optional override frequency ('d','m','q','a').
        
    Returns:
        list of dicts with 'date' and 'value'.
    """
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY is not set")

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }
    if start_date:
        params["observation_start"] = start_date
    if end_date:
        params["observation_end"] = end_date
    if frequency:
        params["frequency"] = frequency

    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("observations", [])
    
    return [
        {"date": obs["date"], "value": None if obs["value"] in (".", "") else float(obs["value"])}
        for obs in data
    ]
    
def ensure_schema(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        for ddl in CREATE_SQL.values():
            cur.execute(ddl)
        con.commit()
    finally:
        con.close()

def insert_rows_ignore(db_path: str, table: str, rows: Iterable[Tuple[str, float]]) -> int:
    """
    Insert rows with 'INSERT OR IGNORE' (idempotent).
    Returns the number of rows attempted (not necessarily inserted if duplicates).
    """
    rows = [(d, float(v)) for (d, v) in rows if v is not None]
    if not rows:
        return 0
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.executemany(
            f"INSERT OR IGNORE INTO {table} (date, value) VALUES (?, ?)",
            rows
        )
        con.commit()
        return len(rows)
    finally:
        con.close()
        
DEFAULT_START = {
    "GDP": "1999-01-01",
    "UNRATE": "1999-01-01",
    "CPIAUCSL": "1999-01-01",
    "HOUST": "1999-01-01",
    "SP500": "1999-01-01",
    "DFF": "1999-01-01",
}

DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(os.getcwd(), "database", "fed_data.db")
)

def _normalize_rows(obs_list):
    """[['date','value'], ...] -> list[(date, value)] and strip time."""
    return [(o["date"][:10], o["value"]) for o in obs_list if o.get("value") is not None]

def run_fetch_and_insert(db_path: str) -> dict:
    """Fetch every series and insert into SQLite. Returns counts attempted per table."""
    ensure_schema(db_path)
    results = {}

    for key, meta in SERIES_MAP.items():
        fred_id = meta["fred_id"]
        table   = meta["table"]

        obs = fetch_fred_series(
            fred_id,
            start_date=DEFAULT_START.get(fred_id)
        )
        rows = _normalize_rows(obs)
        count = insert_rows_ignore(db_path, table, rows)
        results[table] = count

    return results

@callback(
    Output("macro-status-badge", "children"),
    Output("macro-status-badge", "color"),
    Output("macro-status-text", "children"),
    Input("btn-get-latest", "n_clicks"),
    prevent_initial_call=True,
)
def on_get_latest(n_clicks):
    # set badge to Working… immediately
    try:
        badge_text, badge_color = "Working…", "yellow"
        status_line = "Fetching from FRED and inserting into SQLite…"

        counts = run_fetch_and_insert(DB_PATH)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        badge_text, badge_color = "Up to date", "green"
        nice_counts = ", ".join([f"{k}: {v}" for k, v in counts.items()])
        # status_line = f"Success at {ts}. Attempted inserts — {nice_counts}."
        status_line = f"Successfully updated macroeconomic statistics at {ts}."
        return badge_text, badge_color, status_line

    except Exception as e:
        return "Error", "red", f"Failed: {e}"
    
# Functions and callbacks for updating KPI cards

def _query_one(db_path: str, sql: str):
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(sql)
        row = cur.fetchone()
        return row
    finally:
        con.close()

def latest_row(db_path: str, table: str):
    """Returns (date, value) for the latest row in a table, or (None, None)."""
    row = _query_one(db_path, f"SELECT date, value FROM {table} ORDER BY date DESC LIMIT 1")
    return (row[0], row[1]) if row else (None, None)

def fmt_num(x, suffix=""):
    if x is None:
        return "—"
    # pretty formatting by magnitude
    if abs(x) >= 1_000_000_000:
        s = f"{x/1_000_000_000:.1f}B"
    elif abs(x) >= 1_000_000:
        s = f"{x/1_000_000:.1f}M"
    else:
        s = f"{x:,.2f}"
    return s + suffix

def fmt_billions(x):
    """Format a numeric value that is ALREADY in billions (e.g., 26.3 -> '26.3B')."""
    if x is None:
        return "—"
    return f"{x:,.1f}B"

def fmt_int(x):
    """No decimal places, with thousands separator (e.g., housing starts)."""
    if x is None:
        return "—"
    return f"{int(round(x)):,}"

def fmt_currency(x, code="$", decimals=2):
    """Currency with code appended (e.g., S&P 500 -> '5,123.45 USD')."""
    if x is None:
        return "—"
    return f"{code}{x:,.{decimals}f}"

@callback(
    Output("kpi-unemp", "children"),
    Output("kpi-gdp", "children"),
    Output("kpi-rate", "children"),
    Output("kpi-cpi", "children"),
    Output("kpi-housing", "children"),
    Output("kpi-sp500", "children"),

    Output("kpi-unemp-date", "children"),
    Output("kpi-gdp-date", "children"),
    Output("kpi-rate-date", "children"),
    Output("kpi-cpi-date", "children"),
    Output("kpi-housing-date", "children"),
    Output("kpi-sp500-date", "children"),

    Input("btn-get-latest", "n_clicks"),
    prevent_initial_call=False,   # populate on first load
)
def refresh_kpis(_n):
    # read latest rows from each table
    d_un, v_un = latest_row(DB_PATH, "unemployment_monthly")
    d_gdp, v_gdp = latest_row(DB_PATH, "gdp_quarterly")
    d_ffr, v_ffr = latest_row(DB_PATH, "fedfunds_daily")
    d_cpi, v_cpi = latest_row(DB_PATH, "cpi_monthly")
    d_hou, v_hou = latest_row(DB_PATH, "housing_monthly")
    d_spx, v_spx = latest_row(DB_PATH, "sp500_daily")

    # format (add % to rate series)
    unemp = fmt_num(v_un, "%") if v_un is not None else "—"
    ffr   = fmt_num(v_ffr, "%") if v_ffr is not None else "—"

    # series-specific formatting
    gdp   = fmt_billions(v_gdp) if v_gdp is not None else "—"        # GDP in billions -> 'xx.xB'
    cpi   = fmt_num(v_cpi) if v_cpi is not None else "—"
    hous  = fmt_int(v_hou) if v_hou is not None else "—"             # no decimals
    spx   = fmt_currency(v_spx, "$", 2) if v_spx is not None else "—"  # 'USD' appended

    # date captions
    def cap(d): return f"Latest: {d}" if d else "No data yet"

    return (
        unemp, gdp, ffr, cpi, hous, spx,
        cap(d_un), cap(d_gdp), cap(d_ffr), cap(d_cpi), cap(d_hou), cap(d_spx),
    )
    
    
# Helpers for displaying of FRED data
# table -> metadata for labels & native freq
SERIES_META = {
    "unemployment_monthly": {"fred": "UNRATE",  "name": "Unemployment rate",            "source": "BLS via FRED",        "native": "m", "units_hint": "%"},
    "gdp_quarterly":        {"fred": "GDP",     "name": "Real GDP (Ch.2017 $)",         "source": "BEA via FRED",         "native": "q"},
    "fedfunds_daily":       {"fred": "DFF",     "name": "Effective Federal Funds Rate", "source": "FRB/NY via FRED",      "native": "d", "units_hint": "%"},
    "cpi_monthly":          {"fred": "CPIAUCSL","name": "CPI (All items, SA, 82-84=100)","source": "BLS via FRED",         "native": "m"},
    "housing_monthly":      {"fred": "HOUST",   "name": "Housing starts (SAAR, thous.)", "source": "Census via FRED",      "native": "m"},
    "sp500_daily":          {"fred": "SP500",   "name": "S&P 500 index level",          "source": "S&P Dow Jones via FRED","native": "d"},
}

DB_PATH = os.environ.get("DB_PATH", os.path.join(os.getcwd(), "database", "fed_data.db"))

def read_series_df(table: str) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(f"SELECT date, value FROM {table} ORDER BY date", con)
    finally:
        con.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df

def resample_to(df: pd.DataFrame, target: str, native: str) -> pd.DataFrame:
    """Resample to target freq using standard conventions."""
    if df.empty or target in ("native", native):
        return df

    rule_map = {"d": "D", "m": "M", "q": "Q"}
    tgt = rule_map[target]
    # use mean for high-frequency to low-frequency (daily->monthly), and last for same-freq alignment
    if native == "d" and target == "m":
        return df.resample("M").mean()
    if native == "d" and target == "q":
        return df.resample("Q").mean()
    if native == "m" and target == "q":
        return df.resample("Q").mean()
    if native == "q" and target == "m":
        # upsample quarterly to monthly via forward-fill
        return df.resample("M").ffill()
    if native == "m" and target == "d":
        # upsample monthly to daily via forward-fill
        return df.resample("D").ffill()
    if native == "q" and target == "d":
        return df.resample("D").ffill()

    # fallback: last
    return df.resample(tgt).last()

def transform_units(df: pd.DataFrame, units: str, units_hint: str | None) -> tuple[pd.DataFrame, str]:
    """Return (transformed_df, suffix)."""
    if df.empty:
        return df, ""
    sfx = "" if units_hint is None else (units_hint if units == "level" else "%")

    if units == "level":
        return df, sfx

    # percentage changes expressed in %
    if units == "mom":
        out = df.pct_change(periods=1) * 100.0
        return out, "%"
    if units == "ann_mom":
        out = ((1 + df.pct_change()) ** 12 - 1) * 100.0
        return out, "%"
    if units == "yoy":
        # choose lag based on frequency (assume monthly if >= monthly)
        # for daily series, compute YoY using 252 trading days approximation
        if df.index.inferred_freq in ("M", "MS", "ME") or len(df.index) > 1 and (df.index[1] - df.index[0]).days >= 25:
            out = df.pct_change(periods=12) * 100.0
        elif df.index.inferred_freq in ("Q", "QS", "QE"):
            out = df.pct_change(periods=4) * 100.0
        else:
            out = df.pct_change(periods=252) * 100.0
        return out, "%"
    return df, sfx

def make_line_fig(df: pd.DataFrame, title: str, suffix: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["value"],
            mode="lines",
            line=dict(width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}" + (suffix if suffix else "") + "<extra></extra>",
            name=title,
        )
    )
    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title=title + (f" ({suffix})" if suffix and suffix != "%" else ""),
        legend=dict(orientation="h", y=1.05, x=0),
    )
    return fig

@callback(
    Output("disp-graph", "figure"),
    Output("disp-varname", "children"),
    Output("disp-source", "children"),
    Input("disp-indicator", "value"),
    Input("disp-range", "value"),
    Input("disp-frequency", "value"),
    Input("disp-units", "value"),
    Input("btn-get-latest", "n_clicks"),
    prevent_initial_call=False,  # also draws on first page load
)
def update_display(table, range_sel, freq_sel, units_sel, _n):
    meta = SERIES_META[table]
    df = read_series_df(table)
    if df.empty:
        # empty placeholder
        empty_fig = go.Figure().update_layout(template="plotly_white", annotations=[dict(
            text="No data yet. Click 'Get latest data'.", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper"
        )])
        return empty_fig, meta["name"], f"{meta['fred']} — {meta['source']}"

    # resample if needed
    df = resample_to(df, target=freq_sel, native=meta["native"])
    df.columns = ["value"]

    # transform units
    df, suffix = transform_units(df, units_sel, meta.get("units_hint"))

    # range presets
    if range_sel != "MAX":
        end = df.index.max()
        if range_sel == "1Y":
            start = end - pd.DateOffset(years=1)
        elif range_sel == "5Y":
            start = end - pd.DateOffset(years=5)
        else:  # "10Y"
            start = end - pd.DateOffset(years=10)
        df = df.loc[df.index >= start]

    title = f"{meta['name']} — {meta['fred']}"
    fig = make_line_fig(df.dropna(), title, suffix)

    varname = f"{meta['name']} ({meta['fred']})"
    source  = f"Source: {meta['source']}"
    return fig, varname, source