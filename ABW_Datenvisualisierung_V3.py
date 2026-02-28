"""
app.py — Streamlit Sensor-Explorer

Funktionen:
- Verbindet sich mit eurer Postgres-DB (SQLAlchemy)
- Liest Metadaten aus nodes_hierarchy
- Bietet Filter: Gebäude, Stockwerk, Raum, Sensortyp, Channel, Zeitraum
- Holt Zeitreihen aus nodes_history (occupancy/motion) und nodes_history_env (tempc, hum, co2, light)
- Zeigt interaktive Charts + Kennzahlen
- CSV-Export

Annahme zu Tabellen/Spalten:
- nodes_hierarchy(id, vanity, building, floor, area, nodeTypeName, roomtype, placetype, pax, folderId, ...)
- nodes_history(timestamp, value, ID, type) – für occupancy/motion
- nodes_history_env(timestamp, value, ID, type) – für tempc, hum, co2, light

Credential-Handling:
- ⚠️ Wie gewünscht **im Klartext im Code** hinterlegt (siehe Konstanten DB_USER/DB_PASS/DB_HOST/DB_PORT/DB_NAME)
"""

import os
import datetime as dt
from typing import List

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from time import perf_counter

# Optional: lokaler Cache (DuckDB)
try:
    import duckdb  # pip install duckdb
    HAVE_DUCKDB = True
except Exception:
    HAVE_DUCKDB = False

# ============================
# Setup & Config
# ============================
st.set_page_config(page_title="Sensor Explorer", layout="wide")

# --- ⚠️ DB-Zugang im Klartext (wie gewünscht)
DB_USER = "python_writer"
DB_PASS = "#writeonlye888!"
DB_HOST = "iot-abw-dev-postgessrv.postgres.database.azure.com"
DB_PORT = "5432"
DB_NAME = "postgres"

# --- Helper: DB Engine
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    connect_args = {"options": "-c statement_timeout=0"}
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)

engine = get_engine()

# ===============
# Lokaler Cache (DuckDB)
# ===============
CACHE_DB_PATH = os.path.join(os.path.dirname(__file__) if '__file__' in globals() else os.getcwd(), "sensor_cache.duckdb")

def cache_conn():
    if not HAVE_DUCKDB:
        raise RuntimeError("DuckDB nicht installiert. Bitte 'pip install duckdb' ausführen.")
    return duckdb.connect(CACHE_DB_PATH)

def ensure_cache_schema():
    if not HAVE_DUCKDB:
        return
    con = cache_conn()
    con.execute("""
        CREATE TABLE IF NOT EXISTS nodes_hierarchy_cache (
            id BIGINT,
            vanity VARCHAR,
            building VARCHAR,
            floor VARCHAR,
            area VARCHAR,
            nodeTypeName VARCHAR
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS nodes_history_cache (
            timestamp TIMESTAMP,
            value DOUBLE,
            ID BIGINT,
            type VARCHAR
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS nodes_history_env_cache (
            timestamp TIMESTAMP,
            value DOUBLE,
            ID BIGINT,
            type VARCHAR
        );
    """)
    con.close()

ensure_cache_schema()

def clear_cache():
    if not HAVE_DUCKDB:
        return
    con = cache_conn()
    con.execute("DELETE FROM nodes_hierarchy_cache;")
    con.execute("DELETE FROM nodes_history_cache;")
    con.execute("DELETE FROM nodes_history_env_cache;")
    con.close()

@st.cache_data(ttl=0, show_spinner=False)
def cache_stats():
    if not HAVE_DUCKDB:
        return {"enabled": False}
    con = cache_conn()
    stats = {}
    for tbl in ["nodes_hierarchy_cache","nodes_history_cache","nodes_history_env_cache"]:
        try:
            cnt = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        except Exception:
            cnt = 0
        stats[tbl] = cnt
    con.close()
    stats["enabled"] = True
    return stats

# Sync-Helfer
def sync_meta_to_cache(meta_df: pd.DataFrame):
    if not HAVE_DUCKDB or meta_df.empty:
        return
    con = cache_conn()
    con.execute("DELETE FROM nodes_hierarchy_cache;")
    con.register("meta_df", meta_df[["id","vanity","building","floor","area","nodeTypeName"]])
    con.execute("INSERT INTO nodes_hierarchy_cache SELECT * FROM meta_df;")
    con.close()

def upsert_timeseries_to_cache(df: pd.DataFrame):
    if not HAVE_DUCKDB or df.empty:
        return
    con = cache_conn()
    # Aufteilen nach Zieltabellen
    df_hist = df[df["type"].isin(["occupancy","motion"])].copy()
    df_env  = df[~df["type"].isin(["occupancy","motion"])].copy()
    if not df_hist.empty:
        con.execute("CREATE TEMP TABLE tmp_hist AS SELECT * FROM nodes_history_cache WHERE 1=0;")
        con.register("tmp_in", df_hist[["timestamp","value","ID","type"]])
        con.execute("INSERT INTO nodes_history_cache SELECT * FROM tmp_in;")
    if not df_env.empty:
        con.execute("CREATE TEMP TABLE tmp_env AS SELECT * FROM nodes_history_env_cache WHERE 1=0;")
        con.register("tmp_in2", df_env[["timestamp","value","ID","type"]])
        con.execute("INSERT INTO nodes_history_env_cache SELECT * FROM tmp_in2;")
    con.close()

# ============================
# Daten-Layer
# ============================

@st.cache_data(ttl=60*10, show_spinner=False)
def load_meta() -> pd.DataFrame:
    """Lädt Metadaten robust und vereinheitlicht die Spalte 'nodeTypeName',
    ohne Namen zu konvertieren (nur Alias auf vorhandene DB-Spalten).

    Pflicht: id
    Pflicht/gewünscht: nodeTypeName (aus einer der Kandidaten-Spalten)
    Optional: building, floor, area, vanity
    """
    use_cache = st.session_state.get("use_local_cache", False)
    if use_cache and HAVE_DUCKDB:
        try:
            con = cache_conn()
            df = con.execute("SELECT id, vanity, building, floor, area, nodeTypeName FROM nodes_hierarchy_cache").df()
            con.close()
            if not df.empty:
                # sicherstellen, dass Spalten als String vorliegen
                for col in ["nodeTypeName","building","floor","area","vanity"]:
                    if col in df.columns:
                        df[col] = df[col].fillna("").astype(str)
                return df
        except Exception:
            pass  # fällt auf DB-Load zurück

    # Kandidaten-Spalten für nodeTypeName (in DB können Namensvarianten existieren)
    candidates = [
        '"nodeTypeName"', 'nodeTypeName', '"nodetypename"', 'nodetypename',
        '"node_type_name"', 'node_type_name', '"nodetype"', 'nodetype', '"type"', 'type'
    ]

    base_df = None
    for cand in candidates:
        try:
            q = f"""
                SELECT id,
                       COALESCE({cand}, '') AS nodeTypeName
                FROM nodes_hierarchy
            """
            tmp = pd.read_sql(q, engine)
            if 'nodeTypeName' in tmp.columns:
                base_df = tmp
                break
        except Exception:
            continue

    if base_df is None:
        base_df = pd.read_sql("SELECT id FROM nodes_hierarchy", engine)
        base_df['nodeTypeName'] = ''

    df = base_df
    for col in ["building", "floor", "area", "vanity"]:
        try:
            tmp = pd.read_sql(f'SELECT id, {col} FROM nodes_hierarchy', engine)
            if col in tmp.columns:
                df = df.merge(tmp, on="id", how="left")
            if col not in df.columns:
                df[col] = ""
        except Exception:
            if col not in df.columns:
                df[col] = ""

    for col in ["nodeTypeName","building","floor","area","vanity"]:
        df[col] = df[col].fillna("").astype(str)

    # Bei aktivem Cache Metadaten synchronisieren
    if st.session_state.get("use_local_cache", False) and HAVE_DUCKDB:
        sync_meta_to_cache(df)

    return df

@st.cache_data(ttl=60*5, show_spinner=False)
def load_timeseries(
    id_list: List[int],
    channels: List[str],
    start: dt.datetime,
    end: dt.datetime
) -> pd.DataFrame:
    if not id_list or not channels:
        return pd.DataFrame(columns=["timestamp","value","ID","type"])  # leer

    # Wenn lokaler Cache verwendet werden soll, zuerst versuchen, aus Cache zu lesen
    use_cache = st.session_state.get("use_local_cache", False)
    frames = []
    if use_cache and HAVE_DUCKDB:
        try:
            con = cache_conn()
            ids_tuple = tuple(id_list)
            ch_tuple = tuple(channels)
            # history
            dfh = con.execute(
                """
                SELECT timestamp, value, ID, type
                FROM nodes_history_cache
                WHERE ID IN $ids AND type IN $chs AND timestamp >= $start AND timestamp < $end
                """,
                {"ids": ids_tuple, "chs": ch_tuple, "start": start, "end": end}
            ).df()
            # env
            dfe = con.execute(
                """
                SELECT timestamp, value, ID, type
                FROM nodes_history_env_cache
                WHERE ID IN $ids AND type IN $chs AND timestamp >= $start AND timestamp < $end
                """,
                {"ids": ids_tuple, "chs": ch_tuple, "start": start, "end": end}
            ).df()
            con.close()
            if not dfh.empty:
                frames.append(dfh)
            if not dfe.empty:
                frames.append(dfe)
        except Exception:
            pass

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
        df = df.dropna(subset=["timestamp"])  # ungültige raus
        return df

    # Kein Cache oder Cache leer → aus Postgres laden
    env_channels = {"tempc","hum","co2","light"}
    hist_channels = {"occupancy","motion"}

    need_env = [c for c in channels if c in env_channels]
    need_hist = [c for c in channels if c in hist_channels]

    frames = []

    if need_hist:
        ids_sql = ",".join([":hid"+str(i) for i,_ in enumerate(id_list)])
        ch_sql  = ",".join([":hc"+str(i)  for i,_ in enumerate(need_hist)])
        params = {("hid"+str(i)): v for i,v in enumerate(id_list)}
        params.update({("hc"+str(i)): v for i,v in enumerate(need_hist)})
        params.update({"start": start, "end": end})

        q_hist = text(f"""
            SELECT h."timestamp", h.value, h."ID", h."type"
            FROM nodes_history h
            WHERE h."ID" IN ({ids_sql})
              AND h."type" IN ({ch_sql})
              AND h."timestamp" >= :start
              AND h."timestamp" <  :end
        """)
        dfh = pd.read_sql(q_hist, engine, params=params)
        frames.append(dfh)

    if need_env:
        ids_sql = ",".join([":eid"+str(i) for i,_ in enumerate(id_list)])
        ch_sql  = ",".join([":ec"+str(i)  for i,_ in enumerate(need_env)])
        params = {("eid"+str(i)): v for i,v in enumerate(id_list)}
        params.update({("ec"+str(i)): v for i,v in enumerate(need_env)})
        params.update({"start": start, "end": end})

        q_env = text(f"""
            SELECT e."timestamp", e.value, e."ID", e."type"
            FROM nodes_history_env e
            WHERE e."ID" IN ({ids_sql})
              AND e."type" IN ({ch_sql})
              AND e."timestamp" >= :start
              AND e."timestamp" <  :end
        """)
        dfe = pd.read_sql(q_env, engine, params=params)
        frames.append(dfe)

    if not frames:
        return pd.DataFrame(columns=["timestamp","value","ID","type"])  # leer

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    df = df.dropna(subset=["timestamp"])  # ungültige raus

    # Wenn Cache aktiv, frisch geladene Daten in den lokalen Cache schreiben
    if st.session_state.get("use_local_cache", False) and HAVE_DUCKDB:
        upsert_timeseries_to_cache(df)

    return df

# ============================
# Self-Tests (kleine Checks statt Unit-Tests)
# ============================

def run_self_tests() -> pd.DataFrame:
    """Führt einfache Integrations-Checks aus und liefert eine Ergebnis-Tabelle."""
    checks = []
    # 1) DB-Verbindung
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks.append(("DB-Verbindung", True, "OK"))
    except Exception as e:
        checks.append(("DB-Verbindung", False, str(e)))

    # 2) Tabellen & Spalten
    expected = {
        # nodes_hierarchy: nur Minimal-Annahme, optionale Spalten werden zur Laufzeit ergänzt
        "nodes_hierarchy": {"id","nodeTypeName"},
        "nodes_history": {"timestamp","value","ID","type"},
        "nodes_history_env": {"timestamp","value","ID","type"},
    }
    for tbl, cols in expected.items():
        try:
            df_cols = pd.read_sql(text(f"SELECT * FROM {tbl} LIMIT 0"), engine).columns
            missing = cols - set(df_cols)
            if missing:
                checks.append((f"{tbl} Spalten", False, f"Fehlend: {sorted(missing)}"))
            else:
                checks.append((f"{tbl} Spalten", True, "OK"))
        except Exception as e:
            checks.append((f"{tbl} vorhanden", False, str(e)))

    # 3) Kleine Abfrage: ein beliebiger Sensor, 1 Tag, 1 Channel
    try:
        meta = load_meta()
        if not meta.empty:
            any_id = int(meta["id"].iloc[0])
            start = dt.datetime.now() - dt.timedelta(days=7)
            end = dt.datetime.now()
            df_try = load_timeseries([any_id], ["occupancy","tempc","hum","co2","light","motion"], start, end)
            if {"timestamp","value","ID","type"}.issubset(df_try.columns):
                checks.append(("Timeseries Probe", True, f"{len(df_try)} Zeilen"))
            else:
                checks.append(("Timeseries Probe", False, "Spalten unvollständig"))
        else:
            checks.append(("Timeseries Probe", False, "nodes_hierarchy leer"))
    except Exception as e:
        checks.append(("Timeseries Probe", False, str(e)))

    # 4) Meta sollte eine 'vanity'-Spalte haben
    try:
        meta = load_meta()
        has_name = "vanity" in meta.columns and meta["vanity"].notna().any()
        checks.append(("Meta enthält 'vanity'", bool(has_name), "OK" if has_name else "nicht vorhanden"))
    except Exception as e:
        checks.append(("Meta enthält 'vanity'", False, str(e)))
    return pd.DataFrame(checks, columns=["Check","OK","Details"])

# ============================
# UI — Header
# ============================

st.title("🔎 Sensor Explorer — SQL → Filters → Charts")
meta = load_meta()

c_top1, c_top2, c_top3 = st.columns([2,1,1])
with c_top1:
    st.caption("Select metadata & date range → data loads live from Postgres.")
with c_top2:
    if st.button("🔁 Clear app cache", help="Invalidate cached meta/queries in Streamlit"):
        load_meta.clear(); load_timeseries.clear()
        st.toast("Cache cleared")
with c_top3:
    if st.button("✅ Run self-test", help="Quick DB/schema/query checks"):
        res = run_self_tests()
        st.dataframe(res, use_container_width=True)

# ============================
# Sidebar (Cache, Preload, Filter)
# ============================
with st.sidebar:
    st.header("Cache & Filters")
    # Local cache switch
    use_local_cache = st.checkbox(
        "⚡ Use local cache (DuckDB)", value=False,
        help="Speeds up queries via local copy. Requires 'pip install duckdb'."
    )
    st.session_state["use_local_cache"] = use_local_cache

    cols = st.columns([1,1,1])
    with cols[0]:
        if st.button("🔃 Sync meta → cache", help="Load nodes_hierarchy from Postgres into local cache"):
            m = load_meta(); sync_meta_to_cache(m); st.toast("Meta synced to cache")
    with cols[1]:
        if st.button("🗑️ Clear cache", help="Delete locally cached data"):
            clear_cache(); cache_stats.clear(); st.toast("Cache cleared")
    with cols[2]:
        stats = cache_stats()
        if stats.get("enabled"):
            st.caption(
                f"Cache: meta={stats.get('nodes_hierarchy_cache',0)} "
                f"hist={stats.get('nodes_history_cache',0)} "
                f"env={stats.get('nodes_history_env_cache',0)}"
            )

    # Progress controls
    st.subheader("Progress settings")
    batch_size_ids = st.number_input(
        "Batch size (IDs per step)", min_value=1, max_value=2000, value=50, step=10,
        help="Controls how many sensor IDs are fetched per batch."
    )
    progress_mode = st.selectbox(
        "Progress detail", ["Minimal","Verbose"], index=1,
        help="Verbose shows rows/MB per batch and cumulative stats."
    )

    # ==== Preload a date range into cache ====
    st.subheader("Preload range → cache")
    preload_col1, preload_col2 = st.columns(2)
    with preload_col1:
        _today = dt.date.today()
        preload_date_sel = st.date_input("Preload date range", value=(_today, _today))
        if isinstance(preload_date_sel, (list, tuple)) and len(preload_date_sel) == 2:
            preload_date_from, preload_date_to = preload_date_sel
        else:
            preload_date_from = preload_date_sel
            preload_date_to = preload_date_sel
    with preload_col2:
        preload_scope = st.selectbox(
            "Scope", ["All sensors", "Current filter"], index=1,
            help="Choose whether to preload for all sensors or only those matching current Building/Floor/Area filters."
        )
    st.caption("Loads all selected channels for the chosen date **range** into the local DuckDB cache for instant reuse.")

    # Preview count based on current Building/Floor/Area/Type/Vanity filters
    _pre_meta = meta.copy()
    if 'sel_buildings' in locals() and sel_buildings:
        _pre_meta = _pre_meta[_pre_meta["building"].isin(sel_buildings)]
    if 'sel_floors' in locals() and sel_floors:
        _pre_meta = _pre_meta[_pre_meta["floor"].isin(sel_floors)]
    if 'sel_rooms' in locals() and sel_rooms:
        _pre_meta = _pre_meta[_pre_meta["area"].isin(sel_rooms)]
    if 'sel_types' in locals() and sel_types:
        _pre_meta = _pre_meta[_pre_meta["nodeTypeName"].isin(sel_types)]
    if 'sel_vanity' in locals() and sel_vanity:
        _pre_meta = _pre_meta[_pre_meta["vanity"].isin(sel_vanity)]
    st.caption(f"Sensors matching current filter: {_pre_meta['id'].nunique()}")

    if st.button("⚡ Preload to cache"):
        if not HAVE_DUCKDB:
            st.error("DuckDB not installed. Run: pip install duckdb")
        else:
            # Determine IDs (respect scope)
            if preload_scope == "All sensors":
                ids_for_preload = meta["id"].tolist()
            else:
                ids_for_preload = _pre_meta["id"].tolist()

            # Determine channels (use user-selected or all default)
            channels_for_preload = sel_channels if 'sel_channels' in locals() and sel_channels else ["occupancy","motion","tempc","hum","co2","light"]

            # Compute range [from, to)
            start_pre = dt.datetime.combine(preload_date_from, dt.time(0,0))
            end_pre   = dt.datetime.combine(preload_date_to, dt.time(23,59,59))

            # Ensure cache is on so load_timeseries will upsert to cache
            st.session_state["use_local_cache"] = True

            total_ids = len(ids_for_preload)
            if total_ids == 0:
                st.info("No sensors to preload for the chosen scope.")
            else:
                total_batches = (total_ids + batch_size_ids - 1) // batch_size_ids
                progress = st.progress(0, text=f"Starting preload… 0/{total_batches} batches")
                status = st.empty()
                log_area = st.empty(); logs = []
                total_rows = 0; total_bytes = 0
                t0 = perf_counter()

                for b, i in enumerate(range(0, total_ids, batch_size_ids), start=1):
                    batch_ids = ids_for_preload[i:i+batch_size_ids]
                    status.write(
                        f"Batch {b}/{total_batches} — IDs {i+1}–{i+len(batch_ids)} / {total_ids} | "
                        f"Channels: {', '.join(channels_for_preload)} | {start_pre:%Y-%m-%d}→{end_pre:%Y-%m-%d}"
                    )
                    tb = perf_counter()
                    df_pre = load_timeseries(batch_ids, channels_for_preload, start_pre, end_pre)
                    dt_s = perf_counter() - tb

                    # Metrics for this batch
                    rows = len(df_pre)
                    bytes_ = int(df_pre.memory_usage(deep=True).sum()) if rows else 0
                    mb_ = bytes_ / (1024*1024)
                    rps = rows / dt_s if dt_s > 0 else 0

                    total_rows += rows
                    total_bytes += bytes_
                    pct = min(1.0, b / total_batches)
                    progress.progress(pct, text=f"Preloading… {b}/{total_batches} batches")

                    if progress_mode == "Verbose":
                        logs.append(f"Batch {b}: {rows:,} rows ({mb_:.2f} MB) in {dt_s:.2f}s → {rps:,.0f} rows/s".replace(",","."))
                        log_area.code("\n".join(logs), language="text")

                progress.empty(); status.empty()
                # Keep batch log visible after finishing
                if progress_mode == "Verbose" and logs:
                    with st.expander("Preload log (batches)", expanded=False):
                        # Final batch log
                        st.code("\n".join(logs), language="text")

                mb_total = total_bytes / (1024*1024)
                elapsed = perf_counter() - t0
                st.success(f"Preloaded ~{total_rows:,} rows (~{mb_total:.2f} MB) in {elapsed:.2f}s".replace(",","."))
                cache_stats.clear()

    # --- Kaskadierende Auswahllisten (Gebäude → Stockwerk → Area)
    st.header("Filters")

    # 1) Gebäude (aus allen Metadaten)
    buildings_all = sorted([b for b in meta["building"].unique() if b])
    sel_buildings = st.multiselect("Building", buildings_all, default=buildings_all[:1] if buildings_all else [])

    # 2) Stockwerke gefiltert nach Gebäude(n)
    meta_b = meta[meta["building"].isin(sel_buildings)] if sel_buildings else meta
    floors_opts = sorted([f for f in meta_b["floor"].unique() if f])
    sel_floors  = st.multiselect("Floor", floors_opts)

    # 3) Areas gefiltert nach Gebäude + Stockwerk
    meta_bf = meta_b[meta_b["floor"].isin(sel_floors)] if sel_floors else meta_b
    areas_opts = sorted([r for r in meta_bf["area"].unique() if r])
    sel_rooms  = st.multiselect("Area (filtered)", areas_opts)

    # 4) Sensortypen gefiltert nach Gebäude + Stockwerk + Area
    meta_bfa = meta_bf[meta_bf["area"].isin(sel_rooms)] if sel_rooms else meta_bf
    types     = sorted([t for t in meta_bfa["nodeTypeName"].unique() if t])
    sel_types = st.multiselect("Sensor type", types, help="e.g., ems_desk, ers_eye, ers_co2 …")

    # 5) Vanity gefiltert (optional)
    vanitys   = sorted([v for v in meta_bfa["vanity"].unique() if v])
    sel_vanity= st.multiselect("Vanity", vanitys)

    # Channels je Sensortyp (nur für Anzeige-Auswahl; Laden bleibt optional)
    SENSOR_CHANNELS = {
        'ems_desk': ['occupancy', 'tempc', 'hum'],
        'ers_eye':  ['occupancy', 'motion', 'light', 'tempc', 'hum'],
        'ers_co2':  ['tempc', 'hum', 'co2'],
        'ers_desk': ['occupancy', 'tempc', 'hum']
    }
    ALL_CHANNELS = sorted(list({c for lst in SENSOR_CHANNELS.values() for c in lst}))
    sel_channels = st.multiselect("Channel (optional)", ALL_CHANNELS, default=[], help="Leave empty to load all channels.")

    # Zeitraum
    today = dt.datetime.now().date()
    date_sel = st.date_input(
        "Date range",
        value=(today - dt.timedelta(days=7), today),
        max_value=today
    )
    if isinstance(date_sel, (list, tuple)) and len(date_sel) == 2:
        date_from, date_to = date_sel
    else:
        date_from = date_sel
        date_to = date_sel

    c1, c2 = st.columns(2)
    with c1:
        from_time = st.time_input("from", value=dt.time(0,0))
    with c2:
        to_time   = st.time_input("to", value=dt.time(23,59))

    # Resampling
    agg_options = ["None (raw data)", "15Min", "H", "D", "W", "M"]
    agg = st.selectbox(
        "Aggregation (resampling)", agg_options, index=0,
        help="Zwischenwerte mitteln/summieren: 15Min, Stunde (H), Tag (D), Woche (W), Monat (M). Choose 'None (raw data)' to use original values."
    )

    agg_disabled = (agg == "None (raw data)")
    agg_func = st.selectbox(
        "Aggregation function",
        ["mean", "sum", "max", "min", "median"], index=0,
        disabled=agg_disabled,
        help="Only relevant when a resampling interval is selected."
    )

    # Button
    run = st.button("🔍 Load data")

# ============================
# Filter anwenden & Zeitraum
# ============================
f = meta.copy()
if sel_buildings:
    f = f[f["building"].isin(sel_buildings)]
if sel_floors:
    f = f[f["floor"].isin(sel_floors)]
if sel_rooms:
    f = f[f["area"].isin(sel_rooms)]
if sel_types:
    f = f[f["nodeTypeName"].isin(sel_types)]
if sel_vanity:
    f = f[f["vanity"].isin(sel_vanity)]

id_list = f["id"].tolist()

# quick preview of affected sensors before loading
st.caption(f"Selected sensors: {len(id_list)}")

start_dt = dt.datetime.combine(date_from, from_time)
end_dt   = dt.datetime.combine(date_to, to_time)

# ============================
# Query & Anzeige
# ============================
if run:
    if not id_list:
        st.warning("No sensors found for the selected filters.")
        st.stop()

    # Batch progress for normal loading
    channels_to_use = sel_channels if sel_channels else ["occupancy","motion","tempc","hum","co2","light"]

    total_ids = len(id_list)
    frames = []
    if total_ids > 0:
        total_batches = (total_ids + batch_size_ids - 1) // batch_size_ids
        progress = st.progress(0, text=f"Starting load… 0/{total_batches} batches")
        status = st.empty()
        log_area = st.empty(); logs = []
        t0 = perf_counter()

        for b, i in enumerate(range(0, total_ids, batch_size_ids), start=1):
            batch_ids = id_list[i:i+batch_size_ids]
            status.write(
                f"Batch {b}/{total_batches} — IDs {i+1}–{i+len(batch_ids)} / {total_ids} | "
                f"Channels: {', '.join(channels_to_use)} | {start_dt:%Y-%m-%d}→{end_dt:%Y-%m-%d}"
            )
            tb = perf_counter()
            _df = load_timeseries(batch_ids, channels_to_use, start_dt, end_dt)
            dt_s = perf_counter() - tb

            if not _df.empty:
                frames.append(_df)
                rows = len(_df)
                bytes_ = int(_df.memory_usage(deep=True).sum())
                mb_ = bytes_ / (1024*1024)
                rps = rows / dt_s if dt_s > 0 else 0
                if progress_mode == "Verbose":
                    logs.append(f"Batch {b}: {rows:,} rows ({mb_:.2f} MB) in {dt_s:.2f}s → {rps:,.0f} rows/s".replace(",","."))
                    log_area.code("\n".join(logs), language="text")

            pct = min(1.0, b / total_batches)
            progress.progress(pct, text=f"Loading… {b}/{total_batches} batches")

        progress.empty(); status.empty()
        # Keep batch log visible
        if progress_mode == "Verbose" and logs:
            with st.expander("Load log (batches)", expanded=False):
                # Final batch log
                st.code("\n".join(logs), language="text")

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp","value","ID","type"])

    if df.empty:
        st.info("No data in the selected date range/channels.")
        st.stop()

    # Join Meta für Anzeige
    df = df.merge(meta[["id","vanity","building","floor","area","nodeTypeName"]], left_on="ID", right_on="id", how="left")
    df.drop(columns=["id"], inplace=True)

    # Aggregation optional
    if agg != "None (raw data)":
        # Resampling pro (ID,type) Gruppe
        df = (df
              .set_index("timestamp")
              .groupby(["ID","type","vanity","building","floor","area","nodeTypeName"])
              .resample(agg)["value"]
        )
        # Aggregationsfunktion wählen
        if agg_func == "mean":
            df = df.mean()
        elif agg_func == "sum":
            df = df.sum()
        elif agg_func == "max":
            df = df.max()
        elif agg_func == "min":
            df = df.min()
        else:
            df = df.median()

        df = df.reset_index()
    else:
        # sicherstellen, dass nach Zeit sortiert
        df = df.sort_values(["ID","type","timestamp"])

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.metric("Data points", f"{len(df):,}".replace(",","."))
    with k2:
        st.metric("Sensors", f["id"].nunique())
    with k3:
        st.metric("Channels", len(set(df["type"].unique().tolist())))
    with k4:
        st.metric("Date range", f"{start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d}")

    # Memory footprint of the loaded dataframe in Python
    try:
        _bytes = int(df.memory_usage(deep=True).sum())
        _mb = _bytes / (1024*1024)
        st.caption(f"Loaded ~{_mb:.2f} MB into memory for this view.")
    except Exception:
        pass

    # Chart — Multi-Line (eine Linie pro (ID, type))
    st.subheader("Time series — filtered by Building/Floor/Area")

    # Für klare Legende / Darstellung
    if sel_vanity:
        # Ein Graph je Vanity
        for v in sel_vanity:
            df_v = df[df["vanity"] == v]
            st.subheader(f"Time series — {v}")
            if df_v.empty:
                st.info("No data for this vanity in the selected date range/channels.")
                continue
            fig = px.line(
                df_v,
                x="timestamp",
                y="value",
                color="type",
                hover_data=["type","vanity","building","floor","area","nodeTypeName"],
                title=None
            )
            fig.update_layout(legend_title_text="Channel")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    else:
        # Sammelgraph
        df["series"] = df.apply(lambda r: f"{r['type']} | {r.get('vanity','')} ({r['building']} {r['floor']}/{r['area']})", axis=1)
        fig = px.line(
            df,
            x="timestamp",
            y="value",
            color="series",
            hover_data=["type","vanity","building","floor","area","nodeTypeName"],
            title="Time series"
        )
        fig.update_layout(legend_title_text="Series")
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # Tabellarische Ansicht
    st.subheader("Table")
    st.dataframe(df.sort_values(["timestamp"]).reset_index(drop=True), use_container_width=True)

    # Export
    st.download_button(
        "⬇️ Export CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"sensor_export_{start_dt:%Y%m%d}_{end_dt:%Y%m%d}.csv",
        mime="text/csv"
    )

# ============================
# Footer / Hilfe
# ============================
with st.expander("Help & tips"):
    st.markdown(
        """
        **Start in VS Code / Terminal**
        - Run as a Streamlit app, **not** with `python your_script.py`.
        - Command:
          ```bash
          streamlit run "path/to/ABW_Datenvisualisierung.py"
          ```

        **Tips**
        - Use *resampling* (e.g., `H`) to smooth large raw datasets.
        - Filter top-down (Building → Floor → **Area**), then Sensor type & Channel.
        - If fields like `building/floor/area` are missing in `nodes_hierarchy`, the app will treat them as empty strings.

        **Performance**
        - `@st.cache_data` and `@st.cache_resource` reduce DB load.
        - For huge ranges, prefer pre-aggregated DB views/materialized views.

        **Security**
        - ⚠️ Plaintext credentials are convenient but insecure. Prefer secrets/Key Vault in production.
        """
    )
