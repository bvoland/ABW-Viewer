"""
app.py — Streamlit Sensor-Explorer (mit Chart-Anpassungen)

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
- lokale Datei `db_config.local.json` bzw. `/app/data/db_config.local.json`
- alternativ Umgebungsvariablen `ABW_DB_*`
"""

import json
import os
import datetime as dt
from typing import List

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
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

# --- DB-Zugang aus lokaler Datei oder Umgebungsvariablen
APP_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.join(APP_DIR, "data")
LEGACY_CACHE_DB_PATH = os.path.join(APP_DIR, "sensor_cache.duckdb")
CACHE_DB_PATH = os.path.join(DATA_DIR, "sensor_cache.duckdb")
DB_CONFIG_CANDIDATES = [
    os.path.join(DATA_DIR, "db_config.local.json"),
    os.path.join(APP_DIR, "db_config.local.json"),
]


def get_db_config_path() -> str:
    for path in DB_CONFIG_CANDIDATES:
        if os.path.exists(path):
            return path
    return DB_CONFIG_CANDIDATES[0]


def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def migrate_legacy_cache() -> None:
    ensure_data_dir()
    if os.path.exists(CACHE_DB_PATH) or not os.path.exists(LEGACY_CACHE_DB_PATH):
        return
    try:
        os.replace(LEGACY_CACHE_DB_PATH, CACHE_DB_PATH)
    except OSError:
        pass


def load_local_db_config() -> dict:
    for path in DB_CONFIG_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def save_local_db_config(settings: dict) -> str:
    path = get_db_config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "user": str(settings.get("user", "")).strip(),
        "password": str(settings.get("password", "")).strip(),
        "host": str(settings.get("host", "")).strip(),
        "port": str(settings.get("port", "")).strip(),
        "name": str(settings.get("name", "")).strip(),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return path


def get_db_settings() -> dict:
    local_cfg = load_local_db_config()
    return {
        "user": os.getenv("ABW_DB_USER", str(local_cfg.get("user", "")).strip()),
        "password": os.getenv("ABW_DB_PASS", str(local_cfg.get("password", "")).strip()),
        "host": os.getenv("ABW_DB_HOST", str(local_cfg.get("host", "")).strip()),
        "port": os.getenv("ABW_DB_PORT", str(local_cfg.get("port", "")).strip()),
        "name": os.getenv("ABW_DB_NAME", str(local_cfg.get("name", "")).strip()),
    }


def get_missing_db_settings() -> List[str]:
    settings = get_db_settings()
    return [key for key, value in settings.items() if not value]


def get_env_db_overrides() -> List[str]:
    mapping = {
        "user": "ABW_DB_USER",
        "password": "ABW_DB_PASS",
        "host": "ABW_DB_HOST",
        "port": "ABW_DB_PORT",
        "name": "ABW_DB_NAME",
    }
    return [key for key, env_name in mapping.items() if os.getenv(env_name, "").strip()]


def is_db_configured() -> bool:
    return not get_missing_db_settings()


def get_empty_meta() -> pd.DataFrame:
    return pd.DataFrame(columns=["id", "vanity", "building", "floor", "area", "nodeTypeName"])


def build_db_url(settings: dict) -> URL:
    return URL.create(
        "postgresql+psycopg2",
        username=settings["user"],
        password=settings["password"],
        host=settings["host"],
        port=int(settings["port"]),
        database=settings["name"],
    )


def test_db_connection(settings: dict) -> tuple[bool, str]:
    missing = [key for key, value in settings.items() if not str(value).strip()]
    if missing:
        return False, "Fehlende Felder: " + ", ".join(missing)
    try:
        engine = create_engine(
            build_db_url(settings),
            connect_args={"options": "-c statement_timeout=0"},
            pool_pre_ping=True,
        )
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
        engine.dispose()
        return result == 1, "Verbindung erfolgreich."
    except Exception as exc:
        return False, str(exc)


def get_request_header(name: str, default: str = "") -> str:
    try:
        headers = st.context.headers
        value = headers.get(name)
        if value is not None:
            return str(value)
        for key, current_value in headers.items():
            if str(key).lower() == name.lower():
                return str(current_value)
    except Exception:
        return default
    return default


def parse_group_values(raw_value: str) -> List[str]:
    raw_value = (raw_value or "").strip()
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list):
            return sorted({str(item).strip() for item in parsed if str(item).strip()})
    except Exception:
        pass
    normalized = raw_value
    for sep in [";", "|", "\n"]:
        normalized = normalized.replace(sep, ",")
    return sorted({item.strip() for item in normalized.split(",") if item.strip()})


def get_auth_context() -> dict:
    username = get_request_header("X-Authentik-Username")
    email = get_request_header("X-Authentik-Email")
    name = get_request_header("X-Authentik-Name")
    groups = parse_group_values(get_request_header("X-Authentik-Groups"))
    return {
        "username": username,
        "email": email,
        "name": name,
        "groups": groups,
        "is_authenticated": bool(username or email or name),
        "is_admin": "abw-admin" in groups,
    }

# --- Helper: DB Engine
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    settings = get_db_settings()
    missing = [key for key, value in settings.items() if not value]
    if missing:
        raise RuntimeError(
            "DB-Konfiguration unvollstaendig. Fehlende Felder: "
            + ", ".join(missing)
            + ". Nutze Umgebungsvariablen ABW_DB_* oder db_config.local.json."
        )
    connect_args = {"options": "-c statement_timeout=0"}
    return create_engine(build_db_url(settings), connect_args=connect_args, pool_pre_ping=True)

# ===============
# Lokaler Cache (DuckDB)
# ===============
def cache_conn():
    if not HAVE_DUCKDB:
        raise RuntimeError("DuckDB nicht installiert. Bitte 'pip install duckdb' ausführen.")
    ensure_data_dir()
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

migrate_legacy_cache()
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
    use_cache = (
        st.session_state.get("use_local_cache", False)
        or st.session_state.get("use_cache_only", False)
    )
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
    if st.session_state.get("use_cache_only"):
        return get_empty_meta()

    try:
        engine = get_engine()
    except RuntimeError:
        return get_empty_meta()

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

    use_cache = (
        st.session_state.get("use_local_cache", False)
        or st.session_state.get("use_cache_only", False)
    )
    frames = []

    # ---------- Versuch: aus DuckDB-Cache laden ----------
    if use_cache and HAVE_DUCKDB:
        try:
            con = cache_conn()

            # IDs/Channels als kleine temp Tabellen registrieren
            ids_df = pd.DataFrame({"ID": list(set(id_list))})
            chs_df = pd.DataFrame({"type": list(set(channels))})
            con.register("ids_req", ids_df)
            con.register("chs_req", chs_df)

            # history (occupancy/motion)
            dfh = con.execute(
                """
                SELECT c.timestamp, c.value, c.ID, c.type
                FROM nodes_history_cache AS c
                JOIN ids_req  USING (ID)
                JOIN chs_req  ON c.type = chs_req.type
                WHERE c.timestamp >= ? AND c.timestamp < ?
                """,
                [start, end],
            ).df()

            # env (tempc/hum/co2/light)
            dfe = con.execute(
                """
                SELECT c.timestamp, c.value, c.ID, c.type
                FROM nodes_history_env_cache AS c
                JOIN ids_req  USING (ID)
                JOIN chs_req  ON c.type = chs_req.type
                WHERE c.timestamp >= ? AND c.timestamp < ?
                """,
                [start, end],
            ).df()

            con.close()

            if not dfh.empty:
                frames.append(dfh)
            if not dfe.empty:
                frames.append(dfe)

            if frames:
                df = pd.concat(frames, ignore_index=True)
                # Doppelte vermeiden (falls Preload + Live-Load gemischt wurde)
                df = df.drop_duplicates(subset=["timestamp", "ID", "type"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
                df = df.dropna(subset=["timestamp"])
                return df
        except Exception:
            # Fallback: einfach mit Postgres weiter unten fortfahren
            pass

    if st.session_state.get("use_cache_only"):
        return pd.DataFrame(columns=["timestamp","value","ID","type"])

    try:
        engine = get_engine()
    except RuntimeError:
        return pd.DataFrame(columns=["timestamp","value","ID","type"])

    # ---------- Postgres laden (mit ANY() Arrays) ----------
    env_channels = {"tempc","hum","co2","light"}
    hist_channels = {"occupancy","motion"}
    need_env  = [c for c in channels if c in env_channels]
    need_hist = [c for c in channels if c in hist_channels]

    frames = []

    if need_hist:
        q_hist = text("""
            SELECT h."timestamp", h.value, h."ID", h."type"
            FROM nodes_history h
            WHERE h."ID" = ANY(:ids)
              AND h."type" = ANY(:types)
              AND h."timestamp" >= :start
              AND h."timestamp" <  :end
        """)
        dfh = pd.read_sql(
            q_hist, engine,
            params={"ids": id_list, "types": need_hist, "start": start, "end": end},
        )
        frames.append(dfh)

    if need_env:
        q_env = text("""
            SELECT e."timestamp", e.value, e."ID", e."type"
            FROM nodes_history_env e
            WHERE e."ID" = ANY(:ids)
              AND e."type" = ANY(:types)
              AND e."timestamp" >= :start
              AND e."timestamp" <  :end
        """)
        dfe = pd.read_sql(
            q_env, engine,
            params={"ids": id_list, "types": need_env, "start": start, "end": end},
        )
        frames.append(dfe)

    if not frames:
        return pd.DataFrame(columns=["timestamp","value","ID","type"])

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Frisch geladene Daten in den Cache schreiben (wenn gewünscht)
    if st.session_state.get("use_local_cache", False) and HAVE_DUCKDB and not df.empty:
        upsert_timeseries_to_cache(df)

    return df

# ============================
# Self-Tests (kleine Checks statt Unit-Tests)
# ============================

def run_self_tests() -> pd.DataFrame:
    """Führt einfache Integrations-Checks aus und liefert eine Ergebnis-Tabelle."""
    checks = []
    try:
        engine = get_engine()
    except RuntimeError as e:
        return pd.DataFrame(
            [
                ("DB-Konfiguration", False, str(e)),
                ("Hinweis", False, f"Erwartete Datei: {DB_CONFIG_PATH}"),
            ],
            columns=["Check","OK","Details"],
        )
    # 1) DB-Verbindung
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks.append(("DB-Verbindung", True, "OK"))
    except Exception as e:
        checks.append(("DB-Verbindung", False, str(e)))

    # 2) Tabellen & Spalten
    expected = {
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

    try:
        df_cols = set(pd.read_sql(text("SELECT * FROM nodes_hierarchy LIMIT 0"), engine).columns)
        node_type_candidates = {"nodeTypeName", "nodetypename", "node_type_name", "nodetype", "type"}
        missing = []
        if "id" not in df_cols:
            missing.append("id")
        if not (df_cols & node_type_candidates):
            missing.append("nodeTypeName|nodetypename|node_type_name|nodetype|type")
        if missing:
            checks.append(("nodes_hierarchy Spalten", False, f"Fehlend: {missing}"))
        else:
            checks.append(("nodes_hierarchy Spalten", True, "OK"))
    except Exception as e:
        checks.append(("nodes_hierarchy vorhanden", False, str(e)))

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

st.session_state.setdefault("use_local_cache", False)
st.session_state.setdefault("use_cache_only", False)

auth_ctx = get_auth_context()

st.title("🔎 Sensor Explorer — SQL → Filters → Charts")
if not is_db_configured():
    st.warning(
        "DB-Konfiguration fehlt. Lege `db_config.local.json` an oder setze "
        "`ABW_DB_USER`, `ABW_DB_PASS`, `ABW_DB_HOST`, `ABW_DB_PORT`, `ABW_DB_NAME`."
    )
meta = load_meta()

c_top1, c_top2, c_top3 = st.columns([2,1,1])
with c_top1:
    if auth_ctx["is_authenticated"]:
        identity = auth_ctx["email"] or auth_ctx["username"] or auth_ctx["name"]
        groups_text = ", ".join(auth_ctx["groups"]) if auth_ctx["groups"] else "keine Gruppen"
        st.caption(f"Angemeldet als {identity} | Gruppen: {groups_text}")
    else:
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
# Sidebar (Cache, Preload, Filter, Charts)
# ============================
with st.sidebar:
    st.header("Cache & Filters")
    with st.expander("Admin", expanded=auth_ctx["is_admin"]):
        current_config_path = get_db_config_path()
        env_overrides = get_env_db_overrides()
        if auth_ctx["is_authenticated"]:
            st.caption(
                "Angemeldet als "
                + (auth_ctx["email"] or auth_ctx["username"] or auth_ctx["name"])
            )
        if env_overrides:
            st.warning(
                "Umgebungsvariablen überschreiben aktuell diese Felder: "
                + ", ".join(env_overrides)
            )
        if not auth_ctx["is_admin"]:
            st.info("DB-Konfiguration ist nur für Mitglieder der Gruppe `abw-admin` sichtbar.")
        else:
            current_cfg = load_local_db_config()
            with st.form("db_config_form", clear_on_submit=False):
                db_user = st.text_input("DB User", value=str(current_cfg.get("user", "")))
                db_password = st.text_input(
                    "DB Passwort", value=str(current_cfg.get("password", "")), type="password"
                )
                db_host = st.text_input("DB Host", value=str(current_cfg.get("host", "")))
                db_port = st.text_input("DB Port", value=str(current_cfg.get("port", "5432")))
                db_name = st.text_input("DB Name", value=str(current_cfg.get("name", "postgres")))
                test_clicked = st.form_submit_button("Verbindung testen")
                save_clicked = st.form_submit_button("Konfiguration speichern")

            draft_settings = {
                "user": db_user,
                "password": db_password,
                "host": db_host,
                "port": db_port,
                "name": db_name,
            }
            if test_clicked:
                ok, message = test_db_connection(draft_settings)
                if ok:
                    st.success(message)
                else:
                    st.error(message)
            if save_clicked:
                ok, message = test_db_connection(draft_settings)
                if not ok:
                    st.error("Speichern abgebrochen. " + message)
                else:
                    saved_path = save_local_db_config(draft_settings)
                    get_engine.clear()
                    load_meta.clear()
                    load_timeseries.clear()
                    st.success(f"Konfiguration gespeichert: {saved_path}")
                    st.rerun()
            st.caption(f"Aktiver Konfigurationspfad: {current_config_path}")

    # Local cache switch
    use_local_cache = st.checkbox(
        "⚡ Use local cache (DuckDB)", value=False,
        help="Speeds up queries via local copy. Requires 'pip install duckdb'."
    )
    st.session_state["use_local_cache"] = use_local_cache

    # Optional: Cache-only (zum Verifizieren, dass Preload greift)
    use_cache_only = st.checkbox(
        "Use cache only (no DB fetch)",
        value=False,
        help="Force DuckDB only. Good to verify that preloaded ranges fully cover your queries. Only locally stored data is used; no data is retrieved from the PostgreSQL database."
    )
    st.session_state["use_cache_only"] = use_cache_only

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

    # Platzhalter – wird später unten anhand tatsächlicher Filter ersetzt
    sensors_per_chart = st.number_input(
        "Sensors per chart", min_value=1, max_value=1000, value=12, step=1,
        help="Controls how many sensors are plotted per figure."
    )
    chart_mode = st.selectbox(
        "Chart mode", ["By channel", "All channels"],
        help="By channel: one figure per channel. All channels: combined figure."
    )
    rolling_window = st.number_input(
        "Rolling window (points)", min_value=0, max_value=500, value=0, step=1,
        help="Apply moving average smoothing (0 = off)."
    )
    show_markers = st.checkbox("Show markers", value=False)
    show_rangeslider = st.checkbox("Show range slider", value=True)

    # Channels je Sensortyp (nur für Anzeige-Auswahl; Laden bleibt optional)
    st.header("Filters")
    # (Der Rest der Sidebar kommt gleich nach der Preload-Logik)

# ============================
# Sidebar – eigentliche Filter (abhängig von Meta)
# ============================
with st.sidebar:
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

    # Preview count based on current Building/Floor/Area/Type/Vanity filters (für Preload-Info)
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

    # Button
    run = st.button("🔍 Load data")
    if st.button("⚡ Preload to cache"):
        if not HAVE_DUCKDB:
            st.error("DuckDB not installed. Run: pip install duckdb")
        else:
            # Determine IDs (respect scope)
            if 'preload_scope' in locals() and preload_scope == "All sensors":
                ids_for_preload = meta["id"].tolist()
            else:
                ids_for_preload = _pre_meta["id"].tolist()

            # Determine channels (use user-selected or all default)
            channels_for_preload = sel_channels if 'sel_channels' in locals() and sel_channels else ["occupancy","motion","tempc","hum","co2","light"]

            # Compute range [from, to) — Ende = nächster Tagesbeginn (exklusiv)
            start_pre = dt.datetime.combine(preload_date_from, dt.time(0,0))
            end_pre   = dt.datetime.combine(preload_date_to + dt.timedelta(days=1), dt.time(0,0))

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
                        f"Channels: {', '.join(channels_for_preload)} | {start_pre:%Y-%m-%d}→{(end_pre - dt.timedelta(seconds=1)):%Y-%m-%d}"
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
                        logs.append(f"Batch {b}: {rows:,} rows ({mb_:.2f} MB) in {dt_s:.2f}s → {rps:,.0f} rows/s".replace(",","."))  # DE-Decimal hack
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

# ============================
# Filter anwenden & Zeitraum
# ============================
f = meta.copy()
if 'sel_buildings' in locals() and sel_buildings:
    f = f[f["building"].isin(sel_buildings)]
if 'sel_floors' in locals() and sel_floors:
    f = f[f["floor"].isin(sel_floors)]
if 'sel_rooms' in locals() and sel_rooms:
    f = f[f["area"].isin(sel_rooms)]
if 'sel_types' in locals() and sel_types:
    f = f[f["nodeTypeName"].isin(sel_types)]
if 'sel_vanity' in locals() and sel_vanity:
    f = f[f["vanity"].isin(sel_vanity)]

id_list = f["id"].tolist()

# quick preview of affected sensors before loading
st.caption(f"Selected sensors: {len(id_list)}")

start_dt = dt.datetime.combine(date_from, from_time)
end_dt   = dt.datetime.combine(date_to, to_time) + dt.timedelta(minutes=1)

# ============================
# Query & Anzeige
# ============================
if run:
    if not is_db_configured() and not (
        st.session_state.get("use_local_cache")
        or st.session_state.get("use_cache_only")
    ):
        st.error(
            "Keine DB-Konfiguration gefunden. Ohne konfigurierten Datenbankzugang "
            "sind nur bereits lokal gecachte Daten nutzbar."
        )
        st.stop()

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
                    logs.append(f"Batch {b}: {rows:,} rows ({mb_:.2f} MB) in {dt_s:.2f}s → {rps:,.0f} rows/s".replace(",","."))  # DE-Decimal hack
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

    # Rolling (optional)
    if rolling_window and rolling_window > 0:
        df = df.sort_values(["ID","type","timestamp"])
        df["value"] = (df
                       .groupby(["ID","type"])["value"]
                       .transform(lambda s: s.rolling(rolling_window, min_periods=1).mean())
                      )

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.metric("Data points", f"{len(df):,}".replace(",","."))  # DE-Decimal hack
    with k2:
        st.metric("Sensors", df["ID"].nunique())
    with k3:
        st.metric("Channels", len(set(df["type"].unique().tolist())))
    with k4:
        st.metric("Date range", f"{df['timestamp'].min():%Y-%m-%d} → {df['timestamp'].max():%Y-%m-%d}")

    # Memory footprint of the loaded dataframe in Python
    try:
        _bytes = int(df.memory_usage(deep=True).sum())
        _mb = _bytes / (1024*1024)
        st.caption(f"Loaded ~{_mb:.2f} MB into memory for this view.")
    except Exception:
        pass

    # ============================
    # Charts
    # ============================
    st.subheader("Time series — filtered by Building/Floor/Area")

    def chunked(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    def make_series_labels(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["series"] = frame.apply(
            lambda r: f"{r['type']} | {r.get('vanity','')} ({r['building']} {r['floor']}/{r['area']})",
            axis=1
        )
        return frame

    def post_tune_layout(fig, is_binary: bool):
        if show_rangeslider:
            fig.update_xaxes(rangeslider_visible=True)
        if is_binary:
            # Bei (nahezu) binären Werten Achse klemmen
            fig.update_yaxes(range=[-0.05, 1.05])
            fig.update_traces(line_shape="hv")
        if show_markers:
            fig.update_traces(mode="lines+markers")
        fig.update_layout(legend_title_text="Series")
        return fig

    def plot_group(df_group: pd.DataFrame, title_text: str):
        # Erzeuge Plotly-Express Line
        fig = px.line(
            df_group,
            x="timestamp",
            y="value",
            color="series",
            hover_data=["type","vanity","building","floor","area","nodeTypeName"],
            title=None
        )
        # Binary?
        vals = df_group["value"].dropna()
        is_binary = False
        if not vals.empty:
            unique_vals = set(np.unique(vals.values))
            # toleranter Check: nur 0/1 (oder sehr nahe daran)
            approx = {round(float(v)) for v in unique_vals if np.isfinite(v)}
            if approx.issubset({0,1}) and len(approx) <= 2:
                is_binary = True
        fig = post_tune_layout(fig, is_binary)
        st.subheader(title_text)
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # Vanity-gesplittet oder global
    def plot_all(df_src: pd.DataFrame):
        all_ids = sorted(df_src["ID"].dropna().unique().tolist())
        if not all_ids:
            st.info("No data to plot.")
            return
        for gi, id_chunk in enumerate(chunked(all_ids, sensors_per_chart), start=1):
            df_chunk = df_src[df_src["ID"].isin(id_chunk)].copy()
            df_chunk = make_series_labels(df_chunk)

            if chart_mode == "By channel":
                for ch in sorted(df_chunk["type"].unique()):
                    df_c = df_chunk[df_chunk["type"] == ch].copy()
                    if df_c.empty:
                        continue
                    plot_group(df_c, f"Time series — {ch} (group {gi}/{(len(all_ids)+sensors_per_chart-1)//sensors_per_chart})")
            else:
                plot_group(df_chunk, f"Time series (group {gi}/{(len(all_ids)+sensors_per_chart-1)//sensors_per_chart})")

    if sel_vanity:
        # Pro Vanity ein Abschnitt
        for v in sel_vanity:
            df_v = df[df["vanity"] == v].copy()
            if df_v.empty:
                st.info(f"No data for vanity '{v}' in the selected date range/channels.")
                continue
            plot_all(df_v)
    else:
        # Keine Vanity-Filter: alles zusammen
        plot_all(df)

    # ============================
    # Tabellarische Ansicht & Export
    # ============================
    st.subheader("Table")
    st.dataframe(df.sort_values(["timestamp"]).reset_index(drop=True), use_container_width=True)

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
        - Use **Preload → cache** for ranges you'll reuse; toggle **Use cache only** to verify coverage.

        **Security**
        - ⚠️ Plaintext credentials are convenient but insecure. Prefer secrets/Key Vault in production.
        """
    )
