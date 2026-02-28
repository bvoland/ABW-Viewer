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

# ============================
# Setup & Config
# ============================
st.set_page_config(page_title="Sensor-Explorer", layout="wide")

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
    # Kandidaten-Spalten für nodeTypeName (in DB können Namensvarianten existieren)
    candidates = [
        '"nodeTypeName"',      # exakt quotiert
        'nodeTypeName',         # unquoted
        '"nodetypename"',      # exakt kleingeschrieben quotiert
        'nodetypename',         # unquoted klein
        '"node_type_name"',    # snake_case
        'node_type_name',
        '"nodetype"',
        'nodetype',
        '"type"',
        'type'
    ]

    base_df = None
    last_err = None
    for cand in candidates:
        try:
            q = f"""
                SELECT id,
                       COALESCE({cand}, '') AS nodeTypeName
                FROM nodes_hierarchy
            """
            tmp = pd.read_sql(q, engine)
            # Wenn die Query klappt, übernehmen
            if 'nodeTypeName' in tmp.columns:
                base_df = tmp
                break
        except Exception as e:
            last_err = e
            continue

    if base_df is None:
        # Letzter Versuch: nur id laden und später ohne Typ filtern lassen
        base_df = pd.read_sql("SELECT id FROM nodes_hierarchy", engine)
        base_df['nodeTypeName'] = ''

    # Optionale Spalten nachladen (ohne Fehler abbrechen)
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

    # Strings bereinigen
    for col in ["nodeTypeName","building","floor","area","vanity"]:
        df[col] = df[col].fillna("").astype(str)

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

    # Split für history vs env
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
    # Sicherheit: Datentypen
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    df = df.dropna(subset=["timestamp"])  # ungültige raus

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

        # 4) Meta sollte eine 'name'‑Spalte haben (synthetisch oder echt)
    try:
        meta = load_meta()
        has_name = "vanity" in meta.columns and meta["vanity"].notna().any()
        checks.append(("Meta enthält 'vanity'", bool(has_name), "OK" if has_name else "nicht vorhanden"))
    except Exception as e:
        checks.append(("Meta enthält 'vanity'", False, str(e)))
    return pd.DataFrame(checks, columns=["Check","OK","Details"])

# ============================
# UI — Sidebar Filter
# ============================

st.title("🔎 Sensor-Explorer — SQL → Filter → Charts")
meta = load_meta()

c_top1, c_top2, c_top3 = st.columns([2,1,1])
with c_top1:
    st.caption("Wähle Metadaten & Zeitraum → Daten werden live aus Postgres geladen.")
with c_top2:
    if st.button("🔁 Cache leeren", help="Cache für Meta/Abfragen invalidieren"):
        load_meta.clear(); load_timeseries.clear()
        st.toast("Cache geleert")
with c_top3:
    if st.button("✅ Self‑Test ausführen", help="Kurze DB/Schema/Query‑Checks"):
        res = run_self_tests()
        st.dataframe(res, use_container_width=True)

with st.sidebar:
    st.header("Filter")
    buildings = sorted([b for b in meta["building"].unique() if b])
    floors    = sorted([f for f in meta["floor"].unique() if f])
    rooms     = sorted([r for r in meta["area"].unique() if r])
    types     = sorted([t for t in meta["nodeTypeName"].unique() if t])

    sel_buildings = st.multiselect("Gebäude", buildings, default=buildings[:1] if buildings else [])
    sel_floors    = st.multiselect("Stockwerk", floors)
    sel_rooms     = st.multiselect("Area", rooms)
    sel_types     = st.multiselect("Sensortyp", types, help="z. B. ems_desk, ers_eye, ers_co2 …")

    # Channels je Sensortyp
    SENSOR_CHANNELS = {
        'ems_desk': ['occupancy', 'tempc', 'hum'],
        'ers_eye':  ['occupancy', 'motion', 'light', 'tempc', 'hum'],
        'ers_co2':  ['tempc', 'hum', 'co2'],
        'ers_desk': ['occupancy', 'tempc', 'hum']
    }
    # Alle möglichen Channels aus Auswahl ableiten
    possible_channels = set()
    if sel_types:
        for t in sel_types:
            possible_channels.update(SENSOR_CHANNELS.get(t, []))
    else:
        for lst in SENSOR_CHANNELS.values():
            possible_channels.update(lst)
    possible_channels = sorted(list(possible_channels))

    sel_channels = st.multiselect("Channel", possible_channels, default=possible_channels[:1] if possible_channels else [])

    # Zeitraum
    today = dt.datetime.now().date()
    date_from, date_to = st.date_input(
        "Zeitraum",
        value=(today - dt.timedelta(days=7), today),
        max_value=today
    )
    # Uhrzeit-Feintuning
    c1, c2 = st.columns(2)
    with c1:
        from_time = st.time_input("von", value=dt.time(0,0))
    with c2:
        to_time   = st.time_input("bis", value=dt.time(23,59))

    # Resampling
    agg = st.selectbox("Aggregation (Resampling)", ["keine","15Min","H","D","W","M"], index=2,
                       help="Zwischenwerte mitteln/summieren: 15Min, Stunde (H), Tag (D), Woche (W), Monat (M)")

    agg_func = st.selectbox("Aggregationsfunktion", ["mean","sum","max","min","median"], index=0)

    # Button
    run = st.button("🔍 Daten laden")

# Filter auf IDs anwenden
f = meta.copy()
if sel_buildings:
    f = f[f["building"].isin(sel_buildings)]
if sel_floors:
    f = f[f["floor"].isin(sel_floors)]
if sel_rooms:
    f = f[f["area"].isin(sel_rooms)]
if sel_types:
    f = f[f["nodeTypeName"].isin(sel_types)]

id_list = f["id"].tolist()

# Zeitraum zusammensetzen
start_dt = dt.datetime.combine(date_from, from_time)
end_dt   = dt.datetime.combine(date_to, to_time)

# ============================
# Query & Anzeige
# ============================

if run:
    if not id_list:
        st.warning("Keine Sensoren für die gewählten Filter gefunden.")
        st.stop()
    if not sel_channels:
        st.warning("Bitte mindestens einen Channel wählen.")
        st.stop()

    with st.spinner("Hole Zeitreihen aus Postgres…"):
        df = load_timeseries(id_list, sel_channels, start_dt, end_dt)

    if df.empty:
        st.info("Keine Daten im gewählten Zeitraum/Channel.")
        st.stop()

    # Join Meta für Anzeige
    df = df.merge(meta[["id","vanity","building","floor","area","nodeTypeName"]], left_on="ID", right_on="id", how="left")
    df.drop(columns=["id"], inplace=True)

    # Aggregation optional
    if agg != "keine":
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
        st.metric("Messpunkte", f"{len(df):,}".replace(",","."))
    with k2:
        st.metric("Sensoren", f["id"].nunique())
    with k3:
        st.metric("Channels", len(set(sel_channels)))
    with k4:
        st.metric("Zeitraum", f"{start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d}")

    # Chart — Multi-Line (eine Linie pro (ID, type))
    st.subheader("Zeitreihe — gefiltert nach Gebäude/Stockwerk/Area")

    # Für klare Legende: Label bauen
    df["series"] = df.apply(lambda r: f"{r['type']} | {r.get('vanity','')} ({r['building']} {r['floor']}/{r['area']})", axis=1)

    fig = px.line(
        df,
        x="timestamp",
        y="value",
        color="series",
        hover_data=["type","vanity","building","floor","area","nodeTypeName"],
        title="Zeitverlauf"
    )
    fig.update_layout(legend_title_text="Serie")
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # Tabellarische Ansicht
    st.subheader("Tabelle")
    st.dataframe(df.sort_values(["timestamp"]).reset_index(drop=True), use_container_width=True)

    # Export
    st.download_button(
        "⬇️ CSV exportieren",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"sensor_export_{start_dt:%Y%m%d}_{end_dt:%Y%m%d}.csv",
        mime="text/csv"
    )

# ============================
# Footer / Hilfe
# ============================
with st.expander("Hilfe & Hinweise"):
    st.markdown(
        """
        **Start in VS Code / Terminal**
        - Bitte als Streamlit-App starten, **nicht** mit `python your_script.py`.
        - Kommando:
          ```bash
          streamlit run "Pfad/zur/ABW_Datenvisualisierung.py"
          ```

        **Tipps**
        - Nutze *Resampling* (z. B. `H`), um große Rohdatenmengen zu glätten.
        - Filter zuerst grob (Gebäude → Stockwerk → **Area**), dann Sensortyp & Channel.
        - Falls Felder wie `building/floor/room` in `nodes_hierarchy` fehlen, ist das ok – die App füllt sie leer auf.

        **Performance**
        - `@st.cache_data` und `@st.cache_resource` reduzieren DB-Last.
        - Für riesige Zeiträume lieber voraggregierte Views/Materialized Views nutzen.

        **Sicherheit**
        - ⚠️ Plaintext-Credentials sind bequem, aber unsicher. Nutze sie nur in internen Umgebungen. Für Prod besser: Key Vault/Secrets.
        """
    )
