import datetime as dt
import io
import json
import os
import sys
from typing import Iterable

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


SOURCE_TABLES = ("nodes_hierarchy", "nodes_history", "nodes_history_env")
FULL_REFRESH_TABLES = {"nodes_hierarchy"}
INCREMENTAL_TABLES = {"nodes_history", "nodes_history_env"}
BASELINE_TS = dt.datetime(1900, 1, 1)


def log(message: str) -> None:
    print(f"[{dt.datetime.utcnow().isoformat()}Z] {message}", flush=True)


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_source_settings() -> dict:
    config_path = require_env("ABW_SOURCE_CONFIG")
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return {
        "user": str(cfg["user"]).strip(),
        "password": str(cfg["password"]).strip(),
        "host": str(cfg["host"]).strip(),
        "port": int(str(cfg["port"]).strip()),
        "name": str(cfg["name"]).strip(),
    }


def load_target_settings() -> dict:
    return {
        "user": require_env("ABW_TARGET_USER"),
        "password": require_env("ABW_TARGET_PASS"),
        "host": require_env("ABW_TARGET_HOST"),
        "port": int(require_env("ABW_TARGET_PORT")),
        "name": require_env("ABW_TARGET_NAME"),
    }


def build_url(settings: dict) -> URL:
    return URL.create(
        "postgresql+psycopg2",
        username=settings["user"],
        password=settings["password"],
        host=settings["host"],
        port=settings["port"],
        database=settings["name"],
    )


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def qname(schema: str, table: str) -> str:
    return f"{quote_ident(schema)}.{quote_ident(table)}"


def fetch_columns(source_engine, table: str) -> list[dict]:
    query = text(
        """
        SELECT
            column_name,
            data_type,
            udt_name,
            is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = :table_name
        ORDER BY ordinal_position
        """
    )
    with source_engine.connect() as conn:
        rows = conn.execute(query, {"table_name": table}).mappings().all()
    if not rows:
        raise RuntimeError(f"Source table public.{table} not found")
    return [dict(row) for row in rows]


def map_data_type(column: dict) -> str:
    data_type = str(column["data_type"]).lower()
    udt_name = str(column["udt_name"]).lower()
    if data_type == "timestamp without time zone":
        return "TIMESTAMP WITHOUT TIME ZONE"
    if data_type == "timestamp with time zone":
        return "TIMESTAMP WITH TIME ZONE"
    if data_type == "character varying":
        return "TEXT"
    if data_type == "text":
        return "TEXT"
    if data_type == "boolean":
        return "BOOLEAN"
    if data_type == "bigint" or udt_name == "int8":
        return "BIGINT"
    if data_type == "integer" or udt_name == "int4":
        return "INTEGER"
    if data_type == "smallint" or udt_name == "int2":
        return "SMALLINT"
    if data_type == "double precision" or udt_name == "float8":
        return "DOUBLE PRECISION"
    if data_type == "real" or udt_name == "float4":
        return "REAL"
    if data_type == "numeric":
        return "NUMERIC"
    return "TEXT"


def ensure_target_layout(target_engine, source_columns_by_table: dict[str, list[dict]]) -> None:
    statements: list[str] = [
        "CREATE SCHEMA IF NOT EXISTS abw_app",
        """
        CREATE TABLE IF NOT EXISTS abw_app.sync_runs (
            id BIGSERIAL PRIMARY KEY,
            table_name TEXT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            finished_at TIMESTAMPTZ,
            mode TEXT NOT NULL,
            rows_loaded BIGINT NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'running',
            details TEXT
        )
        """,
    ]
    for table, columns in source_columns_by_table.items():
        defs = []
        for col in columns:
            nullable = "" if str(col["is_nullable"]).upper() == "YES" else " NOT NULL"
            defs.append(f"{quote_ident(col['column_name'])} {map_data_type(col)}{nullable}")
        statements.append(f"CREATE TABLE IF NOT EXISTS {qname('public', table)} ({', '.join(defs)})")

    statements.extend(
        [
            f'CREATE UNIQUE INDEX IF NOT EXISTS ix_nodes_hierarchy_id ON {qname("public", "nodes_hierarchy")} ("id")',
            f'CREATE INDEX IF NOT EXISTS ix_nodes_history_id_type_ts ON {qname("public", "nodes_history")} ("ID", "type", "timestamp")',
            f'CREATE INDEX IF NOT EXISTS ix_nodes_history_env_id_type_ts ON {qname("public", "nodes_history_env")} ("ID", "type", "timestamp")',
        ]
    )
    with target_engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))


def column_names(columns: Iterable[dict]) -> list[str]:
    return [str(col["column_name"]) for col in columns]


def select_sql(table: str, columns: list[dict], where_sql: str = "", order_sql: str = "") -> str:
    quoted = ", ".join(quote_ident(col["column_name"]) for col in columns)
    base = f"SELECT {quoted} FROM {qname('public', table)}"
    if where_sql:
        base += f" WHERE {where_sql}"
    if order_sql:
        base += f" ORDER BY {order_sql}"
    return base


def copy_dataframe(target_engine, schema: str, table: str, columns: list[str], df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, na_rep="\\N")
    buffer.seek(0)
    copy_sql = (
        f"COPY {qname(schema, table)} ({', '.join(quote_ident(col) for col in columns)}) "
        "FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
    )
    raw_conn = target_engine.raw_connection()
    try:
        cur = raw_conn.cursor()
        cur.copy_expert(copy_sql, buffer)
        raw_conn.commit()
    finally:
        raw_conn.close()
    return len(df)


def begin_run(target_engine, table: str, mode: str) -> int:
    with target_engine.begin() as conn:
        run_id = conn.execute(
            text(
                """
                INSERT INTO abw_app.sync_runs (table_name, mode)
                VALUES (:table_name, :mode)
                RETURNING id
                """
            ),
            {"table_name": table, "mode": mode},
        ).scalar_one()
    return int(run_id)


def finish_run(target_engine, run_id: int, rows_loaded: int, status: str, details: str = "") -> None:
    with target_engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE abw_app.sync_runs
                SET finished_at = NOW(),
                    rows_loaded = :rows_loaded,
                    status = :status,
                    details = :details
                WHERE id = :run_id
                """
            ),
            {
                "run_id": run_id,
                "rows_loaded": rows_loaded,
                "status": status,
                "details": details[:4000],
            },
        )


def sync_full_table(source_engine, target_engine, table: str, columns: list[dict], chunk_size: int) -> None:
    cols = column_names(columns)
    run_id = begin_run(target_engine, table, "full")
    row_count = 0
    try:
        log(f"Full refresh for {table} started")
        with target_engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {qname('public', table)}"))
        query = text(select_sql(table, columns, order_sql=quote_ident(cols[0])))
        chunks = pd.read_sql(query, source_engine, chunksize=chunk_size)
        for chunk in chunks:
            row_count += copy_dataframe(target_engine, "public", table, cols, chunk)
            log(f"{table}: loaded {row_count} rows")
        with target_engine.begin() as conn:
            conn.execute(text(f"ANALYZE {qname('public', table)}"))
        finish_run(target_engine, run_id, row_count, "success")
        log(f"Full refresh for {table} finished with {row_count} rows")
    except Exception as exc:
        finish_run(target_engine, run_id, row_count, "failed", str(exc))
        raise


def sync_incremental_table(
    source_engine,
    target_engine,
    table: str,
    columns: list[dict],
    chunk_size: int,
    overlap_hours: int,
) -> None:
    cols = column_names(columns)
    run_id = begin_run(target_engine, table, "incremental")
    row_count = 0
    try:
        with target_engine.connect() as conn:
            max_ts = conn.execute(
                text(f'SELECT MAX("timestamp") FROM {qname("public", table)}')
            ).scalar()
        if max_ts is None:
            cutoff = BASELINE_TS
        else:
            cutoff = max_ts - dt.timedelta(hours=overlap_hours)

        log(f"Incremental sync for {table} started from cutoff {cutoff.isoformat(sep=' ')}")
        with target_engine.begin() as conn:
            conn.execute(
                text(f'DELETE FROM {qname("public", table)} WHERE "timestamp" >= :cutoff'),
                {"cutoff": cutoff},
            )

        query = text(
            select_sql(
                table,
                columns,
                where_sql=f'{quote_ident("timestamp")} >= :cutoff',
                order_sql=quote_ident("timestamp"),
            )
        )
        chunks = pd.read_sql(query, source_engine, params={"cutoff": cutoff}, chunksize=chunk_size)
        for chunk in chunks:
            row_count += copy_dataframe(target_engine, "public", table, cols, chunk)
            log(f"{table}: loaded {row_count} rows since cutoff")
        with target_engine.begin() as conn:
            conn.execute(text(f"ANALYZE {qname('public', table)}"))
        finish_run(target_engine, run_id, row_count, "success", f"cutoff={cutoff.isoformat()}")
        log(f"Incremental sync for {table} finished with {row_count} rows")
    except Exception as exc:
        finish_run(target_engine, run_id, row_count, "failed", str(exc))
        raise


def main() -> int:
    source_engine = create_engine(build_url(load_source_settings()), pool_pre_ping=True)
    target_engine = create_engine(build_url(load_target_settings()), pool_pre_ping=True)
    chunk_size = int(os.getenv("ABW_SYNC_CHUNK_SIZE", "50000"))
    overlap_hours = int(os.getenv("ABW_SYNC_OVERLAP_HOURS", "336"))

    source_columns_by_table = {table: fetch_columns(source_engine, table) for table in SOURCE_TABLES}
    ensure_target_layout(target_engine, source_columns_by_table)

    for table in SOURCE_TABLES:
        if table in FULL_REFRESH_TABLES:
            sync_full_table(source_engine, target_engine, table, source_columns_by_table[table], chunk_size)
        elif table in INCREMENTAL_TABLES:
            sync_incremental_table(
                source_engine,
                target_engine,
                table,
                source_columns_by_table[table],
                chunk_size,
                overlap_hours,
            )
        else:
            raise RuntimeError(f"No sync mode configured for table: {table}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"Sync failed: {exc}")
        raise
