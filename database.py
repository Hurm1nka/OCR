# -*- coding: utf-8 -*-
"""SQLite: пользователи, журнал распознаваний, открытия шлагбаума, белый/чёрный списки."""
import os
import sqlite3
import time
import threading
from contextlib import contextmanager

from werkzeug.security import check_password_hash, generate_password_hash

_lock = threading.Lock()


def get_db_path():
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "arm.db")
    return os.environ.get("DATABASE_PATH", default)


@contextmanager
def get_conn():
    path = get_db_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _migrate_schema(conn):
    """Добавление колонок/таблиц для существующих БД."""
    rows = conn.execute("PRAGMA table_info(recognition_events)").fetchall()
    col_names = {r[1] for r in rows}
    if "visit_id" not in col_names:
        conn.execute("ALTER TABLE recognition_events ADD COLUMN visit_id INTEGER")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicle_visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_base TEXT NOT NULL,
            entered_at INTEGER,
            exited_at INTEGER,
            duration_seconds INTEGER,
            exit_without_entry INTEGER NOT NULL DEFAULT 0,
            entry_user_id INTEGER REFERENCES users(id),
            exit_user_id INTEGER REFERENCES users(id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_vehicle_visits_plate ON vehicle_visits(plate_base)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_vehicle_visits_entered ON vehicle_visits(entered_at DESC)"
    )


def init_db():
    with get_conn() as conn:
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'guard',
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS recognition_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at INTEGER NOT NULL,
                user_id INTEGER NOT NULL REFERENCES users(id),
                direction TEXT NOT NULL,
                plate_base TEXT,
                plate_full TEXT,
                region TEXT,
                confidence REAL,
                source TEXT NOT NULL,
                is_duplicate_recent INTEGER NOT NULL DEFAULT 0,
                list_status TEXT NOT NULL DEFAULT 'none',
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS barrier_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at INTEGER NOT NULL,
                user_id INTEGER NOT NULL REFERENCES users(id),
                direction TEXT NOT NULL,
                plate_base TEXT,
                plate_full TEXT,
                note TEXT,
                stub_message TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS plate_lists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                list_type TEXT NOT NULL,
                plate_base TEXT NOT NULL,
                note TEXT,
                created_at INTEGER NOT NULL,
                created_by INTEGER REFERENCES users(id),
                UNIQUE(list_type, plate_base)
            );

            CREATE INDEX IF NOT EXISTS idx_recognition_created ON recognition_events(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_recognition_plate ON recognition_events(plate_base);
            CREATE INDEX IF NOT EXISTS idx_barrier_created ON barrier_actions(created_at DESC);
            """
        )
        _migrate_schema(conn)


def ensure_default_guard_user():
    """Один пользователь по умолчанию, если таблица пуста."""
    initial_password = os.environ.get("INITIAL_GUARD_PASSWORD", "guard123")
    with _lock:
        with get_conn() as conn:
            n = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
            if n > 0:
                return
            h = generate_password_hash(initial_password)
            now = int(time.time())
            conn.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                ("guard", h, "guard", now),
            )


def get_user_by_username(username):
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, username, password_hash, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()


def verify_login(username, password):
    row = get_user_by_username(username)
    if not row:
        return None
    if not check_password_hash(row["password_hash"], password):
        return None
    return {"id": row["id"], "username": row["username"], "role": row["role"]}


def resolve_list_status(plate_base):
    if not plate_base:
        return "none"
    with get_conn() as conn:
        b = conn.execute(
            "SELECT id FROM plate_lists WHERE list_type = 'black' AND plate_base = ?",
            (plate_base,),
        ).fetchone()
        if b:
            return "blacklist"
        w = conn.execute(
            "SELECT id FROM plate_lists WHERE list_type = 'white' AND plate_base = ?",
            (plate_base,),
        ).fetchone()
        if w:
            return "whitelist"
    return "none"


def insert_recognition_event(
    user_id,
    direction,
    plate_base,
    plate_full,
    region,
    confidence,
    source,
    is_duplicate_recent,
    list_status,
    visit_id=None,
):
    now = int(time.time())
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO recognition_events
            (created_at, user_id, direction, plate_base, plate_full, region, confidence, source, is_duplicate_recent, list_status, visit_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                user_id,
                direction,
                plate_base,
                plate_full,
                region,
                float(confidence) if confidence is not None else None,
                source,
                1 if is_duplicate_recent else 0,
                list_status,
                visit_id,
            ),
        )


def get_open_visit(plate_base):
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT id, plate_base, entered_at, exited_at, exit_without_entry
            FROM vehicle_visits
            WHERE plate_base = ? AND exited_at IS NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (plate_base,),
        ).fetchone()


def apply_visit_for_recognition(plate_base, user_id, exit_without_entry):
    """
    Возвращает (direction, visit_id, duration_seconds, exit_without_entry_applied).
    direction: 'entry' | 'exit' | 'exit_no_entry'
    """
    now = int(time.time())
    exit_without_entry = bool(exit_without_entry)
    with get_conn() as conn:
        open_v = conn.execute(
            """
            SELECT id, plate_base, entered_at, exited_at, exit_without_entry
            FROM vehicle_visits
            WHERE plate_base = ? AND exited_at IS NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (plate_base,),
        ).fetchone()

        if open_v:
            # Нормальный выезд (был зафиксирован въезд)
            vid = open_v["id"]
            entered = open_v["entered_at"]
            dur = None
            if entered is not None:
                dur = max(0, now - int(entered))
            conn.execute(
                """
                UPDATE vehicle_visits
                SET exited_at = ?, duration_seconds = ?, exit_user_id = ?
                WHERE id = ?
                """,
                (now, dur, user_id, vid),
            )
            return "exit", vid, dur, False

        if exit_without_entry:
            cur = conn.execute(
                """
                INSERT INTO vehicle_visits
                (plate_base, entered_at, exited_at, duration_seconds, exit_without_entry, exit_user_id)
                VALUES (?, NULL, ?, NULL, 1, ?)
                """,
                (plate_base, now, user_id),
            )
            return "exit_no_entry", cur.lastrowid, None, True

        cur = conn.execute(
            """
            INSERT INTO vehicle_visits
            (plate_base, entered_at, exited_at, duration_seconds, exit_without_entry, entry_user_id)
            VALUES (?, ?, NULL, NULL, 0, ?)
            """,
            (plate_base, now, user_id),
        )
        return "entry", cur.lastrowid, None, False


def insert_barrier_stub_action(user_id, direction, plate_base, plate_full, note, stub_message):
    now = int(time.time())
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO barrier_actions (created_at, user_id, direction, plate_base, plate_full, note, stub_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (now, user_id, direction, plate_base, plate_full, note, stub_message),
        )
        return cur.lastrowid


def add_plate_list_entry(list_type, plate_base, note, created_by_user_id):
    if list_type not in ("white", "black"):
        raise ValueError("list_type must be 'white' or 'black'")
    now = int(time.time())
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO plate_lists (list_type, plate_base, note, created_at, created_by)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(list_type, plate_base) DO UPDATE SET
                note = excluded.note,
                created_at = excluded.created_at,
                created_by = excluded.created_by
            """,
            (list_type, plate_base, note or "", now, created_by_user_id),
        )


def delete_plate_list_entry(entry_id):
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM plate_lists WHERE id = ?", (entry_id,))
        return cur.rowcount


def fetch_plate_list(list_type):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.list_type, p.plate_base, p.note, p.created_at, u.username AS created_by_name
            FROM plate_lists p
            LEFT JOIN users u ON u.id = p.created_by
            WHERE p.list_type = ?
            ORDER BY p.plate_base
            """,
            (list_type,),
        ).fetchall()
        return [dict(r) for r in rows]


def _journal_recognitions_sql(sort, order, q, limit, offset):
    order = "DESC" if str(order).lower() == "desc" else "ASC"
    if sort not in ("created_at", "plate_base", "confidence", "direction", "list_status"):
        sort = "created_at"
    q_like = f"%{q}%" if q else None
    return sort, order, q_like, limit, offset


def _time_clause(from_ts, to_ts, alias="r"):
    if from_ts is None and to_ts is None:
        return "", []
    parts = []
    args = []
    if from_ts is not None:
        parts.append(f"{alias}.created_at >= ?")
        args.append(int(from_ts))
    if to_ts is not None:
        parts.append(f"{alias}.created_at <= ?")
        args.append(int(to_ts))
    return " AND " + " AND ".join(parts), args


def format_duration_human(seconds):
    if seconds is None:
        return "—"
    try:
        sec = int(seconds)
    except (TypeError, ValueError):
        return "—"
    if sec < 0:
        sec = 0
    days, sec = divmod(sec, 86400)
    hours, sec = divmod(sec, 3600)
    mins, sec = divmod(sec, 60)
    parts = []
    if days:
        parts.append(f"{days} д")
    if hours:
        parts.append(f"{hours} ч")
    if mins or sec or not parts:
        parts.append(f"{mins} м")
    return " ".join(parts)


def journal_recognitions(
    sort="created_at", order="desc", q="", limit=100, offset=0, from_ts=None, to_ts=None
):
    sort, order, q_like, limit, offset = _journal_recognitions_sql(sort, order, q, limit, offset)
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    tclause, targs = _time_clause(from_ts, to_ts, "r")
    with get_conn() as conn:
        if q_like:
            rows = conn.execute(
                f"""
                SELECT r.id, r.created_at, r.direction, r.plate_base, r.plate_full, r.region, r.confidence,
                       r.source, r.is_duplicate_recent, r.list_status, r.visit_id, u.username AS operator
                FROM recognition_events r
                JOIN users u ON u.id = r.user_id
                WHERE (r.plate_base LIKE ? OR r.plate_full LIKE ? OR u.username LIKE ?){tclause}
                ORDER BY r.{sort} {order}
                LIMIT ? OFFSET ?
                """,
                (q_like, q_like, q_like, *targs, limit, offset),
            ).fetchall()
            total = conn.execute(
                f"""
                SELECT COUNT(*) AS c FROM recognition_events r
                JOIN users u ON u.id = r.user_id
                WHERE (r.plate_base LIKE ? OR r.plate_full LIKE ? OR u.username LIKE ?){tclause}
                """,
                (q_like, q_like, q_like, *targs),
            ).fetchone()["c"]
        else:
            rows = conn.execute(
                f"""
                SELECT r.id, r.created_at, r.direction, r.plate_base, r.plate_full, r.region, r.confidence,
                       r.source, r.is_duplicate_recent, r.list_status, r.visit_id, u.username AS operator
                FROM recognition_events r
                JOIN users u ON u.id = r.user_id
                WHERE 1=1{tclause}
                ORDER BY r.{sort} {order}
                LIMIT ? OFFSET ?
                """,
                (*targs, limit, offset),
            ).fetchall()
            total = conn.execute(
                f"SELECT COUNT(*) AS c FROM recognition_events r WHERE 1=1{tclause}",
                tuple(targs),
            ).fetchone()["c"]
        return [dict(r) for r in rows], total


def journal_visits(from_ts=None, to_ts=None, q="", limit=200, offset=0):
    """Визиты: пересечение по времени заезда/выезда с интервалом [from_ts, to_ts]."""
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    q_like = f"%{q}%" if (q or "").strip() else None
    args = []
    overlap = "1=1"
    if from_ts is not None or to_ts is not None:
        ft = int(from_ts) if from_ts is not None else 0
        tt = int(to_ts) if to_ts is not None else 2**31 - 1
        overlap = """(
            (v.entered_at IS NOT NULL AND v.entered_at >= ? AND v.entered_at <= ?)
            OR (v.exited_at IS NOT NULL AND v.exited_at >= ? AND v.exited_at <= ?)
            OR (v.exited_at IS NULL AND v.entered_at IS NOT NULL AND v.entered_at >= ? AND v.entered_at <= ?)
        )"""
        args.extend([ft, tt, ft, tt, ft, tt])
    with get_conn() as conn:
        if q_like:
            sql = f"""
                SELECT v.id, v.plate_base, v.entered_at, v.exited_at, v.duration_seconds,
                       v.exit_without_entry
                FROM vehicle_visits v
                WHERE ({overlap}) AND v.plate_base LIKE ?
                ORDER BY COALESCE(v.exited_at, v.entered_at) DESC
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(sql, (*args, q_like, limit, offset)).fetchall()
            cnt = conn.execute(
                f"SELECT COUNT(*) AS c FROM vehicle_visits v WHERE ({overlap}) AND v.plate_base LIKE ?",
                (*args, q_like),
            ).fetchone()["c"]
        else:
            sql = f"""
                SELECT v.id, v.plate_base, v.entered_at, v.exited_at, v.duration_seconds,
                       v.exit_without_entry
                FROM vehicle_visits v
                WHERE {overlap}
                ORDER BY COALESCE(v.exited_at, v.entered_at) DESC
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(sql, (*args, limit, offset)).fetchall()
            cnt = conn.execute(
                f"SELECT COUNT(*) AS c FROM vehicle_visits v WHERE {overlap}",
                tuple(args),
            ).fetchone()["c"]
        out = []
        for r in rows:
            d = dict(r)
            d["duration_human"] = format_duration_human(d.get("duration_seconds"))
            out.append(d)
        return out, cnt


def journal_barrier_actions(
    sort="created_at", order="desc", q="", limit=100, offset=0, from_ts=None, to_ts=None
):
    order = "DESC" if str(order).lower() == "desc" else "ASC"
    if sort not in ("created_at", "plate_base", "direction"):
        sort = "created_at"
    q_like = f"%{q}%" if q else None
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    tclause, targs = _time_clause(from_ts, to_ts, "b")
    with get_conn() as conn:
        if q_like:
            rows = conn.execute(
                f"""
                SELECT b.id, b.created_at, b.direction, b.plate_base, b.plate_full, b.note, b.stub_message, u.username AS operator
                FROM barrier_actions b
                JOIN users u ON u.id = b.user_id
                WHERE (b.plate_base LIKE ? OR b.plate_full LIKE ? OR u.username LIKE ? OR b.note LIKE ?){tclause}
                ORDER BY b.{sort} {order}
                LIMIT ? OFFSET ?
                """,
                (q_like, q_like, q_like, q_like, *targs, limit, offset),
            ).fetchall()
            total = conn.execute(
                f"""
                SELECT COUNT(*) AS c FROM barrier_actions b
                JOIN users u ON u.id = b.user_id
                WHERE (b.plate_base LIKE ? OR b.plate_full LIKE ? OR u.username LIKE ? OR b.note LIKE ?){tclause}
                """,
                (q_like, q_like, q_like, q_like, *targs),
            ).fetchone()["c"]
        else:
            rows = conn.execute(
                f"""
                SELECT b.id, b.created_at, b.direction, b.plate_base, b.plate_full, b.note, b.stub_message, u.username AS operator
                FROM barrier_actions b
                JOIN users u ON u.id = b.user_id
                WHERE 1=1{tclause}
                ORDER BY b.{sort} {order}
                LIMIT ? OFFSET ?
                """,
                (*targs, limit, offset),
            ).fetchall()
            total = conn.execute(
                f"SELECT COUNT(*) AS c FROM barrier_actions b WHERE 1=1{tclause}",
                tuple(targs),
            ).fetchone()["c"]
        return [dict(r) for r in rows], total
