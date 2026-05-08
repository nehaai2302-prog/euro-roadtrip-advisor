import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional


DB_PATH = Path(__file__).parent / "app_state.sqlite3"


def _connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT PRIMARY KEY,
            traveler_type TEXT,
            trip_style TEXT,
            avoid_tolls INTEGER,
            avoid_highways INTEGER,
            short_drives INTEGER,
            highways INTEGER,
            updated_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS geocode_cache (
            city TEXT PRIMARY KEY,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS route_cache (
            cache_key TEXT PRIMARY KEY,
            response_json TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reverse_geo_cache (
            lat_round REAL NOT NULL,
            lon_round REAL NOT NULL,
            country_code TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (lat_round, lon_round)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS password_resets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token_hash TEXT UNIQUE NOT NULL,
            expires_at TEXT NOT NULL,
            used_at TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS last_route_context (
            user_id TEXT PRIMARY KEY,
            start_city TEXT NOT NULL,
            end_city TEXT NOT NULL,
            routing_mode TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def load_last_route(user_id: Optional[str]) -> Optional[dict]:
    """Persisted start/end cities for toll/fuel follow-ups (per logged-in user or guest id)."""
    if not user_id:
        return None
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT start_city, end_city, routing_mode
        FROM last_route_context
        WHERE user_id = ?
        """,
        (user_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "start_city": row["start_city"],
        "end_city": row["end_city"],
        "routing_mode": row["routing_mode"] or "fast",
    }


def save_last_route(
    user_id: Optional[str],
    start_city: Optional[str],
    end_city: Optional[str],
    routing_mode: Optional[str] = None,
) -> None:
    if not user_id or not start_city or not end_city:
        return
    rm = routing_mode if routing_mode in ("short", "fast") else "fast"
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO last_route_context (user_id, start_city, end_city, routing_mode, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            start_city=excluded.start_city,
            end_city=excluded.end_city,
            routing_mode=excluded.routing_mode,
            updated_at=excluded.updated_at
        """,
        (
            user_id,
            str(start_city).strip(),
            str(end_city).strip(),
            rm,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def load_history(user_id: str, limit: int = 100):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT role, content
        FROM conversations
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def save_message(user_id: str, role: str, content: str):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO conversations (user_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, role, content, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def load_preferences(user_id: str):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "traveler_type": row["traveler_type"],
        "trip_style": row["trip_style"],
        "preferences": {
            "avoid_tolls": bool(row["avoid_tolls"]),
            "avoid_highways": bool(row["avoid_highways"]),
            "short_drives": bool(row["short_drives"]),
            "highways": bool(row["highways"]),
        },
    }


def save_preferences(user_id: str, trip_context: dict):
    prefs = trip_context.get("preferences", {})
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO user_preferences (
            user_id, traveler_type, trip_style, avoid_tolls, avoid_highways, short_drives, highways, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            traveler_type=excluded.traveler_type,
            trip_style=excluded.trip_style,
            avoid_tolls=excluded.avoid_tolls,
            avoid_highways=excluded.avoid_highways,
            short_drives=excluded.short_drives,
            highways=excluded.highways,
            updated_at=excluded.updated_at
        """,
        (
            user_id,
            trip_context.get("traveler_type"),
            trip_context.get("trip_style"),
            int(bool(prefs.get("avoid_tolls"))),
            int(bool(prefs.get("avoid_highways"))),
            int(bool(prefs.get("short_drives"))),
            int(bool(prefs.get("highways"))),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_geocode(city: str):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT lat, lon FROM geocode_cache WHERE city = ?", (city.lower(),))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return row["lat"], row["lon"]


def get_reverse_geo_country(lat_round: float, lon_round: float):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT country_code FROM reverse_geo_cache WHERE lat_round = ? AND lon_round = ?",
        (lat_round, lon_round),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    code = row["country_code"]
    if code == "":
        return ""
    return code


def set_reverse_geo_country(lat_round: float, lon_round: float, country_code: str):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO reverse_geo_cache (lat_round, lon_round, country_code, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(lat_round, lon_round) DO UPDATE SET
            country_code=excluded.country_code,
            updated_at=excluded.updated_at
        """,
        (lat_round, lon_round, country_code or "", datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def set_geocode(city: str, lat: float, lon: float):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO geocode_cache (city, lat, lon, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(city) DO UPDATE SET
            lat=excluded.lat, lon=excluded.lon, updated_at=excluded.updated_at
        """,
        (city.lower(), lat, lon, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_route_cache(cache_key: str):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT response_json, expires_at FROM route_cache WHERE cache_key = ?",
        (cache_key,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
        return None
    return json.loads(row["response_json"])


def set_route_cache(cache_key: str, payload: dict, ttl_days: int = 7):
    expires_at = (datetime.utcnow() + timedelta(days=ttl_days)).isoformat()
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO route_cache (cache_key, response_json, expires_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
            response_json=excluded.response_json,
            expires_at=excluded.expires_at,
            updated_at=excluded.updated_at
        """,
        (cache_key, json.dumps(payload), expires_at, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_user_by_username(username: str):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return row


def create_user(username: str, email: str, password_hash: str, salt: str):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO users (username, email, password_hash, salt, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (username, email, password_hash, salt, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_user_by_identifier(identifier: str):
    normalized = identifier.strip()
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM users
        WHERE username = ? OR email = ?
        """,
        (normalized, normalized.lower()),
    )
    row = cur.fetchone()
    conn.close()
    return row


def update_user_password(user_id: int, password_hash: str, salt: str):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE users
        SET password_hash = ?, salt = ?
        WHERE id = ?
        """,
        (password_hash, salt, user_id),
    )
    conn.commit()
    conn.close()


def create_password_reset(user_id: int, token_hash: str, expires_at: str):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO password_resets (user_id, token_hash, expires_at, used_at, created_at)
        VALUES (?, ?, ?, NULL, ?)
        """,
        (user_id, token_hash, expires_at, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_password_reset_by_token(token_hash: str):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, user_id, expires_at, used_at
        FROM password_resets
        WHERE token_hash = ?
        """,
        (token_hash,),
    )
    row = cur.fetchone()
    conn.close()
    return row


def mark_password_reset_used(reset_id: int):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE password_resets
        SET used_at = ?
        WHERE id = ?
        """,
        (datetime.utcnow().isoformat(), reset_id),
    )
    conn.commit()
    conn.close()


def invalidate_user_password_resets(user_id: int, keep_reset_id: Optional[int] = None):
    conn = _connect()
    cur = conn.cursor()
    if keep_reset_id is None:
        cur.execute(
            """
            UPDATE password_resets
            SET used_at = COALESCE(used_at, ?)
            WHERE user_id = ? AND used_at IS NULL
            """,
            (datetime.utcnow().isoformat(), user_id),
        )
    else:
        cur.execute(
            """
            UPDATE password_resets
            SET used_at = COALESCE(used_at, ?)
            WHERE user_id = ? AND used_at IS NULL AND id != ?
            """,
            (datetime.utcnow().isoformat(), user_id, keep_reset_id),
        )
    conn.commit()
    conn.close()
