import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta


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
