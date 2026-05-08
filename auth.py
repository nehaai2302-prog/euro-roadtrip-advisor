import hashlib
import hmac
import secrets
from datetime import datetime, timedelta

from db import (
    get_user_by_username,
    get_user_by_identifier,
    create_user,
    create_password_reset,
    get_password_reset_by_token,
    update_user_password,
    mark_password_reset_used,
    invalidate_user_password_resets,
)

def _hash_password(password: str, salt: str) -> str:
    raw = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 150_000)
    return raw.hex()


def register_user(username: str, email: str, password: str):
    username = username.strip()
    email = email.strip().lower()
    if not username or not email or not password:
        return False, "Please fill in username, email, and password."
    if get_user_by_username(username):
        return False, "Username already exists."

    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    try:
        create_user(username, email, password_hash, salt)
        return True, "Account created. You can now log in."
    except Exception:
        return False, "Could not create account. Email may already be in use."


def authenticate_user(username: str, password: str):
    user = get_user_by_username(username.strip())
    if not user:
        return False
    expected_hash = user["password_hash"]
    candidate_hash = _hash_password(password, user["salt"])
    return hmac.compare_digest(expected_hash, candidate_hash)


def _hash_reset_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def request_password_reset(identifier: str):
    """
    Creates a one-time reset token. Returns a generic message regardless of account existence.
    The raw token is returned for local/dev flows where email delivery is not configured.
    """
    generic_msg = "If this account exists, password reset instructions have been generated."
    if not identifier or not identifier.strip():
        return False, "Please provide your username or email.", None

    user = get_user_by_identifier(identifier)
    if not user:
        return True, generic_msg, None

    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash_reset_token(raw_token)
    expires_at = (datetime.utcnow() + timedelta(minutes=30)).isoformat()
    create_password_reset(user["id"], token_hash, expires_at)
    return True, generic_msg, raw_token


def reset_password_with_token(token: str, new_password: str, confirm_password: str):
    if not token or not new_password or not confirm_password:
        return False, "Please fill in token, new password, and confirm password."
    if new_password != confirm_password:
        return False, "Passwords do not match."
    if len(new_password) < 8:
        return False, "Password must be at least 8 characters long."

    token_hash = _hash_reset_token(token.strip())
    reset_row = get_password_reset_by_token(token_hash)
    if not reset_row:
        return False, "Invalid or expired reset token."
    if reset_row["used_at"]:
        return False, "This reset token has already been used."
    if datetime.fromisoformat(reset_row["expires_at"]) < datetime.utcnow():
        return False, "This reset token has expired."

    salt = secrets.token_hex(16)
    password_hash = _hash_password(new_password, salt)
    update_user_password(reset_row["user_id"], password_hash, salt)
    mark_password_reset_used(reset_row["id"])
    invalidate_user_password_resets(reset_row["user_id"], keep_reset_id=reset_row["id"])
    return True, "Password reset successful. You can now log in."
