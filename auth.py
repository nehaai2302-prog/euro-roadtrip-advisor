from pathlib import Path
import yaml
import streamlit_authenticator as stauth


CONFIG_PATH = Path(__file__).parent / "auth_config.yaml"


def _default_config():
    return {
        "credentials": {
            "usernames": {
                "user1": {
                    "email": "user1@example.com",
                    "name": "User 1",
                    "password": "ChangeMe123!",
                }
            }
        },
        "cookie": {
            "name": "euroroad_auth",
            "key": "replace_with_secure_cookie_key",
            "expiry_days": 30,
        },
        "preauthorized": {"emails": []},
    }


def _ensure_config():
    if not CONFIG_PATH.exists():
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(_default_config(), f, sort_keys=False)


def _load_config():
    _ensure_config()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or _default_config()

    usernames = config.get("credentials", {}).get("usernames", {})
    updated = False
    for username, user_data in usernames.items():
        pwd = user_data.get("password", "")
        if pwd and not str(pwd).startswith("$2"):
            usernames[username]["password"] = stauth.Hasher([pwd]).generate()[0]
            updated = True

    if updated:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    return config


def get_authenticator():
    config = _load_config()
    return stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )
