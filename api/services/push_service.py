from __future__ import annotations

import base64
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
except ImportError:  # pragma: no cover - dependency installed in runtime images
    serialization = None
    ec = None


logger = logging.getLogger("scrooge.push")
PUSH_ENABLED = os.getenv("SCROOGE_PUSH_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
PUSH_SUBJECT = os.getenv("SCROOGE_PUSH_VAPID_SUBJECT", "mailto:scrooge@example.local").strip() or "mailto:scrooge@example.local"


def _runtime_dir() -> Path:
    for env_name in (
        "SCROOGE_STATE_FILE",
        "SCROOGE_STATE_PATH",
        "SCROOGE_LOG_FILE",
        "SCROOGE_LOG_PATH",
    ):
        raw_value = os.getenv(env_name, "").strip()
        if raw_value:
            return Path(raw_value).expanduser().parent
    return Path.cwd()


SUBSCRIPTIONS_PATH = Path(
    os.getenv("SCROOGE_PUSH_SUBSCRIPTIONS_FILE", str(_runtime_dir() / "push_subscriptions.json"))
).expanduser()
PRIVATE_KEY_PATH = Path(
    os.getenv("SCROOGE_PUSH_VAPID_PRIVATE_KEY_FILE", str(_runtime_dir() / "push_vapid_private.pem"))
).expanduser()
PUBLIC_KEY_PATH = Path(
    os.getenv("SCROOGE_PUSH_VAPID_PUBLIC_KEY_FILE", str(_runtime_dir() / "push_vapid_public.txt"))
).expanduser()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _urlsafe_b64encode(raw_bytes: bytes) -> str:
    return base64.urlsafe_b64encode(raw_bytes).rstrip(b"=").decode("ascii")


def _urlsafe_b64decode(raw_text: str) -> bytes:
    padding = "=" * ((4 - len(raw_text) % 4) % 4)
    return base64.urlsafe_b64decode(raw_text + padding)


def _subscription_record(subscription: dict[str, Any], requested_by: str | None = None) -> dict[str, Any]:
    keys = subscription.get("keys") if isinstance(subscription.get("keys"), dict) else {}
    endpoint = str(subscription.get("endpoint", "")).strip()
    p256dh = str(keys.get("p256dh", "")).strip()
    auth = str(keys.get("auth", "")).strip()
    if not endpoint or not p256dh or not auth:
        raise ValueError("Push subscription must include endpoint, keys.p256dh, and keys.auth")
    return {
        "endpoint": endpoint,
        "keys": {
            "p256dh": p256dh,
            "auth": auth,
        },
        "requested_by": requested_by or "unknown",
        "updated_at": _now_iso(),
    }


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("push_json_read_failed path=%s error=%s", path, exc)
        return default


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _load_private_key_from_text(raw_text: str):
    if serialization is None:
        raise RuntimeError("cryptography is not installed")
    normalized = raw_text.strip()
    if not normalized:
        raise ValueError("Empty VAPID private key")
    try:
        return serialization.load_pem_private_key(normalized.encode("utf-8"), password=None)
    except ValueError:
        der_bytes = _urlsafe_b64decode(normalized)
        return serialization.load_der_private_key(der_bytes, password=None)


def _public_key_from_private_key(private_key: Any) -> str:
    public_key = private_key.public_key()
    numbers = public_key.public_numbers()
    public_bytes = b"\x04" + numbers.x.to_bytes(32, "big") + numbers.y.to_bytes(32, "big")
    return _urlsafe_b64encode(public_bytes)


def _persist_generated_keys(private_key: Any, public_key_text: str) -> None:
    if serialization is None:
        raise RuntimeError("cryptography is not installed")
    PRIVATE_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    PRIVATE_KEY_PATH.write_bytes(private_pem)
    PUBLIC_KEY_PATH.write_text(public_key_text, encoding="utf-8")


def get_vapid_config() -> dict[str, str | None]:
    if serialization is None or ec is None:
        return {
            "private_key_path": None,
            "public_key": None,
            "subject": PUSH_SUBJECT,
        }

    env_private = os.getenv("SCROOGE_PUSH_VAPID_PRIVATE_KEY", "").strip()
    env_public = os.getenv("SCROOGE_PUSH_VAPID_PUBLIC_KEY", "").strip()

    private_key = None
    public_key = env_public or None

    if env_private:
        private_key = _load_private_key_from_text(env_private)
        if public_key is None:
            public_key = _public_key_from_private_key(private_key)
        _persist_generated_keys(private_key, public_key)
        return {
            "private_key_path": str(PRIVATE_KEY_PATH),
            "public_key": public_key,
            "subject": PUSH_SUBJECT,
        }

    if PRIVATE_KEY_PATH.exists():
        private_key = _load_private_key_from_text(PRIVATE_KEY_PATH.read_text(encoding="utf-8"))
        if public_key is None:
            if PUBLIC_KEY_PATH.exists():
                public_key = PUBLIC_KEY_PATH.read_text(encoding="utf-8").strip() or None
            if public_key is None:
                public_key = _public_key_from_private_key(private_key)
                PUBLIC_KEY_PATH.write_text(public_key, encoding="utf-8")
        return {
            "private_key_path": str(PRIVATE_KEY_PATH),
            "public_key": public_key,
            "subject": PUSH_SUBJECT,
        }

    if not PUSH_ENABLED:
        return {
            "private_key_path": None,
            "public_key": None,
            "subject": PUSH_SUBJECT,
        }

    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = _public_key_from_private_key(private_key)
    _persist_generated_keys(private_key, public_key)
    logger.info("push_vapid_keys_generated private=%s public=%s", PRIVATE_KEY_PATH, PUBLIC_KEY_PATH)
    return {
        "private_key_path": str(PRIVATE_KEY_PATH),
        "public_key": public_key,
        "subject": PUSH_SUBJECT,
    }


def get_push_status() -> dict[str, Any]:
    config = get_vapid_config()
    subscriptions = load_push_subscriptions()
    return {
        "enabled": PUSH_ENABLED,
        "configured": bool(config.get("public_key")),
        "public_key": config.get("public_key"),
        "subject": config.get("subject"),
        "subscription_count": len(subscriptions),
    }


def load_push_subscriptions() -> list[dict[str, Any]]:
    raw_rows = _load_json(SUBSCRIPTIONS_PATH, default=[])
    if not isinstance(raw_rows, list):
        return []
    output: list[dict[str, Any]] = []
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        try:
            output.append(_subscription_record(item, requested_by=item.get("requested_by")))
        except ValueError:
            continue
    return output


def upsert_push_subscription(subscription: dict[str, Any], requested_by: str | None = None) -> dict[str, Any]:
    record = _subscription_record(subscription, requested_by=requested_by)
    subscriptions = load_push_subscriptions()
    for existing in subscriptions:
        if existing["endpoint"] == record["endpoint"]:
            existing["keys"] = record["keys"]
            existing["requested_by"] = record["requested_by"]
            existing["updated_at"] = record["updated_at"]
            _save_json(SUBSCRIPTIONS_PATH, subscriptions)
            return existing

    subscriptions.append(
        {
            **record,
            "created_at": record["updated_at"],
        }
    )
    _save_json(SUBSCRIPTIONS_PATH, subscriptions)
    return subscriptions[-1]


def remove_push_subscription(endpoint: str) -> bool:
    normalized_endpoint = str(endpoint or "").strip()
    if not normalized_endpoint:
        raise ValueError("Subscription endpoint is required")

    subscriptions = load_push_subscriptions()
    filtered = [item for item in subscriptions if item.get("endpoint") != normalized_endpoint]
    removed = len(filtered) != len(subscriptions)
    if removed:
        _save_json(SUBSCRIPTIONS_PATH, filtered)
    return removed


def build_notification_payload(event: dict[str, Any]) -> dict[str, Any]:
    category = str(event.get("category") or "system").strip().lower()
    code = str(event.get("code") or "").strip().lower()
    title_map = {
        "trade": "Scrooge Trade Update",
        "risk": "Scrooge Risk Alert",
        "error": "Scrooge Trouble",
        "lifecycle": "Scrooge Office Update",
        "command": "Scrooge Order Update",
    }
    target_url = "/logs" if category in {"error"} or code == "command_failed" else "/dashboard"
    return {
        "title": title_map.get(category, "Scrooge Control"),
        "body": str(event.get("ui_message") or "Scrooge has an update."),
        "tag": code or f"scrooge-{category}",
        "url": target_url,
        "code": code,
        "category": category,
        "timestamp": event.get("ts"),
    }


def send_test_push(subscription: dict[str, Any]) -> bool:
    payload = {
        "title": "Scrooge Control",
        "body": "The bell is wired. I shall send word when something important happens.",
        "tag": "scrooge-test",
        "url": "/dashboard",
        "code": "push_test",
        "category": "system",
        "timestamp": _now_iso(),
    }
    return _send_to_single_subscription(subscription, payload)


def _send_to_single_subscription(subscription: dict[str, Any], payload: dict[str, Any]) -> bool:
    config = get_vapid_config()
    private_key_path = config.get("private_key_path")
    public_key = config.get("public_key")
    if not PUSH_ENABLED or not private_key_path or not public_key:
        return False

    try:
        from pywebpush import WebPushException, webpush
    except ImportError:  # pragma: no cover - dependency installed in runtime images
        logger.warning("push_send_skipped reason=pywebpush_missing")
        return False

    try:
        webpush(
            subscription_info={
                "endpoint": subscription["endpoint"],
                "keys": subscription["keys"],
            },
            data=json.dumps(payload),
            vapid_private_key=private_key_path,
            vapid_claims={"sub": str(config.get("subject") or PUSH_SUBJECT)},
        )
        return True
    except WebPushException as exc:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code in {404, 410}:
            try:
                remove_push_subscription(subscription["endpoint"])
            except ValueError:
                pass
        logger.warning(
            "push_send_failed endpoint=%s status=%s error=%s",
            subscription.get("endpoint"),
            status_code,
            exc,
        )
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("push_send_failed endpoint=%s error=%s", subscription.get("endpoint"), exc)
        return False


def dispatch_event_push(event: dict[str, Any]) -> None:
    if not PUSH_ENABLED or not bool(event.get("notify")):
        return

    subscriptions = load_push_subscriptions()
    if not subscriptions:
        return

    payload = build_notification_payload(event)
    for subscription in subscriptions:
        _send_to_single_subscription(subscription, payload)
