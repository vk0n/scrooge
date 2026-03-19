from __future__ import annotations

import logging
import os
import socket
import time
from typing import Any, Callable, TypeVar

import requests
from binance.client import Client
from urllib3.exceptions import MaxRetryError, NameResolutionError, NewConnectionError, ProtocolError


T = TypeVar("T")

DEFAULT_ATTEMPTS = 5
DEFAULT_INITIAL_DELAY_SECONDS = 2.0
DEFAULT_MAX_DELAY_SECONDS = 20.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, default)).strip()
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__context__ is not None:
            current = current.__context__
        else:
            current = None
    return chain


def is_retryable_binance_network_error(exc: BaseException) -> bool:
    retryable_types = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
        MaxRetryError,
        NameResolutionError,
        NewConnectionError,
        ProtocolError,
        socket.gaierror,
        TimeoutError,
    )
    return any(isinstance(item, retryable_types) for item in _iter_exception_chain(exc))


def _resolve_logger(logger: Any | None) -> logging.Logger:
    if isinstance(logger, logging.Logger):
        return logger
    return logging.getLogger("scrooge.bot")


def run_binance_with_retries(
    operation: Callable[[], T],
    *,
    operation_name: str,
    logger: Any | None = None,
    attempts: int | None = None,
    initial_delay_seconds: float | None = None,
    max_delay_seconds: float | None = None,
    backoff_multiplier: float | None = None,
) -> T:
    log = _resolve_logger(logger)
    resolved_attempts = attempts or _env_int("SCROOGE_BINANCE_RETRY_ATTEMPTS", DEFAULT_ATTEMPTS)
    resolved_initial_delay = (
        initial_delay_seconds
        if initial_delay_seconds is not None
        else _env_float("SCROOGE_BINANCE_RETRY_INITIAL_DELAY_SECONDS", DEFAULT_INITIAL_DELAY_SECONDS)
    )
    resolved_max_delay = (
        max_delay_seconds
        if max_delay_seconds is not None
        else _env_float("SCROOGE_BINANCE_RETRY_MAX_DELAY_SECONDS", DEFAULT_MAX_DELAY_SECONDS)
    )
    resolved_backoff = (
        backoff_multiplier
        if backoff_multiplier is not None
        else _env_float("SCROOGE_BINANCE_RETRY_BACKOFF_MULTIPLIER", DEFAULT_BACKOFF_MULTIPLIER)
    )

    delay_seconds = max(0.1, resolved_initial_delay)
    for attempt in range(1, max(1, resolved_attempts) + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            retryable = is_retryable_binance_network_error(exc)
            if not retryable or attempt >= resolved_attempts:
                raise

            log.warning(
                "binance_retry_scheduled operation=%s attempt=%s/%s retry_in_seconds=%.2f error=%s",
                operation_name,
                attempt,
                resolved_attempts,
                delay_seconds,
                exc,
            )
            time.sleep(delay_seconds)
            delay_seconds = min(resolved_max_delay, delay_seconds * max(1.0, resolved_backoff))

    raise RuntimeError(f"unreachable retry loop for {operation_name}")


def create_binance_client(
    api_key: str | None,
    api_secret: str | None,
    *,
    logger: Any | None = None,
) -> Client:
    return run_binance_with_retries(
        lambda: Client(api_key, api_secret),
        operation_name="binance_client_init",
        logger=logger,
    )
