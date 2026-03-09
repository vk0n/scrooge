from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


SYSTEMCTL_BIN = os.getenv("SCROOGE_SYSTEMCTL_BIN", "systemctl")
SERVICE_NAME = os.getenv("SCROOGE_SYSTEMD_SERVICE", "scrooge.service")
SYSTEMCTL_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class ServiceStatus:
    service_name: str
    running: bool
    active_state: str
    sub_state: str
    unit_file_state: str


def _get_systemctl_bin() -> str:
    resolved = shutil.which(SYSTEMCTL_BIN)
    if resolved:
        return resolved
    raise RuntimeError(f"systemctl binary not found: {SYSTEMCTL_BIN}")


def _run_systemctl(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_get_systemctl_bin(), *args],
        capture_output=True,
        text=True,
        timeout=SYSTEMCTL_TIMEOUT_SECONDS,
        check=False,
    )


def _parse_show_output(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def get_service_status(service_name: str | None = None) -> ServiceStatus:
    effective_service_name = service_name or SERVICE_NAME
    result = _run_systemctl(
        [
            "show",
            effective_service_name,
            "--property=ActiveState",
            "--property=SubState",
            "--property=UnitFileState",
            "--no-pager",
        ]
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or "unknown systemctl error"
        raise RuntimeError(f"Failed to read service status for {effective_service_name}: {detail}")

    parsed = _parse_show_output(result.stdout or "")
    active_state = parsed.get("ActiveState", "unknown")
    sub_state = parsed.get("SubState", "unknown")
    unit_file_state = parsed.get("UnitFileState", "unknown")
    return ServiceStatus(
        service_name=effective_service_name,
        running=active_state == "active",
        active_state=active_state,
        sub_state=sub_state,
        unit_file_state=unit_file_state,
    )


def _control_service(action: str, service_name: str | None = None) -> ServiceStatus:
    effective_service_name = service_name or SERVICE_NAME
    result = _run_systemctl([action, effective_service_name])
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or "unknown systemctl error"
        raise RuntimeError(f"Failed to {action} {effective_service_name}: {detail}")
    return get_service_status(effective_service_name)


def start_service(service_name: str | None = None) -> ServiceStatus:
    return _control_service("start", service_name)


def stop_service(service_name: str | None = None) -> ServiceStatus:
    return _control_service("stop", service_name)


def restart_service(service_name: str | None = None) -> ServiceStatus:
    return _control_service("restart", service_name)

