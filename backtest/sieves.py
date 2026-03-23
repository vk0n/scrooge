from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backtest.time_windows import resolve_backtest_time_range


@dataclass(slots=True)
class CompareSieve:
    stage_name: str
    name: str
    start_time: str
    end_time: str
    description: str = ""

    @property
    def duration_days(self) -> int:
        return resolve_backtest_time_range(
            backtest_period_days=1,
            backtest_period_start_time=self.start_time,
            backtest_period_end_time=self.end_time,
        ).duration_days


SIEVE_PRESETS: dict[str, list[CompareSieve]] = {
    "btcusdt-three-sieves-v1": [
        CompareSieve(
            stage_name="month",
            name="bull",
            start_time="2021-10-01T00:00:00",
            end_time="2021-10-31T23:59:59",
            description="Strong bullish month with sustained upside and elevated range.",
        ),
        CompareSieve(
            stage_name="month",
            name="bear",
            start_time="2022-11-01T00:00:00",
            end_time="2022-11-30T23:59:59",
            description="Representative bearish month with clear downside pressure.",
        ),
        CompareSieve(
            stage_name="month",
            name="neutral",
            start_time="2025-03-01T00:00:00",
            end_time="2025-03-31T23:59:59",
            description="Sideways-to-mildly-negative month with tradable intramonth swings.",
        ),
        CompareSieve(
            stage_name="quarter",
            name="bull",
            start_time="2024-10-14T00:00:00",
            end_time="2025-01-11T23:59:59",
            description="Representative bullish 90-day upswing.",
        ),
        CompareSieve(
            stage_name="quarter",
            name="bear",
            start_time="2022-03-10T00:00:00",
            end_time="2022-06-07T23:59:59",
            description="Representative bearish 90-day decline.",
        ),
        CompareSieve(
            stage_name="quarter",
            name="neutral",
            start_time="2022-07-03T00:00:00",
            end_time="2022-10-01T23:59:59",
            description="Representative 90-day sideways/choppy regime.",
        ),
        CompareSieve(
            stage_name="half_year",
            name="bull",
            start_time="2024-06-19T00:00:00",
            end_time="2024-12-15T23:59:59",
            description="Representative bullish 180-day expansion.",
        ),
        CompareSieve(
            stage_name="half_year",
            name="bear",
            start_time="2021-11-07T00:00:00",
            end_time="2022-05-05T23:59:59",
            description="Representative bearish 180-day drawdown.",
        ),
        CompareSieve(
            stage_name="half_year",
            name="neutral",
            start_time="2022-08-04T00:00:00",
            end_time="2023-01-30T23:59:59",
            description="Representative 180-day neutral accumulation/chop regime.",
        ),
    ],
    "btcusdt-one-sieve-365-2022-2023-v1": [
        CompareSieve(
            stage_name="year",
            name="mixed",
            start_time="2022-06-01T00:00:00",
            end_time="2023-05-31T23:59:59",
            description=(
                "Single 365-day mixed-regime window spanning the 2022 bear, "
                "sideways bottoming, and early 2023 bull recovery."
            ),
        ),
    ],
}


def resolve_compare_sieves(
    *,
    preset: str | None,
    raw_sieves: Any,
) -> list[CompareSieve]:
    if raw_sieves is not None:
        if not isinstance(raw_sieves, list) or not raw_sieves:
            raise ValueError("sieves must be a non-empty list when provided")
        resolved: list[CompareSieve] = []
        seen: set[str] = set()
        for idx, item in enumerate(raw_sieves, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"sieve #{idx} must be a mapping")
            name = str(item.get("name", "") or "").strip().lower()
            if not name:
                raise ValueError(f"sieve #{idx} must include name")
            if name in seen:
                raise ValueError(f"duplicate sieve name: {name}")
            start_time = str(item.get("start_time", "") or "").strip()
            end_time = str(item.get("end_time", "") or "").strip()
            if not start_time or not end_time:
                raise ValueError(f"sieve {name} must include start_time and end_time")
            resolve_backtest_time_range(
                backtest_period_days=1,
                backtest_period_start_time=start_time,
                backtest_period_end_time=end_time,
            )
            resolved.append(
                CompareSieve(
                    stage_name=str(item.get("stage_name", "custom") or "custom").strip().lower(),
                    name=name,
                    start_time=start_time,
                    end_time=end_time,
                    description=str(item.get("description", "") or "").strip(),
                )
            )
            seen.add(name)
        return resolved

    preset_name = str(preset or "").strip()
    if not preset_name:
        return []
    if preset_name not in SIEVE_PRESETS:
        raise ValueError(
            "Unknown sieve_preset: "
            f"{preset_name}. Available presets: {', '.join(sorted(SIEVE_PRESETS))}"
        )
    return [
        CompareSieve(
            stage_name=item.stage_name,
            name=item.name,
            start_time=item.start_time,
            end_time=item.end_time,
            description=item.description,
        )
        for item in SIEVE_PRESETS[preset_name]
    ]
