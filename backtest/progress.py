from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from tqdm import tqdm


class BacktestProgressReporter(Protocol):
    def start_stage(self, label: str, total: int) -> None:
        ...

    def advance(self, amount: int = 1) -> None:
        ...

    def complete_stage(self) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass(slots=True)
class ScenarioProgressBar:
    scenario_name: str
    position: int | None = None
    leave: bool = False
    _bar: tqdm = field(init=False)
    _stage_total: int = field(init=False, default=0)
    _stage_done: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        tqdm_kwargs: dict[str, object] = {
            "total": 0,
            "desc": f"[{self.scenario_name}] Starting",
            "dynamic_ncols": True,
            "leave": self.leave,
        }
        if self.position is not None:
            tqdm_kwargs["position"] = self.position
        self._bar = tqdm(**tqdm_kwargs)
        self._stage_total = 0
        self._stage_done = 0

    def start_stage(self, label: str, total: int) -> None:
        normalized_total = max(0, int(total))
        self._stage_total = normalized_total
        self._stage_done = 0
        self._bar.total = int(self._bar.total or 0) + normalized_total
        self._bar.set_description(f"[{self.scenario_name}] {label}")
        self._bar.refresh()

    def advance(self, amount: int = 1) -> None:
        if amount <= 0:
            return
        self._stage_done += int(amount)
        self._bar.update(int(amount))

    def complete_stage(self) -> None:
        remaining = self._stage_total - self._stage_done
        if remaining > 0:
            self._bar.update(remaining)
        self._stage_done = self._stage_total

    def close(self) -> None:
        self._bar.close()
