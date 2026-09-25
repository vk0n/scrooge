from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IndicatorSizingConfig:
    weak_modifier: float = 0.5
    neutral_modifier: float = 1.0
    strong_modifier: float = 1.25
    very_strong_modifier: float = 1.5
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    def __post_init__(self) -> None:
        modifiers = (
            float(self.weak_modifier),
            float(self.neutral_modifier),
            float(self.strong_modifier),
            float(self.very_strong_modifier),
        )
        if any(not math.isfinite(value) or value <= 0 for value in modifiers):
            raise ValueError("Spot indicator sizing modifiers must be finite positive numbers.")
        if any(current < previous for previous, current in zip(modifiers, modifiers[1:])):
            raise ValueError("Spot indicator sizing modifiers must be ordered from weak to very strong.")
        oversold = float(self.rsi_oversold)
        overbought = float(self.rsi_overbought)
        if not 0 < oversold < overbought < 100:
            raise ValueError("Spot RSI sizing thresholds must satisfy 0 < oversold < overbought < 100.")
        object.__setattr__(self, "weak_modifier", modifiers[0])
        object.__setattr__(self, "neutral_modifier", modifiers[1])
        object.__setattr__(self, "strong_modifier", modifiers[2])
        object.__setattr__(self, "very_strong_modifier", modifiers[3])
        object.__setattr__(self, "rsi_oversold", oversold)
        object.__setattr__(self, "rsi_overbought", overbought)


def _finite_number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def apply_indicator_sizing(
    signal_result: dict[str, Any],
    indicator_context: dict[str, Any] | None,
    *,
    config: IndicatorSizingConfig | None = None,
    indicator_error: str | None = None,
) -> dict[str, Any]:
    resolved_config = config or IndicatorSizingConfig()
    opportunity = str(signal_result.get("opportunity") or "").strip().lower()
    if opportunity not in {"hold", "buy", "sell"}:
        raise ValueError("Spot indicator sizing requires a HOLD, BUY, or SELL opportunity.")

    base_tranche_pct = _finite_number(signal_result.get("base_tranche_pct")) or 0.0
    if opportunity == "hold":
        return {
            **signal_result,
            "indicator_status": "not_applicable",
            "indicator_context": indicator_context or {},
            "indicator_assessment": {
                "tier": "not_applicable",
                "confirmations": [],
                "conflicts": [],
                "available_directional_indicators": 0,
                "reason": "Rolling 24H movement did not create a trading opportunity.",
            },
            "sizing_modifier": resolved_config.neutral_modifier,
            "final_tranche_pct": 0.0,
        }
    if base_tranche_pct <= 0:
        raise ValueError("BUY/SELL indicator sizing requires a positive base tranche percentage.")

    context = dict(indicator_context or {})
    current_price = _finite_number(signal_result.get("current_price"))
    if current_price is None or current_price <= 0:
        raise ValueError("Spot indicator sizing requires the signal's positive current price.")

    confirmations: list[str] = []
    conflicts: list[str] = []
    available = 0
    rsi = _finite_number(context.get("rsi"))
    if rsi is not None:
        available += 1
        if (opportunity == "buy" and rsi <= resolved_config.rsi_oversold) or (
            opportunity == "sell" and rsi >= resolved_config.rsi_overbought
        ):
            confirmations.append("rsi")
        elif (opportunity == "buy" and rsi >= resolved_config.rsi_overbought) or (
            opportunity == "sell" and rsi <= resolved_config.rsi_oversold
        ):
            conflicts.append("rsi")

    bb_lower = _finite_number(context.get("bb_lower"))
    bb_upper = _finite_number(context.get("bb_upper"))
    if bb_lower is not None and bb_upper is not None and bb_lower <= bb_upper:
        available += 1
        if (opportunity == "buy" and current_price <= bb_lower) or (
            opportunity == "sell" and current_price >= bb_upper
        ):
            confirmations.append("bollinger")
        elif (opportunity == "buy" and current_price >= bb_upper) or (
            opportunity == "sell" and current_price <= bb_lower
        ):
            conflicts.append("bollinger")

    ema = _finite_number(context.get("ema"))
    if ema is not None and ema > 0:
        available += 1
        if (opportunity == "buy" and current_price < ema) or (
            opportunity == "sell" and current_price > ema
        ):
            confirmations.append("ema")
        elif (opportunity == "buy" and current_price > ema) or (
            opportunity == "sell" and current_price < ema
        ):
            conflicts.append("ema")

    if available == 0 or len(conflicts) > len(confirmations):
        tier = "weak"
        modifier = resolved_config.weak_modifier
    elif len(confirmations) == 3 and not conflicts:
        tier = "very_strong"
        modifier = resolved_config.very_strong_modifier
    elif len(confirmations) >= 2 and not conflicts:
        tier = "strong"
        modifier = resolved_config.strong_modifier
    else:
        tier = "neutral"
        modifier = resolved_config.neutral_modifier

    indicator_status = "ok" if available else "unavailable"
    assessment_reason = indicator_error or (
        f"{len(confirmations)} confirmation(s), {len(conflicts)} conflict(s) "
        f"from {available} directional indicator(s)."
    )
    return {
        **signal_result,
        "indicator_status": indicator_status,
        "indicator_error": str(indicator_error)[:500] if indicator_error else None,
        "indicator_context": context,
        "indicator_assessment": {
            "tier": tier,
            "confirmations": confirmations,
            "conflicts": conflicts,
            "available_directional_indicators": available,
            "reason": assessment_reason,
            "atr_role": "volatility_context_only",
        },
        "sizing_modifier": modifier,
        # The modifier is research telemetry only. Execution always uses the fixed level allocation.
        "final_tranche_pct": base_tranche_pct,
    }
