import os
import time
import pandas as pd
from datetime import datetime
from backtest.time_windows import resolve_backtest_time_range
from bot.event_log import get_technical_logger
from core.binance_retry import run_binance_with_retries
from core.feature_engine import FeatureEngine, build_feature_frame
from core.indicator_inputs import INDICATOR_COLUMNS, normalize_indicator_inputs, uses_realtime_indicator_inputs


REQUIRED_INDICATOR_COLUMNS = ("BBL", "BBM", "BBU", "ATR", "RSI", "EMA")

_client = None
technical_logger = get_technical_logger()


def set_client(client):
    global _client
    _client = client


def _get_client():
    if _client is None:
        raise ValueError("Binance client not initialized. Call set_client().")
    return _client


def fetch_historical(symbol="BTCUSDT", interval="15m", limit=500):
    """Fetch historical klines from Binance Futures (include high/low)."""
    client = _get_client()
    raw = run_binance_with_retries(
        lambda: client.futures_klines(symbol=symbol, interval=interval, limit=limit),
        operation_name=f"futures_klines symbol={symbol} interval={interval} limit={limit}",
        logger=technical_logger,
    )
    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume","close_time","qav",
        "num_trades","taker_base_vol","taker_quote_vol","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    # numeric conversion for price columns
    df[["open","high","low","close"]] = df[["open","high","low","close"]].apply(pd.to_numeric)
    return df[["open_time","open","high","low","close","volume"]]


def fetch_historical_paginated(symbol="BTCUSDT", interval="1m", start_time=None, end_time=None, sleep=0):
    """Fetch long historical klines from Binance Futures with pagination."""
    client = _get_client()
    dfs = []
    limit = 1500
    start_ts = int(start_time.timestamp() * 1000) if start_time else None
    end_ts = int(end_time.timestamp() * 1000) if end_time else None

    while True:
        current_start_ts = start_ts
        klines = run_binance_with_retries(
            lambda current_start_ts=current_start_ts: client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=current_start_ts,
                endTime=end_ts,
                limit=limit,
            ),
            operation_name=(
                f"futures_klines_paginated symbol={symbol} interval={interval} "
                f"start_ts={current_start_ts} end_ts={end_ts} limit={limit}"
            ),
            logger=technical_logger,
        )
        if not klines:
            break

        df = pd.DataFrame(klines, columns=[
            "open_time","open","high","low","close","volume","close_time",
            "qav","num_trades","taker_base_vol","taker_quote_vol","ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df[["open","high","low","close"]] = df[["open","high","low","close"]].apply(pd.to_numeric)
        dfs.append(df[["open_time","open","high","low","close","volume"]])

        last_ts = int(klines[-1][0]) + 1
        if end_ts and last_ts >= end_ts:
            break
        if len(klines) < limit:
            break
        start_ts = last_ts
        if sleep > 0:
            time.sleep(sleep)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["open_time", "close"])


def prepare_multi_tf(df_small, df_medium, df_big):
    """Merge multi-timeframe data (small=price, medium=BB/ATR, big=RSI/EMA)."""
    return build_feature_frame(
        df_small=df_small,
        df_medium=df_medium,
        df_big=df_big,
    )


def _normalize_candle_frame(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
    out = df.copy()
    out["open_time"] = pd.to_datetime(out["open_time"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["open_time", "open", "high", "low", "close"])
    out = out.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
    return out[["open_time", "open", "high", "low", "close", "volume"]]


def _interval_to_pandas_freq(interval: str) -> str:
    normalized = str(interval or "").strip().lower()
    if not normalized:
        raise ValueError("interval must not be empty")
    unit = normalized[-1]
    value = normalized[:-1]
    if unit == "m":
        return f"{value}min"
    if unit == "h":
        return f"{value}h"
    if unit == "d":
        return f"{value}d"
    raise ValueError(f"Unsupported interval unit: {interval}")


def _materialize_live_timeframe_frame(df_small, closed_df, *, interval: str):
    normalized_small = _normalize_candle_frame(df_small)
    normalized_closed = _normalize_candle_frame(closed_df)
    if normalized_small.empty:
        return normalized_closed

    current_open_time = normalized_small["open_time"].iloc[-1].floor(_interval_to_pandas_freq(interval))
    current_slice = normalized_small[
        normalized_small["open_time"].dt.floor(_interval_to_pandas_freq(interval)) == current_open_time
    ].copy()
    if current_slice.empty:
        return normalized_closed

    latest_closed_open_time = normalized_closed["open_time"].iloc[-1] if not normalized_closed.empty else None
    if latest_closed_open_time is not None and pd.Timestamp(latest_closed_open_time) == current_open_time:
        return normalized_closed

    forming_row = pd.DataFrame(
        [
            {
                "open_time": current_open_time,
                "open": float(current_slice["open"].iloc[0]),
                "high": float(current_slice["high"].max()),
                "low": float(current_slice["low"].min()),
                "close": float(current_slice["close"].iloc[-1]),
                "volume": float(current_slice["volume"].sum()),
            }
        ]
    )
    return _normalize_candle_frame(pd.concat([normalized_closed, forming_row], ignore_index=True))


def compute_latest_realtime_indicator_values(df_small, df_medium, df_big, *, intervals):
    engine = FeatureEngine(
        intervals={key: str(value) for key, value in intervals.items()},
    )
    engine.bootstrap_from_frames(
        df_small=df_small,
        df_medium=df_medium,
        df_big=df_big,
    )
    return engine.realtime_values()


def prepare_live_strategy_frame(
    df_small,
    df_medium,
    df_big,
    *,
    intervals,
    indicator_inputs=None,
    strategy_mode: str = "discrete",
):
    discrete_frame = prepare_multi_tf(df_small, df_medium, df_big)
    if discrete_frame.empty:
        return discrete_frame

    latest = discrete_frame.tail(1).copy()
    normalized_inputs = normalize_indicator_inputs(
        indicator_inputs,
        strategy_mode=strategy_mode,
    )
    if not uses_realtime_indicator_inputs(normalized_inputs):
        return latest

    realtime_values = compute_latest_realtime_indicator_values(
        df_small,
        df_medium,
        df_big,
        intervals=intervals,
    )
    for column in INDICATOR_COLUMNS:
        latest[f"{column}_RT"] = None if realtime_values is None else realtime_values.get(column)
    return latest


def _sanitize_dataset(df, required_columns=REQUIRED_INDICATOR_COLUMNS):
    """
    Drop warmup rows where key indicators are not ready yet.
    This keeps dataset clean for backtests/charts/ML and avoids useless head tail.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    missing_required_columns = [column for column in required_columns if column not in out.columns]
    if missing_required_columns:
        raise ValueError(
            "Dataset is missing required indicator columns: "
            + ", ".join(missing_required_columns)
        )

    initial_rows = len(out)
    out = out.dropna(subset=list(required_columns)).copy()
    out.reset_index(drop=True, inplace=True)
    dropped_rows = initial_rows - len(out)
    if dropped_rows > 0:
        technical_logger.info("dataset_warmup_rows_dropped count=%s", dropped_rows)

    if out.empty:
        raise ValueError("Dataset became empty after dropping warmup rows.")

    return out


def build_dataset(
    symbol="BTCUSDT",
    intervals=None,
    backtest_period_days=365,
    backtest_period_end_time="",
    backtest_period_start_time="",
    output_dir="data"
):
    """Fetch and build dataset from Binance, or load if already cached."""
    if intervals is None:
        intervals = {"small": "1m", "medium": "1h", "big": "4h"}

    interval_small = intervals["small"]
    interval_medium = intervals["medium"]
    interval_big = intervals["big"]

    os.makedirs(output_dir, exist_ok=True)
    time_range = resolve_backtest_time_range(
        backtest_period_days=backtest_period_days,
        backtest_period_end_time=backtest_period_end_time,
        backtest_period_start_time=backtest_period_start_time,
    )
    output_filename = (
        f"{symbol}_{interval_small}_{interval_medium}_{interval_big}_"
        f"{time_range.cache_key}.pkl"
    )
    output_path = os.path.join(output_dir, output_filename)
    end_time = time_range.end_time
    start_time = time_range.start_time

    if os.path.exists(output_path):
        technical_logger.info("dataset_cache_found path=%s", output_path)
        cached = pd.read_pickle(output_path)
        sanitized = _sanitize_dataset(cached)
        if len(sanitized) != len(cached):
            sanitized.to_pickle(output_path)
            sanitized.to_csv(output_path.replace(".pkl", ".csv"), index=False)
            technical_logger.info("dataset_cache_rewritten_sanitized path=%s", output_path)
        return sanitized

    technical_logger.info(
        "dataset_fetch_started symbol=%s days=%s intervals=%s,%s,%s",
        symbol,
        time_range.duration_days,
        interval_small,
        interval_medium,
        interval_big,
    )
    df_small = fetch_historical_paginated(symbol, interval_small, start_time, end_time)
    df_medium = fetch_historical_paginated(symbol, interval_medium, start_time, end_time)
    df_big = fetch_historical_paginated(symbol, interval_big, start_time, end_time)

    technical_logger.info("dataset_merge_started symbol=%s", symbol)
    df_merged = prepare_multi_tf(df_small, df_medium, df_big)
    df_merged = _sanitize_dataset(df_merged)

    df_merged.to_pickle(output_path)
    df_merged.to_csv(output_path.replace(".pkl", ".csv"), index=False)
    technical_logger.info("dataset_saved path=%s rows=%s", output_path, len(df_merged))

    return df_merged


if __name__ == "__main__":
    df = build_dataset()
