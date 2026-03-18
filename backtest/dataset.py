import os
import time
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from bot.event_log import get_technical_logger
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
    raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
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
        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_ts,
            endTime=end_ts,
            limit=limit
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
    df_small["open_time"] = pd.to_datetime(df_small["open_time"], unit="ms")
    df_medium["open_time"] = pd.to_datetime(df_medium["open_time"], unit="ms")
    df_big["open_time"] = pd.to_datetime(df_big["open_time"], unit="ms")

    # --- RSI and EMA on big timeframe ---
    df_big = df_big.set_index("open_time")
    df_big["RSI"] = ta.rsi(df_big["close"], length=11)
    df_big["EMA"] = ta.ema(df_big["close"], length=50)
    df_big = df_big[["RSI", "EMA"]]

    # --- Bollinger Bands and ATR on medium timeframe ---
    df_medium = df_medium.set_index("open_time")
    bb = ta.bbands(df_medium["close"], length=20, std=2)
    atr = ta.atr(df_medium["high"], df_medium["low"], df_medium["close"], length=14)
    df_medium["BBL"] = bb["BBL_20_2.0_2.0"]
    df_medium["BBM"] = bb["BBM_20_2.0_2.0"]
    df_medium["BBU"] = bb["BBU_20_2.0_2.0"]
    df_medium["ATR"] = atr
    df_medium = df_medium[["BBL", "BBM", "BBU", "ATR"]]

    # --- Merge into small timeframe ---
    df_merged = df_small.set_index("open_time")
    df_merged = df_merged.merge(df_medium, left_index=True, right_index=True, how="left").ffill()
    df_merged = df_merged.merge(df_big, left_index=True, right_index=True, how="left").ffill()
    df_merged.reset_index(inplace=True)
    return df_merged


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
    medium_frame = _materialize_live_timeframe_frame(
        df_small,
        df_medium,
        interval=str(intervals["medium"]),
    )
    big_frame = _materialize_live_timeframe_frame(
        df_small,
        df_big,
        interval=str(intervals["big"]),
    )
    if medium_frame.empty or big_frame.empty:
        return None

    medium = medium_frame.copy().set_index("open_time")
    bb = ta.bbands(medium["close"], length=20, std=2)
    atr = ta.atr(medium["high"], medium["low"], medium["close"], length=14)
    if bb is None or atr is None:
        return None

    big = big_frame.copy().set_index("open_time")
    rsi = ta.rsi(big["close"], length=11)
    ema = ta.ema(big["close"], length=50)
    if rsi is None or ema is None:
        return None

    latest = {
        "BBL": float(bb["BBL_20_2.0_2.0"].iloc[-1]) if "BBL_20_2.0_2.0" in bb and pd.notna(bb["BBL_20_2.0_2.0"].iloc[-1]) else None,
        "BBM": float(bb["BBM_20_2.0_2.0"].iloc[-1]) if "BBM_20_2.0_2.0" in bb and pd.notna(bb["BBM_20_2.0_2.0"].iloc[-1]) else None,
        "BBU": float(bb["BBU_20_2.0_2.0"].iloc[-1]) if "BBU_20_2.0_2.0" in bb and pd.notna(bb["BBU_20_2.0_2.0"].iloc[-1]) else None,
        "ATR": float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else None,
        "RSI": float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None,
        "EMA": float(ema.iloc[-1]) if pd.notna(ema.iloc[-1]) else None,
    }
    return latest


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
    output_dir="data"
):
    """Fetch and build dataset from Binance, or load if already cached."""
    if intervals is None:
        intervals = {"small": "1m", "medium": "1h", "big": "4h"}

    interval_small = intervals["small"]
    interval_medium = intervals["medium"]
    interval_big = intervals["big"]

    os.makedirs(output_dir, exist_ok=True)
    output_filename = (
        f"{symbol}_{interval_small}_{interval_medium}_{interval_big}_"
        f"{backtest_period_days}_{backtest_period_end_time}.pkl"
    )
    output_path = os.path.join(output_dir, output_filename)

    if backtest_period_end_time == "":
        end_time = datetime.now()
    else:
        end_time = datetime.fromisoformat(backtest_period_end_time)
    start_time = end_time - timedelta(days=backtest_period_days)

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
        backtest_period_days,
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
