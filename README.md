# Scrooge Trading Bot

Simple Binance Futures trading bot using Bollinger Bands strategy with SL/TP, dynamic stop, and session logging.

## Features

- Multi-timeframe analysis (1m, 15m, 1h)
- Bollinger Bands and RSI-based entries
- Stop-loss, take-profit, and dynamic stop management
- Live trading on Binance Futures
- State persistence (`state.json`) to recover after restarts
- Logging of trades and events
- Session report with price chart, trades, and equity curve

## Requirements

```bash
pip install python-binance pandas pandas_ta matplotlib numpy
```

Set your Binance API keys in environment variables:

```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

## Usage

### Live Trading

```bash
python main.py
```

The bot will run indefinitely, checking every 1 minute. Press `Ctrl+C` to stop; session data will be saved automatically.

### Backtesting

Set `live = False` in `main.py`. The bot will simulate trading on historical data and plot results with charts.

### Session Report

After a live session, you can generate a session report:

```bash
python report.py
```

It will plot:

- Price chart with Bollinger Bands
- Entry/exit markers for trades
- Equity curve
- Basic session statistics

## State File

`state.json` contains:

- `balance` – current balance  
- `position` – current open position  
- `trade_history` – list of closed trades  
- `balance_history` – equity over time  
- `session_start` / `session_end` – timestamps of the session  

## Logging

Events are logged to `trading_log.txt` with timestamps.

