# Scrooge Trading Bot

Scrooge is a trading bot for Binance Futures with **backtest** and **live trading** support. It uses a strategy based on **Bollinger Bands**, **RSI**, dynamic **Stop Loss** and **Take Profit**, and keeps a history of trades and balances.

---

## Features

* Backtest on historical Binance data
* Live trading via Binance Futures API
* Dynamic SL/TP and trailing
* Multi-timeframe analysis (1m, 15m, 1h)
* Trade logs in `trading_log.txt`
* Visualization of results: price chart, Bollinger Bands, RSI, equity curve
* Session state saved in `state.json`

---

## Installation

1. Clone the repository:

```bash
git clone <your_repo_url>
cd scrooge-bot
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your Binance keys:

```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

---

## Usage

### Backtest

```bash
python main.py
```

* By default, the bot runs a backtest for the last day (`backtest_period_days = 1`).

### Live Trading

* In `main.py`, set:

```python
live = True
```

* Configure parameters: symbol, balance, leverage, timeframes.
* Run:

```bash
python main.py
```

### Session Report

* After the session, generate charts and statistics:

```bash
python report.py
```

---

## Project Structure

```
.
├── main.py           # Main bot runner
├── strategy.py       # Trading strategy and backtest logic
├── trade.py          # Binance functions (open/close positions, SL/TP)
├── state.py          # Session state and trade history
├── report.py         # Generate charts and statistics
├── requirements.txt  # Dependencies
└── README.md         # This file
```

---

## Configuration Parameters

* `symbol` — trading symbol, e.g., "BTCUSDT"
* `initial_balance` — starting balance for backtest
* `interval_small`, `interval_medium`, `interval_big` — timeframes for analysis
* `sl_mult`, `tp_mult` — multipliers for Stop Loss and Take Profit
* `leverage` — leverage for live trading
* `use_full_balance` — whether to use full balance for positions

---

## License

MIT License
