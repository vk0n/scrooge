# Scrooge — Multi-Timeframe Binance Futures Trading Bot

Scrooge is a fully automated **algorithmic trading system** built for **Binance Futures**, combining advanced **technical analysis (Bollinger Bands + RSI + EMA)**, adaptive **stop-loss / take-profit control**, and a modular backtesting & optimization engine.  

Originally designed as an experiment in intraday mean-reversion trading, Scrooge has evolved into a robust **multi-timeframe swing-trading bot** capable of live trading, historical simulation, and parameter optimization.

---

## ⚙️ Key Features

- **Multi-Timeframe Decision Engine**  
  Uses three synchronized timeframes (e.g., `1m–1h–4h` or `1h–4h–1d`) to blend micro-price action with higher-level RSI and EMA trends.

- **Dynamic Bollinger Band Strategy**  
  Entry conditions based on lower/upper Bollinger Band crossovers with RSI confirmation.

- **Customizable Risk/Reward Framework**  
  Parameterized Stop-Loss (`sl_mult`) and Take-Profit (`tp_mult`) multipliers with optional trailing ATR stop logic.

- **Stateful Execution**  
  Trade state, open positions, and balance history are saved between runs via `state.py`.

- **Optimized for Research and Deployment**  
  Supports both:
  - **Offline backtesting** (fast evaluation of historical data)
  - **Live trading** on Binance Futures (via `trade.py`)

- **Automatic Data Handling**  
  Historical klines are fetched and merged from multiple intervals using `data.py`, then serialized to `.pkl` for reuse.

- **Parameter Optimization**  
  Grid-search optimization of key strategy parameters using `optimize.py` with parallel execution and YAML logging.

- **Visual Performance Reports**  
  `report.py` generates detailed matplotlib charts for price, RSI, and equity curves with trade markers.

---

## 🧪 Project Structure

```
scrooge/
├── config.yaml              # Main configuration (symbol, intervals, leverage, etc.)
├── param_grid.yaml          # Parameter grid for optimization
├── data.py                  # Historical data fetcher and preprocessor
├── strategy.py              # Core multi-TF trading logic (RSI, EMA, Bollinger Bands)
├── trade.py                 # Binance Futures trading operations (open/close positions)
├── state.py                 # Persistent session management (positions, balance, etc.)
├── optimize.py              # Parameter grid optimization engine
├── report.py                # Visualization and reporting utilities
├── main.py                  # Entry point for backtesting or live trading
├── results/                 # Output charts and configs from experiments
└── requirements.txt         # Python dependencies
```

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/vk0n/scrooge.git
cd scrooge
```

### 2. Create a Virtual Environment
```bash
python -m venv scrooge-env
source scrooge-env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```bash
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

---

## ⚙️ Configuration (`config.yaml`)

All runtime parameters are centralized in `config.yaml`.

Example:
```yaml
symbol: BTCUSDT
leverage: 3
initial_balance: 10000

intervals:
  small: 1m     # Price base timeframe
  medium: 1h    # Bollinger Bands timeframe
  big: 4h       # RSI & EMA timeframe

backtest_period_days: 365
use_state: false
enable_logs: true
```

---

## 🚀 Usage

### 1. Generate Historical Data
Fetch and prepare synchronized multi-timeframe data:

```bash
python data.py
```
This will output a merged file named after your intervals:
```
1m1h4h.pkl
```

---

### 2. Backtesting

Run a simulation using your `config.yaml`:
```bash
python main.py
```

Results will include:
- Final balance and performance metrics
- Trade log (`trading_log.txt`)
- Graphs in `results/{intervals}/`

---

### 3. Parameter Optimization

Explore optimal values for stop-loss, take-profit, and RSI thresholds:
```bash
python optimize.py
```

Results are automatically stored in YAML format (e.g., `best_metrics.yaml`), with visual summaries in `results/`.

---

### 4. Live Trading (Experimental)

Ensure you have a Futures account and sufficient balance.  
Then simply set `live: true` in `config.yaml` or launch manually:

```bash
python main.py --live
```

Logs and states are persistently stored under:
```
state.json
trading_log.txt
```

---

## 🧠 Strategy Logic Overview

1. **Entry Conditions**
   - Long when price touches the lower Bollinger Band and RSI < `rsi_long_open_threshold`
   - Short when price touches the upper Bollinger Band and RSI > `rsi_short_open_threshold`

2. **Exit Conditions**
   - Stop-loss: `price ± ATR × sl_mult`
   - Take-profit: `price ± ATR × tp_mult`
   - Optional trailing ATR stop (`trail_atr_mult`)
   - RSI-based extreme exit (e.g. RSI > 90 or RSI < 10)

3. **Position Sizing**
   - Calculated dynamically based on balance, leverage, and asset price.

4. **State Persistence**
   - Every open position and balance update is stored via `state.py` to resume interrupted sessions.

---

## 📊 Example Results

Scrooge demonstrates robust performance across multiple timeframes:

| Period | Intervals | Return % | Profit Factor | Max Drawdown % |
|:-------|:-----------|:----------|:---------------|:----------------|
| 1 year | 1m–1h–4h | +62.8% | 1.35 | -36.9 |
| 2 years | 1m–1h–4h | +120.8% | 1.22 | -42.2 |
| 5 years | 1m–1h–4h | +1736% | 1.22 | -61.2 |

See `/results/` folder for charts and configurations.

---

## ⚡ Tips for Advanced Use

- For **intraday trading**, try `15m–1h–4h` intervals.  
- For **swing trading**, use `1h–4h–1d`.  
- Backtesting large datasets (5+ years) can be accelerated by pre-saving `.pkl` data files via `data.py`.  
- Binance API rate limits: insert `time.sleep(0.05)` between paginated requests to stay below 2400 req/min.

---

## ⚠️ Disclaimer

Scrooge is for **educational and research purposes only**.  
Trading cryptocurrencies involves **significant financial risk**.  
The author assumes **no responsibility** for any financial loss resulting from the use of this software.
---

## License

MIT License
---

