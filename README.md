# Scrooge — Multi-Timeframe Binance Futures Trading Bot

Scrooge is a fully automated **algorithmic trading system** for **Binance Futures**, combining advanced **technical analysis** (Bollinger Bands, RSI, and EMA), adaptive **stop-loss/take-profit logic**, and a modular **optimization engine** for both intraday and swing trading.

---

## ⚙️ Key Features

- **Multi-Timeframe Decision Framework**  
  Integrates three timeframes (e.g., `1m–1h–4h` or `1h–4h–1d`) for price action, trend confirmation, and momentum detection.

- **Dynamic Bollinger Band Strategy**  
  Entry logic is based on Bollinger Band crossovers filtered by RSI and EMA trend bias.

- **Adaptive RSI Filtering (Length = 11)**  
  Extensive backtesting determined that RSI with a length of **11** provides the best balance between signal responsiveness and trend stability. Shorter RSIs produced excessive noise; longer ones lagged in volatile conditions.

- **Optimized EMA Trend Baseline (Length = 50)**  
  Comparative backtests between EMA(40–60) showed that **EMA(50)** achieves the most consistent equity growth and lowest volatility.  
  - EMA(40): Stable, but too sensitive to minor corrections.  
  - EMA(60): Captured large trends but increased drawdowns.  
  - **EMA(50): Ideal midpoint** with Profit Factor ~1.5 and controlled -40% max drawdown.

- **Configurable Risk Management**  
  Adjustable Stop-Loss (`sl_mult`), Take-Profit (`tp_mult`), and Trailing ATR multiplier (`trail_atr_mult`).

- **Stateful Backtesting & Live Mode**  
  Retains open trades, balances, and equity curves between sessions for continuous operation.

- **Comprehensive Analytics & Visualization**  
  Generates detailed matplotlib charts showing Bollinger bands, RSI, and equity progression.

---

## 🧪 Project Structure

```
scrooge/
├── config.yaml              # Main configuration file
├── param_grid.yaml          # Optimization parameter grid
├── data.py                  # Multi-timeframe data fetching and preparation
├── strategy.py              # Core trading logic (RSI/EMA/Bollinger)
├── trade.py                 # Binance Futures order management
├── state.py                 # Persistent storage for open positions
├── optimize.py              # Automated parameter optimization script
├── report.py                # Plotting and performance visualization
├── main.py                  # Entry point for backtesting or live trading
├── results/                 # Historical charts and configuration snapshots
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
Then simply set `live: true` in `config.yaml` and launch manually:

```bash
python main.py
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

## 🧠 Technical Insights

### EMA Optimization Summary

| EMA Length | Final Balance | Profit Factor | Max Drawdown | Observation |
|-------------|----------------|----------------|----------------|--------------|
| 40 | $526,754 | 1.50 | -34% | Stable but reactive to noise |
| 45 | $485,604 | 1.47 | -40% | Slightly weaker trend capture |
| **50** | **$601,173** | **1.50** | **-40%** | **Optimal balance point** |
| 55 | $581,026 | 1.48 | -40% | Nearly equivalent performance |
| 60 | $684,556 | 1.33 | -41% | Stronger trends, higher volatility |

EMA(50) was thus selected as the **baseline trend filter** across all production configurations.

### RSI Length Analysis

| RSI Length | Final Balance | Profit Factor | Max Drawdown | Observation |
|-------------|----------------|----------------|----------------|--------------|
| 6 | $183,000 | 1.15 | -58% | Too reactive; many false entries |
| 8 | $183,839 | 1.33 | -58% | Better, but still unstable |
| 10 | $263,197 | 1.39 | -43% | Excellent trend capture |
| **11** | **$405,207** | **1.48** | **-43%** | **Optimal signal-to-noise ratio** |
| 12 | $361,338 | 1.42 | -43% | Minor over-smoothing |

RSI(11) was adopted as the **default momentum filter**, providing strong balance between early entries and trend validation.

---

## 📊 Example Results

Scrooge demonstrates robust performance across multiple timeframes:

| Period | Intervals | Return % | Profit Factor | Max Drawdown % |
|:-------|:-----------|:----------|:---------------|:----------------|
| 1 year | 1m–1h–4h | +62.8% | 1.35 | -36.9 |
| 2 years | 1m–1h–4h | +120.8% | 1.22 | -42.2 |
| 5 years | 1m–1h–4h | +1736% | 1.22 | -61.2 |

See `/results/` folder for charts and configurations.

With optimized stop-loss and take-profit parameters (`sl_mult=3.7`, `tp_mult=1.9`, `trail_atr_mult=0.04`), Scrooge achieved the following 5-year benchmark:

> **Final Balance:** $601,173 (from $10,000 initial)  
> **Profit Factor:** 1.5  
> **Max Drawdown:** -40.3%  
> **Win Rate:** 72.1%

---

## ⚠️ Disclaimer

Scrooge is provided for **educational and research purposes only**.  
Cryptocurrency trading involves substantial risk and may result in total capital loss.  
Use this software at your own discretion.

---

## License

MIT License


## 🧘 Philosophy of Scrooge

Scrooge is not human — and that’s his greatest strength.
He does not hope, fear, hesitate, or overthink.
Where humans trade with emotion, Scrooge trades with mathematics.

He sees the market as a field of probabilities, not possibilities.
He doesn’t chase euphoria or revenge after loss.
Every position is just data, every outcome a statistical event.

While traders battle psychology, Scrooge operates in logic.
He never over-leverages from greed, never panics on a red candle.
He acts only when signals align, exits only when math demands it.

Scrooge represents the elimination of emotion and the automation of discipline.
His only loyalty is to the algorithm — consistent, adaptive, and entirely emotionless.