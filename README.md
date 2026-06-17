# Swing Trading Research Dashboard

Local research and alert system only. It does not connect to a brokerage and cannot place trades.

## Setup

```powershell
py -3 -m pip install -r requirements.txt
```

## Configuration

Edit `config.toml` to change paths, market-data settings, headline limits, and alert destinations.

Alerts are disabled by default:

```toml
[alerts]
enabled = false
discord_webhook_url = ""
```

Email alerts require SMTP settings in `[alerts.email]`.

## Run The Daily Research Job

```powershell
py -3 trading_research.py
```

The job reads `watchlist.csv`, pulls public daily OHLCV data, fetches earnings dates and recent headlines, writes `daily_candidates.csv`, stores news and ranking history in SQLite, and records alerts.

OHLCV uses a direct public chart endpoint first, then falls back to `yfinance`. If providers are rate-limiting or unavailable, the script still writes `daily_candidates.csv` with clear `Status` values so the dashboard can open cleanly.

## Run The Dashboard

```powershell
py -3 -m streamlit run app.py
```

The dashboard includes:

- Top 10 Candidates
- Ranked Watchlist
- Upcoming Earnings
- Recent Catalysts
- Strong Relative Strength Stocks
- High Volume Movers
- Latest Headlines
- Historical Rankings

## Scoring

Final score is 0-100:

- Technical Score: 40%
- Catalyst Score: 40%
- Relative Strength Score: 20%

Ratings:

- `Buy Candidate`: final score >= 75
- `Watch`: final score >= 50
- `Pass`: final score < 50

Catalyst score is 0-10 and is detected from headline language across these categories:

- Earnings Beat
- Earnings Miss
- Analyst Upgrade
- Analyst Downgrade
- New Partnership
- Product Launch
- Government Contract
- Regulatory Approval
- Insider Buying
- Insider Selling
- Acquisition/Merger

## Alerts

Alerts are generated for:

- Relative volume > 2x
- Price crossing above the 50-day moving average
- Price crossing above the 200-day moving average
- 52-week highs
- Catalyst score >= 8

Alert records are stored in the `alerts` table. Discord and email delivery happen only when enabled in `config.toml`.

## Database

The SQLite database defaults to `research.db` and stores:

- `news`: de-duplicated ticker headlines with catalyst categories and scores
- `earnings`: upcoming earnings dates and days until earnings
- `rankings`: daily historical ranking snapshots
- `alerts`: generated alert records and delivery status

## Ranking Performance Analysis

```powershell
py -3 performance_analysis.py
```

This analyzes stored ranking history for stocks that scored `90+`, `80-89`, and `70-79`. It calculates 1/5/10/20/60-day forward returns, win rate, average return, maximum drawdown, Sharpe ratio, and factor-value correlations.

Outputs:

- `ranking_forward_returns.csv`
- `ranking_performance_summary.csv`
- `ranking_factor_value.csv`
- `ranking_recommendations.txt`

The analysis only includes horizons that have matured. For example, a ranking from today cannot contribute to the 20-day or 60-day return tables yet.

## Trade Setup Generator

The dashboard includes a `Trade Setups` tab. It generates research-only swing trade setups for high-ranking tickers using OHLCV, moving averages, ATR, recent support/resistance, relative volume, relative strength, catalysts, portfolio size, and max risk per trade.

Exports:

- `trade_setups.csv`
- `trade_setups.json`
- `trade_setups_rejected.csv`

Safety filters block setups when average volume is too low, price is below $3, relative volume is below 1.2, earnings are within 2 trading days, Target 2 risk/reward is below 2:1, or trade quality is below 75. The system does not connect to a brokerage and does not place trades.

Generated setups are stored in the `trade_setups` SQLite table for later 1/5/10/20/60-day performance analysis. The `Performance` dashboard tab includes both ranking-signal performance and generated-setup performance once enough time has passed.

## Tests

```powershell
py -3 -m unittest discover -s tests -v
```
