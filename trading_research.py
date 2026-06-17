from __future__ import annotations

import hashlib
import json
import logging
import smtplib
import sqlite3
import ssl
import time
import tomllib
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

try:
    from urllib3.exceptions import InsecureRequestWarning

    warnings.simplefilter("ignore", InsecureRequestWarning)
except Exception:
    pass


WATCHLIST_PATH = Path("watchlist.csv")
OUTPUT_PATH = Path("daily_candidates.csv")
CONFIG_PATH = Path("config.toml")
DATABASE_PATH = Path("research.db")
TRADING_DAYS_3M = 63
ProgressCallback = Callable[[str, int, int, str], None]

CATALYST_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Earnings Beat": ("beats estimates", "beat estimates", "beats expectations", "earnings beat", "raises guidance"),
    "Earnings Miss": ("misses estimates", "missed estimates", "earnings miss", "cuts guidance", "lowers guidance"),
    "Analyst Upgrade": ("upgraded", "upgrade", "raises price target", "price target raised", "initiated buy"),
    "Analyst Downgrade": ("downgraded", "downgrade", "cuts price target", "price target cut", "initiated sell"),
    "New Partnership": ("partnership", "partners with", "collaboration", "strategic alliance"),
    "Product Launch": ("launches", "unveils", "announces new product", "product launch", "rolls out"),
    "Government Contract": ("government contract", "defense contract", "federal contract", "awarded contract"),
    "Regulatory Approval": ("fda approval", "regulatory approval", "approved by", "clearance", "authorization"),
    "Insider Buying": ("insider buying", "insider buys", "director buys", "ceo buys"),
    "Insider Selling": ("insider selling", "insider sells", "director sells", "ceo sells"),
    "Acquisition/Merger": ("acquisition", "merger", "to acquire", "buyout", "takeover"),
}

CATALYST_WEIGHTS = {
    "Earnings Beat": 9,
    "Earnings Miss": 2,
    "Analyst Upgrade": 7,
    "Analyst Downgrade": 2,
    "New Partnership": 7,
    "Product Launch": 6,
    "Government Contract": 8,
    "Regulatory Approval": 8,
    "Insider Buying": 7,
    "Insider Selling": 3,
    "Acquisition/Merger": 8,
}


@dataclass(frozen=True)
class AlertConfig:
    discord_webhook_url: str = ""
    email_enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: str = ""


@dataclass(frozen=True)
class ResearchConfig:
    watchlist_path: Path = WATCHLIST_PATH
    output_path: Path = OUTPUT_PATH
    database_path: Path = DATABASE_PATH
    period: str = "18mo"
    min_history_days: int = 210
    earnings_window_days: int = 14
    news_limit: int = 5
    send_alerts: bool = False
    alert_config: AlertConfig = AlertConfig()


def setup_logging(log_path: Path = Path("research.log")) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def load_config(path: Path | str = CONFIG_PATH) -> ResearchConfig:
    path = Path(path)
    if not path.exists():
        return ResearchConfig()

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    paths = data.get("paths", {})
    market_data = data.get("market_data", {})
    alerts = data.get("alerts", {})
    email = alerts.get("email", {})

    alert_config = AlertConfig(
        discord_webhook_url=str(alerts.get("discord_webhook_url", "")),
        email_enabled=bool(email.get("enabled", False)),
        smtp_host=str(email.get("smtp_host", "")),
        smtp_port=int(email.get("smtp_port", 587)),
        smtp_username=str(email.get("smtp_username", "")),
        smtp_password=str(email.get("smtp_password", "")),
        email_from=str(email.get("from", "")),
        email_to=str(email.get("to", "")),
    )

    return ResearchConfig(
        watchlist_path=Path(paths.get("watchlist", WATCHLIST_PATH)),
        output_path=Path(paths.get("daily_candidates", OUTPUT_PATH)),
        database_path=Path(paths.get("database", DATABASE_PATH)),
        period=str(market_data.get("period", "18mo")),
        min_history_days=int(market_data.get("min_history_days", 210)),
        earnings_window_days=int(data.get("earnings", {}).get("window_days", 14)),
        news_limit=int(data.get("news", {}).get("headlines_per_ticker", 5)),
        send_alerts=bool(alerts.get("enabled", False)),
        alert_config=alert_config,
    )


def load_watchlist(path: Path | str = WATCHLIST_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Watchlist not found: {path}")

    watchlist = pd.read_csv(path)
    if watchlist.empty:
        raise ValueError(f"Watchlist is empty: {path}")

    if "Symbol" not in watchlist.columns:
        watchlist = watchlist.rename(columns={watchlist.columns[0]: "Symbol"})

    watchlist["Symbol"] = watchlist["Symbol"].astype(str).str.strip().str.upper().replace("", np.nan)
    watchlist = watchlist.dropna(subset=["Symbol"]).drop_duplicates("Symbol")

    for column in ("Tier", "Category"):
        if column not in watchlist.columns:
            watchlist[column] = ""

    return watchlist[["Symbol", "Tier", "Category"]].reset_index(drop=True)


def yahoo_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


def _period_to_days(period: str) -> int:
    value = period.strip().lower()
    try:
        if value.endswith("mo"):
            return int(value[:-2]) * 31
        if value.endswith("y"):
            return int(value[:-1]) * 366
        if value.endswith("d"):
            return int(value[:-1])
    except ValueError:
        pass
    return 550


def init_db(database_path: Path | str = DATABASE_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(database_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            headline TEXT NOT NULL,
            normalized_headline TEXT NOT NULL,
            publisher TEXT,
            link TEXT,
            published_at TEXT,
            catalyst_category TEXT,
            catalyst_score REAL NOT NULL DEFAULT 0,
            inserted_at TEXT NOT NULL,
            UNIQUE(symbol, normalized_headline)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS earnings (
            symbol TEXT PRIMARY KEY,
            earnings_date TEXT,
            days_until_earnings INTEGER,
            earnings_within_14_days INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rankings (
            id INTEGER PRIMARY KEY,
            run_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            final_score REAL NOT NULL,
            technical_score REAL NOT NULL,
            catalyst_score REAL NOT NULL,
            relative_strength_score REAL NOT NULL,
            rating TEXT NOT NULL,
            rank INTEGER NOT NULL,
            payload TEXT NOT NULL,
            inserted_at TEXT NOT NULL,
            UNIQUE(run_date, symbol)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY,
            run_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            message TEXT NOT NULL,
            delivered INTEGER NOT NULL DEFAULT 0,
            inserted_at TEXT NOT NULL,
            UNIQUE(run_date, symbol, alert_type)
        )
        """
    )
    conn.commit()
    return conn


def normalize_headline(headline: str) -> str:
    cleaned = " ".join(headline.lower().split())
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def detect_catalyst(headline: str) -> tuple[str, float]:
    text = headline.lower()
    matches = [category for category, needles in CATALYST_KEYWORDS.items() if any(needle in text for needle in needles)]
    if not matches:
        return "", 0.0
    score = max(CATALYST_WEIGHTS[category] for category in matches)
    return ", ".join(matches), float(score)


def _download_ohlcv(symbols: Iterable[str], period: str) -> pd.DataFrame:
    direct = _download_yahoo_chart_ohlcv(symbols, period)
    if not direct.empty:
        return direct

    try:
        import yfinance as yf
        from curl_cffi import requests
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install requirements with `py -3 -m pip install -r requirements.txt`.") from exc

    session = requests.Session(verify=False)
    yahoo_symbols = sorted({yahoo_symbol(symbol) for symbol in symbols} | {"SPY"})
    frames: list[pd.DataFrame] = []
    chunks = [yahoo_symbols[index : index + 25] for index in range(0, len(yahoo_symbols), 25)]

    for chunk in chunks:
        try:
            data = yf.download(
                tickers=chunk,
                period=period,
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                threads=False,
                progress=False,
                timeout=20,
                session=session,
            )
        except Exception as exc:
            logging.warning("OHLCV download failed for chunk %s: %s", chunk, exc)
            continue
        if data is not None and not data.empty:
            frames.append(data)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, axis=1)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.loc[:, ~data.columns.duplicated()]
    return data


def _download_yahoo_chart_ohlcv(symbols: Iterable[str], period: str) -> pd.DataFrame:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install requirements with `py -3 -m pip install -r requirements.txt`.") from exc

    end = int(time.time())
    start = end - (_period_to_days(period) * 24 * 60 * 60)
    frames: dict[str, pd.DataFrame] = {}

    for symbol in sorted({*symbols, "SPY"}):
        yf_symbol = yahoo_symbol(symbol)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_symbol}"
        try:
            response = requests.get(
                url,
                params={"period1": start, "period2": end, "interval": "1d", "events": "history"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=20,
                verify=False,
            )
            response.raise_for_status()
            payload = response.json()
            result = (payload.get("chart", {}).get("result") or [None])[0]
            if not result:
                continue
            timestamps = result.get("timestamp") or []
            quote = ((result.get("indicators") or {}).get("quote") or [None])[0]
            adjclose = ((result.get("indicators") or {}).get("adjclose") or [None])[0]
            if not timestamps or not quote:
                continue

            frame = pd.DataFrame(
                {
                    "Open": quote.get("open"),
                    "High": quote.get("high"),
                    "Low": quote.get("low"),
                    "Close": quote.get("close"),
                    "Volume": quote.get("volume"),
                },
                index=pd.to_datetime(timestamps, unit="s").normalize(),
            )
            if adjclose and adjclose.get("adjclose"):
                frame["Adj Close"] = adjclose.get("adjclose")
            else:
                frame["Adj Close"] = frame["Close"]
            frame = frame[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].dropna(how="all")
            if not frame.empty:
                frames[yf_symbol] = frame
        except Exception as exc:
            logging.info("Yahoo chart OHLCV unavailable for %s: %s", symbol, exc)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=1)


def _frame_for_symbol(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    yf_symbol = yahoo_symbol(symbol)
    if isinstance(data.columns, pd.MultiIndex):
        if yf_symbol not in data.columns.get_level_values(0):
            return pd.DataFrame()
        frame = data[yf_symbol].copy()
    else:
        frame = data.copy()
    expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    return frame[[column for column in expected if column in frame.columns]].dropna(how="all")


def _pct_return(close: pd.Series, days: int) -> float:
    close = close.dropna()
    if len(close) <= days:
        return np.nan
    return float((close.iloc[-1] / close.iloc[-days - 1]) - 1)


def percentile_score(values: pd.Series) -> pd.Series:
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index)
    ranks = pd.Series(1.0, index=valid.index) if len(valid) == 1 else valid.rank(pct=True)
    return (ranks.reindex(values.index) * 100).round(1)


def calculate_technical_metrics(frame: pd.DataFrame, spy_return_3m: float) -> dict[str, object]:
    frame = frame.sort_index().copy()
    frame["MA20"] = frame["Close"].rolling(20).mean()
    frame["MA50"] = frame["Close"].rolling(50).mean()
    frame["MA200"] = frame["Close"].rolling(200).mean()
    frame["AvgVolume20"] = frame["Volume"].rolling(20).mean()
    frame["High52Week"] = frame["High"].rolling(252, min_periods=20).max()

    usable = frame.dropna(subset=["Close", "Volume"])
    if usable.empty:
        return {"Status": "No usable close/volume"}

    latest = usable.iloc[-1]
    previous = usable.iloc[-2] if len(usable) > 1 else None
    close = float(latest["Close"])
    volume = float(latest["Volume"])
    avg_volume_20 = float(latest.get("AvgVolume20", np.nan))
    ma20 = float(latest.get("MA20", np.nan))
    ma50 = float(latest.get("MA50", np.nan))
    ma200 = float(latest.get("MA200", np.nan))
    rel_volume = volume / avg_volume_20 if avg_volume_20 and not np.isnan(avg_volume_20) else np.nan
    return_3m = _pct_return(frame["Close"], TRADING_DAYS_3M)
    rs_3m = return_3m - spy_return_3m if not np.isnan(spy_return_3m) else np.nan

    trend_score = 0
    trend_score += 15 if close > ma20 else 0
    trend_score += 20 if close > ma50 else 0
    trend_score += 20 if close > ma200 else 0
    trend_score += 10 if ma20 > ma50 else 0
    trend_score += 10 if ma50 > ma200 else 0
    volume_score = min(max(rel_volume, 0), 2) / 2 * 25 if not np.isnan(rel_volume) else 0
    technical_score = min(trend_score + volume_score, 100)

    crossed_ma50 = False
    crossed_ma200 = False
    if previous is not None:
        prev_close = float(previous["Close"])
        prev_ma50 = float(previous.get("MA50", np.nan))
        prev_ma200 = float(previous.get("MA200", np.nan))
        crossed_ma50 = not np.isnan(prev_ma50) and not np.isnan(ma50) and prev_close <= prev_ma50 and close > ma50
        crossed_ma200 = not np.isnan(prev_ma200) and not np.isnan(ma200) and prev_close <= prev_ma200 and close > ma200

    high_52_week = float(latest.get("High52Week", np.nan))
    made_52_week_high = not np.isnan(high_52_week) and close >= high_52_week

    return {
        "AsOf": latest.name.date().isoformat(),
        "Close": round(close, 2),
        "Volume": int(volume),
        "MA20": round(ma20, 2) if not np.isnan(ma20) else np.nan,
        "MA50": round(ma50, 2) if not np.isnan(ma50) else np.nan,
        "MA200": round(ma200, 2) if not np.isnan(ma200) else np.nan,
        "RelativeVolume": round(rel_volume, 2) if not np.isnan(rel_volume) else np.nan,
        "Return3M": round(return_3m * 100, 2) if not np.isnan(return_3m) else np.nan,
        "SPYReturn3M": round(spy_return_3m * 100, 2) if not np.isnan(spy_return_3m) else np.nan,
        "RelativeStrength3M": round(rs_3m * 100, 2) if not np.isnan(rs_3m) else np.nan,
        "TechnicalScore": round(technical_score, 1),
        "TrendScore": round(trend_score, 1),
        "VolumeScore": round(volume_score, 1),
        "CrossedMA50": bool(crossed_ma50),
        "CrossedMA200": bool(crossed_ma200),
        "Made52WeekHigh": bool(made_52_week_high),
        "Status": "OK",
    }


def fetch_earnings(symbol: str) -> tuple[str, int | None, bool]:
    ticker = None
    try:
        import yfinance as yf
        from curl_cffi import requests

        ticker = yf.Ticker(yahoo_symbol(symbol), session=requests.Session(verify=False))
        dates = ticker.get_earnings_dates(limit=8)
    except Exception as exc:
        logging.info("Earnings unavailable for %s: %s", symbol, exc)
        dates = None

    future_dates: list[date] = []
    today = date.today()

    if dates is not None and not dates.empty:
        parsed_dates = pd.to_datetime(dates.index, errors="coerce").dropna()
        future_dates.extend(value.date() for value in parsed_dates if value.date() >= today)

    if not future_dates and ticker is not None:
        try:
            calendar = ticker.calendar
            if isinstance(calendar, dict):
                values = calendar.values()
            elif isinstance(calendar, pd.DataFrame):
                values = calendar.to_numpy().ravel()
            else:
                values = []
            parsed = pd.to_datetime(list(values), errors="coerce").dropna()
            future_dates.extend(value.date() for value in parsed if value.date() >= today)
        except Exception as exc:
            logging.info("Calendar earnings fallback unavailable for %s: %s", symbol, exc)

    if not future_dates:
        return "", None, False

    next_date = min(future_dates)
    days = (next_date - today).days
    return next_date.isoformat(), days, days <= 14


def store_earnings(conn: sqlite3.Connection, symbol: str, earnings_date: str, days: int | None, within_14: bool) -> None:
    conn.execute(
        """
        INSERT INTO earnings(symbol, earnings_date, days_until_earnings, earnings_within_14_days, updated_at)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            earnings_date=excluded.earnings_date,
            days_until_earnings=excluded.days_until_earnings,
            earnings_within_14_days=excluded.earnings_within_14_days,
            updated_at=excluded.updated_at
        """,
        (symbol, earnings_date, days, int(within_14), datetime.now(timezone.utc).isoformat()),
    )


def fetch_news(symbol: str, limit: int) -> list[dict[str, object]]:
    try:
        import yfinance as yf
        from curl_cffi import requests

        ticker = yf.Ticker(yahoo_symbol(symbol), session=requests.Session(verify=False))
        items = ticker.news or []
    except Exception as exc:
        logging.info("News unavailable for %s: %s", symbol, exc)
        return []

    normalized_items: list[dict[str, object]] = []
    for item in items[:limit]:
        content = item.get("content", item) if isinstance(item, dict) else {}
        headline = str(content.get("title") or item.get("title") or "").strip()
        if not headline:
            continue
        provider = content.get("provider", {}) if isinstance(content.get("provider"), dict) else {}
        link = content.get("canonicalUrl", {}) if isinstance(content.get("canonicalUrl"), dict) else {}
        published = content.get("pubDate") or item.get("providerPublishTime") or ""
        normalized_items.append(
            {
                "headline": headline,
                "publisher": provider.get("displayName") or item.get("publisher", ""),
                "link": link.get("url") or item.get("link", ""),
                "published_at": str(published),
            }
        )
    return normalized_items


def store_news(conn: sqlite3.Connection, symbol: str, news_items: list[dict[str, object]]) -> float:
    best_score = 0.0
    now = datetime.now(timezone.utc).isoformat()
    for item in news_items:
        headline = str(item["headline"])
        category, score = detect_catalyst(headline)
        best_score = max(best_score, score)
        conn.execute(
            """
            INSERT OR IGNORE INTO news(
                symbol, headline, normalized_headline, publisher, link, published_at,
                catalyst_category, catalyst_score, inserted_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                headline,
                normalize_headline(headline),
                item.get("publisher", ""),
                item.get("link", ""),
                item.get("published_at", ""),
                category,
                score,
                now,
            ),
        )
    return best_score


def latest_news(conn: sqlite3.Connection, limit: int = 5) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT symbol, headline, publisher, link, published_at, catalyst_category, catalyst_score
        FROM news
        ORDER BY COALESCE(published_at, inserted_at) DESC, id DESC
        LIMIT ?
        """,
        conn,
        params=(limit,),
    )


def latest_catalysts(conn: sqlite3.Connection, limit: int = 25) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT symbol, headline, catalyst_category, catalyst_score, published_at
        FROM news
        WHERE catalyst_score > 0
        ORDER BY catalyst_score DESC, COALESCE(published_at, inserted_at) DESC
        LIMIT ?
        """,
        conn,
        params=(limit,),
    )


def generate_alerts(row: pd.Series) -> list[tuple[str, str]]:
    alerts: list[tuple[str, str]] = []
    symbol = row["Symbol"]
    rel_volume = row.get("RelativeVolume")
    if pd.notna(rel_volume) and float(rel_volume) > 2:
        alerts.append(("Relative volume > 2x", f"{symbol} relative volume is {float(rel_volume):.2f}x."))
    if bool(row.get("CrossedMA50", False)):
        alerts.append(("50-day MA cross", f"{symbol} crossed above its 50-day moving average."))
    if bool(row.get("CrossedMA200", False)):
        alerts.append(("200-day MA cross", f"{symbol} crossed above its 200-day moving average."))
    if bool(row.get("Made52WeekHigh", False)):
        alerts.append(("52-week high", f"{symbol} made a 52-week high."))
    catalyst_score = row.get("CatalystScore")
    if pd.notna(catalyst_score) and float(catalyst_score) >= 8:
        alerts.append(("Catalyst score >= 8", f"{symbol} catalyst score is {float(catalyst_score):.1f}."))
    return alerts


def send_discord(webhook_url: str, messages: list[str]) -> bool:
    if not webhook_url or not messages:
        return False
    try:
        import requests

        response = requests.post(webhook_url, json={"content": "\n".join(messages)}, timeout=15)
        response.raise_for_status()
        return True
    except Exception as exc:
        logging.warning("Discord alert failed: %s", exc)
        return False


def send_email(alert_config: AlertConfig, subject: str, messages: list[str]) -> bool:
    if not alert_config.email_enabled or not messages:
        return False
    required = [
        alert_config.smtp_host,
        alert_config.smtp_username,
        alert_config.smtp_password,
        alert_config.email_from,
        alert_config.email_to,
    ]
    if not all(required):
        logging.warning("Email alerts enabled but SMTP settings are incomplete.")
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = alert_config.email_from
    message["To"] = alert_config.email_to
    message.set_content("\n".join(messages))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(alert_config.smtp_host, alert_config.smtp_port, timeout=20) as server:
            server.starttls(context=context)
            server.login(alert_config.smtp_username, alert_config.smtp_password)
            server.send_message(message)
        return True
    except Exception as exc:
        logging.warning("Email alert failed: %s", exc)
        return False


def store_alerts(conn: sqlite3.Connection, candidates: pd.DataFrame, config: ResearchConfig) -> None:
    run_date = date.today().isoformat()
    alert_messages: list[str] = []
    now = datetime.now(timezone.utc).isoformat()

    for _, row in candidates.iterrows():
        for alert_type, message in generate_alerts(row):
            conn.execute(
                """
                INSERT OR IGNORE INTO alerts(run_date, symbol, alert_type, message, delivered, inserted_at)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (run_date, row["Symbol"], alert_type, message, 0, now),
            )
            alert_messages.append(message)

    delivered = False
    if config.send_alerts and alert_messages:
        discord_sent = send_discord(config.alert_config.discord_webhook_url, alert_messages)
        email_sent = send_email(config.alert_config, "Swing Trading Research Alerts", alert_messages)
        delivered = discord_sent or email_sent

    if delivered:
        conn.execute("UPDATE alerts SET delivered = 1 WHERE run_date = ?", (run_date,))


def classify(final_score: float) -> str:
    if final_score >= 75:
        return "Buy Candidate"
    if final_score >= 50:
        return "Watch"
    return "Pass"


def build_candidates(config: ResearchConfig | None = None, progress_callback: ProgressCallback | None = None) -> pd.DataFrame:
    config = config or load_config()
    setup_logging()
    if progress_callback:
        progress_callback("Starting", 0, 100, "Loading watchlist")
    watchlist = load_watchlist(config.watchlist_path)
    total_tickers = len(watchlist)
    if progress_callback:
        progress_callback("Database", 5, 100, "Opening research database")
    conn = init_db(config.database_path)

    if progress_callback:
        progress_callback("Market Data", 10, 100, f"Downloading OHLCV for {total_tickers} tickers")
    raw = _download_ohlcv(watchlist["Symbol"], config.period)
    spy = _frame_for_symbol(raw, "SPY")
    spy_return_3m = _pct_return(spy["Close"], TRADING_DAYS_3M) if not spy.empty else np.nan

    rows: list[dict[str, object]] = []
    for index, item in enumerate(watchlist.itertuples(index=False), start=1):
        symbol = item.Symbol
        if progress_callback:
            progress_value = 15 + int((index - 1) / max(total_tickers, 1) * 70)
            progress_callback("Ticker Research", progress_value, 100, f"Analyzing {symbol} ({index}/{total_tickers})")
        frame = _frame_for_symbol(raw, symbol)
        base = {"Symbol": symbol, "Tier": item.Tier, "Category": item.Category}

        if frame.empty or "Close" not in frame or "Volume" not in frame:
            metrics = {"Status": "No data", "TechnicalScore": 0, "RelativeStrength3M": np.nan}
        else:
            metrics = calculate_technical_metrics(frame, spy_return_3m)
            if len(frame.dropna(subset=["Close"])) < config.min_history_days and metrics.get("Status") == "OK":
                metrics["Status"] = "Limited history"

        earnings_date, days_until, within_14 = fetch_earnings(symbol)
        store_earnings(conn, symbol, earnings_date, days_until, within_14)

        news_items = fetch_news(symbol, config.news_limit)
        catalyst_score = store_news(conn, symbol, news_items)

        rows.append(
            {
                **base,
                **metrics,
                "EarningsDate": earnings_date,
                "DaysUntilEarnings": days_until,
                "EarningsWithin14Days": within_14,
                "CatalystScore": catalyst_score,
            }
        )

    conn.commit()
    if progress_callback:
        progress_callback("Scoring", 88, 100, "Calculating final scores and ranks")
    candidates = pd.DataFrame(rows)
    candidates["RelativeStrengthScore"] = percentile_score(candidates["RelativeStrength3M"]).fillna(0)
    candidates["TechnicalScore"] = candidates["TechnicalScore"].fillna(0).astype(float)
    candidates["CatalystScore"] = candidates["CatalystScore"].fillna(0).clip(0, 10).astype(float)
    candidates["FinalScore"] = (
        candidates["TechnicalScore"] * 0.40
        + (candidates["CatalystScore"] * 10) * 0.40
        + candidates["RelativeStrengthScore"] * 0.20
    ).clip(0, 100).round(1)
    candidates["Rating"] = candidates["FinalScore"].apply(classify)
    candidates = candidates.sort_values(["FinalScore", "RelativeStrengthScore"], ascending=[False, False]).reset_index(drop=True)
    candidates["Rank"] = candidates.index + 1

    preferred_columns = [
        "Rank",
        "Symbol",
        "Tier",
        "Category",
        "FinalScore",
        "Rating",
        "TechnicalScore",
        "CatalystScore",
        "RelativeStrengthScore",
        "Close",
        "MA20",
        "MA50",
        "MA200",
        "RelativeVolume",
        "Return3M",
        "SPYReturn3M",
        "RelativeStrength3M",
        "EarningsDate",
        "DaysUntilEarnings",
        "EarningsWithin14Days",
        "Volume",
        "AsOf",
        "Status",
        "CrossedMA50",
        "CrossedMA200",
        "Made52WeekHigh",
    ]
    candidates = candidates[[column for column in preferred_columns if column in candidates.columns]]
    if progress_callback:
        progress_callback("Saving", 94, 100, "Writing CSV, ranking history, and alerts")
    candidates.to_csv(config.output_path, index=False)
    store_rankings(conn, candidates)
    store_alerts(conn, candidates, config)
    conn.commit()
    conn.close()
    if progress_callback:
        progress_callback("Complete", 100, 100, f"Refresh complete: {len(candidates)} tickers ranked")
    return candidates


def store_rankings(conn: sqlite3.Connection, candidates: pd.DataFrame) -> None:
    run_date = date.today().isoformat()
    inserted_at = datetime.now(timezone.utc).isoformat()
    for _, row in candidates.iterrows():
        payload = row.where(pd.notna(row), None).to_dict()
        conn.execute(
            """
            INSERT INTO rankings(
                run_date, symbol, final_score, technical_score, catalyst_score,
                relative_strength_score, rating, rank, payload, inserted_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_date, symbol) DO UPDATE SET
                final_score=excluded.final_score,
                technical_score=excluded.technical_score,
                catalyst_score=excluded.catalyst_score,
                relative_strength_score=excluded.relative_strength_score,
                rating=excluded.rating,
                rank=excluded.rank,
                payload=excluded.payload,
                inserted_at=excluded.inserted_at
            """,
            (
                run_date,
                row["Symbol"],
                float(row["FinalScore"]),
                float(row["TechnicalScore"]),
                float(row["CatalystScore"]),
                float(row["RelativeStrengthScore"]),
                row["Rating"],
                int(row["Rank"]),
                json.dumps(payload, default=str),
                inserted_at,
            ),
        )


def ranking_history(conn: sqlite3.Connection, limit: int = 500) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT run_date, symbol, rank, final_score, rating
        FROM rankings
        ORDER BY run_date DESC, rank ASC
        LIMIT ?
        """,
        conn,
        params=(limit,),
    )


if __name__ == "__main__":
    results = build_candidates(load_config())
    print(f"Wrote {OUTPUT_PATH} with {len(results)} rows.")
