from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research import DATABASE_PATH, init_db, _download_ohlcv, _frame_for_symbol


SETUPS_CSV = Path("trade_setups.csv")
SETUPS_JSON = Path("trade_setups.json")
REJECTED_SETUPS_CSV = Path("trade_setups_rejected.csv")
SETUP_COLUMNS = [
    "Ticker",
    "Company",
    "Setup Class",
    "Setup Type",
    "Current Price",
    "Entry Price",
    "Stop Loss",
    "Target 1",
    "Target 2",
    "Target 3",
    "Risk Per Share",
    "Reward Per Share",
    "Position Size",
    "Position Cost",
    "Max Dollar Risk",
    "Portfolio Size",
    "Max Risk %",
    "Shares To Sell At Target 1",
    "Shares To Sell At Target 2",
    "Shares To Sell At Target 3",
    "Risk/Reward T1",
    "Risk/Reward T2",
    "Risk/Reward T3",
    "Trade Quality Score",
    "Timeframe",
    "Bull Thesis",
    "Bear Thesis",
    "Catalysts",
    "Key Support",
    "Key Resistance",
    "Invalidation",
    "Take-Profit Notes",
    "Notes",
    "Final Research Score",
    "Technical Score",
    "Catalyst Score",
    "Relative Strength Score",
    "Relative Volume",
    "ATR",
    "Support",
    "Resistance",
]


@dataclass(frozen=True)
class SetupConfig:
    portfolio_size: float = 100_000
    max_risk_percent: float = 1.0
    minimum_average_volume: int = 500_000
    minimum_relative_volume: float = 1.2
    minimum_price: float = 3.0
    minimum_quality_score: float = 75.0
    minimum_target_2_rr: float = 2.0
    top_ranked_limit: int = 30
    output_csv: Path = SETUPS_CSV
    output_json: Path = SETUPS_JSON
    rejected_output_csv: Path = REJECTED_SETUPS_CSV


def setup_class(quality_score: float) -> str:
    if quality_score >= 90:
        return "Elite Setup"
    if quality_score >= 80:
        return "Strong Candidate"
    if quality_score >= 75:
        return "Watchlist"
    return "Reject"


def take_profit_plan(shares: int) -> tuple[int, int, int]:
    if shares <= 0:
        return 0, 0, 0
    target_1 = math.floor(shares * 0.33)
    target_2 = math.floor(shares * 0.33)
    target_3 = shares - target_1 - target_2
    return target_1, target_2, target_3


def calculate_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = frame["High"] - frame["Low"]
    high_prev_close = (frame["High"] - frame["Close"].shift()).abs()
    low_prev_close = (frame["Low"] - frame["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def recent_support_resistance(frame: pd.DataFrame, lookback: int = 60) -> tuple[float, float]:
    recent = frame.tail(lookback).dropna(subset=["High", "Low"])
    if recent.empty:
        return np.nan, np.nan
    support = float(recent["Low"].rolling(5).min().dropna().tail(10).min())
    resistance = float(recent["High"].rolling(5).max().dropna().tail(10).max())
    return support, resistance


def classify_entry_type(
    current_price: float,
    ma20: float,
    ma50: float,
    ma200: float,
    support: float,
    resistance: float,
    atr: float,
) -> tuple[str, float]:
    buffer = max(atr * 0.10, current_price * 0.002)
    near_support = pd.notna(support) and abs(current_price - support) <= max(atr, current_price * 0.03)
    near_ma = any(
        pd.notna(value) and abs(current_price - value) <= max(atr * 0.75, current_price * 0.02)
        for value in (ma20, ma50)
    )

    if pd.notna(resistance) and current_price >= resistance * 0.985:
        return "Breakout entry", round(resistance + buffer, 2)
    if near_support or near_ma:
        anchor = max(value for value in (support, ma20, ma50) if pd.notna(value))
        return "Pullback entry", round(max(current_price, anchor + buffer), 2)
    if pd.notna(ma50) and current_price > ma50 and current_price <= ma50 + max(atr, current_price * 0.03):
        return "Reclaim entry", round(current_price + buffer, 2)

    if pd.notna(resistance):
        return "Breakout entry", round(resistance + buffer, 2)
    return "Pullback entry", round(current_price + buffer, 2)


def calculate_stop_loss(entry_price: float, support: float, ma20: float, ma50: float, atr: float) -> float:
    candidates = []
    if pd.notna(support):
        candidates.append(float(support) * 0.985)
    for average in (ma20, ma50):
        if pd.notna(average):
            candidates.append(float(average) * 0.99)
    if pd.notna(atr) and atr > 0:
        candidates.append(entry_price - (1.5 * float(atr)))

    below_entry = [candidate for candidate in candidates if candidate < entry_price]
    if not below_entry:
        return round(entry_price * 0.92, 2)
    return round(min(below_entry), 2)


def calculate_targets(entry_price: float, stop_loss: float, resistance: float) -> tuple[float, float, float]:
    risk = entry_price - stop_loss
    target_1 = entry_price + risk
    target_2 = entry_price + (2 * risk)
    target_3 = entry_price + (3 * risk)
    if pd.notna(resistance) and resistance > target_2:
        target_3 = max(target_3, resistance)
    return round(target_1, 2), round(target_2, 2), round(target_3, 2)


def position_size(portfolio_size: float, max_risk_percent: float, entry_price: float, stop_loss: float) -> tuple[int, float]:
    risk_dollars = portfolio_size * (max_risk_percent / 100)
    risk_per_share = entry_price - stop_loss
    if risk_dollars <= 0 or risk_per_share <= 0:
        return 0, round(risk_dollars, 2)
    return math.floor(risk_dollars / risk_per_share), round(risk_dollars, 2)


def risk_reward(entry_price: float, stop_loss: float, target: float) -> float:
    risk = entry_price - stop_loss
    reward = target - entry_price
    if risk <= 0:
        return np.nan
    return round(reward / risk, 2)


def market_condition_score(candidates: pd.DataFrame) -> float:
    if candidates.empty or "FinalScore" not in candidates.columns:
        return 50.0
    watch_or_better = (pd.to_numeric(candidates["FinalScore"], errors="coerce") >= 50).mean()
    return round(float(watch_or_better) * 100, 1)


def trade_quality_score(row: pd.Series, relative_volume: float, market_score: float) -> float:
    technical = float(row.get("TechnicalScore", 0) or 0)
    catalyst = float(row.get("CatalystScore", 0) or 0) * 10
    relative_strength = float(row.get("RelativeStrengthScore", 0) or 0)
    volume_score = min(max(relative_volume, 0), 2) / 2 * 100 if pd.notna(relative_volume) else 0
    return round(
        technical * 0.35
        + catalyst * 0.25
        + volume_score * 0.15
        + relative_strength * 0.15
        + market_score * 0.10,
        1,
    )


def setup_filter_reasons(
    *,
    avg_volume: float,
    relative_volume: float,
    current_price: float,
    days_until_earnings: float | None,
    rr_target_2: float,
    quality_score: float,
    config: SetupConfig,
    earnings_risk: bool = False,
) -> list[str]:
    reasons: list[str] = []
    if pd.isna(avg_volume) or avg_volume < config.minimum_average_volume:
        reasons.append("Average volume too low")
    if pd.isna(relative_volume) or relative_volume < config.minimum_relative_volume:
        reasons.append("Relative volume below 1.2")
    if pd.isna(current_price) or current_price < config.minimum_price:
        reasons.append("Price below minimum")
    if days_until_earnings is not None and pd.notna(days_until_earnings) and days_until_earnings <= 2 and not earnings_risk:
        reasons.append("Earnings within 2 trading days")
    if pd.isna(rr_target_2) or rr_target_2 < config.minimum_target_2_rr:
        reasons.append("Risk/reward to Target 2 below 2:1")
    if quality_score < config.minimum_quality_score:
        reasons.append("Trade quality score below threshold")
    return reasons


def build_trade_setup(
    row: pd.Series,
    frame: pd.DataFrame,
    candidates: pd.DataFrame,
    config: SetupConfig,
) -> dict[str, object] | None:
    frame = frame.sort_index().dropna(subset=["Close", "High", "Low", "Volume"]).copy()
    if len(frame) < 60:
        return None

    frame["MA20"] = frame["Close"].rolling(20).mean()
    frame["MA50"] = frame["Close"].rolling(50).mean()
    frame["MA200"] = frame["Close"].rolling(200).mean()
    frame["ATR"] = calculate_atr(frame)
    frame["AvgVolume20"] = frame["Volume"].rolling(20).mean()

    latest = frame.iloc[-1]
    symbol = str(row["Symbol"])
    current_price = float(latest["Close"])
    ma20 = float(latest.get("MA20", np.nan))
    ma50 = float(latest.get("MA50", np.nan))
    ma200 = float(latest.get("MA200", np.nan))
    atr = float(latest.get("ATR", np.nan))
    avg_volume = float(latest.get("AvgVolume20", np.nan))
    relative_volume = current_relative_volume(latest)
    support, resistance = recent_support_resistance(frame)

    setup_type, entry_price = classify_entry_type(current_price, ma20, ma50, ma200, support, resistance, atr)
    stop_loss = calculate_stop_loss(entry_price, support, ma20, ma50, atr)
    target_1, target_2, target_3 = calculate_targets(entry_price, stop_loss, resistance)
    risk_per_share = round(entry_price - stop_loss, 2)
    shares, max_dollar_risk = position_size(config.portfolio_size, config.max_risk_percent, entry_price, stop_loss)
    rr_1 = risk_reward(entry_price, stop_loss, target_1)
    rr_2 = risk_reward(entry_price, stop_loss, target_2)
    rr_3 = risk_reward(entry_price, stop_loss, target_3)
    quality = trade_quality_score(row, relative_volume, market_condition_score(candidates))
    days_until_earnings = row.get("DaysUntilEarnings")

    reasons = setup_filter_reasons(
        avg_volume=avg_volume,
        relative_volume=relative_volume,
        current_price=current_price,
        days_until_earnings=days_until_earnings,
        rr_target_2=rr_2,
        quality_score=quality,
        config=config,
        earnings_risk=False,
    )
    if reasons:
        return None

    reward_per_share = round(target_2 - entry_price, 2)
    target_1_shares, target_2_shares, target_3_shares = take_profit_plan(shares)
    return {
        "Ticker": symbol,
        "Company": str(row.get("Company", symbol) or symbol),
        "Setup Class": setup_class(quality),
        "Setup Type": setup_type,
        "Current Price": round(current_price, 2),
        "Entry Price": entry_price,
        "Stop Loss": stop_loss,
        "Target 1": target_1,
        "Target 2": target_2,
        "Target 3": target_3,
        "Risk Per Share": risk_per_share,
        "Reward Per Share": reward_per_share,
        "Position Size": shares,
        "Position Cost": round(shares * entry_price, 2),
        "Max Dollar Risk": max_dollar_risk,
        "Portfolio Size": config.portfolio_size,
        "Max Risk %": config.max_risk_percent,
        "Shares To Sell At Target 1": target_1_shares,
        "Shares To Sell At Target 2": target_2_shares,
        "Shares To Sell At Target 3": target_3_shares,
        "Risk/Reward T1": rr_1,
        "Risk/Reward T2": rr_2,
        "Risk/Reward T3": rr_3,
        "Trade Quality Score": quality,
        "Timeframe": "Swing trade: 2-8 weeks",
        "Bull Thesis": bull_thesis(row, setup_type),
        "Bear Thesis": bear_thesis(row),
        "Catalysts": catalyst_text(row),
        "Key Support": round(support, 2) if pd.notna(support) else np.nan,
        "Key Resistance": round(resistance, 2) if pd.notna(resistance) else np.nan,
        "Invalidation": invalidation_text(stop_loss, ma50, support),
        "Take-Profit Notes": take_profit_notes(ma20, atr, current_price),
        "Notes": "Research-only setup. No brokerage connection, no trade placement, and no guaranteed outcome.",
        "Final Research Score": round(float(row.get("FinalScore", 0) or 0), 1),
        "Technical Score": round(float(row.get("TechnicalScore", 0) or 0), 1),
        "Catalyst Score": round(float(row.get("CatalystScore", 0) or 0), 1),
        "Relative Strength Score": round(float(row.get("RelativeStrengthScore", 0) or 0), 1),
        "Relative Volume": round(relative_volume, 2) if pd.notna(relative_volume) else np.nan,
        "ATR": round(atr, 2) if pd.notna(atr) else np.nan,
        "Support": round(support, 2) if pd.notna(support) else np.nan,
        "Resistance": round(resistance, 2) if pd.notna(resistance) else np.nan,
    }


def evaluate_trade_setup(
    row: pd.Series,
    frame: pd.DataFrame,
    candidates: pd.DataFrame,
    config: SetupConfig,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    frame = frame.sort_index().dropna(subset=["Close", "High", "Low", "Volume"]).copy()
    if len(frame) < 60:
        return None, {"Ticker": row.get("Symbol", ""), "Rejected Reasons": "Insufficient OHLCV history"}

    frame["MA20"] = frame["Close"].rolling(20).mean()
    frame["MA50"] = frame["Close"].rolling(50).mean()
    frame["MA200"] = frame["Close"].rolling(200).mean()
    frame["ATR"] = calculate_atr(frame)
    frame["AvgVolume20"] = frame["Volume"].rolling(20).mean()

    latest = frame.iloc[-1]
    symbol = str(row["Symbol"])
    current_price = float(latest["Close"])
    ma20 = float(latest.get("MA20", np.nan))
    ma50 = float(latest.get("MA50", np.nan))
    ma200 = float(latest.get("MA200", np.nan))
    atr = float(latest.get("ATR", np.nan))
    avg_volume = float(latest.get("AvgVolume20", np.nan))
    relative_volume = current_relative_volume(latest)
    support, resistance = recent_support_resistance(frame)

    setup_type, entry_price = classify_entry_type(current_price, ma20, ma50, ma200, support, resistance, atr)
    stop_loss = calculate_stop_loss(entry_price, support, ma20, ma50, atr)
    target_1, target_2, target_3 = calculate_targets(entry_price, stop_loss, resistance)
    risk_per_share = round(entry_price - stop_loss, 2)
    shares, max_dollar_risk = position_size(config.portfolio_size, config.max_risk_percent, entry_price, stop_loss)
    rr_1 = risk_reward(entry_price, stop_loss, target_1)
    rr_2 = risk_reward(entry_price, stop_loss, target_2)
    rr_3 = risk_reward(entry_price, stop_loss, target_3)
    quality = trade_quality_score(row, relative_volume, market_condition_score(candidates))
    days_until_earnings = row.get("DaysUntilEarnings")
    reasons = setup_filter_reasons(
        avg_volume=avg_volume,
        relative_volume=relative_volume,
        current_price=current_price,
        days_until_earnings=days_until_earnings,
        rr_target_2=rr_2,
        quality_score=quality,
        config=config,
        earnings_risk=False,
    )
    target_1_shares, target_2_shares, target_3_shares = take_profit_plan(shares)

    base = {
        "Ticker": symbol,
        "Company": str(row.get("Company", symbol) or symbol),
        "Setup Class": setup_class(quality),
        "Setup Type": setup_type,
        "Current Price": round(current_price, 2),
        "Entry Price": entry_price,
        "Stop Loss": stop_loss,
        "Target 1": target_1,
        "Target 2": target_2,
        "Target 3": target_3,
        "Risk Per Share": risk_per_share,
        "Reward Per Share": round(target_2 - entry_price, 2),
        "Position Size": shares,
        "Position Cost": round(shares * entry_price, 2),
        "Max Dollar Risk": max_dollar_risk,
        "Portfolio Size": config.portfolio_size,
        "Max Risk %": config.max_risk_percent,
        "Shares To Sell At Target 1": target_1_shares,
        "Shares To Sell At Target 2": target_2_shares,
        "Shares To Sell At Target 3": target_3_shares,
        "Risk/Reward T1": rr_1,
        "Risk/Reward T2": rr_2,
        "Risk/Reward T3": rr_3,
        "Trade Quality Score": quality,
        "Timeframe": "Swing trade: 2-8 weeks",
        "Bull Thesis": bull_thesis(row, setup_type),
        "Bear Thesis": bear_thesis(row),
        "Catalysts": catalyst_text(row),
        "Key Support": round(support, 2) if pd.notna(support) else np.nan,
        "Key Resistance": round(resistance, 2) if pd.notna(resistance) else np.nan,
        "Invalidation": invalidation_text(stop_loss, ma50, support),
        "Take-Profit Notes": take_profit_notes(ma20, atr, current_price),
        "Notes": "Research-only setup. No brokerage connection, no trade placement, and no guaranteed outcome.",
        "Final Research Score": round(float(row.get("FinalScore", 0) or 0), 1),
        "Technical Score": round(float(row.get("TechnicalScore", 0) or 0), 1),
        "Catalyst Score": round(float(row.get("CatalystScore", 0) or 0), 1),
        "Relative Strength Score": round(float(row.get("RelativeStrengthScore", 0) or 0), 1),
        "Relative Volume": round(relative_volume, 2) if pd.notna(relative_volume) else np.nan,
        "ATR": round(atr, 2) if pd.notna(atr) else np.nan,
        "Support": round(support, 2) if pd.notna(support) else np.nan,
        "Resistance": round(resistance, 2) if pd.notna(resistance) else np.nan,
    }
    if reasons:
        rejected = base | {"Rejected Reasons": "; ".join(reasons)}
        return None, rejected
    return base, None


def current_relative_volume(latest: pd.Series) -> float:
    avg_volume = latest.get("AvgVolume20", np.nan)
    volume = latest.get("Volume", np.nan)
    if pd.isna(avg_volume) or avg_volume <= 0 or pd.isna(volume):
        return np.nan
    return float(volume) / float(avg_volume)


def bull_thesis(row: pd.Series, setup_type: str) -> str:
    return (
        f"{setup_type} aligned with final research score {float(row.get('FinalScore', 0) or 0):.1f}, "
        f"relative strength score {float(row.get('RelativeStrengthScore', 0) or 0):.1f}, "
        f"and catalyst score {float(row.get('CatalystScore', 0) or 0):.1f}."
    )


def bear_thesis(row: pd.Series) -> str:
    return (
        "Setup weakens if price loses key moving-average support, relative volume fades, "
        f"or the catalyst score of {float(row.get('CatalystScore', 0) or 0):.1f} is not confirmed by follow-through."
    )


def catalyst_text(row: pd.Series) -> str:
    score = float(row.get("CatalystScore", 0) or 0)
    if score <= 0:
        return "No scored catalyst detected."
    return f"Detected catalyst strength score: {score:.1f}/10."


def invalidation_text(stop_loss: float, ma50: float, support: float) -> str:
    anchors = [f"stop loss {stop_loss:.2f}"]
    if pd.notna(ma50):
        anchors.append(f"50-day moving average {ma50:.2f}")
    if pd.notna(support):
        anchors.append(f"recent support {support:.2f}")
    return "Invalidate on a decisive close below " + " or ".join(anchors) + "."


def take_profit_notes(ma20: float, atr: float, current_price: float) -> str:
    trail_atr = current_price - (1.5 * atr) if pd.notna(atr) else np.nan
    trail_parts = []
    if pd.notna(ma20):
        trail_parts.append(f"20-day moving average ({ma20:.2f})")
    if pd.notna(trail_atr):
        trail_parts.append(f"1.5 ATR below current price ({trail_atr:.2f})")
    trail = " or ".join(trail_parts) if trail_parts else "the tighter valid trailing stop"
    return (
        "Sell 33% at Target 1, 33% at Target 2, and the remaining shares at Target 3. "
        "After Target 1, move stop to breakeven. After Target 2, trail stop using "
        f"{trail}."
    )


def init_setup_history(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_setups (
            id INTEGER PRIMARY KEY,
            setup_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            trade_quality_score REAL NOT NULL,
            setup_class TEXT NOT NULL,
            setup_type TEXT NOT NULL,
            entry_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            target_1 REAL NOT NULL,
            target_2 REAL NOT NULL,
            target_3 REAL NOT NULL,
            position_size INTEGER NOT NULL,
            payload TEXT NOT NULL,
            inserted_at TEXT NOT NULL,
            UNIQUE(setup_date, ticker, entry_price, stop_loss)
        )
        """
    )
    conn.commit()


def store_trade_setups(setups: pd.DataFrame, database_path: Path | str = DATABASE_PATH) -> None:
    conn = init_db(database_path)
    init_setup_history(conn)
    setup_date = date.today().isoformat()
    inserted_at = datetime.now(timezone.utc).isoformat()
    for _, row in setups.iterrows():
        conn.execute(
            """
            INSERT OR IGNORE INTO trade_setups(
                setup_date, ticker, trade_quality_score, setup_class, setup_type,
                entry_price, stop_loss, target_1, target_2, target_3,
                position_size, payload, inserted_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                setup_date,
                row["Ticker"],
                float(row["Trade Quality Score"]),
                row["Setup Class"],
                row["Setup Type"],
                float(row["Entry Price"]),
                float(row["Stop Loss"]),
                float(row["Target 1"]),
                float(row["Target 2"]),
                float(row["Target 3"]),
                int(row["Position Size"]),
                json.dumps(row.where(pd.notna(row), None).to_dict(), default=str),
                inserted_at,
            ),
        )
    conn.commit()
    conn.close()


def generate_trade_setups(
    candidates: pd.DataFrame,
    config: SetupConfig,
    ohlcv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ranked = candidates[pd.to_numeric(candidates["FinalScore"], errors="coerce") >= 50].head(config.top_ranked_limit)
    if ranked.empty:
        return pd.DataFrame()

    data = ohlcv if ohlcv is not None else _download_ohlcv(ranked["Symbol"], "18mo")
    setups: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for _, row in ranked.iterrows():
        frame = _frame_for_symbol(data, str(row["Symbol"]))
        if frame.empty:
            rejected.append({"Ticker": row["Symbol"], "Rejected Reasons": "No OHLCV data"})
            continue
        setup, rejection = evaluate_trade_setup(row, frame, candidates, config)
        if setup is not None:
            setups.append(setup)
        if rejection is not None:
            rejected.append(rejection)

    setup_frame = pd.DataFrame(setups)
    rejected_frame = pd.DataFrame(rejected)
    rejected_frame.to_csv(config.rejected_output_csv, index=False)
    if setup_frame.empty:
        return pd.DataFrame(columns=SETUP_COLUMNS)
    setup_frame = setup_frame.sort_values(["Trade Quality Score", "Final Research Score"], ascending=[False, False])
    return setup_frame.reset_index(drop=True)


def export_trade_setups(setups: pd.DataFrame, config: SetupConfig) -> None:
    setups.to_csv(config.output_csv, index=False)
    records = setups.to_dict(orient="records")
    config.output_json.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")


def generate_and_export_trade_setups(candidates: pd.DataFrame, config: SetupConfig) -> pd.DataFrame:
    setups = generate_trade_setups(candidates, config)
    export_trade_setups(setups, config)
    if not setups.empty:
        store_trade_setups(setups)
    return setups
