from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from trading_research import DATABASE_PATH, _download_ohlcv, _frame_for_symbol


HORIZONS = (1, 5, 10, 20, 60)
SCORE_BUCKETS = (
    ("90+", 90, 100),
    ("80-89", 80, 89.9999),
    ("70-79", 70, 79.9999),
)
FACTOR_COLUMNS = ("final_score", "technical_score", "catalyst_score", "relative_strength_score")
TRADE_COLUMNS = [
    "run_date",
    "symbol",
    "score_bucket",
    "horizon",
    "forward_return",
    "max_drawdown",
    "final_score",
    "technical_score",
    "catalyst_score",
    "relative_strength_score",
    "relative_volume",
    "return_3m",
    "relative_strength_3m",
]


@dataclass(frozen=True)
class PerformanceResult:
    trades: pd.DataFrame
    bucket_metrics: pd.DataFrame
    factor_value: pd.DataFrame
    recommendations: list[str]


def score_bucket(score: float) -> str:
    if score >= 90:
        return "90+"
    if score >= 80:
        return "80-89"
    if score >= 70:
        return "70-79"
    return ""


def max_drawdown(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan
    running_peak = values.cummax()
    drawdown = values / running_peak - 1
    return float(drawdown.min())


def sharpe_ratio(returns: pd.Series, horizon: int) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if len(returns) < 2:
        return np.nan
    std = returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float((returns.mean() / std) * np.sqrt(252 / horizon))


def load_ranking_events(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = pd.read_sql_query(
        """
        SELECT run_date, symbol, final_score, technical_score, catalyst_score,
               relative_strength_score, rank, payload
        FROM rankings
        WHERE final_score >= 70
        ORDER BY run_date, symbol
        """,
        conn,
    )
    if rows.empty:
        return rows

    payload_rows: list[dict[str, object]] = []
    for payload in rows["payload"]:
        try:
            payload_rows.append(json.loads(payload or "{}"))
        except json.JSONDecodeError:
            payload_rows.append({})

    payload_frame = pd.DataFrame(payload_rows)
    for column in ("RelativeVolume", "Return3M", "RelativeStrength3M"):
        rows[column] = pd.to_numeric(payload_frame.get(column), errors="coerce")

    rows["run_date"] = pd.to_datetime(rows["run_date"], errors="coerce")
    rows["score_bucket"] = rows["final_score"].apply(score_bucket)
    return rows.dropna(subset=["run_date"]).reset_index(drop=True)


def _date_position(index: pd.DatetimeIndex, signal_date: pd.Timestamp) -> int | None:
    normalized = pd.Timestamp(signal_date).normalize()
    positions = np.flatnonzero(index.normalize() >= normalized)
    if len(positions) == 0:
        return None
    return int(positions[0])


def _forward_trade_rows(events: pd.DataFrame, ohlcv: pd.DataFrame) -> list[dict[str, object]]:
    trade_rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        frame = _frame_for_symbol(ohlcv, event.symbol)
        if frame.empty or "Close" not in frame:
            continue

        frame = frame.sort_index().dropna(subset=["Close"])
        if frame.empty:
            continue

        position = _date_position(pd.DatetimeIndex(frame.index), event.run_date)
        if position is None:
            continue

        entry_close = float(frame["Close"].iloc[position])
        if entry_close <= 0:
            continue

        for horizon in HORIZONS:
            exit_position = position + horizon
            if exit_position >= len(frame):
                continue

            window = frame.iloc[position : exit_position + 1]
            exit_close = float(frame["Close"].iloc[exit_position])
            forward_return = exit_close / entry_close - 1
            drawdown = float(window["Close"].min() / entry_close - 1)

            trade_rows.append(
                {
                    "run_date": event.run_date.date().isoformat(),
                    "symbol": event.symbol,
                    "score_bucket": event.score_bucket,
                    "horizon": horizon,
                    "forward_return": forward_return,
                    "max_drawdown": drawdown,
                    "final_score": event.final_score,
                    "technical_score": event.technical_score,
                    "catalyst_score": event.catalyst_score,
                    "relative_strength_score": event.relative_strength_score,
                    "relative_volume": getattr(event, "RelativeVolume", np.nan),
                    "return_3m": getattr(event, "Return3M", np.nan),
                    "relative_strength_3m": getattr(event, "RelativeStrength3M", np.nan),
                }
            )
    return trade_rows


def calculate_bucket_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "score_bucket",
                "horizon",
                "observations",
                "win_rate",
                "average_return",
                "maximum_drawdown",
                "sharpe_ratio",
            ]
        )

    rows: list[dict[str, object]] = []
    for (bucket, horizon), group in trades.groupby(["score_bucket", "horizon"], dropna=False):
        returns = group["forward_return"]
        rows.append(
            {
                "score_bucket": bucket,
                "horizon": int(horizon),
                "observations": int(len(group)),
                "win_rate": round(float((returns > 0).mean()) * 100, 2),
                "average_return": round(float(returns.mean()) * 100, 2),
                "maximum_drawdown": round(float(group["max_drawdown"].min()) * 100, 2),
                "sharpe_ratio": round(sharpe_ratio(returns, int(horizon)), 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["score_bucket", "horizon"]).reset_index(drop=True)


def calculate_factor_value(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["factor", "horizon", "observations", "pearson", "spearman"])

    rows: list[dict[str, object]] = []
    factors = [
        "final_score",
        "technical_score",
        "catalyst_score",
        "relative_strength_score",
        "relative_volume",
        "return_3m",
        "relative_strength_3m",
    ]
    for horizon, group in trades.groupby("horizon"):
        for factor in factors:
            sample = group[[factor, "forward_return"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sample) < 3 or sample[factor].nunique() < 2:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = sample[factor].corr(sample["forward_return"], method="pearson")
                spearman = sample[factor].rank().corr(sample["forward_return"].rank(), method="pearson")
            rows.append(
                {
                    "factor": factor,
                    "horizon": int(horizon),
                    "observations": int(len(sample)),
                    "pearson": round(float(pearson), 4) if pd.notna(pearson) else np.nan,
                    "spearman": round(float(spearman), 4) if pd.notna(spearman) else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon", "spearman"], ascending=[True, False]).reset_index(drop=True)


def make_recommendations(bucket_metrics: pd.DataFrame, factor_value: pd.DataFrame) -> list[str]:
    recommendations: list[str] = []
    if bucket_metrics.empty:
        return [
            "Not enough matured forward-return observations yet. Keep storing daily rankings until at least the 20-day and 60-day windows have completed.",
            "Avoid changing score weights until each target bucket has a useful sample size, ideally 30 or more observations per horizon.",
        ]

    mature = bucket_metrics[bucket_metrics["observations"] >= 10]
    if mature.empty:
        recommendations.append("Sample size is still thin. Treat all performance statistics as directional, not conclusive.")

    usable_factors = factor_value.dropna(subset=["spearman"])
    if not usable_factors.empty:
        strongest = usable_factors.iloc[usable_factors["spearman"].abs().argmax()]
        recommendations.append(
            f"Highest observed predictive factor so far: {strongest['factor']} at the {int(strongest['horizon'])}-day horizon "
            f"(Spearman {strongest['spearman']})."
        )

    negative = usable_factors[usable_factors["spearman"] < -0.05]
    if not negative.empty:
        factors = ", ".join(sorted(negative["factor"].unique()))
        recommendations.append(f"Consider reducing or transforming negatively correlated factors: {factors}.")

    positive = usable_factors[usable_factors["spearman"] > 0.05]
    if not positive.empty:
        factors = ", ".join(sorted(positive["factor"].unique()))
        recommendations.append(f"Consider modestly increasing weight on persistently positive factors: {factors}.")

    by_bucket = bucket_metrics.groupby("score_bucket")["average_return"].mean().sort_values(ascending=False)
    if len(by_bucket) >= 2 and by_bucket.index[0] != "90+":
        recommendations.append(
            "The highest score bucket is not leading yet. Recalibrate score thresholds or add penalties for low-liquidity/no-data names before using 90+ as the strongest signal."
        )

    return recommendations or ["No adjustment recommended yet; continue collecting daily ranking history."]


def analyze_performance(database_path: Path | str = DATABASE_PATH, price_period: str = "5y") -> PerformanceResult:
    conn = sqlite3.connect(database_path)
    events = load_ranking_events(conn)
    conn.close()

    if events.empty:
        empty_trades = pd.DataFrame(columns=TRADE_COLUMNS)
        empty_metrics = calculate_bucket_metrics(empty_trades)
        empty_factors = calculate_factor_value(empty_trades)
        return PerformanceResult(empty_trades, empty_metrics, empty_factors, make_recommendations(empty_metrics, empty_factors))

    symbols = sorted(events["symbol"].unique())
    ohlcv = _download_ohlcv(symbols, price_period)
    trades = pd.DataFrame(_forward_trade_rows(events, ohlcv), columns=TRADE_COLUMNS)
    bucket_metrics = calculate_bucket_metrics(trades)
    factor_value = calculate_factor_value(trades)
    recommendations = make_recommendations(bucket_metrics, factor_value)
    return PerformanceResult(trades, bucket_metrics, factor_value, recommendations)


def write_performance_outputs(result: PerformanceResult, output_dir: Path | str = ".") -> None:
    output_dir = Path(output_dir)
    result.trades.to_csv(output_dir / "ranking_forward_returns.csv", index=False)
    result.bucket_metrics.to_csv(output_dir / "ranking_performance_summary.csv", index=False)
    result.factor_value.to_csv(output_dir / "ranking_factor_value.csv", index=False)
    (output_dir / "ranking_recommendations.txt").write_text("\n".join(result.recommendations), encoding="utf-8")


def load_setup_events(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        rows = pd.read_sql_query(
            """
            SELECT setup_date, ticker AS symbol, trade_quality_score, setup_class, setup_type,
                   entry_price, stop_loss, target_1, target_2, target_3, payload
            FROM trade_setups
            ORDER BY setup_date, ticker
            """,
            conn,
        )
    except Exception:
        return pd.DataFrame()
    if rows.empty:
        return rows
    rows["setup_date"] = pd.to_datetime(rows["setup_date"], errors="coerce")
    return rows.dropna(subset=["setup_date"]).reset_index(drop=True)


def analyze_setup_performance(database_path: Path | str = DATABASE_PATH, price_period: str = "5y") -> PerformanceResult:
    conn = sqlite3.connect(database_path)
    events = load_setup_events(conn)
    conn.close()
    if events.empty:
        empty_trades = pd.DataFrame(columns=TRADE_COLUMNS)
        empty_metrics = calculate_bucket_metrics(empty_trades)
        empty_factors = calculate_factor_value(empty_trades)
        return PerformanceResult(empty_trades, empty_metrics, empty_factors, ["No generated trade setups have been stored yet."])

    symbols = sorted(events["symbol"].unique())
    ohlcv = _download_ohlcv(symbols, price_period)
    synthetic_events = events.rename(
        columns={
            "setup_date": "run_date",
            "trade_quality_score": "final_score",
        }
    )
    synthetic_events["technical_score"] = np.nan
    synthetic_events["catalyst_score"] = np.nan
    synthetic_events["relative_strength_score"] = np.nan
    synthetic_events["score_bucket"] = synthetic_events["final_score"].apply(score_bucket)
    synthetic_events = synthetic_events[synthetic_events["score_bucket"] != ""]
    trades = pd.DataFrame(_forward_trade_rows(synthetic_events, ohlcv), columns=TRADE_COLUMNS)
    bucket_metrics = calculate_bucket_metrics(trades)
    factor_value = calculate_factor_value(trades)
    recommendations = make_recommendations(bucket_metrics, factor_value)
    return PerformanceResult(trades, bucket_metrics, factor_value, recommendations)


if __name__ == "__main__":
    performance = analyze_performance(DATABASE_PATH)
    write_performance_outputs(performance)
    print(f"Forward-return rows: {len(performance.trades)}")
    print(f"Performance summary rows: {len(performance.bucket_metrics)}")
