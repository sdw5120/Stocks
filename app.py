from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

from trading_research import (
    init_db,
    latest_catalysts,
    latest_news,
    load_config,
    load_watchlist,
    ranking_history,
    build_candidates,
)
from performance_analysis import analyze_performance, analyze_setup_performance
from trade_setups import (
    REJECTED_SETUPS_CSV,
    SETUPS_CSV,
    SETUPS_JSON,
    SetupConfig,
    export_trade_setups,
    generate_trade_setups,
    take_profit_plan,
)


CONFIG = load_config()
WATCHLIST_PATH = CONFIG.watchlist_path
OUTPUT_PATH = CONFIG.output_path
DATABASE_PATH = CONFIG.database_path
SNAPSHOT_DIR = Path("data")


st.set_page_config(page_title="Swing Trading Research", layout="wide")
st.title("Swing Trading Research Dashboard")
st.caption("Research and alerts only. This app does not connect to a brokerage or place trades.")


@st.cache_data(show_spinner=False)
def cached_watchlist(watchlist_mtime: float) -> pd.DataFrame:
    return load_watchlist(WATCHLIST_PATH)


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def snapshot_path(path: Path) -> Path:
    return SNAPSHOT_DIR / path.name


def read_csv_with_snapshot(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    fallback = snapshot_path(path)
    if fallback.exists():
        return pd.read_csv(fallback)
    return pd.DataFrame()


def snapshot_mtime(path: Path) -> float:
    return max(file_mtime(path), file_mtime(snapshot_path(path)))


@st.cache_data(show_spinner=False)
def cached_candidates(refresh_key: int, output_mtime: float) -> pd.DataFrame:
    if OUTPUT_PATH.exists() and refresh_key == 0:
        existing = pd.read_csv(OUTPUT_PATH)
        if "FinalScore" in existing.columns:
            return existing
    snapshot = snapshot_path(OUTPUT_PATH)
    if snapshot.exists() and refresh_key == 0:
        existing = pd.read_csv(snapshot)
        if "FinalScore" in existing.columns:
            return existing
    return build_candidates(CONFIG)


def refresh_candidates_with_progress() -> pd.DataFrame:
    progress_bar = st.progress(0, text="Starting refresh")
    status_text = st.empty()
    started_at = time.monotonic()

    def update_progress(stage: str, current: int, total: int, message: str) -> None:
        total = max(total, 1)
        current = min(max(current, 0), total)
        percent = int(current / total * 100)
        elapsed = time.monotonic() - started_at
        eta = ""
        if percent > 3:
            remaining = max((elapsed / percent) * (100 - percent), 0)
            eta = f" | about {remaining / 60:.1f} min left" if remaining >= 60 else f" | about {remaining:.0f} sec left"
        progress_bar.progress(percent, text=f"{stage}: {message}")
        status_text.caption(f"{percent}% complete | elapsed {elapsed / 60:.1f} min{eta}")

    with st.status("Refreshing research data", expanded=True) as status:
        status.write("Pulling market data, earnings, news, catalysts, rankings, and alerts.")
        candidates_frame = build_candidates(CONFIG, progress_callback=update_progress)
        status.update(label="Research data refreshed", state="complete", expanded=False)
    progress_bar.progress(100, text="Refresh complete")
    return candidates_frame


@st.cache_data(show_spinner=False)
def cached_db_tables(refresh_key: int, database_mtime: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not DATABASE_PATH.exists():
        return cached_snapshot_db_tables(database_mtime)
    conn = init_db(DATABASE_PATH)
    earnings = pd.read_sql_query(
        """
        SELECT symbol, earnings_date, days_until_earnings, earnings_within_14_days, updated_at
        FROM earnings
        WHERE earnings_date IS NOT NULL AND earnings_date != ''
        ORDER BY days_until_earnings ASC
        """,
        conn,
    )
    news = latest_news(conn, 500)
    catalysts = latest_catalysts(conn, 100)
    history = ranking_history(conn, 1000)
    conn.close()
    if earnings.empty and news.empty and catalysts.empty and history.empty:
        return cached_snapshot_db_tables(database_mtime)
    return earnings, news, catalysts, history


@st.cache_data(show_spinner=False)
def cached_snapshot_db_tables(database_mtime: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    earnings = pd.read_csv(SNAPSHOT_DIR / "earnings.csv") if (SNAPSHOT_DIR / "earnings.csv").exists() else pd.DataFrame()
    news = pd.read_csv(SNAPSHOT_DIR / "news.csv") if (SNAPSHOT_DIR / "news.csv").exists() else pd.DataFrame()
    catalysts = pd.DataFrame()
    if not news.empty and "catalyst_score" in news.columns:
        catalyst_score = pd.to_numeric(news["catalyst_score"], errors="coerce").fillna(0)
        catalysts = news[catalyst_score > 0].copy()
        if "published_at" in catalysts.columns:
            catalysts = catalysts.sort_values(["catalyst_score", "published_at"], ascending=[False, False])
        else:
            catalysts = catalysts.sort_values("catalyst_score", ascending=False)
        catalysts = catalysts.head(100)
    history = pd.read_csv(SNAPSHOT_DIR / "rankings_history.csv") if (SNAPSHOT_DIR / "rankings_history.csv").exists() else pd.DataFrame()
    return earnings, news, catalysts, history


@st.cache_data(show_spinner=False)
def cached_performance(refresh_key: int, database_mtime: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    if refresh_key == 0 and Path("ranking_performance_summary.csv").exists():
        return cached_saved_performance(database_mtime, "ranking", Path("."))
    if not DATABASE_PATH.exists() and (SNAPSHOT_DIR / "ranking_performance_summary.csv").exists():
        return cached_snapshot_performance(database_mtime, "ranking")
    result = analyze_performance(DATABASE_PATH)
    if result.bucket_metrics.empty and (SNAPSHOT_DIR / "ranking_performance_summary.csv").exists():
        return cached_snapshot_performance(database_mtime, "ranking")
    return result.trades, result.bucket_metrics, result.factor_value, result.recommendations


@st.cache_data(show_spinner=False)
def cached_setup_performance(refresh_key: int, database_mtime: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    if refresh_key == 0 and Path("setup_performance_summary.csv").exists():
        return cached_saved_performance(database_mtime, "setup", Path("."))
    if not DATABASE_PATH.exists() and (SNAPSHOT_DIR / "setup_performance_summary.csv").exists():
        return cached_snapshot_performance(database_mtime, "setup")
    result = analyze_setup_performance(DATABASE_PATH)
    if result.bucket_metrics.empty and (SNAPSHOT_DIR / "setup_performance_summary.csv").exists():
        return cached_snapshot_performance(database_mtime, "setup")
    return result.trades, result.bucket_metrics, result.factor_value, result.recommendations


@st.cache_data(show_spinner=False)
def cached_snapshot_performance(database_mtime: float, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    return cached_saved_performance(database_mtime, prefix, SNAPSHOT_DIR)


@st.cache_data(show_spinner=False)
def cached_saved_performance(database_mtime: float, prefix: str, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    if prefix == "ranking":
        trades_path = output_dir / "ranking_forward_returns.csv"
        summary_path = output_dir / "ranking_performance_summary.csv"
        factors_path = output_dir / "ranking_factor_value.csv"
        recommendations_path = output_dir / "ranking_recommendations.txt"
    else:
        trades_path = output_dir / "setup_forward_returns.csv"
        summary_path = output_dir / "setup_performance_summary.csv"
        factors_path = output_dir / "setup_factor_value.csv"
        recommendations_path = output_dir / "setup_recommendations.txt"

    trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    factors = pd.read_csv(factors_path) if factors_path.exists() else pd.DataFrame()
    recommendations = (
        recommendations_path.read_text(encoding="utf-8").splitlines()
        if recommendations_path.exists()
        else ["No snapshot recommendations are available yet."]
    )
    return trades, summary, factors, recommendations


@st.cache_data(show_spinner=False)
def cached_trade_setups(
    refresh_key: int,
    output_mtime: float,
    portfolio_size: float,
    max_risk_percent: float,
) -> pd.DataFrame:
    existing = read_csv_with_snapshot(SETUPS_CSV)
    if not existing.empty or SETUPS_CSV.exists() or snapshot_path(SETUPS_CSV).exists():
        return apply_risk_inputs(existing, portfolio_size, max_risk_percent)
    candidates = cached_candidates(refresh_key, output_mtime)
    setup_config = SetupConfig(portfolio_size=portfolio_size, max_risk_percent=max_risk_percent)
    setups = generate_trade_setups(candidates, setup_config)
    export_trade_setups(setups, setup_config)
    return apply_risk_inputs(setups, portfolio_size, max_risk_percent)


@st.cache_data(show_spinner=False)
def cached_rejected_trade_setups(rejected_mtime: float, portfolio_size: float, max_risk_percent: float) -> pd.DataFrame:
    rejected = read_csv_with_snapshot(REJECTED_SETUPS_CSV)
    if rejected.empty and not REJECTED_SETUPS_CSV.exists() and not snapshot_path(REJECTED_SETUPS_CSV).exists():
        return pd.DataFrame()
    return apply_risk_inputs(rejected, portfolio_size, max_risk_percent)


def regenerate_trade_setup_files(candidates: pd.DataFrame, portfolio_size: float, max_risk_percent: float) -> pd.DataFrame:
    setup_config = SetupConfig(portfolio_size=portfolio_size, max_risk_percent=max_risk_percent)
    setups = generate_trade_setups(candidates, setup_config)
    export_trade_setups(setups, setup_config)
    cached_trade_setups.clear()
    cached_rejected_trade_setups.clear()
    return apply_risk_inputs(setups, portfolio_size, max_risk_percent)


def apply_risk_inputs(setups: pd.DataFrame, portfolio_size: float, max_risk_percent: float) -> pd.DataFrame:
    if setups.empty or "Entry Price" not in setups.columns or "Stop Loss" not in setups.columns:
        return setups

    adjusted = setups.copy()
    risk_dollars = round(float(portfolio_size) * (float(max_risk_percent) / 100), 2)
    adjusted["Portfolio Size"] = float(portfolio_size)
    adjusted["Max Risk %"] = float(max_risk_percent)
    adjusted["Max Dollar Risk"] = risk_dollars

    for index, row in adjusted.iterrows():
        entry = pd.to_numeric(row.get("Entry Price"), errors="coerce")
        stop = pd.to_numeric(row.get("Stop Loss"), errors="coerce")
        target_2 = pd.to_numeric(row.get("Target 2"), errors="coerce")
        direction = str(row.get("Direction", "LONG")).upper()
        if pd.isna(entry) or pd.isna(stop):
            continue

        risk_per_share = stop - entry if direction == "SHORT" else entry - stop
        if pd.isna(risk_per_share) or risk_per_share <= 0:
            shares = 0
        else:
            shares = int(np.floor(risk_dollars / risk_per_share))

        adjusted.at[index, "Risk Per Share"] = round(float(risk_per_share), 2) if risk_per_share > 0 else np.nan
        adjusted.at[index, "Position Size"] = shares
        adjusted.at[index, "Position Cost"] = round(shares * float(entry), 2)

        if pd.notna(target_2):
            reward_per_share = entry - target_2 if direction == "SHORT" else target_2 - entry
            adjusted.at[index, "Reward Per Share"] = round(float(reward_per_share), 2)

        first_slice, second_slice, third_slice = take_profit_plan(shares)
        if direction == "SHORT":
            adjusted.at[index, "Shares To Sell At Target 1"] = 0
            adjusted.at[index, "Shares To Sell At Target 2"] = 0
            adjusted.at[index, "Shares To Sell At Target 3"] = 0
            adjusted.at[index, "Shares To Cover At Target 1"] = first_slice
            adjusted.at[index, "Shares To Cover At Target 2"] = second_slice
            adjusted.at[index, "Shares To Cover At Target 3"] = third_slice
        else:
            adjusted.at[index, "Shares To Sell At Target 1"] = first_slice
            adjusted.at[index, "Shares To Sell At Target 2"] = second_slice
            adjusted.at[index, "Shares To Sell At Target 3"] = third_slice
            adjusted.at[index, "Shares To Cover At Target 1"] = 0
            adjusted.at[index, "Shares To Cover At Target 2"] = 0
            adjusted.at[index, "Shares To Cover At Target 3"] = 0

    return adjusted


with st.sidebar:
    st.header("Controls")
    refresh = st.button("Refresh research data", type="primary")
    min_score = st.slider("Minimum final score", 0, 100, 50)
    top_n = st.number_input("Rows to show", min_value=5, max_value=250, value=50, step=5)

    st.divider()
    watchlist = cached_watchlist(file_mtime(WATCHLIST_PATH))
    st.metric("Watchlist tickers", len(watchlist))
    st.caption(f"Database: {DATABASE_PATH}")

refresh_key = 1 if refresh else 0
if refresh:
    cached_candidates.clear()
    cached_db_tables.clear()
    cached_performance.clear()
    cached_setup_performance.clear()
    cached_trade_setups.clear()

try:
    if refresh:
        candidates = refresh_candidates_with_progress()
    else:
        candidates = cached_candidates(refresh_key, snapshot_mtime(OUTPUT_PATH))
except Exception as exc:
    st.error(str(exc))
    st.info("Install dependencies with: py -3 -m pip install -r requirements.txt")
    st.stop()

earnings, news, catalysts, history = cached_db_tables(refresh_key, snapshot_mtime(DATABASE_PATH))
perf_trades, perf_summary, factor_value, perf_recommendations = cached_performance(refresh_key, snapshot_mtime(DATABASE_PATH))
setup_perf_trades, setup_perf_summary, setup_factor_value, setup_perf_recommendations = cached_setup_performance(
    refresh_key, snapshot_mtime(DATABASE_PATH)
)
filtered = candidates[candidates["FinalScore"] >= min_score].head(int(top_n))
earnings_flag = (
    candidates["EarningsWithin14Days"].fillna(False).astype(bool)
    if "EarningsWithin14Days" in candidates.columns
    else pd.Series(False, index=candidates.index)
)

metric_cols = st.columns(5)
metric_cols[0].metric("Tickers", len(candidates))
metric_cols[1].metric("Buy Candidates", int((candidates["Rating"] == "Buy Candidate").sum()))
metric_cols[2].metric("Watch", int((candidates["Rating"] == "Watch").sum()))
metric_cols[3].metric("Top Final Score", f"{candidates['FinalScore'].max():.1f}")
metric_cols[4].metric("Upcoming Earnings", int(earnings_flag.sum()))

top_10 = candidates.head(10)
upcoming_earnings = candidates[earnings_flag].sort_values("DaysUntilEarnings")
strong_rs = candidates.sort_values("RelativeStrengthScore", ascending=False).head(25)
if "RelativeVolume" in candidates.columns:
    relative_volume = pd.to_numeric(candidates["RelativeVolume"], errors="coerce")
    high_volume = candidates[relative_volume > 1.5].sort_values("RelativeVolume", ascending=False)
else:
    high_volume = candidates.iloc[0:0].copy()

tabs = st.tabs(
    [
        "Top 10 Candidates",
        "Ranked Watchlist",
        "Upcoming Earnings",
        "Recent Catalysts",
        "Strong RS",
        "High Volume",
        "News",
        "History",
        "Performance",
        "Trade Setups",
    ]
)

score_columns = {
    "FinalScore": st.column_config.ProgressColumn("Final", min_value=0, max_value=100),
    "TechnicalScore": st.column_config.ProgressColumn("Technical", min_value=0, max_value=100),
    "RelativeStrengthScore": st.column_config.ProgressColumn("RS", min_value=0, max_value=100),
    "CatalystScore": st.column_config.NumberColumn("Catalyst", min_value=0, max_value=10, format="%.1f"),
    "RelativeVolume": st.column_config.NumberColumn("Rel Vol", format="%.2f"),
}

with tabs[0]:
    st.subheader("Top 10 Candidates")
    st.dataframe(top_10, use_container_width=True, hide_index=True, column_config=score_columns)

with tabs[1]:
    st.subheader("Ranked Watchlist")
    st.dataframe(filtered, use_container_width=True, hide_index=True, column_config=score_columns)
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("rb") as output_file:
            st.download_button("Download daily_candidates.csv", output_file, file_name=OUTPUT_PATH.name, mime="text/csv")

with tabs[2]:
    st.subheader("Upcoming Earnings")
    if upcoming_earnings.empty:
        st.info("No upcoming earnings inside the configured window.")
    else:
        st.dataframe(
            upcoming_earnings[
                ["Symbol", "Tier", "Category", "EarningsDate", "DaysUntilEarnings", "FinalScore", "Rating"]
            ],
            use_container_width=True,
            hide_index=True,
            column_config=score_columns,
        )
    if not earnings.empty:
        st.caption("All stored upcoming earnings")
        st.dataframe(earnings, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Recent Catalysts")
    if catalysts.empty:
        st.info("No catalyst headlines stored yet.")
    else:
        st.dataframe(catalysts, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Strong Relative Strength Stocks")
    st.dataframe(
        strong_rs[
            ["Rank", "Symbol", "FinalScore", "RelativeStrengthScore", "RelativeStrength3M", "Rating", "Tier", "Category"]
        ],
        use_container_width=True,
        hide_index=True,
        column_config=score_columns,
    )

with tabs[5]:
    st.subheader("High Volume Movers")
    if high_volume.empty:
        st.info("No stocks above the high-volume threshold.")
    else:
        st.dataframe(
            high_volume[
                ["Rank", "Symbol", "FinalScore", "RelativeVolume", "Volume", "Close", "Rating", "Tier", "Category"]
            ],
            use_container_width=True,
            hide_index=True,
            column_config=score_columns,
        )

with tabs[6]:
    st.subheader("Latest Headlines")
    symbol_options = ["All"] + sorted(watchlist["Symbol"].tolist())
    selected_symbol = st.selectbox("Ticker", symbol_options)
    if selected_symbol == "All" and not news.empty:
        shown_news = news.groupby("symbol", group_keys=False).head(5)
    else:
        shown_news = news[news["symbol"] == selected_symbol].head(5) if not news.empty else news
    if shown_news.empty:
        st.info("No stored headlines yet. Refresh research data to pull headlines.")
    else:
        st.dataframe(shown_news.head(100), use_container_width=True, hide_index=True)

with tabs[7]:
    st.subheader("Historical Rankings")
    if history.empty:
        st.info("No ranking history stored yet.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)

with tabs[8]:
    st.subheader("Ranking System Performance")
    st.caption("Forward returns are calculated only after each horizon has matured.")

    if perf_summary.empty:
        st.info("No matured 70+ score observations yet for 1/5/10/20/60-day forward-return analysis.")
    else:
        st.dataframe(
            perf_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "win_rate": st.column_config.NumberColumn("Win Rate %", format="%.2f"),
                "average_return": st.column_config.NumberColumn("Average Return %", format="%.2f"),
                "maximum_drawdown": st.column_config.NumberColumn("Max Drawdown %", format="%.2f"),
                "sharpe_ratio": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            },
        )

    st.subheader("Predictive Factor Value")
    if factor_value.empty:
        st.info("No factor-value correlations yet.")
    else:
        st.dataframe(factor_value, use_container_width=True, hide_index=True)

    st.subheader("Recommendations")
    for recommendation in perf_recommendations:
        st.write(f"- {recommendation}")

    if not perf_trades.empty:
        st.subheader("Forward Return Samples")
        st.dataframe(perf_trades.head(500), use_container_width=True, hide_index=True)

    st.subheader("Generated Trade Setup Performance")
    if setup_perf_summary.empty:
        st.info("No matured generated trade setups yet.")
    else:
        st.dataframe(setup_perf_summary, use_container_width=True, hide_index=True)
    for recommendation in setup_perf_recommendations:
        st.write(f"- {recommendation}")

@st.fragment
def render_trade_setups_tab(output_mtime: float, rejected_mtime: float) -> None:
    with st.form("trade_setup_risk_form"):
        input_cols = st.columns(2)
        portfolio_size = input_cols[0].number_input(
            "Portfolio size",
            min_value=1_000.0,
            max_value=100_000_000.0,
            value=100_000.0,
            step=1_000.0,
            key="trade_setups_portfolio_size",
        )
        max_risk_percent = input_cols[1].number_input(
            "Max risk per trade %",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key="trade_setups_max_risk_percent",
        )
        st.form_submit_button("Recalculate trade setups", type="primary")

    regenerate = st.button(
        "Regenerate trade setups from latest research data",
        help="Rebuild current price, entry, stop, targets, and rejected setup candidates from the latest ranked watchlist and OHLCV data.",
    )
    if regenerate:
        with st.status("Regenerating trade setups", expanded=True) as status:
            status.write("Loading latest ranked candidates.")
            latest_candidates = cached_candidates(0, snapshot_mtime(OUTPUT_PATH))
            status.write("Pulling OHLCV and rebuilding setup prices, entries, stops, targets, and filters.")
            trade_setups = regenerate_trade_setup_files(latest_candidates, portfolio_size, max_risk_percent)
            status.update(label="Trade setups regenerated", state="complete", expanded=False)
        rejected_trade_setups = cached_rejected_trade_setups(file_mtime(REJECTED_SETUPS_CSV), portfolio_size, max_risk_percent)
        st.success("Trade setups regenerated from the latest research data.")
    else:
        trade_setups = cached_trade_setups(0, output_mtime, portfolio_size, max_risk_percent)
        rejected_trade_setups = cached_rejected_trade_setups(rejected_mtime, portfolio_size, max_risk_percent)

    st.subheader("Trade Setups")
    st.caption("Research-only setups. The system does not connect to a brokerage or place trades.")

    if trade_setups.empty:
        st.info("No setups passed the safety filters. Try refreshing market data or lowering risk only after reviewing the filters.")
    else:
        visible_columns = [
            "Ticker",
            "Company",
            "Direction",
            "Setup Class",
            "Setup Type",
            "Market Regime",
            "Current Price",
            "Entry Price",
            "Stop Loss",
            "Target 1",
            "Target 2",
            "Target 3",
            "Risk/Reward T2",
            "Position Size",
            "Position Cost",
            "Max Dollar Risk",
            "Trade Quality Score",
            "Timeframe",
            "Estimated Holding Period",
            "Expected Time to Target 1",
            "Expected Time to Target 2",
            "Expected Time to Target 3",
            "Holding Period Confidence",
        ]
        st.dataframe(
            trade_setups[visible_columns].head(10),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Trade Quality Score": st.column_config.ProgressColumn("Quality", min_value=0, max_value=100),
                "Risk/Reward T2": st.column_config.NumberColumn("R/R T2", format="%.2f"),
            },
        )

        selected_ticker = st.selectbox("Setup details", trade_setups["Ticker"].tolist())
        selected_setup = trade_setups[trade_setups["Ticker"] == selected_ticker].iloc[0]
        detail_cols = st.columns(3)
        detail_cols[0].metric("Entry", f"{selected_setup['Entry Price']:.2f}")
        detail_cols[1].metric("Stop", f"{selected_setup['Stop Loss']:.2f}")
        detail_cols[2].metric("Quality", f"{selected_setup['Trade Quality Score']:.1f}")
        st.write(f"**Estimated Holding Period:** {selected_setup['Estimated Holding Period']}")
        st.write(f"**Expected Time to Target 1:** {selected_setup['Expected Time to Target 1']}")
        st.write(f"**Expected Time to Target 2:** {selected_setup['Expected Time to Target 2']}")
        st.write(f"**Expected Time to Target 3:** {selected_setup['Expected Time to Target 3']}")
        st.write(f"**Confidence:** {selected_setup['Holding Period Confidence']}")
        st.write(f"**Bull Thesis:** {selected_setup['Bull Thesis']}")
        st.write(f"**Bear Thesis:** {selected_setup['Bear Thesis']}")
        st.write(f"**Catalysts:** {selected_setup['Catalysts']}")
        st.write(f"**Invalidation:** {selected_setup['Invalidation']}")
        st.write(f"**Take-Profit Notes:** {selected_setup['Take-Profit Notes']}")
        if selected_setup.get("Direction") == "SHORT":
            st.write(f"**Short Sale Warnings:** {selected_setup.get('Short Sale Warnings', '')}")
        st.write(f"**Notes:** {selected_setup['Notes']}")

    st.subheader("Rejected Setup Candidates")
    if rejected_trade_setups.empty:
        st.info("No rejected setup candidates are available yet. Refresh research data after market data loads.")
    else:
        rejected_columns = [
            column
            for column in [
                "Ticker",
                "Direction",
                "Setup Type",
                "Market Regime",
                "Current Price",
                "Entry Price",
                "Stop Loss",
                "Target 1",
                "Target 2",
                "Target 3",
                "Risk/Reward T2",
                "Trade Quality Score",
                "Long Trade Quality Score",
                "Short Trade Quality Score",
                "Estimated Holding Period",
                "Expected Time to Target 1",
                "Expected Time to Target 2",
                "Expected Time to Target 3",
                "Holding Period Confidence",
                "Rejected Reasons",
            ]
            if column in rejected_trade_setups.columns
        ]
        st.dataframe(
            rejected_trade_setups[rejected_columns].head(50),
            use_container_width=True,
            hide_index=True,
        )

    export_cols = st.columns(2)
    if SETUPS_CSV.exists():
        with SETUPS_CSV.open("rb") as setup_csv:
            export_cols[0].download_button("Download trade_setups.csv", setup_csv, file_name=SETUPS_CSV.name, mime="text/csv")
    if SETUPS_JSON.exists():
        with SETUPS_JSON.open("rb") as setup_json:
            export_cols[1].download_button("Download trade_setups.json", setup_json, file_name=SETUPS_JSON.name, mime="application/json")


with tabs[9]:
    render_trade_setups_tab(snapshot_mtime(OUTPUT_PATH), snapshot_mtime(REJECTED_SETUPS_CSV))
