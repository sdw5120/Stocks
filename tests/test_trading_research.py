from __future__ import annotations

import sqlite3
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from trading_research import (
    calculate_technical_metrics,
    classify,
    detect_catalyst,
    generate_alerts,
    init_db,
    normalize_headline,
    percentile_score,
    store_news,
)
from performance_analysis import (
    calculate_bucket_metrics,
    calculate_factor_value,
    max_drawdown,
    score_bucket,
    sharpe_ratio,
)
from trade_setups import (
    SetupConfig,
    calculate_stop_loss,
    classify_entry_type,
    position_size,
    risk_reward,
    setup_filter_reasons,
    setup_class,
    take_profit_plan,
)


class TradingResearchTests(unittest.TestCase):
    def test_detect_catalyst_categories(self) -> None:
        category, score = detect_catalyst("Company wins government contract and raises guidance")

        self.assertIn("Government Contract", category)
        self.assertIn("Earnings Beat", category)
        self.assertEqual(score, 9)

    def test_duplicate_news_is_ignored(self) -> None:
        conn = init_db(":memory:")
        item = {
            "headline": "Acme announces new product launch",
            "publisher": "Example",
            "link": "https://example.com/news",
            "published_at": "2026-06-17T12:00:00Z",
        }

        first_score = store_news(conn, "ACME", [item])
        second_score = store_news(conn, "ACME", [item])
        count = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]

        self.assertEqual(count, 1)
        self.assertEqual(first_score, second_score)

    def test_technical_metrics_detects_alert_conditions(self) -> None:
        index = pd.date_range(end=datetime(2026, 6, 17), periods=260, freq="B")
        close = np.linspace(80, 120, len(index))
        close[-2] = 99
        close[-1] = 130
        high = close + 1
        high[-1] = 130
        volume = np.full(len(index), 1_000_000)
        volume[-1] = 2_500_000
        frame = pd.DataFrame(
            {
                "Open": close - 0.5,
                "High": high,
                "Low": close - 1,
                "Close": close,
                "Volume": volume,
            },
            index=index,
        )

        metrics = calculate_technical_metrics(frame, spy_return_3m=0.02)

        self.assertGreater(metrics["TechnicalScore"], 0)
        self.assertTrue(metrics["CrossedMA50"])
        self.assertTrue(metrics["CrossedMA200"])
        self.assertTrue(metrics["Made52WeekHigh"])
        self.assertGreater(metrics["RelativeVolume"], 2)

    def test_alert_generation(self) -> None:
        row = pd.Series(
            {
                "Symbol": "ACME",
                "RelativeVolume": 2.5,
                "CrossedMA50": True,
                "CrossedMA200": False,
                "Made52WeekHigh": True,
                "CatalystScore": 8,
            }
        )

        alerts = generate_alerts(row)
        alert_types = {alert_type for alert_type, _ in alerts}

        self.assertIn("Relative volume > 2x", alert_types)
        self.assertIn("50-day MA cross", alert_types)
        self.assertIn("52-week high", alert_types)
        self.assertIn("Catalyst score >= 8", alert_types)

    def test_classification_thresholds(self) -> None:
        self.assertEqual(classify(80), "Buy Candidate")
        self.assertEqual(classify(60), "Watch")
        self.assertEqual(classify(20), "Pass")

    def test_percentile_score_handles_missing_values(self) -> None:
        scores = percentile_score(pd.Series([np.nan, 1.0, 3.0, 2.0]))

        self.assertTrue(np.isnan(scores.iloc[0]))
        self.assertEqual(scores.iloc[2], 100.0)

    def test_normalized_headline_is_stable(self) -> None:
        self.assertEqual(normalize_headline(" Big   News "), normalize_headline("big news"))

    def test_score_bucket(self) -> None:
        self.assertEqual(score_bucket(95), "90+")
        self.assertEqual(score_bucket(85), "80-89")
        self.assertEqual(score_bucket(72), "70-79")
        self.assertEqual(score_bucket(69), "")

    def test_max_drawdown(self) -> None:
        result = max_drawdown(pd.Series([100, 110, 90, 120]))

        self.assertAlmostEqual(result, -0.1818, places=4)

    def test_sharpe_ratio_requires_variance(self) -> None:
        self.assertTrue(np.isnan(sharpe_ratio(pd.Series([0.01]), 5)))
        self.assertTrue(np.isnan(sharpe_ratio(pd.Series([0.01, 0.01]), 5)))
        self.assertFalse(np.isnan(sharpe_ratio(pd.Series([0.01, 0.03, -0.01]), 5)))

    def test_bucket_metrics(self) -> None:
        trades = pd.DataFrame(
            [
                {"score_bucket": "90+", "horizon": 5, "forward_return": 0.10, "max_drawdown": -0.02},
                {"score_bucket": "90+", "horizon": 5, "forward_return": -0.05, "max_drawdown": -0.08},
                {"score_bucket": "80-89", "horizon": 5, "forward_return": 0.02, "max_drawdown": -0.01},
            ]
        )

        metrics = calculate_bucket_metrics(trades)
        row = metrics[(metrics["score_bucket"] == "90+") & (metrics["horizon"] == 5)].iloc[0]

        self.assertEqual(row["observations"], 2)
        self.assertEqual(row["win_rate"], 50.0)
        self.assertEqual(row["average_return"], 2.5)
        self.assertEqual(row["maximum_drawdown"], -8.0)

    def test_factor_value_correlations(self) -> None:
        trades = pd.DataFrame(
            {
                "horizon": [10, 10, 10, 10],
                "forward_return": [0.01, 0.03, 0.05, 0.07],
                "final_score": [70, 80, 90, 95],
                "technical_score": [10, 20, 30, 40],
                "catalyst_score": [0, 1, 2, 3],
                "relative_strength_score": [40, 50, 60, 70],
                "relative_volume": [1, 2, 3, 4],
                "return_3m": [5, 6, 7, 8],
                "relative_strength_3m": [1, 2, 3, 4],
            }
        )

        factors = calculate_factor_value(trades)
        final_score_row = factors[factors["factor"] == "final_score"].iloc[0]

        self.assertGreater(final_score_row["spearman"], 0.9)

    def test_position_sizing(self) -> None:
        shares, max_risk = position_size(100_000, 1.0, 50.0, 45.0)

        self.assertEqual(shares, 200)
        self.assertEqual(max_risk, 1000.0)

    def test_stop_loss_uses_conservative_candidate(self) -> None:
        stop = calculate_stop_loss(entry_price=100, support=96, ma20=94, ma50=92, atr=3)

        self.assertEqual(stop, 91.08)

    def test_risk_reward_calculation(self) -> None:
        self.assertEqual(risk_reward(100, 95, 110), 2.0)
        self.assertTrue(np.isnan(risk_reward(100, 100, 110)))

    def test_entry_type_classification(self) -> None:
        setup_type, entry = classify_entry_type(
            current_price=99,
            ma20=95,
            ma50=90,
            ma200=80,
            support=94,
            resistance=100,
            atr=2,
        )

        self.assertEqual(setup_type, "Breakout entry")
        self.assertGreater(entry, 100)

    def test_setup_filtering_rules(self) -> None:
        reasons = setup_filter_reasons(
            avg_volume=100_000,
            relative_volume=0.5,
            current_price=2.5,
            days_until_earnings=1,
            rr_target_2=1.5,
            quality_score=60,
            config=SetupConfig(),
        )

        self.assertIn("Average volume too low", reasons)
        self.assertIn("Relative volume below 1.2", reasons)
        self.assertIn("Price below minimum", reasons)
        self.assertIn("Earnings within 2 trading days", reasons)
        self.assertIn("Risk/reward to Target 2 below 2:1", reasons)
        self.assertIn("Trade quality score below threshold", reasons)

    def test_setup_classification(self) -> None:
        self.assertEqual(setup_class(95), "Elite Setup")
        self.assertEqual(setup_class(85), "Strong Candidate")
        self.assertEqual(setup_class(77), "Watchlist")
        self.assertEqual(setup_class(70), "Reject")

    def test_take_profit_plan_allocates_all_shares(self) -> None:
        t1, t2, t3 = take_profit_plan(101)

        self.assertEqual((t1, t2, t3), (33, 33, 35))
        self.assertEqual(t1 + t2 + t3, 101)


if __name__ == "__main__":
    unittest.main()
