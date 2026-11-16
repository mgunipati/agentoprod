import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

import finance_tools as ft


class LoanEmiTests(unittest.TestCase):
    def test_matches_known_emi(self):
        result = ft.loan_emi(principal=1_000_000, annual_rate_percent=8.5, months=240)
        self.assertEqual(result["emi"], 8678.23)
        self.assertEqual(result["total_paid"], 2082775.76)

    def test_zero_rate_splits_principal_evenly(self):
        self.assertEqual(ft.loan_emi(principal=1200, annual_rate_percent=0, months=12)["emi"], 100)

    def test_rejects_bad_inputs(self):
        self.assertIn("error", ft.loan_emi(principal=-5, annual_rate_percent=8.5, months=12))
        self.assertIn("error", ft.loan_emi(principal=100, annual_rate_percent=150, months=12))
        self.assertIn("error", ft.loan_emi(principal=100, annual_rate_percent=8, months=0))


class SymbolValidationTests(unittest.TestCase):
    def test_accepts_common_ticker_formats(self):
        for symbol in ("AAPL", "reliance.ns", "BRK-B", "^NSEI", "INR=X"):
            self.assertIsNotNone(ft._clean_symbol(symbol), symbol)

    def test_rejects_non_tickers(self):
        for symbol in ("", "AAPL; rm -rf /", "a" * 30, "../etc", None):
            self.assertIsNone(ft._clean_symbol(symbol), symbol)

    def test_tools_reject_invalid_symbol_without_network(self):
        with patch.object(ft.yf, "Ticker") as ticker:
            self.assertIn("error", ft.get_stock_price(symbol="not a ticker"))
            ticker.assert_not_called()


class StockPriceTests(unittest.TestCase):
    def test_returns_price_in_listed_currency(self):
        stock = MagicMock(info={"currentPrice": 2950.456, "currency": "INR", "shortName": "Reliance"})
        with patch.object(ft.yf, "Ticker", return_value=stock) as ticker:
            result = ft.get_stock_price(symbol="reliance.ns")

        ticker.assert_called_once_with("RELIANCE.NS")
        self.assertEqual(result["currency"], "INR")
        self.assertEqual(result["price"], 2950.46)
        self.assertIsNone(result["pe_ratio"])

    def test_missing_price_is_an_error(self):
        with patch.object(ft.yf, "Ticker", return_value=MagicMock(info={})):
            self.assertIn("error", ft.get_stock_price(symbol="ZZZZ"))

    def test_fetch_failure_hides_exception_message(self):
        with patch.object(ft.yf, "Ticker", side_effect=RuntimeError("https://x?apikey=secret")):
            result = ft.get_stock_price(symbol="AAPL")
        self.assertIn("RuntimeError", result["error"])
        self.assertNotIn("secret", result["error"])


class StockHistoryTests(unittest.TestCase):
    def test_rejects_unknown_period(self):
        self.assertIn("error", ft.get_stock_history(symbol="AAPL", period="7w"))

    def test_summarises_history(self):
        index = pd.date_range("2026-11-02", periods=3, freq="D")
        hist = pd.DataFrame(
            {"Close": [100.0, 105.0, 110.0], "High": [101, 106, 112], "Low": [99, 104, 108], "Volume": [10, 20, 30]},
            index=index,
        )
        stock = MagicMock(history_metadata={"currency": "USD"})
        stock.history.return_value = hist
        with patch.object(ft.yf, "Ticker", return_value=stock):
            result = ft.get_stock_history(symbol="AAPL", period="5d")

        self.assertEqual(result["change_percent"], 10.0)
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["high"], 112)
        self.assertEqual(list(result["last_5_closes"]), ["2026-11-02", "2026-11-03", "2026-11-04"])


class AlphaVantageTests(unittest.TestCase):
    def test_reports_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIn("not configured", ft.get_alpha_vantage_quote(symbol="AAPL")["error"])

    def test_normalises_quote_keys(self):
        client = MagicMock()
        client.get_quote_endpoint.return_value = ({"01. symbol": "AAPL", "05. price": "190.10"}, None)
        with (
            patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test-key"}),
            patch.object(ft, "TimeSeries", return_value=client),
        ):
            result = ft.get_alpha_vantage_quote(symbol="aapl")

        self.assertEqual(result["price"], "190.10")
        self.assertEqual(result["symbol"], "AAPL")


if __name__ == "__main__":
    unittest.main()
