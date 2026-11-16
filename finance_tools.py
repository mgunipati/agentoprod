"""Financial tools for the agentoprod agent.

Template for adding a tool:
1. Write a plain function with type hints and a docstring the model can read.
2. Validate every argument: the model chooses them, so treat them as untrusted.
3. Return a dict with the numbers and the inputs used, never free text only.
4. On failure, return {"error": ...} instead of raising, and never include
   secrets (API keys, URLs with keys) in the error.
5. Add it to FINANCE_TOOLS at the bottom.
"""

import logging
import os
import re

import boto3
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from strands import tool

logger = logging.getLogger("agentoprod.tools")

# Tickers such as AAPL, RELIANCE.NS, BRK-B, ^NSEI (index) or INR=X (currency).
_SYMBOL = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=]{0,19}$")
_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}


def _clean_symbol(symbol: str) -> str | None:
    """Return the upper-cased ticker, or None if it doesn't look like one."""
    if not isinstance(symbol, str):
        return None
    symbol = symbol.strip().upper()
    return symbol if _SYMBOL.match(symbol) else None


def _invalid_symbol(symbol) -> dict:
    # Echo at most 30 characters of what the model sent.
    return {"error": f"{str(symbol)[:30]!r} is not a valid ticker symbol"}


def _ticker_info(tool_name: str, ticker: str) -> dict:
    """Fetch Yahoo's info dict; returns {"error": ...} on failure."""
    try:
        return yf.Ticker(ticker).info or {}
    except Exception as exc:
        return _fetch_failed(tool_name, ticker, exc)


def _number(value, digits: int = 2):
    """Round numeric values; Yahoo leaves some fields empty, so keep None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), digits)


def _fetch_failed(tool_name: str, symbol: str, exc: Exception) -> dict:
    # Log only the error type: some client errors include request URLs.
    logger.warning("tool_fetch_failed", extra={"tool_name": tool_name, "error_type": type(exc).__name__})
    return {"error": f"Could not fetch data for {symbol} ({type(exc).__name__}). Try again later."}


@tool
def loan_emi(principal: float, annual_rate_percent: float, months: int) -> dict:
    """Calculate the fixed monthly instalment (EMI) for a loan.

    Use this whenever the user asks about a monthly loan payment, EMI,
    total interest, or total repayment for a fixed-rate loan.

    Args:
        principal: Amount borrowed, greater than 0.
        annual_rate_percent: Yearly interest rate in percent, e.g. 8.5 for 8.5%.
        months: Loan length in months, between 1 and 600.
    """
    if principal <= 0:
        return {"error": "principal must be greater than 0"}
    if not 0 <= annual_rate_percent <= 100:
        return {"error": "annual_rate_percent must be between 0 and 100"}
    if not 1 <= months <= 600:
        return {"error": "months must be between 1 and 600"}

    monthly_rate = annual_rate_percent / 12 / 100
    if monthly_rate == 0:
        emi = principal / months
    else:
        growth = (1 + monthly_rate) ** months
        emi = principal * monthly_rate * growth / (growth - 1)

    total_paid = emi * months
    return {
        "emi": round(emi, 2),
        "total_interest": round(total_paid - principal, 2),
        "total_paid": round(total_paid, 2),
        "inputs": {
            "principal": principal,
            "annual_rate_percent": annual_rate_percent,
            "months": months,
        },
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """Get the current price and key metrics for a stock from Yahoo Finance.

    Use this for questions about what a stock or index is trading at now.
    Prices are in the listed `currency`, which depends on the exchange.

    Args:
        symbol: Yahoo ticker, e.g. 'AAPL', 'MSFT', 'RELIANCE.NS' (NSE), 'TCS.BO' (BSE), '^NSEI'.
    """
    ticker = _clean_symbol(symbol)
    if ticker is None:
        return _invalid_symbol(symbol)
    info = _ticker_info("get_stock_price", ticker)
    if "error" in info:
        return info

    price = _number(info.get("currentPrice") or info.get("regularMarketPrice"))
    if price is None:
        return {"error": f"No price found for {ticker}. Check the symbol and exchange suffix."}
    return {
        "symbol": ticker,
        "name": info.get("shortName"),
        "currency": info.get("currency"),
        "price": price,
        "market_cap": _number(info.get("marketCap"), 0),
        "pe_ratio": _number(info.get("trailingPE")),
        "week52_high": _number(info.get("fiftyTwoWeekHigh")),
        "week52_low": _number(info.get("fiftyTwoWeekLow")),
        "average_volume": _number(info.get("averageVolume"), 0),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "source": "Yahoo Finance (may be delayed)",
    }


@tool
def get_stock_history(symbol: str, period: str = "1mo") -> dict:
    """Get how a stock's price moved over a period, from Yahoo Finance.

    Use this for questions about performance, trends, highs and lows over time.

    Args:
        symbol: Yahoo ticker, e.g. 'AAPL', 'RELIANCE.NS', '^NSEI'.
        period: One of 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
    """
    ticker = _clean_symbol(symbol)
    if ticker is None:
        return _invalid_symbol(symbol)
    if period not in _PERIODS:
        return {"error": f"period must be one of {', '.join(sorted(_PERIODS))}"}
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        currency = stock.history_metadata.get("currency") if not hist.empty else None
    except Exception as exc:
        return _fetch_failed("get_stock_history", ticker, exc)
    if hist.empty:
        return {"error": f"No historical data available for {ticker} over {period}"}

    start_price = float(hist["Close"].iloc[0])
    end_price = float(hist["Close"].iloc[-1])
    return {
        "symbol": ticker,
        "period": period,
        "currency": currency,
        "start_price": round(start_price, 2),
        "end_price": round(end_price, 2),
        "change_percent": round((end_price - start_price) / start_price * 100, 2),
        "high": round(float(hist["High"].max()), 2),
        "low": round(float(hist["Low"].min()), 2),
        "average_volume": round(float(hist["Volume"].mean())),
        "last_5_closes": {
            day.strftime("%Y-%m-%d"): round(float(close), 2)
            for day, close in hist["Close"].tail(5).items()
        },
        "source": "Yahoo Finance (may be delayed)",
    }


@tool
def get_company_financials(symbol: str) -> dict:
    """Get a company's financial metrics: revenue, margins, growth and leverage.

    Use this for questions about a company's fundamentals or profitability.
    Ratios are returned as percentages (e.g. 25.3 means 25.3%).

    Args:
        symbol: Yahoo ticker, e.g. 'AAPL', 'INFY.NS'.
    """
    ticker = _clean_symbol(symbol)
    if ticker is None:
        return _invalid_symbol(symbol)
    info = _ticker_info("get_company_financials", ticker)
    if "error" in info:
        return info

    def percent(key: str):
        value = _number(info.get(key), 6)
        return None if value is None else round(value * 100, 2)

    return {
        "symbol": ticker,
        "name": info.get("shortName"),
        "currency": info.get("financialCurrency") or info.get("currency"),
        "revenue": _number(info.get("totalRevenue"), 0),
        "revenue_growth_percent": percent("revenueGrowth"),
        "gross_profit": _number(info.get("grossProfits"), 0),
        "profit_margin_percent": percent("profitMargins"),
        "operating_margin_percent": percent("operatingMargins"),
        "ebitda": _number(info.get("ebitda"), 0),
        "earnings_growth_percent": percent("earningsGrowth"),
        "return_on_equity_percent": percent("returnOnEquity"),
        "debt_to_equity": _number(info.get("debtToEquity")),
        "free_cash_flow": _number(info.get("freeCashflow"), 0),
        "source": "Yahoo Finance",
    }


# Name of the AgentCore Identity credential that holds the Alpha Vantage key.
# The AgentCore CLI sets CREDENTIAL_ALPHA_VANTAGE_NAME in the runtime for a
# credential added with `agentcore add credentials api-key --name alpha-vantage`.
_ALPHA_VANTAGE_PROVIDER_ENV = "CREDENTIAL_ALPHA_VANTAGE_NAME"


def _alpha_vantage_key() -> str | dict:
    """Return the Alpha Vantage key, or an {"error": ...} dict.

    On a laptop the key comes from ALPHA_VANTAGE_API_KEY (e.g. `uv run --env-file .env`).
    In AgentCore Runtime it is fetched from AgentCore Identity on each call, so the
    key is never stored in the code, the package or the runtime's settings.
    """
    key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if key:
        return key

    provider = os.getenv(_ALPHA_VANTAGE_PROVIDER_ENV)
    if not provider:
        return {"error": "Alpha Vantage is not configured (no API key or AgentCore credential)."}

    # Only available inside AgentCore Runtime, so import it here.
    from bedrock_agentcore.runtime.context import BedrockAgentCoreContext

    token = BedrockAgentCoreContext.get_workload_access_token()
    if token is None:
        return {"error": "Alpha Vantage needs a user identity: invoke the runtime with a user ID."}

    client = boto3.client("bedrock-agentcore", region_name=os.environ.get("AWS_REGION"))
    response = client.get_resource_api_key(
        resourceCredentialProviderName=provider, workloadIdentityToken=token,
    )
    return response["apiKey"]


@tool
def get_alpha_vantage_quote(symbol: str) -> dict:
    """Get the latest stock quote from Alpha Vantage.

    Use this only if get_stock_price fails or the user asks for Alpha Vantage.
    The free plan allows about 25 requests a day.

    Args:
        symbol: Ticker, e.g. 'AAPL', 'MSFT', 'RELIANCE.BSE'.
    """
    ticker = _clean_symbol(symbol)
    if ticker is None:
        return _invalid_symbol(symbol)
    try:
        key = _alpha_vantage_key()
    except Exception as exc:
        return _fetch_failed("get_alpha_vantage_quote", ticker, exc)
    if isinstance(key, dict):
        return key
    try:
        data, _ = TimeSeries(key=key, output_format="json").get_quote_endpoint(symbol=ticker)
    except Exception as exc:
        return _fetch_failed("get_alpha_vantage_quote", ticker, exc)
    if not data:
        return {"error": f"No quote found for {ticker}"}

    # Alpha Vantage keys look like "05. price"; drop the number prefix.
    quote = {name.split(". ", 1)[-1].replace(" ", "_"): value for name, value in data.items()}
    quote["source"] = "Alpha Vantage"
    return quote


FINANCE_TOOLS = [
    loan_emi,
    get_stock_price,
    get_stock_history,
    get_company_financials,
    get_alpha_vantage_quote,
]
