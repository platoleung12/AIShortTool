"""AI Short-Selling & Undervalued Stock Screener

Run with:
    streamlit run ai_short_selling_screener.py

This application is for educational and research purposes only. It is not
investment, legal, or tax advice, and it does not place trades.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="AI Short-Selling & Undervalued Stock Screener",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


MARKET_UNIVERSES: dict[str, list[str]] = {
    "US Tech - S&P 500 / NASDAQ": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO",
        "TSLA", "AMD", "NFLX", "ADBE", "CRM", "ORCL", "CSCO", "QCOM",
        "INTC", "INTU", "AMAT", "MU", "ADI", "TXN", "PANW", "SNPS",
        "CDNS", "NOW", "SHOP", "PLTR", "CRWD", "MELI",
    ],
    "Hong Kong - Hang Seng Index": [
        "0700.HK", "9988.HK", "3690.HK", "9618.HK", "1810.HK", "0005.HK",
        "1299.HK", "0941.HK", "0388.HK", "2318.HK", "0883.HK", "1398.HK",
        "3988.HK", "2628.HK", "2388.HK", "0001.HK", "0011.HK", "0002.HK",
        "0066.HK", "1928.HK", "2018.HK", "1038.HK", "0267.HK", "1113.HK",
    ],
    "Global Major Stocks": [
        "AAPL", "MSFT", "0700.HK", "9988.HK", "005930.KS", "7203.T",
        "6758.T", "601398.SS", "NESN.SW", "ASML", "SAP", "SHEL", "TTE",
        "NVO", "ROG.SW", "MC.PA", "OR.PA", "BHP", "RIO", "RELIANCE.NS",
        "INFY", "SONY", "TM", "UL", "AZN", "HSBA.L", "SHOP", "TSM",
    ],
}


@st.cache_data(ttl=900, show_spinner=False)
def download_prices(tickers: tuple[str, ...], period: str = "1y") -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data in one Yahoo Finance request."""
    raw = yf.download(
        list(tickers),
        period=period,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    result: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return result

    # yfinance returns a two-level column index for multiple tickers and a
    # one-level index for a single ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        first_level = set(raw.columns.get_level_values(0))
        for ticker in tickers:
            if ticker in first_level:
                frame = raw[ticker].copy()
            elif ticker in set(raw.columns.get_level_values(1)):
                frame = raw.xs(ticker, axis=1, level=1).copy()
            else:
                continue
            result[ticker] = frame.dropna(how="all")
    else:
        result[tickers[0]] = raw.copy().dropna(how="all")
    return result


def calculate_rsi(close: pd.Series, period: int) -> float:
    """Calculate Wilder-style RSI from closing prices."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # A continuously rising series has RSI=100 rather than missing RSI.
    rsi = rsi.where(avg_loss != 0, 100.0)
    return float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else float("nan")


def safe_number(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def get_company_name_and_market_cap(ticker: str) -> tuple[str, float]:
    """Read company metadata, tolerating Yahoo Finance field differences."""
    instrument = yf.Ticker(ticker)
    name = ticker
    market_cap = float("nan")
    try:
        info = instrument.info
        name = info.get("longName") or info.get("shortName") or ticker
        market_cap = safe_number(info.get("marketCap")) / 1_000_000_000
    except Exception:
        # Metadata failures should not prevent the remaining symbols from
        # being analyzed; the price-based calculations are still useful.
        pass
    return str(name), market_cap


def analyze_market(
    tickers: list[str], rsi_period: int, min_market_cap: float, ma_offset: float,
    overbought: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Calculate indicators and return only symbols matching the scan rules."""
    price_data = download_prices(tuple(tickers))
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for ticker in tickers:
        frame = price_data.get(ticker)
        if frame is None or frame.empty or "Close" not in frame or "Volume" not in frame:
            errors.append(f"{ticker}: no usable historical price data")
            continue

        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        volume = pd.to_numeric(frame["Volume"], errors="coerce").reindex(close.index).fillna(0)
        if len(close) < max(rsi_period + 1, 20):
            errors.append(f"{ticker}: insufficient history")
            continue

        # Use a 20-day SMA as the stable, widely available VWAP alternative.
        moving_average = float(close.rolling(20).mean().iloc[-1])
        current_price = float(close.iloc[-1])
        rsi_value = calculate_rsi(close, rsi_period)
        name, market_cap = get_company_name_and_market_cap(ticker)
        price_threshold = moving_average * (1 + ma_offset / 100)

        # A short-selling watch signal is generated when momentum is
        # overbought and price is above its moving-average threshold.
        matches = (
            np.isfinite(rsi_value)
            and np.isfinite(market_cap)
            and market_cap >= min_market_cap
            and rsi_value > overbought
            and current_price > price_threshold
        )
        if matches:
            rows.append(
                {
                    "Ticker": ticker,
                    "Company Name": name,
                    "Current Price": current_price,
                    "RSI Value": rsi_value,
                    "Market Cap ($B)": market_cap,
                    "Signal Generated": "SHORT-SELL WATCH",
                }
            )

    columns = [
        "Ticker", "Company Name", "Current Price", "RSI Value",
        "Market Cap ($B)", "Signal Generated",
    ]
    return pd.DataFrame(rows, columns=columns), errors


def format_results(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    if not formatted.empty:
        formatted["Current Price"] = formatted["Current Price"].map(lambda x: f"{x:,.2f}")
        formatted["RSI Value"] = formatted["RSI Value"].map(lambda x: f"{x:.2f}")
        formatted["Market Cap ($B)"] = formatted["Market Cap ($B)"].map(lambda x: f"${x:,.2f}")
    return formatted


st.title("AI Short-Selling & Undervalued Stock Screener")
st.caption(
    "A quantitative research dashboard that flags potentially overbought stocks "
    "for further short-selling research. Data provided by Yahoo Finance."
)

with st.sidebar:
    st.header("Control Panel")
    selected_market = st.selectbox("Target Market Index", list(MARKET_UNIVERSES))
    st.divider()
    overbought = st.slider("RSI Overbought Level", min_value=50, max_value=90, value=70)
    rsi_period = st.number_input("RSI Period Length", min_value=2, max_value=100, value=14, step=1)
    min_market_cap = st.number_input(
        "Minimum Market Cap ($B)", min_value=0.0, max_value=10_000.0, value=1.0, step=0.5,
    )
    ma_offset = st.number_input(
        "Price vs Moving Average Offset (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.5,
    )
    st.divider()
    st.info(
        "Rule: RSI > selected overbought level, price > 20-day SMA plus the selected offset, "
        "and market cap ≥ the selected minimum."
    )
    run_scan = st.button("🔍 RUN MARKET SCAN", type="primary", use_container_width=True)

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
    st.session_state.scanned = 0
    st.session_state.errors = []
    st.session_state.scan_market = ""

if run_scan:
    universe = MARKET_UNIVERSES[selected_market]
    with st.spinner(f"Scanning {len(universe)} symbols in {selected_market}..."):
        results, errors = analyze_market(
            universe, int(rsi_period), float(min_market_cap), float(ma_offset), int(overbought)
        )
    st.session_state.results = results
    st.session_state.scanned = len(universe)
    st.session_state.errors = errors
    st.session_state.scan_market = selected_market

results = st.session_state.results
if st.session_state.scanned == 0:
    st.info("Choose your scan parameters in the sidebar and click **RUN MARKET SCAN** to begin.")
else:
    average_rsi = float(results["RSI Value"].mean()) if not results.empty else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Stocks Scanned", st.session_state.scanned)
    c2.metric("Matches Found", len(results))
    c3.metric("Average RSI of Matches", f"{average_rsi:.2f}" if not results.empty else "—")

    st.subheader(f"Flagged Stocks — {st.session_state.scan_market}")
    if results.empty:
        st.success("No stocks matched all selected criteria.")
    else:
        st.dataframe(
            format_results(results),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Current Price": st.column_config.TextColumn("Current Price"),
                "RSI Value": st.column_config.TextColumn("RSI Value"),
                "Market Cap ($B)": st.column_config.TextColumn("Market Cap ($B)"),
            },
        )
        csv_data = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Flagged Stocks (CSV)",
            data=csv_data,
            file_name="flagged_stocks.csv",
            mime="text/csv",
        )

    if st.session_state.errors:
        with st.expander(f"Data warnings ({len(st.session_state.errors)})"):
            st.write("Some symbols could not be fully analyzed; they were skipped.")
            st.code("\n".join(st.session_state.errors))

st.divider()
st.caption(
    "Disclaimer: This tool is for educational purposes only. Yahoo Finance data may be delayed, "
    "incomplete, or inaccurate. A signal is not a recommendation to sell short or trade any security."
)

# Keep io imported intentionally for compatibility with environments that
# monkey-patch Streamlit download handling; no file-system state is required.
_UNUSED = io

if __name__ == "__main__":
    pass
