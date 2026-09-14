"""
AI Short-Selling & Undervalued Stock Screener with Backtesting Engine
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="AI Short Screener & Backtester",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

MARKET_UNIVERSES = {
    "US Tech - S&P 500 / NASDAQ": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "NFLX", "PLTR",
    ],
    "Hong Kong - Hang Seng Index": [
        "0700.HK", "9988.HK", "3690.HK", "9618.HK", "1810.HK", "0005.HK", "1299.HK", "0941.HK",
    ],
    "Global Major Stocks": [
        "AAPL", "MSFT", "0700.HK", "9988.HK", "ASML", "SAP", "SHEL", "TSM",
    ],
}

@st.cache_data(ttl=900, show_spinner=False)
def download_prices(tickers: tuple[str, ...], period: str = "2y") -> dict[str, pd.DataFrame]:
    raw = yf.download(
        list(tickers),
        period=period,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    result = {}
    if raw is None or raw.empty:
        return result

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

def calculate_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.where(avg_loss != 0, 100.0)

def analyze_market(tickers: list[str], rsi_period: int, ma_offset: float, overbought: float):
    price_data = download_prices(tuple(tickers), period="1y")
    rows = []
    errors = []

    for ticker in tickers:
        frame = price_data.get(ticker)
        if frame is None or frame.empty or "Close" not in frame:
            errors.append(f"{ticker}: no data")
            continue

        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        if len(close) < max(rsi_period + 1, 20):
            errors.append(f"{ticker}: insufficient history")
            continue

        moving_average = float(close.rolling(20).mean().iloc[-1])
        current_price = float(close.iloc[-1])
        rsi_series = calculate_rsi_series(close, rsi_period)
        rsi_value = float(rsi_series.dropna().iloc[-1])
        price_threshold = moving_average * (1 + ma_offset / 100)

        if rsi_value > overbought and current_price > price_threshold:
            rows.append({
                "Ticker": ticker,
                "Current Price": current_price,
                "RSI Value": rsi_value,
                "20-SMA": moving_average,
                "Signal Generated": "SHORT-SELL WATCH",
            })

    return pd.DataFrame(rows), errors

def run_backtest_simulation(
    ticker: str,
    rsi_period: int,
    overbought: float,
    ma_offset: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_hold_days: int,
    initial_capital: float,
):
    df_raw = yf.download(ticker, period="2y", interval="1d", progress=False)
    if df_raw.empty or "Close" not in df_raw:
        return pd.DataFrame(), {}

    close = pd.to_numeric(df_raw["Close"].squeeze(), errors="coerce").dropna()
    rsi = calculate_rsi_series(close, rsi_period)
    sma = close.rolling(20).mean()

    df = pd.DataFrame({"Close": close, "RSI": rsi, "SMA": sma}).dropna()

    trades = []
    in_position = False
    entry_price = 0.0
    entry_date = None
    days_held = 0

    capital = initial_capital
    portfolio_history = []

    for i in range(len(df)):
        date = df.index[i]
        curr_price = float(df["Close"].iloc[i])
        curr_rsi = float(df["RSI"].iloc[i])
        curr_sma = float(df["SMA"].iloc[i])

        # Evaluate Exit logic if currently in a short trade
        if in_position:
            days_held += 1
            price_change_pct = (curr_price - entry_price) / entry_price  # Short loses if price rises
            exit_reason = None
            pnl_pct = 0.0

            if price_change_pct <= -take_profit_pct / 100:
                exit_reason = "Take Profit Target Hit"
                pnl_pct = take_profit_pct / 100
            elif price_change_pct >= stop_loss_pct / 100:
                exit_reason = "Stop Loss Hit"
                pnl_pct = -stop_loss_pct / 100
            elif days_held >= max_hold_days:
                exit_reason = "Max Hold Time Reached"
                pnl_pct = -price_change_pct

            if exit_reason:
                pnl_dollar = capital * pnl_pct
                capital += pnl_dollar
                trades.append({
                    "Ticker": ticker,
                    "Entry Date": entry_date.strftime("%Y-%m-%d"),
                    "Exit Date": date.strftime("%Y-%m-%d"),
                    "Entry Price ($)": round(entry_price, 2),
                    "Exit Price ($)": round(curr_price, 2),
                    "Days Held": days_held,
                    "Result": "WIN" if pnl_pct > 0 else "LOSS",
                    "P&L (%)": round(pnl_pct * 100, 2),
                    "P&L ($)": round(pnl_dollar, 2),
                    "Exit Reason": exit_reason,
                })
                in_position = False
                days_held = 0

        # Evaluate Entry logic if not in a trade
        elif not in_position:
            price_threshold = curr_sma * (1 + ma_offset / 100)
            if curr_rsi > overbought and curr_price > price_threshold:
                in_position = True
                entry_price = curr_price
                entry_date = date
                days_held = 0

        portfolio_history.append({"Date": date, "Portfolio Value": capital})

    trades_df = pd.DataFrame(trades)
    perf_df = pd.DataFrame(portfolio_history).set_index("Date")

    summary = {
        "Starting Capital": initial_capital,
        "Ending Capital": round(capital, 2),
        "Total Return ($)": round(capital - initial_capital, 2),
        "Total Return (%)": round(((capital - initial_capital) / initial_capital) * 100, 2),
        "Total Trades": len(trades),
        "Win Rate (%)": round((len(trades_df[trades_df["Result"] == "WIN"]) / len(trades_df) * 100), 2) if not trades_df.empty else 0.0,
    }

    return trades_df, perf_df, summary

# --- UI LAYOUT ---
st.title("AI Short-Selling Screener & Backtester")

tab1, tab2 = st.tabs(["🔍 Live Market Screener", "📊 Historical Backtest Engine"])

with st.sidebar:
    st.header("Strategy Rules")
    selected_market = st.selectbox("Market Index", list(MARKET_UNIVERSES))
    overbought = st.slider("RSI Overbought Target", 50, 90, 70)
    rsi_period = st.number_input("RSI Period", 2, 50, 14)
    ma_offset = st.number_input("SMA Offset (%)", 0.0, 20.0, 2.0, step=0.5)

    st.divider()
    st.header("Simulation Settings")
    take_profit = st.number_input("Take Profit Target (%)", 1.0, 50.0, 5.0, step=0.5)
    stop_loss = st.number_input("Stop Loss Limit (%)", 1.0, 50.0, 3.0, step=0.5)
    max_days = st.number_input("Max Position Hold (Days)", 1, 60, 10)
    initial_cap = st.number_input("Initial Portfolio ($)", 1000, 1000000, 10000, step=1000)

with tab1:
    st.subheader("Market Scan Controls")
    if st.button("🔍 RUN MARKET SCAN", type="primary"):
        universe = MARKET_UNIVERSES[selected_market]
        with st.spinner("Analyzing universe..."):
            res, errs = analyze_market(universe, int(rsi_period), float(ma_offset), float(overbought))
            st.session_state["scan_results"] = res

    if "scan_results" in st.session_state and not st.session_state["scan_results"].empty:
        st.dataframe(st.session_state["scan_results"], use_container_width=True)
    else:
        st.info("Run a market scan to locate active candidates.")

with tab2:
    st.subheader("Simulate Short-Selling Strategy")
    
    # Allow choosing from flagged scan results or manual ticker input
    default_tickers = ["TSLA", "NVDA", "AAPL", "0700.HK", "9988.HK"]
    if "scan_results" in st.session_state and not st.session_state["scan_results"].empty:
        ticker_options = list(st.session_state["scan_results"]["Ticker"].unique())
    else:
        ticker_options = default_tickers

    selected_ticker = st.selectbox("Select Stock to Backtest (Past 2 Years Data)", ticker_options)

    if st.button("🚀 RUN HISTORICAL BACKTEST"):
        with st.spinner(f"Simulating trade performance for {selected_ticker}..."):
            trades_df, perf_df, summary = run_backtest_simulation(
                ticker=selected_ticker,
                rsi_period=int(rsi_period),
                overbought=float(overbought),
                ma_offset=float(ma_offset),
                take_profit_pct=float(take_profit),
                stop_loss_pct=float(stop_loss),
                max_hold_days=int(max_days),
                initial_capital=float(initial_cap),
            )

            if trades_df.empty:
                st.warning("No trade signals were triggered historically for this ticker with these exact parameters.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Trades", summary["Total Trades"])
                c2.metric("Win Rate", f"{summary['Win Rate (%)']}%")
                c3.metric("Net P&L ($)", f"${summary['Total Return ($)']}")
                c4.metric("Return on Capital", f"{summary['Total Return (%)']}%")

                st.subheader("Portfolio Growth Curve ($)")
                st.line_chart(perf_df["Portfolio Value"])

                st.subheader("Trade Log Details")
                st.dataframe(trades_df, use_container_width=True)
