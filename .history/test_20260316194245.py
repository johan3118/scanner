#!/usr/bin/env python3
"""
Single-Pair Z-Score Backtester — Binance Futures
================================================
Simulates trading a specific pair over the last 365 days using 4h candles.
Enters when |Z| > threshold, exits when Z crosses 0.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Configuration ---
SYM_A = "BNB/USDT"
SYM_B = "BTC/USDT"
TIMEFRAME = "4h"
DAYS_BACK = 365
Z_WINDOW = 20         # Rolling window for mean/std
ENTRY_Z = 2.0         # Z-score threshold to enter trade
EXIT_Z = 0.0          # Z-score threshold to exit trade
FEE_RATE = 0.0005     # 0.05% taker fee per leg (0.10% total per trade)

exchange = ccxt.binance({"options": {"defaultType": "future"}})


def fetch_historical_data(symbol, days, timeframe):
    print(f"Fetching {days} days of {timeframe} data for {symbol}...")
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_ohlcv = []

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if all_ohlcv[-1][0] >= int(datetime.now().timestamp() * 1000):
                break
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=[
                      "ts", "open", "high", "low", "close", "vol"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df["close"]


def run_backtest():
    # 1. Fetch Data
    price_a = fetch_historical_data(SYM_A, DAYS_BACK, TIMEFRAME)
    price_b = fetch_historical_data(SYM_B, DAYS_BACK, TIMEFRAME)

    # Align dataframes
    df = pd.DataFrame({"A": price_a, "B": price_b}).dropna()

    if df.empty:
        print("Not enough data to backtest.")
        return

    # 2. Calculate Spread and Z-Score
    df["log_A"] = np.log(df["A"])
    df["log_B"] = np.log(df["B"])
    df["spread"] = df["log_A"] - df["log_B"]

    df["roll_mean"] = df["spread"].rolling(window=Z_WINDOW).mean()
    df["roll_std"] = df["spread"].rolling(window=Z_WINDOW).std()
    df["zscore"] = (df["spread"] - df["roll_mean"]) / df["roll_std"]

    df.dropna(inplace=True)

    # 3. Generate Trading Signals
    # Position: 1 (Long Spread: Buy A, Sell B), -1 (Short Spread: Sell A, Buy B), 0 (Flat)
    df["signal"] = 0
    position = 0

    for i in range(len(df)):
        z = df["zscore"].iloc[i]

        # Entry Logic
        if position == 0:
            if z < -ENTRY_Z:
                position = 1   # Spread is cheap, go Long
            elif z > ENTRY_Z:
                position = -1  # Spread is rich, go Short

        # Exit Logic
        elif position == 1 and z >= EXIT_Z:
            position = 0       # Mean reverted, exit Long
        elif position == -1 and z <= EXIT_Z:
            position = 0       # Mean reverted, exit Short

        df.iloc[i, df.columns.get_loc("signal")] = position

    # 4. Calculate Returns
    # Shift signal by 1 to avoid lookahead bias (we trade on the close, realize return on next candle)
    df["position"] = df["signal"].shift(1).fillna(0)

    # Log returns of individual assets
    df["ret_A"] = df["log_A"].diff()
    df["ret_B"] = df["log_B"].diff()

    # Strategy Return:
    # If position is 1: we get return of A, minus return of B
    # If position is -1: we get return of B, minus return of A
    df["strat_ret"] = df["position"] * (df["ret_A"] - df["ret_B"])

    # 5. Factor in Trading Fees
    # Every time position changes, we pay fees on BOTH legs (A and B)
    df["trades"] = df["position"].diff().abs().fillna(0)
    df["fee_impact"] = df["trades"] * (FEE_RATE * 2)

    # Net Returns
    df["net_ret"] = df["strat_ret"] - df["fee_impact"]

    # Cumulative Returns
    df["cum_return"] = df["net_ret"].cumsum()
    df["cum_return_pct"] = (np.exp(df["cum_return"]) - 1) * 100

    # 6. Display Results
    # Divide by 2 because entry+exit = 1 round trip
    total_trades = df["trades"].sum() / 2
    final_return = df["cum_return_pct"].iloc[-1]

    print("\n" + "="*50)
    print(f" BACKTEST RESULTS: {SYM_A} / {SYM_B}")
    print(f" Period: Last {DAYS_BACK} days | Timeframe: {TIMEFRAME}")
    print("="*50)
    print(f" Total Round-Trip Trades : {int(total_trades)}")
    print(f" Estimated Net Profit    : {final_return:.2f}%")
    print(f" Total Fees Paid (Est.)  : {(df['fee_impact'].sum() * 100):.2f}%")
    print("="*50)


if __name__ == "__main__":
    run_backtest()
