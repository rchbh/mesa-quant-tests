"""
H-007  200-day trend filter on hyped themes -- PRE-REGISTERED 2026-11-11.

Rule   : at each month-end, if the equal-weight basket's value > its 200-day
         simple moving average, hold the basket next month; else hold cash
         (13-week T-bill yield, ^IRX). 10 bps per switch. Until 200 days of
         history exist the rule just holds the basket.
Main window (reported, NOT decisive -- one episode only):
         AI power: GRID, XLU, URA, NLR, VOLT  2023-01-01 .. 2026-10-31
Decisive analog windows (fixed before running):
  W1 CSCO           1999-01-01 .. 2003-12-31
  W2 QQQ            1999-03-10 .. 2003-12-31
  W3 ICLN + TAN     2008-06-25 .. 2012-12-31
  W4 ICLN + TAN     2020-01-01 .. 2023-12-31
  W5 ARKK           2019-01-01 .. 2023-12-31
A window is a WIN if the filter has a smaller max drawdown than buy-and-hold
AND its CAGR is no more than 2.0 %/yr below buy-and-hold.
PASS: wins in >= 4 of 5 analog windows. Otherwise the filter is a story.
Output: results_h007_theme_trend.txt (utf-8)
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUTF = os.path.join(HERE, "results_h007_theme_trend.txt")
LOG = []


def out(s=""):
    print(s); LOG.append(s)


WINDOWS = [
    ("MAIN AI power", ["GRID", "XLU", "URA", "NLR", "VOLT"], "2023-01-01", "2026-10-31", False),
    ("W1 CSCO 99-03", ["CSCO"], "1999-01-01", "2003-12-31", True),
    ("W2 QQQ 99-03", ["QQQ"], "1999-03-10", "2003-12-31", True),
    ("W3 ICLN+TAN 08-12", ["ICLN", "TAN"], "2008-06-25", "2012-12-31", True),
    ("W4 ICLN+TAN 20-23", ["ICLN", "TAN"], "2020-01-01", "2023-12-31", True),
    ("W5 ARKK 19-23", ["ARKK"], "2019-01-01", "2023-12-31", True),
]
COST = 0.0010

irx = yf.download("^IRX", start="1997-01-01", end="2026-12-31", progress=False, auto_adjust=True)["Close"]
irx = irx.squeeze() if hasattr(irx, "squeeze") else irx
cash_daily = (irx / 100 / 252).ffill()


def stats(eq):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    return cagr, dd


out("H-007 results (filter vs buy-and-hold)\n")
out(f"{'window':20s} {'B&H CAGR':>9s} {'B&H DD':>8s} {'FLT CAGR':>9s} {'FLT DD':>8s} {'switches':>8s}  result")
wins = 0
for name, tick, start, end, decisive in WINDOWS:
    pre = (pd.Timestamp(start) - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
    px = yf.download(tick, start=pre, end=(pd.Timestamp(end) + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                     progress=False, auto_adjust=True)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tick[0])
    px = px.dropna(how="all")
    rets = px.pct_change()
    # equal weight, monthly rebalanced; members join when they start trading
    basket_r = rets.mean(axis=1, skipna=True).fillna(0)
    idx = (1 + basket_r).cumprod()
    first_valid = px.notna().any(axis=1).idxmax()
    sma = idx.rolling(200, min_periods=200).mean()
    hist_ok = px.notna().sum(axis=1).cumsum() > 0
    in_win = (idx.index >= pd.Timestamp(start)) & (idx.index <= pd.Timestamp(end))
    dates = idx.index[in_win]
    # month-end signals
    me = idx.loc[: end].resample("ME").last().index
    sig = pd.Series(np.nan, index=idx.index)
    for m in me:
        loc = idx.index[idx.index <= m]
        if len(loc) == 0:
            continue
        d = loc[-1]
        s = sma.loc[d]
        sig.loc[d] = 1.0 if (np.isnan(s) or idx.loc[d] > s) else 0.0
    pos = sig.ffill().shift(1).fillna(1.0)          # act from next day
    cash = cash_daily.reindex(idx.index).ffill().fillna(0)
    strat_r = pos * basket_r + (1 - pos) * cash
    sw = pos.diff().abs().fillna(0)
    strat_r = strat_r - sw * COST
    br = basket_r.loc[dates]; sr = strat_r.loc[dates]
    bh = (1 + br).cumprod(); fl = (1 + sr).cumprod()
    bc, bd = stats(bh); fc, fd = stats(fl)
    nsw = int(sw.loc[dates].sum())
    win = (fd > bd) and (fc >= bc - 0.02)
    tag = ("WIN " if win else "LOSS") if decisive else ("win*" if win else "loss*")
    if decisive and win:
        wins += 1
    out(f"{name:20s} {bc*100:+8.1f}% {bd*100:+7.1f}% {fc*100:+8.1f}% {fd*100:+7.1f}% {nsw:8d}  {tag}")

out("\n* main window is informational only (single episode)")
out(f"\nAnalog wins: {wins}/5")
out("VERDICT: " + ("PASSES -> filter is a usable risk rule for theme bets" if wins >= 4
                    else "DIES -> no reliable edge from the trend filter"))
open(OUTF, "w", encoding="utf-8").write("\n".join(LOG))
