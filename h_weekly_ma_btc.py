"""Weekly moving-average trend filter for Bitcoin: an open stand-in for the x100u "Sniper Sell" and
"The 4 Seasons of Bitcoin" indicators (both invite-only; their exact averages are not public). Both
are described by their author as weekly-BTC tools whose signals come from price crossing
"strategic moving averages", with three alerts in a row read as a trend change.

Question: does holding BTC only while the weekly trend is up beat simply holding BTC?

Pre-registered rules (fixed 2026-11-10, before running on real data):
  * Data: BTC-USD daily from Yahoo (starts Sept 2014), resampled to weeks ending Sunday.
  * Signal on each weekly close, applied to the FOLLOWING week (no look-ahead).
  * Grid, 8 variants, nothing else tried:
      MA length (weeks, simple MA):   20, 50, 100, 200
      rule "cross":   in BTC while the close is above the MA, in cash below it
      rule "3 strikes": exit only after 3 weekly closes in a row below the MA (the Sniper Sell
                        idea), re-enter on the first close back above it
  * Costs: 0.2% per switch (exchange fee + slippage, each way). Cash earns nothing.
  * Each variant is scored only once its MA exists; buy-and-hold is measured over the SAME weeks.
  * Two periods: before 2021 (in-sample) and 2021 on (out-of-sample, the one that decides).
  * PASS for a variant, in the 2021+ period:  max drawdown at most HALF of buy-and-hold's AND
    CAGR at least 75% of buy-and-hold's. The idea PASSES only if at least half of the 8 variants
    pass (one lucky setting among eight is not evidence).

    python h_weekly_ma_btc.py       (downloads the data on first run; needs yfinance)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "data" / "btc_usd_1d_full.csv"
LENGTHS = [20, 50, 100, 200]
RULES = ["cross", "3 strikes"]
COST = 0.002
SPLIT = pd.Timestamp("2021-01-01")


def load():
    if not CSV.exists():
        try:
            import yfinance as yf
        except ImportError:
            sys.exit("yfinance missing: activate the stockbot .venv first")
        df = yf.download("BTC-USD", period="max", interval="1d", auto_adjust=False, progress=False)
        if hasattr(df.columns, "levels"):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns=str.lower)[["close"]].dropna()
        CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CSV)
        print(f"downloaded {len(df)} days")
    d = pd.read_csv(CSV, index_col=0, parse_dates=True)["close"]
    w = d.resample("W-SUN").last().dropna()
    return w[w.index <= d.index[-1]]          # drop the current, unfinished week


def positions(close, n, rule):
    ma = close.rolling(n).mean()
    above = close > ma
    pos = pd.Series(np.nan, index=close.index)
    state, strikes = 0, 0
    for i, (t, a) in enumerate(above.items()):
        if pd.isna(ma.iloc[i]):
            continue
        if a:
            state, strikes = 1, 0
        else:
            strikes += 1
            if rule == "cross" or strikes >= 3:
                state = 0
        pos.iloc[i] = state
    return pos.shift(1)          # decided at this close, held next week


def stats(r):
    r = r.dropna()
    if len(r) < 10:
        return dict(cagr=np.nan, mdd=np.nan, weeks=len(r))
    eq = (1 + r).cumprod()
    years = len(r) / 52.18
    return dict(cagr=eq.iloc[-1] ** (1 / years) - 1, mdd=(eq / eq.cummax() - 1).min(), weeks=len(r),
                final=eq.iloc[-1])


def main():
    close = load()
    ret = close.pct_change()
    print(f"BTC weekly closes {close.index[0]:%Y-%m-%d} .. {close.index[-1]:%Y-%m-%d} ({len(close)} weeks)\n")
    rows = []
    for n in LENGTHS:
        for rule in RULES:
            pos = positions(close, n, rule)
            switches = pos.diff().abs().fillna(0)
            strat = pos * ret - switches * COST
            valid = pos.notna()
            for period, mask in (("before 2021", close.index < SPLIT), ("2021 on", close.index >= SPLIT)):
                m = valid & mask
                s, b = stats(strat[m]), stats(ret[m])
                rows.append(dict(ma=n, rule=rule, period=period, cagr=s["cagr"], mdd=s["mdd"],
                                 bh_cagr=b["cagr"], bh_mdd=b["mdd"], weeks=s["weeks"],
                                 in_mkt=pos[m].mean(), switches=int(switches[m].sum())))
    t = pd.DataFrame(rows)
    for period in ("before 2021", "2021 on"):
        g = t[t["period"] == period]
        print(f"=== {period} ===")
        print(f"{'variant':<20} {'weeks':>5} {'in BTC':>7} {'switches':>8} {'CAGR':>8} {'max DD':>8} | {'B&H CAGR':>8} {'B&H DD':>8}")
        for _, r in g.iterrows():
            print(f"{str(r['ma'])+'W '+r['rule']:<20} {r['weeks']:>5} {r['in_mkt']*100:>6.0f}% {r['switches']:>8} "
                  f"{r['cagr']*100:>7.1f}% {r['mdd']*100:>7.1f}% | {r['bh_cagr']*100:>7.1f}% {r['bh_mdd']*100:>7.1f}%")
        print()

    oos = t[t["period"] == "2021 on"].copy()
    oos["pass"] = (oos["mdd"].abs() <= 0.5 * oos["bh_mdd"].abs()) & (oos["cagr"] >= 0.75 * oos["bh_cagr"])
    print("PASS RULE (2021 on): drawdown <= half of buy-and-hold's AND CAGR >= 75% of buy-and-hold's")
    for _, r in oos.iterrows():
        print(f"   [{'PASS' if r['pass'] else 'FAIL'}] {r['ma']}W {r['rule']}: CAGR {r['cagr']*100:.1f}% vs "
              f"{r['bh_cagr']*100:.1f}%, DD {r['mdd']*100:.1f}% vs {r['bh_mdd']*100:.1f}%")
    k = int(oos["pass"].sum())
    print(f"   => {k}/8 variants pass: the weekly MA filter {'PASSES' if k >= 4 else 'FAILS'}")

    last = close.index[-1]
    print(f"\nWhere BTC stands now ({last:%Y-%m-%d}, close ${close.iloc[-1]:,.0f}):")
    for n in LENGTHS:
        ma = close.rolling(n).mean().iloc[-1]
        print(f"   {n}W MA ${ma:,.0f}: price is {'ABOVE' if close.iloc[-1] > ma else 'BELOW'}")


if __name__ == "__main__":
    main()
