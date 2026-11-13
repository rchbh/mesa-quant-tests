"""H-004 on Polymarket: buy favourites (90 to 99 cents) and hold to settlement. Does it survive?

Same hypothesis and death rule as h004_kalshi_favorites.py (Kalshi is closed to Venezuela, so the
question is whether the edge also exists on Polymarket). Formula: net per share = win - price - fee.
Death rule, written before any data:  dies if N < 200 or net per share (maker) < $0.01.

Pre-registered rules (fixed 2026-11-10, before the Polymarket data was downloaded):
  * Universe: the sample in fetch_polymarket.py (top-volume binary markets closed in each quarter
    since 2024, volume >= $10k), resolved cleanly to 1/0. 50-50 or unresolved markets are dropped.
  * Event time = the earlier of closedTime and endDate. Entry: the first daily price point on which
    one side is priced 0.90 to 0.99 and that is at least 24 hours before the event time, so the
    price is never one set after the outcome is known.
  * Polymarket's free history gives ONE price per day (the midpoint), no bid/ask. So:
      - maker version: buy at that price with a limit order, assume it fills, no rebate counted.
        Makers pay no fee on Polymarket. The fill assumption makes this an UPPER bound.
      - taker version: pay price + 1 cent (assumed half-spread), plus the taker fee
        0.05 x p x (1-p) for markets created on/after 2026-03-30 (fee start; 0.05 is the most
        common category rate, crypto is 0.07, geopolitics 0). Markets created before then: no fee.
  * One entry per market. Significance: t-statistic clustered by event (all "No" legs of one
    "who will win" event move together, so they count as one draw, not twenty).

    python h004_polymarket.py
"""
import gzip
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "polymarket"
LO, HI = 0.90, 0.99
MIN_LEAD_S = 24 * 3600
FEE_START = datetime(2026, 3, 30, tzinfo=timezone.utc)
TAKER_RATE = 0.05
HALF_SPREAD = 0.01
DEATH_NET = 0.01
N_MIN = 200


def ts(s):
    if not s:
        return None
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def outcome(m):
    """Index of the winning outcome (0 or 1), or None if not a clean 1/0 resolution."""
    try:
        p = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
    except (TypeError, ValueError):
        return None
    if len(p) != 2:
        return None
    if p[0] >= 0.99 and p[1] <= 0.01:
        return 0
    if p[1] >= 0.99 and p[0] <= 0.01:
        return 1
    return None


def event_id(m):
    ev = m.get("events") or []
    if ev and isinstance(ev, list):
        return str(ev[0].get("id") or ev[0].get("slug"))
    return str(m.get("id"))


def episodes(markets, hist):
    rows, skipped = [], {"not clean 1/0": 0, "no history": 0, "no favourite": 0}
    for m in markets:
        win_idx = outcome(m)
        if win_idx is None:
            skipped["not clean 1/0"] += 1
            continue
        times = [t for t in (ts(m.get("closedTime")), ts(m.get("endDate"))) if t]
        if not times:
            skipped["not clean 1/0"] += 1
            continue
        event_time = min(times)
        h = hist.get(str(m["id"])) or hist.get(m["id"]) or []
        if not h:
            skipped["no history"] += 1
            continue
        created = ts(m.get("createdAt")) or ts(m.get("startDate"))
        fee_on = created is not None and created >= FEE_START
        for pt in sorted(h, key=lambda x: x["t"]):
            when = datetime.fromtimestamp(int(pt["t"]), tz=timezone.utc)
            if (event_time - when).total_seconds() < MIN_LEAD_S:
                skipped["no favourite"] += 1
                break
            p0 = float(pt["p"])
            side = 0 if LO <= p0 <= HI else 1 if LO <= 1 - p0 <= HI else None
            if side is None:
                continue
            price = p0 if side == 0 else 1 - p0
            win = 1.0 if side == win_idx else 0.0
            tp = min(price + HALF_SPREAD, 0.999)
            fee = TAKER_RATE * tp * (1 - tp) if fee_on else 0.0
            rows.append(dict(id=m["id"], event=event_id(m), question=m.get("question", "")[:90],
                             quarter=m.get("_sample_quarter"), date=when, price=price, win=win,
                             volume=float(m.get("volumeNum") or 0), neg_risk=bool(m.get("negRisk")),
                             net_maker=win - price, net_taker=win - tp - fee))
            break
        else:
            skipped["no favourite"] += 1
    return pd.DataFrame(rows), skipped


def clustered_t(x, groups):
    g = pd.DataFrame({"x": x, "g": groups}).dropna()
    if g["g"].nunique() < 3:
        return np.nan
    resid = (g["x"] - g["x"].mean()).groupby(g["g"]).sum()
    se = math.sqrt((resid ** 2).sum()) / len(g)
    return g["x"].mean() / se if se > 0 else np.nan


def c(x):
    return "   n/a" if pd.isna(x) else f"{x*100:+.2f}c"


def table(t, key, label):
    print(f"\n{label:<14} {'entries':>7} {'events':>6} {'avg price':>9} {'won':>7} {'net maker':>10} {'net taker':>10} {'t (maker)':>9}")
    for k, g in t.groupby(key, observed=True):
        print(f"{str(k):<14} {len(g):>7} {g['event'].nunique():>6} {g['price'].mean():>9.3f} {g['win'].mean()*100:>6.1f}% "
              f"{c(g['net_maker'].mean()):>10} {c(g['net_taker'].mean()):>10} {clustered_t(g['net_maker'], g['event']):>9.2f}")


def main():
    mfile, hfile = DATA / "markets.json.gz", DATA / "history_1d.json.gz"
    if not (mfile.exists() and hfile.exists()):
        sys.exit("No Polymarket data. Run fetch_polymarket.py first.")
    markets = json.load(gzip.open(mfile, "rt"))
    hist = json.load(gzip.open(hfile, "rt"))
    t, skipped = episodes(markets, hist)
    if t.empty:
        sys.exit(f"No favourite entries found. Skipped: {skipped}")
    print(f"{len(markets)} markets -> {len(t)} favourite entries in {t['event'].nunique()} events, "
          f"{t['date'].min():%Y-%m-%d} .. {t['date'].max():%Y-%m-%d} | skipped: {skipped}")

    table(t, "quarter", "quarter")
    t["bucket"] = pd.cut(t["price"], [0.899, 0.93, 0.96, 0.99], labels=["0.90-0.93", "0.93-0.96", "0.96-0.99"])
    table(t, "bucket", "price bucket")
    t["vol"] = pd.cut(t["volume"], [0, 1e5, 1e6, np.inf], labels=["<$100k", "$100k-1M", ">$1M"])
    table(t, "vol", "volume")
    table(t, "neg_risk", "multi-outcome")

    tt = clustered_t(t["net_maker"], t["event"])
    tk = clustered_t(t["net_taker"], t["event"])
    lost = t[t["win"] == 0]
    print(f"\nALL: net per share maker {c(t['net_maker'].mean())} (event-clustered t = {tt:.2f}), "
          f"taker {c(t['net_taker'].mean())} (t = {tk:.2f})")
    print(f"won {t['win'].mean()*100:.1f}% at an average price of {t['price'].mean()*100:.1f}c; "
          f"{len(lost)} losers each lost ~{lost['price'].mean()*100:.0f}c" if len(lost) else "no losers")
    recent = t[t["date"] >= datetime(2026, 1, 1, tzinfo=timezone.utc)]
    if len(recent):
        print(f"2026 only: {len(recent)} entries, net maker {c(recent['net_maker'].mean())} "
              f"(t = {clustered_t(recent['net_maker'], recent['event']):.2f})")

    print("\nDEATH RULE (same as the Kalshi test)")
    checks = [(f"N >= {N_MIN}", len(t) >= N_MIN, f"{len(t)}"),
              ("net per share >= $0.01 (maker, upper bound)", t["net_maker"].mean() >= DEATH_NET, c(t["net_maker"].mean()))]
    for text, ok, val in checks:
        print(f"   [{'PASS' if ok else 'FAIL'}] {text}   (measured: {val})")
    print(f"   => H-004 on Polymarket {'SURVIVES' if all(ok for _, ok, _ in checks) else 'DIES'} "
          f"(event-clustered t = {tt:.2f})")
    t.drop(columns=["bucket", "vol"]).to_csv(HERE / "results_h004_polymarket_entries.csv", index=False)


if __name__ == "__main__":
    main()
