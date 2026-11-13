"""H-004: buy Kalshi favourites (90 cents or more) and hold to settlement. Does it survive?

Mesa quant's H-004 (docs/hipotesis-semilla.json), from the favourite-longshot bias measured by
Bürgi, Deng and Whelan (CESifo WP 12122). Formula F18: net per contract = win - price - fee.
Death rule, written before any data:  dies if net per contract < $0.01 (or N < 200).
(Its second rule, CVaR_5 of the book below -30% of capital, depends on position sizing, which
belongs to the spec's Sizer; it is reported per position, not judged here.)

Pre-registered rules (fixed before the Kalshi data was downloaded):
  * Universe: settled binary markets of the downloaded series, from 2024 on.
  * Entry: the first daily candle on which one side is a favourite (price 0.90 to 0.99) and that
    ENDS at least 24 hours before the event happens (occurrence time, or close time if absent),
    so the price is never one set after the outcome is already known.
  * Maker version (as H-004 says): buy at the bid of the favourite side (YES bid, or 1 - YES ask
    for NO), assuming the limit order fills. That fill assumption makes it an UPPER bound.
    Maker fee 0.0175 x P x (1-P) on KXFED, KXCPI and KXPAYROLLS, zero elsewhere (spec 4.3).
  * Taker version for reference: buy at the ask, fee 0.07 x P x (1-P) rounded up to the cent per
    100-contract order.
  * One entry per market. Significance: t-statistic clustered by event (strikes of the same event
    move together).

    python h004_kalshi_favorites.py
"""
import gzip
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data" / "kalshi"
LO, HI = 0.90, 0.99
MIN_LEAD_S = 24 * 3600
MAKER_FEE_SERIES = {"KXFED", "KXCPI", "KXPAYROLLS"}
START = datetime(2024, 1, 1, tzinfo=timezone.utc)
DEATH_NET = 0.01
N_MIN = 200


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def to_dollars(v, key):
    """Kalshi has used several formats: '*_dollars' strings, dollar strings under 'close'
    (e.g. "0.0200"), and integer cents under 'close' (older API). Strings and floats <= 1 are
    dollars; integers are cents."""
    if v is None or v == "":
        return None
    if "dollars" in key or isinstance(v, str):
        return float(v)
    if isinstance(v, int):
        return v / 100
    return float(v) if v <= 1 else float(v) / 100


def price(obj, field):
    """Candle sub-object -> close price in dollars, whatever the format."""
    d = obj.get(field) or {}
    if isinstance(d, dict):
        for k in ("close_dollars", "close"):
            if d.get(k) not in (None, ""):
                return to_dollars(d[k], k)
    for k in (f"{field}_close_dollars", f"{field}_close"):
        if obj.get(k) is not None:
            return to_dollars(obj[k], k)
    return None


def taker_fee(p):
    return math.ceil(0.07 * 100 * p * (1 - p) * 100 - 1e-9) / 100 / 100   # per contract, 100-lot order


def maker_fee(p, series):
    return 0.0175 * p * (1 - p) if series in MAKER_FEE_SERIES else 0.0


def episodes():
    rows, skipped = [], {"not binary/settled": 0, "no candles": 0, "no favourite": 0}
    for mfile in sorted(DATA.glob("*_markets.json.gz")):
        series = mfile.name.replace("_markets.json.gz", "")
        cfile = DATA / f"{series}_candles_1d.json.gz"
        if not cfile.exists():
            continue
        markets = json.load(gzip.open(mfile, "rt"))
        candles = json.load(gzip.open(cfile, "rt"))
        for m in markets:
            res = m.get("result")
            if m.get("market_type", "binary") != "binary" or res not in ("yes", "no"):
                skipped["not binary/settled"] += 1
                continue
            event_time = ts(m.get("occurrence_datetime")) or ts(m.get("close_time"))
            if event_time is None or event_time < START:
                continue
            cs = (candles.get(m["ticker"]) or {}).get("candlesticks") or []
            if not cs:
                skipped["no candles"] += 1
                continue
            for c in sorted(cs, key=lambda c: c.get("end_period_ts", 0)):
                end = datetime.fromtimestamp(c["end_period_ts"], tz=timezone.utc)
                if (event_time - end).total_seconds() < MIN_LEAD_S:
                    break
                yb, ya = price(c, "yes_bid"), price(c, "yes_ask")
                side = None
                if yb is not None and LO <= yb <= HI:
                    side, maker_p, taker_p = "yes", yb, ya
                elif ya is not None and LO <= 1 - ya <= HI:
                    side, maker_p, taker_p = "no", 1 - ya, (1 - yb) if yb is not None else None
                if side:
                    win = 1.0 if res == side else 0.0
                    rows.append(dict(series=series, event=m.get("event_ticker"), ticker=m["ticker"],
                                     date=end, side=side, price=maker_p, win=win,
                                     net_maker=win - maker_p - maker_fee(maker_p, series),
                                     net_taker=(win - taker_p - taker_fee(taker_p))
                                     if taker_p is not None and 0 < taker_p < 1 else np.nan))
                    break
            else:
                skipped["no favourite"] += 1
    return pd.DataFrame(rows), skipped


def clustered_t(x, groups):
    g = pd.DataFrame({"x": x, "g": groups}).dropna()
    if g["g"].nunique() < 3:
        return np.nan
    mean = g["x"].mean()
    resid = (g["x"] - mean).groupby(g["g"]).sum()
    se = math.sqrt((resid ** 2).sum()) / len(g)
    return mean / se if se > 0 else np.nan


def c(x):
    return "   n/a" if pd.isna(x) else f"{x*100:+.2f}c"


def main():
    if not DATA.exists():
        sys.exit("No Kalshi data. Run fetch_public.py first.")
    t, skipped = episodes()
    if t.empty:
        sys.exit(f"No favourite episodes found. Skipped: {skipped}")
    print(f"{len(t)} favourite entries in {t['event'].nunique()} events, {t['date'].min():%Y-%m-%d} .. "
          f"{t['date'].max():%Y-%m-%d} | skipped: {skipped}")
    print(f"\n{'series':<11} {'entries':>7} {'avg price':>9} {'won':>7} {'implied':>8} {'net maker':>10} {'net taker':>10}")
    for s, g in list(t.groupby("series")) + [("ALL", t)]:
        print(f"{s:<11} {len(g):>7} {g['price'].mean():>9.3f} {g['win'].mean()*100:>6.1f}% {g['price'].mean()*100:>7.1f}% "
              f"{c(g['net_maker'].mean()):>10} {c(g['net_taker'].mean()):>10}")
    t["quarter"] = t["date"].dt.tz_localize(None).dt.to_period("Q")
    print("\nby quarter (the spec expects the bias to shrink over time):")
    for q, g in t.groupby("quarter"):
        print(f"   {q}: {len(g):>5} entries, won {g['win'].mean()*100:5.1f}% at avg {g['price'].mean():.3f}, net maker {c(g['net_maker'].mean())}")
    tt = clustered_t(t["net_maker"], t["event"])
    loss_rate = 1 - t["win"].mean()
    print(f"\nnet per contract (maker): {c(t['net_maker'].mean())}, event-clustered t = {tt:.2f}")
    print(f"a losing position loses its whole price: {loss_rate*100:.1f}% of positions lost ~{t['price'].mean()*100:.0f}c")
    print("\nDEATH RULE (from hipotesis-semilla.json, H-004)")
    checks = [(f"N >= {N_MIN}", len(t) >= N_MIN, f"{len(t)}"),
              ("net per contract >= $0.01 (maker, upper bound)", t["net_maker"].mean() >= DEATH_NET, c(t['net_maker'].mean()))]
    for text, ok, val in checks:
        print(f"   [{'PASS' if ok else 'FAIL'}] {text}   (measured: {val})")
    print(f"   => H-004 {'SURVIVES' if all(ok for _, ok, _ in checks) else 'DIES'}"
          f"{' (and significance: event-clustered t = %.2f)' % tt}")
    t.to_csv(Path(__file__).resolve().parent / "results_h004_entries.csv", index=False)


if __name__ == "__main__":
    main()
