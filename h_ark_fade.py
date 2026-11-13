"""
H-006  Fade ARK  -- PRE-REGISTERED 2026-11-11 before any data was downloaded.

Hypothesis: stocks ARK SELLS beat stocks ARK BUYS over the next 20 trading days.
Data     : arkfunds.io /v2/etf/trades for ARKK, ARKW, ARKG, ARKQ, ARKF,
           2020-09-09 .. 2026-10-31; prices from yfinance (adjusted).
Events   : one per (ticker, trade date, direction); a ticker both bought and
           sold on the same date (different funds) is dropped as conflicting.
Entry    : ARK publishes after the close on day t -> enter at the OPEN of t+1.
Returns  : t+1 open -> close of 5th / 20th trading day, minus the matched
           sector ETF (XLK XLV XLF XLC XLY XLI XLP XLE XLU XLB XLRE; SPY if no sector).
Cost     : 20 bps per side = 0.40% per position round trip.
Primary  : weekly calendar-time portfolio: each week, mean 20d excess of SELL
           events minus mean 20d excess of BUY events (weeks with both sides),
           net of 0.80% (two positions). t-stat Newey-West, 4 weekly lags.
DIES if any: t < 2.0 | mean net spread < +0.50% per 20d |
             sign differs between 2020-22 and 2023-26 |
             top-5 tickers > 40% of total positive contribution.
Secondary (context, practical for a long-only account): buy what ARK sells,
           20d excess net of 0.40%.
Output: results_h006_ark_fade.txt (utf-8) + data/ark/events.csv
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "ark")
os.makedirs(D, exist_ok=True)
OUTF = os.path.join(HERE, "results_h006_ark_fade.txt")
LOG = []


def out(s=""):
    print(s); LOG.append(s)


FUNDS = ["ARKK", "ARKW", "ARKG", "ARKQ", "ARKF"]
COST = 0.0040
SECTOR_ETF = {"Technology": "XLK", "Healthcare": "XLV", "Financial Services": "XLF",
              "Energy": "XLE", "Utilities": "XLU", "Industrials": "XLI",
              "Basic Materials": "XLB", "Consumer Defensive": "XLP",
              "Consumer Cyclical": "XLY", "Communication Services": "XLC",
              "Real Estate": "XLRE"}

# ------------------------------------------------------------- trades
TF = os.path.join(D, "trades.json")
if not os.path.exists(TF):
    rows = []
    for f in FUNDS:
        for y in range(2020, 2027):
            url = "https://arkfunds.io/api/v2/etf/trades"
            p = {"symbol": f, "date_from": f"{y}-01-01", "date_to": f"{y}-12-31", "limit": 100000}
            for i in range(4):
                try:
                    r = requests.get(url, params=p, timeout=60)
                    if r.status_code == 404:
                        js = {"trades": []}; break
                    r.raise_for_status(); js = r.json(); break
                except Exception as e:
                    print("  retry", f, y, e); time.sleep(5 * (i + 1))
            else:
                raise SystemExit(f"arkfunds.io failed for {f} {y}")
            t = js.get("trades", [])
            for x in t:
                x["fund"] = x.get("fund") or f
            rows += t
            print(f"  {f} {y}: {len(t)} trades")
            time.sleep(0.5)
    json.dump(rows, open(TF, "w"))
rows = json.load(open(TF))
tr = pd.DataFrame(rows)
out(f"ARK trades downloaded: {len(tr)}  ({tr.date.min()} .. {tr.date.max()})")
tr = tr[(tr.date >= "2020-09-09") & (tr.date <= "2026-10-31")]
tr["ticker"] = tr.ticker.astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
tr = tr[tr.ticker.str.fullmatch(r"[A-Z][A-Z\-]{0,6}")]
tr["direction"] = tr.direction.str.lower()
tr = tr[tr.direction.isin(["buy", "sell"])]
g = tr.groupby(["ticker", "date"]).direction.nunique()
conflict = g[g > 1].index
tr = tr.set_index(["ticker", "date"]).drop(conflict, errors="ignore").reset_index()
ev = tr.groupby(["ticker", "date", "direction"]).size().reset_index(name="nfunds")
out(f"events after cleaning: {len(ev)}  (conflicting ticker-days dropped: {len(conflict)})")

# ------------------------------------------------------------- sectors
SC = os.path.join(D, "sector_cache.json")
sc = json.load(open(SC)) if os.path.exists(SC) else {}


def info(t):
    try:
        i = yf.Ticker(t).info
        return t, i.get("sector")
    except Exception:
        return t, None


need = [t for t in ev.ticker.unique() if t not in sc]
print(f"fetching sectors for {len(need)} tickers ...")
with ThreadPoolExecutor(6) as ex:
    for t, s in ex.map(info, need):
        sc[t] = s
json.dump(sc, open(SC, "w"))
ev["bench"] = ev.ticker.map(lambda t: SECTOR_ETF.get(sc.get(t), "SPY"))

# ------------------------------------------------------------- prices
tick = sorted(set(ev.ticker) | set(SECTOR_ETF.values()) | {"SPY"})
print(f"downloading prices for {len(tick)} tickers ...")
px = yf.download(tick, start="2020-09-01", end="2026-12-31", auto_adjust=True,
                 group_by="ticker", progress=False, threads=True)
days = px.index


def ret(t, i0, h):
    try:
        o = px[(t, "Open")].iloc[i0]; c = px[(t, "Close")].iloc[i0 + h - 1]
        return c / o - 1 if (o > 0 and c > 0) else np.nan
    except (KeyError, IndexError):
        return np.nan


recs = []
for e in ev.itertuples():
    d = pd.Timestamp(e.date)
    i0 = days.searchsorted(d + pd.Timedelta(days=1))
    if i0 + 20 > len(days):
        continue
    r5, r20 = ret(e.ticker, i0, 5), ret(e.ticker, i0, 20)
    b5, b20 = ret(e.bench, i0, 5), ret(e.bench, i0, 20)
    recs.append({"ticker": e.ticker, "date": e.date, "dir": e.direction,
                 "week": days[i0].strftime("%G-%V"), "x5": r5 - b5, "x20": r20 - b20})
R = pd.DataFrame(recs).dropna()
R.to_csv(os.path.join(D, "events.csv"), index=False)
out(f"events with prices: {len(R)}  (buys {sum(R.dir=='buy')}, sells {sum(R.dir=='sell')})")


def nw_t(x, lags=4):
    x = np.asarray(x, float); n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean(); v = (e @ e) / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        v += 2 * w * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(v / n)


def spread(df, col):
    w = df.groupby(["week", "dir"])[col].mean().unstack()
    w = w.dropna()
    return (w["sell"] - w["buy"]).sort_index()


out("\n================ H-006 FADE ARK ================")
for col, lab in (("x20", "20d"), ("x5", " 5d")):
    s = spread(R, col) - 2 * COST
    out(f"  sell-minus-buy {lab} net   weeks={len(s):4d}  mean={s.mean()*100:+6.2f}%  NW t={nw_t(s):+5.2f}")
s20 = spread(R, "x20") - 2 * COST
early = s20[s20.index < "2023"]; late = s20[s20.index >= "2023"]
out(f"  2020-22: mean {early.mean()*100:+.2f}% (weeks {len(early)})   2023-26: mean {late.mean()*100:+.2f}% (weeks {len(late)})")

R["contrib"] = np.where(R.dir == "sell", R.x20, -R.x20)
ct = R.groupby("ticker").contrib.sum().sort_values(ascending=False)
pos = ct[ct > 0].sum()
top5 = ct.head(5)
share = top5.sum() / pos if pos > 0 else np.nan
out(f"  top-5 tickers share of positive contribution: {share*100:.0f}%  ({', '.join(top5.index)})")

out("--- context only ---")
for d_ in ("sell", "buy"):
    sub = R[R.dir == d_]
    wk = sub.groupby("week").x20.mean() - COST
    out(f"  long what ARK {d_:4s}s, 20d vs sector, net  events={len(sub):5d} mean={wk.mean()*100:+6.2f}%  NW t={nw_t(wk):+5.2f}")

m, t = s20.mean(), nw_t(s20)
fails = []
if not (t >= 2.0): fails.append(f"t {t:+.2f} < 2")
if not (m >= 0.005): fails.append(f"mean {m*100:+.2f}% < +0.50%")
if np.sign(early.mean()) != np.sign(late.mean()): fails.append("sign flips between periods")
if not (share <= 0.40): fails.append(f"top-5 share {share*100:.0f}% > 40%")
out("\nVERDICT: " + ("PASSES -> paper-trade next" if not fails else "DIES (" + "; ".join(fails) + ")"))
open(OUTF, "w", encoding="utf-8").write("\n".join(LOG))
