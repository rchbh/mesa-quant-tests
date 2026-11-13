"""
H-005  Congressional trades -- committee-jurisdiction purchases
PRE-REGISTERED 2026-11-11, before any results were seen.

Primary hypothesis:
  Stock purchases by members of Congress whose committee oversees the stock's sector,
  bought at the OPEN of the first trading day after the report was received
  by the Senate, beat the stock's sector ETF over 20 trading days.
Universe : electronic PTRs received 2025-01-03 .. today-30 trading days,
           House (and Senate if its data exists -- Senate eFD blocks Venezuela,
           so in practice House only; amendment made 2026-11-11 BEFORE results),
           Asset Type == Stock, common equity (yfinance quoteType EQUITY).
Events   : one event per (ticker, disclosure date, side); several senators
           buying the same name on the same day count once.
Cost     : 0.10% round trip.
PASS     : mean 20d excess vs sector ETF >= +0.50%  AND  clustered t >= 2.0
           AND  n >= 100 events.            Otherwise: DIES.
           If n < 100: INCONCLUSIVE (not a pass).
Context only (does not decide verdict): 5d horizon; vs SPY; all purchases;
  non-jurisdiction purchases; sales (an informed sale should be NEGATIVE).
t-stats are clustered by ISO week of entry.
"""
import csv, json, os, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "congress")
COST = 0.0010

COMMITTEE_SECTORS = {  # thomas_id prefix -> yfinance sectors (pre-registered)
    "SSAF": ["Consumer Defensive", "Basic Materials"],
    "SSAS": ["Industrials"],
    "SSBK": ["Financial Services", "Real Estate"],
    "SSCM": ["Communication Services", "Technology", "Industrials", "Consumer Cyclical"],
    "SSEG": ["Energy", "Utilities", "Basic Materials"],
    "SSEV": ["Industrials", "Utilities", "Basic Materials"],
    "SSFI": ["Healthcare", "Financial Services"],
    "SSHR": ["Healthcare"],
    "SSGA": ["Technology", "Industrials"],
    "SLIN": ["Technology", "Industrials"],
    # House
    "HSAG": ["Consumer Defensive", "Basic Materials"],
    "HSAS": ["Industrials"],
    "HSBA": ["Financial Services", "Real Estate"],
    "HSIF": ["Energy", "Utilities", "Healthcare", "Communication Services", "Technology"],
    "HSPW": ["Industrials", "Utilities", "Basic Materials"],
    "HSII": ["Energy", "Basic Materials"],
    "HSSY": ["Technology", "Industrials"],
    "HSWM": ["Healthcare", "Financial Services"],
    "HSHM": ["Technology", "Industrials"],
    "HLIG": ["Technology", "Industrials"],
}
SECTOR_ETF = {"Technology": "XLK", "Healthcare": "XLV", "Financial Services": "XLF",
              "Energy": "XLE", "Utilities": "XLU", "Industrials": "XLI",
              "Basic Materials": "XLB", "Consumer Defensive": "XLP",
              "Consumer Cyclical": "XLY", "Communication Services": "XLC",
              "Real Estate": "XLRE"}


def norm(s):
    return re.sub(r"[^a-z]", "", (s or "").lower().replace(" jr", "").replace(" iii", "").replace(" ii", ""))


# ---------------------------------------------------------------- committees
SUFFIX = ("jr", "sr", "ii", "iii", "iv")


def lastname(name):
    p = name.replace(",", " ").split()
    while len(p) > 1 and p[-1].lower().rstrip(".") in SUFFIX:
        p = p[:-1]
    return norm(p[-1]) if p else ""


mem = json.load(open(os.path.join(D, "committees.json"), encoding="utf-8"))
comm = defaultdict(set)   # (chamber, last, first initial) -> committee prefixes
for m in mem:
    tid = str(m["committee"].get("thomasId", ""))
    ch = (m.get("chamber") or ("senate" if tid[:1] == "S" else "house")).lower()
    nm = m.get("memberName") or ""
    if not nm.split():
        continue
    comm[(ch, lastname(nm), norm(nm.split()[0])[:1])].add(tid[:4])
by_last = defaultdict(list)
for k in comm:
    by_last[k[:2]].append(k)


def committees_of(chamber, first, last):
    keys = by_last.get((chamber, lastname(last)), [])
    if len(keys) > 1:
        keys = [k for k in keys if k[2] == norm(first)[:1]]
    return comm[keys[0]] if len(keys) == 1 else None


# ---------------------------------------------------------------- trades
parts = []
for fn, ch in (("house_trades.csv", "house"), ("senate_trades.csv", "senate")):
    fp = os.path.join(D, fn)
    if os.path.exists(fp):
        x = pd.read_csv(fp, dtype=str).fillna("")
        x["chamber"] = ch
        parts.append(x)
tr = pd.concat(parts, ignore_index=True)
print("chambers:", tr.chamber.value_counts().to_dict())
tr = tr[tr.asset_type.str.contains("Stock", case=False) & ~tr.asset_type.str.contains("Option", case=False)]
tr["ticker"] = tr.ticker.str.strip().str.upper().str.replace(".", "-", regex=False)
tr = tr[tr.ticker.str.fullmatch(r"[A-Z][A-Z\-]{0,5}")]
tr["side"] = np.where(tr.type.str.contains("Purchase", case=False), "buy",
                      np.where(tr.type.str.contains("Sale", case=False), "sell", "other"))
tr = tr[tr.side != "other"]
tr["disc"] = pd.to_datetime(tr.received, errors="coerce").dt.normalize()
tr = tr.dropna(subset=["disc"])
tr["comms"] = [committees_of(c, f, l) for c, f, l in zip(tr["chamber"], tr["first"], tr["last"])]
unmatched = tr[tr.comms.isna()][["first", "last"]].drop_duplicates()
print(f"transactions: {len(tr)}, members unmatched to committees: {len(unmatched)}")
if len(unmatched):
    print("  ", ", ".join(f"{a} {b}" for a, b in unmatched.values[:20]))

# ---------------------------------------------------------------- sector info
SC = os.path.join(D, "sector_cache.json")
sc = json.load(open(SC)) if os.path.exists(SC) else {}


def info(t):
    try:
        i = yf.Ticker(t).info
        return t, {"sector": i.get("sector"), "qt": i.get("quoteType")}
    except Exception:
        return t, {"sector": None, "qt": None}


need = [t for t in tr.ticker.unique() if t not in sc]
print(f"fetching sector for {len(need)} tickers ...")
with ThreadPoolExecutor(6) as ex:
    for t, v in ex.map(info, need):
        sc[t] = v
json.dump(sc, open(SC, "w"))
tr["sector"] = tr.ticker.map(lambda t: sc.get(t, {}).get("sector"))
tr = tr[tr.ticker.map(lambda t: sc.get(t, {}).get("qt") == "EQUITY") & tr.sector.isin(SECTOR_ETF)]


def relevant(row):
    if not row.comms:
        return False
    return any(row.sector in COMMITTEE_SECTORS.get(c, []) for c in row.comms)


tr["relevant"] = tr.apply(relevant, axis=1)

# ---------------------------------------------------------------- prices
tickers = sorted(set(tr.ticker) | set(SECTOR_ETF.values()) | {"SPY"})
start = (tr.disc.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
print(f"downloading prices for {len(tickers)} tickers ...")
px = yf.download(tickers, start=start, auto_adjust=True, group_by="ticker",
                 progress=False, threads=True)
days = px.index


def ret(t, i0, h):
    try:
        o = px[(t, "Open")].iloc[i0]; c = px[(t, "Close")].iloc[i0 + h - 1]
        return c / o - 1 if o > 0 and c > 0 else np.nan
    except (KeyError, IndexError):
        return np.nan


ev = (tr.groupby(["ticker", "disc", "side"])
        .agg(sector=("sector", "first"), relevant=("relevant", "max"), n=("ticker", "size"))
        .reset_index())
rows = []
for e in ev.itertuples():
    i0 = days.searchsorted(e.disc + pd.Timedelta(days=1))
    if i0 + 20 > len(days):
        continue  # not enough future data yet
    rec = {"ticker": e.ticker, "disc": e.disc, "side": e.side, "relevant": e.relevant,
           "week": days[i0].strftime("%G-%V")}
    for h in (5, 20):
        r, s, m = ret(e.ticker, i0, h), ret(SECTOR_ETF[e.sector], i0, h), ret("SPY", i0, h)
        rec[f"x{h}"] = r - s - COST
        rec[f"m{h}"] = r - m - COST
    rows.append(rec)
R = pd.DataFrame(rows).dropna(subset=["x20", "x5"])


def stats(df, col):
    x = df[col].values
    n = len(x)
    if n < 3:
        return n, np.nan, np.nan
    e = x - x.mean()
    g = pd.Series(e).groupby(df["week"].values).sum()
    se = np.sqrt((g ** 2).sum()) / n
    return n, x.mean(), x.mean() / se if se > 0 else np.nan


def show(label, df):
    for col in ("x20", "x5", "m20"):
        n, m, t = stats(df, col)
        name = {"x20": "20d vs sector", "x5": " 5d vs sector", "m20": "20d vs SPY  "}[col]
        print(f"  {label:34s} {name}  n={n:4d}  mean={m*100:+6.2f}%  t={t:+5.2f}")


print("\n================ H-005 RESULTS ================")
P = R[(R.side == "buy") & R.relevant]
show("PRIMARY jurisdiction purchases", P)
print("--- context only ---")
show("all purchases", R[R.side == "buy"])
show("non-jurisdiction purchases", R[(R.side == "buy") & ~R.relevant])
show("jurisdiction sales", R[(R.side == "sell") & R.relevant])
show("all sales", R[R.side == "sell"])

n, m, t = stats(P, "x20")
print("\nVERDICT:", end=" ")
if n < 100:
    print(f"INCONCLUSIVE (n={n} < 100)")
elif m >= 0.005 and t >= 2.0:
    print(f"PASSES (mean {m*100:+.2f}%, t {t:+.2f}) -> paper-trade next")
else:
    print(f"DIES (mean {m*100:+.2f}%, t {t:+.2f})")
R.to_csv(os.path.join(D, "h005_events.csv"), index=False)
