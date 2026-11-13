"""Download the public data needed to test two Mesa quant hypotheses. No accounts or keys.

  H-005  Bitcoin implied volatility (Deribit DVOL index, daily) + BTC daily prices (Yahoo).
  H-004  Settled Kalshi markets and their daily candlesticks, for the favourite-longshot study.

Raw API responses are saved as-is (gzip JSON) so field-name changes do not need a new download.

    python fetch_public.py            # both
    python fetch_public.py --only h005
    python fetch_public.py --only h004
"""
import argparse
import gzip
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests is missing: pip install requests yfinance pandas")

OUT = Path(__file__).resolve().parent / "data"
UA = {"User-Agent": "mesa-quant-tests/0.1"}
DERIBIT = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
KALSHI_HOSTS = ["https://api.elections.kalshi.com/trade-api/v2", "https://external-api.kalshi.com/trade-api/v2"]
KALSHI_SERIES = ["KXFED", "KXCPI", "KXPAYROLLS", "KXBTCD", "KXHIGHNY", "KXINXU", "KXETHD", "KXGDP", "KXU3"]
MAX_MARKETS_PER_SERIES = 2500
START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f)


def get(url, params, tries=4):
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
            if r.status_code == 429:
                time.sleep(2 * attempt)
                continue
            return r
        except requests.RequestException as exc:
            print(f"   network error ({exc}), retry {attempt}")
            time.sleep(2 * attempt)
    return None


# ---------------------------------------------------------------- H-005

def fetch_h005():
    print("H-005: Deribit DVOL (BTC implied volatility index), daily")
    end = int(time.time() * 1000)
    start = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    rows, cursor_end = [], end
    for _ in range(50):
        r = get(DERIBIT, {"currency": "BTC", "start_timestamp": start, "end_timestamp": cursor_end,
                          "resolution": "1D"})
        if r is None or r.status_code != 200:
            print(f"   Deribit error: {None if r is None else (r.status_code, r.text[:300])}")
            break
        res = r.json().get("result", {})
        data = res.get("data", [])
        rows += data
        cont = res.get("continuation")
        print(f"   got {len(data)} days, continuation={cont}")
        if not cont or not data or cont <= start:
            break
        cursor_end = cont
        time.sleep(0.3)
    save(OUT / "deribit_dvol_btc_1d.json.gz", rows)
    print(f"   saved {len(rows)} rows")

    print("H-005: BTC-USD daily prices (Yahoo)")
    try:
        import yfinance as yf
        df = yf.download("BTC-USD", start="2020-06-01", interval="1d", auto_adjust=False, progress=False)
        if hasattr(df.columns, "levels"):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
        df.index = df.index.strftime("%Y-%m-%d")
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / "btc_usd_1d.csv")
        print(f"   saved {len(df)} days")
    except Exception as exc:
        print(f"   Yahoo error: {exc}")


# ---------------------------------------------------------------- H-004

def kalshi_host():
    for h in KALSHI_HOSTS:
        r = get(h + "/exchange/status", {})
        if r is not None and r.status_code == 200:
            print(f"   using {h}")
            return h
        print(f"   {h} -> {None if r is None else r.status_code}")
    return None


def list_markets(host, path, params):
    out, cursor = [], None
    while len(out) < MAX_MARKETS_PER_SERIES:
        p = dict(params, limit=1000)
        if cursor:
            p["cursor"] = cursor
        r = get(host + path, p)
        if r is None or r.status_code != 200:
            print(f"      {path} -> {None if r is None else (r.status_code, r.text[:200])}")
            break
        js = r.json()
        out += js.get("markets", [])
        cursor = js.get("cursor")
        if not cursor:
            break
        time.sleep(0.12)
    return out


def fetch_h004():
    print("H-004: Kalshi settled markets + daily candlesticks")
    host = kalshi_host()
    if host is None:
        print("   Kalshi API not reachable from here")
        return
    r = get(host + "/historical/cutoff", {})
    cutoff = r.json() if r is not None and r.status_code == 200 else {}
    save(OUT / "kalshi" / "cutoff.json.gz", cutoff)
    print(f"   historical cutoff: {cutoff}")
    for series in KALSHI_SERIES:
        recent = list_markets(host, "/markets", {"series_ticker": series, "status": "settled"})
        old = list_markets(host, "/historical/markets", {"series_ticker": series})
        markets = {m["ticker"]: (m, "historical" if i >= len(recent) else "live")
                   for i, m in enumerate(recent + old) if "ticker" in m}
        print(f"   {series}: {len(recent)} recent + {len(old)} historical settled markets")
        save(OUT / "kalshi" / f"{series}_markets.json.gz", [m for m, _ in markets.values()])
        candles, fails = {}, 0
        for k, (ticker, (m, where)) in enumerate(markets.items()):
            close_s = m.get("close_time") or m.get("expiration_time")
            open_s = m.get("open_time")
            try:
                end_ts = int(datetime.fromisoformat(close_s.replace("Z", "+00:00")).timestamp())
                start_ts = int(datetime.fromisoformat(open_s.replace("Z", "+00:00")).timestamp()) if open_s else end_ts - 30 * 86400
            except Exception:
                continue
            if end_ts < START.timestamp():
                continue
            params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": 1440}
            paths = ([f"/historical/markets/{ticker}/candlesticks", f"/series/{series}/markets/{ticker}/candlesticks"]
                     if where == "historical" else
                     [f"/series/{series}/markets/{ticker}/candlesticks", f"/historical/markets/{ticker}/candlesticks"])
            got = None
            for p in paths:
                rr = get(host + p, params)
                if rr is not None and rr.status_code == 200:
                    got = rr.json()
                    break
            if got is None:
                fails += 1
                if fails <= 3:
                    print(f"      candles failed for {ticker}: {None if rr is None else (rr.status_code, rr.text[:200])}")
            else:
                candles[ticker] = got
            time.sleep(0.11)   # stay under 10 requests per second
            if (k + 1) % 250 == 0:
                print(f"      {series}: {k + 1}/{len(markets)} markets")
        save(OUT / "kalshi" / f"{series}_candles_1d.json.gz", candles)
        print(f"   {series}: candles for {len(candles)} markets ({fails} failed)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["h004", "h005"])
    args = ap.parse_args()
    if args.only in (None, "h005"):
        fetch_h005()
    if args.only in (None, "h004"):
        fetch_h004()
    print("done")


if __name__ == "__main__":
    main()
