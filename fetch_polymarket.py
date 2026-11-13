"""Download settled Polymarket markets and their daily price history, for H-004 on Polymarket.
Public APIs, no account or keys. Run on your PC (needs: pip install requests).

    python fetch_polymarket.py              # full download (about 7,000 markets, 20-40 min)
    python fetch_polymarket.py --per-q 50   # quick trial run

Sample (fixed before any data was seen): for each calendar quarter from 2024Q1 to now, the
PER_Q highest-volume binary markets that CLOSED in that quarter (volume >= $10k). Spreading the
sample by quarter keeps one busy period (e.g. the 2024 US election) from dominating. Volume is a
liquidity filter; it says nothing about which side won.

Raw API responses are saved as-is (gzip JSON) so a field change does not need a new download.
Re-running skips markets whose history is already saved.
"""
import argparse
import gzip
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests is missing: pip install requests")

OUT = Path(__file__).resolve().parent / "data" / "polymarket"
GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB = "https://clob.polymarket.com/prices-history"
UA = {"User-Agent": "mesa-quant-tests/0.1"}
MIN_VOLUME = 10_000


_local = threading.local()


def session():
    """One reusable connection per thread: avoids a new TLS handshake on every request."""
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(UA)
    return _local.s


def get(url, params, tries=5):
    for attempt in range(1, tries + 1):
        try:
            r = session().get(url, params=params, timeout=20)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 * attempt)
                continue
            return r
        except requests.RequestException as exc:
            print(f"   network error ({exc}), retry {attempt}")
            time.sleep(2 * attempt)
    return None


def quarters(start=datetime(2024, 1, 1, tzinfo=timezone.utc)):
    now = datetime.now(timezone.utc)
    y, q = start.year, 1
    while True:
        a = datetime(y, 3 * (q - 1) + 1, 1, tzinfo=timezone.utc)
        if a > now:
            return
        b = datetime(y + (q == 4), (3 * q) % 12 + 1, 1, tzinfo=timezone.utc)
        yield f"{y}Q{q}", a, min(b, now)
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)


def is_binary(m):
    try:
        return len(json.loads(m.get("clobTokenIds") or "[]")) == 2 and len(json.loads(m.get("outcomes") or "[]")) == 2
    except (TypeError, ValueError):
        return False


def list_quarter(a, b, per_q):
    out, offset = [], 0
    while len(out) < per_q and offset < 5000:
        r = get(GAMMA, {"closed": "true", "end_date_min": a.isoformat(), "end_date_max": b.isoformat(),
                        "volume_num_min": MIN_VOLUME, "order": "volumeNum", "ascending": "false",
                        "limit": 100, "offset": offset})
        if r is None or r.status_code != 200:
            print(f"      gamma error: {None if r is None else (r.status_code, r.text[:200])}")
            break
        page = r.json()
        if not page:
            break
        out += [m for m in page if is_binary(m)]
        offset += len(page)
        time.sleep(0.15)
    return out[:per_q]


def history(token, m):
    """Daily points for one outcome token. Resolved markets only return coarse (>= 12h) data."""
    def ts(s):
        try:
            return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
        except Exception:
            return None
    end = ts(m.get("closedTime") or "") or ts(m.get("endDate") or "")
    start = ts(m.get("startDate") or m.get("createdAt") or "")
    tries = [{"market": token, "interval": "max", "fidelity": 1440}]
    if start and end:
        tries.append({"market": token, "startTs": start, "endTs": end + 86400, "fidelity": 1440})
    for p in tries:
        r = get(CLOB, p)
        if r is not None and r.status_code == 200:
            h = r.json().get("history") or []
            if h:
                return h
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-q", type=int, default=600, help="markets per quarter (default 600)")
    ap.add_argument("--workers", type=int, default=6, help="parallel downloads (default 6)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mfile, hfile = OUT / "markets.json.gz", OUT / "history_1d.json.gz"
    markets = {}
    for name, a, b in quarters():
        got = list_quarter(a, b, args.per_q)
        for m in got:
            m["_sample_quarter"] = name
            markets[m["id"]] = m
        print(f"   {name}: {len(got)} binary markets with volume >= ${MIN_VOLUME:,}")
    with gzip.open(mfile, "wt", encoding="utf-8") as f:
        json.dump(list(markets.values()), f)
    print(f"saved {len(markets)} markets")

    hist = json.load(gzip.open(hfile, "rt")) if hfile.exists() else {}
    todo = [m for m in markets.values() if m["id"] not in hist]
    print(f"price history: {len(hist)} already saved, {len(todo)} to fetch")
    def one(m):
        return m["id"], history(json.loads(m["clobTokenIds"])[0], m)   # outcome 0; outcome 1 is 1 - p

    empty, t0 = 0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, m) for m in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            mid, h = fut.result()
            empty += not h
            hist[mid] = h
            if k % 50 == 0 or k == len(todo):
                rate = k / max(time.time() - t0, 1e-9)
                print(f"   {k}/{len(todo)} done ({empty} empty), {rate:.1f}/s, "
                      f"~{(len(todo) - k) / rate / 60:.0f} min left", flush=True)
            if k % 500 == 0 or k == len(todo):
                with gzip.open(hfile, "wt", encoding="utf-8") as f:   # checkpoint
                    json.dump(hist, f)
    print("done")


if __name__ == "__main__":
    main()
