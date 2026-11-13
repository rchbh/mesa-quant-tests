"""
fetch_senate.py -- H-005 data step (replaces fetch_congress.py for trades)

1. Scrapes every electronic Periodic Transaction Report (PTR) on the official
   Senate eFD site (efdsearch.senate.gov) received since 2025-01-03
   (start of the 119th Congress). Paper filings (scanned images) are skipped.
2. Downloads current committee membership (LuxAlgo market-trackers-data,
   sourced from unitedstates/congress-legislators).

Output in data/congress/:
   senate_trades.csv      one row per transaction
   committees.json        membership rows (bioguideId, memberName, chamber, committee)
Resumable: already-parsed reports are cached in senate_ptr_cache.json.

Run:  python fetch_senate.py          (full run, ~5-15 min)
      python fetch_senate.py --test   (first 3 reports only, prints them)
"""
import csv, gzip, json, os, re, sys, time
from html import unescape
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "congress")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "senate_ptr_cache.json")
TEST = "--test" in sys.argv
START = "01/03/2025 00:00:00"

BASE = "https://efdsearch.senate.gov"
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (research; mesa-quant-tests)"})


def req(method, url, **kw):
    for i in range(5):
        try:
            r = S.request(method, url, timeout=60, **kw)
            if r.status_code in (429, 503):
                time.sleep(10 * (i + 1)); continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"  retry {i+1} {url[-60:]}: {e}")
            time.sleep(4 * (i + 1))
    raise SystemExit(f"FAILED {url}")


# ------------------------------------------------------------ committees
print("== committee membership ==")
r = req("GET", "https://raw.githubusercontent.com/LuxAlgo/market-trackers-data/main/congress/committees/snapshot.json.gz")
obj = json.loads(gzip.decompress(r.content).decode("utf-8"))
if isinstance(obj, dict):
    obj = max((v for v in obj.values() if isinstance(v, list)), key=len)
mem = [o for o in obj if isinstance(o, dict) and o.get("bioguideId") and o.get("committee")]
with open(os.path.join(OUT, "committees.json"), "w", encoding="utf-8") as f:
    json.dump(mem, f)
print(f"  {len(mem)} membership rows saved")

# ------------------------------------------------------------ eFD session
print("== Senate eFD ==")
home = req("GET", f"{BASE}/search/home/")
tok = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', home.text).group(1)
req("POST", f"{BASE}/search/home/",
    data={"prohibition_agreement": "1", "csrfmiddlewaretoken": tok},
    headers={"Referer": f"{BASE}/search/home/"})
csrf = S.cookies.get("csrftoken")

reports, start = [], 0
while True:
    r = req("POST", f"{BASE}/search/report/data/", data={
        "start": str(start), "length": "100", "report_types": "[11]",
        "filer_types": "[]", "submitted_start_date": START, "submitted_end_date": "",
        "candidate_state": "", "senator_state": "", "office_id": "",
        "first_name": "", "last_name": "", "csrfmiddlewaretoken": csrf},
        headers={"Referer": f"{BASE}/search/", "X-CSRFToken": csrf})
    js = r.json()
    rows = js.get("data", [])
    for first, last, filer, link, received in rows:
        m = re.search(r'href="([^"]+)"', link)
        reports.append({"first": unescape(first).strip(), "last": unescape(last).strip(),
                        "filer": filer, "url": m.group(1) if m else "", "received": received,
                        "title": re.sub("<[^>]+>", "", unescape(link)).strip()})
    start += 100
    print(f"  listed {len(reports)} / {js.get('recordsTotal')}")
    if not rows or start >= int(js.get("recordsTotal", 0)) or (TEST and start >= 100):
        break
    time.sleep(0.5)

ptr = [x for x in reports if "/view/ptr/" in x["url"]]
print(f"  {len(ptr)} electronic PTRs, {len(reports) - len(ptr)} paper/other skipped")

cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}


def cell(td):
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", td))).strip()


def parse(html):
    body = html.split("<tbody>", 1)[-1].split("</tbody>", 1)[0]
    out = []
    for tr in re.findall(r"<tr.*?</tr>", body, flags=re.S):
        tds = [cell(t) for t in re.findall(r"<td.*?</td>", tr, flags=re.S)]
        if len(tds) >= 8:
            out.append(tds)
    return out


todo = ptr[:3] if TEST else ptr
for i, rep in enumerate(todo):
    if rep["url"] in cache and not TEST:
        continue
    html = req("GET", BASE + rep["url"], headers={"Referer": f"{BASE}/search/"}).text
    cache[rep["url"]] = parse(html)
    if TEST:
        print(rep); print(cache[rep["url"]][:5])
    if i % 25 == 0:
        print(f"  parsed {i+1}/{len(todo)}")
        json.dump(cache, open(CACHE, "w", encoding="utf-8"))
    time.sleep(0.4)
json.dump(cache, open(CACHE, "w", encoding="utf-8"))
if TEST:
    raise SystemExit("test done")

# ------------------------------------------------------------ write csv
n = 0
with open(os.path.join(OUT, "senate_trades.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["first", "last", "filer", "received", "url", "tx_date", "owner",
                "ticker", "asset", "asset_type", "type", "amount"])
    for rep in ptr:
        for t in cache.get(rep["url"], []):
            # columns: #, Transaction Date, Owner, Ticker, Asset Name, Asset Type, Type, Amount, Comment
            w.writerow([rep["first"], rep["last"], rep["filer"], rep["received"], rep["url"],
                        t[1], t[2], t[3], t[4], t[5], t[6], t[7]])
            n += 1
print(f"Saved {n} transactions -> data/congress/senate_trades.csv")
