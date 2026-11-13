"""
fetch_house.py -- H-005 data step (House of Representatives)

The Senate eFD site blocks connections from Venezuela (Akamai 403), so H-005
uses House trades. Source: official House Clerk disclosures.
 1. Yearly index zips (2025FD.zip, 2026FD.zip) -> list of PTR filings
    (FilingType P) with filing date = disclosure date.
 2. Each electronic PTR PDF (DocID starting with 2; 8xxxx are scanned paper,
    skipped) is downloaded and its text parsed for stock rows "(TICKER) [ST]".
 3. Committee membership (LuxAlgo / congress-legislators) -> committees.json

Output: data/congress/house_trades.csv, committees.json. Resumable cache.
Needs:  pip install pypdf
Run:    python fetch_house.py --test   (3 filings, prints parsed rows)
        python fetch_house.py
"""
import csv, gzip, io, json, os, re, sys, time, zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
import requests
from pypdf import PdfReader

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "congress")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "house_ptr_cache.json")
TEST = "--test" in sys.argv
BASE = "https://disclosures-clerk.house.gov/public_disc"
FIRST_DAY = datetime(2025, 1, 3)

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (research; mesa-quant-tests)"


def get(url):
    for i in range(5):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"  retry {i+1} {url[-50:]}: {e}")
            time.sleep(4 * (i + 1))
    print(f"  GAVE UP {url}")
    return None


# ------------------------------------------------------------ committees
if not os.path.exists(os.path.join(OUT, "committees.json")) or "--refresh" in sys.argv:
    r = get("https://raw.githubusercontent.com/LuxAlgo/market-trackers-data/main/congress/committees/snapshot.json.gz")
    obj = json.loads(gzip.decompress(r.content).decode("utf-8"))
    if isinstance(obj, dict):
        obj = max((v for v in obj.values() if isinstance(v, list)), key=len)
    mem = [o for o in obj if isinstance(o, dict) and o.get("bioguideId") and o.get("committee")]
    json.dump(mem, open(os.path.join(OUT, "committees.json"), "w", encoding="utf-8"))
    print(f"committees: {len(mem)} rows")

# ------------------------------------------------------------ index
filings = []
for year in (2025, 2026):
    r = get(f"{BASE}/financial-pdfs/{year}FD.zip")
    if r is None:
        continue
    z = zipfile.ZipFile(io.BytesIO(r.content))
    xmlname = [n for n in z.namelist() if n.lower().endswith(".xml")][0]
    root = ET.fromstring(z.read(xmlname))
    for m in root.iter("Member"):
        g = lambda k: (m.findtext(k) or "").strip()
        if g("FilingType") != "P":
            continue
        try:
            fd = datetime.strptime(g("FilingDate"), "%m/%d/%Y")
        except ValueError:
            continue
        if fd < FIRST_DAY:
            continue
        filings.append({"first": g("First"), "last": g("Last"), "state": g("StateDst"),
                        "filed": fd.strftime("%Y-%m-%d"), "doc": g("DocID"), "year": year})
elec = [f for f in filings if f["doc"].startswith("2")]
print(f"PTR filings since 2025-01-03: {len(filings)}  electronic: {len(elec)}  (paper skipped: {len(filings)-len(elec)})")

# ------------------------------------------------------------ parse PDFs
ROW = re.compile(
    r"\(([A-Z][A-Z.\-]{0,6})\)\s*\[(ST|OP|ET|EF|MF|CS|GS|PS|AB|OT)\]"   # ticker + asset class
    r".{0,80}?\s(P|S|S \(partial\)|E)\s+"                                # transaction type
    r"(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})", re.S)                   # tx date, notification date

cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
todo = elec[:3] if TEST else elec
for i, f in enumerate(todo):
    if f["doc"] in cache and not TEST:
        continue
    r = get(f"{BASE}/ptr-pdfs/{f['year']}/{f['doc']}.pdf")
    rows = []
    if r is not None:
        try:
            txt = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(r.content)).pages)
            txt = txt.replace("\x00", " ")
            for t, cls, typ, txd, nd in ROW.findall(txt):
                rows.append([t, cls, typ, txd, nd])
            if TEST:
                print(f, "\n--- text sample ---\n", txt[:1200], "\n--- parsed ---\n", rows[:8])
        except Exception as e:
            print("  pdf error", f["doc"], e)
    cache[f["doc"]] = rows
    if i % 50 == 0:
        print(f"  parsed {i+1}/{len(todo)}")
        json.dump(cache, open(CACHE, "w", encoding="utf-8"))
    time.sleep(0.25)
json.dump(cache, open(CACHE, "w", encoding="utf-8"))
if TEST:
    raise SystemExit("test done")

TYPE = {"P": "Purchase", "S": "Sale", "S (partial)": "Sale (Partial)", "E": "Exchange"}
n = 0
with open(os.path.join(OUT, "house_trades.csv"), "w", newline="", encoding="utf-8") as fo:
    w = csv.writer(fo)
    w.writerow(["chamber", "first", "last", "state", "received", "doc", "tx_date",
                "ticker", "asset_type", "type"])
    for f in elec:
        for t, cls, typ, txd, nd in cache.get(f["doc"], []):
            w.writerow(["house", f["first"], f["last"], f["state"], f["filed"], f["doc"], txd,
                        t, "Stock" if cls == "ST" else cls, TYPE.get(typ, typ)])
            n += 1
print(f"Saved {n} transactions -> data/congress/house_trades.csv")
