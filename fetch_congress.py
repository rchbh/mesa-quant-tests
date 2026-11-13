"""
fetch_congress.py  --  Step 1 of H-005 (congressional trades test)

Downloads, into mesa-quant-tests/data/congress/:
  1. LuxAlgo market-trackers-data congress-trade snapshots (CC0, built from
     Senate eFD + House Clerk filings). The repo layout is discovered via the
     GitHub API, because it isn't documented on the project page.
  2. unitedstates/congress-legislators: current members + committee membership.

Then prints the schema and a sample so the test script can be locked to the
real field names. Run from mesa-quant-tests with the stockbot venv active:
    python fetch_congress.py
"""
import gzip, io, json, os, sys, time
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "congress")
os.makedirs(OUT, exist_ok=True)

S = requests.Session()
S.headers["User-Agent"] = "mesa-quant-tests/1.0 (research)"
REPO = "LuxAlgo/market-trackers-data"
API = f"https://api.github.com/repos/{REPO}"


def get(url, **kw):
    for i in range(4):
        try:
            r = S.get(url, timeout=60, **kw)
            if r.status_code == 403 and "rate limit" in r.text.lower():
                print("  GitHub rate limit hit; waiting 60s ...")
                time.sleep(60)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"  retry {i+1}: {e}")
            time.sleep(3 * (i + 1))
    raise SystemExit(f"FAILED: {url}")


# ---------------------------------------------------------------- 1. discover
print("== Discovering repo layout ==")
info = get(API).json()
branch = info.get("default_branch", "main")
print("default branch:", branch)

tree = get(f"{API}/git/trees/{branch}", params={"recursive": "1"}).json()
paths = [t["path"] for t in tree.get("tree", []) if t["type"] == "blob"]
if tree.get("truncated"):
    print("  (tree listing truncated by GitHub; walking 'congress' dirs only)")
    def walk(p, depth=0):
        if depth > 4:
            return
        for it in get(f"{API}/contents/{p}", params={"ref": branch}).json():
            if it["type"] == "dir":
                walk(it["path"], depth + 1)
            elif it["type"] == "file":
                paths.append(it["path"])
    roots = [it["path"] for it in get(f"{API}/contents", params={"ref": branch}).json()
             if it["type"] == "dir"]
    for rt in roots:
        if "congress" in rt.lower():
            walk(rt, 1)
        else:
            for it in get(f"{API}/contents/{rt}", params={"ref": branch}).json():
                if it["type"] == "dir" and "congress" in it["path"].lower():
                    walk(it["path"], 2)

cong = sorted(p for p in paths if "congress" in p.lower())
print(f"{len(cong)} congress-related files. First 40:")
for p in cong[:40]:
    print("  ", p)

# year snapshots: prefer .json.gz (stdlib only), else .json
snaps = [p for p in cong if p.endswith(".json.gz")]
if not snaps:
    snaps = [p for p in cong if p.endswith(".json") and "latest" not in p.lower()
             and any(str(y) in os.path.basename(p) for y in range(2012, 2027))]
if not snaps:
    snaps = [p for p in cong if p.endswith("latest.json")]
print("\nSnapshot files chosen:")
for p in snaps:
    print("  ", p)
if not snaps:
    raise SystemExit("No congress snapshot files found -- send me this output.")

# ---------------------------------------------------------------- 2. download
rows = []
for p in snaps:
    url = f"https://raw.githubusercontent.com/{REPO}/{branch}/{p}"
    print("downloading", p)
    raw = get(url).content
    if p.endswith(".gz"):
        raw = gzip.decompress(raw)
    txt = raw.decode("utf-8")
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):  # envelope: find the list
            lists = [v for v in obj.values() if isinstance(v, list)]
            obj = max(lists, key=len) if lists else [obj]
        rows.extend(obj)
    except json.JSONDecodeError:  # JSON lines
        rows.extend(json.loads(l) for l in txt.splitlines() if l.strip())

with open(os.path.join(OUT, "trades_raw.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f)
print(f"\nSaved {len(rows):,} rows -> data/congress/trades_raw.json")

# ---------------------------------------------------------------- 3. legislators
LEG = "https://theunitedstates.io/congress-legislators/"
for name in ["legislators-current.json", "committee-membership-current.json",
             "committees-current.json"]:
    r = get(LEG + name)
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(r.content)
    print("saved", name)

# ---------------------------------------------------------------- 4. schema
print("\n== SCHEMA (send me everything below) ==")
keys = {}
for r in rows[:5000]:
    for k, v in (r.items() if isinstance(r, dict) else []):
        keys.setdefault(k, repr(v)[:70])
for k, v in keys.items():
    print(f"  {k:30s} e.g. {v}")
print("\nSample rows:")
for r in rows[:3]:
    print(json.dumps(r, default=str)[:600])
for k in keys:
    if "date" in k.lower() or "filed" in k.lower() or "disclos" in k.lower():
        vals = sorted(str(r.get(k)) for r in rows if isinstance(r, dict) and r.get(k))
        if vals:
            print(f"range {k}: {vals[0]} -> {vals[-1]}")
