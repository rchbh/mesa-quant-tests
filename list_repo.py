"""Save the LuxAlgo market-trackers-data file list to data/congress/repo_paths.txt"""
import os, requests
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "congress", "repo_paths.txt")
API = "https://api.github.com/repos/LuxAlgo/market-trackers-data"
s = requests.Session()
branch = s.get(API, timeout=60).json().get("default_branch", "main")
tree = s.get(f"{API}/git/trees/{branch}", params={"recursive": "1"}, timeout=120).json()
lines = [f"branch={branch} truncated={tree.get('truncated')}"]
lines += [f"{t['type']}\t{t.get('size', '')}\t{t['path']}" for t in tree.get("tree", [])]
if tree.get("truncated") or len(lines) < 3:
    # fall back: list top two levels via contents API
    for it in s.get(f"{API}/contents", params={"ref": branch}, timeout=60).json():
        lines.append(f"{it['type']}\t{it.get('size','')}\t{it['path']}")
        if it["type"] == "dir":
            for sub in s.get(f"{API}/contents/{it['path']}", params={"ref": branch}, timeout=60).json():
                lines.append(f"{sub['type']}\t{sub.get('size','')}\t{sub['path']}")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"saved {len(lines)} lines -> {OUT}")
