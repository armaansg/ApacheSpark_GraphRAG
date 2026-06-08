import json, time, urllib.parse, urllib.request
from pathlib import Path

CONFIG = Path("config/jira.yaml")
OUT = Path("data/raw/jira/issues")
FIELDS = (
    "key,summary,description,status,issuetype,priority,components,labels,"
    "fixVersions,versions,created,updated,comment,assignee,reporter"
)


def load_config(path: Path) -> dict:
    cfg = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        cfg[key.strip()] = value.strip().strip('"').strip("'")
    return cfg


cfg = load_config(CONFIG)
BASE = cfg["base_url"].rstrip("/") + "/rest/api/2/search"
JQL = cfg["jql"]
PAGE = int(cfg.get("page_size", 50))
OUT.mkdir(parents=True, exist_ok=True)

start = 0
total = None
seen = 0

while total is None or start < total:
    params = urllib.parse.urlencode({
        "jql": JQL,
        "startAt": start,
        "maxResults": PAGE,
        "fields": FIELDS,
    })
    url = f"{BASE}?{params}"
    with urllib.request.urlopen(url) as resp:
        data = json.load(resp)
    if total is None:
        total = data["total"]
        print(f"JQL: {JQL}")
        print(f"Total issues to fetch: {total}")
    issues = data.get("issues", [])
    if not issues:
        break
    for issue in issues:
        key = issue["key"]
        path = OUT / f"{key}.json"
        path.write_text(json.dumps(issue, indent=2))
        seen += 1
    start += PAGE
    print(f"Fetched {seen}/{total}")
    time.sleep(1.5)
print("Done:", seen)