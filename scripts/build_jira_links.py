import json 
from pathlib import Path 
ISSUES_DIR = Path("data/raw/jira/issues")
CONFIG_PATH = Path("config/jira_components.yaml")
OUT_PATH = Path("data/normalized/links/jira_modules.jsonl")

def load_component_config(path: Path) -> dict:
    mappings = {}
    unmapped = set()
    section = None

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "mappings:":
            section = "mappings"
            continue
        if line == "unmapped:":
            section = "unmapped"
            continue
        if line.startswith("- "):
            if section == "unmapped":
                unmapped.add(line[2:].strip())
            continue
        if section == "mappings" and ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().strip('"').strip("'")
            value = value.strip()
            if value.startswith("["):
                modules = json.loads(value.replace("'", '"'))
                mappings[key] = modules
    return {"mappings": mappings, "unmapped": unmapped}


def main():
    cfg = load_component_config(CONFIG_PATH)
    mappings = cfg["mappings"]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_issues = 0
    total_links = 0
    mapped_links = 0
    unmapped_links = 0

    with OUT_PATH.open("w") as out:
        for issue_path in sorted(ISSUES_DIR.glob("SPARK-*.json")):
            issue = json.loads(issue_path.read_text())
            key = issue["key"]
            fields = issue["fields"]
            summary = fields.get("summary")
            status = (fields.get("status") or {}).get("name")
            components = fields.get("components") or []
            total_issues += 1

            if not components:
                row = {
                    "jira_key": key,
                    "component": None,
                    "module": None,
                    "mapped": False,
                    "summary": summary,
                    "status": status,
                    "link_type": "RELATES_TO",
                    "reason": "no_component",
                }
                out.write(json.dumps(row) + "\n")
                total_links += 1
                unmapped_links += 1
                continue
            
            for comp in components:
                comp_name = comp["name"]
                modules = mappings.get(comp_name)
                if modules:
                    for module in modules:
                        row = {
                            "jira_key": key,
                            "component": comp_name,
                            "module": module,
                            "mapped": True,
                            "summary": summary,
                            "status": status,
                            "link_type": "RELATES_TO",
                        }
                        out.write(json.dumps(row) + "\n")
                        total_links += 1
                        mapped_links += 1
                else:
                    row = {
                        "jira_key": key,
                        "component": comp_name,
                        "module": None,
                        "mapped": False,
                        "summary": summary,
                        "status": status,
                        "link_type": "RELATES_TO",
                        "reason": "unmapped_component",
                    }
                    out.write(json.dumps(row) + "\n")
                    total_links += 1
                    unmapped_links += 1

    print(f"Issues processed: {total_issues}")
    print(f"Links written:   {total_links}")
    print(f"Mapped:          {mapped_links}")
    print(f"Unmapped:        {unmapped_links}")
    print(f"Wrote {OUT_PATH}")

    
if __name__ == "__main__":
    main()