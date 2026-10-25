"""Compare raw file names, sizes, modification times and SHA-256 with a saved baseline."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.preparation import ROOT, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--ignore-mtime", action="store_true", help="For server copies, compare content and paths only")
    args = parser.parse_args()
    expected = {r["path"]: r for r in json.loads((ROOT / args.inventory).read_text(encoding="utf-8"))}
    actual = {r["path"]: r for r in inventory()}
    if args.ignore_mtime:
        for collection in (expected, actual):
            for item in collection.values():
                item.pop("mtime_ns", None)
    result = dict(added=sorted(actual.keys()-expected.keys()), removed=sorted(expected.keys()-actual.keys()),
        changed=[p for p in sorted(actual.keys() & expected.keys()) if actual[p] != expected[p]])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if any(result.values()):
        raise SystemExit(1)
    print(f"PASS: {len(actual)} raw files unchanged")


if __name__ == "__main__":
    main()
