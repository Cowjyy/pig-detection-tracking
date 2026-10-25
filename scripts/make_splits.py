"""Create reproducible group-level detection splits from an audited data manifest."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.preparation import ROOT, load_checked_records, new_output, sha256, source_path, write_json


def assign_groups(groups, ratios, seed):
    if len(groups) < 3:
        raise ValueError("At least three independent groups are needed for three splits")
    rng = random.Random(seed)
    ordered = sorted(groups)
    rng.shuffle(ordered)
    ordered.sort(key=lambda group: len(groups[group]), reverse=True)
    names = ("train", "val", "test")
    count = sum(len(ids) for ids in groups.values())
    targets = dict(zip(names, (count * v for v in ratios)))
    assigned = {name: [] for name in names}
    counts = Counter()
    for index, group in enumerate(ordered):
        empty = [name for name in names if not assigned[name]]
        if len(ordered) - index == len(empty):
            choice = max(empty, key=lambda name: targets[name])
        else:
            choice = max(names, key=lambda name: (targets[name] - counts[name]) / targets[name])
        assigned[choice].append(group)
        counts[choice] += len(groups[group])
    return assigned, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="Auditor's detection_records.json")
    parser.add_argument("--groups", help="CSV columns: image_id,group_id; sources/near duplicates share a group")
    parser.add_argument("--write-group-template", action="store_true", help="Write an unassigned review CSV only; do not split")
    parser.add_argument("--output", required=True, help="New datasets/splits/<version> directory")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1])
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    if any(v <= 0 for v in args.ratios) or abs(sum(args.ratios) - 1) > 1e-9:
        parser.error("Three positive ratios must sum to 1")
    record_path = ROOT / args.records
    records = load_checked_records(record_path)
    if args.write_group_template:
        output = new_output(args.output)
        with (output / "groups_to_review.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_id", "group_id", "source_note"])
            writer.writerows([r["id"], "", ""] for r in records)
        write_json(output / "status.json", dict(status="unassigned_template_not_a_split", records_sha256=sha256(record_path)))
        print(f"Created {output}; group_id fields intentionally blank")
        return
    if not args.groups:
        parser.error("--groups is required unless --write-group-template is used")
    group_path = ROOT / args.groups
    lookup = {r["id"]: r for r in records}
    if len(lookup) != len(records):
        raise ValueError("Duplicate image IDs")
    groups, membership = {}, {}
    with group_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            image_id, group_id = row["image_id"].strip(), row["group_id"].strip()
            if image_id not in lookup or image_id in membership or not group_id:
                raise ValueError(f"Invalid or duplicate group assignment: {row}")
            membership[image_id] = group_id
            groups.setdefault(group_id, []).append(image_id)
    if set(membership) != set(lookup):
        raise ValueError("Every audited image must have exactly one group assignment")
    # Identical bytes must never cross groups, even if the grouping CSV is mistaken.
    hash_groups = {}
    for image_id, record in lookup.items():
        digest = sha256(source_path(record["image"]))
        prior = hash_groups.setdefault(digest, membership[image_id])
        if prior != membership[image_id]:
            raise ValueError("Identical images are assigned to different groups")
    names = ("train", "val", "test")
    assigned, counts = assign_groups(groups, args.ratios, args.seed)
    result = {name: sorted(image_id for group in assigned[name] for image_id in groups[group]) for name in names}
    assert all(result.values())
    output = new_output(args.output)
    write_json(output / "splits.json", dict(seed=args.seed, requested_ratios=args.ratios,
        counts=dict(counts), group_counts={name: len(assigned[name]) for name in names},
        records_sha256=sha256(record_path), groups_sha256=sha256(group_path),
        groups=assigned, splits=result,
        limitations="Grouped split prevents declared-group overlap; image/video source overlap requires separate review."))
    for name in names:
        (output / f"{name}.txt").write_text("\n".join(lookup[i]["image"] for i in result[name]) + "\n", encoding="utf-8")
    print(json.dumps(dict(counts=counts, output=str(output)), indent=2))


if __name__ == "__main__":
    main()
