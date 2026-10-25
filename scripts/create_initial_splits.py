"""Create an explicit initial split when acquisition groups remain unknown.

Numeric filename order is a proxy, NOT verified temporal/source independence.
No image, XML, video or GT is moved, copied, repaired or deleted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.preparation import ROOT, load_checked_records, new_output, relative, sha256, source_path, write_json


def ordered_split(records, ratios=(0.7, 0.2, 0.1)):
    if len(ratios) != 3 or any(v <= 0 or not math.isfinite(v) for v in ratios) or abs(sum(ratios)-1) > 1e-9:
        raise ValueError("Three positive finite ratios must sum to 1")
    if len(records) < 3 or any(not r["id"].isdigit() for r in records):
        raise ValueError("At least three numerically named images are required")
    if len({int(r["id"]) for r in records}) != len(records):
        raise ValueError("Numeric image IDs must be unique")
    ordered = sorted(records, key=lambda r: int(r["id"]))
    ideal = [len(ordered)*ratio for ratio in ratios]
    counts = [math.floor(value) for value in ideal]
    remainder_order = sorted(range(3), key=lambda i: (-(ideal[i]-counts[i]), i))
    for i in remainder_order[:len(ordered)-sum(counts)]:
        counts[i] += 1
    if not all(counts):
        raise ValueError("Every split must be nonempty")
    result, start = {}, 0
    for name, count in zip(("train", "val", "test"), counts):
        result[name] = ordered[start:start+count]
        start += count
    return result


def similarity_review(records, membership):
    from PIL import Image, ImageOps
    values = []
    for record in records:
        with Image.open(source_path(record["image"])) as im:
            small = ImageOps.grayscale(im).resize((9, 8))
            pixels = list(small.get_flattened_data() if hasattr(small, "get_flattened_data") else small.getdata())
            bits = sum(int(pixels[y*9+x] > pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
        values.append((record["id"], bits))
    counts, examples = Counter(), []
    for i, (a, ha) in enumerate(values):
        for b, hb in values[i+1:]:
            distance = (ha ^ hb).bit_count()
            if distance <= 4 and membership[a] != membership[b]:
                key = "__".join(sorted([membership[a], membership[b]]))
                counts[key] += 1
                if len(examples) < 100:
                    examples.append(dict(a=a, b=b, split_a=membership[a], split_b=membership[b], distance=distance))
    exact = {}
    for record in records:
        exact.setdefault(record["image_sha256"], set()).add(membership[record["id"]])
    conflicts = [digest for digest, sets in exact.items() if len(sets) > 1]
    if conflicts:
        raise ValueError("Identical image bytes would cross subsets; refuse this initial split")
    return dict(exact_duplicate_cross_split_groups=0, dhash_cross_split_candidate_counts=dict(counts),
                examples=examples, note="dHash<=4 measures appearance only; candidates are not confirmed duplicates. Cross-source/video overlap and temporal independence remain unverified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="experiments/preparation/audit_20261022_03")
    parser.add_argument("--output", required=True, help="New datasets/splits/<version> directory")
    args = parser.parse_args()
    audit = ROOT / args.audit
    record_path = audit / "detection_records.json"
    records = load_checked_records(record_path)
    summary = json.loads((audit / "summary.json").read_text(encoding="utf-8"))
    split_records = ordered_split(records)
    membership = {r["id"]: name for name, items in split_records.items() for r in items}
    similarity = similarity_review(records, membership)
    # Prespecified, no model scores: choose the earliest sequence from the largest
    # population (10 IDs) and the earliest from the smallest population (7 IDs).
    sequences = sorted(summary["tracking"], key=lambda r: r["sequence"])
    identity_counts = [r["identity_count"] for r in sequences]
    tuning = sorted({next(r["sequence"] for r in sequences if r["identity_count"] == n)
                     for n in (max(identity_counts), min(identity_counts))})
    if len(tuning) != 2:
        raise ValueError("Unable to select two distinct tuning sequences using this rule")
    track_records = []
    for record in sequences:
        track_records.append(dict(sequence=record["sequence"],
            subset="tuning" if record["sequence"] in tuning else "test",
            video=record["video"], gt=record["gt"], frames=record["frames_metadata"],
            video_sha256=sha256(source_path(record["video"])),
            gt_sha256=sha256(source_path(record["gt"])),
            audit_issues=record["issues"],
            evaluation_status="blocked_gt_review" if record["issues"] else "awaiting_protocol_and_overlap_review"))
    output = new_output(args.output)
    detection_dir, tracking_dir = output / "detection", output / "tracking"
    detection_dir.mkdir()
    tracking_dir.mkdir()
    manifest = dict(status="initial_v1_source_independence_unverified", strategy="numeric_order_contiguous_70_20_10",
        seed=None, requested_ratios=[0.7, 0.2, 0.1],
        records_sha256=sha256(record_path),
        counts={name: len(items) for name, items in split_records.items()},
        splits={name: [r["id"] for r in items] for name, items in split_records.items()},
        limitations=["Filename number is not verified capture time", "No gap is omitted at subset boundaries",
            "Adjacent views may remain dependent", "Image provenance and detection/MOT video overlap unknown",
            "Not a claim of independent cross-source generalization"], similarity_review=similarity)
    write_json(detection_dir / "splits.json", manifest)
    for name, items in split_records.items():
        (detection_dir / f"{name}.txt").write_text("\n".join(r["image"] for r in items)+"\n", encoding="utf-8")
        with (detection_dir / f"{name}.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_id", "image", "xml", "width", "height", "object_count"])
            writer.writerows([r["id"], r["image"], r["xml"], r["width"], r["height"], len(r["objects"])] for r in items)
    tracking = dict(status="sequence_assignment_v1_gt_and_provenance_review_pending",
        selection_rule="Earliest sequence at largest and smallest identity counts for tuning; no model scores used",
        training_sequences=[], tuning_sequences=tuning,
        test_sequences=[r["sequence"] for r in track_records if r["subset"] == "test"],
        split_unit="entire_video", sequences=track_records,
        blocked_test_sequences=[r["sequence"] for r in track_records if r["evaluation_status"] == "blocked_gt_review"],
        limitations="No tracking training subset; this task validates trackers. Do not tune on test or assume distinct videos are independent captures.")
    write_json(tracking_dir / "splits.json", tracking)
    for subset in ("tuning", "test"):
        (tracking_dir / f"{subset}.txt").write_text("\n".join(r["sequence"] for r in track_records if r["subset"] == subset)+"\n", encoding="utf-8")
    write_json(output / "manifest.json", dict(version="initial_v1", audit=relative(audit),
        detection="detection/splits.json", tracking="tracking/splits.json",
        source_data_modified=False, source_independence_verified=False,
        generator_sha256=sha256(Path(__file__))))
    print(json.dumps(dict(output=relative(output), detection=manifest["counts"],
        tuning=tuning, test=tracking["test_sequences"], blocked=tracking["blocked_test_sequences"],
        similarity=similarity["dhash_cross_split_candidate_counts"]), indent=2))


if __name__ == "__main__":
    main()
