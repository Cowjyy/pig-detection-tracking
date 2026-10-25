"""Publish guarded within-scene splits and validated derived tracking GT links.

Source independence is not invented. All held-out/reserved source files remain intact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.create_initial_splits import ordered_split
from tools.mot_geometry import validate_gt
from tools.preparation import ROOT, load_checked_records, new_output, relative, sha256, source_path, write_json


def content_components(ids, hashes, threshold=4):
    parent = {key: key for key in ids}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    for i, a in enumerate(ids):
        for b in ids[i+1:]:
            if (hashes[a] ^ hashes[b]).bit_count() <= threshold:
                ra, rb = find(a), find(b)
                parent[max(ra, rb, key=int)] = min(ra, rb, key=int)
    return {key: find(key) for key in ids}


def buffered_assignment(records, gap):
    if gap < 1:
        raise ValueError("Positive boundary gap required")
    ordered = sorted(records, key=lambda r: int(r["id"]))
    usable = len(ordered)-2*gap
    if usable < 30:
        raise ValueError("Not enough images after two boundary gaps")
    count = {name: len(items) for name,items in ordered_split(ordered[:usable]).items()}
    assigned, reserve = {}, {}
    offset = 0
    for name in ("train", "val", "test"):
        for record in ordered[offset:offset+count[name]]:
            assigned[record["id"]] = name
        offset += count[name]
        if name != "test":
            for record in ordered[offset:offset+gap]:
                reserve[record["id"]] = "boundary_gap_numeric_order_proxy"
            offset += gap
    assert offset == len(ordered)
    return assigned, reserve


def quarantine_components(assigned, reserve, components):
    priority = {"train": 0, "val": 1, "test": 2}
    destinations = {}
    for key, subset in assigned.items():
        component = components[key]
        if component not in destinations or priority[subset] > priority[destinations[component]]:
            destinations[component] = subset
    active = dict(assigned)
    reserved = dict(reserve)
    for key, subset in assigned.items():
        if subset != destinations[components[key]]:
            reserved[key] = "cross_subset_dhash_component_quarantine"
            del active[key]
    return active, reserved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gap", type=int, default=20, help="Number of source images reserved at EACH numeric boundary")
    parser.add_argument("--records", default="experiments/preparation/audit_20261022_03/detection_records.json")
    parser.add_argument("--overlap", default="experiments/preparation/content_overlap_v2/summary.json")
    parser.add_argument("--gt-manifest", default="datasets/annotations/mot_clean_v1/manifest.json")
    args = parser.parse_args()
    from PIL import Image, ImageOps
    records = sorted(load_checked_records(ROOT/args.records), key=lambda r: int(r["id"]))
    overlap = json.loads((ROOT/args.overlap).read_text(encoding="utf-8"))
    if overlap["status"] != "completed_content_search_not_provenance_certification" or not all(c["accepted"] for c in overlap["jpeg_positive_controls"]):
        raise ValueError("Only a completed, self-checked content search may be used")
    if overlap["records_sha256"] != sha256(ROOT/args.records):
        raise ValueError("Content search uses a different audit")
    gt_manifest = json.loads((ROOT/args.gt_manifest).read_text(encoding="utf-8"))
    if gt_manifest["status"] != "geometry_valid_derived_gt":
        raise ValueError("Tracking requires a validated derived manifest")
    hashes = {}
    for record in records:
        with Image.open(source_path(record["image"])) as im:
            small = ImageOps.grayscale(im).resize((9, 8))
            pixels = list(small.get_flattened_data() if hasattr(small,"get_flattened_data") else small.getdata())
        hashes[record["id"]] = sum(int(pixels[y*9+x]>pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
    ids = [r["id"] for r in records]
    components = content_components(ids, hashes)
    assigned, reserve = buffered_assignment(records, args.gap)
    assigned, reserve = quarantine_components(assigned, reserve, components)
    # Directly matched video content is not allowed in detection training/validation,
    # regardless of which video will be used for tuning or final tracking evaluation.
    video_match_components = {components[r["image_id"]] for r in overlap["observed_duplicate_matches"]}
    for key in list(assigned):
        if components[key] in video_match_components:
            reserve[key] = "video_duplicate_component_quarantine"
            del assigned[key]
    splits = {name: [key for key in ids if assigned.get(key) == name] for name in ("train","val","test")}
    if not all(splits.values()):
        raise ValueError("Conservative filtering leaves an empty subset; cannot publish")
    assert set(assigned).isdisjoint(reserve) and set(assigned)|set(reserve) == set(ids)
    component_sets = {name: {components[key] for key in keys} for name,keys in splits.items()}
    assert not (component_sets["train"] & component_sets["val"] or component_sets["train"] & component_sets["test"] or component_sets["val"] & component_sets["test"])
    lookup = {r["id"]: r for r in records}
    track_rows = []
    for record in gt_manifest["sequences"]:
        gt = (ROOT/record["gt"]).resolve()
        if not gt.is_relative_to((ROOT/"datasets"/"annotations").resolve()) or sha256(gt) != record["gt_sha256"]:
            raise ValueError("Derived GT path or hash invalid")
        if sha256(source_path(record["source_gt"])) != record["source_gt_sha256"] or sha256(source_path(record["video"])) != record["video_sha256"]:
            raise ValueError("Raw tracking source changed")
        validate_gt(gt.read_text().splitlines(), record["width"], record["height"], record["frames"])
        track_rows.append(dict(**record, subset="tuning" if record["sequence"] in ("191","194") else "test",
            evaluation_status="geometry_valid_for_derived_gt_within_scene_protocol"))
    output = new_output(args.output)
    det, track = output/"detection", output/"tracking"
    det.mkdir()
    track.mkdir()
    counts = {name: len(keys) for name,keys in splits.items()}
    reserved_ids = [key for key in ids if key in reserve]
    write_json(det/"splits.json", dict(status="within_scene_buffered_content_guarded_v2", seed=None,
        strategy="numeric_contiguous_with_20_image_boundary_gaps_then_dhash_component_quarantine",
        evaluation_scope="same_scene_holdout_not_source_independent_generalization",
        source_independence_verified=False, records_sha256=sha256(ROOT/args.records),
        requested_ratios=[0.7,0.2,0.1], counts=counts, splits=splits, reserved_ids=reserved_ids,
        reserve_counts=dict(Counter(reserve.values())), boundary_gap_images=args.gap,
        dhash_threshold=4, dhash_groups=len(set(components.values())),
        cross_subset_dhash_candidate_pairs=0, direct_video_duplicate_images=overlap["observed_duplicate_images"],
        overlap_report=args.overlap, overlap_report_sha256=sha256(ROOT/args.overlap),
        limitations=["Filename order is a proxy; a 20-image gap is not a known elapsed time",
            "Conservative appearance quarantine may reserve nonduplicate views of the same background",
            "No match in a finite visual search is not proof of different animal or recording source",
            "Use only for declared within-scene evaluation; independent-source claims need new provenance or collection"]))
    for name, keys in {**splits, "reserve":reserved_ids}.items():
        (det/f"{name}.txt").write_text("\n".join(lookup[key]["image"] for key in keys)+"\n", encoding="utf-8")
        with (det/f"{name}.csv").open("x",encoding="utf-8-sig",newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_id","image","xml","object_count","component","reserve_reason"])
            writer.writerows([key,lookup[key]["image"],lookup[key]["xml"],len(lookup[key]["objects"]),components[key],reserve.get(key,"")] for key in keys)
    tracking = dict(status="derived_gt_geometry_valid_within_scene_protocol", gt_protocol="mot_clean_v1",
        gt_manifest=args.gt_manifest, gt_manifest_sha256=sha256(ROOT/args.gt_manifest),
        tuning_sequences=["191","194"], test_sequences=["192","193","195","196","197"],
        blocked_test_sequences=[], raw_gt_repaired_in_place=False, sequences=track_rows,
        source_independence_verified=False, scope="derived-label within-scene benchmark; disclose exclusions/clipping")
    write_json(track/"splits.json",tracking)
    for name in ("tuning","test"):
        (track/f"{name}.txt").write_text("\n".join(tracking[f"{name}_sequences"])+"\n",encoding="utf-8")
    write_json(output/"manifest.json",dict(version="protocol_v2",detection="detection/splits.json",tracking="tracking/splits.json",
        raw_data_modified=False, evaluation_scope="within_scene_content_guarded_derived_gt",
        source_independence_verified=False, generator_sha256=sha256(Path(__file__))))
    print(json.dumps(dict(output=relative(output),detection=counts,reserve=len(reserve),
        reserve_reasons=dict(Counter(reserve.values())),tracking_gt_geometry_valid=True),indent=2))


if __name__ == "__main__":
    main()
