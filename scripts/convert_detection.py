"""Convert audited VOC boxes to COCO and YOLO labels without changing source data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.preparation import ROOT, load_checked_records, new_output, sha256, source_path, write_json


def convert_box(box, width, height, convention):
    x1, y1, x2, y2 = box
    if convention == "voc_1based_inclusive":
        if x1 < 1 or y1 < 1:
            raise ValueError("Zero minimum coordinate conflicts with 1-based VOC; verify annotation convention")
        x1 -= 1
        y1 -= 1
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid box after conversion: {box}")
    return [x1, y1, x2 - x1, y2 - y1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--coordinate-convention", required=True,
                        choices=["voc_1based_inclusive", "zero_based_half_open"])
    parser.add_argument("--difficult-policy", required=True, choices=["include", "exclude"],
                        help="Use the SAME policy for all detectors; excluded boxes are not silently converted to iscrowd")
    parser.add_argument("--coordinate-review", help="Evidence JSON declaring the coordinate protocol")
    parser.add_argument("--output", required=True, help="New datasets/annotations/<version> directory")
    args = parser.parse_args()
    record_path, split_path = ROOT / args.records, ROOT / args.splits
    records = load_checked_records(record_path)
    coordinate_review = None
    if args.coordinate_review:
        review_path = ROOT / args.coordinate_review
        coordinate_review = json.loads(review_path.read_text(encoding='utf-8'))
        if (coordinate_review['selected_convention'] != args.coordinate_convention
                or coordinate_review['records_sha256'] != sha256(record_path)):
            raise ValueError('Coordinate review does not match this conversion')
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    if split_data["records_sha256"] != sha256(record_path):
        raise ValueError("Split manifest was generated from a different audit")
    if any(r["issues"] for r in records):
        raise ValueError("Audit contains unresolved detection issues")
    lookup = {r["id"]: r for r in records}
    all_ids = [i for ids in split_data["splits"].values() for i in ids]
    reserved_ids = split_data.get("reserved_ids", [])
    if (len(all_ids) != len(set(all_ids)) or len(reserved_ids) != len(set(reserved_ids))
            or set(all_ids)&set(reserved_ids) or set(all_ids)|set(reserved_ids) != set(lookup)):
        raise ValueError("Active split IDs and explicit reserve IDs must cover the dataset exactly once")
    datasets, yolo, sources = {}, {}, []
    excluded = 0
    for split in ("train", "val", "test"):
        data = dict(images=[], annotations=[], categories=[dict(id=1, name="pig")])
        for image_id, key in enumerate(split_data["splits"][split], 1):
            record = lookup[key]
            source_path(record["image"])  # Validate portability and source boundary.
            width, height = record["width"], record["height"]
            data["images"].append(dict(id=image_id, file_name=record["image"], width=width, height=height))
            lines = []
            for obj in record["objects"]:
                if obj["difficult"] and args.difficult_policy == "exclude":
                    excluded += 1
                    continue
                x, y, w, h = convert_box(obj["bbox_xyxy"], width, height, args.coordinate_convention)
                data["annotations"].append(dict(id=len(data["annotations"]) + 1, image_id=image_id,
                    category_id=1, bbox=[x, y, w, h], area=w*h, iscrowd=0))
                lines.append(f"0 {(x+w/2)/width:.9f} {(y+h/2)/height:.9f} {w/width:.9f} {h/height:.9f}")
            yolo[(split, key)] = lines
            sources.append(dict(image_id=key, split=split, image=record["image"], source_xml=record["xml"]))
        datasets[split] = data
    output = new_output(args.output)
    for split, data in datasets.items():
        write_json(output / f"{split}.coco.json", data)
        (output / "yolo" / split).mkdir(parents=True)
    for (split, key), lines in yolo.items():
        (output / "yolo" / split / f"{key}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    write_json(output / "conversion.json", dict(coordinate_convention=args.coordinate_convention,
        coordinate_review=args.coordinate_review,
        coordinate_review_sha256=sha256(ROOT / args.coordinate_review) if args.coordinate_review else None,
        original_annotation_tool_verified=coordinate_review.get('original_annotation_tool_verified') if coordinate_review else None,
        difficult_policy=args.difficult_policy, excluded_objects=excluded,
        records_sha256=sha256(record_path), splits_sha256=sha256(split_path), sources=sources,
        reserved_ids=reserved_ids, evaluation_scope=split_data.get("evaluation_scope", "see split manifest"),
        coco_image_root="project root (file_name is project-relative)",
        yolo_note="Labels only; use a framework adapter to pair source images with these labels. No training YAML is implied."))
    print(f"Created {output}; source images and XML were not modified")


if __name__ == "__main__":
    main()
