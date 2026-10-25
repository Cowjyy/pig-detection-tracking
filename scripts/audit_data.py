"""Audit images, VOC labels and MOT sequences without modifying source data."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.preparation import ROOT, RAW_ROOTS, inventory, new_output, relative, write_json


def image_audit():
    from PIL import Image, ImageOps
    root = RAW_ROOTS[0]
    images = {p.stem: p for p in (root / "data").iterdir() if p.suffix.lower() == ".jpg"}
    labels = {p.stem: p for p in (root / "label").iterdir() if p.suffix.lower() == ".xml"}
    records, issues, hashes = [], [], []
    for stem, path in sorted(images.items()):
        item = dict(id=stem, image=relative(path), xml=relative(labels[stem]) if stem in labels else None)
        errors = []
        try:
            with Image.open(path) as im:
                im.load()  # Decode every image, not just the header.
                width, height = im.size
                small = ImageOps.grayscale(im).resize((9, 8))
                values = list(small.get_flattened_data() if hasattr(small, "get_flattened_data") else small.getdata())
                bits = [values[y * 9 + x] > values[y * 9 + x + 1] for y in range(8) for x in range(8)]
                dhash = sum(int(v) << i for i, v in enumerate(bits))
                hashes.append((stem, dhash))
            item.update(width=width, height=height)
            objects = []
            if stem not in labels:
                errors.append("missing_xml")
            else:
                xml = ET.parse(labels[stem]).getroot()
                item["xml_filename"] = xml.findtext("filename")
                item["xml_original_path"] = xml.findtext("path")
                if Path(xml.findtext("filename", "")).name != path.name:
                    errors.append("xml_filename_mismatch")
                if (int(xml.findtext("size/width", "0")), int(xml.findtext("size/height", "0"))) != (width, height):
                    errors.append("xml_size_mismatch")
                for index, obj in enumerate(xml.findall("object")):
                    box = [float(obj.findtext(f"bndbox/{key}", "nan")) for key in ("xmin", "ymin", "xmax", "ymax")]
                    name = obj.findtext("name", "").strip()
                    difficult = int(obj.findtext("difficult", "0"))
                    truncated = int(obj.findtext("truncated", "0"))
                    if name != "pig":
                        errors.append(f"object_{index}:unexpected_class:{name}")
                    if difficult not in (0, 1) or truncated not in (0, 1):
                        errors.append(f"object_{index}:invalid_flags")
                    if not all(math.isfinite(v) for v in box):
                        errors.append(f"object_{index}:nonfinite_box")
                        continue
                    x1, y1, x2, y2 = box
                    # Check broad bounds here; conversion requires an explicit origin convention.
                    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                        errors.append(f"object_{index}:invalid_or_outside_box")
                    objects.append(dict(category=name, bbox_xyxy=box, difficult=difficult, truncated=truncated))
                item["objects"] = objects
        except Exception as exc:
            errors.append(f"decode_or_parse:{type(exc).__name__}:{exc}")
        item["issues"] = errors
        records.append(item)
        issues.extend(dict(image=item["image"], issue=error) for error in errors)
    candidates = []
    candidate_count = 0
    for i, (a, ha) in enumerate(hashes):
        for b, hb in hashes[i + 1:]:
            distance = (ha ^ hb).bit_count()
            if distance <= 4:
                candidate_count += 1
                if len(candidates) < 200:
                    candidates.append(dict(a=a, b=b, dhash_distance=distance))
    boxes = [o["bbox_xyxy"] for r in records for o in r.get("objects", [])]
    return records, dict(images=len(images), xmls=len(labels), paired=len(images.keys() & labels.keys()),
        orphan_xmls=sorted(labels.keys() - images.keys()), issues=issues,
        object_count=len(boxes), classes=dict(Counter(o["category"] for r in records for o in r.get("objects", []))),
        difficult_objects=sum(o["difficult"] for r in records for o in r.get("objects", [])),
        boxes_with_zero_minimum=sum(b[0] == 0 or b[1] == 0 for b in boxes),
        image_sizes=dict(Counter(f'{r.get("width")}x{r.get("height")}' for r in records)),
        near_duplicate_candidate_pairs=candidate_count, near_duplicate_examples=candidates,
        near_duplicate_note="dHash <= 4 is a screening heuristic, not proof of duplication or common video source.")


def mot_audit(full_decode):
    import cv2
    root = RAW_ROOTS[1]
    records = []
    for seq in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        video = root / f"MyVideo_{seq.name}.mp4"
        gt = seq / "gt" / "gt.txt"
        issues = Counter()
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        decoded = 0
        if full_decode:
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                decoded += 1
            if decoded != frames:
                issues["decoded_frame_count_mismatch"] += 1
        capture.release()
        seen, ids, frame_ids = set(), set(), set()
        bad_examples, bad_ids = [], Counter()
        row_count = 0
        for line in gt.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            row_count += 1
            try:
                row = [float(v.strip()) for v in line.split(",")]
                if len(row) != 9 or not all(math.isfinite(v) for v in row):
                    raise ValueError("MOT GT needs nine finite numbers")
                frame, identity, x, y, w, h, mark, category, visibility = row
                if frame != int(frame) or identity != int(identity) or frame < 1 or identity < 1:
                    issues["invalid_frame_or_id"] += 1
                if frame > frames:
                    issues["frame_beyond_video"] += 1
                if w <= 0 or h <= 0:
                    issues["nonpositive_box"] += 1
                    bad_ids[f"nonpositive_box:id_{int(identity)}"] += 1
                    if len(bad_examples) < 12:
                        bad_examples.append(dict(issue="nonpositive_box", row=row))
                if x < 0 or y < 0 or x + w > width + 1e-6 or y + h > height + 1e-6:
                    issues["box_outside_image"] += 1
                    bad_ids[f"box_outside_image:id_{int(identity)}"] += 1
                    if len(bad_examples) < 12:
                        bad_examples.append(dict(issue="box_outside_image", row=row))
                if mark not in (0, 1) or category != 1 or not 0 <= visibility <= 1:
                    issues["unexpected_flags_class_visibility"] += 1
                key = (int(frame), int(identity))
                if key in seen:
                    issues["duplicate_frame_id"] += 1
                seen.add(key)
                ids.add(int(identity))
                frame_ids.add(int(frame))
            except ValueError:
                issues["invalid_row"] += 1
        records.append(dict(sequence=seq.name, video=relative(video), gt=relative(gt),
            labels=(seq / "gt" / "labels.txt").read_text(encoding="utf-8-sig").splitlines(),
            frames_metadata=frames, fps=fps, width=width, height=height,
            full_decode_checked=full_decode, frames_decoded=decoded if full_decode else None,
            gt_rows=row_count, identity_count=len(ids), annotated_frames=len(frame_ids),
            first_gt_frame=min(frame_ids) if frame_ids else None,
            last_gt_frame=max(frame_ids) if frame_ids else None, issues=dict(issues),
            issue_counts_by_id=dict(bad_ids), issue_examples=bad_examples))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New experiments/preparation/<run> directory")
    parser.add_argument("--full-video-decode", action="store_true", help="Decode every frame; slower")
    args = parser.parse_args()
    before = inventory()
    images, detection = image_audit()
    tracking = mot_audit(args.full_video_decode)
    inventory_map = {r["path"]: r for r in before}
    for item in images:
        item["image_sha256"] = inventory_map[item["image"]]["sha256"]
        if item["xml"]:
            item["xml_sha256"] = inventory_map[item["xml"]]["sha256"]
    exact_images = {}
    for record in before:
        if record["path"].startswith("pig-objection detection/data/"):
            exact_images.setdefault(record["sha256"], []).append(record["path"])
    detection["exact_duplicate_image_groups"] = [paths for paths in exact_images.values() if len(paths) > 1]
    nested = RAW_ROOTS[1] / "PIG_MOT_TEST_DATA"
    duplicates = []
    for path in sorted(nested.rglob("*")):
        if path.is_file():
            outer = RAW_ROOTS[1] / path.relative_to(nested)
            a, b = inventory_map[relative(path)], inventory_map.get(relative(outer))
            duplicates.append(dict(inner=a["path"], outer=relative(outer), identical=b is not None and a["sha256"] == b["sha256"]))
    after = inventory()
    if before != after:
        raise RuntimeError("Raw data changed during audit; no report published")
    output = new_output(args.output)
    write_json(output / "raw_inventory.json", before)
    write_json(output / "detection_records.json", images)
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), raw_unchanged=True,
        raw_files=len(before), raw_bytes=sum(r["bytes"] for r in before), detection=detection,
        tracking=tracking, nested_duplicate_pairs=duplicates,
        unresolved=["VOC coordinate convention", "Image source groups / temporal adjacency",
                    "Overlap between detection images and MOT videos", "Held-out tracking tuning sequences",
                    "Server GPU / CUDA / framework versions", "Pig-domain ReID weights and protocol"],
        environment=dict(python=sys.version))
    write_json(output / "summary.json", summary)
    print(json.dumps(dict(output=relative(output), raw_files=len(before), detection_images=len(images),
        detection_issues=len(detection["issues"]), sequences=len(tracking),
        tracking_issues={r["sequence"]: r["issues"] for r in tracking}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
