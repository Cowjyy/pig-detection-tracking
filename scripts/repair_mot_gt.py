"""Publish derived MOT GT with auditable exclusion/clipping, preserving every source."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.mot_geometry import clip_small_overflow, parse_gt, validate_gt
from tools.preparation import ROOT, new_output, relative, sha256, source_path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="experiments/preparation/audit_20261022_03")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = ROOT/args.audit
    summary = json.loads((audit/"summary.json").read_text(encoding="utf-8"))
    baseline = {r["path"]: r["sha256"] for r in json.loads((audit/"raw_inventory.json").read_text(encoding="utf-8"))}
    prepared, changes, checks = {}, [], []
    for sequence in summary["tracking"]:
        name, width, height, frames = sequence["sequence"], sequence["width"], sequence["height"], sequence["frames_metadata"]
        source = source_path(sequence["gt"])
        video = source_path(sequence["video"])
        if sha256(source) != baseline[sequence["gt"]] or sha256(video) != baseline[sequence["video"]]:
            raise ValueError("Source changed since audit")
        lines = source.read_text(encoding="utf-8-sig").splitlines()
        parsed = [parse_gt(line) for line in lines]
        tracks = defaultdict(list)
        for row in parsed:
            tracks[int(row[1])].append(row)
        excluded = set()
        for identity, rows in tracks.items():
            bad = [r for r in rows if r[4] <= 0 or r[5] <= 0]
            if not bad:
                continue
            # A stable zero-area placeholder throughout the entire video is not
            # an interpolatable missing box. Partial defects deliberately fail.
            if (name != "193" or identity != 6 or len(bad) != len(rows) or len(rows) != frames
                    or {int(r[0]) for r in rows} != set(range(1, frames+1))
                    or len({tuple(r[2:6]) for r in rows}) != 1
                    or rows[0][4] != 0):
                raise ValueError(f"Unreviewed degenerate trajectory: {name}/{identity}")
            excluded.add(identity)
        output_lines = []
        for line_no, (line, row) in enumerate(zip(lines, parsed), 1):
            if int(row[1]) in excluded:
                changes.append(dict(sequence=name, source_line=line_no, frame=int(row[0]), identity=int(row[1]),
                    action="exclude_constant_zero_width_placeholder", before=line, after=None,
                    reason="zero-width at the exact same position throughout all 1904 frames; not a recoverable observed trajectory"))
                continue
            updated, changed = clip_small_overflow(row, width, height)
            new_line = ",".join(format(v, ".15g") for v in updated) if changed else line
            if changed:
                changes.append(dict(sequence=name, source_line=line_no, frame=int(row[0]), identity=int(row[1]),
                    action="clip_small_overflow_to_image", before=line, after=new_line,
                    reason="at most 5 pixels overflow and at least 90% of original area retained"))
            output_lines.append(new_line)
        validated = validate_gt(output_lines, width, height, frames)
        prepared[name] = "\n".join(output_lines)+"\n"
        checks.append(dict(sequence=name, source_gt=sequence["gt"], source_gt_sha256=baseline[sequence["gt"]],
            video=sequence["video"], video_sha256=baseline[sequence["video"]],
            width=width, height=height, frames=frames, fps=sequence["fps"],
            input_rows=len(parsed), output_rows=len(validated), excluded_identities=sorted(excluded),
            output_identity_count=len({int(r[1]) for r in validated}), geometry_valid=True))
    expected = Counter({("193", "exclude_constant_zero_width_placeholder"): 1904,
                        ("197", "clip_small_overflow_to_image"): 20})
    observed = Counter((c["sequence"],c["action"]) for c in changes)
    if observed != expected:
        raise ValueError(f"Unexpected repair scope: {observed}; stop rather than silently expanding changes")
    output = new_output(args.output)
    for item in checks:
        path = output/item["sequence"]/"gt"/"gt.txt"
        path.parent.mkdir(parents=True)
        path.write_text(prepared[item["sequence"]], encoding="utf-8")
        item["gt"] = relative(path)
        item["gt_sha256"] = sha256(path)
        if sha256(source_path(item["source_gt"])) != item["source_gt_sha256"]:
            raise ValueError("Source GT changed during processing")
    write_json(output/"changes.json", changes)
    write_json(output/"manifest.json", dict(version="mot_clean_v1", status="geometry_valid_derived_gt",
        policy=dict(exclude="193/ID6 constant zero-width placeholder only", clip_max_pixels=5.0,
                    minimum_retained_area=0.90, renumber_ids=False, infer_or_interpolate_boxes=False),
        changes=dict(excluded_rows=1904, clipped_rows=20), sequences=checks,
        scope="Derived-label benchmark, not the unmodified original benchmark; disclose the policy in every report.",
        limitations="Geometry and reviewed placeholder exclusion only; does not certify every identity annotation or independent image provenance.",
        audit_sha256=sha256(audit/"summary.json"), generator_sha256=sha256(Path(__file__))))
    print(json.dumps(dict(output=relative(output), excluded=1904, clipped=20,
        output_rows=sum(c["output_rows"] for c in checks), all_geometry_valid=True), indent=2))


if __name__ == "__main__":
    main()
