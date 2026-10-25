"""Export the seven canonical MOT sequences; optional frame extraction, no tracking."""
from __future__ import annotations

import argparse
import configparser
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.preparation import ROOT, RAW_ROOTS, new_output, relative, sha256, source_path, write_json
from tools.mot_geometry import validate_gt
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New datasets/processed/mot/<version> directory")
    parser.add_argument("--extract-frames", action="store_true", help="Expensive; estimate storage first")
    parser.add_argument("--gt-manifest", help="Manifest from repair_mot_gt.py; uses verified derived GT instead of source GT")
    args = parser.parse_args()
    import cv2
    derived = {}
    if args.gt_manifest:
        manifest_path = (ROOT/args.gt_manifest).resolve()
        if not manifest_path.is_relative_to((ROOT/"datasets"/"annotations").resolve()):
            raise ValueError("Derived GT manifest must be inside datasets/annotations")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "geometry_valid_derived_gt":
            raise ValueError("Derived GT manifest is not validated")
        derived = {r["sequence"]: r for r in manifest["sequences"]}
        if len(derived) != 7:
            raise ValueError("Expected seven derived sequences")
    else:
        from audit_data import mot_audit
        checked = mot_audit(False)
        failures = {row["sequence"]: row["issues"] for row in checked if row["issues"]}
        if failures:
            raise ValueError(f"Unresolved raw MOT GT issues: {failures}. Use a validated derived --gt-manifest; source GT is never repaired here.")
    sources = []
    for seq in sorted(p for p in RAW_ROOTS[1].iterdir() if p.is_dir() and p.name.isdigit()):
        video = RAW_ROOTS[1] / f"MyVideo_{seq.name}.mp4"
        gt = seq / "gt" / "gt.txt"
        if derived:
            item = derived[seq.name]
            gt = (ROOT/item["gt"]).resolve()
            if not gt.is_relative_to((ROOT/"datasets"/"annotations").resolve()):
                raise ValueError("Derived GT must belong to datasets/annotations")
            if sha256(gt) != item["gt_sha256"] or sha256(source_path(item["source_gt"])) != item["source_gt_sha256"]:
                raise ValueError("Derived or original GT changed after repair")
            if sha256(video) != item["video_sha256"]:
                raise ValueError("Source video changed after repair")
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {video}")
        fps, frames, width, height = (capture.get(key) for key in
            (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        validate_gt(gt.read_text(encoding="utf-8-sig").splitlines(), int(width), int(height), int(frames))
        sources.append(dict(sequence=seq.name, video=relative(video), gt=relative(gt),
            video_sha256=sha256(video), gt_sha256=sha256(gt), fps=fps,
            frames=int(frames), width=int(width), height=int(height)))
    output = new_output(args.output)
    # Incomplete outputs have no READY marker. They are never overwritten on rerun.
    for item in sources:
        seq_out = output / item["sequence"]
        (seq_out / "gt").mkdir(parents=True)
        source_gt = ROOT/item["gt"]
        shutil.copyfile(source_gt, seq_out / "gt" / "gt.txt")
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        cfg["Sequence"] = dict(name=item["sequence"], imDir="img1", frameRate=str(item["fps"]),
            seqLength=str(item["frames"]), imWidth=str(item["width"]), imHeight=str(item["height"]), imExt=".jpg")
        with (seq_out / "seqinfo.ini").open("x", encoding="utf-8") as stream:
            cfg.write(stream)
        if args.extract_frames:
            frame_dir = seq_out / "img1"
            frame_dir.mkdir()
            capture = cv2.VideoCapture(str(RAW_ROOTS[1] / f'MyVideo_{item["sequence"]}.mp4'))
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                index += 1
                if not cv2.imwrite(str(frame_dir / f"{index:06d}.jpg"), frame):
                    raise RuntimeError("Failed to write extracted frame")
            capture.release()
            if index != item["frames"]:
                raise RuntimeError(f"Frame count mismatch for {item['sequence']}")
        if sha256(source_gt) != item["gt_sha256"]:
            raise RuntimeError("Source GT changed during export")
    write_json(output / "sources.json", dict(sequences=sources, frames_extracted=args.extract_frames,
        gt_protocol="derived_clean_gt" if derived else "original_gt",
        gt_manifest=relative(ROOT/args.gt_manifest) if derived else None,
        gt_manifest_sha256=sha256(ROOT/args.gt_manifest) if derived else None,
        note="Metadata-only export is not ready for image-based tracking. Exported GT bytes match the selected GT input; class 1 means pig, not pedestrian. Derived-label scores must be identified as such."))
    (output / "READY").write_text("frames\n" if args.extract_frames else "metadata_only\n", encoding="utf-8")
    print(f"Exported {len(sources)} sequences to {output}")


if __name__ == "__main__":
    main()
