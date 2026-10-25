"""Independently verify repaired GT, buffered split coverage and raw-file integrity."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.preparation import ROOT, inventory, new_output, sha256, write_json
from tools.mot_geometry import parse_gt, validate_gt


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from PIL import Image, ImageOps
    import cv2
    output = new_output(args.output)
    tests = subprocess.run([sys.executable,"-B","-m","unittest","discover","-s","tests","-v"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",errors="replace")
    (output/"tests.log").write_text(tests.stdout+tests.stderr,encoding="utf-8")
    if tests.returncode:
        raise RuntimeError("Unit tests failed")
    baseline = read_json(ROOT/"experiments/preparation/audit_20261022_03/raw_inventory.json")
    if baseline != inventory():
        raise RuntimeError("Original files differ from audit baseline")
    gt_manifest = read_json(ROOT/"datasets/annotations/mot_clean_v1/manifest.json")
    edits = read_json(ROOT/"datasets/annotations/mot_clean_v1/changes.json")
    edit_map = {(r["sequence"],r["frame"],r["identity"]):r for r in edits}
    assert len(edit_map) == 1924
    gt_results = []
    for record in gt_manifest["sequences"]:
        name = record["sequence"]
        original_path, derived_path = ROOT/record["source_gt"], ROOT/record["gt"]
        assert sha256(original_path)==record["source_gt_sha256"]
        assert sha256(derived_path)==record["gt_sha256"]
        original = {tuple(row[:2]):row for row in map(parse_gt,original_path.read_text().splitlines())}
        checked = validate_gt(derived_path.read_text().splitlines(),record["width"],record["height"],record["frames"])
        derived = {tuple(row[:2]):row for row in checked}
        missing = set(original)-set(derived)
        assert not set(derived)-set(original)
        if name == "193":
            assert missing == {(float(frame),6.0) for frame in range(1,1905)}
        else:
            assert not missing
        changed = 0
        for key,row in derived.items():
            before = original[key]
            if any(not math.isclose(a,b,rel_tol=0,abs_tol=1e-9) for a,b in zip(row,before)):
                changed += 1
                assert name=="197" and int(row[1])==7
                x,y,w,h = before[2:6]
                expected = [max(0,x),max(0,y),min(record["width"],x+w)-max(0,x),min(record["height"],y+h)-max(0,y)]
                assert all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(row[2:6],expected))
                assert row[:2]==before[:2] and row[6:]==before[6:]
                assert (name,int(row[0]),int(row[1])) in edit_map
        assert changed == (20 if name=="197" else 0)
        exported = ROOT/"datasets/processed/mot/clean_v1"/name/"gt/gt.txt"
        assert sha256(exported)==sha256(derived_path)
        gt_results.append(dict(sequence=name,rows=len(derived),excluded=len(missing),clipped=changed,geometry_valid=True))
        if name in ("193","197"):
            capture=cv2.VideoCapture(str(ROOT/record["video"]))
            ok,frame=capture.read()
            capture.release()
            assert ok
            for row in checked:
                if row[0]!=1:
                    continue
                x,y,w,h = row[2:6]
                cv2.rectangle(frame,(round(x),round(y)),(min(record["width"]-1,round(x+w)),min(record["height"]-1,round(y+h))),(0,220,0),2)
                cv2.putText(frame,f'ID {int(row[1])}',(round(x),max(25,round(y)-8)),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,220,0),2)
            assert cv2.imwrite(str(output/f"{name}_clean_frame1.jpg"),frame)
    detection = read_json(ROOT/"datasets/splits/protocol_v2/detection/splits.json")
    records = read_json(ROOT/"experiments/preparation/audit_20261022_03/detection_records.json")
    lookup = {r["id"]:r for r in records}
    lists = {**detection["splits"],"reserve":detection["reserved_ids"]}
    coverage = [key for keys in lists.values() for key in keys]
    assert len(coverage)==len(set(coverage))==len(lookup) and set(coverage)==set(lookup)
    hashes = {}
    for name,keys in lists.items():
        expected = [lookup[key]["image"] for key in keys]
        assert (ROOT/"datasets/splits/protocol_v2/detection"/f"{name}.txt").read_text(encoding="utf-8").splitlines()==expected
        for key in keys:
            record = lookup[key]
            assert (ROOT/record["xml"]).is_file()
            if name=="reserve":
                continue
            with Image.open(ROOT/record["image"]) as im:
                small=ImageOps.grayscale(im).resize((9,8))
                pixels=list(small.get_flattened_data() if hasattr(small,"get_flattened_data") else small.getdata())
            hashes[key]=sum(int(pixels[y*9+x]>pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
    conflicts=[]
    for left,right in (("train","val"),("train","test"),("val","test")):
        for a in lists[left]:
            for b in lists[right]:
                if (hashes[a]^hashes[b]).bit_count() <= 4:
                    conflicts.append((a,b))
    assert not conflicts
    track = read_json(ROOT/"datasets/splits/protocol_v2/tracking/splits.json")
    assert set(track["tuning_sequences"]).isdisjoint(track["test_sequences"])
    assert set(track["tuning_sequences"]+track["test_sequences"])=={r["sequence"] for r in gt_results}
    assert not track["blocked_test_sequences"]
    config=read_json(ROOT/"configs/data.json")
    assert config["detection"]["split_manifest"]=="datasets/splits/protocol_v2/detection/splits.json"
    assert config["tracking"]["gt_manifest"]=="datasets/annotations/mot_clean_v1/manifest.json"
    # All validation writes belong to the new output; recheck sources at the end.
    assert baseline==inventory()
    result=dict(raw_files_unchanged=len(baseline),gt=gt_results,
        detection_counts={k:len(v) for k,v in lists.items()},all_images_accounted_for=True,
        cross_subset_dhash_candidates=0, unit_tests_passed=True,
        source_independence_verified=False,evaluation_scope="within_scene_derived_gt_protocol",
        provenance_caveat="Historical source/animal/time independence cannot be certified from existing files")
    write_json(output/"results.json",result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
