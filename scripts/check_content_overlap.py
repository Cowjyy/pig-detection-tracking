"""Search EVERY video frame for resized visual duplicates of detection images.

This establishes observed content overlap, not historical provenance or animal independence.
It does not run a detector, tracker or training job. Sources remain untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.preparation import ROOT, RAW_ROOTS, load_checked_records, new_output, relative, sha256, write_json


def ssim(a, b):
    import cv2
    import numpy as np
    a, b = a.astype(np.float32), b.astype(np.float32)
    ma, mb = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    va = cv2.GaussianBlur(a*a, (11, 11), 1.5)-ma*ma
    vb = cv2.GaussianBlur(b*b, (11, 11), 1.5)-mb*mb
    cov = cv2.GaussianBlur(a*b, (11, 11), 1.5)-ma*mb
    value = ((2*ma*mb+6.5025)*(2*cov+58.5225))/((ma*ma+mb*mb+6.5025)*(va+vb+58.5225))
    return float(value[5:-5, 5:-5].mean())


def comparison(a, b, objects, width, height):
    import numpy as np
    error = np.abs(a.astype(np.float32)-b.astype(np.float32))
    mask = np.zeros(a.shape, dtype=bool)
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        # Broad ROI bounds intentionally avoid assuming VOC's unconfirmed origin.
        left, right = max(0, int(x1*a.shape[1]/width)-1), min(a.shape[1], int(x2*a.shape[1]/width)+1)
        top, bottom = max(0, int(y1*a.shape[0]/height)-1), min(a.shape[0], int(y2*a.shape[0]/height)+1)
        mask[top:bottom, left:right] = True
    return dict(ssim=ssim(a, b), mae=float(error.mean()),
                object_roi_mae=float(error[mask].mean()) if mask.any() else float(error.mean()))


def qualifies(metrics):
    return metrics["ssim"] >= 0.98 and metrics["mae"] <= 3.0 and metrics["object_roi_mae"] <= 4.0


def descriptor(bgr):
    import cv2
    return cv2.resize(bgr, (32, 18), interpolation=cv2.INTER_AREA).reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="experiments/preparation/audit_20261022_03/detection_records.json")
    parser.add_argument("--splits", default="datasets/splits/initial_v1/detection/splits.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    records = sorted(load_checked_records(ROOT/args.records), key=lambda r: int(r["id"]))
    output = new_output(args.output)
    image_descriptors, image_gray = [], []
    image_paths = []
    for record in records:
        path = ROOT/record["image"]
        bgr = cv2.imread(str(path))
        image_descriptors.append(descriptor(bgr))
        image_gray.append(cv2.cvtColor(cv2.resize(bgr, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
        image_paths.append(record["image"])
    descriptors, gray, frames = [], [], []
    video_hashes = {}
    for path in sorted(RAW_ROOTS[1].glob("MyVideo_*.mp4")):
        sequence = path.stem.split("_")[-1]
        video_hashes[relative(path)] = sha256(path)
        capture = cv2.VideoCapture(str(path))
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            count += 1
            descriptors.append(descriptor(bgr))
            gray.append(cv2.cvtColor(cv2.resize(bgr, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
            frames.append((sequence, count))
        capture.release()
        if count != expected:
            raise ValueError(f"Incomplete decoding: {sequence}: {count}/{expected}")
        print(f"Indexed sequence {sequence}: {count} frames", flush=True)
    features = np.asarray(descriptors, dtype=np.float32)/255
    image_features = np.asarray(image_descriptors, dtype=np.float32)/255
    video_gray = np.asarray(gray)
    image_gray = np.asarray(image_gray)
    del descriptors, gray
    np.savez_compressed(output/"thumbnail_cache.npz", video_gray=video_gray, image_gray=image_gray)
    fnorm = (features*features).sum(axis=1)
    top_candidates, matches = [], []
    # Exhaustive low-resolution frame retrieval followed by structural/pig-ROI comparison.
    for begin in range(0, len(records), 64):
        batch = image_features[begin:begin+64]
        distances = (batch*batch).sum(axis=1)[:, None] + fnorm[None, :] - 2*(batch @ features.T)
        indices = np.argpartition(distances, 4, axis=1)[:, :5]
        for local, candidates in enumerate(indices):
            index = begin+local
            record = records[index]
            results = []
            for video_index in candidates:
                metrics = comparison(image_gray[index], video_gray[video_index], record["objects"], record["width"], record["height"])
                sequence, frame = frames[int(video_index)]
                results.append(dict(sequence=sequence, frame=frame, video_index=int(video_index), **metrics))
            results.sort(key=lambda r: (-r["ssim"], r["object_roi_mae"]))
            best = results[0]
            accepted = [r for r in results if qualifies(r)]
            top_candidates.append(dict(image_id=record["id"], image=record["image"], image_index=index,
                status="observed_resized_frame_duplicate" if accepted else "no_direct_duplicate_observed",
                best=best, candidates=results))
            if accepted:
                matches.append(dict(image_id=record["id"], matches=accepted))
        print(f"Compared {min(begin+64, len(records))}/{len(records)} images", flush=True)
    # Inspect all cross-subset pairs structurally, rather than treating dHash as proof.
    initial = json.loads((ROOT/args.splits).read_text(encoding="utf-8"))
    membership = {image_id: name for name, ids in initial["splits"].items() for image_id in ids}
    cross_pairs = []
    image_norm = (image_features*image_features).sum(axis=1)
    image_dist = image_norm[:,None]+image_norm[None,:]-2*(image_features @ image_features.T)
    for i, record in enumerate(records):
        # Includes nearest 20 images plus every original dHash cross-split candidate.
        neighbors = set(int(j) for j in np.argsort(image_dist[i])[:21] if i != j)
        for item in initial["similarity_review"]["examples"]:
            if item["a"] == record["id"]:
                neighbors.add(next(j for j,r in enumerate(records) if r["id"] == item["b"]))
            elif item["b"] == record["id"]:
                neighbors.add(next(j for j,r in enumerate(records) if r["id"] == item["a"]))
        for j in sorted(neighbors):
            if j <= i or membership[record["id"]] == membership[records[j]["id"]]:
                continue
            metrics = comparison(image_gray[i], image_gray[j], record["objects"], record["width"], record["height"])
            if qualifies(metrics):
                cross_pairs.append(dict(a=record["id"], b=records[j]["id"], **metrics))
    # Positive controls follow the real transformation: full-resolution video ->
    # 1280x720 JPEG -> thumbnail. Compressing the 160x90 cache itself destroys much
    # more texture and is not representative of the supplied 1280x720 JPG files.
    controls = []
    for sequence in sorted(set(s for s,_ in frames)):
        index = next(i for i,(seq,frame) in enumerate(frames) if seq == sequence and frame == 1)
        original = video_gray[index]
        capture = cv2.VideoCapture(str(RAW_ROOTS[1]/f"MyVideo_{sequence}.mp4"))
        ok, full_frame = capture.read()
        capture.release()
        if not ok:
            raise ValueError("Cannot read positive-control source frame")
        resized = cv2.resize(full_frame, (1280, 720), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        reconstructed = cv2.cvtColor(cv2.resize(decoded, (160,90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        metrics = comparison(reconstructed, original, [], 160, 90)
        query = descriptor(decoded).astype(np.float32)/255
        control_distance = (query*query).sum()+fnorm-2*(features @ query)
        candidates = np.argpartition(control_distance, 4)[:5]
        retrieval_ok = any(frames[int(j)][0] == sequence and frames[int(j)][1] <= 30 for j in candidates)
        controls.append(dict(sequence=sequence, accepted=qualifies(metrics) and retrieval_ok,
            retrieval_within_first_second=retrieval_ok, **metrics))
    write_json(output/"control_checks.json", controls)
    if not all(c["accepted"] for c in controls):
        raise ValueError("JPEG positive controls failed; check thresholds")
    write_json(output/"retrieval.json", dict(images=top_candidates, video_frames=frames,
        image_order=[r["id"] for r in records], video_hashes=video_hashes))
    result = dict(status="completed_content_search_not_provenance_certification", images=len(records),
        video_frames_indexed=len(frames), sequences=len(video_hashes), retrieved_candidates_per_image=5,
        observed_duplicate_images=len(matches), observed_duplicate_matches=matches,
        cross_subset_duplicate_pairs=cross_pairs, jpeg_positive_controls=controls,
        thresholds=dict(ssim_min=0.98, mae_max=3.0, object_roi_mae_max=4.0),
        records_sha256=sha256(ROOT/args.records), initial_splits_sha256=sha256(ROOT/args.splits),
        limitations=["No-direct-match is not proof of different recording, animal, pen or time",
            "Retrieval may miss crops, geometric warps, heavy edits or strong lighting changes",
            "Sparse 32x18 retrieval and 160x90 comparisons have finite sensitivity",
            "Shared pen/animal or adjacent-time dependence requires source metadata or separate collection"],
        generator_sha256=sha256(Path(__file__)))
    write_json(output/"summary.json", result)
    print(json.dumps({k:result[k] for k in ("images", "video_frames_indexed", "observed_duplicate_images", "cross_subset_duplicate_pairs")}, indent=2))


if __name__ == "__main__":
    main()
