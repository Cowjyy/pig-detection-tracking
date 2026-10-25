"""Create source-review contact sheets, not experimental predictions."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.preparation import ROOT, RAW_ROOTS, new_output, source_path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from PIL import Image, ImageDraw, ImageOps
    import cv2
    records = json.loads((ROOT / args.records).read_text(encoding="utf-8"))
    output = new_output(args.output)
    tile_w, tile_h, label_h = 320, 180, 25
    canvas = Image.new("RGB", (tile_w * 4, (tile_h + label_h) * 3), "white")
    draw = ImageDraw.Draw(canvas)
    selected = [records[round(i * (len(records)-1) / 11)] for i in range(12)]
    for index, record in enumerate(selected):
        with Image.open(source_path(record["image"])) as im:
            tile = ImageOps.fit(im.convert("RGB"), (tile_w, tile_h))
        x, y = index % 4 * tile_w, index // 4 * (tile_h + label_h)
        canvas.paste(tile, (x, y))
        draw.text((x+5, y+tile_h+4), f'Image {record["id"]} (source sample)', fill="black")
    canvas.save(output / "detection_samples.jpg", quality=92)
    sequences = sorted(p for p in RAW_ROOTS[1].iterdir() if p.is_dir() and p.name.isdigit())
    canvas = Image.new("RGB", (tile_w * 3, (tile_h + label_h) * len(sequences)), "white")
    draw = ImageDraw.Draw(canvas)
    sampled = []
    for row, seq in enumerate(sequences):
        capture = cv2.VideoCapture(str(RAW_ROOTS[1] / f"MyVideo_{seq.name}.mp4"))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for col, frame_index in enumerate((0, count // 2, count-1)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                capture.release()
                raise RuntimeError(f"Cannot preview {seq.name} frame {frame_index+1}")
            tile = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((tile_w, tile_h))
            x, y = col * tile_w, row * (tile_h + label_h)
            canvas.paste(tile, (x, y))
            draw.text((x+5, y+tile_h+4), f'Seq {seq.name} / frame {frame_index+1}', fill="black")
            sampled.append(dict(sequence=seq.name, frame=frame_index+1))
        capture.release()
    canvas.save(output / "tracking_samples.jpg", quality=92)
    write_json(output / "sampling.json", dict(detection_image_ids=[r["id"] for r in selected],
        tracking_frames=sampled, note="Sparse visual review only. These samples cannot prove absence of cross-dataset overlap."))
    print(f"Created review sheets: {output}")


if __name__ == "__main__":
    main()
