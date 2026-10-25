"""Independently compare converted COCO/YOLO labels with source XML and splits."""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.preparation import ROOT, new_output, sha256, write_json


def read(value):
    return json.loads((ROOT / value).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = read('configs/data.json')['detection']
    experiment = read('configs/detection/experiments.json')
    folder = ROOT / experiment['annotation_directory']
    conversion = read(folder / 'conversion.json')
    split = read(data['split_manifest'])
    if conversion['splits_sha256'] != sha256(ROOT / data['split_manifest']):
        raise ValueError('Conversion split hash is stale')
    if conversion['coordinate_convention'] != data['coordinate_convention']:
        raise ValueError('Coordinate convention mismatch')
    if data['coordinate_convention'] != 'voc_1based_inclusive':
        raise ValueError('This independent verifier expects the frozen standard VOC protocol')
    if sha256(ROOT / conversion['coordinate_review']) != conversion['coordinate_review_sha256']:
        raise ValueError('Coordinate review hash mismatch')
    records = {r['id']: r for r in read('experiments/preparation/audit_20261022_03/detection_records.json')}
    result, previews = {}, []
    for subset in ('train', 'val', 'test'):
        coco = read(folder / f'{subset}.coco.json')
        assert coco['categories'] == [dict(id=1, name='pig')]
        images = {im['id']: im for im in coco['images']}
        assert len(images) == len(split['splits'][subset])
        assert len({a['id'] for a in coco['annotations']}) == len(coco['annotations'])
        by_image = {image_id: [] for image_id in images}
        for ann in coco['annotations']:
            by_image[ann['image_id']].append(ann)
        for key, image in zip(split['splits'][subset], coco['images']):
            record = records[key]
            assert image['file_name'] == record['image']
            assert (image['width'], image['height']) == (record['width'], record['height'])
            assert sha256(ROOT / record['xml']) == record['xml_sha256']
            objects = ET.parse(ROOT / record['xml']).getroot().findall('object')
            labels = (folder / 'yolo' / subset / f'{key}.txt').read_text().splitlines()
            assert len(objects) == len(by_image[image['id']]) == len(labels)
            for obj, ann, line in zip(objects, by_image[image['id']], labels):
                box = obj.find('bndbox')
                x1, y1, x2, y2 = [float(box.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
                x, y, w, h = ann['bbox']
                assert [x+1, y+1, x+w, y+h] == [x1, y1, x2, y2]
                assert 0 <= x < x+w <= image['width'] and 0 <= y < y+h <= image['height']
                assert ann['category_id'] == 1 and ann['iscrowd'] == 0 and ann['area'] == w*h
                label = list(map(float, line.split()))
                expected = [0, (x+w/2)/image['width'], (y+h/2)/image['height'], w/image['width'], h/image['height']]
                assert all(abs(a-b) < 1e-8 for a, b in zip(label, expected))
            if len(previews) < (('train', 'val', 'test').index(subset)+1)*2:
                previews.append((subset, image, by_image[image['id']]))
        result[subset] = dict(images=len(images), boxes=len(coco['annotations']))
    out = new_output(args.output)
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (1280, 3*390), 'white')
    for index, (subset, image, boxes) in enumerate(previews):
        with Image.open(ROOT / image['file_name']) as source:
            panel = source.convert('RGB')
        draw = ImageDraw.Draw(panel)
        for ann in boxes:
            x, y, w, h = ann['bbox']
            draw.rectangle((x, y, x+w-1, y+h-1), outline=(0, 255, 0), width=2)
        panel = panel.resize((640, 360))
        x, y = (index % 2)*640, (index // 2)*390
        canvas.paste(panel, (x, y+30))
        ImageDraw.Draw(canvas).text((x+8, y+8), f'{subset}: {Path(image["file_name"]).name}', fill='black')
    canvas.save(out / 'conversion_preview.jpg', quality=90)
    write_json(out / 'verification.json', dict(status='passed', subsets=result,
        convention=data['coordinate_convention'], split_sha256=conversion['splits_sha256'],
        xml_inverse_mapping_verified=True, yolo_consistency_verified=True,
        reserved_images=len(split['reserved_ids']), original_data_modified=False))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
