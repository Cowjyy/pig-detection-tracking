"""Audit coordinate evidence and freeze an explicit standard-VOC experiment policy.

This establishes a reproducible interpretation, not proof of the original tool.
Only derived annotations are written; raw images and XML are never changed.
"""
import argparse
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.convert_detection import convert_box
from tools.preparation import ROOT, load_checked_records, new_output, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', default='experiments/preparation/audit_20261022_03/detection_records.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    records = load_checked_records(ROOT / args.records)
    counts = dict(boxes=0, zero_minimum=0, xmin_one=0, ymin_one=0,
                  xmax_width=0, ymax_height=0, truncated_rule_mismatch=0)
    area_differences = []
    examples = []
    for record in records:
        for obj in record['objects']:
            x1, y1, x2, y2 = obj['bbox_xyxy']
            converted = convert_box(obj['bbox_xyxy'], record['width'], record['height'], 'voc_1based_inclusive')
            counts['boxes'] += 1
            counts['zero_minimum'] += x1 == 0 or y1 == 0
            counts['xmin_one'] += x1 == 1
            counts['ymin_one'] += y1 == 1
            counts['xmax_width'] += x2 == record['width']
            counts['ymax_height'] += y2 == record['height']
            expected_truncated = x1 == 1 or y1 == 1 or x2 == record['width'] or y2 == record['height']
            counts['truncated_rule_mismatch'] += bool(obj['truncated']) != expected_truncated
            area_differences.append(((x2-x1+1)*(y2-y1+1))/((x2-x1)*(y2-y1))-1)
            if len(examples) < 8 and expected_truncated:
                examples.append(dict(image=record['image'], xml_box=obj['bbox_xyxy'], coco_box=converted))
    out = new_output(args.output)
    write_json(out / 'review.json', dict(
        schema_version=1, images=len(records), counts=counts,
        records_sha256=sha256(ROOT / args.records),
        selected_convention='voc_1based_inclusive',
        status='explicit_standard_voc_protocol_supported_by_file_evidence',
        original_annotation_tool_verified=False,
        rationale=['VOC XML structure matches common LabelImg exports',
                   'All boxes satisfy the standard VOC coordinate domain',
                   'Boundary values and truncated flags are compared to the public LabelImg writer',
                   'This is a declared experimental convention; the historical tool cannot be proven from XML alone'],
        transform='COCO x=xmin-1, y=ymin-1, width=xmax-xmin+1, height=ymax-ymin+1',
        sensitivity=dict(alternative='zero_based_half_open', left_top_shift_pixels=1,
                         box_extent_difference_pixels=1,
                         median_relative_area_difference=statistics.median(area_differences),
                         max_relative_area_difference=max(area_differences)),
        examples=examples,
        references=['https://github.com/HumanSignal/labelImg/blob/master/libs/pascal_voc_io.py',
                    'https://github.com/HumanSignal/labelImg/blob/master/libs/labelFile.py'],
        raw_files_modified=False))
    print(f'Coordinate policy evidence written to {out}; tool history remains unverified')


if __name__ == '__main__':
    main()
