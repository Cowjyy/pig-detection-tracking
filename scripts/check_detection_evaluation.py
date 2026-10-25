"""Exercise COCO evaluation on two validation images using a smoke checkpoint.

The output is a software self-check and must never enter comparison tables.
"""
import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.runtime import ROOT, load_json, path, run_directory, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    training = load_json(str(path(args.run) / 'run.json'))
    if training['status'] != 'complete' or not training['smoke']:
        raise ValueError('Self-check requires a completed smoke checkpoint')
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    register_all_modules()
    cfg = Config.fromfile(str(path(args.run) / 'config.py'))
    annotations = load_json(str(path(load_json('configs/detection/experiments.json')['annotation_directory']) / 'val.coco.json'))
    annotations['images'] = annotations['images'][:2]
    image_ids = {image['id'] for image in annotations['images']}
    annotations['annotations'] = [ann for ann in annotations['annotations'] if ann['image_id'] in image_ids]
    checkpoints = list(path(args.run).glob('iter_*.pth'))
    checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    with run_directory(args.output, 'detection_evaluator_selfcheck',
                       dict(smoke=True, benchmark_result=False, images=len(image_ids))) as (out, state):
        save_json(out / 'fixture.coco.json', annotations)
        cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
        cfg.test_dataloader.dataset.ann_file = str(out / 'fixture.coco.json')
        cfg.test_dataloader.dataset.data_root = str(ROOT)
        cfg.test_dataloader.num_workers = 0
        cfg.test_dataloader.persistent_workers = False
        cfg.test_evaluator = dict(type='CocoMetric', ann_file=str(out / 'fixture.coco.json'), metric='bbox')
        cfg.work_dir = str(out)
        cfg.load_from = str(checkpoint)
        cfg.resume = False
        runner = Runner.from_cfg(cfg)
        metrics = runner.test()
        if 'coco/bbox_mAP' not in metrics:
            raise AssertionError('Expected COCO metric keys not returned')
        save_json(out / 'verification.json', dict(status='passed', metric_names=list(metrics),
            smoke_checkpoint=True, benchmark_result=False, score_values_intentionally_not_exported=True))
        print('PASS: detection inference and COCO evaluation interface')


if __name__ == '__main__':
    main()
