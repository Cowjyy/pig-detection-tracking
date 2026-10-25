"""Evaluate a completed detector; val drives selection, test is final only."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.runtime import ROOT,load_json,path,fingerprint,run_directory,save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,help='Completed train run with a best validation checkpoint')
    parser.add_argument('--split',choices=['val','test'],default='val')
    parser.add_argument('--output',required=True)
    parser.add_argument('--benchmark-images',type=int,default=20)
    parser.add_argument('--skip-benchmark',action='store_true',help='Evaluate AP now; measure speed separately when the GPU is exclusive')
    args=parser.parse_args()
    training=load_json(str(path(args.run)/'run.json'))
    if training['kind']!='detection_train' or training['status']!='complete' or training['smoke']:
        raise ValueError('Only completed NON-smoke training may produce comparison metrics')
    checkpoint=training['best_checkpoint']
    data=load_json('configs/data.json')['detection']
    if training['split']!=fingerprint(data['split_manifest']):
        raise ValueError('Training split differs from active protocol')
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    register_all_modules()
    cfg=Config.fromfile(str(path(args.run)/'config.py'))
    ann=path(load_json('configs/detection/experiments.json')['annotation_directory'])
    cfg.test_dataloader=cfg[f'{args.split}_dataloader']
    cfg.test_dataloader.dataset.data_root=str(ROOT)
    cfg.test_dataloader.dataset.ann_file=str(ann/f'{args.split}.coco.json')
    cfg.test_evaluator=dict(type='CocoMetric',ann_file=str(ann/f'{args.split}.coco.json'),metric='bbox')
    cfg.load_from=str(path(checkpoint))
    cfg.resume=False
    with run_directory(args.output,'detection_eval',dict(model=training['model'],seed=training['seed'],
        split=args.split,data_split=training['split'],training_run=args.run,checkpoint=fingerprint(checkpoint),smoke=False,
        evaluation_scope=data['evaluation_scope'])) as (output,state):
        cfg.work_dir=str(output)
        runner=Runner.from_cfg(cfg)
        metrics=runner.test()
        if args.skip_benchmark:
            save_json(output/'metrics.json',dict(metrics=metrics,metric_units='AP fractions (0..1)',
                parameter_count=sum(p.numel() for p in runner.model.parameters()),fps=None,
                benchmark_status='deferred_to_exclusive_gpu',precision='fp32'))
            return
        import cv2
        import torch
        from models.detectors.mmdet_adapter import Detector
        detector=Detector(str(path(args.run)/'config.py'),checkpoint)
        annotations=load_json(str(ann/f'{args.split}.coco.json'))
        images=[cv2.imread(str(path(item['file_name']))) for item in annotations['images'][:args.benchmark_images]]
        if not images or any(image is None for image in images):
            raise ValueError('No valid benchmark images')
        for i in range(5):
            detector.predict(images[i%len(images)])
        torch.cuda.synchronize()
        elapsed=0.0
        for image in images:
            torch.cuda.synchronize()
            start=time.perf_counter()
            detector.predict(image)
            torch.cuda.synchronize()
            elapsed+=time.perf_counter()-start
        result=dict(metrics=metrics,metric_units='AP fractions (0..1)',
            parameter_count=sum(p.numel() for p in runner.model.parameters()),
            fps=len(images)/elapsed,benchmark_images=len(images),warmup_images=5,
            timing_scope='batch1 FP32; preprocessing + host/device transfer + model + postprocessing; image decode excluded',
            cuda_name=torch.cuda.get_device_name(0),precision='fp32')
        save_json(output/'metrics.json',result)


if __name__=='__main__':
    main()
