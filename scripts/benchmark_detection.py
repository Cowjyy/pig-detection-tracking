"""Measure a frozen detector separately from AP evaluation on an exclusive GPU."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.runtime import load_json,path,fingerprint,run_directory,save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--images',type=int,default=20)
    args=parser.parse_args()
    if args.images<1:
        parser.error('--images must be positive')
    training=load_json(str(path(args.run)/'run.json'))
    if training['status']!='complete' or training['smoke']:
        raise ValueError('Benchmark needs a completed formal training run')
    import cv2
    import torch
    from models.detectors.mmdet_adapter import Detector
    ann=path(load_json('configs/detection/experiments.json')['annotation_directory'])
    samples=load_json(str(ann/'val.coco.json'))['images'][:args.images]
    images=[cv2.imread(str(path(row['file_name']))) for row in samples]
    if not images or any(image is None for image in images):
        raise ValueError('Invalid benchmark images')
    checkpoint=training['best_checkpoint']
    with run_directory(args.output,'detection_benchmark',dict(model=training['model'],seed=training['seed'],
            smoke=False,training_run=args.run,checkpoint=fingerprint(checkpoint))) as (out,state):
        detector=Detector(str(path(args.run)/'config.py'),checkpoint)
        for index in range(5):
            detector.predict(images[index%len(images)])
        torch.cuda.synchronize()
        times=[]
        for image in images:
            torch.cuda.synchronize()
            start=time.perf_counter()
            detector.predict(image)
            torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
        save_json(out/'metrics.json',dict(fps=len(images)/sum(times),seconds_per_image=times,
            images=[row['file_name'] for row in samples],warmup_images=5,precision='fp32',batch_size=1,
            parameter_count=sum(p.numel() for p in detector.model.parameters()),
            cuda_name=torch.cuda.get_device_name(0),
            timing_scope='preprocessing + transfer + model + postprocessing; decode excluded'))


if __name__=='__main__':
    main()
