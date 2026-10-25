"""Run the frozen detector once; every tracker consumes identical candidate boxes."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.runtime import ROOT,load_json,path,fingerprint,run_directory,save_json,tracking_protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',help='Completed select_detector.py output')
    parser.add_argument('--subset',choices=['tuning','test'],required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--score-min',type=float,default=.001)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--training-run',help='Smoke-only bypass of formal detector selection')
    parser.add_argument('--max-frames',type=int,default=0)
    args=parser.parse_args()
    if not .001<=args.score_min<=.01:
        parser.error('Cache threshold must lie in [0.001,0.01] to preserve low-score candidates')
    if args.smoke:
        if not args.training_run or args.max_frames<1:
            parser.error('Smoke requires --training-run and positive --max-frames')
        training=load_json(str(path(args.training_run)/'run.json'))
        if training['kind']!='detection_train' or training['status']!='complete':
            raise ValueError('Smoke checkpoint must come from a completed detector run')
        candidates=list(path(args.training_run).glob('iter_*.pth'))
        if not candidates:
            raise ValueError('No smoke checkpoint found')
        checkpoint=str(max(candidates,key=lambda p:int(p.stem.split('_')[-1])).relative_to(ROOT))
        config=str((path(args.training_run)/'config.py').relative_to(ROOT))
    else:
        if not args.selection or args.training_run or args.max_frames:
            parser.error('Formal cache requires --selection and complete videos')
        selection_meta=load_json(str(path(args.selection)/'run.json'))
        if selection_meta['status']!='complete' or selection_meta['kind']!='detector_selection':
            raise ValueError('Incomplete detector selection')
        selection=load_json(str(path(args.selection)/'selection.json'))
        if fingerprint(selection['checkpoint']['path'])!=selection['checkpoint']:
            raise ValueError('Frozen detector weights changed')
        checkpoint=selection['checkpoint']['path']
        config=selection['config']
    records,protocol=tracking_protocol(args.subset)
    import cv2
    import numpy as np
    import torch
    from models.detectors.mmdet_adapter import Detector
    detector=Detector(config,checkpoint)
    with run_directory(args.output,'detection_cache',dict(smoke=args.smoke,subset=args.subset,
        checkpoint=fingerprint(checkpoint),config=fingerprint(config),score_min=args.score_min,
        protocol=protocol,selection=args.selection,frames_limited=bool(args.max_frames))) as (output,state):
        manifest=[]
        for record in records:
            capture=cv2.VideoCapture(str(path(record['video'])))
            ok,first=capture.read()
            if not ok:
                raise RuntimeError(f'Cannot read {record["video"]}')
            for _ in range(3):
                detector.predict(first,args.score_min)
            capture.set(cv2.CAP_PROP_POS_FRAMES,0)
            torch.cuda.synchronize()
            offsets=[0]
            all_detections=[]
            count=0
            inference_seconds=0.0
            start_stream=time.perf_counter()
            while not args.max_frames or count<args.max_frames:
                ok,bgr=capture.read()
                if not ok:
                    break
                torch.cuda.synchronize()
                start=time.perf_counter()
                det=detector.predict(bgr,args.score_min)
                torch.cuda.synchronize()
                inference_seconds+=time.perf_counter()-start
                all_detections.append(det)
                offsets.append(offsets[-1]+len(det))
                count+=1
                if count%300==0:
                    print(f'{record["sequence"]}: cached {count} frames',flush=True)
            stream_seconds=time.perf_counter()-start_stream
            capture.release()
            expected=min(record['frames'],args.max_frames) if args.max_frames else record['frames']
            if count!=expected:
                raise RuntimeError('Incomplete video decode; refuse to publish a complete cache')
            filename=f'{record["sequence"]}.npz'
            np.savez_compressed(output/filename,offsets=np.asarray(offsets,dtype=np.int64),
                detections=np.concatenate(all_detections) if all_detections else np.empty((0,6),dtype=np.float32))
            manifest.append(dict(sequence=record['sequence'],video=record['video'],video_sha256=record['video_sha256'],
                frames=count,original_frames=record['frames'],width=record['width'],height=record['height'],
                cache_file=filename,cache_sha256=fingerprint(str(output/filename))['sha256'],
                inference_seconds=inference_seconds,decode_and_inference_seconds=stream_seconds,
                warmup_frames=3,detector_fps=count/inference_seconds))
        save_json(output/'cache.json',dict(schema_version=1,sequences=manifest,subset=args.subset,
            smoke=args.smoke,score_min=args.score_min,checkpoint=fingerprint(checkpoint),protocol=protocol,
            detection_schema=['x1','y1','x2','y2','score','class_0_pig']))


if __name__=='__main__':
    main()
