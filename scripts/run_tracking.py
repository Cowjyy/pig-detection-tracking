"""Run one pinned tracker on a shared detector cache, never reading GT boxes."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.runtime import load_json,path,fingerprint,run_directory,save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    methods=load_json('configs/tracking/experiments.json')['methods']
    parser.add_argument('--method',required=True,choices=list(methods))
    parser.add_argument('--cache',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--parameters',help='Frozen JSON from tune_tracking.py, or tuning-only parameter dict')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    cache_meta=load_json(str(path(args.cache)/'run.json'))
    cache=load_json(str(path(args.cache)/'cache.json'))
    if cache_meta['status']!='complete' or cache_meta['kind']!='detection_cache' or cache['smoke']!=args.smoke:
        raise ValueError('Wrong or incomplete detector cache; smoke must be explicit')
    parameters={}
    parameter_record=None
    if args.parameters:
        parameter_record=load_json(args.parameters)
        if parameter_record.get('kind')=='frozen_tracker_parameters':
            if parameter_record['method']!=args.method or parameter_record['selected_on']!='tuning':
                raise ValueError('Invalid frozen parameter source')
            if parameter_record.get('protocol')!=cache['protocol']:
                raise ValueError('Frozen parameters use a different sequence split')
            if parameter_record.get('detector_checkpoint')!=cache['checkpoint']:
                raise ValueError('Frozen parameters use a different detector checkpoint')
            parameters=parameter_record['parameters']
        elif cache['subset']=='tuning' and not parameter_record.get('kind'):
            parameters=parameter_record
        else:
            raise ValueError('Test runs require parameters frozen on tuning sequences')
    elif cache['subset']=='test' and not args.smoke:
        raise ValueError('Freeze tracker parameters on tuning data before test')
    import cv2
    import numpy as np
    import torch
    from models.trackers.boxmot_adapter import Tracker,mot_lines
    reid=load_json('configs/tracking/experiments.json')['reid_weights'] if methods[args.method]['reid'] else None
    with run_directory(args.output,'tracking',dict(method=args.method,subset=cache['subset'],smoke=args.smoke,
        cache=fingerprint(str(path(args.cache)/'cache.json')),detector_checkpoint=cache['checkpoint'],protocol=cache['protocol'],
        reid_weights=fingerprint(reid) if reid else None,
        reid_domain=load_json('configs/tracking/experiments.json')['reid_domain'] if reid else None,
        parameter_source=fingerprint(args.parameters) if args.parameters else None)) as (output,state):
        (output/'data').mkdir()
        timings=[]
        for sequence in cache['sequences']:
            cache_file=path(args.cache)/sequence['cache_file']
            if fingerprint(str(cache_file))['sha256']!=sequence['cache_sha256']:
                raise ValueError('Cache content changed')
            if fingerprint(sequence['video'])['sha256']!=sequence['video_sha256']:
                raise ValueError('Video changed since detector caching')
            with np.load(cache_file,allow_pickle=False) as content:
                offsets=content['offsets'].copy()
                detections=content['detections'].copy()
            if offsets.shape!=(sequence['frames']+1,) or offsets[0]!=0 or offsets[-1]!=len(detections) or np.any(np.diff(offsets)<0):
                raise ValueError('Invalid frame offsets in cache')
            tracker=Tracker(args.method,parameters=parameters)
            capture=cv2.VideoCapture(str(path(sequence['video'])))
            rows=[]
            tracking_seconds=0.0
            start_stream=time.perf_counter()
            for index in range(sequence['frames']):
                ok,bgr=capture.read()
                if not ok:
                    raise RuntimeError('Video ended before detector cache')
                det=detections[offsets[index]:offsets[index+1]]
                torch.cuda.synchronize()
                start=time.perf_counter()
                tracks=tracker.update(det,bgr)
                torch.cuda.synchronize()
                tracking_seconds+=time.perf_counter()-start
                rows.extend(mot_lines(index+1,tracks))
            wall=time.perf_counter()-start_stream
            capture.release()
            (output/'data'/f'{sequence["sequence"]}.txt').write_text('\n'.join(rows)+('\n' if rows else ''),encoding='utf-8')
            timings.append(dict(sequence=sequence['sequence'],frames=sequence['frames'],track_seconds=tracking_seconds,
                detector_seconds=sequence['inference_seconds'],tracking_fps=sequence['frames']/tracking_seconds,
                detector_plus_tracker_fps=sequence['frames']/(tracking_seconds+sequence['inference_seconds']),
                decode_plus_tracker_seconds=wall,
                timing_note='synchronized serial processing; detector+tracker excludes video decode and file I/O; no claimed live-pipeline end-to-end FPS'))
            state['effective_parameters']=tracker.parameters
        save_json(output/'timing.json',dict(sequences=timings,
            overall_tracking_fps=sum(r['frames'] for r in timings)/sum(r['track_seconds'] for r in timings),
            overall_detector_plus_tracker_fps=sum(r['frames'] for r in timings)/sum(r['track_seconds']+r['detector_seconds'] for r in timings)))


if __name__=='__main__':
    main()
