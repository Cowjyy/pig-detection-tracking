"""Preserve tracker-specific association logic behind an explicit shared interface."""
import importlib
import inspect
import sys

import numpy as np

from tools.runtime import ROOT,load_json,path,verify_source,fingerprint


def default_parameters(method):
    import yaml
    verify_source('boxmot')
    config=ROOT/'tools/vendor/boxmot/boxmot/configs'/f'{method}.yaml'
    values=yaml.safe_load(config.read_text())
    return {name:spec['default'] for name,spec in values.items()}


class Tracker:
    def __init__(self,method,device='cuda:0',parameters=None,reid_weights=None):
        import torch
        protocol=load_json('configs/tracking/experiments.json')
        spec=protocol['methods'][method]
        verify_source('boxmot')
        sys.path.insert(0,str(ROOT/'tools/vendor/boxmot'))
        boxmot=importlib.import_module('boxmot')
        if not path(str(boxmot.__file__)).is_relative_to(ROOT/'tools/vendor/boxmot'):
            raise RuntimeError('Imported BoxMOT is not the pinned implementation')
        actual=default_parameters(method)
        actual.update(spec['overrides'])
        if parameters:
            unknown=set(parameters)-set(actual)
            if unknown:
                raise ValueError(f'Unknown tracker parameters: {unknown}')
            actual.update(parameters)
        self.reid_weights=path(reid_weights or protocol['reid_weights']) if spec['reid'] else None
        if self.reid_weights is not None and not self.reid_weights.is_file():
            raise ValueError('Missing frozen ReID weights; run download_weights.py --reid')
        if self.reid_weights is not None and fingerprint(str(self.reid_weights))['sha256']!=protocol['reid_source']['sha256']:
            raise ValueError('ReID checkpoint differs from the frozen author-published weights')
        self.parameters=actual
        self.method=method
        self.impl=boxmot.create_tracker(method,tracker_config=None,reid_weights=self.reid_weights,
            device=torch.device(device),half=False,per_class=False,evolve_param_dict=dict(actual))

    def update(self,detections,bgr):
        detections=np.asarray(detections,dtype=np.float32).reshape(-1,6)
        if not np.isfinite(detections).all() or not (detections[:,5]==0).all():
            raise ValueError('Tracker cache must contain finite single-class pig detections')
        tracks=np.asarray(self.impl.update(detections,bgr))
        if tracks.size==0:
            return np.empty((0,7),dtype=np.float64)
        if tracks.ndim!=2 or tracks.shape[1]<7 or not np.isfinite(tracks[:,:7]).all():
            raise ValueError('Unexpected pinned tracker output schema')
        # Upstream: x1,y1,x2,y2,ID,confidence,class,(detection index).
        output=tracks[:,:7].astype(np.float64,copy=True)
        h,w=bgr.shape[:2]
        output[:,[0,2]]=np.clip(output[:,[0,2]],0,w)
        output[:,[1,3]]=np.clip(output[:,[1,3]],0,h)
        valid=(output[:,2]>output[:,0])&(output[:,3]>output[:,1])
        output=output[valid]
        if len(set(output[:,4]))!=len(output) or np.any(output[:,4]<1) or np.any(output[:,4]!=output[:,4].astype(int)):
            raise ValueError('Duplicate or invalid identity returned within one frame')
        if np.any(output[:,6]!=0):
            raise ValueError('Tracker produced a non-pig class')
        return output


def mot_lines(frame,tracks):
    rows=[]
    for x1,y1,x2,y2,identity,confidence,category in tracks:
        rows.append(f'{frame},{int(identity)},{x1:.6f},{y1:.6f},{x2-x1:.6f},{y2-y1:.6f},{confidence:.8f},1,-1,-1')
    return rows
