"""Build all six controlled pig detection configurations from pinned MMDetection."""
from __future__ import annotations

import copy
from pathlib import Path

from tools.runtime import ROOT,load_json,path,verify_source


def specialize_model(value):
    if isinstance(value,dict):
        for key,item in list(value.items()):
            if key=='num_classes':
                value[key]=1
            elif key=='init_cfg' and isinstance(item,dict) and item.get('type')=='Pretrained':
                value[key]=None
            else:
                specialize_model(item)
    elif isinstance(value,list):
        for item in value:
            specialize_model(item)


def build_config(model_name,seed=20261022,weights=None,epochs=None,batch_size=None,workers=None,smoke_iters=0):
    from mmengine.config import Config
    verify_source('mmdetection')
    protocol=load_json('configs/detection/experiments.json')
    spec=protocol['models'][model_name]
    ann=path(protocol['annotation_directory'])
    conversion=load_json(str(ann/'conversion.json'))
    data_protocol=load_json('configs/data.json')['detection']
    split_path=data_protocol['split_manifest']
    from tools.preparation import sha256
    if conversion['splits_sha256']!=sha256(path(split_path)):
        raise ValueError('Converted labels do not correspond to the active split')
    if conversion['coordinate_convention']!=data_protocol['coordinate_convention']:
        raise ValueError('Converted labels use a different coordinate convention')
    if conversion.get('coordinate_review') and conversion['coordinate_review_sha256']!=sha256(path(conversion['coordinate_review'])):
        raise ValueError('Coordinate evidence changed after annotation conversion')
    cfg=Config.fromfile(str(ROOT/'tools/vendor/mmdetection'/spec['config']))
    specialize_model(cfg.model)
    if 'batch_augments' in cfg.model.get('data_preprocessor',{}):
        cfg.model.data_preprocessor.batch_augments=None
    if 'test_cfg' in cfg.model:
        test_cfg=cfg.model.test_cfg
        if 'score_thr' in test_cfg:
            test_cfg.score_thr=0.001
        if 'rcnn' in test_cfg:
            test_cfg.rcnn.score_thr=0.001
    scale=tuple(protocol['image_scale'])
    train_pipeline=[dict(type='LoadImageFromFile'),dict(type='LoadAnnotations',with_bbox=True),
        dict(type='Resize',scale=scale,keep_ratio=True),dict(type='RandomFlip',prob=0.5),
        dict(type='Pad',size=scale,pad_val=dict(img=(114,114,114))),dict(type='PackDetInputs')]
    test_pipeline=[dict(type='LoadImageFromFile'),dict(type='Resize',scale=scale,keep_ratio=True),
        dict(type='Pad',size=scale,pad_val=dict(img=(114,114,114))),dict(type='LoadAnnotations',with_bbox=True),
        dict(type='PackDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))]
    batch=batch_size or spec['batch_size']
    worker_count=protocol['workers'] if workers is None else workers
    def dataset(subset,training):
        return dict(type='CocoDataset',data_root=str(ROOT),data_prefix=dict(img=''),
            ann_file=str(ann/f'{subset}.coco.json'),metainfo=dict(classes=('pig',),palette=[(30,200,30)]),
            test_mode=not training,filter_cfg=dict(filter_empty_gt=False,min_size=1) if training else None,
            pipeline=copy.deepcopy(train_pipeline if training else test_pipeline))
    cfg.train_dataloader=dict(batch_size=batch,num_workers=worker_count,persistent_workers=worker_count>0,
        sampler=dict(type='DefaultSampler',shuffle=True),dataset=dataset('train',True))
    for subset,name in [('val','val'),('test','test')]:
        cfg[f'{name}_dataloader']=dict(batch_size=1,num_workers=worker_count,persistent_workers=worker_count>0,
            drop_last=False,sampler=dict(type='DefaultSampler',shuffle=False),dataset=dataset(subset,False))
        cfg[f'{name}_evaluator']=dict(type='CocoMetric',ann_file=str(ann/f'{subset}.coco.json'),metric='bbox',classwise=False)
    max_epochs=epochs or protocol['epochs']
    cfg.train_cfg=dict(type='EpochBasedTrainLoop',max_epochs=max_epochs,val_interval=protocol['validation_interval'])
    cfg.val_cfg=dict(type='ValLoop')
    cfg.test_cfg=dict(type='TestLoop')
    # Preserve each method's optimizer type, parameter groups and gradient clipping.
    base_batch=cfg.get('auto_scale_lr',{}).get('base_batch_size',16)
    cfg.optim_wrapper.optimizer.lr*=batch/base_batch
    cfg.auto_scale_lr=dict(enable=False,base_batch_size=batch)
    cfg.optim_wrapper.type='OptimWrapper'
    cfg.param_scheduler=[dict(type='MultiStepLR',begin=0,end=max_epochs,by_epoch=True,
        milestones=sorted(set([max(1,int(max_epochs*.7)),max(1,int(max_epochs*.9))])),gamma=0.1)]
    cfg.custom_hooks=[]
    cfg.default_hooks.checkpoint=dict(type='CheckpointHook',interval=1,max_keep_ckpts=2,
        save_best='coco/bbox_mAP',rule='greater')
    cfg.default_hooks.logger=dict(type='LoggerHook',interval=20)
    cfg.visualizer=dict(type='DetLocalVisualizer',vis_backends=[dict(type='LocalVisBackend')],name='visualizer')
    cfg.randomness=dict(seed=seed,deterministic=True)
    cfg.load_from=str(path(weights)) if weights else None
    cfg.resume=False
    cfg.launcher='none'
    if smoke_iters:
        cfg.train_cfg=dict(type='IterBasedTrainLoop',max_iters=smoke_iters,val_interval=smoke_iters+1)
        cfg.param_scheduler=[]
        cfg.default_hooks.checkpoint=dict(type='CheckpointHook',by_epoch=False,interval=smoke_iters,max_keep_ckpts=1)
        cfg.default_hooks.logger.interval=1
    return cfg


class Detector:
    def __init__(self,config,checkpoint,device='cuda:0'):
        from mmdet.apis import init_detector
        from mmcv.transforms import Compose
        from mmdet.utils import get_test_pipeline_cfg
        self.model=init_detector(str(path(config)),str(path(checkpoint)),device=device,palette='random')
        self.model.dataset_meta['classes']=('pig',)
        pipeline=copy.deepcopy(get_test_pipeline_cfg(self.model.cfg))
        pipeline=[step for step in pipeline if step['type']!='LoadAnnotations']
        pipeline[0]['type']='mmdet.LoadImageFromNDArray'
        self.pipeline=Compose(pipeline)

    def predict(self,bgr,score_min=0.001):
        import numpy as np
        from mmdet.apis import inference_detector
        pred=inference_detector(self.model,bgr,test_pipeline=self.pipeline).pred_instances.cpu()
        mask=(pred.labels==0)&(pred.scores>=score_min)
        boxes=pred.bboxes[mask].numpy()
        scores=pred.scores[mask].numpy()
        h,w=bgr.shape[:2]
        boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,w)
        boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,h)
        valid=(boxes[:,2]-boxes[:,0]>=1)&(boxes[:,3]-boxes[:,1]>=1)
        return np.concatenate([boxes[valid],scores[valid,None],np.zeros((int(valid.sum()),1))],axis=1).astype(np.float32)
