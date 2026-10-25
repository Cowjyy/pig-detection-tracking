"""Train one detector, or perform explicit short server-only smoke verification."""
import argparse
import os
from pathlib import Path
import sys

# Required by PyTorch's deterministic CUDA linear algebra before CUDA starts.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.runtime import load_json,path,fingerprint,run_directory,save_json
from models.detectors.mmdet_adapter import build_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,choices=list(load_json('configs/detection/experiments.json')['models']))
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,default=20261022)
    parser.add_argument('--weights',help='Defaults to checkpoints/pretrained/<model>.pth')
    parser.add_argument('--resume',help='Checkpoint to resume into a NEW output directory')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--workers',type=int)
    parser.add_argument('--smoke-iters',type=int,default=0)
    parser.add_argument('--random-init',action='store_true',help='Allowed only for smoke verification, never scored as formal training')
    args=parser.parse_args()
    if args.random_init and args.smoke_iters<1:
        parser.error('--random-init is restricted to --smoke-iters')
    if args.smoke_iters<0 or args.epochs is not None and args.epochs<1 or args.batch_size is not None and args.batch_size<1:
        parser.error('Invalid training budget')
    weights=args.resume or args.weights or f'checkpoints/pretrained/{args.model}.pth'
    if args.random_init:
        weights=None
    elif not path(weights).is_file():
        parser.error(f'Missing checkpoint {weights}; run download_weights.py')
    config=build_config(args.model,args.seed,weights,args.epochs,args.batch_size,args.workers,args.smoke_iters)
    if args.resume:
        config.resume=True
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Training/smoke must run on the laboratory CUDA server')
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    register_all_modules()
    data=load_json('configs/data.json')['detection']
    meta=dict(model=args.model,seed=args.seed,smoke=bool(args.smoke_iters),
        split=fingerprint(data['split_manifest']),weights=fingerprint(weights) if weights else None,
        protocol=fingerprint('configs/detection/experiments.json'),
        evaluation_scope=data['evaluation_scope'])
    with run_directory(args.output,'detection_train',meta) as (output,state):
        config.work_dir=str(output)
        config.dump(str(output/'config.py'))
        runner=Runner.from_cfg(config)
        runner.train()
        state['parameter_count']=sum(p.numel() for p in runner.model.parameters())
        best=runner.message_hub.runtime_info.get('best_ckpt')
        if best:
            state['best_checkpoint']=str(Path(best).relative_to(path('.')))


if __name__=='__main__':
    main()
