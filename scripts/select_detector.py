"""Freeze the highest validation-AP detector and checkpoint, never selecting on test."""
import argparse
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.runtime import load_json,path,run_directory,save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',required=True,help='Validation evaluation directories')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    rows=[]
    for directory in args.runs:
        meta=load_json(str(path(directory)/'run.json'))
        if meta['status']!='complete' or meta['kind']!='detection_eval' or meta['split']!='val' or meta['smoke']:
            raise ValueError('Selection accepts completed validation evaluations only')
        metrics=load_json(str(path(directory)/'metrics.json'))
        rows.append(dict(run=directory,meta=meta,ap=metrics['metrics']['coco/bbox_mAP']))
    reference=rows[0]['meta']['data_split']
    if any(row['meta']['data_split']!=reference for row in rows):
        raise ValueError('Do not compare different split versions')
    grouped={}
    for row in rows:
        grouped.setdefault(row['meta']['model'],[]).append(row)
    for name,items in grouped.items():
        if len({r['meta']['seed'] for r in items})!=len(items):
            raise ValueError(f'Duplicate seed results for {name}')
    seed_sets={tuple(sorted(r['meta']['seed'] for r in items)) for items in grouped.values()}
    if len(seed_sets)!=1:
        raise ValueError('All compared models must have the same evaluated seeds')
    winner=max(sorted(grouped),key=lambda name:statistics.mean(r['ap'] for r in grouped[name]))
    # The selected model's highest-validation-AP seed is explicit and reproducible.
    selected=max(grouped[winner],key=lambda row:(row['ap'],-row['meta']['seed']))
    checkpoint=selected['meta']['checkpoint']
    with run_directory(args.output,'detector_selection',dict(smoke=False,data_split=reference)) as (output,state):
        save_json(output/'selection.json',dict(model=winner,seed=selected['meta']['seed'],
            checkpoint=checkpoint,training_run=selected['meta']['training_run'],validation_run=selected['run'],
            config=str((path(selected['meta']['training_run'])/'config.py').relative_to(path('.'))),
            selection_metric=f'validation AP50:95 at fixed seed {selected["meta"]["seed"]}' if len(next(iter(seed_sets)))==1 else 'validation AP50:95; mean across supplied equal seed sets',
            candidates={name:dict(mean_ap=statistics.mean(r['ap'] for r in items),seeds=[r['meta']['seed'] for r in items]) for name,items in grouped.items()},
            compared_models=len(grouped),data_split=reference))


if __name__=='__main__':
    main()
