"""Collect completed formal runs into per-seed and aggregate comparison tables."""
import argparse
import csv
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.runtime import load_json,path,save_json,fingerprint


def completed(directory,kind):
    meta=load_json(str(path(directory)/'run.json'))
    if meta['status']!='complete' or meta['kind']!=kind or meta.get('smoke',False):
        raise ValueError(f'Not a completed formal {kind}: {directory}')
    return meta


def write_csv(filename,rows):
    with filename.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tracking_metrics(group):
    return dict(HOTA_pct=100*statistics.mean(group['HOTA']['HOTA']),
        IDF1_pct=100*group['Identity']['IDF1'],MOTA_pct=100*group['CLEAR']['MOTA'],
        IDSW=group['CLEAR']['IDSW'],FP=group['CLEAR']['CLR_FP'],FN=group['CLEAR']['CLR_FN'])


def summarize_detection_rows(rows):
    if len({row['parameters'] for row in rows})!=1:
        raise ValueError('Parameter count differs across seeds of the same model')
    first=rows[0]
    summary=dict(model=first['model'],method=first['method'],seeds=len(rows),parameters=first['parameters'])
    if len(rows)==1:
        summary['seed']=first['seed']
        for metric in ('AP50_pct','AP50_95_pct','FPS'):
            summary[metric]=first[metric]
    else:
        for metric in ('AP50_pct','AP50_95_pct','FPS'):
            summary[metric+'_mean']=statistics.mean(row[metric] for row in rows)
            summary[metric+'_std']=statistics.stdev(row[metric] for row in rows)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',required=True)
    args=parser.parse_args()
    batch=path(args.batch)
    detection=load_json('configs/detection/experiments.json')
    tracking=load_json('configs/tracking/experiments.json')
    seeds=load_json(str(batch/'plan.json'))['seeds']
    per_seed=[]
    aggregate=[]
    for model,spec in detection['models'].items():
        rows=[]
        for seed in seeds:
            evaluation=batch/'detection/test'/f'{model}_s{seed}'
            speed=batch/'detection/speed'/f'{model}_s{seed}'
            completed(evaluation,'detection_eval')
            completed(speed,'detection_benchmark')
            metric=load_json(str(evaluation/'metrics.json'))
            timing=load_json(str(speed/'metrics.json'))
            row=dict(model=model,method=spec['name'],seed=seed,
                AP50_pct=100*metric['metrics']['coco/bbox_mAP_50'],
                AP50_95_pct=100*metric['metrics']['coco/bbox_mAP'],
                parameters=metric['parameter_count'],FPS=timing['fps'])
            rows.append(row)
            per_seed.append(row)
        aggregate.append(summarize_detection_rows(rows))
    track_rows=[]
    per_sequence=[]
    for method,spec in tracking['methods'].items():
        evaluation=batch/'tracking/test_eval'/method
        run=batch/'tracking/test'/method
        completed(evaluation,'tracking_eval')
        completed(run,'tracking')
        metrics=load_json(str(evaluation/'metrics.json'))['metrics']
        timing=load_json(str(run/'timing.json'))
        track_rows.append(dict(method=method,name=spec['name'],HOTA_pct=100*metrics['HOTA'],
            IDF1_pct=100*metrics['IDF1'],MOTA_pct=100*metrics['MOTA'],IDSW=metrics['IDSW'],
            FP=metrics['FP'],FN=metrics['FN'],association_FPS=timing['overall_tracking_fps'],
            detector_plus_tracker_FPS=timing['overall_detector_plus_tracker_fps']))
        raw=load_json(str(evaluation/'trackeval.json'))['results']
        sequences=next(iter(next(iter(raw.values())).values()))
        for sequence,classes in sequences.items():
            if sequence!='COMBINED_SEQ':
                per_sequence.append(dict(method=method,sequence=sequence,**tracking_metrics(classes['pedestrian'])))
    out=batch/'results'
    out.mkdir(exist_ok=False)
    write_csv(out/'detection_per_seed.csv',per_seed)
    write_csv(out/'detection_comparison.csv',aggregate)
    write_csv(out/'tracking_comparison.csv',track_rows)
    write_csv(out/'tracking_per_sequence.csv',per_sequence)
    save_json(out/'results.json',dict(detection_per_seed=per_seed,detection=aggregate,
        tracking=track_rows,tracking_per_sequence=per_sequence,
        data_protocol=fingerprint('configs/data.json'),selected_detector=load_json(str(batch/'selection/selection.json')),
        seeds=seeds,seed_protocol='fixed_single_seed' if len(seeds)==1 else 'multiple_independent_seeds',
        AP_units='percent',std='not_estimated_single_run' if len(seeds)==1 else 'sample standard deviation across configured seeds',
        overall_tracking_metrics='TrackEval COMBINED_SEQ, never arithmetic means of per-sequence percentages',
        speed_scope='serial FP32 batch 1; decode excluded; detector+tracker is not live end-to-end FPS',
        source_independence_verified=False,gt_protocol='mot_clean_v1'))
    print(f'Wrote comparison tables to {out}')


if __name__=='__main__':
    main()
