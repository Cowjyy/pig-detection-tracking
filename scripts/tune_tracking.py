"""Freeze the best completed tracker trial using tuning-sequence HOTA only.

Run candidate parameter settings with run_tracking.py --subset tuning, evaluate
each output with evaluate_tracking.py, then pass those evaluation directories
here. This selector never opens test GT or chooses a tracker using test metrics.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.runtime import fingerprint, load_json, path, run_directory, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True)
    parser.add_argument('--evaluations', nargs='+', required=True,
                        help='Completed evaluate_tracking.py directories on tuning sequences')
    parser.add_argument('--output', required=True, help='New child of experiments/')
    args = parser.parse_args()
    protocol = load_json('configs/tracking/experiments.json')
    if args.method not in protocol['methods']:
        parser.error(f'Unknown tracker method: {args.method}')
    trials = []
    for evaluation in args.evaluations:
        eval_dir = path(evaluation)
        eval_meta = load_json(str(eval_dir / 'run.json'))
        if (eval_meta.get('kind') != 'tracking_eval' or eval_meta.get('status') != 'complete'
                or eval_meta.get('method') != args.method or eval_meta.get('subset') != 'tuning'):
            raise ValueError('Every tuning candidate must have a completed tuning-only evaluation')
        tracking_dir = path(eval_meta['tracking_run'])
        tracking_meta = load_json(str(tracking_dir / 'run.json'))
        if (tracking_meta.get('kind') != 'tracking' or tracking_meta.get('status') != 'complete'
                or tracking_meta.get('method') != args.method or tracking_meta.get('subset') != 'tuning'
                or tracking_meta.get('smoke')):
            raise ValueError('Evaluation does not point to a completed formal tuning run')
        metrics = load_json(str(eval_dir / 'metrics.json'))['metrics']
        parameters = tracking_meta.get('effective_parameters')
        if not isinstance(parameters, dict):
            raise ValueError('Tracking run did not record its effective parameters')
        trials.append(dict(evaluation=evaluation, tracking_run=eval_meta['tracking_run'],
            parameters=parameters, metrics=metrics,
            cache=tracking_meta['cache'], protocol=tracking_meta['protocol'],
            detector_checkpoint=tracking_meta['detector_checkpoint']))
    cache_fingerprints = {str(row['cache']) for row in trials}
    protocol_fingerprints = {str(row['protocol']) for row in trials}
    checkpoint_fingerprints = {str(row['detector_checkpoint']) for row in trials}
    if len(cache_fingerprints) != 1 or len(protocol_fingerprints) != 1 or len(checkpoint_fingerprints) != 1:
        raise ValueError('All tuning trials must share the exact detector cache and split protocol')
    winner = max(trials, key=lambda row: (row['metrics']['HOTA'], row['metrics']['IDF1'],
                                          row['metrics']['MOTA']))
    with run_directory(args.output, 'tracker_tuning', dict(method=args.method,
            subset='tuning', smoke=False, protocol=winner['protocol'],
            cache=winner['cache'], selection_metric='combined HOTA; ties IDF1 then MOTA')) as (out, state):
        frozen = dict(schema_version=1, kind='frozen_tracker_parameters', method=args.method,
            selected_on='tuning', parameters=winner['parameters'],
            selected_tracking_run=winner['tracking_run'], selected_evaluation=winner['evaluation'],
            metrics=winner['metrics'], detector_cache=winner['cache'],
            detector_checkpoint=winner['detector_checkpoint'], protocol=winner['protocol'])
        save_json(out / 'frozen_parameters.json', frozen)
        save_json(out / 'trials.json', dict(trials=trials, winner=winner['evaluation'],
            selection_rule='highest combined HOTA; ties IDF1 then MOTA', test_data_used=False))
        state['frozen_parameters'] = str((out / 'frozen_parameters.json').relative_to(path('.')))


if __name__ == '__main__':
    main()
