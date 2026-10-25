"""Evaluate saved MOT-format tracks with the pinned TrackEval implementation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.runtime import ROOT, fingerprint, load_json, path, run_directory, save_json, verify_source


def evaluate(run, subset, output):
    verify_source('TrackEval')
    metadata = load_json(str(path(run) / 'run.json'))
    if metadata.get('kind') != 'tracking' or metadata.get('status') != 'complete':
        raise ValueError('Expected a completed tracking run')
    if metadata.get('smoke'):
        raise ValueError('Smoke runs cannot produce benchmark metrics')
    if metadata.get('subset') != subset:
        raise ValueError('Requested subset does not match the tracking run')
    protocol = load_json('configs/data.json')['tracking']
    manifest = load_json(protocol['split_manifest'])
    if metadata['protocol'] != fingerprint(protocol['split_manifest']):
        raise ValueError('Tracking run uses a different active split protocol')
    expected = manifest[f'{subset}_sequences']
    actual = [row['sequence'] for row in load_json(str(path(run) / 'timing.json'))['sequences']]
    if actual != expected:
        raise ValueError(f'Run sequence list differs from frozen {subset} split')
    gt_root = path('datasets/processed/mot/clean_v1')
    sequence_info = {row['sequence']: row['frames'] for row in manifest['sequences']
                     if row['sequence'] in expected}
    for row in manifest['sequences']:
        if row['sequence'] in expected:
            gt = gt_root / row['sequence'] / 'gt/gt.txt'
            if fingerprint(str(gt))['sha256'] != row['gt_sha256']:
                raise ValueError('Evaluation GT does not match the frozen derived GT')
    return score_files(run, subset, output, gt_root, sequence_info)


def score_files(run, subset, output, gt_root, sequence_info):
    """Pinned scorer, also used by the independent synthetic evaluator self-check."""
    verify_source('TrackEval')

    sys.path.insert(0, str(ROOT / 'tools/vendor/TrackEval'))
    import numpy as np
    # This pinned TrackEval revision still uses NumPy's removed scalar aliases.
    for alias, builtin in (('float', float), ('int', int), ('bool', bool)):
        if not hasattr(np, alias):
            setattr(np, alias, builtin)
    import trackeval

    dataset_config = dict(
        GT_FOLDER=str(gt_root), TRACKERS_FOLDER=str(path(run).parent),
        TRACKERS_TO_EVAL=[path(run).name], CLASSES_TO_EVAL=['pedestrian'],
        BENCHMARK='PIG', SPLIT_TO_EVAL=subset, SKIP_SPLIT_FOL=True,
        SEQ_INFO=sequence_info, TRACKER_SUB_FOLDER='data',
        GT_LOC_FORMAT='{gt_folder}/{seq}/gt/gt.txt', DO_PREPROC=False,
        PRINT_CONFIG=False, OUTPUT_FOLDER=str(path(output)),
        OUTPUT_SUB_FOLDER='trackeval')
    evaluator_config = trackeval.Evaluator.get_default_eval_config()
    evaluator_config.update(dict(PRINT_RESULTS=True, PRINT_ONLY_COMBINED=False,
                                 OUTPUT_SUMMARY=True, OUTPUT_DETAILED=True,
                                 PLOT_CURVES=False, USE_PARALLEL=False,
                                 BREAK_ON_ERROR=True, PRINT_CONFIG=False))
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    evaluator = trackeval.Evaluator(evaluator_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    results, messages = evaluator.evaluate([dataset], metrics)
    dataset_name = dataset.get_name()
    if messages.get(dataset_name, {}).get(path(run).name) != 'Success':
        raise RuntimeError(f'TrackEval failed: {messages}')
    combined = results[dataset_name][path(run).name]['COMBINED_SEQ']['pedestrian']
    hota = combined['HOTA']['HOTA']
    summary = dict(HOTA=float(np.mean(hota)), HOTA_at_0_5=float(hota[9]),
        DetA_at_0_5=float(combined['HOTA']['DetA'][9]),
        AssA_at_0_5=float(combined['HOTA']['AssA'][9]),
        IDF1=float(combined['Identity']['IDF1']),
        MOTA=float(combined['CLEAR']['MOTA']),
        IDSW=int(combined['CLEAR']['IDSW']),
        FP=int(combined['CLEAR']['CLR_FP']), FN=int(combined['CLEAR']['CLR_FN']))
    return summary, results, messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Completed scripts/run_tracking.py output')
    parser.add_argument('--output', required=True, help='New child of experiments/')
    args = parser.parse_args()
    meta = load_json(str(path(args.run) / 'run.json'))
    with run_directory(args.output, 'tracking_eval', dict(
            tracking_run=args.run, method=meta['method'], subset=meta['subset'],
            smoke=False, protocol=meta['protocol'],
            tracking_run_fingerprint=fingerprint(str(path(args.run) / 'run.json')))) as (out, state):
        summary, result, messages = evaluate(args.run, meta['subset'], out)
        save_json(out / 'metrics.json', dict(metrics=summary, unit='fractions for HOTA/IDF1/MOTA; counts for IDSW/FP/FN',
            sequences=load_json(str(path(args.run) / 'timing.json'))['sequences'],
            interpretation='Combined sequence metrics are recomputed by TrackEval; sequence percentages must not be averaged.',
            gt_protocol='mot_clean_v1', subset=meta['subset']))
        save_json(out / 'trackeval.json', dict(results=result, messages=messages,
            evaluator='pinned TrackEval; MOTChallenge pedestrian class slot aliases pig class 1',
            preprocessing=False, gt_protocol='mot_clean_v1', subset=meta['subset']))


if __name__ == '__main__':
    main()
