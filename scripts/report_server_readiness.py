"""Publish readiness only after model, tracker, evaluator and source checks pass."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.runtime import ROOT, fingerprint, load_json, path, save_json
from tools.preparation import inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = path(args.output)
    if not target.is_relative_to(ROOT / 'experiments/setup') or target.exists():
        raise ValueError('Readiness report must be a new file under experiments/setup/')
    first = 'experiments/smoke/readiness_20261022_01'
    second = 'experiments/smoke/readiness_20261022_02'
    detector_runs = {}
    for model in load_json('configs/detection/experiments.json')['models']:
        root = second if model in ('faster_rcnn', 'deformable_detr') else first
        run = f'{root}/detection/{model}/run.json'
        record = load_json(run)
        if record['status'] != 'complete' or not record['smoke'] or record['weights'] is None:
            raise ValueError(f'Detector smoke/pretrained loading not verified: {model}')
        detector_runs[model] = fingerprint(run)
    trackers = {}
    for method in load_json('configs/tracking/experiments.json')['methods']:
        run = f'{second}/tracking/{method}/run.json'
        record = load_json(run)
        if record['status'] != 'complete' or not record['smoke']:
            raise ValueError(f'Tracker smoke not verified: {method}')
        trackers[method] = fingerprint(run)
    checks = {}
    for name, filename in {
        'coco_evaluator': 'experiments/setup/coco_check_20261022_01/verification.json',
        'trackeval': 'experiments/setup/trackeval_check_20261022_01/verification.json',
        'tracker_association': 'experiments/setup/tracker_interface_check_20261022_01/verification.json',
        'conversion': 'experiments/preparation/conversion_verification_v1/verification.json',
    }.items():
        if load_json(filename)['status'] != 'passed':
            raise ValueError(f'Failed prerequisite: {name}')
        checks[name] = fingerprint(filename)
    expected = {r['path']: (r['bytes'], r['sha256']) for r in load_json(
        'experiments/preparation/audit_20261022_03/raw_inventory.json')}
    actual = {r['path']: (r['bytes'], r['sha256']) for r in inventory()}
    if expected != actual:
        raise ValueError('Raw dataset contents differ from the original audit')
    environments = {name: load_json(f'experiments/setup/{name}_environment.json')
                    for name in ('detection', 'tracking')}
    for name, env in environments.items():
        if not env['torch_cuda']['available']:
            raise ValueError(f'CUDA unavailable in {name}')
    checkpoints = {}
    for model in detector_runs:
        weight = f'checkpoints/pretrained/{model}.pth'
        recorded = load_json(weight+'.json')
        checkpoints[model] = fingerprint(weight)
        if recorded['sha256'] != checkpoints[model]['sha256']:
            raise ValueError('Detector weight fingerprint changed')
    tracking = load_json('configs/tracking/experiments.json')
    checkpoints['reid'] = fingerprint(tracking['reid_weights'])
    if checkpoints['reid']['sha256'] != tracking['reid_source']['sha256']:
        raise ValueError('ReID weight differs from author hash')
    save_json(target, dict(status='ready_for_formal_experiments', server_project=str(ROOT),
        formal_experiments_started=False, smoke_results_are_not_benchmark_results=True,
        detectors=detector_runs, trackers=trackers, checks=checks, checkpoints=checkpoints,
        environments=environments, raw_files_verified=len(actual),
        protocol=fingerprint('configs/data.json'),
        coordinate_tool_history_verified=False, source_independence_verified=False,
        evaluation_scope='declared standard-VOC / within-scene / derived MOT GT protocol',
        locks={name: fingerprint(f'experiments/setup/{name}.lock.txt')
               for name in ('detection', 'tracking')}))
    print(f'READY: six detectors, six trackers, both evaluators; {len(actual)} raw files verified')


if __name__ == '__main__':
    main()
