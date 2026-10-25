"""Verify all tracker adapters associate two synthetic moving objects."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.runtime import load_json, run_directory, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import numpy as np
    from models.trackers.boxmot_adapter import Tracker
    image = np.random.default_rng(17).integers(0, 256, (320, 640, 3), dtype=np.uint8)
    with run_directory(args.output, 'tracker_interface_selfcheck', dict(
            synthetic=True, benchmark_result=False)) as (out, state):
        checked = {}
        for method in load_json('configs/tracking/experiments.json')['methods']:
            tracker = Tracker(method)
            history = []
            for frame in range(10):
                delta = frame*2
                detections = np.array([[40+delta, 50, 140+delta, 210, .95, 0],
                                       [350+delta, 70, 450+delta, 230, .95, 0]], dtype=np.float32)
                tracks = tracker.update(detections, image)
                history.append(sorted(int(row[4]) for row in tracks))
            if not (len(history[-1]) == 2 and history[-1] == history[-2] == history[-3]):
                raise AssertionError(f'{method} did not preserve two simple synthetic trajectories: {history}')
            checked[method] = dict(status='passed', final_ids=history[-1],
                                   frames=10, tracked_objects=2)
        save_json(out / 'verification.json', dict(status='passed', methods=checked,
            synthetic_fixture=True, pig_data_used=False, benchmark_result=False))
        print('PASS: all six tracker adapters return stable IDs on the synthetic association fixture')


if __name__ == '__main__':
    main()
