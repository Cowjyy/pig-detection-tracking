"""Check the exact evaluator on synthetic perfect and erroneous trajectories.

These fixture scores are software checks, never experimental pig-data results.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_tracking import score_files
from tools.runtime import run_directory, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with run_directory(args.output, 'evaluator_selfcheck', dict(synthetic=True)) as (out, state):
        ground_truth = {'A': '1,1,0,0,10,10,1,1,1\n2,1,0,0,10,10,1,1,1\n3,1,0,0,10,10,1,1,1\n',
                        'B': '1,1,20,20,10,10,1,1,1\n'}
        perfect = {'A': '1,1,0,0,10,10,1,1,-1,-1\n2,1,0,0,10,10,1,1,-1,-1\n3,1,0,0,10,10,1,1,-1,-1\n',
                   'B': '1,1,20,20,10,10,1,1,-1,-1\n'}
        imperfect = {'A': '1,1,0,0,10,10,1,1,-1,-1\n2,2,0,0,10,10,1,1,-1,-1\n',
                     'B': '1,1,20,20,10,10,1,1,-1,-1\n1,2,100,100,10,10,1,1,-1,-1\n'}
        for seq, rows in ground_truth.items():
            dest = out / 'gt' / seq / 'gt'
            dest.mkdir(parents=True)
            (dest / 'gt.txt').write_text(rows)
        summaries = {}
        for name, predictions in [('perfect', perfect), ('imperfect', imperfect)]:
            dest = out / name / 'data'
            dest.mkdir(parents=True)
            for seq, rows in predictions.items():
                (dest / f'{seq}.txt').write_text(rows)
            summary, raw, messages = score_files(out / name, 'fixture', out / 'scores',
                                                  out / 'gt', {'A': 3, 'B': 1})
            summaries[name] = summary
        for field in ('HOTA', 'IDF1', 'MOTA'):
            if abs(summaries['perfect'][field]-1) > 1e-9:
                raise AssertionError(f'Perfect fixture failed: {field}')
        for field in ('FP', 'FN', 'IDSW'):
            if summaries['perfect'][field] != 0 or summaries['imperfect'][field] != 1:
                raise AssertionError(f'Fixture count failed: {field}')
        if abs(summaries['imperfect']['MOTA']-.25) > 1e-9:
            raise AssertionError('Combined MOTA must use total counts, not the mean of per-sequence MOTA')
        if abs(summaries['imperfect']['IDF1']-.5) > 1e-9:
            raise AssertionError('Identity matching fixture failed')
        save_json(out / 'verification.json', dict(status='passed', synthetic_fixture=True,
            pig_data_used=False, benchmark_result=False, expected_imperfect_counts=dict(FP=1, FN=1, IDSW=1),
            summaries=summaries))
        print('PASS: pinned TrackEval perfect match, identity switch, false positive, false negative and sequence aggregation')


if __name__ == '__main__':
    main()
