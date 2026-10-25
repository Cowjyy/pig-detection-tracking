"""Check that the launched matrix follows the frozen evaluation boundaries."""
import unittest
from scripts.run_experiments import build_plan
from scripts.collect_experiment_results import summarize_detection_rows
from tools.experiment_resume import validate_seed_reduction


class ExperimentPlanTests(unittest.TestCase):
    def test_all_formal_models_seeds_and_test_after_selection(self):
        jobs=build_plan('experiments/formal/plan_check')['jobs']
        training=[j for j in jobs if j['key'].startswith('train_')]
        self.assertEqual(len(training),6)
        self.assertEqual(len({j['command'][j['command'].index('--model')+1] for j in training}),6)
        for job in jobs:
            self.assertNotIn('--smoke',job['command'])
            self.assertNotIn('--random-init',job['command'])
        self.assertEqual({j['command'][j['command'].index('--seed')+1] for j in training},{'20261022'})
        selection=next(i for i,j in enumerate(jobs) if j['key']=='select_detector')
        tests=[i for i,j in enumerate(jobs) if j['key'].startswith('test_')]
        self.assertEqual(len(tests),6)
        self.assertTrue(all(index>selection for index in tests))

    def test_tracking_is_frozen_before_test_and_speed_is_exclusive(self):
        plan=build_plan('experiments/formal/plan_check')
        jobs=plan['jobs']
        for method in plan['tracking_methods']:
            freeze=next(i for i,j in enumerate(jobs) if j['key']==f'freeze_{method}')
            test=next(i for i,j in enumerate(jobs) if j['key']==f'tracking_test_{method}')
            self.assertLess(freeze,test)
            self.assertIn(f'experiments/formal/plan_check/tracking/frozen/{method}/frozen_parameters.json',jobs[test]['command'])
        for job in jobs:
            if job['key'].startswith(('speed_','cache_','tune_run_','tracking_test_')):
                self.assertTrue(job['exclusive_gpu'])

    def test_seed_reduction_retains_running_job_and_removes_future_repeats(self):
        previous=build_plan('experiments/formal/plan_check',seeds=[20261022,20261023,20261024])
        current=build_plan('experiments/formal/plan_check')
        validate_seed_reduction(previous,current,{'verify_raw'},'train_faster_rcnn_20261022')
        self.assertEqual(len(previous['jobs']),131)
        self.assertEqual(len(current['jobs']),83)
        with self.assertRaises(ValueError):
            validate_seed_reduction(previous,current,{'verify_raw'},'train_faster_rcnn_20261023')

    def test_single_seed_table_does_not_claim_zero_variance(self):
        row=dict(model='test',method='Test',seed=20261022,parameters=100,AP50_pct=80,AP50_95_pct=60,FPS=30)
        summary=summarize_detection_rows([row])
        self.assertEqual(summary['seed'],20261022)
        self.assertEqual(summary['AP50_95_pct'],60)
        self.assertFalse(any(key.endswith(('_std','_mean')) for key in summary))
        repeated=summarize_detection_rows([row,dict(row,seed=20261023,AP50_95_pct=70)])
        self.assertGreater(repeated['AP50_95_pct_std'],0)


if __name__=='__main__':
    unittest.main()
