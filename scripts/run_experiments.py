"""Durable serial detector-seed matrix and six-tracker experiment controller.

Training can share a GPU with sufficient free memory. Speed-related stages wait
for an exclusive idle GPU. A failed stage stops the pipeline and preserves all
logs and checkpoints. Existing experiment outputs are never overwritten.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.runtime import ROOT,load_json,path,fingerprint,save_json


def now():
    return datetime.now(timezone.utc).isoformat()


def build_plan(output,seeds=None):
    detection=load_json('configs/detection/experiments.json')
    tracking=load_json('configs/tracking/experiments.json')
    seeds=list(detection['seeds'] if seeds is None else seeds)
    if not seeds or len(seeds)!=len(set(seeds)) or any(not isinstance(seed,int) or seed<0 for seed in seeds):
        raise ValueError('Seeds must be a nonempty list of unique nonnegative integers')
    det=str(ROOT/'.venvs/detection/bin/python')
    trk=str(ROOT/'.venvs/tracking/bin/python')
    jobs=[]
    def job(key,python,script,arguments,exclusive=False):
        jobs.append(dict(key=key,command=[python,'-B','-u',f'scripts/{script}']+arguments,
                         exclusive_gpu=exclusive))
    job('verify_raw',det,'verify_raw.py',['--inventory','experiments/preparation/audit_20261022_03/raw_inventory.json','--ignore-mtime'])
    validation=[]
    # Complete one seed across all methods before progressing to the next seed.
    for seed in seeds:
        for model in detection['models']:
            train=f'{output}/detection/train/{model}_s{seed}'
            val=f'{output}/detection/val/{model}_s{seed}'
            job(f'train_{model}_{seed}',det,'train_detection.py',
                ['--model',model,'--seed',str(seed),'--output',train])
            job(f'val_{model}_{seed}',det,'evaluate_detection.py',
                ['--run',train,'--split','val','--skip-benchmark','--output',val])
            validation.append(val)
    selection=f'{output}/selection'
    job('select_detector',det,'select_detector.py',['--runs']+validation+['--output',selection])
    for seed in seeds:
        for model in detection['models']:
            train=f'{output}/detection/train/{model}_s{seed}'
            job(f'test_{model}_{seed}',det,'evaluate_detection.py',
                ['--run',train,'--split','test','--skip-benchmark','--output',f'{output}/detection/test/{model}_s{seed}'])
    for seed in seeds:
        for model in detection['models']:
            job(f'speed_{model}_{seed}',det,'benchmark_detection.py',
                ['--run',f'{output}/detection/train/{model}_s{seed}',
                 '--output',f'{output}/detection/speed/{model}_s{seed}'],True)
    for subset in ('tuning','test'):
        job(f'cache_{subset}',det,'cache_detections.py',
            ['--selection',selection,'--subset',subset,'--score-min','0.001','--output',f'{output}/cache/{subset}'],True)
    variants={
        'bytetrack':[{},dict(track_thresh=.4),dict(track_thresh=.8)],
        'botsort':[{},dict(track_high_thresh=.4,new_track_thresh=.5),dict(track_high_thresh=.8,new_track_thresh=.9)],
        'ocsort':[{},dict(det_thresh=.4),dict(det_thresh=.8)],
        'strongsort':[{},dict(min_conf=.4),dict(min_conf=.8)],
        'deepocsort':[{},dict(det_thresh=.3),dict(det_thresh=.7)],
        'boosttrack':[{},dict(det_thresh=.4),dict(det_thresh=.8)]}
    parameter_files={}
    for method in tracking['methods']:
        evaluations=[]
        for index,parameters in enumerate(variants[method]):
            param=f'{output}/parameters/{method}_trial{index}.json'
            parameter_files[param]=parameters
            run=f'{output}/tracking/tuning/{method}_trial{index}'
            evaluation=f'{output}/tracking/tuning_eval/{method}_trial{index}'
            job(f'tune_run_{method}_{index}',trk,'run_tracking.py',
                ['--method',method,'--cache',f'{output}/cache/tuning','--parameters',param,'--output',run],True)
            job(f'tune_eval_{method}_{index}',trk,'evaluate_tracking.py',['--run',run,'--output',evaluation])
            evaluations.append(evaluation)
        job(f'freeze_{method}',trk,'tune_tracking.py',
            ['--method',method,'--evaluations']+evaluations+['--output',f'{output}/tracking/frozen/{method}'])
    for method in tracking['methods']:
        run=f'{output}/tracking/test/{method}'
        job(f'tracking_test_{method}',trk,'run_tracking.py',
            ['--method',method,'--cache',f'{output}/cache/test',
             '--parameters',f'{output}/tracking/frozen/{method}/frozen_parameters.json','--output',run],True)
        job(f'tracking_eval_{method}',trk,'evaluate_tracking.py',
            ['--run',run,'--output',f'{output}/tracking/test_eval/{method}'])
    job('collect_results',det,'collect_experiment_results.py',['--batch',output])
    return dict(schema_version=1,jobs=jobs,parameters=parameter_files,
        detection_models=list(detection['models']),seeds=seeds,epochs=detection['epochs'],
        seed_protocol='fixed_single_seed' if len(seeds)==1 else 'multiple_independent_seeds',
        tracking_methods=list(tracking['methods']),tracking_trials_per_method=3,
        selection_rule=f'validation AP at fixed seed {seeds[0]}; no test-driven tuning' if len(seeds)==1 else 'mean validation AP over supplied seeds; no test-driven tuning',
        speed_rule='serial, exclusive GPU before every timed stage')


def resources():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    index,free,util=result.stdout.splitlines()[0].split(',')
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                        capture_output=True,text=True,check=True,timeout=15)
    return dict(gpu=int(index),free_mib=int(free),utilization=int(util),
                compute_pids=[int(line.strip()) for line in apps.stdout.splitlines() if line.strip().isdigit()])


def execute(output,plan):
    import fcntl
    project_lock=ROOT/'experiments/formal/controller.lock'
    project_lock.parent.mkdir(parents=True,exist_ok=True)
    lock=project_lock.open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    readiness=load_json('experiments/setup/readiness_20261022.json')
    if readiness['status']!='ready_for_formal_experiments' or readiness['protocol']!=fingerprint('configs/data.json'):
        raise ValueError('Readiness report does not match current data protocol')
    for record in readiness['checkpoints'].values():
        if fingerprint(record['path'])!=record:
            raise ValueError('A verified initial checkpoint changed')
    out=path(output)
    out.mkdir(parents=True,exist_ok=False)
    (out/'logs').mkdir()
    frozen={}
    for folder in ('scripts','models','tools'):
        for filename in (ROOT/folder).rglob('*.py'):
            if 'vendor' not in filename.relative_to(ROOT).parts:
                frozen[filename.relative_to(ROOT).as_posix()]=fingerprint(str(filename))
    for filename in ('configs/data.json','configs/detection/experiments.json','configs/tracking/experiments.json',
                     'configs/upstream_sources.json'):
        frozen[filename]=fingerprint(filename)
    save_json(out/'plan.json',dict(**plan,created_utc=now(),frozen_inputs=frozen))
    for filename,parameters in plan['parameters'].items():
        path(filename).parent.mkdir(parents=True,exist_ok=True)
        save_json(path(filename),parameters)
    state=dict(status='starting',pid=os.getpid(),created_utc=now(),completed_jobs=0,
               total_jobs=len(plan['jobs']),current_job=None,project=str(ROOT),formal=True)
    (out/'controller.pid').write_text(str(os.getpid())+'\n')
    child=None
    def publish(**changes):
        state.update(changes,updated_utc=now())
        save_json(out/'status.json',state)
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
             MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    try:
        for number,job in enumerate(plan['jobs'],1):
            for filename,expected in frozen.items():
                if fingerprint(filename)!=expected:
                    raise ValueError(f'Frozen experiment source changed: {filename}')
            publish(current_job=job['key'],job_number=number,status='waiting_for_resources',child_pid=None)
            idle_samples=0
            while True:
                gpu=resources()
                eligible=gpu['free_mib']>=8000
                if job['exclusive_gpu']:
                    eligible=eligible and not gpu['compute_pids'] and gpu['utilization']<=10
                idle_samples=idle_samples+1 if eligible else 0
                if eligible and (not job['exclusive_gpu'] or idle_samples>=3):
                    break
                publish(status='waiting_for_exclusive_gpu' if job['exclusive_gpu'] else 'waiting_for_gpu_memory',gpu=gpu)
                time.sleep(10 if eligible else 30)
            logfile=out/'logs'/f'{number:03d}_{job["key"]}.log'
            started=time.monotonic()
            with logfile.open('x') as stream:
                child=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                                       start_new_session=True)
                publish(status='running',child_pid=child.pid,current_log=logfile.relative_to(ROOT).as_posix(),
                        command=job['command'],gpu_at_start=gpu,job_started_utc=now())
                print(f'[{now()}] START {number}/{len(plan["jobs"])} {job["key"]} PID={child.pid}',flush=True)
                last_update=0
                while child.poll() is None:
                    elapsed=time.monotonic()-started
                    if elapsed-last_update>=30:
                        publish(elapsed_seconds=round(elapsed,1))
                        last_update=elapsed
                    time.sleep(3)
                code=child.returncode
                child=None
            with (out/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(job=job['key'],exit_code=code,finished_utc=now(),
                    elapsed_seconds=time.monotonic()-started,log=logfile.relative_to(ROOT).as_posix()))+'\n')
            if code:
                raise RuntimeError(f'{job["key"]} exited {code}; see {logfile}')
            publish(completed_jobs=number)
            print(f'[{now()}] DONE {job["key"]}',flush=True)
        publish(status='complete',current_job=None,child_pid=None,finished_utc=now())
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
        publish(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        raise
    finally:
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Continue an existing batch using an audited handover record')
    parser.add_argument('--handover',help='JSON recording the verified live training process to preserve')
    args=parser.parse_args()
    target=path(args.output)
    if not target.is_relative_to(ROOT/'experiments/formal') or target==ROOT/'experiments/formal':
        parser.error('Batch output must be a new child of experiments/formal/')
    plan=build_plan(target.relative_to(ROOT).as_posix())
    if args.plan_only:
        print(json.dumps(plan,indent=2))
        return
    if args.resume:
        if not args.handover:
            parser.error('--resume requires --handover')
        from tools.experiment_resume import resume
        resume(args.output,plan,args.handover,resources)
    else:
        execute(args.output,plan)


if __name__=='__main__':
    main()
