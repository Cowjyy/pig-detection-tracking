"""Audited seed-budget reduction without restarting a live training process."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from tools.runtime import ROOT,load_json,path,fingerprint,save_json


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_seed_reduction(previous,replacement,completed,current_key):
    if not set(replacement['seeds']).issubset(previous['seeds']):
        raise ValueError('Handover may only reduce the previous seed budget')
    for field in ('detection_models','epochs','tracking_methods','tracking_trials_per_method','parameters','speed_rule'):
        if previous[field]!=replacement[field]:
            raise ValueError(f'Handover changed more than the seed budget: {field}')
    before={job['key']:job for job in previous['jobs']}
    after={job['key']:job for job in replacement['jobs']}
    if not set(after).issubset(before):
        raise ValueError('Handover cannot introduce unreviewed jobs')
    for key in set(completed)|{current_key}:
        if key not in after or after[key]!=before[key]:
            raise ValueError(f'An executed or running job was altered: {key}')
    for key,job in after.items():
        if key!='select_detector' and job!=before[key]:
            raise ValueError(f'A retained job command changed: {key}')


def process_matches(record):
    folder=Path(f'/proc/{record["child_pid"]}')
    try:
        stat=folder.joinpath('stat').read_text().rsplit(')',1)[1].split()
        args=[item.decode() for item in folder.joinpath('cmdline').read_bytes().split(b'\0') if item]
    except FileNotFoundError:
        return False
    if stat[0]=='Z':
        return False
    if stat[19]!=str(record['child_start_ticks']) or args!=record['command']:
        raise ValueError('Training PID identity changed; refuse to adopt or signal it')
    return True


def resume(output,plan,handover_file,resource_reader):
    import fcntl
    out=path(output)
    record=load_json(handover_file)
    if record['batch']!=out.relative_to(ROOT).as_posix():
        raise ValueError('Handover belongs to another experiment batch')
    lock=(ROOT/'experiments/formal/controller.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=load_json(str(out/'plan.json'))
    previous_status=record['prior_status']
    completed=set()
    for line in (out/'events.jsonl').read_text().splitlines():
        event=json.loads(line)
        if event.get('status')=='complete' or event.get('exit_code')==0:
            completed.add(event['job'])
    current_key=previous_status['current_job']
    validate_seed_reduction(prior,plan,completed,current_key)
    current=next(job for job in plan['jobs'] if job['key']==current_key)
    if current['command']!=record['command'] or current['command']!=previous_status['command']:
        raise ValueError('Live training command does not match the retained job')
    alive=process_matches(record)
    training=path(current['command'][current['command'].index('--output')+1])
    training_meta=load_json(str(training/'run.json'))
    if training_meta['seed'] not in plan['seeds'] or training_meta['smoke']:
        raise ValueError('Cannot retain this training run in the new budget')
    if not alive and training_meta['status']!='complete':
        raise ValueError('The recorded trainer is gone and its output is incomplete')
    readiness=load_json('experiments/setup/readiness_20261022.json')
    if readiness['protocol']!=fingerprint('configs/data.json'):
        raise ValueError('Data protocol changed during the handover')
    for item in readiness['checkpoints'].values():
        if fingerprint(item['path'])!=item:
            raise ValueError('An initial weight changed during the handover')
    frozen={}
    for folder in ('scripts','models','tools'):
        for filename in (ROOT/folder).rglob('*.py'):
            if 'vendor' not in filename.relative_to(ROOT).parts:
                frozen[filename.relative_to(ROOT).as_posix()]=fingerprint(str(filename))
    for filename in ('configs/data.json','configs/detection/experiments.json','configs/tracking/experiments.json',
                     'configs/upstream_sources.json'):
        frozen[filename]=fingerprint(filename)
    replacement=dict(**plan,created_utc=prior['created_utc'],revised_utc=now(),frozen_inputs=frozen,
        revision_reason=record['reason'],previous_plan=fingerprint(str(out/'plan.json')),
        handover=fingerprint(handover_file),preserved_training_pid=record['child_pid'],
        note='Only the repeat-seed budget changed; original training metadata and checkpoint history are retained')
    save_json(out/'plan.json',replacement)
    for filename,parameters in plan['parameters'].items():
        if path(filename).exists():
            if load_json(filename)!=parameters:
                raise ValueError('Existing tracker parameter trial changed')
        else:
            path(filename).parent.mkdir(parents=True,exist_ok=True)
            save_json(path(filename),parameters)
    state=dict(previous_status,pid=os.getpid(),status='resuming',updated_utc=now(),
        total_jobs=len(plan['jobs']),completed_jobs=len(completed),seeds=plan['seeds'],
        seed_protocol=plan['seed_protocol'],adopted_training_pid=record['child_pid'],
        resumed_utc=now(),previous_controller_pid=record['old_controller_pid'])
    (out/'controller.pid').write_text(str(os.getpid())+'\n')
    def publish(**changes):
        state.update(changes,updated_utc=now())
        save_json(out/'status.json',state)
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
             MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    child=None
    adopting=False
    try:
        for number,job in enumerate(plan['jobs'],1):
            if job['key'] in completed:
                continue
            for filename,expected in frozen.items():
                if fingerprint(filename)!=expected:
                    raise ValueError(f'Frozen experiment source changed: {filename}')
            adopting=job['key']==current_key
            if adopting:
                logfile=path(previous_status['current_log'])
                elapsed_before=max(0,(datetime.now(timezone.utc)-datetime.fromisoformat(previous_status['job_started_utc'])).total_seconds())
                started=time.monotonic()-elapsed_before
                publish(status='running',current_job=job['key'],job_number=number,
                        child_pid=record['child_pid'],current_log=logfile.relative_to(ROOT).as_posix(),
                        command=job['command'],adopted=True)
                print(f'[{now()}] ADOPT {job["key"]} PID={record["child_pid"]}; training continues unchanged',flush=True)
                while process_matches(record):
                    publish(elapsed_seconds=round(time.monotonic()-started,1))
                    time.sleep(10)
                finished=load_json(str(training/'run.json'))
                if finished['status']!='complete' or not finished.get('best_checkpoint'):
                    raise RuntimeError('Adopted trainer did not finish with a completed best checkpoint')
                code=None
                adopting=False
            else:
                publish(status='waiting_for_resources',current_job=job['key'],job_number=number,child_pid=None,adopted=False)
                idle_samples=0
                while True:
                    gpu=resource_reader()
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
                    child=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdout=stream,
                                           stderr=subprocess.STDOUT,start_new_session=True)
                    publish(status='running',child_pid=child.pid,command=job['command'],gpu_at_start=gpu,
                        current_log=logfile.relative_to(ROOT).as_posix(),job_started_utc=now())
                    print(f'[{now()}] START {number}/{len(plan["jobs"])} {job["key"]} PID={child.pid}',flush=True)
                    while child.poll() is None:
                        publish(elapsed_seconds=round(time.monotonic()-started,1))
                        time.sleep(10)
                    code=child.returncode
                    child=None
                if code:
                    raise RuntimeError(f'{job["key"]} exited {code}; see {logfile}')
            with (out/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(job=job['key'],status='complete',exit_code=code,
                    adopted=(job['key']==current_key),finished_utc=now(),
                    elapsed_seconds=time.monotonic()-started,log=logfile.relative_to(ROOT).as_posix()))+'\n')
            completed.add(job['key'])
            publish(completed_jobs=len(completed))
            print(f'[{now()}] DONE {job["key"]}',flush=True)
        publish(status='complete',current_job=None,child_pid=None,finished_utc=now())
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
        if isinstance(error,KeyboardInterrupt) and adopting and process_matches(record):
            os.killpg(record['child_pid'],signal.SIGTERM)
        publish(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        raise
    finally:
        lock.close()
