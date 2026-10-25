"""Shared experiment paths, provenance records and crash-visible run states."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys
import traceback

from tools.preparation import ROOT, sha256

# Runner may enable deterministic algorithms in both training and evaluation.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def path(value):
    target=(ROOT/value).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError(f'Path must stay inside the project: {value}')
    return target


def load_json(value):
    return json.loads(path(value).read_text(encoding='utf-8'))


def save_json(target,value):
    def convert(item):
        if hasattr(item,'tolist'):
            return item.tolist()
        if isinstance(item,Path):
            return str(item)
        raise TypeError(type(item).__name__)
    temporary=target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=convert,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(target)


def fingerprint(value):
    target=path(value)
    return dict(path=target.relative_to(ROOT).as_posix(),sha256=sha256(target))


def verify_source(name):
    requested=load_json('configs/upstream_sources.json')[name]
    marker=load_json(f'tools/vendor/{name}/SOURCE_LOCK.json')
    if requested['commit']!=marker['commit']:
        raise ValueError(f'Wrong upstream {name}; run bootstrap_sources.py')
    return requested


def environment():
    packages={}
    for name in ['torch','torchvision','numpy','opencv-python','mmcv','mmengine','mmdet','boxmot','pycocotools']:
        try:
            packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name]=None
    return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=packages,
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


@contextmanager
def run_directory(value,kind,metadata):
    output=path(value)
    if not output.is_relative_to((ROOT/'experiments').resolve()) or output==(ROOT/'experiments').resolve():
        raise ValueError('Run output must be a new child of experiments/')
    output.mkdir(parents=True,exist_ok=False)
    state=dict(schema_version=1,kind=kind,status='running',created_utc=datetime.now(timezone.utc).isoformat(),
               environment=environment(),**metadata)
    save_json(output/'run.json',state)
    try:
        yield output,state
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}')
        (output/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        save_json(output/'run.json',state)
        raise
    else:
        state.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(output/'run.json',state)


def tracking_protocol(subset):
    if subset not in ('tuning','test'):
        raise ValueError('Tracking subset must be tuning or test')
    config=load_json('configs/data.json')['tracking']
    protocol=load_json(config['split_manifest'])
    wanted=protocol[f'{subset}_sequences']
    records=[r for r in protocol['sequences'] if r['sequence'] in wanted]
    if sorted(r['sequence'] for r in records)!=sorted(wanted):
        raise ValueError('Missing or duplicate sequence assignments')
    for record in records:
        for field,digest in [('video','video_sha256'),('gt','gt_sha256')]:
            if sha256(path(record[field]))!=record[digest]:
                raise ValueError(f'Changed tracking source: {record[field]}')
    return records,fingerprint(config['split_manifest'])
