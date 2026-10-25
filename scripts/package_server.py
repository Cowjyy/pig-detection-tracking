"""Package source, derived data and both intact datasets for the lab server."""
import argparse
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.preparation import ROOT, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New .tar.gz inside experiments/setup/')
    args = parser.parse_args()
    target = (ROOT / args.output).resolve()
    if not target.is_relative_to(ROOT / 'experiments/setup') or target.exists():
        raise ValueError('Bundle must be a new file under experiments/setup/')
    target.parent.mkdir(parents=True, exist_ok=True)
    included = ['README.md', 'requirements.txt', '.gitignore', 'configs', 'models', 'losses',
                'tools', 'scripts', 'docs', 'tests', 'datasets', 'experiments/preparation',
                'pig-objection detection', 'PIG_MOT_TEST_DATA']
    files = []
    for name in included:
        source = ROOT / name
        files.extend([source] if source.is_file() else source.rglob('*'))
    files = sorted(p for p in files if p.is_file() and '__pycache__' not in p.parts
                   and not p.name.endswith(('.pyc', '.pyo')))
    with tarfile.open(target, 'x:gz', compresslevel=1) as archive:
        for source in files:
            archive.add(source, arcname=source.relative_to(ROOT).as_posix(), recursive=False)
    write_json(target.with_suffix(target.suffix + '.json'), dict(
        files=len(files), bytes=target.stat().st_size, sha256=sha256(target),
        raw_data_modified=False, exclude_server_environment=True,
        included_roots=included))
    print(f'Packaged {len(files)} files into {target} ({target.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
