"""Small, testable Colab preparation/backup/run helpers. No cloud SDK required."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from uuid import uuid4


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _signature(stat):
    return (stat.st_size, stat.st_mtime_ns, stat.st_ino)


def copy_stable(source, destination, *, expected_sha256=None):
    """Publish only a complete snapshot; leave the previous backup on races."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name('.' + destination.name + '.' + uuid4().hex + '.tmp')
    try:
        checksum = hashlib.sha256() if expected_sha256 else None
        with source.open('rb') as reader, temporary.open('wb') as writer:
            before = _signature(os.fstat(reader.fileno()))
            for block in iter(lambda: reader.read(1024 * 1024), b''):
                writer.write(block)
                if checksum is not None:
                    checksum.update(block)
            after = _signature(os.fstat(reader.fileno()))
        if before != after or before != _signature(source.stat()):
            raise RuntimeError(f'Source changed during backup; retry on next sync: {source}')
        if checksum is not None and checksum.hexdigest() != expected_sha256:
            raise ValueError(f'Reconstruction content differs from audited manifest: {source}')
        os.replace(temporary, destination)
        return before
    finally:
        temporary.unlink(missing_ok=True)


def mirror_tree(source, destination, signatures=None, expected_hashes=None):
    source, destination = Path(source), Path(destination)
    cache = signatures if signatures is not None else {}
    count = 0
    expected_hashes = expected_hashes or {}
    for relative in expected_hashes:
        if not (source / relative).is_file():
            raise FileNotFoundError(f'Missing reconstruction in backup source: {source / relative}')
    if not source.is_dir():
        return count
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source)
        if not path.is_file() or any(p.startswith('.') for p in relative.parts) or path.suffix == '.tmp':
            continue
        target = destination / relative
        if cache.get(str(relative)) == _signature(path.stat()) and target.is_file():
            continue
        cache[str(relative)] = copy_stable(path, target, expected_sha256=expected_hashes.get(str(relative)))
        count += 1
    return count


def _reconstruction_hashes(manifest, root):
    from data.drct_manifest import read_drct_manifest
    result = {}
    root = Path(root).resolve()
    for row in read_drct_manifest(manifest):
        if row['sample_kind'].endswith('_reconstruction'):
            path = Path(row['path']).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f'Reconstruction must be inside the backed-up reconstruction root: {path}')
            if not row.get('sha256'):
                raise ValueError(f'Reconstruction has no audited hash: {path}')
            result[str(path.relative_to(root))] = row['sha256']
    return result


def save_preparation(manifest, bundle, reconstruction_root):
    """Save hash-audited CSV/gate plus reconstructions, never copy original images."""
    from data.manifest import verify_manifest_gate
    manifest, bundle, reconstruction_root = map(Path, (manifest, bundle, reconstruction_root))
    verify_manifest_gate(manifest, require_hashes=True)
    expected = _reconstruction_hashes(manifest, reconstruction_root)
    mirror_tree(reconstruction_root, bundle / 'reconstructions', expected_hashes=expected)
    copy_stable(manifest, bundle / 'selected_manifest.csv')
    copy_stable(manifest.with_suffix('.verified.json'), bundle / 'selected_manifest.verified.json')
    atomic_json(bundle / 'bundle.json', dict(
        manifest_path=str(manifest.resolve()), reconstruction_root=str(reconstruction_root.resolve()),
        manifest_sha256=digest(manifest), gate_sha256=digest(manifest.with_suffix('.verified.json'))))


def restore_preparation(bundle, manifest, reconstruction_root, *, immutable_dataset=False):
    """Restore into the original paths after extracting the SAME original dataset."""
    from data.manifest import verify_manifest_gate
    from data.drct_manifest import read_drct_manifest
    if not immutable_dataset:
        raise ValueError('Restore requires explicit unchanged-dataset confirmation; otherwise prepare/audit again.')
    bundle, manifest, reconstruction_root = map(Path, (bundle, manifest, reconstruction_root))
    meta = json.loads((bundle / 'bundle.json').read_text(encoding='utf-8'))
    if str(manifest.resolve()) != meta['manifest_path'] or str(reconstruction_root.resolve()) != meta['reconstruction_root']:
        raise ValueError('Restore to the recorded paths; changing CSV paths requires a new audit.')
    for filename, key in [('selected_manifest.csv', 'manifest_sha256'), ('selected_manifest.verified.json', 'gate_sha256')]:
        if digest(bundle / filename) != meta[key]:
            raise ValueError(f'Preparation bundle changed: {filename}')
    # Always restore reconstruction files, even if the manifest already exists.
    expected = _reconstruction_hashes(bundle / 'selected_manifest.csv', reconstruction_root)
    mirror_tree(bundle / 'reconstructions', reconstruction_root, expected_hashes=expected)
    copy_stable(bundle / 'selected_manifest.csv', manifest)
    copy_stable(bundle / 'selected_manifest.verified.json', manifest.with_suffix('.verified.json'))
    verify_manifest_gate(manifest, require_hashes=True)
    for row in read_drct_manifest(manifest):
        if not Path(row['path']).is_file():
            raise FileNotFoundError(f'Restore the original dataset/reconstruction: {row["path"]}')
    return manifest


def current_commit(repo):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()


def create_run(repo, manifest, output_base, drive_runs, name, training_args):
    """Persist the exact argument vector, run identity and data digest before launch."""
    from data.manifest import verify_manifest_gate
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Use a simple experiment name without paths.')
    args = [str(a) for a in training_args]
    reserved = {'--name', '--run_id', '--checkpoints_dir', '--manifest', '--val_manifest', '--continue_train', '--resume_checkpoint'}
    if any(a.split('=')[0] in reserved for a in args):
        raise ValueError('Run identity and manifests are managed by create_run, not training_args.')
    if '--epochs' not in args:
        raise ValueError('Declare the total epoch budget explicitly.')
    manifest = Path(manifest).resolve()
    verify_manifest_gate(manifest, require_hashes=True)
    run_id = uuid4().hex[:12]
    experiment = Path(output_base).resolve() / f'{name}_{run_id}'
    drive_dir = Path(drive_runs).resolve() / experiment.name
    args += ['--manifest', str(manifest), '--val_manifest', str(manifest),
             '--name', name, '--run_id', run_id, '--checkpoints_dir', str(Path(output_base).resolve())]
    descriptor = dict(version=1, commit=current_commit(repo), args=args,
                      experiment_dir=str(experiment), drive_dir=str(drive_dir),
                      manifest=str(manifest), manifest_sha256=digest(manifest),
                      gate_sha256=digest(manifest.with_suffix('.verified.json')))
    # Keep descriptor outside local experiment: ExperimentManager creates it.
    atomic_json(drive_dir / 'run_descriptor.json', descriptor)
    return descriptor


def read_run(drive_dir):
    descriptor = json.loads((Path(drive_dir) / 'run_descriptor.json').read_text(encoding='utf-8'))
    if Path(descriptor['drive_dir']).resolve() != Path(drive_dir).resolve():
        raise ValueError('Run descriptor does not identify this backup directory.')
    return descriptor


def run_training(descriptor, repo, *, resume=False, interval_seconds=60):
    """Run the saved command; mirror the entire experiment during and after exit."""
    from data.manifest import verify_manifest_gate
    if current_commit(repo) != descriptor['commit']:
        raise ValueError(f'Restore repository commit {descriptor["commit"]} before resuming/running.')
    manifest = Path(descriptor['manifest'])
    if digest(manifest) != descriptor['manifest_sha256'] or digest(manifest.with_suffix('.verified.json')) != descriptor['gate_sha256']:
        raise ValueError('Preparation identity changed since this run was created.')
    verify_manifest_gate(manifest, require_hashes=True)
    experiment, drive_dir = Path(descriptor['experiment_dir']), Path(descriptor['drive_dir'])
    args = list(descriptor['args'])
    if resume:
        local_last = experiment / 'checkpoints' / 'last.pth'
        if not local_last.is_file():
            mirror_tree(drive_dir, experiment)
        if not local_last.is_file():
            raise FileNotFoundError(f'No saved last.pth in {drive_dir}; do not start a replacement run silently.')
        args += ['--continue_train', '--resume_checkpoint', str(local_last)]
    elif experiment.exists():
        raise FileExistsError(f'Run directory exists: {experiment}. Resume it or create a new run.')
    stop = threading.Event()
    signatures = {}
    def sync_loop():
        while not stop.wait(interval_seconds):
            try:
                mirror_tree(experiment, drive_dir, signatures)
            except Exception as exc:
                print(f'Backup retry needed: {exc}', flush=True)
    thread = threading.Thread(target=sync_loop, daemon=True)
    thread.start()
    process = None
    try:
        process = subprocess.Popen([sys.executable, '-u', 'train.py', *args], cwd=repo,
                                   env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, process.args)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        stop.set()
        thread.join()
        # After writer exit, include logs/config/CSV and complete checkpoints.
        mirror_tree(experiment, drive_dir, signatures)
    expected = experiment / 'checkpoints' / 'best.pth'
    if not expected.is_file():
        raise FileNotFoundError(f'Run returned success without best.pth: {expected}')
    print(f'Run saved to Drive: {drive_dir}', flush=True)
    return expected
