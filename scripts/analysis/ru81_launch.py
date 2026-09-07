"""Bounded eight-GPU shard scheduler; no Slurm and no agent-side polling."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from src.utils.metrics import auroc

TASKS = ["cptac_lscc/ARID1A_mutation", "cptac_lscc/Histologic_Grade",
         "cptac_lscc/KEAP1_mutation", "cptac_luad/KRAS_mutation",
         "cptac_pda/SMAD4_mutation", "ucla_lung/progression_regression",
         "cptac_ccrcc/PBRM1_mutation"]


def main():
    tag = sys.argv[1]
    output = ROOT / 'predictions' / tag
    output.mkdir(parents=True, exist_ok=False)
    logdir = ROOT / 'logs' / tag
    logdir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    deadline = start + 900
    manifest = {'tag': tag, 'tasks': TASKS, 'gpus': list(range(8)), 'wall_limit_seconds': 900,
                'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff'])).hexdigest(),
                'files': {}, 'fold_files': {}, 'commands': [], 'jobs': []}
    for path in [*ROOT.glob('scripts/analysis/ru81_*.py'), ROOT/'docs/ru/RU-81.json']:
        manifest['files'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for task in TASKS:
        path = Path(os.environ['OFFICIAL']) / task / 'k=all.tsv'
        manifest['fold_files'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    work = queue.Queue()
    for task in TASKS:
        for offset in (0, 25):
            work.put((task, offset))
    lock, failed = threading.Lock(), threading.Event()
    processes = []

    def worker(gpu):
        while not failed.is_set():
            try:
                task, offset = work.get_nowait()
            except queue.Empty:
                return
            name = task.replace('/', '_') + f'_{offset}'
            cmd = [sys.executable, str(ROOT/'scripts/analysis/ru81_worker.py'),
                   '--diagnostic-output', str(output/f'{name}_diag.pt'),
                   '--config', str(ROOT/'configs/archive/v94_v102_cell_value/train_v98_p1_reverse_1536_1gpu.yaml'),
                   '--official-folds', str(Path(os.environ['OFFICIAL'])/task),
                   '--features', os.environ['FEATURES'], '--input-dim', '1536',
                   '--precision', 'bf16-mixed', '--official-fold-start', str(offset),
                   '--official-nfolds', '25', '--output', str(output/f'{name}_base.pt')]
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
            with lock:
                manifest['commands'].append({'gpu': gpu, 'task': task, 'offset': offset, 'argv': cmd})
                (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
            then = time.monotonic()
            try:
                with (logdir/f'{name}.log').open('w') as log:
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    with lock:
                        processes.append(proc)
                    rc = proc.wait(timeout=max(1, deadline-time.monotonic()))
                    if rc:
                        raise RuntimeError(f'{name}: exit {rc}; see {logdir/name}.log')
            except Exception:
                failed.set()
                with lock:
                    for proc in processes:
                        if proc.poll() is None:
                            proc.kill()
                raise
            finally:
                with lock:
                    manifest['jobs'].append({'gpu': gpu, 'task': task, 'offset': offset,
                                             'seconds': time.monotonic()-then})
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker, gpu) for gpu in range(8)]
            for future in futures:
                future.result()
        for task in TASKS:
            name = task.replace('/', '_')
            parts = [torch.load(output/f'{name}_{o}_base.pt', weights_only=False) for o in (0,25)]
            if [i for p in parts for i in p['fold_indices']] != list(range(50)):
                raise ValueError(f'{task}: missing/duplicate folds')
            rows = [r for p in parts for r in p['per_fold']]
            diag = [r for o in (0,25) for r in torch.load(output/f'{name}_{o}_diag.pt', weights_only=False)['rows']]
            if len(diag) != 50:
                raise ValueError('Missing diagnostic folds')
            for r,d in zip(rows,diag):
                if r['slide_id'] != d['slide_id'] or not torch.equal(r['label'], d['label']):
                    raise ValueError('Diagnostic alignment failed')
            values = [auroc(r['probability'],r['label']) for r in rows]
            merged = {**parts[0], 'official_folds':50, 'fold_start':0, 'fold_indices':list(range(50)),
                      'per_fold':rows, 'fold_aurocs':values, 'fold_auroc_mean':sum(values)/50,
                      'fold_auroc_std':torch.tensor(values).std(unbiased=False).item(),
                      'auroc_pooled':auroc(torch.cat([r['probability'] for r in rows]),torch.cat([r['label'] for r in rows]))}
            torch.save(merged, ROOT/'predictions'/f'pathobench_{name}_{tag}_official50_bf16.pt')
            torch.save({'rows':diag,'fold_indices':list(range(50))}, output/f'{name}_diag.pt')
            print(f'=== END {task}: 50 folds; baseline fold-mean AUROC {sum(values)/50:.6f}', flush=True)
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error'] = repr(exc)
        raise
    finally:
        manifest['seconds'] = time.monotonic()-start
        manifest['gpu_hours_reserved'] = sum(j['seconds'] for j in manifest['jobs'])/3600
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
