"""Runs colab/run.ipynb locally with tiny budgets, to check the whole pipeline before spending
Colab credits. Colab-only cells (Drive, git, pip, runtime) are skipped; CUDA binaries come from
build-cuda/ (see README), networks are shrunk to the small 4x5 preset so they fit a 4 GB GPU.

    python3 colab/dryrun.py [RUNS_DIR=runs/dryrun]
"""
import json
import os
import subprocess
import sys

RUNS = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else 'runs/dryrun')
SMALL = [('net=4x6', 'net=4x5'), ('net=8x6', 'net=4x5'), ('--net 4x6', '--net 4x5'), ('--net 8x6', '--net 4x5'),
         ('stages=14;14,13', 'stages=9;9,10')]


def sh(cmd, check=True, bg=None):
    cmd = cmd.replace('./build/', './build-cuda/').replace('python3 analysis/plot.py', 'uv run -q --project analysis analysis/plot.py')
    for a, b in SMALL:
        cmd = cmd.replace(a, b)
    print('$', cmd, flush=True)
    if bg:
        return subprocess.Popen(f'nohup {cmd} > {bg} 2>&1', shell=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = list(p.stdout)
    if p.wait() and check:
        print(''.join(lines[-30:]))
        raise RuntimeError(f'command failed ({p.returncode}): {cmd}')
    print(''.join(lines[-6:]), end='')


def main():
    os.makedirs(RUNS, exist_ok=True)
    nb = json.load(open('colab/run.ipynb'))
    G = {'sh': sh, 'RUNS': RUNS, 'REPO': os.getcwd()}
    exec('import os, subprocess, json, math, time', G)
    for i, c in enumerate(c for c in nb['cells'] if c['cell_type'] == 'code'):
        src = ''.join(c['source'])
        if any(k in src for k in ('google.colab', 'pip -q install', 'git', 'MyDrive/2048/final')) and 'def sh' not in src:
            print(f'--- cell {i}: skipped (Colab only)')
            continue
        if 'def sh(' in src:  # cell 1: Drive + helpers -> local helpers above
            print(f'--- cell {i}: skipped (local helpers)')
            continue
        for a, b in SMALL:
            src = src.replace(a, b)
        src = src.replace("'/content/tables'", f"'{RUNS}/tables'").replace('--games 300000', '--games 20000')
        src = src.replace('--log-every 100000', '--log-every 10000').replace('--coverage 50', '--coverage 10')
        print(f'--- cell {i} ---', flush=True)
        exec(src, G)
        if src.startswith('GAMES_ABLATION'):
            G.update(GAMES_ABLATION=20000, GAMES_FINAL=20000, PBT_GENS=3, PBT_TRAIN=5000, PBT_TOURNEY=300, EVAL_GAMES=300,
                     DISTILL_GAMES=200, SEARCH6_GAMES=30, HYBRID_GAMES=30, FINAL_REPLICAS=2, PBT_POP=4,
                     TABLE_CANDIDATES=['0,1,2,3:5', '0,1,2,3:6', '3,7,8,9,10,11,12,13,14,15:6'],
                     TABLE_MAX_STATES=50_000_000, TABLE_MAX_COUNT=2, TABLE_MARGINS=[0.0], TABLE_MIN_COVERAGE=0.1)
            print('budgets shrunk for the dry run')
    print('DRY RUN COMPLETE')


if __name__ == '__main__':
    main()
