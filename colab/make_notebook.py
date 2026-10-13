"""Generates colab/run.ipynb (kept as code so the notebook stays reviewable in diffs)."""
import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# 2048-konsey — Colab A100 pipeline (phases 3–6)

All real training runs here; the local machine is for development and small trials only.
Run the cells in order. **Every job resumes where it stopped**: after a disconnect, reconnect the runtime and run cells 1–4 again, then the cell you were in.

The GPU is used to full capacity: one network plays thousands of games at once (batched TD updates), and many networks train at the same time (ablation arms, final replicas, the PBT population). Population and replica counts are chosen from the free GPU memory. The CPU cores run the CPU jobs (distillation, voting council, deep search, endgame tables).

Runtime: **A100 GPU + High-RAM**. Results go to Google Drive: `MyDrive/2048/runs/`.""")

md("## 1. Drive + helpers")
code("""from google.colab import drive
drive.mount('/content/drive')
import os, subprocess, json, math, time
REPO = '/content/2048-konsey'
RUNS = '/content/drive/MyDrive/2048/runs'
os.makedirs(RUNS, exist_ok=True)

def sh(cmd, check=True, bg=None):
    \"\"\"Runs a shell command in the repo and streams its output. bg='log path' starts it in the background.\"\"\"
    print('$', cmd, flush=True)
    if bg:
        return subprocess.Popen(f'nohup {cmd} > {bg} 2>&1', shell=True, cwd=REPO)
    p = subprocess.Popen(cmd, shell=True, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    if p.wait() and check:
        raise RuntimeError(f'command failed ({p.returncode}): {cmd}')""")

md("## 2. Hardware (also saved to Drive; copy the numbers into PLAN.md)")
code("""hw = subprocess.run('echo "nproc: $(nproc)"; free -h; nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv; g++ --version | head -1; nvcc --version | tail -1',
                    shell=True, capture_output=True, text=True).stdout
print(hw); open(f'{RUNS}/hardware.txt', 'w').write(hw)
GPU_FREE_GB = float(subprocess.run('nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits', shell=True,
                                   capture_output=True, text=True).stdout.split()[0]) / 1024
NPROC = os.cpu_count()
print(f'GPU free: {GPU_FREE_GB:.1f} GB, CPUs: {NPROC}')""")

md("## 3. Code (public repository)")
code("""url = 'https://github.com/mikailustuner/2048-konsey.git'
if not os.path.isdir(REPO):
    subprocess.run(['git', 'clone', '-q', url, REPO], check=True)
else:
    subprocess.run(['git', '-C', REPO, 'pull', '-q', 'origin', 'main'], check=True)
print(subprocess.run(['git', '-C', REPO, 'log', '--oneline', '-1'], capture_output=True, text=True).stdout)""")

md("## 4. Build + test (CUDA; `-march=native` and the GPU architecture of this machine)")
code("""sh('pip -q install cmake ninja')
sh('cmake -S . -B build -G Ninja -DUSE_CUDA=ON > /dev/null && cmake --build build')
sh('./build/tests | tail -3')
sh('./build/gpu_tests | tail -4')   # the GPU engine/values must match the CPU bit for bit""")

md("""## 5. Budgets

Game counts are per network. The calibration cell (6) measures the speed on this GPU; adjust here and re-run if needed. Budgets can be raised later: jobs resume from their checkpoints.""")
code("""GAMES_ABLATION = 10_000_000   # Faz 3: each ablation arm
GAMES_FINAL    = 40_000_000   # Faz 3: final configuration, per replica
PBT_GENS, PBT_TRAIN, PBT_TOURNEY = 30, 300_000, 3000   # Faz 4: generations, training games per generation, tournament games
EVAL_GAMES     = 10_000       # every evaluation (common seeds 1,000,000+)
DISTILL_GAMES  = 300_000      # Faz 5 distillation (CPU, in parallel with the GPU jobs)
SEARCH6_GAMES  = 1_000        # Faz 6a: deep search (CPU; depth 3 ≈ 4 ms per move and core)
HYBRID_GAMES   = 1_000        # Faz 6b: hybrid evaluation
# Faz 6b: candidate patterns ("locked cells:target exponent"). The most frequent ones are built within the cap.
TABLE_CANDIDATES = ['0,1,2,3:7', '0,1,2,3:8', '0,1,2,3:9', '0,1,2,3,7:7', '0,1,2,3,7:8', '0,1,2,3,7:9',
                    '0,1,2,3,4:8', '0,1,2,3,7,11:8', '0,1,2,3,4,8:9', '0,1,2,3,4,5:9', '0,1,2,3,7,6:9']
TABLE_MAX_STATES = 6_000_000_000   # ~72 GB on disk (12 bytes per position); match the /content disk
TABLE_MAX_COUNT  = 3
TABLE_MIN_COVERAGE = 1.0         # % of the network's positions a pattern must fit
TABLE_MARGINS    = [0.0, 0.1]      # table/search switch point (0 = the table always plays, 1 = never)

# Memory per network (weights + TC + batched-update buffers), GB:
MEM = {'4x6': 0.27, '4x6_tc': 0.81, '8x6_ms_tc': 4.83}
MEM_BATCH = {'4x6': 0.54, '8x6_ms': 3.22}
mem = lambda net, tc_ms: MEM[tc_ms] + MEM_BATCH[net]
FINAL_REPLICAS = max(1, min(8, int((GPU_FREE_GB - 3) // mem('8x6_ms', '8x6_ms_tc'))))
PBT_POP = max(4, min(32, int((GPU_FREE_GB - 3) // mem('4x6', '4x6_tc')) // 4 * 4))
print(f'final replicas: {FINAL_REPLICAS}, PBT population: {PBT_POP}')""")

md("## 6. Calibration (~3 min): speed on this GPU")
code("""sh(f'./build/train --gpu --net 4x6 --tc --alpha 1.0 --games 300000 --log-every 100000 --out /tmp/calibration_{int(time.time())} --ckpt-sec 100000')""")

md("""## 7. Faz 3a — ablation (5 arms at once, one GPU process)

Techniques are added in order: A plain TD → B +OI → C +TC → D +multi-stage → E 8×6. TD α = 0.1; TC α = 1.0 (Jaśkowski/Guei); OI = 320k (Guei 2021).""")
code("""jobs = f'''name=A_td net=4x6 alpha=0.1 games={GAMES_ABLATION}
name=B_otd net=4x6 alpha=0.1 oi=320000 games={GAMES_ABLATION}
name=C_otc net=4x6 tc=1 alpha=1.0 oi=320000 games={GAMES_ABLATION}
name=D_otc_ms net=4x6 tc=1 alpha=1.0 oi=320000 stages=14;14,13 games={GAMES_ABLATION}
name=E_8x6 net=8x6 tc=1 alpha=1.0 oi=320000 stages=14;14,13 games={GAMES_ABLATION}
'''
os.makedirs(f'{RUNS}/faz3', exist_ok=True)
open(f'{RUNS}/faz3/ablation.jobs', 'w').write(jobs)
sh(f'./build/train_many --jobs {RUNS}/faz3/ablation.jobs --dir {RUNS}/faz3 --run-id faz3 --faz 3 --log-every 250000 --ckpt-sec 1800')""")

md("### Faz 3a evaluation (1-ply and 2-ply, common seeds, GPU)")
code("""def gpu_eval(net, name, depth, out, agent=None, extra=''):
    agent = agent or ('greedy' if depth == 1 else 'expectimax')
    if os.path.exists(f'{out}/log.jsonl') and f'"ajan_id":"{name}-d{depth}"' in open(f'{out}/log.jsonl').read():
        print('already done:', name, depth); return
    sh(f'./build/eval --gpu --agent {agent} --depth {depth} --net {net} --games {EVAL_GAMES} --run-id {os.path.basename(out)} '
       f'--faz {os.path.basename(out)[3:]} --ajan-id {name}-d{depth} --out {out} --per-game {out}/games.jsonl {extra}')

for arm in ['A_td', 'B_otd', 'C_otc', 'D_otc_ms', 'E_8x6']:
    for d in (1, 2):
        gpu_eval(f'{RUNS}/faz3/{arm}/net.bin', arm, d, f'{RUNS}/faz3')
sh(f'python3 analysis/report.py {RUNS} | head -40')""")

md("""## 8. Faz 3b — final configuration, long training (several replicas)

Default: E (8×6, OI, TC, multi-stage). Change the `FINAL` line if the ablation table points to another arm. Replicas train with different seeds and become the council members of Faz 5.""")
code("""FINAL = 'net=8x6 tc=1 alpha=1.0 oi=320000 stages=14;14,13'
jobs = ''.join(f'name=final_r{i} {FINAL} games={GAMES_FINAL} seed={(100 + i) << 40}\\n' for i in range(1, FINAL_REPLICAS + 1))
open(f'{RUNS}/faz3/final.jobs', 'w').write(jobs)
sh(f'./build/train_many --jobs {RUNS}/faz3/final.jobs --dir {RUNS}/faz3 --run-id faz3-final --faz 3 --log-every 1000000 --ckpt-sec 1800')
for i in range(1, FINAL_REPLICAS + 1):
    for d in (1, 2):
        gpu_eval(f'{RUNS}/faz3/final_r{i}/net.bin', f'final_r{i}', d, f'{RUNS}/faz3')
sh(f"python3 analysis/report.py {RUNS} | sed -n '/Faz 3/,/Faz 4/p'")""")

md("""## 9. Faz 4 — evolution (PBT, population on the GPU)

Genome: TC meta step α ∈ [0.3, 2.0] (log-uniform), OI ∈ [0, 640k]. Each generation: training → common-seed tournament → bottom 25% replaced by mutated copies of the top 25%. Lineage in `lineage.jsonl`.""")
code("""sh(f'./build/evolve --gpu --dir {RUNS}/faz4 --population {PBT_POP} --generations {PBT_GENS} --train-games {PBT_TRAIN} '
   f'--tourney-games {PBT_TOURNEY} --net 4x6 --tc --alpha-lo 0.3 --alpha-hi 2.0 --lambda-hi 0 --oi-hi 640000 '
   f'--save-every 5 --run-id faz4')
sh(f'python3 analysis/plot.py evolve {RUNS}/faz4/generations.jsonl --out {RUNS}/faz4/evolution.png')""")

md("""### Faz 4: winning settings re-trained on 8×6 multi-stage vs hand-picked

Hand-picked = Faz 3 final replica 1 (same network, same number of games). Criterion: evolved ≥ hand-picked.""")
code("""pop = [l.split() for l in open(f'{RUNS}/faz4/population.txt').read().splitlines()[1:]]
best = max(pop, key=lambda r: float(r[6]))   # uid parent alpha lambda oi games fitness
alpha, oi = float(best[2]), float(best[4])
print(f'winner: a{best[0]} alpha={alpha:.4f} oi={oi:.0f} tournament={float(best[6]):,.0f}')
open(f'{RUNS}/faz4/winner.jobs', 'w').write(
    f'name=evolved net=8x6 tc=1 alpha={alpha} oi={oi} stages=14;14,13 games={GAMES_FINAL} seed={101 << 40}\\n')
sh(f'./build/train_many --jobs {RUNS}/faz4/winner.jobs --dir {RUNS}/faz4 --run-id faz4-winner --faz 4 --log-every 1000000 --ckpt-sec 1800')
for d in (1, 2):
    gpu_eval(f'{RUNS}/faz4/evolved/net.bin', 'evolved', d, f'{RUNS}/faz4')
    gpu_eval(f'{RUNS}/faz3/final_r1/net.bin', 'manual', d, f'{RUNS}/faz4')
sh(f"python3 analysis/report.py {RUNS} | sed -n '/Faz 4/,/Faz 5/p'")""")

md("""## 10. Faz 5 — council

Members: the Faz 3 final replicas + the Faz 4 evolved 8×6. Mean and weighted (by solo 2-ply score) councils on the GPU against the best single agent **on the same seeds**. Voting council and distillation on the CPU (in parallel with the GPU work).""")
code("""F5 = f'{RUNS}/faz5'; os.makedirs(F5, exist_ok=True)
members = [f'{RUNS}/faz3/final_r{i}/net.bin' for i in range(1, FINAL_REPLICAS + 1)] + [f'{RUNS}/faz4/evolved/net.bin']
members = [m for m in members if os.path.exists(m)][:8]
names = [m.split('/')[-2] for m in members]
# Distillation (CPU) in the background: teaches the council's values to one 8x6 network.
distill = sh(f'./build/distill --teachers {",".join(members)} --net 8x6 --games {DISTILL_GAMES} --threads {NPROC} '
             f'--out {F5}/student.bin', bg=f'{F5}/distill.log')
for d in (1, 2):
    for m, n in zip(members, names):
        gpu_eval(m, f'single_{n}', d, F5)
rows = [json.loads(l) for l in open(f'{F5}/log.jsonl')]
score = {r['ajan_id']: r['ort_skor'] for r in rows}
weights = ','.join(f"{score[f'single_{n}-d2']:.0f}" for n in names)
for d in (1, 2):
    gpu_eval(','.join(members), 'council_mean', d, F5, agent='council')
    gpu_eval(','.join(members), 'council_weighted', d, F5, agent='council', extra=f'--combine weighted --weights {weights}')
# Voting council: CPU, 1-ply (every member picks its own move)
if 'council_vote-d1' not in open(f'{F5}/log.jsonl').read():
    sh(f'./build/eval --agent council --combine vote --depth 1 --net {",".join(members)} --games 2000 --threads {NPROC} '
       f'--run-id faz5 --faz 5 --ajan-id council_vote-d1 --out {F5} --per-game {F5}/games.jsonl')""")

md("### Faz 5: distilled student (once the distillation finishes)")
code("""distill.wait()
sh(f'tail -3 {F5}/distill.log')
for d in (1, 2):
    gpu_eval(f'{F5}/student.bin', 'student', d, F5)
sh(f"python3 analysis/report.py {RUNS} | sed -n '/Faz 5/,$p'")""")

md("""## 11. Faz 6a — deep search without tables (the "pure RL" number)

The best single network on all CPU cores (exact path: 65536 counts too): depth 3, and depth 4 under a time budget (`--time-ms`) + probability cut, both with tile-downgrading once 32768 appears. Criterion: 32768 ≥ 50%.""")
code("""F6 = f'{RUNS}/faz6'; os.makedirs(F6, exist_ok=True)
rows = [json.loads(l) for l in open(f'{F5}/log.jsonl')]
best = max((r for r in rows if r['ajan_id'].startswith('single') and r['ajan_id'].endswith('-d2')), key=lambda r: r['ort_skor'])
BEST_NET = [m for m, n in zip(members, names) if best['ajan_id'] == f'single_{n}-d2'][0]
print('best single network:', BEST_NET)
SEARCH6 = {'search_d3': '--depth 3 --downgrade 15', 'search_d4t': '--depth 4 --time-ms 50 --prob-cut 0.0001 --downgrade 15'}
def cpu_eval(name, args, games):
    if os.path.exists(f'{F6}/log.jsonl') and f'"ajan_id":"{name}"' in open(f'{F6}/log.jsonl').read():
        print('already done:', name); return
    sh(f'./build/eval {args} --games {games} --threads {NPROC} --run-id faz6 --faz 6 --ajan-id {name} --out {F6} --per-game {F6}/games.jsonl')
for name, a in SEARCH6.items():
    cpu_eval(name, f'--agent expectimax --net {BEST_NET} {a}', SEARCH6_GAMES)""")

md("""## 12. Faz 6b — endgame tables + hybrid

Tables are written to the Colab disk (`/content/tables`; too large for Drive). A pattern is `locked cells:target`: the locked cells hold distinct big tiles whose smallest is 2^target, and the table builds the tile 2^target in the free cells (the one that merges into that chain end); lookups try all 8 orientations. Patterns are chosen by measurement: how often each fits the best network's own games (`--coverage`); the most frequent ones are built under a position cap. Local measurements: `0,1,2,3,4,5:9` 139M positions / 52 s / 1.6 GB; patterns with 12 free cells reach billions. Criterion: hybrid 32768 ≥ 80% and 65536 > 3.5% (1000+ games).""")
code("""TABLES = '/content/tables'
def out(cmd):
    return subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True).stdout
# 1) How often does each candidate pattern occur in the best network's own games (any orientation)?
cov = {}
for spec in TABLE_CANDIDATES:
    o = out(f'./build/tablebase --coverage 50 --net {BEST_NET} --pattern "{spec}"')
    print(o.strip())
    cov[spec] = float(o.split('(')[1].split('%')[0]) if '(' in o else 0.0
# 2) Build the most frequent ones that fit (the position cap aborts oversized patterns).
built = []
for spec in sorted(cov, key=cov.get, reverse=True):
    if cov[spec] < TABLE_MIN_COVERAGE or len(built) == TABLE_MAX_COUNT:
        continue
    d = f"{TABLES}/{spec.replace(',', '_').replace(':', '-')}"
    if not os.path.exists(f'{d}/meta.txt'):
        sh(f'./build/tablebase --build --pattern "{spec}" --out {d} --threads {NPROC} --max-states {TABLE_MAX_STATES}', check=False)
    if os.path.exists(f'{d}/meta.txt'):
        built.append(d)
    else:
        sh(f'rm -rf {d}')   # too large: remove the partial layers
print('tables:', built)
assert built, 'no table could be built: check TABLE_CANDIDATES / TABLE_MAX_STATES'
# Switch point: the table overrides the search only if the search's move is `margin` less likely to succeed.
for m in TABLE_MARGINS:
    cpu_eval(f'hybrid_d3_m{m}', f'--agent hybrid --net {BEST_NET} --tables {",".join(built)} --depth 3 --downgrade 15 '
                                f'--table-margin {m}', HYBRID_GAMES)
sh(f"python3 analysis/report.py {RUNS} | sed -n '/Faz 6/,$p'")""")

md("## 13. Report + plots (Drive: `runs/REPORT.md`)")
code("""import glob
curves = ' '.join(glob.glob(f'{RUNS}/faz3/*/train.jsonl'))
if curves:
    sh(f'python3 analysis/plot.py curve {curves} --out {RUNS}/faz3/curves.png')
sh(f'python3 analysis/report.py {RUNS}')""")

md("## 14. Send the final model to the local UI\n\nThe best single network and the distilled student are copied to `MyDrive/2048/final/`. Locally: `tools/fetch_model.sh` → `gui/launch.sh` → pick the model → Start.")
code("""import shutil
os.makedirs('/content/drive/MyDrive/2048/final', exist_ok=True)
shutil.copy(BEST_NET, '/content/drive/MyDrive/2048/final/best_single.bin')
if os.path.exists(f'{F5}/student.bin'):
    shutil.copy(f'{F5}/student.bin', '/content/drive/MyDrive/2048/final/council_student.bin')
sh('ls -lh /content/drive/MyDrive/2048/final')""")

md("## 15. Shut down (stop spending compute units)")
code("""from google.colab import runtime
runtime.unassign()""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 0}
Path(__file__).with_name("run.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote colab/run.ipynb:", len(cells), "cells")
