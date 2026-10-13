"""Generates colab/run.ipynb (kept as code so the notebook stays reviewable in diffs)."""
import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# 2048-konsey — Colab A100 hattı (Faz 3–5)

Bütün gerçek eğitim burada koşar; yerel bilgisayar yalnızca geliştirme ve deneme içindir.
Sırayla çalıştırın. **Her iş kaldığı yerden devam eder**: bağlantı koparsa runtime'ı yeniden bağlayıp 1–4. hücreleri ve kaldığınız hücreyi tekrar çalıştırın.

GPU'nun tamamı kullanılır: bir ağ binlerce oyunu aynı anda oynar (toplu TD güncellemesi), aynı anda birçok ağ eğitilir (ablasyon kolları, final kopyaları, PBT nüfusu). Nüfus/kopya sayıları GPU belleğine göre otomatik seçilir. Boşta kalan 12 vCPU, GPU işleri sürerken CPU işlerini (damıtma, oylamalı konsey) koşar.

Gerekli: Colab Secrets → `GITHUB_TOKEN` (fine-grained, yalnızca bu repo, *Contents: read-only*).""")

md("## 1. Drive + yardımcılar")
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
        raise RuntimeError(f'komut basarisiz ({p.returncode}): {cmd}')""")

md("## 2. Donanım (çıktıyı PLAN.md'ye yazın; Drive'a da kaydedilir)")
code("""hw = subprocess.run('echo "nproc: $(nproc)"; free -h; nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv; g++ --version | head -1; nvcc --version | tail -1',
                    shell=True, capture_output=True, text=True).stdout
print(hw); open(f'{RUNS}/donanim.txt', 'w').write(hw)
GPU_FREE_GB = float(subprocess.run('nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits', shell=True,
                                   capture_output=True, text=True).stdout.split()[0]) / 1024
NPROC = os.cpu_count()
print(f'GPU bos: {GPU_FREE_GB:.1f} GB, CPU: {NPROC}')""")

md("## 3. Kod (token yalnızca Secrets'tan okunur)")
code("""from google.colab import userdata
url = f"https://x-access-token:{userdata.get('GITHUB_TOKEN')}@github.com/mikailustuner/2048-konsey.git"
if not os.path.isdir(REPO):
    subprocess.run(['git', 'clone', '-q', url, REPO], check=True)
else:
    subprocess.run(['git', '-C', REPO, 'pull', '-q', url, 'main'], check=True)
print(subprocess.run(['git', '-C', REPO, 'log', '--oneline', '-1'], capture_output=True, text=True).stdout)""")

md("## 4. Derle + test (CUDA; `-march=native` ve GPU mimarisi bu makineye göre)")
code("""sh('pip -q install cmake ninja')
sh('cmake -S . -B build -G Ninja -DUSE_CUDA=ON > /dev/null && cmake --build build')
sh('./build/tests | tail -3')
sh('./build/gpu_tests | tail -4')   # GPU motoru/degerleri CPU ile bit-bit ayni olmali""")

md("""## 5. Bütçeler

Oyun sayıları ağ başınadır. Kalibrasyon hücresi (6) bu GPU'daki hızı ölçüp süreleri tahmin eder; gerekirse burada değiştirip yeniden çalıştırın. Bütçeler büyütülebilir: işler checkpoint'ten devam eder.""")
code("""GAMES_ABLATION = 10_000_000   # Faz 3: her ablasyon kolu
GAMES_FINAL    = 40_000_000   # Faz 3: final yapılandırma, kopya başına
PBT_GENS, PBT_TRAIN, PBT_TOURNEY = 30, 300_000, 3000   # Faz 4: nesil, nesil başı eğitim, turnuva oyunu
EVAL_GAMES     = 10_000       # her değerlendirme (ortak tohumlar 1.000.000+)
DISTILL_GAMES  = 300_000      # Faz 5 damıtma (CPU, GPU işleriyle paralel)

# Bellek (ağırlık + TC + toplu güncelleme tamponları), GB:
MEM = {'4x6': 0.27, '4x6_tc': 0.81, '8x6_ms_tc': 4.83}
MEM_BATCH = {'4x6': 0.54, '8x6_ms': 3.22}
mem = lambda net, tc_ms: MEM[tc_ms] + MEM_BATCH[net]
FINAL_REPLICAS = max(1, min(8, int((GPU_FREE_GB - 3) // mem('8x6_ms', '8x6_ms_tc'))))
PBT_POP = max(4, min(32, int((GPU_FREE_GB - 3) // mem('4x6', '4x6_tc')) // 4 * 4))
print(f'final kopya: {FINAL_REPLICAS}, PBT nufusu: {PBT_POP}')""")

md("## 6. Kalibrasyon (~3 dk): bu GPU'da hız")
code("""sh(f'./build/train --gpu --net 4x6 --tc --alpha 1.0 --games 300000 --log-every 100000 --out /tmp/kalibrasyon_{int(time.time())} --ckpt-sec 100000')""")

md("""## 7. Faz 3a — ablasyon (5 kol aynı anda, tek GPU süreci)

Teknikler sırayla eklenir: A düz TD → B +OI → C +TC → D +çok aşama → E 8×6. TD α=0,1; TC α=1,0 (Jaśkowski/Guei); OI = 320k (Guei 2021).""")
code("""jobs = f'''name=A_td net=4x6 alpha=0.1 games={GAMES_ABLATION}
name=B_otd net=4x6 alpha=0.1 oi=320000 games={GAMES_ABLATION}
name=C_otc net=4x6 tc=1 alpha=1.0 oi=320000 games={GAMES_ABLATION}
name=D_otc_ms net=4x6 tc=1 alpha=1.0 oi=320000 stages=14;14,13 games={GAMES_ABLATION}
name=E_8x6 net=8x6 tc=1 alpha=1.0 oi=320000 stages=14;14,13 games={GAMES_ABLATION}
'''
os.makedirs(f'{RUNS}/faz3', exist_ok=True)
open(f'{RUNS}/faz3/ablation.jobs', 'w').write(jobs)
sh(f'./build/train_many --jobs {RUNS}/faz3/ablation.jobs --dir {RUNS}/faz3 --run-id faz3 --faz 3 --log-every 250000 --ckpt-sec 1800')""")

md("### Faz 3a değerlendirme (1-ply ve 2-ply, ortak tohumlar, GPU)")
code("""def gpu_eval(net, name, depth, out, agent=None, extra=''):
    agent = agent or ('greedy' if depth == 1 else 'expectimax')
    if os.path.exists(f'{out}/log.jsonl') and f'"ajan_id":"{name}-d{depth}"' in open(f'{out}/log.jsonl').read():
        print('zaten var:', name, depth); return
    sh(f'./build/eval --gpu --agent {agent} --depth {depth} --net {net} --games {EVAL_GAMES} --run-id {os.path.basename(out)} '
       f'--faz {os.path.basename(out)[3:]} --ajan-id {name}-d{depth} --out {out} --per-game {out}/games.jsonl {extra}')

for arm in ['A_td', 'B_otd', 'C_otc', 'D_otc_ms', 'E_8x6']:
    for d in (1, 2):
        gpu_eval(f'{RUNS}/faz3/{arm}/net.bin', arm, d, f'{RUNS}/faz3')
sh(f'python3 analysis/report.py {RUNS} | head -40')""")

md("""## 8. Faz 3b — final yapılandırma, uzun eğitim (birden çok kopya)

Varsayılan: E (8×6, OI, TC, çok aşama). Ablasyon tablosu başka bir kolu gösterirse `FINAL` satırını değiştirin. Kopyalar farklı tohumlarla eğitilir; Faz 5'te konsey üyeleri olurlar.""")
code("""FINAL = 'net=8x6 tc=1 alpha=1.0 oi=320000 stages=14;14,13'
jobs = ''.join(f'name=final_r{i} {FINAL} games={GAMES_FINAL} seed={(100 + i) << 40}\\n' for i in range(1, FINAL_REPLICAS + 1))
open(f'{RUNS}/faz3/final.jobs', 'w').write(jobs)
sh(f'./build/train_many --jobs {RUNS}/faz3/final.jobs --dir {RUNS}/faz3 --run-id faz3-final --faz 3 --log-every 1000000 --ckpt-sec 1800')
for i in range(1, FINAL_REPLICAS + 1):
    for d in (1, 2):
        gpu_eval(f'{RUNS}/faz3/final_r{i}/net.bin', f'final_r{i}', d, f'{RUNS}/faz3')
sh(f"python3 analysis/report.py {RUNS} | sed -n '/Faz 3/,/Faz 4/p'")""")

md("""## 9. Faz 4 — evrim (PBT, nüfus GPU'da)

Genom: TC meta adımı α ∈ [0,3; 2,0] (log-uniform), OI ∈ [0; 640k]. Her nesil: eğitim → ortak tohumlu turnuva → alt %25 üst %25'in kopyası + mutasyon. Soy ağacı `lineage.jsonl`.""")
code("""sh(f'./build/evolve --gpu --dir {RUNS}/faz4 --population {PBT_POP} --generations {PBT_GENS} --train-games {PBT_TRAIN} '
   f'--tourney-games {PBT_TOURNEY} --net 4x6 --tc --alpha-lo 0.3 --alpha-hi 2.0 --lambda-hi 0 --oi-hi 640000 '
   f'--save-every 5 --run-id faz4')
sh(f'python3 analysis/plot.py evolve {RUNS}/faz4/generations.jsonl --out {RUNS}/faz4/evrim.png')""")

md("""### Faz 4: kazanan ayarlarla 8×6 çok aşamalı yeniden eğitim vs elle seçilen

Elle seçilen = Faz 3 final kopyası 1 (aynı ağ, aynı oyun sayısı). Ölçüt: evrimle bulunan ≥ elle seçilen.""")
code("""pop = [l.split() for l in open(f'{RUNS}/faz4/population.txt').read().splitlines()[1:]]
best = max(pop, key=lambda r: float(r[6]))   # uid ebeveyn alpha lambda oi oyun fitness
alpha, oi = float(best[2]), float(best[4])
print(f'kazanan: a{best[0]} alpha={alpha:.4f} oi={oi:.0f} turnuva={float(best[6]):,.0f}')
open(f'{RUNS}/faz4/winner.jobs', 'w').write(
    f'name=evolved net=8x6 tc=1 alpha={alpha} oi={oi} stages=14;14,13 games={GAMES_FINAL} seed={101 << 40}\\n')
sh(f'./build/train_many --jobs {RUNS}/faz4/winner.jobs --dir {RUNS}/faz4 --run-id faz4-winner --faz 4 --log-every 1000000 --ckpt-sec 1800')
for d in (1, 2):
    gpu_eval(f'{RUNS}/faz4/evolved/net.bin', 'evolved', d, f'{RUNS}/faz4')
    gpu_eval(f'{RUNS}/faz3/final_r1/net.bin', 'manual', d, f'{RUNS}/faz4')
sh(f"python3 analysis/report.py {RUNS} | sed -n '/Faz 4/,/Faz 5/p'")""")

md("""## 10. Faz 5 — konsey

Üyeler: Faz 3 final kopyaları + Faz 4 evrilmiş 8×6. GPU'da ortalama ve ağırlıklı (tek başına 2-ply skorlarıyla) konsey, en iyi tek ajanla **aynı tohumlarda**. Oylamalı konsey ve damıtma CPU'da (12 vCPU, GPU işiyle paralel).""")
code("""F5 = f'{RUNS}/faz5'; os.makedirs(F5, exist_ok=True)
members = [f'{RUNS}/faz3/final_r{i}/net.bin' for i in range(1, FINAL_REPLICAS + 1)] + [f'{RUNS}/faz4/evolved/net.bin']
members = [m for m in members if os.path.exists(m)][:8]
names = [m.split('/')[-2] for m in members]
# Damıtma (CPU) arka planda: konseyin değerlerini tek bir 8x6 ağa öğretir.
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
# Oylamalı konsey: CPU, 1-ply (her üye kendi hamlesini seçer)
if 'council_vote-d1' not in open(f'{F5}/log.jsonl').read():
    sh(f'./build/eval --agent council --combine vote --depth 1 --net {",".join(members)} --games 2000 --threads {NPROC} '
       f'--run-id faz5 --faz 5 --ajan-id council_vote-d1 --out {F5} --per-game {F5}/games.jsonl')""")

md("### Faz 5: damıtılmış öğrenci (damıtma bitince)")
code("""distill.wait()
sh(f'tail -3 {F5}/distill.log')
for d in (1, 2):
    gpu_eval(f'{F5}/student.bin', 'student', d, F5)
sh(f"python3 analysis/report.py {RUNS} | sed -n '/Faz 5/,$p'")""")

md("## 11. Rapor + grafikler (Drive: `runs/RAPOR.md`)")
code("""import glob
curves = ' '.join(glob.glob(f'{RUNS}/faz3/*/train.jsonl'))
if curves:
    sh(f'python3 analysis/plot.py curve {curves} --out {RUNS}/faz3/egriler.png')
sh(f'python3 analysis/report.py {RUNS}')""")

md("## 12. Final modeli yerel arayüze gönder\n\nEn iyi tek ağ ve damıtılmış öğrenci `MyDrive/2048/final/`'a kopyalanır. Yerelde: `tools/fetch_model.sh` → `gui/launch.sh` → modeli seç → Başlat.")
code("""import shutil
os.makedirs('/content/drive/MyDrive/2048/final', exist_ok=True)
rows = [json.loads(l) for l in open(f'{F5}/log.jsonl')]
best = max((r for r in rows if r['ajan_id'].startswith('single') and r['ajan_id'].endswith('-d2')), key=lambda r: r['ort_skor'])
src = [m for m, n in zip(members, names) if best['ajan_id'] == f'single_{n}-d2'][0]
shutil.copy(src, '/content/drive/MyDrive/2048/final/en_iyi_tek.bin')
if os.path.exists(f'{F5}/student.bin'):
    shutil.copy(f'{F5}/student.bin', '/content/drive/MyDrive/2048/final/konsey_ogrencisi.bin')
sh('ls -lh /content/drive/MyDrive/2048/final')""")

md("## 13. Kapat (compute unit harcamamak için)")
code("""from google.colab import runtime
runtime.unassign()""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 0}
Path(__file__).with_name("run.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("colab/run.ipynb yazildi:", len(cells), "hucre")
