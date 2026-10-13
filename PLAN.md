# 2048 Record Speedrun — Implementation Plan

Source: a concept document (PDF) that this file turns into executable phases.
Rule: no phase starts before the previous one is **done**; "done" = its success criterion was measured and recorded.

---

## ★ Working environment: Google Colab — A100 + High-RAM (decision: 2026-10-13)

The code is written for this environment. Development and tests run locally; **every real training/compute job runs on Colab** and uses its full capacity (GPU kernels, large populations, large RAM). Local runs are small trials only.

| | Local (development) | Colab A100 High-RAM (heavy jobs) |
|---|---|---|
| CPU | Ryzen 5 1500X, 8 threads | Xeon, ~12 vCPU (measured by notebook cell 2) |
| RAM | 16 GB | ~80 GB (measured by notebook cell 2) |
| GPU | GTX 1050 Ti 4 GB | A100 40 GB or 80 GB (measured by notebook cell 2) |
| Compiler | g++ 16 | Ubuntu's g++ 11 + CUDA `nvcc` |

**Rules reflected in the code:**
1. **C++20** (not C++23) so Colab's older g++ builds it. `-std=c++20` locally as well.
2. **Rebuild on every machine.** `-march=native` binaries are machine specific; Colab rebuilds at the start of every session (seconds).
3. **CUDA is an optional build flag** (`-DUSE_CUDA=ON`). The CPU path is always complete and is the reference; the GPU path is tested to match it. The local 1050 Ti runs the CUDA tests; a Docker image (Ubuntu 22.04 + CUDA 12.6 + g++ 11) mirrors Colab's toolchain.
4. **Persistent data lives on Google Drive.** The Colab disk is wiped with the session: checkpoints, logs, replays go to `/content/drive/MyDrive/2048/runs/`. Every long job checkpoints periodically and **resumes where it stopped** (disconnects, credits running out). Large endgame tables stay on the Colab disk (`/content/tables`).
5. **Code transfer: public GitHub repository.** Flow: write + test locally → `git push` → Colab notebook (`colab/run.ipynb`): mount Drive → clone/pull → build → run tests → start the job.
6. **Compute credits are expensive.** Only jobs that already ran at small scale locally run on Colab. No idle sessions; the notebook releases the runtime at the end.
7. **Memory budgets are sized for ~80 GB RAM / 40–80 GB GPU**: populations and tables can be much larger than locally.

---

## 0. Corrections to the concept document (know these first)

| Topic | Document says | Reality (checked 2026-10-13) | Consequence |
|---|---|---|---|
| Bar | natureofcode/2048-ai = expectimax + n-tuple, 65536 3.5% | A fork of macroxue/2048-ai; **not n-tuple/RL**: hand-written evaluation + huge endgame tables computed by dynamic programming. ~1450 s per game at depth 8, the big version needs 27 GB RAM. | A "search + table" system, not a learning agent. |
| Current SOTA | — | **game-difficulty/2048EndgameTablebase: 65536 8.4%, 32768 86%.** Also endgame-tablebase based. | The absolute record bar is ~2.4× the document's. |
| Best RL | Yeh 2016: 32768 31.75% | **Guei et al. 2021 (OTD): 32768 72%, mean 625,377, 65536 0.02%** (OI + TC + multi-stage + expectimax + tile-downgrading). | The real bar for pure RL. |
| 131072 | "never seen" | Theoretical maximum 131072 (score ~3.93M). The claim that no AI reached it needs checking. | Keep as a target, no promise. |

**Strategic consequence:** there are two different "records":
1. **RL record** (learning agents): beat Guei 2021 — realistic and fits the story.
2. **Absolute record**: beat 8.4% on 65536 — probably needs a non-learning component such as an endgame table (hybrid).

→ **Decision D1: hybrid** (see section 6). The pure-RL number is measured and reported separately.

---

## 1. Hardware reality

| Resource | Value | Good for |
|---|---|---|
| Local CPU | Ryzen 5 1500X, 4 cores / 8 threads (Zen 1) | Engine, tests, small trials |
| Local RAM | 16 GB (~10 GB free) | Small networks and tables |
| Local GPU | GTX 1050 Ti, 4 GB (~2.4 GB free) | CUDA development and GPU tests |
| Colab A100 + High-RAM | ~80 GB RAM, A100 40/80 GB (D2) | All training, GPU evaluation/tournaments, large tables, record runs |

Measured consequences:
- **N-tuple learning is random memory access.** The GPU wins only with thousands of games in flight; the GPU path (Faz 4b) plays one game per warp and applies TD updates in batches (see section 7).
- A 4×6 network is 268 MB (16⁶ floats × 4 tuples), with TC accumulators 805 MB; 8×6 with 3 stages and TC is 4.8 GB. Populations and replicas are sized from the free GPU memory at run time.

---

## 2. Technical stack (Decision D3)

- **Core: C++20** (`-O3 -march=native -ffp-contract=off`), optional CUDA. Engine, n-tuple, TD, search, population loop, tablebase.
- **Build:** CMake (`uv tool install cmake`) + doctest (single header) unit tests.
- **Analysis / plots / report: Python** (stdlib + matplotlib via uv). C++ talks to Python through files (JSONL + binary weight files) — no bindings.
- **Reproducibility:** every game comes from one seed (xoshiro256**); same seed = same tile sequence. All comparisons use **common seeds** (paired tests).
- **Git**, one tag per finished phase (`faz-0`, `faz-1`, …).

Directory layout:
```
engine/      bitboard, move tables, RNG, replays, evaluation harness
ntuple/      network, symmetries, TD/TC/OI, multi-stage
search/      expectimax, transposition table, tile-downgrading
evolve/      population based training (CPU and GPU backends)
council/     ensembles and distillation
tablebase/   endgame tables (Faz 6b)
gpu/         CUDA training and evaluation (Faz 4b)
tools/       CLIs: bench, train, train_many, eval, evolve, distill, play, replay, tablebase
gui/         local web UI: play yourself / watch a model live
colab/       Colab notebook (generated by make_notebook.py)
tests/
analysis/    Python: plots, phase report, replay player
runs/        (not in git) weights, logs, replays, tables
```

---

## 3. Phases

Each phase: **Goal → Work → Success criterion (measurable) → Outputs**.

### Faz 0 — Setup and measurement infrastructure
- git, directory layout, CMake, test framework, `-O3 -march=native`.
- GitHub repository + `colab/run.ipynb` (mount Drive → pull → build → test). The first Colab session measures the hardware (`nproc`, `free -h`, `nvidia-smi`, `g++ --version`).
- Shared **evaluation harness**: N games, fixed seed range, output = mean score + reach rate per tile + **95% confidence interval** (Wilson). Every phase uses it.
- Log schema (JSONL): `run_id, faz, ajan_id, nesil, oyun_sayisi, hamle_sayisi, oranlar{2048..65536}, guven{…}, ort_skor, maks_skor, sure` (field names are Turkish: phase, agent id, generation, games, moves, rates, intervals, mean score, max score, seconds). The video story comes from these logs.
- **Criterion:** tests green locally and on Colab; `eval` produces a report with a random agent and writes it to Drive.

### Faz 1 — Fast engine
- Board = `uint64_t`, 4 bits per cell (log2 of the tile): 0 = empty, 1..15 = 2..32768. **65536 (=16) and 131072 (=17) do not fit**; decided in this phase (see results).
- 65536-entry row tables (left/right moves + row score), up/down through transposition.
- Spawns (90% → 2, 10% → 4), game-over detection, replays: seed + move string.
- **For the hybrid (D1):** the engine API supports a "fixed-cell mask" — lock some cells with big tiles and enumerate/index the small tiles on the rest. The table builder (Faz 6b) builds on it.
- **Correctness:** compared against a slow reference implementation on millions of random boards.
- **Criterion:** single thread **≥ 20M moves/s** (bench tool), random games match the reference 100%, a replay file reproduces its game exactly.

### Faz 2 — Base agent: n-tuple + TD(0)
- 4×6-tuple network (Yeh/Jaśkowski shapes), 8 symmetries, `float` weights.
- TD(0) on **afterstates**, self-play, greedy (1-ply) moves.
- 8 threads sharing the weights (Hogwild, lock-free).
- Checkpoint save/load; interrupted training resumes.
- **Criterion:** 1-ply, 1000 games: **2048 ≥ 90%**. Learning curve plot (games → mean score, 2048 rate).

### Faz 3 — Strengthening
Each technique is added one at a time and **measured before/after on common seeds** (ablation table, also for the video):
1. Optimistic initialisation (OI) — Guei 2021.
2. Temporal coherence (TC) learning rates.
3. Multi-stage TD (stage boundaries e.g. when 16384 and 16384+8192 appear).
4. Network size: 4×6 → 8×6.
5. **A simple expectimax (2-ply) is added now** — as a measuring tool: 1-ply numbers hide the real strength. The fast search comes in Faz 6.
- **Criterion:** **16384 ≥ 50% with 2-ply search**; a significant increase (non-overlapping confidence intervals) reported for each technique at 1-ply.

### Faz 4 — Evolution (population based training, PBT)
- Genome = the TD **settings** (learning rate, OI value, λ, TC parameters, stage boundaries, optionally tuple shapes). Evolution steers the RL, it does not evolve weights.
- 10–20+ agents, each trained K games → **common-seed tournament** → bottom 25% replaced by copies of the top 25% (weights + settings), settings mutated (PBT exploit/explore).
- Lineage and per-generation statistics are logged (video: "generation 1 → generation 40").
- **Criterion:** significant rise of best and mean score over generations; the evolved settings at least match the hand-picked ones from Faz 3. The winning settings are re-trained on the 8×6 multi-stage network.
- **Faz 4b (GPU, A100):** tournaments and self-play batched on the GPU (thousands of boards at once), validated against the CPU results.

### Faz 5 — Council
- Mean of the values of the best 3–5 agents (alternatives: voting, weighted mean).
- **Paired measurement:** council vs best single agent on the same seeds.
- Cost: a council is N× the evaluation work → N× slower search. Second option: **distil** the council into one network (train one network on the council's values).
- CNN member (concept item 4): kept out of this phase.
- **Criterion:** statistically significant gain of the council over the best single agent on the big-tile rates (or, if there is none, report it and continue with the distilled network).

### Faz 6 — Search + endgame tables
**Faz 6a — Search**
- Fast expectimax: transposition table, chance-node pruning by probability, iterative deepening against a time budget, evaluation cache.
- **Tile-downgrading** (Guei): with big tiles on the board, "shrink" it into the region the network knows — critical for 65536.
- Parallelism across games (one game per thread); no parallel search inside a game.

**Faz 6b — Endgame tables (D1: hybrid).** First the method and licence of macroxue and 2048EndgameTablebase are reviewed (understand the approach, do not copy code).
- Patterns: big tiles locked in a corner/edge, small tiles on the rest (e.g. "build the next 512 next to the chain"). For each pattern, backward dynamic programming gives a **success probability + best move** table.
- Memory: small patterns locally (16 GB), big ones on Colab; tables are read from disk through memory mapping (mmap).
- Integration: when the board fits a table pattern the table plays, otherwise council/network + expectimax. The switch point is measured and tuned.
- Check: on small patterns the table matches a brute-force solver exactly.
- **Criterion (6a):** without tables, depth 3–4: **32768 ≥ 50%** — our "pure RL" number, shown separately in the video.
- **Criterion (6b):** the hybrid reaches 32768 ≥ 80% and beats macroxue's 3.5% on 65536 (1000+ games).

### Faz 7 — Record attempt
- Compute budget: minutes per game at depth 4–5; 1000 games / 8 threads ≈ days. A 100-game pilot first, then the big run.
- A full replay of every game and a "that moment" mark for every new max score/tile.
- Cloud sessions: every finished game writes its result + replay to disk; an interrupted session continues with the remaining seeds.
- **Criteria (stepwise):** 65536 > 3.5% (macroxue) → 65536 > 8.4% (current SOTA) → 1.7M score → 131072 attempt. Side criterion: does the pure-RL part beat Guei 2021 (32768 72%)?

### Parallel track — Video and visualisation
- Replay format fixed since Faz 1; a simple HTML replay player in Faz 2; a live web UI to watch a model play.
- After each phase: plots + "that moment" replays under `runs/`.
- Editing at the very end (HyperFrames): the generation story, council vs single agent, the record moment.

---

## 4. Rough schedule

| Phase | Duration | Cumulative |
|---|---|---|
| 0–1 Infrastructure + engine | 2–3 days | ~3 days |
| 2 Base agent | 2–3 days | ~1 week |
| 3 Strengthening | ~1 week | ~2 weeks |
| 4 Evolution | ~1 week | ~3 weeks |
| 5 Council | 2–3 days | ~3.5 weeks |
| 6 Search + tables | 2–3 weeks | ~6 weeks |
| 7 Record | 1–2 weeks | ~7–8 weeks |

Durations are mostly **training/compute time**; writing the code is shorter.

---

## 5. Risks

- **The absolute record probably does not come from pure RL** (section 0). If the story is built around the "RL record", the risk of failure drops a lot.
- **Memory:** 8×6 multi-stage networks and endgame tables are the bottleneck; table size grows steeply with the number of free cells (measured: 10 free cells 139M positions; 12 free cells several billion).
- **65536 representation:** 4 bits per cell end at 32768 — decided in Faz 1.
- **Statistics:** big-tile rates are small; rates like 0.02% need tens of thousands of games. Every claim comes with a confidence interval.
- **Colab disconnects / credits:** checkpoints to Drive; every job resumes.

---

## 6. Decisions (2026-10-13)

- **D1 — Goal: hybrid from the start.** Target: the absolute record (65536 > 8.4%). The RL council plays most of the game, the endgame tables solve the critical merges. Video story: "a learning council + a perfect memory".
- **D2 — Cloud: Google Colab, A100 + High-RAM**, used to full capacity; local = trials only.
- **D3 — Language: C++20 core (+ optional CUDA) + Python analysis.**
- **D4 — Video: certain.** Logs, replays and lineage in full detail from Faz 0; "that moment" replays kept after every phase.
- **D5 — Licences:** macroxue/2048-ai and game-difficulty/2048EndgameTablebase are **GPL-3.0**. Their approach was studied; no code was copied, so this project is not bound by the GPL. They may be run for comparison.

---

## 7. Phase results (measured)

### Faz 0 — 2026-10-13 ✅
- Skeleton, CMake (C++20, `-O3 -march=native`), doctest, `eval`/`replay`/`bench`, JSONL log, `colab/run.ipynb`.
- Tests green locally (g++ 16.2) and **in a Colab proxy**: Docker `ubuntu:22.04` + g++ 11.4.0 (Colab's compiler). The real Colab hardware measurement is notebook cell 2.
- `eval --agent random`, 10,000 games: mean score 1093, 256 7.95%, 512 0.02%; all 10,000 replays verified exactly by `replay`.

### Faz 1 — 2026-10-13 ✅
- **65536 decision:** fast path = `uint64_t`, 4 bits per cell, up to 32768 (the tables do not merge 15+15). As soon as the board holds two 32768s the game switches to the **exact path** (`Grid` = 16 × `uint8_t`, exponent ≤ 17); the policy sees the board through a **tile-downgrading** view, so agents keep working on 4-bit boards. The set of legal moves is preserved (tested).
- Row tables (65536 entries, left/right + score), vertical moves through transposition.
- **Speed: 165M moves/s** on one thread (criterion ≥ 20M); random games 132k games/s.
- Correctness: 4M random boards × 4 directions match the reference 100%; 2000 full games identical move by move; replays reproduce their games exactly; the 32768+32768 → 65536 transition is tested end to end.
- `MaskedSpace` for the hybrid: fixed-cell mask + mixed-radix indexing of the free cells (round-trip tested).

### Faz 2 — 2026-10-13 ✅
- 4×6 n-tuple (Yeh/Jaśkowski shapes, 8 symmetries, 268 MB), afterstate TD(0), α = 0.1 (split over the features), 8 threads Hogwild, 300,000 training games (~13 min).
- Evaluation (1-ply, 1000 games, seeds 1,000,000+): **2048 94.0% [92.4–95.3]** (criterion ≥ 90% ✅), 4096 84.5%, 8192 47.3%, 16384 0.3%; mean score 103,165, max 237,164.
- Learning curve: `analysis/figures/faz2_curve.png`. Replay player: `analysis/player.html` (its JS engine matched C++ on 2000/2000 games).

### Faz 3–5 — ✅ (code complete and validated; full-scale runs on Colab)
Accepted as complete by the project owner on 2026-10-13; the full-scale numbers come from `colab/run.ipynb` → `runs/REPORT.md`.
- **Faz 4b (GPU) brought forward:** one game per warp; values bit-identical to the CPU (no FMA contraction, ordered warp sum). **Batched TD updates:** the TD errors of a launch are summed per weight (touched-weight list) and each weight takes at most κ/α sequential updates with the mean error (κ = 5). Measured on a GTX 1050 Ti: plain Hogwild diverges with 768 concurrent games × α = 0.1; with batched updates one network on every slot follows the CPU learning curve (40k games: mean 42.5k vs 41.2k); TC with α = 1.0 is stable (60k games: mean 85.5k). With one slot and one move per launch the GPU reproduces CPU training bit for bit (TD and TC).
- Mixed-shape GPU populations (4×6/8×6, with/without TC, multi-stage): ablation arms, final replicas, the PBT population and councils run in one GPU process.
- **Local trial ablation (CPU, 300k games, 1-ply, 1000 games):** A TD mean 103k / 2048 94.0%; B +OI 29.6k (OI 320k does not wear off in this budget); C +TC 46.5k / 2048 96.9% [95.6–97.8]; D +multi-stage 49.1k (no effect: 16384 is never reached); E 8×6 78.4k / 2048 98.6% / 4096 82.4%.
- `colab/run.ipynb` (generated by `colab/make_notebook.py`): Faz 3a ablation (5 arms at once) → 3b final configuration (as many replicas as memory allows) → Faz 4 PBT (population sized from memory) + re-training the winning settings on 8×6 → Faz 5 council (mean/weighted on GPU, vote and distillation on CPU) → Faz 6 → `analysis/report.py` checks every criterion (Wilson intervals, paired sign test, McNemar) into `REPORT.md`. The whole notebook was dry-run locally with tiny budgets.

### Faz 6 — code complete and validated; full-scale measurement on Colab
**6a — search (`search/`):** `Searcher` = expectimax + chance-node transposition table + probability cut + iterative deepening against a per-move time budget + tile-downgrading (largest missing exponent above 64 removed while the board holds 2^threshold).
- With the transposition table and no cut the chosen moves equal plain expectimax exactly (tested); depth 3 visits 2.5× fewer nodes (38.7k vs 98.0k on the test positions).
- Local trial (CPU-trained 8×6 net from the trial ablation, 100 games, common seeds): depth 2 mean 102k / 8192 71%; depth 3 mean 112k / 8192 79% at ~4 ms per move and thread. A probability cut of 1e-4 never triggers at depth 3 (branch probabilities stay above ~0.02); it matters at depth 4–5.

**6b — endgame tables (`tablebase/`):** pattern = locked cells + target tile to build in the free cells; locked tiles are placeholders, so one table serves every board whose locked cells hold distinct tiles ≥ the target. Positions are layered by the sum of their free tiles (moves keep it, spawns add 2 or 4): a forward pass generates every reachable position, a backward pass computes the optimal success probability, with at most three layers in memory and every layer on disk (sorted states + probabilities, read through mmap). Lookups try all 8 symmetries and map the move back.
- Verified against an independent brute-force solver on every position of two patterns (4,607 and 146,391 positions, max difference 7.8e-8 = float rounding).
- Build speed (8 threads): 73M positions in 23 s; `0,1,2,3,4,5:9` (6 locked cells, build 512 in 10 cells) 139M positions in 52 s, 1.6 GB on disk, peak RAM 573 MB. Twelve free cells (`0,1,2,3:7`) exceeded 14 GB after 223 layers → Colab scale; `--max-states` aborts oversized patterns.
- **Pattern coverage** (`tablebase --coverage`, how often a pattern fits the net's own games): with the trial net, top-row patterns fit often (`0,1,2,3:7` 12.2% of positions, `0,1,2,3:8` 8.8%, `0,1,2,3,7:7` 4.7%), L/snake shapes never. The notebook therefore picks the tables by measured coverage of the best Colab network.
- **Hybrid integration, measured** (`--agent hybrid --tables … --table-margin M`, trial net, depth 2, 100 games, common seeds; pure search: mean 102.0k, 8192 71%):
  - Table plays whenever the pattern fits (`0,1,2,3,7:7`, 178M positions, 79 s): 5.9% of moves, **worse** (8192 57%). The table maximises "build the target" only and ignores the board it leaves behind.
  - Rule added: a table applies only when the smallest locked tile equals its target (the chain end the new tile merges into) — still worse alone (8192 58%).
  - Switch point added: the search moves unless its move's success probability is more than `M` below the table's best move. `M` = 0.02 / 0.1 / 0.3: tables override 0.18% / 0.02% / 0.01% of moves; 8192 68% / 71% / 72%, mean 99.7k / 102.0k / 102.5k — equal to pure search within noise.
  - Conclusion: at 128-level targets a 2-ply search already plays these positions well; the tables' value has to be measured where the search horizon is too short (targets 512–2048 next to a 16384–32768 chain), i.e. with the Colab networks and Colab-scale tables. The notebook measures pattern coverage with the best network, builds the most frequent patterns under a position cap and evaluates several margins.
