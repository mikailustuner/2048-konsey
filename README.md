# 2048-konsey

A council of reinforcement-learning agents (n-tuple networks trained by TD learning, tuned by evolution) that attacks the 2048 record together with expectimax search and endgame tables. *Konsey* is Turkish for *council*.

- Plan, phases and success criteria: [PLAN.md](PLAN.md)
- Current state and next steps: [HANDOFF.md](HANDOFF.md)
- All real training runs on Google Colab (A100 + High-RAM): [colab/run.ipynb](colab/run.ipynb)

## Status

| Phase | |
|---|---|
| 0 Infrastructure, 1 Engine, 2 Base agent | ✅ measured (see PLAN.md §7) |
| 3 Strengthening, 4 Evolution (+4b GPU), 5 Council | ✅ code complete and validated; full-scale runs on Colab |
| 6 Search + endgame tables | ✅ code complete and validated; full-scale runs on Colab |
| 7 Record attempt | not started |

## Build and test

```sh
uv tool install cmake                      # once
cmake -S . -B build -G Ninja && cmake --build build
./build/tests
```

C++20, `-O3 -march=native` (binaries are machine specific: rebuild on every machine).

GPU path (CUDA, `-DUSE_CUDA=ON`): on Colab the notebook builds it. Locally, a Docker image mirrors Colab's toolchain (Ubuntu 22.04, CUDA 12.6, g++ 11):

```sh
docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/src -w /src konsey-cuda:12.6 \
  bash -c "cmake -S . -B build-cuda -G Ninja -DUSE_CUDA=ON && cmake --build build-cuda"
./build-cuda/gpu_tests        # the binaries also run on the host
python3 colab/dryrun.py       # the whole Colab notebook with tiny budgets
```

## Tools

| Tool | What it does |
|---|---|
| `bench` | single-thread move speed |
| `eval --agent random\|greedy\|expectimax\|council\|hybrid --net a.bin[,b.bin] --games N [--gpu]` | common-seed evaluation → `log.jsonl` (+ `--record` replays, `--per-game` scores) |
| `replay --file replays.jsonl` | replays recorded games from seed + moves and checks them |
| `train --net 4x6\|8x6 [--tc] [--oi V] [--stages "14;14,13"] --alpha A --games N [--gpu]` | TD self-play training, resumable checkpoints |
| `train_many --jobs file --dir D` | many networks at once on the GPU (ablation arms, replicas) |
| `evolve [--gpu] --population P --generations G` | population based training of the TD settings |
| `distill --teachers a.bin,b.bin --net 8x6` | trains one network on a council's values |
| `tablebase --build --pattern "0,1,2,3,7:9" --out DIR` | endgame table (`--coverage N --net X`: how often a pattern fits a network's games) |
| `play --agent … --seed S` | one game as live JSON lines (used by the GUI) |

Search options for `expectimax` and `hybrid`: `--depth`, `--time-ms` (iterative deepening), `--prob-cut`, `--tt-bits`, `--downgrade` (tile-downgrading threshold); `hybrid`: `--tables dir1,dir2 --table-margin M`.

## Watch a model play / play yourself

```sh
gui/launch.sh      # or the application menu entry "2048 Konsey"; http://127.0.0.1:8048
```

- **Watch the model:** every network under `runs/` and `models/` is listed; pick search depth (1–3), seed and speed. The gateway (`gui/server.py`, standard library only, 127.0.0.1) runs `build/play` and streams the moves over SSE.
- **Play yourself:** arrow keys / WASD / swipe; the same seed gives the same tile sequence the model faced.
- **Models trained on Colab:** the notebook copies them to `MyDrive/2048/final/`; locally run `rclone config create gdrive drive scope=drive.readonly` once, then `tools/fetch_model.sh` → `models/`.

`analysis/player.html` replays recorded games (`replays.jsonl`) in the browser.

## Evaluation and log schema

N games with seeds `seed .. seed+N-1` (same seed = same tile sequence → paired comparisons). Every measurement appends one JSONL line (field names are Turkish):

`run_id, faz` (phase)`, ajan_id` (agent)`, nesil` (generation)`, oyun_sayisi` (games)`, hamle_sayisi` (moves)`, oranlar{2048..65536}` (reach rates)`, guven{2048..65536: [lo, hi]}` (Wilson 95% intervals)`, ort_skor` (mean score)`, maks_skor` (max score)`, sure` (seconds)`, egitim_oyunu` (training games, training logs only).

Replay line: `{run_id, seed, score, max_tile, moves: "URDL..."}`. `analysis/report.py RUNS_DIR` checks every phase criterion and writes `REPORT.md`.

## References and licences

The approach of [macroxue/2048-ai](https://github.com/macroxue/2048-ai) and [game-difficulty/2048EndgameTablebase](https://github.com/game-difficulty/2048EndgameTablebase) (both GPL-3.0) was studied; no code was taken from them. Key papers: Szubert & Jaśkowski 2014, Jaśkowski 2018 (TC, multi-stage), Guei et al. 2021 (optimistic TD, tile-downgrading). `tests/third_party/doctest.h` is MIT-licensed.
