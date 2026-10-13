# HANDOFF — 2048-konsey

Last update: 2026-10-13. A new session starts by reading this file and `PLAN.md`.

## Project
A council of RL agents tuned by evolution (n-tuple + TD) + expectimax + endgame tables (hybrid), attempting the 2048 record. A video is planned as well.
- Detailed plan, phases, success criteria: **`PLAN.md`** (single source of truth; phase results in §7)
- Repository: https://github.com/mikailustuner/2048-konsey (**public**, `main`, logged in with `gh`)

## Working agreement (user preferences)
- Talk to the user in **Turkish**; repository documentation is in **English**.
- **Phase by phase:** finish a phase, measure and report its criterion, then continue.
- **All real training runs on Colab A100 + High-RAM, using it to full capacity; the local PC only runs small trials.**
- Code rules: `~/.agents/CODING_GUIDELINES.md` (simplicity, surgical changes, tests first, never claim "done" without verification).

## Decisions
- **D1 Goal:** hybrid from the start (RL council + endgame tables). The pure-RL result is measured and reported separately.
- **D2 Environment:** development locally, heavy jobs on **Google Colab A100 + High-RAM**. Persistent data on Google Drive (`/content/drive/MyDrive/2048/runs/`), code from the public GitHub repository, large tables on the Colab disk (`/content/tables`).
- **D3 Language:** C++20 core (not C++23, because of Colab's older g++), optional CUDA (`-DUSE_CUDA=ON`), Python analysis.
- **D4 Video:** certain → detailed logs/replays/lineage from Faz 0.
- **D5 Licences:** macroxue/2048-ai and 2048EndgameTablebase are GPL-3.0; approach studied, no code copied.

## Verified bar
- Absolute SOTA: game-difficulty/2048EndgameTablebase — 65536 8.4%, 32768 86% (endgame-table based).
- macroxue/2048-ai: 65536 3.5%, 32768 80.5%, max score 1,704,908 — hand-written evaluation + DP tables, not RL.
- Best RL: Guei et al. 2021 (OTD) — 32768 72%, mean 625,377, 65536 0.02%.

## Hardware
- Local: Ryzen 5 1500X (4C/8T), 16 GB RAM, GTX 1050 Ti 4 GB (~2.4 GB free), g++ 16.2, Python 3.14, uv, cmake (uv tool), Docker with the NVIDIA runtime.
- Colab: A100 + High-RAM — numbers are measured by notebook cell 2 (`runs/hardware.txt` on Drive).

## Status
- Faz 0–2 ✅ measured (tags `faz-0..2`).
- Faz 3–5 ✅ accepted by the user: code + GPU path (Faz 4b) + Colab pipeline complete and validated locally; full-scale numbers come from the Colab run.
- Faz 6 ✅ code complete and validated: `search/` (transposition table, probability cut, iterative deepening, tile-downgrading), `tablebase/` (endgame tables verified against brute force, mmap, 8-orientation lookup, coverage measurement), hybrid agent with a measured table/search switch point; Colab cells 11–12. Full-scale criteria (6a: 32768 ≥ 50% without tables; 6b: hybrid 32768 ≥ 80%, 65536 > 3.5%) are measured on Colab.
- GUI: `gui/launch.sh` (watch a model live / play yourself); models from Drive via `tools/fetch_model.sh` (rclone in `~/.local/bin`, needs `rclone config create gdrive drive scope=drive.readonly` once).
- Local CUDA: Docker image `konsey-cuda:12.6` (Ubuntu 22.04 + CUDA 12.6 + g++ 11, the Colab proxy; built from a two-line Dockerfile: `FROM nvidia/cuda:12.6.3-devel-ubuntu22.04` + `apt-get install g++ ninja-build python3-pip && pip3 install cmake`), build dir `build-cuda/`; its binaries also run on the host. `python3 colab/dryrun.py` runs the whole notebook with tiny budgets.

## Next step
1. User: open `colab/run.ipynb` on Colab (A100 + High-RAM) and run the cells in order; results land in `MyDrive/2048/runs/REPORT.md`.
2. Copy the measured numbers into PLAN.md §7 (Faz 3–6) and tag the phases.
3. Faz 7 (record attempt) after that.

## Open points / caveats
- The GPU path uses the fast board only (no 65536 merge); record/65536 measurements use the CPU `eval` (exact path).
- GPU training is TD(0)/TC; TD(λ) exists on the CPU only, so the GPU PBT evolves α and the OI value.
- Endgame tables: a table only plays when the smallest locked tile equals its target (the chain end it feeds), and only overrides the search when the search's move is `--table-margin` less likely to succeed. Locally (weak trial network, target 128) tables did not improve play — their value is expected at 32768+/65536 with large targets, which needs Colab-scale tables. Twelve free cells means billions of positions.
- The repository has no licence file yet (public, so "all rights reserved" by default) — the owner should choose one.
