#!/usr/bin/env bash
# Faz 3 ablation: each technique added cumulatively, same game budget, then paired 1-ply
# evaluation on common seeds. Resumable: train resumes from checkpoints, finished runs skip.
# usage: tools/faz3_ablation.sh [games=300000] [eval_games=1000]
set -euo pipefail
cd "$(dirname "$0")/.."
G=${1:-300000}
E=${2:-1000}
R=runs/faz3
mkdir -p $R
run() {  # name, train args...
    local name=$1; shift
    ./build/train --out $R/$name --games $G --log-every 10000 --ckpt-sec 300 --run-id faz3-$name --faz 3 --ajan-id $name "$@"
    grep -q "\"ajan_id\":\"$name\"" $R/eval.jsonl 2>/dev/null ||
        ./build/eval --agent greedy --net $R/$name/net.bin --games $E --run-id faz3 --faz 3 --ajan-id $name --out $R |
        sed "s/^/[$name] /" | tee -a $R/eval.txt
}
mkdir -p $R/A_td && [ -f $R/A_td/net.bin ] || cp runs/faz2/net.bin $R/A_td/net.bin   # Faz 2 net = baseline
run A_td   --net 4x6 --alpha 0.1
run B_otd  --net 4x6 --alpha 0.1 --oi 320000
run C_otc  --net 4x6 --alpha 1.0 --oi 320000 --tc
run D_otc_ms --net 4x6 --alpha 1.0 --oi 320000 --tc --stages "14;14,13"
run E_8x6  --net 8x6 --alpha 1.0 --oi 320000 --tc --stages "14;14,13"
