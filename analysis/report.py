"""Phase report: reads the runs directory (local or Drive) and checks every PLAN.md criterion.

    python3 analysis/report.py RUNS_DIR [--out RUNS_DIR/RAPOR.md]

Expects the layout written by colab/run.ipynb:
  faz3/log.jsonl, faz3/games.jsonl   ablation arms + final nets, ajan_id "<name>-d<depth>"
  faz4/generations.jsonl             PBT tournaments
  faz4/log.jsonl, faz4/games.jsonl   evolved settings re-trained vs manual ("<name>-d<depth>")
  faz5/log.jsonl, faz5/games.jsonl   council variants vs best single
Standard library only (runs anywhere). Missing parts are reported as missing.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

TILES = [11, 12, 13, 14, 15, 16]


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, mid - half), min(1.0, mid + half)


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def eval_table(rows: list[dict]) -> str:
    out = ["| ajan | oyun | ort. skor | " + " | ".join(str(1 << t) for t in TILES[:5]) + " |",
           "|---|---|---|" + "---|" * 5]
    for r in rows:
        cells = [r["ajan_id"], str(r["oyun_sayisi"]), f'{r["ort_skor"]:,.0f}']
        for t in TILES[:5]:
            lo, hi = r["guven"][str(1 << t)]
            cells.append(f'{pct(r["oranlar"][str(1 << t)])} [{pct(lo)}, {pct(hi)}]')
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def by_agent(games: list[dict]) -> dict[str, dict[int, dict]]:
    d = defaultdict(dict)
    for g in games:
        d[g["ajan_id"]][g["seed"]] = g
    return d


def paired(games: dict, a: str, b: str) -> dict | None:
    """a vs b on the common seeds: sign test on scores, McNemar on each tile."""
    if a not in games or b not in games:
        return None
    seeds = sorted(set(games[a]) & set(games[b]))
    if not seeds:
        return None
    wins = sum(games[a][s]["score"] > games[b][s]["score"] for s in seeds)
    losses = sum(games[a][s]["score"] < games[b][s]["score"] for s in seeds)
    n = wins + losses
    z = (abs(wins - losses) - 1) / math.sqrt(n) if n else 0.0
    res = {"n": len(seeds), "wins": wins, "losses": losses, "p_sign": math.erfc(max(z, 0) / math.sqrt(2)),
           "mean_a": sum(games[a][s]["score"] for s in seeds) / len(seeds),
           "mean_b": sum(games[b][s]["score"] for s in seeds) / len(seeds), "tiles": {}}
    for t in TILES:
        ra = [games[a][s]["max_exp"] >= t for s in seeds]
        rb = [games[b][s]["max_exp"] >= t for s in seeds]
        bb = sum(x and not y for x, y in zip(ra, rb))
        cc = sum(y and not x for x, y in zip(ra, rb))
        chi = (abs(bb - cc) - 1) ** 2 / (bb + cc) if bb + cc else 0.0
        res["tiles"][t] = {"a": sum(ra) / len(seeds), "b": sum(rb) / len(seeds), "only_a": bb, "only_b": cc,
                           "p": math.erfc(math.sqrt(chi / 2)) if bb + cc else 1.0}
    return res


def paired_md(res: dict | None, a: str, b: str) -> str:
    if res is None:
        return f"- `{a}` vs `{b}`: veri yok\n"
    s = (f"- `{a}` vs `{b}` ({res['n']} ortak tohum): ort. skor {res['mean_a']:,.0f} vs {res['mean_b']:,.0f}; "
         f"skorda {res['wins']} galibiyet / {res['losses']} yenilgi, işaret testi p = {res['p_sign']:.2g}\n")
    for t, v in res["tiles"].items():
        if v["a"] == 0 and v["b"] == 0:
            continue
        s += f"  - {1 << t}: {pct(v['a'])} vs {pct(v['b'])} (yalnız a {v['only_a']}, yalnız b {v['only_b']}, McNemar p = {v['p']:.2g})\n"
    return s


def rate(row: dict, t: int) -> tuple[float, float, float]:
    lo, hi = row["guven"][str(1 << t)]
    return row["oranlar"][str(1 << t)], lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--out")
    args = ap.parse_args()
    R = Path(args.runs)
    md = ["# 2048-konsey — faz raporu\n", f"Kaynak: `{R}`\n"]

    # ---- Faz 3 ----
    md.append("## Faz 3 — güçlendirme\n")
    f3 = [r for r in jsonl(R / "faz3/log.jsonl") if "egitim_oyunu" not in r]
    g3 = by_agent(jsonl(R / "faz3/games.jsonl"))
    if not f3:
        md.append("Değerlendirme yok.\n")
    else:
        md.append(eval_table(f3) + "\n")
        arms = ["A_td", "B_otd", "C_otc", "D_otc_ms", "E_8x6"]
        md.append("\n**Ablasyon (her teknik bir öncekine göre, 1-ply, eşleştirilmiş):**\n")
        for prev, cur in zip(arms, arms[1:]):
            md.append(paired_md(paired(g3, f"{cur}-d1", f"{prev}-d1"), f"{cur}-d1", f"{prev}-d1"))
        finals = [r for r in f3 if r["ajan_id"].endswith("-d2") and r["ajan_id"].startswith("final")]
        if finals:
            best = max(finals, key=lambda r: r["oranlar"]["16384"])
            p, lo, hi = rate(best, 14)
            ok = p >= 0.5
            md.append(f"\n**Ölçüt (2-ply 16384 ≥ %50):** en iyi final `{best['ajan_id']}`: {pct(p)} [{pct(lo)}, {pct(hi)}] → "
                      f"{'✅ sağlandı' if ok else '❌ sağlanmadı'}\n")
        else:
            md.append("\n**Ölçüt (2-ply 16384 ≥ %50):** final 2-ply değerlendirmesi yok.\n")

    # ---- Faz 4 ----
    md.append("\n## Faz 4 — evrim (PBT)\n")
    gens = jsonl(R / "faz4/generations.jsonl")
    if not gens:
        md.append("Nesil kaydı yok.\n")
    else:
        per = defaultdict(list)
        for r in gens:
            per[r["nesil"]].append(r)
        rows = ["| nesil | en iyi ort. skor | nüfus ort. | en iyi α | en iyi OI |", "|---|---|---|---|---|"]
        for gnum in sorted(per):
            b = max(per[gnum], key=lambda r: r["ort_skor"])
            rows.append(f"| {gnum} | {b['ort_skor']:,.0f} | {sum(r['ort_skor'] for r in per[gnum]) / len(per[gnum]):,.0f} | "
                        f"{b['alpha']:.4f} | {b['oi']:,.0f} |")
        md.append("\n".join(rows) + "\n")
        first, last = min(per), max(per)
        fb = max(per[first], key=lambda r: r["ort_skor"])
        lb = max(per[last], key=lambda r: r["ort_skor"])
        # Tournament seeds differ per generation, so compare with intervals on the score mean.
        md.append(f"\nEn iyi ajan: nesil {first} {fb['ort_skor']:,.0f} → nesil {last} {lb['ort_skor']:,.0f}; "
                  f"nüfus ort. {sum(r['ort_skor'] for r in per[first]) / len(per[first]):,.0f} → "
                  f"{sum(r['ort_skor'] for r in per[last]) / len(per[last]):,.0f}.\n")
        g4 = by_agent(jsonl(R / "faz4/games.jsonl"))
        md.append("\n**Evrimle bulunan ayarlar vs elle seçilen (aynı ağ, aynı bütçe, 2-ply):**\n")
        md.append(paired_md(paired(g4, "evolved-d2", "manual-d2"), "evolved-d2", "manual-d2"))
        md.append(paired_md(paired(g4, "evolved-d1", "manual-d1"), "evolved-d1", "manual-d1"))

    # ---- Faz 5 ----
    md.append("\n## Faz 5 — konsey\n")
    f5 = [r for r in jsonl(R / "faz5/log.jsonl") if "egitim_oyunu" not in r]
    g5 = by_agent(jsonl(R / "faz5/games.jsonl"))
    if not f5:
        md.append("Değerlendirme yok.\n")
    else:
        md.append(eval_table(f5) + "\n")
        for depth in (1, 2):
            singles = [r for r in f5 if r["ajan_id"].startswith("single") and r["ajan_id"].endswith(f"-d{depth}")]
            if not singles:
                continue
            best = max(singles, key=lambda r: r["ort_skor"])["ajan_id"]
            md.append(f"\n**Derinlik {depth}: konsey vs en iyi tek ajan (`{best}`):**\n")
            for r in f5:
                if r["ajan_id"].startswith(("council", "student")) and r["ajan_id"].endswith(f"-d{depth}"):
                    md.append(paired_md(paired(g5, r["ajan_id"], best), r["ajan_id"], best))
    text = "\n".join(md)
    out = Path(args.out) if args.out else R / "RAPOR.md"
    out.write_text(text)
    print(text)
    print(f"\nyazildi: {out}")


if __name__ == "__main__":
    main()
