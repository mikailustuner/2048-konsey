// Exact port of engine/rng.hpp (xoshiro256** seeded by splitmix64) and the reference move rules,
// so a seed gives the same spawns as the C++ engine.
const M64 = (1n << 64n) - 1n;
const rotl = (x, k) => ((x << BigInt(k)) | (x >> BigInt(64 - k))) & M64;

export class Rng {
  constructor(seed) {
    seed = BigInt(seed);
    this.s = [];
    for (let i = 0; i < 4; i++) {
      seed = (seed + 0x9e3779b97f4a7c15n) & M64;
      let z = seed;
      z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & M64;
      z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & M64;
      this.s.push(z ^ (z >> 31n));
    }
  }
  next() {
    const s = this.s, r = (rotl((s[1] * 5n) & M64, 7) * 9n) & M64, t = (s[1] << 17n) & M64;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return r;
  }
  below(n) { return Number((this.next() * BigInt(n)) >> 64n); }
}

export function spawn(g, rng) {
  let k = rng.below(g.filter(v => !v).length);
  const v = rng.below(10) === 0 ? 2 : 1;
  for (let i = 0; i < 16; i++) if (!g[i] && k-- === 0) { g[i] = v; return i; }
  return -1;
}

export const DIRS = "URDL";
const lineCell = (d, l, k) => [l + 4 * k, 4 * l + (3 - k), l + 4 * (3 - k), 4 * l + k][d];

// Moves g in place; returns the merge score, or -1 if nothing moved.
export function move(g, d) {
  let score = 0, moved = false;
  for (let l = 0; l < 4; l++) {
    const out = [];
    let mergedLast = false;
    for (let k = 0; k < 4; k++) {
      const v = g[lineCell(d, l, k)];
      if (!v) continue;
      if (out.length && !mergedLast && out[out.length - 1] === v) { out[out.length - 1]++; score += 2 ** (v + 1); mergedLast = true; }
      else { out.push(v); mergedLast = false; }
    }
    for (let k = 0; k < 4; k++) { const c = lineCell(d, l, k), v = out[k] || 0; if (g[c] !== v) moved = true; g[c] = v; }
  }
  return moved ? score : -1;
}

export const canMove = g => [0, 1, 2, 3].some(d => move(g.slice(), d) >= 0);
