# 2048-konsey

Evrimle ayarlanan RL ajanlarından (n-tuple + TD) oluşan bir konseyin, expectimax arama ve oyun sonu tablolarıyla 2048 rekoruna saldırısı.

Plan ve fazlar: [PLAN.md](PLAN.md) · Durum: [HANDOFF.md](HANDOFF.md)

## Derleme ve test

```sh
uv tool install cmake          # bir kez
cmake -S . -B build -G Ninja && cmake --build build
./build/tests
```

C++20, `-O3 -march=native` (ikili dosya makineye özel; her makinede yeniden derle). Colab: `colab/run.ipynb`.

## Araçlar

| Araç | İş |
|---|---|
| `bench` | tek iş parçacığı hamle hızı |
| `eval --agent random --games N --seed S --out DIR [--record]` | ortak tohumlu değerlendirme, `DIR/log.jsonl` (+ `replays.jsonl`) |
| `replay --file DIR/replays.jsonl` | kayıtlı oyunları tohum + hamle dizisinden yeniden oynatıp doğrular |

## Değerlendirme ve log şeması

N oyun, tohumlar `seed .. seed+N-1` (aynı tohum = aynı kare dizisi → eşleştirilmiş karşılaştırma). Her ölçüm `log.jsonl`'a bir satır ekler:

`run_id, faz, ajan_id, nesil, oyun_sayisi, hamle_sayisi, oranlar{2048..65536}, guven{2048..65536: [alt, üst]} (Wilson %95), ort_skor, maks_skor, sure`

Tekrar satırı: `{run_id, seed, score, max_tile, moves: "URDL..."}`.

## Arayüz: kendin oyna / modeli canlı izle

```sh
gui/launch.sh          # veya uygulama menüsü → "2048 Konsey"; http://127.0.0.1:8048
```

- **Modeli izle:** `runs/` ve `models/` altındaki her ağ listelenir; derinlik (1–3), tohum, hız seçilir. Gateway (`gui/server.py`, yalnızca stdlib, 127.0.0.1) `build/play`'i çalıştırır ve hamleleri SSE ile canlı akıtır.
- **Kendin oyna:** ok tuşları/WASD/kaydırma; aynı tohum = modelin gördüğü kare dizisinin aynısı.
- **Colab'da eğitilen modeli getirmek:** Colab notebook'u ağı `MyDrive/2048/final/`'a yazar; yerelde bir kez `rclone config create gdrive drive scope=drive.readonly`, sonra `tools/fetch_model.sh` → `models/`. Ağ dosyası makineden bağımsızdır (yalnızca ikili dosya `-march=native`).
