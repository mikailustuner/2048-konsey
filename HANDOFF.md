# HANDOFF — 2048-konsey

Son güncelleme: 2026-10-13. Yeni oturum bu dosyayı ve `PLAN.md`'yi okuyarak başlamalı.

## Proje
Evrimle ayarlanan RL ajanlarından (n-tuple + TD) oluşan bir konsey + expectimax + oyun sonu tablosu (hibrit) ile 2048 rekor denemesi. Videosu da çekilecek.
- Kaynak konsept: `~/Downloads/2048_rekor_speedrun_plani.pdf`
- Ayrıntılı plan, fazlar, başarı ölçütleri: **`PLAN.md`** (tek doğru kaynak)
- Repo: https://github.com/mikailustuner/2048-konsey (özel, `main`, gh ile giriş yapılı)

## Çalışma şekli (kullanıcı tercihi)
- Türkçe konuş.
- **Faz faz ilerle:** bir fazı bitir, başarı ölçütünü ölç ve raporla, kullanıcının onayını bekle, sonra sıradakine geç.
- Kod: `~/.agents/CODING_GUIDELINES.md` geçerli (basitlik, cerrahi değişiklik, önce test, doğrulamadan "bitti" deme).

## Alınan kararlar
- **D1 Hedef:** baştan hibrit (RL konseyi + oyun sonu tablosu). Saf RL sonucu da ayrıca ölçülüp raporlanacak.
- **D2 Ortam:** geliştirme yerelde, ağır işler **Google Colab A100 + High-RAM**'de. Kalıcı veri Google Drive'da (`/content/drive/MyDrive/2048/runs/`), kod GitHub'dan `git pull`, token Colab Secrets'ta (`GITHUB_TOKEN`, fine-grained, salt okuma; henüz oluşturulmadı, Faz 0 sonunda kullanıcıya anlatılacak).
- **D3 Dil:** C++20 çekirdek (Colab'ın eski g++'ı yüzünden C++23 değil), isteğe bağlı CUDA (`-DUSE_CUDA=ON`), analiz Python 3.14 + uv.
- **D4 Video:** kesin çıktı → ayrıntılı log/tekrar/soy kaydı Faz 0'dan itibaren.

## Doğrulanmış çıta (PDF'teki rakamlar eskiydi)
- Mutlak SOTA: game-difficulty/2048EndgameTablebase — 65536 %8,4, 32768 %86 (oyun sonu tablosu tabanlı).
- macroxue/2048-ai (natureofcode fork'u): 65536 %3,5, 32768 %80,5, maks. skor 1.704.908 — elle yazılmış değerlendirme + DP tabloları, RL değil.
- RL en iyi: Guei vd. 2021 (OTD) — 32768 %72, ort. 625.377, 65536 %0,02.

## Donanım
- Yerel: Ryzen 5 1500X (4C/8T), 16 GB RAM, GTX 1050 Ti 4 GB, g++ 16.2, Python 3.14, uv var; **cmake yok** (`uv tool install cmake` ile kurulacak).
- Colab: A100 + High-RAM — rakamlar **doğrulanmadı** (~12 vCPU, ~80 GB RAM, 40/80 GB GPU tahmini). İlk oturumda `nproc`, `free -h`, `nvidia-smi`, `g++ --version` ölçülüp PLAN.md'ye yazılacak.

## Durum
- Yapıldı: plan, kararlar, repo + ilk commit (`3762f75`: PLAN.md, README.md, .gitignore).
- Kod: **henüz yok.**

## Sıradaki adım: Faz 0 (kullanıcı onayı bekleniyor)
PLAN.md → "Faz 0": dizin yapısı (`engine/ ntuple/ search/ evolve/ council/ tools/ tests/ analysis/ colab/`), CMake (C++20, `-O3 -march=native`), test çatısı (Catch2 veya doctest), ortak değerlendirme düzeneği (sabit tohumlar, Wilson %95 güven aralığı), JSONL log şeması, `colab/run.ipynb` iskeleti.
Ölçüt: testler yerelde ve Colab'da yeşil; `eval` rastgele ajanla rapor üretiyor.

## Açık noktalar / dikkat
- 65536+ karelerin temsili (4 bit/kare 32768'de doluyor) Faz 1'de kararlaştırılacak.
- Oyun sonu tablosu için motor API'si "sabit kare maskesi" desteklemeli (Faz 1).
- macroxue ve 2048EndgameTablebase lisansları Faz 6b'den önce incelenecek.
