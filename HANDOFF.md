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

## Durum (2026-10-13)
- Faz 0–2 ✅ (etiketler `faz-0..2`, sonuçlar PLAN.md §7).
- Faz 3–5: kod + GPU (Faz 4b) + Colab hattı hazır ve yerelde doğrulandı; **ölçümler Colab'da koşulacak** (`colab/run.ipynb`, sırayla).
- Arayüz: `gui/launch.sh` (modeli canlı izle / kendin oyna), model Drive'dan `tools/fetch_model.sh` ile.
- CUDA yerelde: `docker run --gpus all ... konsey-cuda:12.6` (Ubuntu 22.04 + CUDA 12.6 + g++ 11, Colab vekili); derleme dizini `build-cuda/`, ikililer host'ta da çalışır.

## Sıradaki adım
Kullanıcı: Colab Secrets'a `GITHUB_TOKEN` ekle → `colab/run.ipynb` hücre 1–6 → bütçeleri kalibrasyona göre ayarla → 7… Sonra `runs/RAPOR.md` ile faz sonuçları PLAN.md'ye.

## Açık noktalar / dikkat
- GPU yolu yalnızca hızlı tahta (65536 birleşmesi yok); rekor/65536 ölçümleri CPU `eval` ile (kesin yol).
- GPU eğitimi TD(0)/TC; λ>0 yalnızca CPU'da.
- macroxue ve 2048EndgameTablebase lisansları Faz 6b'den önce incelenecek.
