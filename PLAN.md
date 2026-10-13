# 2048 Rekor Speedrun'ı — Uygulama Planı

Kaynak: `~/Downloads/2048_rekor_speedrun_plani.pdf` (konsept). Bu dosya onu uygulanabilir fazlara böler.
Kural: her faz **bitti** sayılmadan bir sonrakine geçilmez; "bitti" = başarı ölçütü ölçüldü ve kaydedildi.

---

## ★ Çalışma ortamı: Google Colab — A100 + High-RAM (karar: 2026-10-13)

Kod bu ortama göre yazılır. Geliştirme ve testler yerelde, ağır işler Colab'da koşar.

| | Yerel (geliştirme) | Colab A100 High-RAM (ağır iş) |
|---|---|---|
| CPU | Ryzen 5 1500X, 8 thread | Xeon, ~12 vCPU (ilk oturumda `nproc` ile ölçülecek) |
| RAM | 16 GB | ~80 GB (`free -h` ile ölçülecek) |
| GPU | GTX 1050 Ti 4 GB | A100 40 GB (veya 80 GB; `nvidia-smi` ile ölçülecek) |
| Derleyici | g++ 16 | Ubuntu'nun g++'ı (muhtemelen 11.x) + CUDA `nvcc` |

**Koda yansıyan kurallar:**
1. **C++20** (C++23 değil) — Colab'ın eski g++'ı ile derlenebilmesi için. Yerelde de `-std=c++20`.
2. **Her makinede yeniden derle.** `-march=native` ikili dosyası makineye özel; Colab'da her oturum başında `cmake --build` (birkaç saniye).
3. **CUDA isteğe bağlı derleme bayrağı** (`-DUSE_CUDA=ON`). CPU yolu her zaman tam ve referans; GPU yolu CPU sonuçlarıyla eşleştiği test edilerek eklenir. Yerel 1050 Ti küçük CUDA testleri için yeterli.
4. **Kalıcı veri Google Drive'da.** Colab diski oturumla silinir: checkpoint'ler, loglar, tekrarlar, tablolar `/content/drive/MyDrive/2048/runs/` altına yazılır. Her uzun iş N dakikada bir checkpoint alır ve **kaldığı yerden devam eder** (bağlantı kopması, kredi bitmesi).
5. **Kod aktarımı: özel GitHub reposu.** Akış: yerelde yaz + test → `git push` → Colab'da hazır notebook (`colab/run.ipynb`): Drive bağla → `git pull` → derle → testleri çalıştır → işi başlat. GitHub token'ı **Colab Secrets**'ta tutulur, notebook'a veya koda yazılmaz.
6. **Hesap kredisi pahalı.** A100 compute unit'i hızlı tüketir: Colab'da sadece doğrulanmış, yerelde küçük ölçekte çalışmış işler koşar. Boşta oturum bırakılmaz; iş bitince notebook runtime'ı kapatır.
7. **Bellek bütçeleri ~80 GB'a göre** (bkz. bölüm 1): evrim nüfusu ve oyun sonu tabloları yereldekinden çok daha büyük olabilir.

---

## 0. PDF'e göre düzeltmeler (önce bunları bilelim)

| Konu | PDF diyor | Gerçek (kontrol edildi, 2026-10-13) | Sonuç |
|---|---|---|---|
| Çıta | natureofcode/2048-ai = expectimax + n-tuple, 65536 %3,5 | Bu, macroxue/2048-ai'nin fork'u; **n-tuple/RL değil**: elle yazılmış değerlendirme + dinamik programlamayla hesaplanmış dev oyun sonu tabloları. Derinlik 8'de oyun başı ~1450 sn, büyük sürüm 27 GB RAM ister. | Bu bir "arama + tablo" sistemi, öğrenen ajan değil. |
| Güncel SOTA | — | **game-difficulty/2048EndgameTablebase: 65536 %8,4, 32768 %86.** Yine oyun sonu tablosu (endgame tablebase) tabanlı. | Mutlak rekor çıtası PDF'tekinin ~2,4 katı. |
| RL tabanlı en iyi | Yeh 2016: 32768 %31,75 | **Guei vd. 2021 (OTD): 32768 %72, ortalama 625.377, 65536 %0,02** (OI + TC + çok aşamalı + expectimax + tile-downgrading). | Saf RL'nin gerçek çıtası bu. |
| 131072 | "henüz görülmeyen" | Teorik üst sınır 131072 (skor ~3,93M). AI'da görülmediği iddiası doğrulanmalı. | Hedef olarak kalsın, söz verilmesin. |

**Stratejik sonuç:** İki ayrı "rekor" var:
1. **RL rekoru** (öğrenen ajanların rekoru): Guei 2021'i geçmek — gerçekçi ve hikâyeye uygun.
2. **Mutlak rekor**: 65536'da %8,4'ü geçmek — muhtemelen oyun sonu tablosu gibi öğrenme dışı bir bileşen gerektirir (hibrit).

→ **Karar D1: hibrit** (bkz. bölüm 6). Saf RL sayısı da ayrıca ölçülüp raporlanır.

---

## 1. Donanım gerçekliği

| Kaynak | Değer | Ne için uygun |
|---|---|---|
| Yerel CPU | Ryzen 5 1500X, 4 çekirdek / 8 iş parçacığı (Zen 1) | Motor, TD eğitimi, arama — **ana iş gücü** |
| Yerel RAM | 16 GB (~10 GB boş) | Aynı anda en fazla ~2–3 tane 8×6 çok aşamalı ağ (her aşama 512 MB) |
| Yerel GPU | GTX 1050 Ti, 4 GB (2,4 GB boş) | Küçük ağlarla toplu oyun denemesi, CUDA prototipi |
| Colab A100 + High-RAM | ~80 GB RAM, A100 40 GB (D2) | Uzun eğitimler, toplu GPU değerlendirme/turnuva, büyük tablo üretimi, rekor koşusu |

Bundan çıkan iki kural:
- **CPU önce.** n-tuple yöntemi bellekten rastgele okuma işidir; GPU ancak binlerce oyun toplu oynanırken kazandırır. GPU işi, CPU tabanı çalışıp profil alındıktan sonra yapılır (Faz 4b, isteğe bağlı).
- **Evrim boyutu ortama göre.** 20 ajan × 8×6 çok aşamalı ≈ 30 GB: yerelde sığmaz, Colab'ın ~80 GB'ına sığar. Yerelde küçük ölçekli denemeler **4×6 ağlarla** (ajan başına 64 MB), asıl evrim Colab'da 8×6 ile yapılabilir.

---

## 2. Teknik yığın (Karar D3)

- **Çekirdek: C++20** (Colab uyumu için; `-O3 -march=native`), isteğe bağlı CUDA. Motor, n-tuple, TD, arama, nüfus döngüsü. Python+Numba de olur ama arama/TD iç döngülerinde C++ hem daha hızlı hem daha öngörülebilir.
- **Yapı:** CMake (`uv tool install cmake`) + Catch2 veya doctest (tek başlık dosyası) birim testleri.
- **Analiz / grafik / video verisi: Python 3.14 + uv** (pandas/polars, matplotlib). C++ ile konuşma dosya üzerinden (CSV/JSONL + ikili ağırlık dosyaları) — bağlama (pybind) yok, basit kalsın.
- **Tekrarlanabilirlik:** her oyun bir tohumdan (xoshiro256**) üretilir; aynı tohum = aynı kare dizisi. Tüm karşılaştırmalar **ortak tohumlarla** (eşleştirilmiş test).
- **Git** ile sürüm, her faz sonunda etiket (`faz-1`, `faz-2`…).

Önerilen dizin yapısı:
```
engine/      bitboard, hamle tabloları, RNG, tekrar kaydı
ntuple/      ağ, simetriler, TD/TC/OI, çok aşamalı
search/      expectimax, transpozisyon tablosu
evolve/      nüfus, turnuva, mutasyon, soy kaydı
council/     topluluk değerlendirmesi
tools/       CLI'lar: bench, train, eval, play, replay
tests/
analysis/    Python: grafikler, raporlar, video verisi
runs/        (git dışı) ağırlıklar, loglar, tekrarlar
```

---

## 3. Fazlar

Her fazda: **Hedef → Yapılacaklar → Başarı ölçütü (ölçülebilir) → Çıktılar**.

### Faz 0 — Kurulum ve ölçüm altyapısı (≈0,5 gün)
- git init, dizin yapısı, CMake, test çatısı, `-O3 -march=native` derleme.
- Özel GitHub reposu + `colab/run.ipynb` iskeleti (Drive bağla → `git pull` → derle → test). İlk Colab oturumunda donanım ölçülüp bu plana yazılır (`nproc`, `free -h`, `nvidia-smi`, `g++ --version`).
- Ortak **değerlendirme düzeneği**: N oyun, sabit tohum aralığı, çıktı = ortalama skor + her kare için ulaşma oranı + **%95 güven aralığı** (Wilson). Bütün fazlar bunu kullanacak.
- Log şeması (JSONL): `run_id, faz, ajan_id, nesil, oyun_sayısı, hamle_sayısı, oranlar{2048..65536}, ort_skor, maks_skor, süre`. Video hikâyesi buradan çıkacak; baştan kurulmalı.
- **Ölçüt:** testler hem yerelde hem Colab'da yeşil, `eval` komutu rastgele ajanla rapor üretiyor ve raporu Drive'a yazıyor.

### Faz 1 — Hızlı motor (≈1–2 gün)
- Tahta = `uint64_t`, kare başına 4 bit (log2 değer). 0 = boş, 1..15 = 2..32768. **65536 (=16) ve 131072 (=17) sığmaz.** Faz 1'de karar: (a) 64 bit tahta + nadir durumlar için yan bilgi, (b) 128 bit tahta (5 bit/kare), (c) tile-downgrading ile hiç 16'ya çıkmadan oynamak. Öneri: motor 32768'e kadar hızlı yolu kullansın, taşma durumu ayrı ve test edilmiş bir yoldan gitsin.
- 65536 girişli satır tabloları (sola/sağa hamle + satır skoru), transpoz ile yukarı/aşağı.
- Kare üretimi (%90 → 2, %10 → 4), oyun sonu tespiti, tekrar (replay) kaydı: tohum + hamle dizisi.
- **Hibrit için (D1):** motor API'si baştan "sabit kare maskesi" desteklesin — belirli hücreleri kilitli büyük karelerle sabitleyip kalan küçük kareler üzerinde durum sayma/sıralama (indeksleme). Tablo üretici (Faz 6b) bunun üzerine kurulur.
- **Doğruluk:** saf/yavaş bir referans uygulamaya karşı milyonlarca rastgele tahtada karşılaştırmalı test.
- **Ölçüt:** tek iş parçacığında **≥ 20M hamle/sn** (bench aracıyla ölçülür), rastgele-oyun testleri referansla %100 eşleşiyor, tekrar dosyası aynı oyunu birebir yeniden üretiyor.

### Faz 2 — Temel ajan: n-tuple + TD(0) (≈2–3 gün, eğitim saatleri dahil)
- n-tuple ağı: 4×6 tuple (Yeh/Jaśkowski deseni), 8 simetri, `float` ağırlıklar.
- **Hamle-sonrası durum (afterstate)** üzerinde TD(0), kendi kendine oyun, açgözlü (1-ply) hamle seçimi.
- Çoklu iş parçacığı: 8 thread, paylaşılan ağırlıklar (Hogwild tarzı, kilitsiz).
- Kontrol noktası (checkpoint) kaydet/yükle; eğitim kesilirse kaldığı yerden devam.
- **Ölçüt:** 1-ply, 1000 oyunda **2048'e ≥ %90** (literatürde 4×6 ile bilinen seviye). Öğrenme eğrisi grafiği (oyun sayısı → ort. skor, 2048 oranı).

### Faz 3 — Güçlendirme (≈1 hafta, çoğu eğitim süresi)
Her teknik tek tek eklenir, **ortak tohumlarla önce/sonra ölçülür** (ablation tablosu = videoda da kullanılabilir):
1. İyimser başlangıç (OI) — Guei 2021.
2. Temporal Coherence (TC) öğrenme oranı.
3. Çok aşamalı TD (aşama sınırları ör. 16384 ve 16384+8192 görülünce).
4. Ağ büyüklüğü: 4×6 → 8×6.
5. **Basit expectimax (2-ply) şimdi eklenir** — sadece ölçüm aracı olarak. Gerekçe: 1-ply sayıları ile gerçek gücü ayırt etmek zor; PDF aramayı Faz 6'ya koymuş, ama değerlendirme için erkenden lazım. Hızlı/optimize arama yine Faz 6'da.
- **Ölçüt:** 2-ply aramayla **16384'e ≥ %50**, 1-ply ile anlamlı (güven aralıkları çakışmayan) artış her teknikte raporlu.

### Faz 4 — Evrim (nüfus tabanlı eğitim, PBT) (≈1 hafta)
- Genom = TD'nin **ayarları** (öğrenme oranı, OI değeri, λ, TC parametreleri, aşama sınırları, isteğe bağlı tuple şekli). PDF'in de dediği gibi: evrim ağırlıkları değil, RL'yi yönetir.
- 10–20 ajan (4×6), her biri K oyun eğitilir → **ortak tohumlu turnuva** → alt %25 elenir, üst %25'in ağırlıkları + ayarları kopyalanır, ayarlar mutasyona uğrar (PBT exploit/explore).
- Soy ağacı ve nesil istatistikleri loglanır (video: "nesil 1 → nesil 40").
- RAM bütçesi: 20 × 64 MB ≈ 1,3 GB.
- **Ölçüt:** en iyi ve ortalama skorun nesiller boyunca anlamlı artışı; evrimle bulunan ayarların Faz 3'teki elle bulunanlarla en az eşit olması. Kazanan ayarlar 8×6 çok aşamalı ağda yeniden eğitilir.
- **Faz 4b (GPU, A100):** turnuva ve kendi kendine oyunları toplu GPU'da oynatma (binlerce tahta aynı anda, ağırlık güncellemesi scatter-add). A100'ün 40 GB belleğine birden çok 8×6 ağ sığar. Önce CPU profili alınır; GPU sürümü CPU sonuçlarıyla birebir karşılaştırılarak doğrulanır.

### Faz 5 — Konsey (≈2–3 gün)
- En iyi 3–5 ajanın değerlerinin ortalaması (ve alternatif: oylama, ağırlıklı ortalama).
- **Eşleştirilmiş ölçüm:** aynı tohumlarda konsey vs en iyi tek ajan.
- Maliyet uyarısı: konsey = N kat değerlendirme → aramada N kat yavaşlık. Bu yüzden ikinci seçenek: konseyi tek bir ağa **damıtmak** (konseyin değerleriyle tek ağı eğitmek).
- CNN üyesi (PDF madde 4): bu fazın dışında tutuldu, rekor yolunda gerekirse Faz 6+ deneyi.
- **Ölçüt:** konsey, en iyi tek ajana göre büyük kare oranlarında istatistiksel olarak anlamlı kazanç (veya kazanç yoksa bunu raporlayıp damıtılmış tek ağla devam).

### Faz 6 — Arama + oyun sonu tablosu (≈2–3 hafta)
**Faz 6a — Arama**
- Hızlı expectimax: transpozisyon tablosu, olasılık eşiğiyle şans düğümü budama, zamanla derinleştirme (iterative deepening), değerlendirme önbelleği.
- **Tile-downgrading** (Guei): büyük kareler varken tahtayı "küçültüp" ağın iyi bildiği bölgeye taşıma — 65536 için kritik teknik.
- Paralellik: oyunlar arası (8 thread = 8 oyun aynı anda); oyun içi paralel arama gerekmez.
- **Faz 6b — Oyun sonu tablosu (D1: hibrit).** Önce macroxue ve 2048EndgameTablebase'in yöntemi ve lisansı incelenir (kod kopyalamak değil, yaklaşımı anlamak; lisans uygunsa karşılaştırma için çalıştırılır).
  - Desen seçimi: büyük kareler köşe/kenarda sabit, kalan bölgede küçük kareler (ör. "yeni 64'ü 64'ün yanına getir" görevi). Her desen için geriye doğru dinamik programlama ile **başarı olasılığı + en iyi hamle** tablosu.
  - Bellek: küçük desenler yerelde (16 GB), büyükleri Colab'da (~80 GB) üretilip Drive'a sıkıştırılarak yazılır; disk üzerinde bellek eşlemeli (mmap) okuma.
  - Bağlama: tahta bir tablo desenine uyduğunda tablo konuşur, uymadığında konsey + expectimax. Geçiş noktası ölçülerek ayarlanır.
  - Kontrol: küçük desenlerde tablo sonuçları kaba kuvvet expectimax ile birebir doğrulanır.
- **Ölçüt (6a):** tablosuz, derinlik 3–4'te **32768 oranı ≥ %50** — bu bizim "saf RL" sayımız, videoda ayrı gösterilir.
- **Ölçüt (6b):** hibrit sistem 32768'de ≥ %80 ve 65536'da macroxue'nun %3,5'ini geçiyor (1000+ oyun).

### Faz 7 — Rekor denemesi (≈1–2 hafta, çoğu hesaplama süresi)
- Hesap bütçesi: derinlik 4–5'te oyun başı dakikalar; 1000 oyun / 8 thread ≈ günler. Önce 100 oyunluk pilot, sonra büyük koşu.
- Her oyunun tam tekrarı, her yeni maks. skor/kare için "o an" işareti.
- Bulut oturumlarında koşu: her oyun bitince sonuç + tekrar diske yazılır; oturum kesilirse kalan tohumlardan devam.
- **Ölçütler (kademeli):** 65536 > %3,5 (macroxue) → 65536 > %8,4 (güncel SOTA) → 1,7M skor → 131072 denemesi. Yan ölçüt: saf RL kısmı Guei 2021'i (32768 %72) geçiyor mu.

### Paralel hat — Video ve görselleştirme (her fazda küçük adımlar)
- Faz 1'den itibaren tekrar formatı sabit; Faz 2'de basit HTML tekrar oynatıcı.
- Her faz sonu: grafikler + "o an" tekrarları `runs/` altında.
- Kurgu en sonda (HyperFrames ile): nesil hikâyesi, konsey vs tek ajan, rekor anı.

---

## 4. Takvim özeti (kaba)

| Faz | Süre | Kümülatif |
|---|---|---|
| 0–1 Altyapı + motor | 2–3 gün | ~3 gün |
| 2 Temel ajan | 2–3 gün | ~1 hafta |
| 3 Güçlendirme | ~1 hafta | ~2 hafta |
| 4 Evrim | ~1 hafta | ~3 hafta |
| 5 Konsey | 2–3 gün | ~3,5 hafta |
| 6 Arama + tablo | 2–3 hafta | ~6 hafta |
| 7 Rekor | 1–2 hafta | ~7–8 hafta |

Süreler çoğunlukla **eğitim/hesap süresi**; kod yazma kısmı daha kısa.

---

## 5. Riskler

- **Mutlak rekor saf RL ile muhtemelen gelmez** (bkz. bölüm 0). Hikâye "RL rekoru" ekseninde kurulursa başarısızlık riski çok düşer.
- **RAM 16 GB:** 8×6 çok aşamalı ağ ve oyun sonu tabloları için asıl darboğaz. Ağırlıkları `float16`'ya sıkıştırma Faz 3'te denenebilir.
- **65536 temsili:** 4 bit/kare 32768'de doluyor; Faz 1'de karar verilmeli (bkz. Faz 1).
- **İstatistik:** büyük kare oranları küçük; %0,02 gibi oranlar için on binlerce oyun gerekir. Her iddia güven aralığıyla.
- **Colab oturum kopması / kredi bitmesi:** checkpoint her N dakikada bir Drive'a; her iş kaldığı yerden devam edebilir.

---

## 6. Kararlar (2026-10-13)

- **D1 — Hedef: Baştan hibrit.** Hedef mutlak rekor (65536 > %8,4). RL konseyi oyunun büyük kısmını oynar, oyun sonu tablosu (tablebase) kritik birleşmeleri çözer. Video hikâyesi: "öğrenen konsey + mükemmel hafıza".
- **D2 — Bulut: Google Colab, A100 + High-RAM.** Ayrıntılar en baştaki "Çalışma ortamı" bölümünde.
- **D3 — Dil: C++20 çekirdek (+ isteğe bağlı CUDA) + Python (uv) analiz.**
- **D4 — Video: kesin çıktı.** Log, tekrar ve soy kaydı Faz 0'dan itibaren tam detaylı; her faz sonunda "o an" tekrarları saklanır.

---

## 7. Faz sonuçları (ölçülmüş)

### Faz 0 — 2026-10-13 ✅ (Colab kısmı bekliyor)
- İskelet, CMake (C++20, `-O3 -march=native`), doctest, `eval`/`replay`/`bench` araçları, JSONL log, `colab/run.ipynb`.
- Testler yerelde yeşil (g++ 16.2) ve **Colab vekilinde yeşil**: Docker `ubuntu:22.04` + g++ 11.4.0 (Colab'ın derleyicisi). Gerçek Colab oturumu ve donanım ölçümü (`nproc`, `free -h`, `nvidia-smi`) kullanıcının ilk oturumunu bekliyor — notebook'un 1. hücresi.
- `eval --agent random`, 10.000 oyun: ort. skor 1093, 256 %7,95, 512 %0,02; 10.000 tekrarın tamamı `replay` ile birebir doğrulandı.

### Faz 1 — 2026-10-13 ✅
- **65536 kararı:** hızlı yol = `uint64_t`, 4 bit/kare, 32768'e kadar (tablolar 15+15'i birleştirmez). Tahtada iki 32768 olduğu an oyun **kesin yola** (`Grid` = 16 × `uint8_t`, üs ≤ 17) geçer; politika tahtayı **tile-downgrading** görünümüyle (`downgrade()`: en küçük eksik üs kaldırılır, üstündekiler bir iner) görür, böylece ajanlar 4 bitlik tahtayla çalışmaya devam eder. Yasal hamle kümesi korunur (test). Ayrıntılı tile-downgrading stratejisi Faz 6a'da.
- Satır tabloları (65536 giriş, sol/sağ + skor), dikey hamleler transpoz ile.
- **Hız: 165 M hamle/sn** tek iş parçacığı (ölçüt ≥ 20M); rastgele oyun 132k oyun/sn.
- Doğruluk: 4M rastgele tahta × 4 yön referansla %100 eşleşme; 2000 tam oyun kare kare referansla aynı; tekrar dosyası oyunu birebir üretiyor; iki 32768 → 65536 geçişi uçtan uca test edildi.
- Hibrit için `MaskedSpace`: sabit kare maskesi + serbest hücrelerin karma tabanlı indekslenmesi (gidiş-dönüş testli).
