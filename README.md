# LLM Davranış Cerrahisi

> **EN summary:** Locating behaviors (sycophancy, length, confidence, emotion, formality, language, refusal) inside small open LLMs and steering them on a single 4 GB GPU. Behaviors sit mid-network, refusal sits late; steering works but only in a narrow range; refusal removal keeps capability (also checked on a 14B model); 'intelligence boosting' via a confidence direction did not work. A blind cross-family check showed the small LLM judge is unreliable outside clean text. Own mistakes and negative results are documented. Harmful artifacts are not published. Full English version: [README.en.md](README.en.md).

Bir dil modeli neden bazen dalkavukluk yapıyor, neden bazen "bunu cevaplayamam" diyor? Bu davranışların modelin içinde **nerede yaşadığını** merak ettim. Küçük açık kaynak modelleri açtım ve yedi davranışın hangi katmanda olduğunu ölçtüm: dalkavukluk, uzunluk, özgüven, duygu, resmiyet, dil ve reddetme. Sonra modeli o yönlere doğru **ittim** ve davranışın gerçekten değişip değişmediğine baktım.

Önceki çalışmam [llm-reddetme-yorumlanabilirlik](https://github.com/emirhnnnk-lab/llm-reddetme-yorumlanabilirlik) sadece reddetmeye bakıyordu. Bu repo onu 7 davranışa, 3 modele ve bir 14B karşılaştırmasına genişletiyor.

**Modeller:** Qwen2.5-0.5B, Qwen2.5-1.5B ve SmolLM2-1.7B (Instruct); karşılaştırma için Qwen2.5-14B (Ollama). **Donanım:** tek bir GTX 1650 (4 GB). Yani büyük bir laboratuvar yok, evdeki bilgisayar var.

## Nasıl yaptım (kısaca)
1. **Yönü bulmak:** her davranış için zıt örnek çiftleri hazırladım (mutlu ↔ üzgün gibi). Model iki tarafı okurken her katmanda içeride neyin farklı yandığına baktım. Aradaki fark, o davranışın **yönü**.
2. **Katmanı seçmek:** farkın en net olduğu katmanı ölçerek seçtim (Cohen's d, yani etki büyüklüğü). Tesadüf olmasın diye veriyi ikiye bölüp yönü iki kez buldum ve iki sonucu karşılaştırdım (split-half kararlılığı).
3. **Modeli itmek:** bulduğum yönü -6 ile +6 arası güçlerle modele ekleyip cevap ürettim. Ayrı bir model (**hakem**) her cevabı 1-5 arası puanladı. Ayrıca cevapların ne kadarının anlamsızlaştığını (**bozulma**) ölçtüm.
4. **Zekâ kontrolü:** müdahalenin modelin yeteneğini bozup bozmadığına standart testlerle baktım: MMLU, ARC-Challenge, GSM8K ve perplexity.

## Neler buldum

![derinlik](grafikler/1_davranis_derinligi.png)

**1. Davranışlar modelin ortasında, reddetme sonlara doğru.** Davranış yönlerinin çoğu modelin %45-65 derinliğinde. Kararlılık üç modelde de 0,99-1,00 çıktı; yani yön her seferinde aynı yerde bulunuyor. Reddetme ise Qwen 1.5B'de katman **21/28**'de (derinlik 0,78). Bu, önceki çalışmamda bulduğum sonucun aynısı; sonuç tekrarlandı.

**2. Küçük modelde güvenlik yüzeysel.** Qwen 0.5B ve SmolLM2'de net bir reddetme yönü yok (kararlılık 0,17 ve 0,08). SmolLM2 zararlı isteklerin zaten sadece %33'ünü reddediyor. Ortada bulunacak bir yön olmayınca onu kaldırmak da bir şey değiştirmiyor.

![doz-tepki](grafikler/2_doz_tepki_15b.png)

**3. Modeli itmek davranışı gerçekten değiştiriyor, ama dar bir aralıkta.** Qwen 1.5B'de +2 güç mutluluğu 3,75'ten 5'e, öfkeyi 2'den 5'e, resmiyeti 3,25'ten 4,25'e çıkarıyor. Mutluluk ve resmiyette cevaplar +2'de hâlâ düzgün cümleler. Öfkede ise işin aslı farklı: +2'de bile cevap büyük harfli, yarı anlamsız bir küfür dizisine dönüyor ve benim otomatik bozulma ölçütüm bunu yakalamadı (aşağıda anlatıyorum). ±4 ve üstünde çıktı dağılmaya başlıyor; öfke +4'te cevapların hepsi bozuk, model Çinceye ya da tekrar döngüsüne kayıyor.

**4. Reddetmeyi kaldırınca model aptallaşmıyor.** Qwen 1.5B'de zararlı istekleri reddetme oranı (HarmBench) %95'ten %50'ye iniyor, genel bilgi puanı (MMLU) 0,56'da kalıyor. Aynı işlemi rastgele bir yönle yaptığımda hiçbir şey olmadı (%90). Yani etki tesadüf değil, bulduğum yönün kendisinden geliyor.

![14b](grafikler/3_14b_karsilastirma.png)

**5. 14B'de de aynı tablo.** Topluluğun yayınladığı, reddetmesi kaldırılmış Qwen2.5-14B ile orijinali karşılaştırdım. Red oranı %82,5'ten %0'a iniyor; MMLU 0,83'ten 0,80'e, GSM8K 0,75'ten 0,75'e. Küçük modellerde gördüğüm desen büyükte de geçerli: **güvenlik eğitimi tek bir yöne sıkışmış ve ucuzca sökülebiliyor.** Bu da bence güvenliği yalnızca modele bırakmamak gerektiğini gösteriyor.

**6. Modele tek konuluk bir "yasak" eklenebiliyor.** Qwen 1.5B'de katman 15'te "yemek tarifi" konusunun yönünü buldum. Sadece bu konu algılandığında reddetme yönünü ekleyen bir kanca yazdım. Sonuç: yemek sorularının %93'ü reddedildi, diğer soruların sadece %7'si. Aynı yöntem kötüye de kullanılabileceği için kanca kodunu paylaşmadım. Bir not: reddedilen cevaplar düzgün cümle değildi; model "üzgünüm" deyip anlamsızlaşıyor.

**7. "Zekâ artırma" çalışmadı.** Qwen 1.5B'de MMLU ve ARC'den 60'ar soruyla ölçtüm; taban puanlar MMLU 0,58, ARC 0,73.
- *Katman atlamak:* ilk iki katmanı atlamak modeli yazı tura seviyesine düşürüyor (K0: MMLU 0,20). Katman 17 de hassas (0,35). Diğer katmanların çoğu tek tek atlanınca puan 0,50-0,62 arasında kalıyor.
- *Özgüven yönünü eklemek:* hiçbir güçte puanı artırmadı. En iyi sonuç hiç dokunmadığım hâl; +1'de 0,55, +4'te 0,22. Yani kendinden emin görünmek doğru cevap vermek demek değil.
- *Reddetme yönünü eklemek:* orta katmanlarda (4-20) MMLU yazı tura seviyesine iniyor; katman 22 ve sonrasında yetenek korunuyor (0,58-0,60). Bu, "reddetme sonlarda" bulgusuyla tutarlı.

Sonuç dosyası: `sonuclar/Qwen2.5-1.5B-Instruct/zeka_haritasi.json`. Örnek sayısı küçük; ±0,05'lik farklar gürültü olabilir.

## Hakeme ne kadar güvenebilirim? (08.11)
Davranış puanlarını küçük bir model veriyor: qwen2.5:3b. Puanladığı model de aynı aileden (Qwen). Açıkçası hakemin güvenilir olup olmadığını hiç ölçmemiştim. 30 cevabı rastgele seçtim (3 davranış × 10 güç) ve **farklı aileden** bir modele (Claude) aynı soruyla **kör** puanlattım; hakemin puanları gizliydi. Okunamayacak kadar bozuk metinler "değerlendirilemez" diye işaretlendi.

| Küme | n | Birebir aynı puan | ±1 içinde | Sıra korelasyonu |
|---|---|---|---|---|
| Düzgün metinler | 13 | 7 (%54) | 9 (%69) | 0,78 |
| Tonu seçilebilen bozuk metinler dahil | 21 | 9 (%43) | 13 (%62) | 0,60 |

- **Hakem ortayı kullanmıyor.** Tarafsız, bilgi veren cevaplara ("As an AI language model…") mutluluk için 3 yerine 1 veriyor. Puanlar uçlara yığılıyor.
- **Anlamsız metne de puan veriyor.** Okunamayan 9 metnin 3'üne 5, 6'sına 1 verdi; "değerlendirilemez" demiyor. Yani yüksek güçlerdeki (±4, ±6) ortalamaların bir kısmı aslında çöp metnin puanı. Bu, 3. bulgudaki "dar aralık" sonucunu destekliyor: güvenilir bölge yaklaşık ±2.
- **Sınırı:** Karşılaştırmayı bir insanla değil, farklı aileden başka bir modelle yaptım. Sıradaki adım benim 10-15 cevabı elle puanlayıp iki hakemi de kontrol etmem.
- **Özetle:** Düzgün metinlerde sıralama büyük ölçüde tutuyor (0,78), ama tek tek puanlara ve bozuk bölgedeki ortalamalara güvenilmez. Ayrıntı: `sonuclar/hakem_kalibrasyonu.json`.

## Hatalarım ve tutmayan denemeler

![gsm](grafikler/4_gsm_artefakt.png)

- **Matematikteki "artış" bir ölçüm hatasıydı.** İlk ölçümde reddetmeyi kaldırınca GSM8K puanı 0,15'ten 0,45'e çıktı. İnanmadım, cevapları tek tek kaydeden bir kontrol yazdım. Sebep 160 tokenlık sınırmış: orijinal modelin 20 cevabından 19'u yarıda kesiliyordu. Sınırı 512 yapınca orijinal 14/20, müdahaleli 13/20 aldı. **Gerçek bir fark yok.** Ham sonuç: [gsm_kontrol_ozet.json](sonuclar/Qwen2.5-1.5B-Instruct/gsm_kontrol_ozet.json).
- **Türkçe yönü ölçülüyor ama işe yaramıyor.** Yönü kararlı şekilde buldum, ama hiçbir güçte modele Türkçe cevap verdiremedim (hakem: 0).
- **Özgüven yönünün "ters çalıştığını" sandım, yanlış teşhismiş.** Hakem puanı +2'de 4,75'ten 3'e, +6'da 1'e düştü. İlk yorumum yönün işaretinin ters olduğuydu. Kodu ve cevapları tek tek okuyunca öyle olmadığını gördüm: yön doğru tanımlı. **Negatif taraf beklendiği gibi çalışıyor;** -2'de model "emin değilim, belki…" diye çekingenleşiyor. Pozitif tarafta iki sorun var. Birincisi, model zaten tavanda: hiç itmeden puanı 4,75, yükselecek yer yok. İkincisi, +4'te çıktı anlamsız bir kelime salatasına dönüyor ("No-InlineInformationInArticleTextDescription" gibi) ve hakem haklı olarak düşük puan veriyor. Benim **bozulma ölçütüm** ise sadece tekrar döngüsünü ve boş cevabı yakalıyordu; kelime salatasını kaçırıp %0 bozulma gösterdi. Yani hata yönde değil, ölçütümdeydi.
- **"Dil merceği" deneyi iki kez tutmadı.** Amacım modelin ara katmanlarda İngilizce mi "düşündüğünü" görmekti. İlk denemede model görevi anlamadı. İkincisinde birkaç örnek göstererek denedim; küçük modeller son katmanda bile doğru Türkçe cevabı sadece %5-10 oranında veriyor. Bu ölçekte bu yöntem işe yaramıyor.

## Neyi yayınlamadım
Bu repoda şunlar **yok**: model ağırlıkları, reddetme yönü vektörleri, zararlı prompt listeleri, reddetmeyi kaldıran müdahale kodu ve sansürsüz model çıktıları. Paylaştıklarım: ölçüm kodu (davranış yönleri, zekâ haritası, dil merceği, hakem), bütün sayısal sonuçlar ve grafikler. Reddetmeyle ilgili sadece **sayıları** yayınladım. `d3_zeka.py` reddetme yönünü yerelde duran özel bir dosyadan okuyor; o dosya olmadan diğer kısımlar yine çalışıyor.

## Çalıştırma
```bash
pip install torch transformers pandas matplotlib requests
# benchmark dosyaları (MMLU, ARC, GSM8K parquet; HarmBench/XSTest csv) veri/benchmark/ altına
python -m deneyler.d1_davranis <model_klasoru>      # davranış yönleri + yönlendirme
python -m deneyler.olc sonuclar/<model>             # LLM hakem puanlaması
python grafik_uret.py
```

## Sınırlar
- Modeller de test setleri de küçük: davranış başına 8 soru, GSM8K'da 20 soru. Sonuçlar kesin hüküm değil, eğilim gösteriyor.
- Tek GPU, fp16. 14B karşılaştırmasını Ollama'nın sıkıştırılmış (kuantize) modelleriyle yaptım.
