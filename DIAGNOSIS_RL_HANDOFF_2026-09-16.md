# Sol Ultra için teşhis — 16 Eylül 2026

Bu not mevcut kodu/raporları okuyarak hazırlanmıştır; bu turda simülasyon değiştirilmedi, yeni yıllık test çalıştırılmadı. Kullanıcı tek ajanla, az tokenla, fiziksel gerçekçilik korunarak çalışılmasını istiyor. Önce mevcut çalışma ağacını koru; toplu revert yapma. Aşağıdaki öneriler henüz uygulanmadı.

## 1. Öncelik: %35 koşusundaki inşaatın durması

Kanıt: `reports/rl_learning_verification_final_20260915/training/train_6/result.json`.
- Seed 42, 51.840 tick, 6 kişi hayatta, sıfır processing error. Skor tick 20.000'den sona kadar kaydedilen bütün checkpointlerde tam 35.
- `supply_audit.active_sites[0]`: greenhouse `struct_53`, (973,1007), başlangıç tick 15638. 127602.4 kg manifest tamamen teslim edilmiş; teslim tick 15665, dört sefer.
- Aşama utility_trenching; 6 segmentin 5'i tamamlanmış. Boru işçiliği 10.990054/12 saat; toplam yapı işçiliği 10.990054/180 saat. Quality hold yok. Final aktif insan/robot sayıları sıfır.
- Hat: (973,1007),(973,1006),(974,1006),(975,1006),(976,1006),(977,1006),(978,1006). `_construction_workfront` sonraki iş hücresini (977,1006) seçer; boru çalışması tam hücreye varmayı gerektirir.
- Son 60 günün eylem sayımında build yok. Üretim durmuş değil: iron çevrimleri train3=1153, train5=1177, train6=1288. Yalnız üretim hızını/ödülü artırmak bu vakayı açıklamaz.
- Final shared_work_order boş. Bunun tek başına tüm yıl boyunca boş kaldığını kanıtlamadığına dikkat et; son tick deadline maskesi ayrıca order silebilir.

Kod bağlantıları:
- `src/agents/decision.py:1562`: `_select_shared_capacity_order`, aktif site varsa adayları yalnız o tip ile sınırlar. Başka yapılar bu site bitmeden açılamaz.
- Aynı dosya 3406+: `_shared_colony_work_decision`; order None ise order temizleniyor. 3490–3630: kalıcı ekip, yerel/uzak ayrımı, tek asgari süpervizör.
- `src/orchestration/engine.py:6347`: `_construction_workfront`; 10185+: fiziksel segment ilerlemesi; 14880–14950: rover, tam hücreye yürüyüş ve EVA kapıları.
- `src/agents/strategic_rl.py:420+`: choose/mask/commitment/stall mantığı.

KESİN BULGU: yarım kalmış, malzemesi teslim edilmiş site + uzun skor platosu var. Hangi branch'in bunu ürettiği henüz kanıtlanmadı. Bunu doğrudan “RL unuttu” veya “kesin pathfinding bug” diye sunma.

İlk iş: train_6/policy_start.json ve tactical_policy_start.json ile mevcut deney yükleme yolunu ve seed42'yi kullanarak yalnız başarısız koşuyu yeniden üret. İlk 15–20 bin tick çevresinde hedefli iz al; altı yılı tekrar koşma. Site işçiliği ilerlemiyorken aday/actionable, commitment/release reason, assigned/shift crew, rover/pending expedition, kişilerin mevcut görevleri, workfront erişimi ve EVA ret nedenini kaydet. İlk durma anını yakaladıktan sonra küçük fixture/regresyon testi üret. Sonrasında yalnız kanıtlanan ret/atama/rota sorununu düzelt. Watchdog eklenecekse gerekçeli fiziksel yeniden planlama yapmalı; bedava inşaat/işçilik/ışınlanma üretmemeli.

Karşı örnek: seed303 control=20.3, trained=33.5 raporlarının final active_sites listesi boş. Bütün düşük skorları aynı site hatasına bağlama; ikinci ayrı inceleme gerektiriyor.

## 2. RL çalışıyor; düzenli gelişim henüz kanıtlı değil

`reports/rl_learning_verification_final_20260915/comparison.json` ve REPORT.md:
- Ardışık eğitim: 90.9,93.4,95.0,89.2,93.4,35.0.
- Altı eşleşmiş dondurulmuş değerlendirme: ortalama 76.17 -> 82.43 (+6.27 puan), üç kazanım/üç kayıp, iki yönlü paired sign-flip p=0.3125.
- Q freeze/source hash kontrolleri geçmiş. Bu örneklem güvenilir genelleme veya %100 başarım kanıtı değil. Öte yandan sıfır öğrenme olduğunu da kanıtlamıyor.
- %35 mekanik durmayı çözmeden hiperparametre/ödül taraması yapma. Q değişmesi tek başına öğrenmenin faydasını göstermez.

## 3. RL'nin kontrol sınırları

Stratejik politika kapasite hedefini seçiyor; taktik/operasyonel politikalar çaba ve bazı davranışları etkiliyor. Üretim reçetesi seçimi `_shared_supply_decision` içindeki uygun refine listesinin ilk elemanı; iş gücü dağıtımının önemli bölümü kurallı. Politika kontrol etmediği üretim/atama kararlarını öğrenerek düzeltemez.

P0 çözüldükten sonra halen darboğaz varsa tek, küçük bir RL karar alanı değerlendir: uygun üretim işlerinin önceliği VEYA üretim/inşaat iş gücü tahsisi. İkisini birden ekleme. Fiziksel uygunluk maskesi + gerçek gecikmeli çıktı kredisi + ziyaret/eylem sayımları + dondurulmuş karşılaştırma olmadan yeni Q tablosu eklemek kanıt değildir. Önce gözlenen darboğazı ölç; yeni mekanik şartmış gibi davranma.

## 4. Uygulama/kanıt sırası

1. Train6 durmasını yeniden üret, nedenini daralt, küçük regresyonla düzelt.
2. Aynı başarısız başlangıçla koşunun artık ilerlediğini doğrula; skor hedefini değiştirme.
3. Seed303 düşük sonuçlarını ayrı incele; başarılı ~%90 koşularının kalan kategori/BOM/işçi zamanı eksiklerini sayısallaştır. Bu not onların tamamının kök nedenini çözmüş değildir.
4. Ancak sonra gereken tek RL kontrol değişikliğini yap. Kod sürümleri arasında eski deneyleri doğrudan öğrenme kanıtı diye birleştirme.
5. Eğitim, checkpoint seçimi için validation ve nihai test seedlerini ayır. İncelenmiş altı seed artık geliştirme verisi sayılmalı. Yayında doğrulanmış checkpoint kullanılabilir; eğitimde keşif sürer. Bu canlı politikada durmadan monoton artış garantisi değildir.

Skoru şişirme, bedava malzeme/robot emeği verme, %100'e zorlayan hedef değişikliği yapma. Canlı ekranda öğrenme iddiası için yalnız final skor değil aynı koşullardaki başlangıç politikası karşılaştırması, tamamlanan kapasite, görev süreleri ve hayatta kalma gösterilebilir. Operasyonel kapasite (ör. sera hizmeti/olgunluğu) fiziksel kurulu kapasiteden ayrı gösterilmeli; yapay heyecan için dalgalanma üretilmemeli.

Teslim iddiası: %100 henüz gösterilmedi; fiziksel olarak imkânsız olduğu da gösterilmedi. Öncelik daha çok yıl eğitmek değil, öğrenmenin değerlendirmesini bozan yürütme sorununu ayırmaktır.
