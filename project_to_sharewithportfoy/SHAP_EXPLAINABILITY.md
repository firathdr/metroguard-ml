# SHAP açıklanabilirlik notu

## Neden SHAP?

Anomali tespitinde model çıktısı genellikle tek bir skordur. Bu skor alarmın ne zaman oluştuğunu söyler, ancak bakım ekibine “hangi sinyaller bu pencereyi olağandışı yaptı?” sorusunun yanıtını vermez. SHAP (SHapley Additive exPlanations), model çıktısını özellik katkılarına ayırarak bu boşluğu kapatır.

MetroGuard’da her örnek 12 adet 5 dakikalık bin içeren 60 dakikalık bir penceredir. Her bin analog sensörler için ortalama, standart sapma, minimum, maksimum ve son değer; dijital sensörler için aktif oranı, geçiş sayısı ve son değer taşır. SHAP önce bu düzleştirilmiş model girdisini açıklar, sonra katkılar aynı fiziksel sensör grubuna toplanır.

## Yorumlama kuralı

Isolation Forest’ın `score_samples` çıktısında düşük değer daha anomali demektir. Kod bu nedenle TreeSHAP katkılarını ters çevirir ve raporlarda şu kuralı kullanır:

`pozitif SHAP katkısı → anomali skorunu artırır → daha şüpheli davranış`

Global önem için `mean(|SHAP|)` kullanılır. Bu, yönü değil katkı büyüklüğünü ölçer. Yerel açıklamada işaret korunur; böylece tek bir pencere için sensörlerin anomali skorunu artırıp azaltması görülebilir.

## Kod akışı

```text
scaled 60-minute windows
        ↓
TreeSHAP on Isolation Forest
        ↓
negative score_samples direction is inverted
        ↓
engineered features → sensor groups
        ↓
global ranking + local explanation + time/sensor heatmap
```

Uygulama [`src/metroguard/explainability.py`](../src/metroguard/explainability.py) içindedir; pipeline entegrasyonu [`src/metroguard/pipeline.py`](../src/metroguard/pipeline.py) içindeki `write_isolation_shap` fonksiyonuyla yapılır.

## Çıktıların anlamı

| Çıktı | Portföy sorusu |
|---|---|
| `isolation_forest_shap.csv` | Genel olarak hangi sensörler daha etkili? |
| `isolation_forest_shap_local.csv` | Seçilen tek anomalik pencerede hangi sensörler etkili? |
| `isolation_forest_shap_time.csv` | Etki 60 dakikalık pencerenin hangi bölümünde yoğunlaşıyor? |
| `isolation_forest_shap.png` | Global sensör önem grafiği |
| `isolation_forest_shap_local.png` | İşaretli yerel katkı grafiği |
| `isolation_forest_shap_heatmap.png` | Zaman-sensör SHAP ısı haritası |

Bu görseller model davranışını açıklamak içindir; fiziksel kök neden, bakım teşhisi veya nedensellik iddiası değildir.
