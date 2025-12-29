# Model ve Algoritma Mantığı

Bu proje, bir "Konuşmacı Tanıma" (Speaker Identification) sistemi geliştirmek için **Derin Öğrenme (Deep Learning)** tekniklerini kullanmaktadır. Aşağıda, projede kullanılan modellerin, veri işleme adımlarının ve eğitim mantığının detayları açıklanmıştır.

## 1. Genel Yaklaşım: Contrastive Learning (Karşıtlıklı Öğrenme)

Bu projede klasik bir sınıflandırma (classification) yaklaşımı yerine, **Metric Learning** veya daha spesifik olarak **Contrastive Learning** (SimCLR benzeri bir yapı) kullanılmıştır.

*   **Neden?** Klasik sınıflandırma modelleri (örn: Bu ses Ahmet mi, Mehmet mi?) yeni bir kişi eklendiğinde modelin tamamen yeniden eğitilmesini gerektirir. Ancak Metric Learning yaklaşımında model, "kişileri tanımayı" değil, "seslerin benzerliğini ölçmeyi" öğrenir. Bu sayede model eğitildikten sonra, sisteme hiç görmediği yeni bir kişi eklense bile yeniden eğitime gerek kalmadan o kişiyi tanıyabilir.
*   **Nasıl?** Model, ses kayıtlarını **Embedding** adı verilen sayısal vektörlere dönüştürür. Hedefimiz şudur:
    *   Aynı kişiye ait seslerin vektörleri birbirine **yakın** olmalı.
    *   Farklı kişilere ait seslerin vektörleri birbirine **uzak** olmalı.

## 2. Model Mimarisi (`SpeakerEncoder`)

Kullanılan model, sesi temsil eden bir vektör (embedding) üreten bir **Evrişimli Sinir Ağıdır (Convolutional Neural Network - CNN)**.

*   **Dosya:** `src/model/speaker_encoder.py`
*   **Yapı:**
    *   Model, standart 2D konvolüsyon yerine **Separable Convolution (Ayrılabilir Evrişim)** blokları kullanır. Bu, modelin parametre sayısını ve işlem yükünü azaltarak daha hızlı çalışmasını sağlar.
    *   Giriş olarak **Mel Spectrogram** görüntülerini (sesin görsel frekans temsili) alır.
    *   Katmanlar boyunca sesin özniteliklerini (feature) çıkarır ve sıkıştırır.
    *   **Çıktı:** 256 boyutlu bir vektör (embedding). Son katmanda `L2 Normalization` uygulanır, böylece tüm vektörlerin boyu 1'e eşitlenir. Bu, benzerlik hesaplamasında (Cosine Similarity) kararlılık sağlar.

## 3. Veri Ön İşleme (`Preprocessing`)

Modelin sesi anlayabilmesi için ham ses dalgaları (wav) işlenmelidir.

*   **Dosya:** `src/audio/preprocessing.py`
*   **Örnekleme Hızı (Sample Rate):** 8000 Hz. (İnsan sesi için yeterlidir ve veri boyutunu düşük tutar).
*   **Mel Spectrogram:** Ses, zaman ve frekans boyutunda bir resme dönüştürülür (`librosa.feature.melspectrogram`).
    *   `n_fft=1024`, `hop_length=256`, `n_mels=256`.
    *   Bu, modelin giriş boyutunu belirler: `(1, 256, 32)`. Burada 32, zaman eksenindeki kare sayısıdır (yaklaşık 1 saniye).
*   **Veri Artırma (Data Augmentation):** Modelin gürültülü ortamlarda da çalışabilmesi için eğitim sırasında seslere rastgele bozulmalar eklenir:
    *   Arkaplan gürültüsü ekleme.
    *   Yankı (reverb/impulse response) ekleme.
    *   Zaman kaydırma.
    *   *Mantık:* Modele "temiz ses" ile "gürültülü ses"in aynı kişiye ait olduğu öğretilerek dayanıklılık artırılır.

## 4. Eğitim Süreci (`Training`)

Eğitim döngüsü, modelin doğru vektörleri üretmesini sağlamak için tasarlanmıştır.

*   **Dosya:** `src/training/train.py`
*   **Loss Fonksiyonu:** `NTxent_Loss` (Normalized Temperature-scaled Cross Entropy Loss).
*   **İşleyiş:**
    1.  Her adımda, bir konuşmacıdan iki farklı ses kesiti (veya aynı kesitin iki farklı versiyonu) alınır (`View 1` ve `View 2`).
    2.  Biri temiz bırakılırken, diğeri ağır şekilde veri artırma işlemine (gürültü vb.) tabi tutulur.
    3.  Model her ikisi için de embedding üretir.
    4.  Loss fonksiyonu, bu iki embedding'i birbirine **çekerken**, batch içindeki diğer tüm konuşmacıların embedding'lerinden **iter**.
    5.  Zamanla model, gürültüden bağımsız olarak kişinin ses karakteristiğini (voiceprint) yakalamayı öğrenir.

## 5. Çıkarım ve Tanıma (`Inference`)

Eğitim bittikten sonra sistem şu şekilde çalışır:

1.  **Kayıt (Enrollment):** Tanınması istenen kişilerin (örn: Ahmet, Ayşe) seslerinden embeddingler üretilir ve bir veritabanına (`embeddings/speakers.json`) kaydedilir. Bu onların "dijital imzası"dır.
2.  **Tanıma:** Yeni bir ses geldiğinde, model bu sesin embedding'ini üretir.
3.  **Karşılaştırma:** Yeni sesin vektörü ile veritabanındaki kayıtlı vektörler arasındaki **Cosine Similarity (Kosinüs Benzerliği)** hesaplanır.
4.  **Sonuç:** Benzerlik skoru belirli bir eşik değerin (threshold) üzerindeyse ve en yüksek skor kime aitse, o kişi tahmin edilir.

Bu yapı, modern ve ölçeklenebilir bir biyometrik tanıma sistemi iskeletidir.
