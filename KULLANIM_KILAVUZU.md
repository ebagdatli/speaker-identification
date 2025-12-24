
# Hoparlör Tanıma Sistemi - Kullanım Kılavuzu

Bu proje, ses dosyalarından kişi tanıması yapmak için geliştirilmiştir. Aşağıdaki adımları takip ederek modelleri eğitebilir, veri ekleyebilir ve sistemi çalıştırabilirsiniz.

## 1. Kurulum

Öncelikle gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

## 2. Veri Hazırlığı (Dataları Nasıl Koyacağız?)

Sistemin kişileri tanıması için ses verilerine ihtiyacı vardır. Verilerinizi `data/raw` klasörü altına şu yapıda eklemelisiniz:

```
data/
  raw/
    Ahmet/
      ses_kaydi1.wav
      ses_kaydi2.wav
    Mehmet/
      ses_kaydi1.wav
    Ayse/
      konusma.wav
    Mehmet/
      ses_kaydi1.mp3
```

Her kişi için ayrı bir klasör oluşturun. Eğer dosyalarınız `.mp3`, `.flac` veya `.m4a` formatındaysa, aşağıdaki komutu çalıştırarak onları `.wav` formatına çevirebilirsiniz:

```bash
python src/audio/convert_wav.py
```

Bu işlem tüm alt klasörleri tarar ve bulduğu dosyaların `.wav` kopyalarını oluşturur.

## 3. Model Eğitimi (Nasıl Eğiteceğiz?)

Verileri ekledikten sonra modeli eğitmek için:

1. `notebooks/02_training.ipynb` dosyasını açın.
2. Tüm hücreleri çalıştırın.
   - Bu işlem, `data/raw` altındaki verileri okur.
   - Modeli eğitir.
   - Eğitilen modeli `models/speaker_encoder.pt` olarak kaydeder.

Alternatif olarak, kodları direkt script üzerinden çalıştırmak isterseniz (eğer ayarlandıysa) `src/training/train.py` dosyasını kullanabilirsiniz, ancak notebook üzerinden ilerlemek daha kolaydır.

## 4. Konuşmacı Kaydı (Embedding Oluşturma)

Eğitilen modeli kullanarak, mevcut kişilerin "ses imzalarını" (embedding) oluşturmanız gerekir. Bu sayede sistem, yeni gelen bir sesin kime ait olduğunu bu imzalarla karşılaştırarak bulur.

1. `notebooks/03_embedding_generation.ipynb` dosyasını açın.
2. Tüm hücreleri çalıştırın.
   - Bu işlem, `data/raw` altındaki her kişi için bir ses imzası oluşturur.
   - Sonuçları `embeddings/speakers.json` dosyasına kaydeder.

## 5. Sistemi Çalıştırma (Inference)

Artık sistem kullanıma hazırdır. Yeni bir ses dosyasının kime ait olduğunu bulmak için:

1. `notebooks/04_inference.ipynb` dosyasını açın.
2. `test_file` değişkenine test etmek istediğiniz ses dosyasının yolunu yazın:
   ```python
   test_file = "../data/raw/Ahmet/yeni_ses.wav"
   ```
3. Hücreleri çalıştırın.
   - Sistem size dosyanın kime ait olduğunu (`Ahmet`, `Mehmet` vb.) söyleyecektir.

## Özet Akış

1. **Veri Ekle**: `data/raw/<İsim>/<Dosya>.wav`
2. **Eğit**: `02_training.ipynb`
3. **Kaydet**: `03_embedding_generation.ipynb`
4. **test Et**: `04_inference.ipynb`
