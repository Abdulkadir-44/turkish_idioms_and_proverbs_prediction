# T5 Atasözleri ve Deyimler Üretme Modeli

Bu proje, Türkçe atasözleri ve deyimlerin anlamlarını bağlama dayalı olarak tahmin edebilen ve bu cümleleerin hangi kategoriye(öğüt,sabır,eleştiri,diğer) ait olduğunu tahmin edebilen bir T5 (Text-to-Text Transfer Transformer) modelini geliştirmeyi ve bir web arayüzü aracılığıyla sunmayı amaçlamaktadır. Proje, Hugging Face Transformers kütüphanesi kullanılarak, özel bir veri kümesi üzerinde eğitilmiş ve dağıtılabilir bir Flask uygulaması olarak tasarlanmıştır.

## 🌟 Özellikler

- **T5 Modeli**: Metinden metine dönüşüm yetenekleriyle bilinen güçlü bir transformatör modeli kullanır.
- **Özel Veri Kümesi**: Türkçe atasözleri ve deyimler üzerine odaklanmış özel bir veri kümesiyle eğitilmiştir.
- **Flask Web Arayüzü**: Kullanıcıların metin girişi yaptıktan sonra girilen metinin atasözü/deyim ve hangi kategoriye ait olabileceğini tahmin etmeyi hedefleyen basit ve etkileşimli bir arayüz.
- **Eğitim Betiği**: Modelin veri ön işleme, eğitim ve değerlendirme aşamalarını içeren ayrı bir Python betiği (`training/main.py`).
- **Görselleştirme**: Eğitim sürecindeki kayıp (loss) değerlerini gösteren grafik çıktısı.
- **Metrik Değerlendirme**: Modelin performansını ROUGE ve BLEU metrikleriyle değerlendirme.

## 🚀 Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

### 1. Depoyu Klonlayın

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
(Not: `your-username/your-repo-name` kısmını kendi GitHub kullanıcı adınız ve depo adınızla değiştirin.)

### 2. Sanal Ortam Oluşturma (Önerilir)

Python bağımlılıklarını izole etmek için bir sanal ortam oluşturmanız şiddetle tavsiye edilir.

```bash
python -m venv venv
```

### 3. Sanal Ortamı Etkinleştirin

- **Windows:**
  ```bash
  .\venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 4. Gerekli Kütüphaneleri Yükleyin

```bash
pip install -r requirements.txt
```

## 🛠️ Kullanım

Bu proje iki ana bölümden oluşmaktadır: Model Eğitimi ve Web Uygulaması.

### Model Eğitimi

Modeli kendi veri kümenizle eğitmek veya mevcut modeli yeniden eğitmek isterseniz:

1.  Veri kümesinin `dataset/atasozleri_deyimler_50_yeni.csv` konumunda olduğundan emin olun.
2.  `training/main.py` betiğini çalıştırın:

    ```bash
    python training/main.py
    ```
    Bu betik, modeli eğitecek, `./mt5_atasozleri_checkpoints` altında kontrol noktalarını kaydedecek ve `./mt5_atasozleri_final_model` altında son modeli ve tokenizer'ı depolayacaktır. Ayrıca eğitim kayıp grafiklerini ve değerlendirme metriklerini gösterecektir.

### Web Uygulaması

Eğitilmiş modeli kullanarak web arayüzünü çalıştırmak için:

1.  Modelin (`./training/mt5_atasozleri_checkpoints/checkpoint-1800` gibi bir kontrol noktası veya `./mt5_atasozleri_final_model` gibi kaydedilmiş bir model) mevcut olduğundan emin olun.
2.  Ana Flask uygulamasını çalıştırın:

    ```bash
    python app.py
    ```
3.  Uygulama çalıştıktan sonra, web tarayıcınızı `http://127.0.0.1:5000` adresine yönlendirin.

## 📂 Proje Yapısı

```
.
├── app.py                     # Ana Flask uygulaması
├── requirements.txt           # Python bağımlılıkları
├── training/
│   └── main.py                # Model eğitim betiği
├── templates/
│   └── index.html             # Web arayüzü HTML dosyası
└── dataset/
    └── atasozleri_deyimler_50_yeni.csv # Eğitim veri kümesi
```

## 📈 Gelecek Geliştirmeler

-   Model performansını artırmak için daha büyük ve çeşitli veri kümeleri kullanma.
-   Daha gelişmiş model mimarileri veya ince ayar teknikleri deneme.
-   Kullanıcı arayüzüne ek özellikler ekleme (örneğin, farklı atasözü kategorileri seçme).
-   Model çıkarımı için daha iyi performans optimizasyonları.
-   Docker veya diğer dağıtım yöntemleriyle kolay dağıtım.

---
