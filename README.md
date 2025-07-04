# MultiZoo Hayvan Sınıflandırma Projesi

Bu proje, Vision Transformer (ViT) mimarisini kullanarak çok sınıflı hayvan görüntülerini sınıflandırmak için geliştirilmiş bir derin öğrenme uygulamasıdır.

## 🚀 Proje Özeti

MultiZoo projesi, farklı hayvan türlerini görüntülerden otomatik olarak tanımlayabilen bir yapay zeka modeli ve kullanıcı dostu bir arayüz içermektedir. Proje, modern derin öğrenme tekniklerini kullanarak yüksek doğruluk oranları elde etmeyi hedeflemektedir.

## 🎯 Özellikler

- **Vision Transformer (ViT)** mimarisi ile güçlü görüntü sınıflandırma
- **Tkinter** tabanlı grafiksel kullanıcı arayüzü
- **Gerçek zamanlı tahmin** ve güven skoru gösterimi
- **Detaylı performans analizi** ve görselleştirme araçları
- **Sınıf dağılımı analizi** ve confusion matrix görselleştirme
- **Öğrenme eğrilerinin** izlenmesi ve analizi
- **Çoklu sınıf** destekli yapı (90+ hayvan türü)

## 📂 Proje Yapısı

```
yazlab 3/
├── multizoo_classifier.py      # Ana eğitim ve model dosyası
├── multizoo_app.py             # Tkinter GUI uygulaması
├── analyze_results.py          # Sonuç analizi ve görselleştirme
├── requirements.txt            # Gerekli Python paketleri
├── multizoo_transformer_model.pth  # Eğitilmiş model ağırlıkları
├── results/                    # Çıktı görselleri ve sonuçlar
├── class_distribution.png      # Sınıf dağılımı grafiği
├── confusion_matrix.png        # Confusion matrix
├── learning_curves.png         # Öğrenme eğrileri
└── README.md                   # Bu dosya
```

## 🛠️ Kurulum

### 1. Gereksinimler

Python 3.8 veya üzeri sürümü gereklidir.

### 2. Sanal Ortam Oluşturma (Önerilen)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# veya
source venv/bin/activate  # Linux/Mac
```

### 3. Gerekli Paketleri Yükleme

```bash
pip install -r requirements.txt
```

## 📋 Gerekli Paketler

- `torch` - PyTorch derin öğrenme framework'ü
- `torchvision` - Görüntü işleme ve dönüşüm araçları
- `timm` - Vision Transformer modeli için
- `numpy` - Sayısal hesaplamalar
- `pandas` - Veri analizi
- `scikit-learn` - Makine öğrenmesi metrikleri
- `matplotlib` - Görselleştirme
- `pillow` - Görüntü işleme
- `tqdm` - İlerleme çubuğu

## 🚀 Kullanım

### 1. Model Eğitimi

```bash
python multizoo_classifier.py
```

Bu komut:
- Veri setini yükler ve eğitim/doğrulama setlerine ayırır
- Vision Transformer modelini başlatır
- Modeli eğitir ve performans metriklerini izler
- En iyi modeli `multizoo_transformer_model.pth` dosyasına kaydeder
- Öğrenme eğrilerini görselleştirir

### 2. GUI Uygulaması

```bash
python multizoo_app.py
```

Bu komut grafiksel kullanıcı arayüzünü açar. Arayüz üzerinden:
- Görüntü yükleme
- Tahmin yapma
- Güven skoru görüntüleme
- Top-5 olasılık dağılımı grafiği

### 3. Sonuç Analizi

```bash
python analyze_results.py
```

Bu komut:
- Sınıf dağılımını analiz eder
- Confusion matrix oluşturur
- Detaylı performans metriklerini hesaplar

## 🔧 Yapılandırma

### Temel Parametreler

`multizoo_classifier.py` dosyasında aşağıdaki parametreler yapılandırılabilir:

```python
BATCH_SIZE = 16              # Batch boyutu
NUM_EPOCHS = 30              # Eğitim epoch sayısı
IMAGE_SIZE = 224             # Görüntü boyutu
MODEL_NAME = "vit_base_patch16_224"  # ViT model türü
LEARNING_RATE = 3e-4         # Öğrenme oranı
```

### Veri Artırma Teknikleri

Proje aşağıdaki veri artırma tekniklerini kullanır:

- **Geometrik Dönüşümler**: Yatay/dikey çevirme, döndürme
- **Renk Dönüşümleri**: Parlaklık, kontrast, doygunluk ayarları
- **Afin Dönüşümleri**: Ölçeklendirme, öteleme
- **Perspektif Değişiklikleri**: Derinlik efektleri
- **Gri Tonlama**: Rastgele renk kanalı azaltma

## 📊 Model Performansı

### Teknik Özellikler

- **Mimari**: Vision Transformer (ViT) Base Patch16 224
- **Optimizasyon**: AdamW optimizer
- **Öğrenme Oranı Programı**: OneCycleLR
- **Loss Fonksiyonu**: Weighted Cross-Entropy Loss
- **Regularizasyon**: Weight Decay, Dropout
- **Erken Durdurma**: 5 epoch patience ile

### Performans Metrikleri

Model performansı şu metriklerle değerlendirilir:

- **Doğruluk (Accuracy)**: Genel sınıflandırma doğruluğu
- **Kesinlik (Precision)**: Sınıf bazında kesinlik
- **Duyarlılık (Recall)**: Sınıf bazında duyarlılık
- **F1-Score**: Kesinlik ve duyarlılığın harmonik ortalaması
- **Confusion Matrix**: Sınıf bazında karışıklık matrisi

## 🎨 Görselleştirme

### Öğrenme Eğrileri
- Eğitim ve doğrulama kaybı
- Eğitim ve doğrulama doğruluğu
- Öğrenme oranı değişimi
- Performans metrikleri eğrileri

### Sınıf Analizi
- Sınıf dağılımı histogram
- Confusion matrix ısı haritası
- Performans metriklerinin tablosu

## 🔍 Desteklenen Hayvan Sınıfları

Model 90+ farklı hayvan türünü sınıflandırabilir:

- **Memeli Hayvanlar**: Kedi, köpek, fil, aslan, kaplan, vb.
- **Kuşlar**: Kartal, papağan, baykuş, ördek, vb.
- **Deniz Hayvanları**: Balina, köpekbalığı, ahtapot, vb.
- **Sürüngenler**: Yılan, kertenkele, kaplumbağa, vb.
- **Böcekler**: Kelebek, arı, böcek, vb.

## 🚨 Hata Ayıklama

### Yaygın Sorunlar

1. **CUDA Hatası**: GPU kullanılabilir değilse model otomatik olarak CPU'ya geçer
2. **Bellek Hatası**: Batch boyutunu azaltın
3. **Model Yükleme Hatası**: Model dosyasının varlığını kontrol edin
4. **Görüntü Formatı**: Desteklenen formatlar: jpg, jpeg, png

### Performans Optimizasyonu

- **GPU Kullanımı**: CUDA destekli GPU için otomatik geçiş
- **Veri Yükleme**: Çoklu thread ile hızlandırılmış veri yükleme
- **Bellek Yönetimi**: Pin memory özelliği ile RAM optimizasyonu

## 📈 Gelecek Geliştirmeler

- [ ] Model ensemble teknikleri
- [ ] Test-Time Augmentation (TTA)
- [ ] Grad-CAM görselleştirme
- [ ] Web tabanlı arayüz
- [ ] Mobil uygulama geliştirme
- [ ] Daha fazla hayvan türü desteği

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir ve açık kaynak olarak paylaşılmıştır.

## 📞 İletişim

Proje ile ilgili sorularınız için:
- GitHub Issues bölümünü kullanabilirsiniz
- Proje dokümantasyonunu inceleyebilirsiniz

---

## 🔍 Teknik Detaylar

### Vision Transformer (ViT) Mimarisi

Bu proje, görüntü sınıflandırma için Vision Transformer mimarisini kullanmaktadır:

- **Patch Embedding**: Görüntüler 16x16 patch'lere bölünür
- **Positional Encoding**: Patch'lerin konumsal bilgisi eklenir
- **Multi-Head Attention**: Çok başlı dikkat mekanizması
- **Feed-Forward Network**: MLP katmanları
- **Classification Head**: Son sınıflandırma katmanı

### Eğitim Stratejisi

1. **Transfer Learning**: ImageNet ön-eğitimli model kullanımı
2. **Data Augmentation**: Kapsamlı veri artırma teknikleri
3. **Balanced Sampling**: Sınıf dengesizliği için ağırlıklı loss
4. **Learning Rate Scheduling**: OneCycleLR programı
5. **Early Stopping**: Overfitting'i önleme

### Performans Değerlendirmesi

Model performansı çok boyutlu olarak değerlendirilir:

- **Macro/Micro Averaged Metrics**: Sınıf dengesizliği için
- **Per-Class Analysis**: Sınıf bazında detaylı analiz
- **Confusion Matrix**: Sınıf karışıklığı analizi
- **Learning Curves**: Eğitim sürecinin izlenmesi

Bu README dosyası, projenin tüm özelliklerini ve kullanım şeklini kapsamlı bir şekilde açıklamaktadır.
