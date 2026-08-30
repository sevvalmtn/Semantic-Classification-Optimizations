# Semantic Classification Optimizations

Türkçe soru-cevap çiftleri üzerinde **semantik sınıflandırma** yapan ve farklı optimizasyon algoritmalarını (GD, SGD, Adam) karşılaştıran bir projedir. Diferansiyel Denklemler dersi Ödev 1 kapsamında hazırlanmıştır.

## 🎯 Proje Özeti

Bu projede:
- **LM Studio** kullanılarak yerel olarak çalıştırılan **Turkish-Gemma-9B** modeli ile Türkçe soru-cevap çiftleri üretilmiştir
- Üretilen çiftler, doğru eşleşme (`+1`) ve yanlış eşleşme (`-1`) olarak etiketlenmiştir
- **HuggingFace**'den çekilen [`ytu-ce-cosmos/turkish-e5-large`](https://huggingface.co/ytu-ce-cosmos/turkish-e5-large) embedding modeli ile metinler vektörleştirilmiştir
- Gradient Descent (GD), Stochastic Gradient Descent (SGD) ve Adam optimizasyon algoritmaları karşılaştırılmıştır
- Sonuçlar **Loss**, **Accuracy** ve **t-SNE** grafikleri ile görselleştirilmiştir

## 📁 Proje Yapısı

```
├── optimizations.ipynb   # Ana notebook — model eğitimi ve görselleştirme
├── train_test.csv        # LM Studio ile üretilmiş soru-cevap veri seti (200 satır)
├── .gitignore
└── README.md
```

## 🗂️ Veri Seti

`train_test.csv` dosyası 4 sütundan oluşur:

| Sütun   | Açıklama |
|---------|----------|
| `Soru`  | Türkçe soru metni |
| `Cevap` | Türkçe cevap metni |
| `Etiket`| `+1` (doğru eşleşme) veya `-1` (yanlış eşleşme) |
| `Küme`  | `Eğitim` (100 satır) veya `Test` (100 satır) |

### Veri Seti Nasıl Oluşturuldu?

1. **[LM Studio](https://lmstudio.ai/)** indirilip kurulur
2. LM Studio içinden **Turkish-Gemma-9B-T1** modeli indirilir
3. Model yerel olarak çalıştırılarak Türkçe soru-cevap çiftleri üretilir
4. Doğru soru-cevap eşleşmeleri `+1`, yanlış eşleşmeler `-1` olarak etiketlenir
5. Veriler `train_test.csv` formatında kaydedilir

## 🤗 HuggingFace Model Kullanımı

Projede embedding için [`ytu-ce-cosmos/turkish-e5-large`](https://huggingface.co/ytu-ce-cosmos/turkish-e5-large) modeli kullanılmaktadır. Model, `sentence-transformers` kütüphanesi ile otomatik olarak çekilir:

```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("ytu-ce-cosmos/turkish-e5-large")
```

> İlk çalıştırmada model otomatik olarak HuggingFace Hub'dan indirilir (~1.2 GB). Sonraki çalıştırmalarda cache'den yüklenir.

## ⚙️ Kurulum

```bash
# Gerekli kütüphaneleri yükleyin
pip install pandas numpy torch matplotlib sentence-transformers scikit-learn huggingface_hub
```

## 🚀 Çalıştırma

1. Bu repoyu klonlayın:
   ```bash
   git clone https://github.com/KULLANICI_ADINIZ/Semantic-Classification-Optimizations.git
   cd Semantic-Classification-Optimizations
   ```

2. Jupyter Notebook'u açın:
   ```bash
   jupyter notebook optimizations.ipynb
   ```

3. Hücreleri sırasıyla çalıştırın

## 📊 Optimizasyon Algoritmaları

| Algoritma | Learning Rate | Batch Size | Açıklama |
|-----------|:---:|:---:|----------|
| **GD** (Gradient Descent) | 0.5 | Tüm veri | Tüm eğitim verisini kullanarak güncelleme yapar |
| **SGD** (Stochastic GD) | 0.1 | 1 | Her seferinde tek bir örnek ile güncelleme yapar |
| **Adam** | 0.03 | 32 | Adaptif öğrenme oranı kullanan modern optimizasyon |

- Her algoritma **5 Run × 100 Epoch** boyunca eğitilir
- Aynı run'da tüm algoritmalar **aynı başlangıç ağırlıklarından** (w) başlar — adil karşılaştırma için

## 📈 Çıktılar

Notebook çalıştırıldığında aşağıdaki grafikler üretilir:

- **Epoch vs Loss** — Her algoritmanın epoch bazında kayıp değişimi
- **Epoch vs Accuracy** — Her algoritmanın epoch bazında doğruluk değişimi
- **Time vs Loss** — Zaman bazında kayıp karşılaştırması
- **Time vs Accuracy** — Zaman bazında doğruluk karşılaştırması
- **t-SNE Görselleştirmesi** — Ağırlık uzayında optimizasyon yollarının 2D görselleştirmesi

## 🛠️ Kullanılan Teknolojiler

- **Python** — Ana programlama dili
- **PyTorch** — Model oluşturma ve eğitim
- **Sentence Transformers** — Türkçe metin embedding
- **scikit-learn** — t-SNE boyut indirgeme
- **Matplotlib** — Görselleştirme
- **LM Studio** — Veri seti üretimi (Turkish-Gemma-9B)
- **HuggingFace Hub** — Embedding model erişimi
