# 🧠 STL-10 Image Classification: Shrink or Sink Challenge
## 🏗️ 1. Model Architecture

The training pipeline is divided into **three major stages**:

---

## 🔵 Stage 1: Supervised Learning

### 📖 Description

A modified **ResNet-18** is trained on labeled STL-10 data.

### 🔧 Key Modifications

* 7×7 conv → **3×3 conv**
* Removed max-pooling layer
* Label smoothing (0.1)
* Cosine Annealing LR
* Strong augmentation (RandAugment)

### 📊 Results

| Metric     | Value      |
| ---------- | ---------- |
| Accuracy   | **84.25%** |
| Model Size | **42.7 MB**|
| Epochs     | 150        |

### 🧠 Insight

* Strong baseline
* Captures core patterns
* Acts as **teacher model**

---

## 🟡 Stage 2: Semi-Supervised Learning (Pseudo-Labeling)

### 📖 Description

The trained teacher model is used to generate labels for **unlabeled data**.

### ⚙️ Process

1. Predict on unlabeled dataset
2. Select samples with confidence > **0.95**
3. Combine with original training data
4. Retrain model

### 📊 Results

| Metric     | Value      |
| ---------- | ---------- |
| Accuracy   | **86.51%** |
| Model Size | **42.7 MB**|
| Epochs     | 50         |

### 🧠 Insight

* Utilizes unlabeled data effectively
* Improves generalization
* Produces a stronger **teacher model**

---

## 🔴 Stage 3: Knowledge Distillation (Top-K Logits)

### 📖 Description

A **smaller student model** learns from the teacher using **Top-K logits (k=4)**.

* Epochs: 50
* Techniques used:

  * **RandAugment (N=2, M=9)**
  * **Label Smoothing (ε = 0.1)**

---

```text
Loss = 0.3 × CE + 0.7 × KD
```

### 📊 Results

| Metric     | Value       |
| ---------- | ----------- |
| Accuracy   | **79.92%**  |
| Model Size | **10.7 MB** |
| Epochs     | 150         |

### 🧠 Insight

* Significant compression (~73% reduction)
* Some accuracy drop due to:

  * smaller model capacity
  * Top-K information loss

## 🧩 3. Compression Techniques

| Model                        | Accuracy        | Size       |
| ---------------------------- | --------------- | ---------- |
| Supervised                   | 84.25%          | 42.7 MB    |
| Semi-Supervised              | **86.51%**      | 42.7 MB    |
| Distilled (Final Submission) | 79.92%          | **10.7MB** |

## 📊 4. Results & Metrics

| Metric                       | Result                    |
| ---------------------------- | ------------------------- |
| Test Accuracy (Single View)  | **82.66%**                |
| Final Accuracy (10-View TTA) | **85.12%**             |
| Model File Size              | **=22.3 MB (.pth, FP16)** |
| Parameters                   | **11.17 Million**         |

✔ Successfully crossed the **85% threshold**

---

# ⚙️ Setup Guide

## 🔹 1. Clone / Setup Environment

```bash
pip install torch torchvision
```

---

## 🔹 2. Dataset

Download STL-10:

```python
torchvision.datasets.STL10(root='./data', download=True)
```

---

## 🔹 3. Training

Run full pipeline:

```bash
python train.py --data_dir ./data --save_dir ./checkpoints
```

---

## 🔹 4. Testing

```bash
python test.py --data_dir ./data --ckpt_dir ./checkpoints
```

---

## 🖥️ 6. Training Environment

The complete training pipeline was executed on the **Paramrudra HPC Cluster** available at **Indian Institute of Technology (IIT) Patna**.
The cluster enabled efficient large-scale training, especially for:

* Handling 100,000 unlabeled samples
* Accelerating GPU-based deep learning workloads
* Running multi-stage SSL and SWA efficiently

---

## 🚀 Key Highlights

* ✅ Efficient architecture tailored for small images
* ✅ Strong use of semi-supervised learning
* ✅ High accuracy with compact model size
* ✅ Fully reproducible pipeline
* ✅ Trained on high-performance computing infrastructure

---

# 🔥 Key Learnings

* Semi-supervised learning provides **significant gains**
* Knowledge distillation enables **efficient compression**
* Model capacity is critical for KD success
* Top-K distillation trades **accuracy for efficiency**

---

# 🚀 Future Improvements

This project was developed as part of the **Shrink or Sink Challenge**, focusing on achieving **high accuracy under strict model size constraints**.
We also acknowledge the support of the **Paramrudra HPC Cluster at IIT Patna** for enabling efficient model training.

---
