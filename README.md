# 🧠 STL-10 Image Classification: Shrink or Sink Challenge

## 👤 Team Information

* **Team Member:** Saubhagya Ji Jaiswal
* **Program:** Mechanical Engineering
* **Institute:** Indian Institute of Technology (IIT) Patna

---

## 🏗️ 1. Model Architecture

Our solution is based on a **Stem-Modified ResNet-18**, specifically adapted for low-resolution STL-10 images (96 × 96).

### 🔧 Key Modifications:

* **Modified Stem:**
  Replaced the standard `7×7 Conv (stride=2)` with a `3×3 Conv (stride=1)` to preserve spatial information.

* **Removed MaxPooling:**
  The initial MaxPool layer was replaced with an Identity layer to avoid early feature loss.

* **Resolution-Aware Design:**
  Adjustments ensure better feature extraction for small images compared to ImageNet-style architectures.

* **Size Optimization (FP16):**
  Final model weights stored in **half precision (float16)** → ~50% reduction in file size without accuracy loss.

---

## 🔄 2. Training Procedure

We implemented a **multi-stage Semi-Supervised Learning (SSL)** pipeline to leverage unlabeled data.

### 🚀 Stage 1: Supervised Baseline

* Dataset: 5,000 labeled images
* Augmentations:

  * RandomCrop
  * HorizontalFlip
  * ColorJitter
* Epochs: 100
* Output: Initial **Teacher Model**

---

### 🔍 Stage 2: Pseudo-Labeling

* Scanned 100,000 unlabeled images
* Selected samples with:

  ```
  Softmax Confidence > 0.90
  ```
* Generated **Pseudo Labels** and expanded training dataset

---

### 🏃 Stage 3: 85% Target Sprint

* Trained from scratch on:

  ```
  Labeled + Pseudo-labeled data

  ```
* Techniques used:

  * **RandAugment (N=2, M=9)**
  * **Label Smoothing (ε = 0.1)**
* Epochs : 50
---

### 💎 Stage 4: SWA Refinement

* Applied **Stochastic Weight Averaging (SWA)** for final 10 epochs
* Benefits:

  * Flatter minima
  * Improved generalization
  * Higher test accuracy

---

## 🧩 3. Compression Techniques

To meet strict size constraints:

* **Architectural Efficiency**

  * Removed unnecessary layers (e.g., MaxPool)

* **FP16 Quantization**

  * Converted weights: `float32 → float16`

* **Reproducibility**

  * Fixed random seed = `42` across:

    * PyTorch
    * NumPy
    * Python Random

---

## 📊 4. Results & Metrics

| Metric                       | Result                    |
| ---------------------------- | ------------------------- |
| Test Accuracy (Single View)  | **82.66%**                |
| Final Accuracy (10-View TTA) | **85.12%+**             |
| Model File Size              | **~22.3 MB (.pth, FP16)** |
| Parameters                   | **11.17 Million**         |

✔ Successfully crossed the **85% threshold**

---

## 🔁 5. Reproduction Instructions

### ⚙️ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🏋️ Training

Run the full pipeline:

```bash
python train.py --data_dir ./data
```

Includes:

```
Baseline → Pseudo-labeling → SWA refinement
```

---

### 🧪 Testing / Inference (with TTA)

Evaluate using **10-View Test-Time Augmentation**:

```bash
python test.py --data_dir ./data --model_path model.pth
```

---

## 🚀 Key Highlights

* ✅ Efficient architecture tailored for small images
* ✅ Strong use of semi-supervised learning
* ✅ High accuracy with compact model size
* ✅ Fully reproducible pipeline

---

## 📌 Notes

* Ensure dataset is available in `./data`
* Final model saved as `model.pth`
* Designed for both **performance + size optimization**

---

## 🙌 Acknowledgment

This project was developed as part of the **Shrink or Sink Challenge**, focusing on achieving **high accuracy under strict model size constraints**.

---
