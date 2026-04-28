# ECG Denoiser: Robust Deep Learning Framework for ECG Signal Denoising

This repository provides the complete pipeline for dataset construction, model training, evaluation, and benchmarking of ECG denoising methods, including the proposed architecture, its ablations, and multiple baseline models.

---

## 📌 Overview

The project is structured into two main stages:

* **Part 1 — Dataset Construction:**
  Preprocessing and augmentation of ECG signals from multiple sources.

* **Part 2 — Model Training & Evaluation:**
  Training across multiple seeds, benchmarking, and final ranking using a composite evaluation metric defined in the associated paper.

---

## ⚠️ Reproducibility Notice

Due to the use of **NVIDIA CuDNN-accelerated operations**, certain components of the training pipeline are **non-deterministic**.

* Exact numerical reproducibility across runs is **not guaranteed**
* Results may show **minor variations across executions and hardware setups**
* This is expected behavior for GPU-accelerated deep learning workflows

To mitigate this:

* Experiments are conducted across **7 independent random seeds**
* Final results are reported as **aggregated performance metrics**

---
## 📥 Google Drive Resources

The following resources are provided via Google Drive to support dataset setup and quick experimentation.

---

### 🔗 Downloads

* **Prebuilt Dataset**
  `Ultimate_Denoiser_Dataset_Fixed2`
  → Use this to skip Part 1 (dataset construction)

* **Additional Data Folder**
  `Hand Position`
  → Required only for Part 1 (dataset creation pipeline)

---

### 📦 Usage Instructions

#### Option A — Skip Dataset Creation

1. Download `Ultimate_Denoiser_Dataset_Fixed2`
2. Place it inside:

   ```
   ECG Final Version 2/
   ```
3. Proceed directly to **Part 2**

---

#### Option B — Full Dataset Construction

1. Download `Hand Position`
2. Place it inside:

   ```
   ECG Finalized2/
   ```
3. Follow Part 1 instructions to build the dataset from scratch

---

### ⚠️ Notes

* Ensure correct folder placement before running scripts
* Do **not rename folders**, as scripts may depend on fixed paths
* Large files may require sufficient storage and stable internet connection

---

### 📌 Link

https://drive.google.com/drive/folders/16ITzxfP3qeXFnaH273o1cLkkHqJ-7oNy?usp=sharing

## ⚡ Quick Start (Skip Dataset Creation)

If you prefer not to build the dataset:

➡️ Download the prebuilt dataset from the provided Google Drive link
➡️ Place `Ultimate_Denoiser_Dataset_Fixed2` inside:

```
ECG Final Version 2/
```

➡️ Proceed directly to **Part 2**

---

## 🧩 Part 1: Dataset Construction

### 1. Repository Setup

Clone the repository:

```
git clone <repo-url>
```

Recommended directory name:

```
ECG_Denoiser/
```

---

### 2. Data Sources

This pipeline integrates:

* ECG signals collected during this study:

  * Chest recordings
  * Palm recordings
* Public dataset:

  * **PTB-XL ECG Dataset**

---

### 3. Required Downloads

1. Download the **PTB-XL dataset**
2. Extract into:

```
ECG Finalized2/
```

3. From the Google Drive link:

   * Copy the folder:

     ```
     Hand Position
     ```
   * Place it inside:

     ```
     ECG Finalized2/
     ```

---

### 4. Execution Pipeline

Run all scripts in **sequential order** (as numbered).

This pipeline performs:

* Signal preprocessing
* Data augmentation
* Data integrity checks
* Patient-wise dataset splitting

---

### 5. Output

Final dataset generated:

```
Ultimate_Denoiser_Dataset_Fixed2
```

✔ Ensures:

* No data leakage
* Strict **patient-wise split**
* Balanced and validated dataset construction

---

## 🤖 Part 2: Model Training & Evaluation

The folder **`ECG Final Version 2`** contains:

* Training scripts for:

  * Proposed denoiser
  * Ablation variants
  * Baseline architectures
* Multi-seed experimentation (**7 seeds**)
* Testing and benchmarking scripts
* Final scoring and ranking framework

---

### 1. Dataset Setup

Place:

```
Ultimate_Denoiser_Dataset_Fixed2
```

inside:

```
ECG Final Version 2/
```

---

### 2. Execution

Run scripts in **sequential order**.

Pipeline performs:

* Model training across multiple seeds
* Performance evaluation
* Metric aggregation
* Final ranking using the scoring formula defined in the paper

---

## 📊 Evaluation Protocol

* Multi-metric evaluation framework
* Cross-seed averaging for robustness
* Final ranking based on composite scoring methodology

(Refer to the paper for detailed formulation)

---

## 📁 Outputs

The pipeline generates:

* Trained model checkpoints
* Evaluation metrics
* Aggregated performance scores
* Final ranked comparison of models

---

## 🧪 Experimental Rigor

* Capacity-matched baselines
* Controlled ablation studies
* Patient-wise data isolation
* Leakage prevention mechanisms

---

## 📜 License & Data Usage

This project uses the **PTB-XL dataset**, which is **publicly available for research purposes**.

* Users must comply with the original PTB-XL license and citation requirements
* This repository does **not redistribute PTB-XL data directly**
* Locally collected data (if included) is for **research use only**

If you use this work, please cite:

* The PTB-XL dataset
* The associated research paper (to be added)

---

## 🚀 Summary

| Component | Description                                 |
| --------- | ------------------------------------------- |
| Part 1    | Dataset construction from raw + PTB-XL data |
| Part 2    | Training, evaluation, and benchmarking      |

---

## 📌 Notes for Reviewers

* Dataset pipeline includes explicit safeguards against leakage
* Multi-seed evaluation ensures statistical robustness
* Non-determinism due to CuDNN is acknowledged and mitigated via aggregation

---

## 👤 Author

1. Hiritish Chidambaram N, Student at VIT Vellore
2. Dr. Anisha M. Lal, Grade 1 Professor at VIT Vellore

---

## 📄 Paper

(Add IEEE paper link here once available)
