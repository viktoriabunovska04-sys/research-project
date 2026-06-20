# 📌 Hate Speech Alignment Research Project

![Python](https://img.shields.io/badge/python-3.12+-blue)

This repository contains code for analyzing how different hate speech definitions affect model predictions for Flan-T5 and Qwen using the extended HateCheck dataset. It was created as part of the CSE3000 Research Project 2026 at TU Delft.

For more information about the course, see the TU Delft CSE3000 Research Project page: https://github.com/TU-Delft-CSE/Research-Project.

It includes:

- data relabeling according to multiple hate speech definition
- prediction scripts for Flan-T5-XL and Qwen2.5-3B-Instruct models
- evaluation utilities (precision, recall, F1)
- flip analysis (beneficial and harmful prediction changes)
- qualitative and categorical analysis of model behavior

---

## 🎯 Research Goal

This project investigates how variations in hate speech definitions affect model behavior, specifically:

- changes in binary classification decisions
- beneficial vs harmful prediction flips relative to a baseline model

---

## 📁 Project structure

### Data processing

- `relabelling.py`  
  Applies definition-specific labeling rules to the extended HateCheck dataset for:
  - Bulgaria
  - Croatia
  - Meta
  - Reddit
  - Theoretical definitions

### Datasets
The relabeled datasets used in this project can be found in `data/`. The dataset used for this project is the extended version of HateCheck proposed by Khurana et al. (2025). References for both the extended and original HateCheck datasets are provided below:

- **Extended HateCheck** \
Khurana, U., Nalisnick, E., & Fokkens, A. (2025). DefVerify: Do Hate Speech Models Reflect Their Dataset's Definition? In Proceedings of COLING 2025, pp. 4341–4358. Available at: https://aclanthology.org/2025.coling-main.293/

- **HateCheck** \
Röttger, P., Vidgen, B., Nguyen, D., Waseem, Z., Margetts, H., & Pierrehumbert, J. (2021). HateCheck: Functional Tests for Hate Speech Detection Models. In Proceedings of ACL-IJCNLP 2021, pp. 41–58. https://doi.org/10.18653/v1/2021.acl-long.4

---

### Model inference

- `flan/flan_predictions_own.py`  
  Runs FLAN-T5 without definition prompts (baseline)

- `flan/flan_predictions_given.py`  
  Runs FLAN-T5 with definition-conditioned prompts

- `qwen/qwen_predictions_own.py`  
  Runs Qwen without definition prompts (baseline)

- `qwen/qwen_predictions_given.py`  
  Runs Qwen definition-conditioned predictions

---

### Evaluation utilities (`src/`)

- `src/general_utils.py`  
  Loads saved results and provides helper functions

- `src/eval_metrics_utils.py`  
  Computes:
  - precision
  - recall
  - F1 score
  - accuracy-related metrics

- `src/eval_flips_utils.py`  
  Computes flip statistics:
  - total flips
  - beneficial flips
  - harmful flips

- `src/definitions.py`  
  Stores all definition prompts used in experiments

---

### Analysis modules

- `common_analysis/metrics_analysis.py`  
  Aggregated metric comparison across models and definitions

- `common_analysis/flip_analysis.py`  
  Flip statistics across datasets and definitions

- `common_analysis/distribution_of_flips.py`  
  Heatmaps of beneficial/harmful flip distributions (FLAN vs Qwen)

- `common_analysis/qualitative_analysis.py`  
  Extracts qualitative examples of interesting prediction flips

---

## ⚙️ Setup

### 1. Clone repository

```bash
git clone <repo-url>
cd hate-speech-alignment
```

### 2. Create environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

The required dependencies are: pandas, numpy, matplotlib, scikit-learn, transformers, torch, openpyxl, tqdm.

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the pipeline

### Step 1: Relabel dataset

```bash
python -m relabelling.py
```

### Step 2: Run model predictions

#### FLAN-T5

```bash
python -m flan/flan_predictions_own.py
python -m flan/flan_predictions_given.py
```

#### Qwen

```bash
python -m qwen/qwen_predictions_own.py
python -m qwen/qwen_predictions_given.py
```

⚠️ Note: These scripts require access to HuggingFace models and may require GPU support. For this project, these files were executed in Kaggle with their GPU-T4. More about the setup can be read here: 
* https://huggingface.co/docs/transformers/en/quicktour
* https://www.kaggle.com/discussions/product-feedback/114053

### Step 3: Evaluation and analysis

```bash
python flan/flan_categories.py
python common_analysis/metrics_analysis.py
python common_analysis/flip_analysis.py
python common_analysis/distribution_of_flips.py
python common_analysis/qualitative_analysis.py
```

---

## 📦 Outputs

- results/flan/ — FLAN prediction outputs
- results/qwen/ — Qwen prediction outputs
- HTML tables for evaluation summaries
- .png heatmaps for flip distributions
- qualitative analysis tables

---

## 🧪 Key Dependencies

- pandas
- numpy
- matplotlib
- scikit-learn
- torch
- transformers
- tqdm
- openpyxl

---

## Notes

- All scripts should be run from the repository root
- Prediction scripts may require GPU or external runtime (e.g., Kaggle)
- Dataset structure is based on HateCheck-derived samples with multiple relabeling schemes
