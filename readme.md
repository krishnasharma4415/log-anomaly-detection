# Cross-Source Transfer for Rare Anomaly Detection in Diverse System Logs

A research-level project benchmarking classical, unsupervised, and advanced transformer-based approaches for rare anomaly detection across **16 diverse log sources**.  
Implemented in **PySpark 3.4+** with GPU-accelerated models on a single machine.

---

## 📚 Motivation & Hypothesis
Modern IT environments generate heterogeneous logs from operating systems, distributed frameworks, and applications.  
Anomalies are rare and domain-specific, making supervised detection difficult.

> **Hypothesis:** Pretraining and fine-tuning transformer models across multiple log sources improves rare anomaly detection on unseen domains compared with classical and single-source approaches.

---

## 📊 Dataset

Labeled log dataset spanning **16 diverse sources**:

- **Operating Systems:** Windows, Linux, Mac  
- **Distributed Systems:** Hadoop, HDFS, Zookeeper, Spark  
- **Applications:** Apache, Thunderbird, Proxifier, HealthApp  
- **Infrastructure:** OpenStack, OpenSSH, BGL, HPC  

**Directory Structure:**
raw_data/ # Original unstructured .log files
structured_data/ # Parsed .csv logs
labeled_data/ # Structured logs with anomaly labels

yaml
Copy code

---

## ⚙️ Infrastructure

- **CPU:** Intel i7 13th Gen  
- **RAM:** 24 GB (Spark configured to use ~18 GB)  
- **GPU:** NVIDIA RTX 4060 8 GB (for BERT models)  
- **Frameworks:** PySpark 3.4+, Spark MLlib, PyTorch/HuggingFace Transformers  

Spark handles data loading, preprocessing, and feature engineering.  
GPU accelerates transformer-based models.

---

## 🧪 Experimental Design

### Cross-Source Protocol
- **Leave-One-Source-Out:** Train on 15 sources, test on the held-out source.
- **Few-Shot Adaptation:** Fine-tune on 10 and 50 labeled examples from the held-out domain.
- **Reproducibility:** Multiple random seeds and stratified sampling.

### Models

**Classical Baselines (Spark MLlib):**
- Naïve Bayes
- k-NN
- SVM

**Unsupervised Baselines:**
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM

**Advanced Models (implemented and compared):**
1. **Domain-Adversarial BERT (DANN-BERT):**  
   BERT encoder with anomaly-classification head plus a domain-discriminator head trained via gradient reversal to encourage domain-invariant representations.
2. **Parameter-Efficient Fine-Tuning (LoRA/Adapters):**  
   Pretrain a base BERT model across multiple sources, then adapt to a held-out source using small adapter layers for efficient few-shot learning.
3. **Hybrid Template + BERT Embeddings:**  
   Combine structured template features (Drain/Spell) with BERT [CLS] embeddings to test whether structured + unstructured features improve cross-source generalization.

---

## 📝 Feature Variants

- Raw logs vs. parsed logs
- Template features + BERT embeddings vs. BERT alone

---

## 📈 Evaluation Metrics

- PR-AUC  
- Macro-F1  
- Balanced Accuracy  
- Matthews Correlation Coefficient (MCC)  
- Time-to-Detect (for streaming scenarios)

---

## 🔍 Interpretability & Ablation Studies

- Attention heatmaps on log tokens  
- SHAP values for template features  
- Ablations:
  - Template features vs. BERT alone
  - Structured vs. unstructured logs
  - Varying number of training sources

---

## 🏗️ Implementation Pipeline

1. **Spark:** Load logs → schema validation → template extraction (Drain/Spell) → tokenization.
2. **GPU (PyTorch):** Feed tokens into advanced models (DANN-BERT, LoRA/Adapters, Template+BERT).
3. **Heads:** Anomaly classifier and domain discriminator (for DANN-BERT).
4. **Evaluation:** Compute metrics on held-out source; compare few-shot results.

---

## 🎯 Expected Contributions

- First systematic benchmark of classical, unsupervised, and advanced transformer-based models for cross-source anomaly detection on a 16-source log dataset.
- Empirical evidence on the effectiveness of domain-adversarial training and parameter-efficient few-shot adaptation in unseen log domains.
- Insights from interpretability analyses on cross-domain generalization.
- Open, reproducible Spark + GPU pipeline for large-scale log anomaly detection.

---

## 🚀 Getting Started

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/cross-source-anomaly-detection.git
   cd cross-source-anomaly-detection
Install Dependencies

bash
Copy code
conda env create -f environment.yml
conda activate anomaly-detection
Run Jupyter Notebook

bash
Copy code
jupyter notebook
Configure Spark

Allocate ~18 GB RAM.

Ensure PySpark 3.4+ is installed.

GPU support enabled for PyTorch.

