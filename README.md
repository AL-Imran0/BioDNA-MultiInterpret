## **Project Overview**

**BioDNA-MultiInterpret** is a robust multimodal deep learning framework that combines **BioBERT** for biomedical text and **DNABERT** for DNA sequences. The model is designed for **classification tasks** where each sample has both text and DNA modalities. The project includes:

* Preprocessing and tokenization for BioBERT and DNABERT
* Multimodal fusion model with pooled embeddings and fully connected layers
* Training, validation, and test evaluation with classification metrics and confusion matrix
* **SHAP explainability** for per-modality and multimodal interpretations
* Attention visualization for understanding model focus on text or DNA sequences

This repository is ideal for bioinformatics, genomics, and biomedical NLP researchers seeking interpretable multimodal models.

---

## **Project Structure**

```
BioDNA-MultiInterpret/
│
├─ README.md
├─ main.py                 # Main script: training, evaluation, SHAP, attention
├─ requirements.txt        # Python dependencies
├─ utils.py                # Helper functions (tokenization, dataset creation)
├─ models.py               # MultiModalModel class
├─ notebooks/              # Optional Jupyter notebooks for experiments
└─ results/                # Saved plots, SHAP outputs, metrics
```

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/your-username/BioDNA-MultiInterpret.git
cd BioDNA-MultiInterpret
```

2. Install required packages (recommend using a virtual environment or Conda):

```bash
pip install -r requirements.txt
```

**Required packages include:**

* `transformers`, `datasets`, `torch`, `shap`, `seaborn`, `matplotlib`, `scikit-learn`, `genomic-benchmarks`, `tqdm`, `accelerate`

---

## **Usage**

### **1. Run Training & Evaluation**

```bash
python main.py
```

* Trains multimodal BioBERT + DNABERT model
* Logs training & validation loss per epoch
* Produces test classification report and confusion matrix

### **2. SHAP Interpretability**

* Multimodal SHAP (if possible) or fallback to per-modality SHAP
* Generates token-level and global importance plots

### **3. Attention Visualization**

```python
from main import plot_attention_for_text
plot_attention_for_text(model_bio, tokenizer_bio, "Example biomedical text here")
plot_attention_for_text(model_dna, tokenizer_dna, "ACGTACGTACGTAC")
```

* Plots attention heatmaps (average over heads & layers)

---

## **Datasets**

1. **Bio Text:**
   PubMed-RCT dataset (`armanc/pubmed-rct20k`)

   * Labels: e.g., `Background`, `Objective`, `Method`, `Result`

2. **DNA Sequence:**
   Genomic Benchmarks dataset (`human_nontata_promoters`)

   * DNA sequences k-mer tokenized for DNABERT

---

## **Project Features**

* **Multimodal fusion:** Combines embeddings from BioBERT and DNABERT
* **Robust label mapping:** Supports string labels like `'Background'`
* **Explainability:** SHAP for both text and DNA sequences
* **Visualization:** Confusion matrix, global SHAP importance, attention heatmaps

------

