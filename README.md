# CAD Score

**Context-Aware Semantic Diversity for Scalable Creativity Evaluation of Large Language Models**

---

## Table of Contents

1. [Description](#description)
2. [Dataset Information](#dataset-information)
3. [Code Information](#code-information)
4. [Usage Instructions](#usage-instructions)
5. [Requirements](#requirements)
6. [Materials & Methods](#materials--methods)
7. [License & Contribution Guidelines](#license--contribution-guidelines)

---

## Description

This repository implements the **CAD Score** (“Context-Aware Semantic Diversity”), a reproducible pipeline to (1) generate creative responses with an LLM, (2) compute embedding-based diversity metrics, and (3) compare them against proxy human evaluations. It was used in the manuscript *“Context-Aware Semantic Diversity for Scalable Creativity Evaluation of LLMs”* (under review).

---

## Dataset Information

* **Sample file**: `dataset/creative_tasks_sample.json`
  * This dataset is inspired by the Torrance Tests of Creative Thinking (TTCT), which are evenly distributed across seven distinct verbal task categories.
  * A small subset of prompts and answer lists drawn from Zhao *et al.* (2025). Only the portions published in the article are shared here for reproducibility.
  The complete dataset is under the authors’ copyright and has **not** been made public.
  Only the prompts for each task and a limited subset of queries disclosed in the above article are included here solely for reproducibility.
* **Third-party source**:

  * Original dataset described in:
    Zhao, Y., Zhang, R., Li, W. *et al.* *Assessing and Understanding Creativity in Large Language Models.* Mach. Intell. Res. 22, 417–436 (2025).
    DOI: [https://doi.org/10.1007/s11633-025-1546-4](https://doi.org/10.1007/s11633-025-1546-4)

---

## Code Information

* **Core scripts**:

  * `a1_inference.py` – Generate creative outputs from a local or remote LLM
  * `a2_embedding.py` – Convert generated text to embeddings
  * `a3_scoring.py` – Compute CAD scores (kernel-entropy on normalized cosine similarity)
* **Batch & API**:

  * `b1_gpt_batch_upload.py` – Create and upload OpenAI batch requests for GPT evaluation
  * `b2_gpt_batch_download.py` – Download and save batch results as CSV for GPT evaluation
  * `b3_correlation_between_scores.py` – Compute Spearman correlations between CAD scores and GPT evaluations
  * `c1_gpt_emb_batch_upload.py` – Create and upload OpenAI batch requests for OpenAI embedding model
  * `c2_gpt_emb_batch_download.py` – Download and save completed embedding-model batch results
* **Shell wrappers**:

  * `run.sh` – End-to-end local run (inference → embedding → scoring)
  * `openai_embedding.sh` – Embed answers produced by `a1_inference.py` using the OpenAI embedding model and compute CAD scores
  * `openai_judge.sh` – Evaluate generated answers through the OpenAI judge model for proxy human scoring
* **Configuration**:

  * `.env` – Shared variables for models, env tags, directories

---

## Usage Instructions

1. **Clone or unzip** this repository.
2. **Configure**:
    Edit .env to set GEN_MODEL, EMB_MODEL, JUDGE_MODEL, GEN_ENV, directories, and generation params
3. **Local pipeline**:

   ```bash
   bash run.sh
   ```

   Outputs in `$GEN_DIR/`, `$EMB_DIR/`, and `$SCORING_DIR/`.
4. (optional) **OpenAI models as a judge**:
   * Edit .env to set JUDGE_MODEL
   * Run
    
   ```bash
   bash openai_judge.sh
   ```

5. (optional) **OpenAI models as a embedding model**:
   * Edit .env to set API_EMB_MODEL
   * Run
    
   ```bash
   bash openai_embedding.sh
   ```

---

## Requirements

* **OS**: Linux 22.04 LTS
* **Python**: ≥ 3.10
* **Python packages**:

  * `vllm`, `nltk`, `pandas`, `numpy`, `zstandard`, `scipy`, `scikit-learn`
  * `openai` (optional)

---

## Materials & Methods

* **Computing infrastructure**:

  * NVIDIA Quadro RTX 8000 (CUDA 12.8), 128 GB RAM
* **Third-party data**:

  * Sample prompts from Zhao *et al.* (2025) DOI:10.1007/s11633-025-1546-4
* **Preprocessing**:

  * JSON parsing → NLTK tokenization → optional zstandard compression of embeddings
* **Evaluation methods**:

  * Spectral entropy for diversity (CAD)
  * Correlation analysis (ablation via varying *N*, cross-dataset testing via partial/full sample comparisons)

---

## License & Contribution Guidelines

* **Dataset**: Sample data for non-commercial research only.
* **Contributions**: Feel free to open issues or pull requests. Please follow the existing code style and include tests where applicable.
* **Supplemental files**: This README and all `.py` scripts are provided as supplemental materials for review.
* **Repository DOI**: (optional, if available)

*Last updated: 20 June 2025*
