# ViT-Robustness-with-and-without-AT-QAT-W4A8-
Evaluating adversarial robustness of Vision Transformer (ViT-Small) after Quantization-Aware Training (QAT) on Tiny ImageNet. Two Colab notebooks: AutoAttack (APGD-CE, FAB, Square) and Foolbox (FGSM). Compares FP32, full QAT (int8 activations / int4 weights), and hybrid QAT (only last 6 transformer blocks quantised).

This repository contains two complete Jupyter notebooks that evaluate the adversarial robustness of Vision Transformers (ViT‑Small) after **Quantization‑Aware Training (QAT)** using `torchao`.  
The experiments compare:

- **FP32** baseline (full precision)
- **Full QAT** – all linear layers quantised to int8 activations / int4 weights
- **Hybrid QAT** – only the last 6 transformer blocks (6–11) are quantised, the rest stay FP32

All models are fine‑tuned on **Tiny ImageNet** (200 classes) and tested against adversarial attacks under two different evaluation frameworks:

1. **Notebook 1 – AutoAttack** (APGD‑CE, FAB, Square) – the strongest standard ensemble for L∞ robustness.
2. **Notebook 2 – Foolbox (FGSM)** – a lightweight, fast evaluation using the Fast Gradient Sign Method.

Both notebooks share the same training pipeline, data preparation, reproducibility artifacts (JSON manifests, environment snapshots, git provenance), and automatic synchronisation to GitHub (optional).

---

