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

## Configuration Highlights

You can modify the shared configuration dictionary (`CONFIG_SHARED`) at the beginning of each notebook:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 64 | Training / validation batch size |
| `num_epochs` | 10 | Fine‑tuning epochs for FP32 |
| `qat_num_epochs` | 10 | Fine‑tuning epochs for QAT (after prepare) |
| `adv_epsilon` | 0.01 | PGD attack strength during adversarial training |
| `epsilons` | `[0.001,0.005,0.01,0.03,0.05]` | L∞ perturbation budgets for evaluation |
| `autoattack_budget` | `'tiny'` | Speed/accuracy trade‑off for AutoAttack |
| `run_baseline_no_adv` | `True` | Also run the non‑adversarial training series |
| `hybrid_quantize_blocks` | `(6, 12)` | Transformer block indices to quantise (0‑based) |
---

