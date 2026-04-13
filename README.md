# ViT-Robustness-with-and-without-AT-QAT-W4A8-
Evaluating adversarial robustness of Vision Transformer (ViT-Small) after Quantization-Aware Training (QAT) on Tiny ImageNet. Two Colab notebooks: AutoAttack (APGD-CE, FAB, Square) and Foolbox (FGSM). Compares FP32, full QAT (int8 activations / int4 weights), and hybrid QAT (only last 6 transformer blocks quantised).
