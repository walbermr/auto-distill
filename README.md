# ⚗️ Auto-Distill

A collection of lightweight, minimalist distillation architectures. Designed for automatically test lightweight layers on different tasks. The priority is clarity, speed, and low resource usage — ideal for research prototypes, edge devices, or anyone who wants to learn by reading clean code.

## ✨ Features

✅ Compact implementations — no unnecessary layers or complexity

- 🚀 Fast and memory-efficient
- 🧰 PyTorch backend
- 📚 Well-documented and easy to extend
- 🔍 Includes training scripts & evaluation utilities

## 📂 Repository Structure

    auto_distill/
    ├── models/                 # Core model definitions
    │   ├── convolution/        # Implementation of convolutional architectures
    │       ├── separable/      # Separable convolution architectures
    ├── scripts/                # Training & evaluation scripts
    |   ├── classification      # Training task
    └── requirements.txt        # Python dependencies