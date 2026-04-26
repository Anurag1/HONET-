# HONet: A Composable Architecture for Lifelong Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2310.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2310.XXXXX)

This repository contains the official PyTorch implementation and demonstration of **HONet (Hierarchical Octave Network)**, a novel architecture that enables AI models to learn new skills sequentially **without catastrophic forgetting**.

## Core Concept: AI Without Amnesia

HONet solves the "plasticity-stability dilemma" by treating learned skills as immutable, foundational layers.

1. **Isolate & Learn:** A task-specific `Octave` module (a conditional VAE) learns a single skill.
2. **Freeze & Preserve:** The module's weights are frozen, guaranteeing its knowledge is never lost.
3. **Distill & Summarize:** The Octave's *function* is distilled into a compact "Master-Tone" vector.
4. **Condition & Build:** The next skill is learned by a new `Octave` conditioned on the previous Master-Tone, enabling positive knowledge transfer with **practical, linear `O(N)` scalability**.

This approach combines the perfect memory of architectural methods with a practical scaling law, making it ideal for building truly adaptive AI systems.

---

## Strong Proof of Capabilities

### 1. Zero Catastrophic Forgetting

HONet can learn new tasks without any degradation to its prior skills. This is the gold standard for continual learning.

*Figure 1: After learning a new task, HONet can still perfectly generate samples from a previously learned task.*

### 2. Linear Scalability

HONet's design avoids the `O(N^2)` parameter explosion of older methods like Progressive Neural Networks, making it viable for real-world systems that must learn many skills.

*Figure 2: HONet's linear scaling vs. the impractical quadratic scaling of PNNs.*

### 3. True Multi-Modal Generality

The HONet framework is data-agnostic. This demonstration shows a single HONet instance learning sequentially across three different data types: **Images (CNNs) -> Tabular Data (MLPs) -> Time-Series Data (LSTMs)**. It retains all three skills perfectly.

*Figure 3: Generated output from all three distinct modalities, produced by the single, fully-trained HONet.*

---

## Getting Started

### 1. Setup Environment

Clone the repository and install the required packages.

```bash
git clone https://github.com/your-username/HONet-Lifelong-Learning.git
cd HONet-Lifelong-Learning
pip install -r requirements.txt
```

### 2. Run the Multi-Modal Demo

This script demonstrates HONet learning three sequentially different tasks (image, tabular, and sequential data) and verifying zero forgetting at the end.

```bash
python run_demo.py
```

Output images will be saved to `live_output/`.

### 3. Run the Strong Evidence Benchmark (Split CIFAR-10)

This runs a rigorous benchmark with a control experiment (naive finetuning → catastrophic forgetting) compared to HONet (zero forgetting).

```bash
python run_strong_evidence.py
```

Output images will be saved to `strong_evidence_output/`.

---

## Repository Structure

```
HONet/
├── honet/
│   ├── __init__.py          # Package exports
│   ├── octaves.py           # ImageOctave, TabularOctave, SequentialOctave (CVAEs)
│   ├── distiller.py         # MasterToneProducer & distillation pipeline
│   └── data_factory.py      # Task DataLoader factories
├── data/
│   └── cifar-10-batches-py/ # Auto-downloaded CIFAR-10 data
├── live_output/             # Demo output images
├── strong_evidence_output/  # Benchmark output images
├── run_demo.py              # Multi-modal lifelong learning demo
├── run_strong_evidence.py   # Rigorous Split CIFAR-10 benchmark
├── requirements.txt
└── README.md
```

---

## Architecture Deep Dive

### The Octave Module

Each `Octave` is a **Conditional Variational Autoencoder (CVAE)**:

```
Input x ──► [Encoder G-Net] ──► (μ, σ²) ──► z ~ N(μ, σ²)
               ▲                                    │
               │  Master-Tone I                     ▼
               └────────────────────────── [Decoder F-Net] ──► x̂
```

The conditioning on Master-Tone `I` allows each new Octave to build upon prior knowledge without accessing any old data (no replay buffer required).

### The Distillation Step

After training each Octave, a lightweight `MasterToneProducer` is trained to compress the Octave's learned distribution into a single vector `I`. This vector is then passed as a conditioning signal to all future Octaves.

```
Trained Octave (frozen) ──► [MasterToneProducer] ──► I_new
                                                         │
                                                         ▼
                                              Condition next Octave
```

---

## Comparison with Existing Methods

| Method                 | Forgetting | Scalability | Data-Agnostic |
|------------------------|------------|-------------|---------------|
| Fine-tuning            | ❌ Severe  | ✅ O(1)     | ✅            |
| EWC / Regularization   | ⚠️ Partial | ✅ O(N)     | ✅            |
| Progressive Nets (PNN) | ✅ Zero    | ❌ O(N²)    | ✅            |
| **HONet (ours)**       | ✅ **Zero**| ✅ **O(N)** | ✅            |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{honet2023,
  title   = {HONet: A Composable Architecture for Lifelong Learning without Catastrophic Forgetting},
  author  = {Anurag},
  journal = {arXiv preprint arXiv:2310.XXXXX},
  year    = {2023}
}
```
