# Profiling Small LLM Workloads

**Performance analysis of fine-tuning DistilBERT (67M parameters) on IMDB sentiment classification using PyTorch.**

This project systematically benchmarks and profiles the training pipeline to identify bottlenecks and quantify the impact of common optimization strategies on an NVIDIA A100-SXM4-80GB.

📊 **[W&B Experiment Dashboard](https://wandb.ai/ar4678-columbia-university/hpml-hw2-llm)**

---

## Overview

Fine-tuning pre-trained language models is a common ML workflow, but few practitioners systematically profile *where* time is spent or *which* optimizations actually matter. This project answers those questions by benchmarking every stage of the training pipeline.

**Model:** DistilBERT (66.9M parameters) — a distilled version of BERT, ~40% smaller and ~60% faster while retaining 97% of BERT's language understanding.

**Task:** Binary sentiment classification on the IMDB dataset (25K train / 25K test).

**Best result:** 91.3% test accuracy (batch size 16, lr=5e-5, AdamW, 5 epochs).

## Experiments & Key Findings

### 1. Baseline Training & Timing Instrumentation

Custom training loop with `torch.cuda.synchronize()` barriers to accurately separate data loading time from compute time. No HuggingFace Trainer — everything is manual PyTorch for full control and transparency.

- Data loading: ~3.5% of epoch time
- Compute (forward + backward + optimizer): ~96.5%

### 2. DataLoader Optimization

Swept `num_workers ∈ {0, 2, 4, 8}` with fixed hyperparameters:

| `num_workers` | Data Time (s) | Compute Time (s) | Epoch Time (s) |
|:---:|---:|---:|---:|
| 0 | 8.6 | 109.5 | 118.1 |
| 2 | 3.6 | 110.5 | 114.1 |
| 4 | 3.8 | 110.7 | 114.5 |
| 8 | 4.4 | 110.6 | 115.0 |

**Finding:** `num_workers=2` is optimal. Beyond that, IPC overhead from multiprocessing exceeds any data prefetch gains. The pipeline is already GPU-bound — data loading is only ~3% of wall time.

### 3. PyTorch Profiler Analysis

Profiled with `torch.profiler.profile` + TensorBoard trace handler:

| Category | Time (%) |
|---|---:|
| GPU Kernel Execution | 97.14 |
| CPU Execution | 1.83 |
| Other | 0.94 |
| Memory Operations | 0.09 |

- **GPU utilization:** 97.48%
- **SM efficiency:** 96.86%
- **Achieved occupancy:** 37.51%

**Insight:** The pipeline is near-perfectly GPU-bound. Optimization efforts should focus on GPU kernel efficiency (mixed precision, operator fusion) rather than CPU-side data loading.

### 4. Hyperparameter Sensitivity (3×3 Grid Search)

| Batch Size | Learning Rate | Test Acc | Time/Epoch (s) |
|:---:|:---:|:---:|---:|
| 16 | 5e-5 | **91.3%** | 134.4 |
| 32 | 5e-5 | 91.3% | 123.3 |
| 64 | 5e-5 | 91.2% | 115.5 |
| 16 | 1e-4 | 88.5% | 134.4 |
| 32 | 1e-4 | 90.2% | 123.3 |
| 64 | 1e-4 | 91.0% | 115.5 |
| * | 5e-4 | 50.0% | — |

**Findings:**
- Learning rate is the dominant factor — 5e-4 causes complete divergence (50% = random)
- Batch size trades ~14% speed for <0.2% accuracy difference
- Sweet spot: batch 32, lr 5e-5 (best speed/accuracy tradeoff)

### 5. Optimizer Comparison

| Optimizer | Test Acc | Train Loss | Avg Epoch Time (s) |
|---|:---:|:---:|---:|
| SGD (momentum=0.9) | 88.5% | 0.278 | 227.4 |
| Adam | 90.2% | 0.006 | 236.3 |
| **AdamW** | **90.3%** | 0.009 | 238.4 |

**Finding:** Adaptive optimizers (Adam/AdamW) converge dramatically faster than SGD for fine-tuning. AdamW's weight decay decoupling gives marginally better generalization. SGD's per-parameter learning rates are insufficient when different layers need different update magnitudes.

### 6. `torch.compile` (Inductor Backend)

| Mode | First Epoch (s) | Steady-State Avg (s) | Speedup |
|---|---:|---:|:---:|
| Eager | 114.9 | 114.6 | — |
| Compiled | 152.1 | 112.7 | **1.7%** |

**Finding:** Compilation overhead is ~32% on the first epoch. Steady-state speedup is only 1.7% because DistilBERT's dominant operations (large matrix multiplications) are already well-optimized via cuBLAS/cuDNN in eager mode. `torch.compile` benefits models with more fusible small operations.

### 7. Advanced Profiling & Bottleneck Analysis

Operator-level trace with `record_shapes=True`, `profile_memory=True`, `with_stack=True`:

**Top bottlenecks identified:**

1. **Float32 matrix multiplications** — `aten::mm` accounts for ~40% of CUDA time across 570+ calls. Running on CUDA cores instead of Tensor Cores.
   - *Optimization:* Mixed precision (`torch.cuda.amp`) would utilize TF32/FP16 Tensor Cores for 2–3× GEMM throughput.

2. **Memory-bound element-wise operations** — Softmax, dropout, GELU, LayerNorm collectively take ~25% of CUDA time despite low arithmetic intensity.
   - *Optimization:* Operator fusion via `torch.compile` or custom CUDA kernels (FlashAttention).

3. **Gradient synchronization overhead** — `aten::zero_()` called 200+ times per step for gradient clearing.
   - *Optimization:* `optimizer.zero_grad(set_to_none=True)` to avoid memset operations.

### 8. Model Architecture

- **Embedding input:** 30,522 (vocab size) → 768-dimensional vectors
- **Classifier head:** Linear(768→768) → ReLU → Dropout(0.1) → Linear(768→2)
- **Total parameters:** 66,955,010 (all trainable, all receive gradients)

## Environment

| Component | Version |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| CUDA | 12.8 |
| PyTorch | 2.10.0 |
| Transformers | 5.0 |
| Platform | Google Colab Pro |

## Usage

Open the notebook in Google Colab or any Jupyter environment with GPU access:

```bash
pip install transformers datasets wandb accelerate
```

All experiments log to [Weights & Biases](https://wandb.ai/ar4678-columbia-university/hpml-hw2-llm) for reproducible comparison.

## License

MIT
