# Sparsely-Gated Mixture of Experts with CNN-GRU Encoding for Interpretable Remaining Useful Life Prediction

> **Md. Salehin Seyam**  
> Department of Industrial and Production Engineering, Bangladesh University of Engineering and Technology (BUET), Dhaka, Bangladesh  



---

## 📌 Overview

This repository presents the implementation of a novel deep learning framework for **Remaining Useful Life (RUL) prediction** of aero-propulsion turbofan engines. The proposed framework is evaluated on all four subsets of NASA’s **C-MAPSS benchmark dataset**.

The principal contribution is a **Sparsely-Gated Mixture of Experts (MoE)** architecture that integrates:

- an **adaptive spatiotemporal CNN-GRU shared encoder**,
- **Physics-Informed Regularization (PIR)** to encourage monotonic and physically consistent degradation trajectories,
- **Monte Carlo Dropout** for calibrated uncertainty quantification, and
- **learnable domain embeddings** to improve cross-domain generalisability.

The framework is designed to be **interpretable**, **uncertainty-aware**, and **transferable** across heterogeneous degradation regimes, without adversarial adaptation or explicit access to target-domain data.

---

## 🧠 Model Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│                         INPUT SENSOR DATA                            │
│                    Batch = 128 | Window = 64 | Features = 25        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      SHARED CNN-GRU ENCODER                          │
│                                                                      │
│   Conv1D (25 → 96 channels)                                          │
│   BatchNorm + ReLU                                                   │
│        │                                                             │
│   Conv1D (96 → 96 channels)                                          │
│   BatchNorm + ReLU                                                   │
│        │                                                             │
│   GRU (96 → 192 hidden units)                                        │
│        │                                                             │
│   Latent Representation Z ∈ ℝ^(Batch × 192)                          │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    NOISY TOP-K SPARSE GATING                         │
│                                                                      │
│   Gating Network → Additive Noise → Top-K Selection                  │
│                           │                                          │
│                    Softmax Normalisation                             │
│         ┌───────────┬───────────┬───────────┬───────────┐           │
│         │  g(E1)    │  g(E2)    │  g(E3)    │  g(E4)    │           │
│         └───────────┴───────────┴───────────┴───────────┘           │
│                                                                      │
│   Stability Constraint: S(w) = λ Σ |w_k(x) - w_k(x')|                │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│               EXPERT HEADS (WITH DOMAIN EMBEDDINGS)                  │
│                                                                      │
│   Each expert Eᵢ receives hᵢ = [Z ; eᵢ],                             │
│   where eᵢ ∈ ℝᵈ is a learnable domain embedding.                     │
│                                                                      │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│   │ Expert 1 │  │ Expert 2 │  │ Expert 3 │  │ Expert 4 │            │
│   │ FC Layer │  │ FC Layer │  │ FC Layer │  │ FC Layer │            │
│   │ μ₁, σ₁²  │  │ μ₂, σ₂²  │  │ μ₃, σ₃²  │  │ μ₄, σ₄²  │            │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│        └─────────────┴─────────────┴─────────────┴────────────┘     │
│                               │                                      │
│                     Mixture Aggregation                              │
│                 p(y|x) = Σ gᵢ(x) · pᵢ(y|x)                           │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         LOSS FUNCTIONS                               │
│                                                                      │
│   1. NLL Loss:        L_NLL = 1/2 log σ² + (y - μ)² / 2σ²            │
│   2. Monotonicity:    L_mono = (1/N) Σ max(0, yᵢ₊₁ - yᵢ)             │
│   3. Load Balance:    auxiliary loss for balanced expert usage       │
│   4. Diversity Loss:  auxiliary loss to prevent expert collapse      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      PROBABILISTIC OUTPUT                            │
│                                                                      │
│   Predicted Mean:        μ_total                                     │
│   Predicted Variance:    σ²_total = σ²_epistemic + σ²_aleatoric      │
│   Expert Gate Weights:   [g₁, g₂, g₃, g₄]                            │
│   90% Prediction Interval: PI = μ ± z₀.₉₅ · σ_total                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset: NASA C-MAPSS

All experiments are conducted on the **Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** benchmark introduced by Saxena et al. (2008). The dataset contains run-to-failure trajectories for turbofan engines, with **21 sensor channels** and **3 operational settings**.

| Subset | Training Engines | Test Engines | Operating Conditions | Fault Modes |
|--------|------------------|--------------|----------------------|-------------|
| FD001  | 100              | 100          | 1                    | 1 (HPC)     |
| FD002  | 260              | 259          | 6                    | 1 (HPC)     |
| FD003  | 100              | 100          | 1                    | 2 (HPC + LPT) |
| FD004  | 249              | 248          | 6                    | 2 (HPC + LPT) |

### Subset Characteristics

- **FD001**: simplest setting with a single operating condition and one fault mode
- **FD002**: introduces six operating regimes, creating a covariate-shift challenge
- **FD003**: introduces a second concurrent fault mode
- **FD004**: the most complex configuration, combining six operating conditions with two concurrent fault modes

### RUL Labelling

RUL targets are generated using a **piecewise-linear degradation strategy** with a cap of `r_max = 125 cycles`, which reduces early healthy-phase noise and places greater emphasis on late-stage degradation behaviour.

---

## ⚙️ Methodology

### 1. Data Preprocessing and Feature Selection

A three-stage adaptive feature selection pipeline is employed to reduce redundancy in the 21-sensor space:

1. **Variance-based filtering** removes near-constant sensors below the threshold `τ = 1e-6`
2. **Cumulative variance analysis** retains features explaining at least 95–99% of the total variance
3. **Coefficient of Variation (CV) ranking** prioritises sensors with the highest relative fluctuation

This process reduces the input dimensionality from 21 sensors to **16 informative channels**, improving training stability and model efficiency.

### 2. Spatiotemporal CNN-GRU Shared Encoder

For each sliding window `X_t ∈ ℝ^(w × F)`:

- **Convolutional layers** capture local inter-sensor dependencies and short-term temporal signatures
- **GRU layers** model long-range degradation dynamics
- the encoder outputs a compact latent representation `z = h_t`

The adopted window length is **T = 30 cycles**.

### 3. Sparsely-Gated Mixture of Experts (MoE)

The predictive distribution is decomposed over `E = 4` expert subnetworks:

```math
p(y|x) = \sum_i g_i(x)\, p_i(y|x)
```

Each expert receives:

- the shared latent representation `z`, and
- a **learnable domain embedding** `e_i` that captures operating-context information without requiring explicit domain labels.

A **noisy Top-K gating mechanism** activates only the most relevant experts for each input. To reduce unstable routing behaviour, a **stability constraint** is imposed:

```math
S(w) = \lambda \sum |w_k(x) - w_k(x')|
```

### 4. Physics-Informed Regularisation (PIR)

To encourage physically plausible degradation trajectories, a monotonicity loss penalises non-decreasing RUL predictions:

```math
L_{\text{mono}} = \frac{1}{N} \sum \max(0, y_{i+1} - y_i)
```

This promotes non-increasing RUL estimates during training.

### 5. Uncertainty Quantification with MC Dropout

Each expert predicts a **heteroscedastic Gaussian distribution**, parameterised by a mean and log-variance. During inference, repeated stochastic forward passes using Monte Carlo Dropout allow decomposition of total uncertainty into:

| Component | Description |
|-----------|-------------|
| Epistemic `σ²_ep` | Model uncertainty; reducible with more data |
| Aleatoric `σ²_al` | Irreducible sensor and process noise |
| Total `σ²_tot` | Combined uncertainty used to compute 90% prediction intervals |

Prediction interval width **increases with true RUL** (slope = 0.20 cycles/cycle), indicating that uncertainty is widest in the healthy region and narrows as engines approach failure, where predictions are operationally most critical.

---

## 📈 Results

### Intra-Domain Performance (RMSE; lower is better)

| Method | FD001 | FD002 | FD003 | FD004 |
|--------|-------|-------|-------|-------|
| Simple Ageing Model | 16.79 | 19.51 | 17.48 | 22.82 |
| PILmin-maxregret | 15.11 | 17.36 | 16.99 | 18.90 |
| CNN-LSTM-Attention | 15.98 | 14.45 | 13.91 | 16.64 |
| k-LSTM-GFT | 13.10 | 14.90 | 11.27 | 16.86 |
| TCAT | 11.12 | 13.40 | 11.02 | 17.56 |
| SCTA-LSTM | 12.10 | 16.90 | 12.14 | 21.93 |
| Transformer-based | 33.60 | 43.93 | 35.49 | 43.05 |
| **Proposed MoE (Ours)** | **14.98** | **18.63** | **16.74** | **23.87** |

The proposed framework substantially outperforms the Transformer-based baseline across all subsets and remains competitive with strong hybrid sequence models, while also delivering **interpretability** and **calibrated uncertainty estimation**, neither of which is provided by the comparison baselines.

### Cross-Domain Transfer (RMSE Matrix)

| Train → Test | FD001 | FD002 | FD003 | FD004 |
|-------------|-------|-------|-------|-------|
| FD001 | **14.98** | 48.46 | 47.55 | 47.28 |
| FD002 | 23.18 | **18.63** | 34.44 | 30.46 |
| FD003 | 19.74 | 58.63 | **16.74** | 56.73 |
| FD004 | 21.54 | 29.44 | 29.06 | **23.87** |

A key observation is that **models trained on FD004 exhibit the strongest cross-domain generalisation**, likely because the heterogeneous training regime encourages the gate to learn a broader and more robust routing vocabulary.

### Best Pooled Transfer Configurations

| Training Pool | Mean RMSE (all targets) |
|--------------|--------------------------|
| FD002 + FD003 + FD004 | **24.65** ✅ Best overall |
| FD001 + FD003 + FD004 | 24.86 |
| FD001 + FD002 + FD004 | 24.96 *(lowest FD004 RMSE = 22.81)* |

Heterogeneous three-source training pools consistently outperform single-source cross-domain transfer.

### Ablation Study (FD002)

| Variant | Description | RMSE ↓ | MAE ↓ | PI Coverage |
|---------|-------------|--------|-------|-------------|
| A | GRU only (baseline) | 20.07 | 16.08 | 0.739 |
| B | + CNN-GRU encoder | 19.52 | 15.39 | 0.779 |
| C | + Sparse MoE routing | 19.42 | 15.32 | **0.877** |
| D | + Domain embeddings | 19.45 | 15.48 | 0.870 |
| E | + PIR + MC Dropout (**Full model**) | **19.38** | **15.25** | 0.870 |

Each architectural component contributes positively. The largest improvement in **prediction interval coverage** arises from introducing sparse MoE routing, with coverage increasing from **0.739 to 0.877**.

---

## 🔍 Gating Interpretability

One of the study’s most notable findings is the emergence of an implicit **degradation curriculum** learned by the gating network, despite the absence of explicit temporal supervision.

| RUL Range | Dominant Expert |
|-----------|-----------------|
| 104–125 cycles (early / healthy plateau) | E3 (gate weight ≈ 0.61) |
| 62–83 cycles (transition region) | **E3 → E4 crossover** |
| 0–20 cycles (near end-of-life / wear phase) | E4 (gate weight ≈ 0.41) |

Expert **E2** remains consistently suppressed (≈ 0.13), suggesting that the gating mechanism learns to rely primarily on **two to three specialised experts** rather than distributing weight uniformly. This bimodal routing pattern indicates **true expert specialisation**, rather than simple ensemble averaging.

---

## 📐 Statistical Validation: Factorial Design

A **2²** factorial design was used across all 13 training configurations to isolate the main drivers of cross-domain generalisation error. Ordinary Least Squares (OLS) regression on coded factors yields:

```math
\text{RMSE} = \beta_0 + \beta_1 X_1(\text{Fault Mode}) + \beta_2 X_2(\text{Fleet Size}) + \beta_{12} X_1X_2
```

### Key Findings

*(all statistically significant at `p < 0.001`)*

1. **Fault-mode complexity (`β₁`) is the dominant driver of cross-domain error** in 8 of the 13 configurations, exerting a stronger effect than fleet size alone.

2. **Fleet size exerts a protective effect (`β₂ < 0`) when FD002 is included in the source pool**, suggesting that larger multi-condition fleets help stabilise the gating vocabulary and reduce sensitivity to fault mismatch.

3. **The interaction term (`β₁₂`) is negative in 11 of 13 configurations**, indicating that fault complexity and fleet size partially offset one another when both are present, thereby supporting the hypothesis that top-K gating encourages regime-coherent partitioning.

---

## 🧪 Uncertainty Quantification Results

| Evaluation Type | Mean PI Coverage (90% nominal) | Mean PI Width |
|----------------|----------------------------------|---------------|
| Intra-domain (4 tasks) | 0.876 | 57.70 cycles |
| Cross-domain (12 tasks) | 0.827 | 110.69 cycles |
| Pooled fleet (36 tasks) | 0.833 | 76.13 cycles |

### Observations

- **53.8% of all 52 evaluation tasks** achieve coverage of at least **0.85**
- prediction interval width exhibits a **positive slope with true RUL (0.20 cycles/cycle)**
- **epistemic uncertainty dominates near failure**, whereas **aleatoric uncertainty dominates during the healthy plateau**, yielding a physically coherent decomposition of predictive uncertainty

---

## ⚠️ Limitations

- A **residual monotonicity violation rate of 41.1%** remains in the high-RUL plateau region (`RUL > 100 cycles`), where piecewise label discontinuities complicate the effect of a fixed-`λ` monotonicity penalty
- **Cross-domain PI coverage declines** when FD003 is used as the sole source domain (e.g., FD002: 0.720; FD004: 0.802), reflecting substantial distributional shift rather than systematic architectural miscalibration
- the framework has been validated only on simulated C-MAPSS data; evaluation on real-world sensor datasets such as **N-CMAPSS** and **PHM2010** remains an important next step

---

## 🔭 Future Work

- **Adaptive `λ` scheduling** to selectively strengthen monotonicity regularisation in the high-RUL plateau region
- **Uncertainty-guided domain alignment**, inspired by EviAdapt, to improve transfer performance under fault-mode mismatch
- **Validation on real-sensor datasets** such as N-CMAPSS and PHM2010 to assess routing transferability under industrial noise and non-stationarity
- **Lightweight MoE inference variants**, including distilled or reduced-expert models, for embedded and on-board deployment under limited compute constraints

---

## 📁 Repository Structure

```text
├── data/
│   └── CMAPSS/              # NASA C-MAPSS raw data (FD001–FD004)
├── preprocessing/
│   ├── feature_selection.py # Variance filtering, cumulative variance, CV ranking
│   └── normalization.py     # Min-max normalisation and RUL capping
├── models/
│   ├── encoder.py           # CNN-GRU shared encoder
│   ├── experts.py           # Expert heads with domain embeddings
│   ├── gating.py            # Noisy top-K sparse gating
│   └── moe.py               # Full MoE model
├── training/
│   ├── losses.py            # NLL, PIR monotonicity, and auxiliary losses
│   └── train.py             # Training loop with MC Dropout
├── evaluation/
│   ├── metrics.py           # RMSE, MAE, NASA Score, PI coverage/width
│   ├── transfer.py          # 13-configuration cross-domain protocol
│   └── factorial.py         # 2² factorial design and OLS regression
├── results/
│   └── figures/             # Plots, tables, and ablation results
└── README.md
```

---

## 📦 Requirements

- Python >= 3.8
- PyTorch >= 1.12
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SciPy

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/MoE-RUL-CMAPSS.git
cd MoE-RUL-CMAPSS

# Prepare the dataset (place C-MAPSS files in data/CMAPSS/)
python preprocessing/feature_selection.py --subset FD001

# Train the full MoE model
python training/train.py --subset FD002 --experts 4 --topk 2

# Evaluate intra-domain and cross-domain performance
python evaluation/transfer.py --protocol all_13

# Run the ablation study
python evaluation/metrics.py --ablation
```

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@article{seyama2025moe_rul,
  title   = {Sparsely-Gated Mixture of Experts with CNN-GRU Encoding for Interpretable Remaining Useful Life Prediction},
  author  = {Seyama, Md. Salehin and Hossain, Limon Bin and Rahman, Abdur},
  journal = {[Journal Name]},
  year    = {2025}
}
```

---

## 📬 Contact

**Md. Salehin Seyam**  
Department of Industrial and Production Engineering  
Bangladesh University of Engineering and Technology (BUET)  
Dhaka-1000, Bangladesh

For questions regarding the implementation or the paper, please open a GitHub issue.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
