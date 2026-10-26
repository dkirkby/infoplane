# PLAN.md

# Implementation Plan: Interactive Information Bottleneck Dynamics

Plan to replicate and interactively visualize the key figures (Figures 2, 3, 4, and 5) from *Opening the Black Box of Deep Neural Networks via Information* (Shwartz-Ziv & Tishby, 2017).

---

## Technical Stack Overview

| Component | Technology / Framework |
| --- | --- |
| **Phase 1 Environment** | Python 3.11+, `uv` environment manager |
| **Deep Learning Engine** | PyTorch (Accelerated via Apple Silicon `mps` / CPU fallback) |
| **Parallel Execution** | Python `multiprocessing` / `concurrent.futures` across CPU/MPS cores |
| **Data Export Format** | Formatted JSON (with fallback to Apache Arrow/Feather if file size exceeds limits) |
| **Phase 2 Build Environment** | Vite lightweight frontend project |
| **Visualization Libraries** | D3.js / Observable Plot |

---

## Phase 1: PyTorch Data Generation Pipeline

Phase 1 focuses on creating a high-performance, reproducible Python script and execution environment to train neural networks, log layer activations, compute mutual information trajectories, and export aggregated metrics.

### 1. Project Setup & Dependency Management (`uv`)

* Initialize a `uv`-managed Python environment (`uv venv`, `uv pip install ...`).
* Install dependencies: `torch`, `torchvision`, `numpy`, `scipy`, `tqdm`, `pyarrow` (optional for binary output).
* Configure device detection to utilize Apple Silicon GPU acceleration (`torch.device("mps")`) with fallback to CPU.

### 2. Dataset Generation & Loading

* **12-bit Binary Toy Dataset Generator**:
* Implement a synthetic data generator producing the 12-bit binary dataset described in the paper ($2^{12} = 4096$ possible input patterns $X \in \{0, 1\}^{12}$).
* Map inputs to binary targets $Y \in \{0, 1\}$ using a non-linear 2D boundary surface.
* Implement cross-verification tests against reference implementations (e.g., original paper scripts / replication repositories).


* **Configurable Dataset Loader**:
* Modularize the dataset interface to support alternative datasets (e.g., MNIST, synthetic Gaussian clusters) via simple CLI arguments (`--dataset toy` vs `--dataset mnist`).



### 3. Model Architecture & Training Loop

* **Architecture**:
* Fully connected feedforward neural network with 7 hidden layers of widths: `12 - 10 - 7 - 5 - 4 - 3 - 2` (matching paper specifications).
* Activation functions: Hyperbolic Tangent (`tanh`) for hidden layers, Sigmoid for output.


* **Training Parameters**:
* Total training duration: **9,000 epochs**.
* Batch size: 256 samples per batch (or full batch SGD as specified by configuration).
* Optimizer: SGD / Adam with configurable learning rate.


* **Logarithmic Epoch Sampling Schedule**:
* Collect network activations at **256 log-spaced epoch checkpoints** out of 9,000 (e.g., `np.unique(np.logspace(0, np.log10(9000), 256).astype(int))`).
* Captures rapid initial fitting dynamics in early epochs while avoiding redundant computation in late training phases.



### 4. Mutual Information Calculation (Discrete Binning)

* **Quantization / Discretization**:
* For each hidden layer $T$ and activation output (ranging in $[-1, 1]$ for `tanh`), discretize the activations into $M = 30$ uniform intervals (bins) per neuron.
* Represent joint layer state by binning individual neuron outputs.


* **Information Measures**:
* Estimate empirical joint probability distributions $p(x, t)$ and $p(y, t)$.
* Calculate Mutual Information $I(X; T)$ and $I(T; Y)$ using:

$$I(X; T) = \sum_{x \in X} \sum_{t \in T} p(x, t) \log_2 \left( \frac{p(x, t)}{p(x)p(t)} \right)$$


$$I(T; Y) = \sum_{t \in T} \sum_{y \in Y} p(t, y) \log_2 \left( \frac{p(t, y)}{p(t)p(y)} \right)$$





### 5. Multi-Seed Parallel Execution & Data Serialization

* **Ensemble Execution**:
* Execute multiple independent random seed initializations (default: 50 runs) in parallel across available CPU/MPS cores.
* Parameterized CLI control: `--num-seeds 50`, `--num-workers 8`.


* **Data Serialization Structure**:
* Aggregate per-seed trajectories, layer-wise mean trajectories, and standard deviations into an optimized JSON schema:
```json
{
  "metadata": { "epochs": 9000, "log_points": 256, "bins": 30, "seeds": 50 },
  "epochs": [1, 2, ..., 9000],
  "layers": ["Layer 1", "Layer 2", ..., "Layer 7"],
  "runs": [
    {
      "seed": 42,
      "trajectories": {
        "Layer 1": { "IXT": [...], "ITY": [...] }
      }
    }
  ],
  "aggregated": {
    "Layer 1": { "IXT_mean": [...], "IXT_std": [...], "ITY_mean": [...], "ITY_std": [...] }
  }
}

```


* Fallback to Apache Arrow/Feather format if JSON export exceeds acceptable size limits (~50MB).



---

## Phase 2: Static Interactive Webpages (Vite + D3.js / Observable Plot)

Phase 2 builds a lightweight Vite frontend serving separate static interactive visualization pages for each key figure from the paper.

```
frontend/
├── index.html         # Project overview & navigation hub
├── fig2.html          # Interactive Information Plane (Multi-seed dynamics)
├── fig3.html          # Interactive Layer Fitting vs. Compression Phase
├── fig4.html          # Interactive Mutual Information vs. Epoch Time-series
├── fig5.html          # Interactive Training Set Size (N) Comparison
└── src/
    ├── scripts/       # D3.js / Observable Plot visualization logic
    └── styles/        # Clean CSS layout

```

### 1. Figure 2: Information Plane Trajectories across Seeds

* **Core Visualization**: Scatter plot of $I(X; T)$ (horizontal axis) vs. $I(T; Y)$ (vertical axis) plotting hidden layer representations across sampled epochs and multiple random seeds.
* **Interactions**:
* **Epoch Scrubber Slider**: Drag or animate to move through the 256 logged epoch checkpoints, observing how point clouds shift from fitting to compression.
* **Seed Focus & Hover Linking**: Hovering over any individual point highlights all points belonging to the same random seed across all layers, drawing connecting trajectory lines while dimming non-selected seed data points.



### 2. Figures 3 & 5: Epoch-wise Information Dynamics & Dataset Size Scaling

* **Figure 3 Visualization**: Detailed Information Plane view tracking layer evolution across training phases.
* **Epoch Scrubber**: Highlight and isolate single epochs to inspect spatial layer ordering ($I(X; T_1) \ge I(X; T_2) \ge \dots \ge I(X; T_7)$ per Data Processing Inequality).


* **Figure 5 Visualization**: Information Plane comparison showing convergence dynamics as a function of dataset size $N$ (e.g., $N=50, 100, 500, 4000$).
* **Interactions**: Multi-dataset toggle buttons and epoch slider to overlay theoretical Information Bottleneck bound curves.



### 3. Figure 4: Mutual Information vs. Epoch Time-Series

* **Core Visualization**: Dual line plots showing $I(X; T)$ and $I(T; Y)$ as a function of training epoch (logarithmic scale) for each hidden layer.
* **Interactions**:
* **Layer Hovering**: Hovering over a specific layer's line highlights the mean trajectory and renders a shaded area for $\pm 1$ Standard Deviation across the ensemble runs.
* **Crosshair Scrubber**: Vertical scrub line displaying exact $I(X; T)$ and $I(T; Y)$ values for all layers at the hovered epoch.



---

## Verification & Testing Strategy

1. **Phase 1 Verification**:
* Verify input data distribution ($H(X) = 12$ bits for uniform 12-bit inputs).
* Ensure Data Processing Inequality holds: $I(X; T_1) \ge I(X; T_2) \ge \dots \ge I(X; T_k)$.
* Validate training loss convergence to near 0 for the 12-bit toy dataset.


2. **Phase 2 Verification**:
* Test frontend responsiveness across standard browser window sizes.
* Verify smooth 60fps rendering during slider scrub actions using D3/Observable Plot SVG/Canvas rendering.
