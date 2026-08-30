# Semantic Classification Optimizations

A semantic classification project that compares different optimization algorithms (GD, SGD, Adam) on Turkish question-answer pairs. Built as part of the Differential Equations course — Homework 1.

## Project Overview

- Turkish question-answer pairs were generated locally using **Turkish-Gemma-9B** model via **LM Studio**
- Each pair is labeled as a correct match (`+1`) or incorrect match (`-1`)
- Text embeddings are produced using the [`ytu-ce-cosmos/turkish-e5-large`](https://huggingface.co/ytu-ce-cosmos/turkish-e5-large) model from **HuggingFace**
- Three optimization algorithms — Gradient Descent (GD), Stochastic Gradient Descent (SGD), and Adam — are trained and compared
- Results are visualized with **Loss**, **Accuracy**, and **t-SNE** plots

## Project Structure

```
├── optimizations.ipynb   # Main notebook — training and visualization
├── train_test.csv        # Question-answer dataset generated via LM Studio (200 rows)
├── .gitignore
└── README.md
```

## Dataset

`train_test.csv` consists of 4 columns:

| Column  | Description |
|---------|-------------|
| `Soru`  | Turkish question text |
| `Cevap` | Turkish answer text |
| `Etiket`| `+1` (correct match) or `-1` (incorrect match) |
| `Küme`  | `Eğitim` / Training (100 rows) or `Test` (100 rows) |

### How the Dataset Was Created

1. Download and install **[LM Studio](https://lmstudio.ai/)**
2. Pull the **Turkish-Gemma-9B-T1** model from within LM Studio
3. Run the model locally to generate Turkish question-answer pairs
4. Label correct question-answer matches as `+1` and incorrect ones as `-1`
5. Export the data as `train_test.csv`

## HuggingFace Model Usage

The project uses [`ytu-ce-cosmos/turkish-e5-large`](https://huggingface.co/ytu-ce-cosmos/turkish-e5-large) for text embeddings. The model is automatically downloaded via the `sentence-transformers` library:

```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("ytu-ce-cosmos/turkish-e5-large")
```

> On the first run, the model is automatically downloaded from HuggingFace Hub (~1.2 GB). Subsequent runs load it from cache.

## Installation

```bash
pip install pandas numpy torch matplotlib sentence-transformers scikit-learn huggingface_hub
```

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Semantic-Classification-Optimizations.git
   cd Semantic-Classification-Optimizations
   ```

2. Launch the notebook:
   ```bash
   jupyter notebook optimizations.ipynb
   ```

3. Run all cells sequentially

## Optimization Algorithms

| Algorithm | Learning Rate | Batch Size | Description |
|------------------------------------------------------|
| **GD** (Gradient Descent) | 0.5 | Full dataset | Updates weights using the entire training set |
| **SGD** (Stochastic GD) | 0.1 | 1 | Updates weights using a single sample at a time |
| **Adam** | 0.03 | 32 | Modern optimizer with adaptive learning rates |

- Each algorithm is trained for **5 Runs x 100 Epochs**
- All algorithms start from the **same initial weights** within each run for a fair comparison

## Outputs

Running the notebook produces the following visualizations:

- **Epoch vs Loss** — Loss progression per algorithm across epochs
- **Epoch vs Accuracy** — Accuracy progression per algorithm across epochs
- **Time vs Loss** — Loss comparison over wall-clock time
- **Time vs Accuracy** — Accuracy comparison over wall-clock time
- **t-SNE Visualization** — 2D projection of optimization trajectories in weight space

## Tech Stack

- **Python** — Core language
- **PyTorch** — Model definition and training
- **Sentence Transformers** — Turkish text embeddings
- **scikit-learn** — t-SNE dimensionality reduction
- **Matplotlib** — Plotting and visualization
- **LM Studio** — Dataset generation (Turkish-Gemma-9B)
- **HuggingFace Hub** — Embedding model access
