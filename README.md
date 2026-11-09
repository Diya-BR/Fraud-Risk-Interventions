# Explaining Fraud-Risk Score Changes in Temporal Transaction Graphs

An empirical study of feature-group interventions for fraud-risk scoring on the Elliptic Bitcoin transaction dataset.

> **Reproducibility status:** This repository is being assembled from the project results and methodology notes. The original analysis notebook and executable source code are not currently present in this repository. Results below are reported project results, not a claim that this repository has already reproduced them from a clean environment. Add the notebook/source and verify a clean run before describing the repository as fully reproducible.

## Research question

How much do transaction attributes and graph-derived feature groups influence a trained fraud-risk model's predictions, and how reliable are those estimates when feature groups are dependent and replacement donors are imperfect?

The study uses matched feature-group replacement as a model-behaviour intervention. It measures changes in predicted risk; it does **not** identify causal effects in the broader real-world or structural-causal sense.

## Dataset

The project uses the Elliptic Bitcoin transaction dataset:

- 203,769 transaction nodes and 234,355 directed edges.
- 49 temporal time steps.
- 46,564 labelled transactions: 4,545 illicit and 42,019 licit.
- Chronological split: train steps 1–34, validation 35–41, test 42–49.

Obtain the dataset from the official/public sources below. This repository intentionally does not include raw dataset files; check the dataset's terms before redistributing them.

- [Kaggle: Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
- [PyTorch Geometric: EllipticBitcoinDataset](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.EllipticBitcoinDataset.html)
- [Original Elliptic paper](https://arxiv.org/abs/1908.02591)

## Feature groups

The analysis uses three operational feature groups:

- **T — Transaction:** the first 93 original transaction/local attributes.
- **N — Neighbourhood:** the next 72 neighbourhood-derived attributes.
- **G — Graph structure:** 17 separately engineered graph-structural descriptors.

The groups are not statistically independent. They are a practical partition for controlled model-input interventions, not mutually exclusive causal mechanisms.

Evaluated configurations: T, N, G, T+N, T+G, N+G, and T+N+G.

## Main reported results

### Random Forest feature configurations

| Features | AUROC | AUPRC | F1 |
|---|---:|---:|---:|
| T | 0.7974 | 0.5379 | 0.5857 |
| N | 0.8460 | 0.4677 | 0.4829 |
| G | 0.6336 | 0.0773 | 0.1384 |
| T+N | 0.8570 | 0.5459 | **0.6200** |
| T+G | 0.8147 | 0.5389 | 0.5962 |
| N+G | 0.8503 | 0.4631 | 0.4812 |
| T+N+G | **0.8595** | **0.5462** | 0.6110 |

T+N+G has the highest reported AUROC and AUPRC, while T+N has the highest reported F1. The incremental gains from adding G to T+N are small on these metrics; this does not establish that graph structure is universally uninformative.

### Matched feature-group replacement

For each target, a donor from the same time step is selected using similarity in the complementary feature groups; the target itself is excluded, up to five donors are used, and labels are not used for matching.

| Replaced group | Mean absolute risk change | Reported 95% CI | Prediction flips |
|---|---:|---:|---:|
| T | 0.0502 | [0.0489, 0.0516] | 0.6% |
| N | 0.0125 | [0.0120, 0.0130] | 0.2% |
| G | 0.0033 | [0.0032, 0.0034] | 0% |

The reported ordering is T > N > G. It was reported as robust across seeds and temporal controls. These are changes in model predictions under a particular replacement design, not causal effects. Donor realism remains a limitation: the donor audit reported distinguishability from original rows (C2ST values T=0.786, N=0.665, G=0.703) and approximately 5% out-of-distribution cases.

### Additional analyses

- Joint replacement mean absolute risk changes: T+N 0.0760, T+G 0.0472, N+G 0.0133, and T+N+G 0.1109.
- SHAP grouped attribution on 503 test rows: T 81.17%, N 18.06%, G 0.77% of the reported absolute grouped attribution. SHAP is a model-attribution analysis, not a causal validation.
- A GraphSAGE experiment was also run (reported test AUROC 0.8058, AUPRC 0.4716, F1 0.4819). It is not a controlled apples-to-apples comparison with the Random Forest.
- The edge-removal diagnostic found shortcut and full-recompute mean absolute prediction changes of 0.0107 and 0.0108 respectively (correlation 0.999). The intervention can leave some neighbourhood-derived inputs stale; interpret it as a diagnostic, not a clean isolation of G.

## Important methodological qualifications

1. Feature groups are dependent and their boundaries are operational.
2. Matched replacement estimates sensitivity to a specified donor-replacement procedure; it does not establish causal contribution.
3. Donor rows are not perfectly exchangeable with targets, as indicated by the donor realism checks.
4. Same-time donor matching uses features from the relevant time step. The project notes also report that the scaler was fitted using the test period; this transductive aspect must be disclosed when interpreting strict out-of-sample claims.
5. Joint replacement effects are descriptive and need not equal the sum of individual effects; joint donor pools differ.
6. SHAP values explain the fitted model's output under a background distribution; they do not resolve feature dependence or causal direction.
7. Graph edits may affect derived features beyond the directly edited structural descriptor.

## Repository layout (intended)

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── AFML_Final_Reproducible.ipynb   # to be re-uploaded and added
├── src/                                # analysis modules to be added
├── results/
│   ├── figures/
│   └── metrics/
└── paper/
    └── project_paper.pdf               # add final approved paper if applicable
```

## Setup

Create and activate a Python virtual environment, then install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dependency list is a starting point, not a verified lockfile. Exact package versions should be pinned after confirming the original notebook's environment.

## Data setup

1. Download the Elliptic dataset from one of the sources above.
2. Follow the dataset source's terms and place the files in a local data directory.
3. Do not commit raw data unless redistribution is explicitly permitted.
4. Update the data-loading paths in the notebook/source for your local environment. Remove machine-specific absolute paths before committing.

## Running the analysis

The notebook is not yet included in this initial repository setup. Once `notebooks/AFML_Final_Reproducible.ipynb` and the source modules are added, document the exact execution order and commands here. A clean-kernel, end-to-end run is required before claiming full reproducibility.

## Acknowledgements and references

- Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics*, KDD 2019. [arXiv:1908.02591](https://arxiv.org/abs/1908.02591).
- Elliptic dataset distribution: [Kaggle dataset page](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).
- PyTorch Geometric dataset documentation: [EllipticBitcoinDataset](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.EllipticBitcoinDataset.html).

## AI assistance

If AI tools were used in preparing code, documentation, or analysis, disclose that assistance in accordance with the course/institution policy. Verify every result and citation against the original notebook and sources before submission.
