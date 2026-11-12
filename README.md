# Explaining Fraud-Risk Score Changes in Temporal Transaction Graphs

An empirical study of feature-group interventions for fraud-risk scoring on the Elliptic Bitcoin transaction dataset.

> **Reproducibility status:** The notebook contains the code needed to rerun the Random Forest experiments and the GraphSAGE analysis. Importantly, the GraphSAGE section includes a from-scratch training path when `graphsage_t_explainer_checkpoint.pt` is absent; a pre-existing checkpoint is therefore not required to execute that section. The notebook also sets random seeds for its main NumPy and GraphSAGE workflows. The committed notebook includes saved outputs from a prior run, but a fresh end-to-end Kaggle run has not been independently verified for this repository. Treat the displayed outputs as the recorded run and use the instructions below to regenerate the analysis.

## Research question

How much do transaction attributes and graph-derived feature groups influence a trained fraud-risk model's predictions, and how reliable are those estimates when feature groups are dependent and replacement donors are imperfect?

The study uses matched feature-group replacement as a model-behaviour intervention. It measures changes in predicted risk; it does **not** identify causal effects in the broader real-world or structural-causal sense.

## Repository contents

```text
.
├── .gitignore
├── Fraud_Risk_Detection.ipynb
├── README.md
├── requirements.txt
└── data/
    └── README.md
```

The notebook is currently stored in the repository root. The raw dataset and generated result files are not committed.

## Dataset

The project uses the Elliptic Bitcoin transaction dataset:

- 203,769 transaction nodes and 234,355 directed edges.
- 49 temporal time steps.
- 46,564 labelled transactions: 4,545 illicit and 42,019 licit.
- Chronological split: train steps 1–34, validation 35–41, test 42–49.

Obtain the dataset from the sources below. This repository intentionally does not include raw dataset files; check the dataset's terms before redistributing them.

- [Kaggle: Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
- [PyTorch Geometric: EllipticBitcoinDataset](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.EllipticBitcoinDataset.html)
- [Original Elliptic paper](https://arxiv.org/abs/1908.02591)

### Dataset files in Kaggle

The recommended execution environment is **Kaggle Notebooks**. Open the [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) page and add the dataset to your Kaggle notebook using **Add Input**. The three CSV files required by this project are:

- `elliptic_txs_features.csv`
- `elliptic_txs_edgelist.csv`
- `elliptic_txs_classes.csv`

Kaggle mounts attached datasets as read-only inputs under `/kaggle/input/`. The exact folder name can vary, so do not assume the path from the dataset title. In a Kaggle notebook cell, run the following to locate the files:

```python
from pathlib import Path

for path in Path("/kaggle/input").rglob("*.csv"):
    if path.name in {
        "elliptic_txs_features.csv",
        "elliptic_txs_edgelist.csv",
        "elliptic_txs_classes.csv",
    }:
        print(path)
```

If the notebook does not find the files automatically, update the candidate paths in its dataset-location cell to the paths printed above. Keep the three files together under the same Kaggle input directory. Do not try to write generated files into `/kaggle/input/`; use `/kaggle/working/` for outputs.

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
- The edge-removal diagnostic on 300 test transactions (150 illicit and 150 licit) changed risk by a mean absolute 0.0065, compared with 0.0041 for matched G replacement on the same nodes; about 42% of the neighbourhood set behind N was invalidated on average. A separate full-recomputation check on a 100-edit subset gave mean absolute changes of 0.0107 (global graph features held fixed) and 0.0108 (PageRank/HITS/k-core recomputed), with correlation 0.999 between the two sets of prediction changes. These are different sample sizes/quantities and should not be conflated. Even full recomputation does not refresh the N features, so the edit is best described as “G edited, N stale”, not an isolated G intervention.

## Important methodological qualifications

1. Feature groups are dependent and their boundaries are operational.
2. Matched replacement estimates sensitivity to a specified donor-replacement procedure; it does not establish causal contribution.
3. Donor rows are not perfectly exchangeable with targets, as indicated by the donor realism checks.
4. The main Random Forest is fitted on labelled training-period rows only and uses the original Elliptic features directly; the notebook does not fit a feature scaler on the test set for classifier training. Separately, the primary same-time donor-matching procedure rank-normalises its matching features with `QuantileTransformer` objects fitted on the full transaction pool from the evaluation time steps, including unlabelled transactions. This uses evaluation-period covariates to define donor similarity, but does not use evaluation labels or refit the predictive model. It is a transductive aspect of the donor-selection procedure and should be disclosed when interpreting the intervention analysis. The historical-donor sensitivity analysis fits its matching transformations on the historical pool only.
5. Joint replacement effects are descriptive and need not equal the sum of individual effects; joint donor pools differ.
6. SHAP values explain the fitted model's output under a background distribution; they do not resolve feature dependence or causal direction.
7. Graph edits may affect derived features beyond the directly edited structural descriptor.
8. The GraphSAGE explanation code implements a custom GNNExplainer-inspired mask objective in PyTorch. It is not an invocation of the official PyTorch Geometric GNNExplainer API. The four selected examples are illustrative, not a representative evaluation.

## Setup and execution

### Recommended: Kaggle Notebooks

This repository contains a Jupyter notebook (`.ipynb`), and **Kaggle Notebooks is the recommended environment for running it**. A notebook file is not tied to local Jupyter; it can be opened and executed in Kaggle.

1. Sign in to [Kaggle](https://www.kaggle.com/) and create a new notebook.
2. Import or upload `Fraud_Risk_Detection.ipynb` from this repository. You can also download the notebook from the [GitHub file page](https://github.com/Diya-BR/Fraud-Risk-Interventions/blob/main/Fraud_Risk_Detection.ipynb).
3. In the Kaggle notebook, select **Add Input** and attach the [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).
4. Confirm that the three required CSV files are present under `/kaggle/input/`. Use the file-search snippet in the [Dataset files in Kaggle](#dataset-files-in-kaggle) section if you need to locate them.
5. If the notebook cannot locate the files, update the candidate paths in its dataset-location cell to match the paths shown by the search snippet.
6. Run the notebook from the first cell to the last. Review the output of each cell for errors; do not rely only on the saved notebook outputs.

Kaggle input datasets are read-only. Save any generated files, checkpoints, or result exports under `/kaggle/working/` (or another writable working directory), not under `/kaggle/input/`. Download any outputs you need to retain before the Kaggle session ends.

### Dependencies and compute

Kaggle commonly provides many scientific Python packages, but the exact installed versions can change. The repository's `requirements.txt` is an unpinned dependency list, not a tested environment lockfile. First run the notebook's imports. If a required package is missing, install that package in a Kaggle cell, for example:

```python
%pip install shap
```

Use the corresponding package name for any other missing dependency. Avoid blindly reinstalling every package in `requirements.txt` in an existing Kaggle image, because doing so can upgrade or downgrade packages already provided by Kaggle. PyTorch is required for the GraphSAGE section; PyTorch Geometric is not required by the custom mask implementation described above.

The GraphSAGE section contains a fallback that trains a checkpoint if `graphsage_t_explainer_checkpoint.pt` is not present. This training and the full notebook may be computationally expensive. A Kaggle accelerator may help, but availability and runtime depend on the account, session settings, and workload. The notebook should still be checked for device compatibility before relying on GPU execution.

### Optional: local Jupyter

Local execution is also possible with Python 3.10 or later as a starting point:

```bash
python -m venv .venv

# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Open `Fraud_Risk_Detection.ipynb`, make the three dataset CSV files accessible to the notebook, update the dataset paths if needed, and run all cells in order.

### Reproducibility note

The notebook is designed to regenerate its analyses from the source CSV files. Random Forest models are fitted in the notebook, and the GraphSAGE section trains and saves a checkpoint when `graphsage_t_explainer_checkpoint.pt` is not present. Thus, the GraphSAGE checkpoint from the recorded run is not a required repository dependency. The notebook includes fixed seeds for its principal workflows, although exact numerical identity can still vary with package versions, hardware, and nondeterministic operations.

One execution detail to check in Kaggle: the notebook currently tests a short list of expected dataset directories rather than recursively searching all of `/kaggle/input/`. If Kaggle mounts the attached dataset at a different path, update the candidate paths in the dataset-location cell. The notebook's saved outputs document a previous execution, but the repository has **not yet been independently verified by a clean, end-to-end run in a fresh Kaggle session**. This is a verification limitation, not evidence that the code requires the omitted checkpoint. For a fresh reproduction, attach the three CSV files, confirm that the notebook locates them, and run all cells in order.

## Acknowledgements and references

- Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics*, KDD 2019. [arXiv:1908.02591](https://arxiv.org/abs/1908.02591).
- Elliptic dataset distribution: [Kaggle dataset page](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).
- PyTorch Geometric dataset documentation: [EllipticBitcoinDataset](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.EllipticBitcoinDataset.html).

## AI assistance

If AI tools were used in preparing code, documentation, or analysis, disclose that assistance in accordance with the course/institution policy. Verify every result and citation against the original notebook and sources before submission.
