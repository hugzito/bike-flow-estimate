# bike-flow-estimate
Code Repository for Master's Thesis by Christian Hugo Rasmussen &amp; Manuel Knepper. 

This repository contains the code, data, and documentation for my master's thesis project, which focuses on **predicting and analyzing bike flow in the city of Copenhagen** using **graph learning techniques** such as **Graph Convolutional Networks (GCN)** and **Graph Attention Networks (GAT)**.

---

## Project Overview

This project aims to:
- Model and predict bike flow across the urban road network in Copenhagen.
- Leverage graph-based deep learning architectures (GCN and GAT) to capture both topological and feature-based dependencies.
- Evaluate model performance and explainability through feature importance analysis and case studies.

---

bike-flow-estimate/
├── data/                      # Raw and processed datasets (road network, bike flow data)
│
├── notebooks/                 # Jupyter notebooks for experiments and analysis
│   ├── network-construction.ipynb          # Network construction and feature/target assignment
│   ├── baseline-models.ipynb               # Baseline model experiments (requires constructed networks)
│   ├── case-study.ipynb                    # Reproduces case study from Section 5.6 of the thesis
│   ├── dataset-splitting.ipynb             # Splits data into subsets with selected primary features
│   ├── eda-plots.ipynb                     # Visualizes feature plots from Section 3 of the thesis
│   ├── feature-plot-visualizer.ipynb       # Replicates feature presence table (Appendix A)
│   ├── GNN_explainer_classification.ipynb  # Explains features in GNN classification models
│   ├── GNN_explainer_regression.ipynb      # Explains features in GNN regression models
│   ├── graph_explanations.ipynb            # Visualizes feature importance scores (scripts/explainer.py)
│   ├── model-runs.ipynb                    # Visualizes model results from Weights & Biases exports
│   ├── n2v_baseline.ipynb                  # Node2Vec baseline experiments from the thesis
│
├── scripts/                   # Python scripts for data processing, training, and explainability
│   ├── bike-functions.py                  # Functions for working with OSMNX, NetworkX, and PyTorch Geometric
│   ├── models.py                          # Model architectures from the thesis
│   └── tg_functions.py                    # Training and evaluation utilities for PyTorch Geometric
│
├── requirements.txt          # Python dependencies
├── docker/                   # Containerization files for reproducible training environments
└── README.md                 # This file

