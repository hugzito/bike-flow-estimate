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
## Repository Structure
```
bike-flow-estimate/ <br>
├── data/ # Contains raw and processed datasets (e.g., road network, bike flow data)
├── notebooks/ # Jupyter notebooks for exploration, experiments, and visualization 
│ ├── network-construction.ipynb # Construction of networks and assignment of features + target variable 
│ ├── baseline-models.ipynb # Running of baseline models (requires constructed networks)  
│ ├── case-study.ipynb # Notebook for replication of case study from section 5.6 from thesis paper  
│ ├── dataset-splitting.ipynb # Notebook for splitting into subsets of desired primary features (and replicate feature-sets from study) 
│ ├── eda-plots.ipynb # Notebook to visualize and replicate feature plots from section 3 in the thesis paper 
│ ├── feature-plot-visualizer.ipynb # Notebook to replicate the feature presence table from appendix A 
│ ├── GNN_explainer_classification.ipynb # Notebook to explain features of GNN models trained on classification tasks 
│ ├── GNN_explainer_regression.ipynb # Notebook to explain features of GNN models trained on regression tasks 
│ ├── graph_explanations.ipynb # Notebook to visualize the feature importance scores calculated in script "scripts/explainer.py"
│ ├── model-runs.ipynb # Notebook to visualize the results of models based on wandb results export 
│ ├── n2v_baseline.ipynb # Notebook to calculate the N2V model from the thesis paper 
├── scripts/ # Python scripts for preprocessing, training, evaluation, and explainability 
│ ├── bike-functions.py # Script containing various composite functions of OSMNX, NX and Torch-geometric used mainly to handle and construct graphs 
│ ├── models.py # Script containing the model architectures used in the thesis paper
│ ├── tg_functions.py # Torch-geometric based functions to handle the training and testing of models
│ ├── script.py # The main script used inside the docker container to train the GNN-models
├── requirements.txt # Python dependencies
├── docker # Container directory for training and saving models in same or similar data 
│ ├── script.py # The main script used inside the docker container to train the GNN-models 
│ ├── dockerfile # Dockerfile used to create container usable to run models 
└── README.md # This file 
```
