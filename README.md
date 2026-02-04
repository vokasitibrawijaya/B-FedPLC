# B-FedPLC: Blockchain-Enabled Federated Learning with Prototype-Anchored Learning and Dynamic Community Adaptation

![IEEE Access](https://img.shields.io/badge/IEEE-Access-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

Official implementation of **B-FedPLC** (Blockchain-Enabled Federated Learning with Prototype-Anchored Learning and Dynamic Community Adaptation for Byzantine-Resilient Distributed Machine Learning).

> [!NOTE]  
> This repository contains the source code for the experiments presented in our **IEEE Access (2026)** paper.

---

## 📄 Paper Information

**Title**: B-FedPLC: Blockchain-Enabled Federated Learning with Prototype-Anchored Learning and Dynamic Community Adaptation for Byzantine-Resilient Distributed Machine Learning

**Abstract**:  
Federated Learning (FL) faces critical challenges in non-IID data environments and vulnerability to Byzantine attacks. Existing solutions often trade off between robust security and model performance. We propose **B-FedPLC**, a novel framework integrating Blockchain-based audit trails, **Prototype-Anchored Regularization Layer (PARL)**, and **Label Distribution-based Community Adaptation (LDCA)**. 

Our results demonstrate that B-FedPLC achieves superior resilience (sustaining **67.70% accuracy** under 20% Byzantine attacks) compared to state-of-the-art methods like Krum and Trimmed Mean, while effectively handling concept drift in heterogeneous data distributions.



**Authors**: Rachmad Andri Atmoko, Sholeh Hadi Pramono, Muhammad Fauzan Edy Purnomo, Panca Mudjirahardjo, Mahdin Rohmatillah, Cries Avian  
**Affiliation**: Universitas Brawijaya, Indonesia  
**Journal**: IEEE ACCESS (2026)

---

## 🚀 Key Features

* **Blockchain-IPFS Integration**: Immutable ledger for model updates with Merkle tree verification ensures complete transparency and auditability.
* **Prototype-Anchored Regularization (PARL)**: Prevents catastrophic forgetting in local training by anchoring updates to global prototypes.
* **Dynamic Community Adaptation (LDCA)**: Dynamically clusters clients based on label distribution to handle statistical heterogeneity (Non-IID).
* **Multi-Layered Byzantine Detection**: Combines statistical Z-score filtering with Cosine Similarity checks to robustly identify malicious actors.

---

## 📂 Project Structure

```bash
B-FedPLC/
├── b_fedplc.py                      # Core implementation of B-FedPLC algorithm
├── ieee_comprehensive_experiment.py # Main script to run all IEEE Access experiments
├── check_experiment_status.py       # Utility to monitor experiment progress
├── requirements.txt                 # Project dependencies
└── paper/
    └── ACCESS_latex_template_20240429/
        └── generate_all_figures.py  # Script to generate publication figures
