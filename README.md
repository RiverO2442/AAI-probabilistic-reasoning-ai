# 🧠 Advanced AI – Probabilistic Reasoning

This repository contains an implementation of **probabilistic reasoning** for two domains:  
- 🏦 **Fraud Detection**  
- ❤️ **Heart Disease Diagnosis**  

using **Bayesian Networks (BNs)** and **Gaussian Processes (GPs)**.

It extends the Advanced AI module at the University of Lincoln by combining **discrete** and **continuous** probabilistic models to perform:
- Bayesian inference on complex datasets  
- Probabilistic reasoning with learned network structures  
- Comprehensive evaluation using modern metrics (AUC, ECL, KL Divergence, etc.)

---
| 🧩 **Stage**                         | **Task**                      | **Description**                                                                    | **Status**     |
| ------------------------------------ | ----------------------------- | ---------------------------------------------------------------------------------- | -------------- |
| 🏗️ **Setup**                        | Project Initialization        | Integrated workshop code (`BayesNetReader`, `BayesNetUtil`, `CPT_Generator`, etc.) | ✅ Done         |
| 💾 **Data Loading**                  | Dataset Import                | Loaded Heart Disease and Fraud Detection datasets                                  | ✅ Done         |
| 🧹 **Preprocessing**                 | Data Cleaning                 | Removed unnecessary columns, handled missing values                                | ✅ Done         |
| ⚖️ **Balancing (Fraud)**             | Class Rebalancing             | Applied **SMOTE** to handle fraud imbalance                                        | ✅ Done         |
| 🧮 **Discretization (Heart)**        | Feature Binning               | Converted continuous features into discrete bins                                   | ✅ Done         |
| 🧠 **Bayesian Network Setup**        | Manual Structure Definition   | Defined causal relationships between variables                                     | ✅ Done         |
| 🔧 **Structure Fixing**              | Parent Node Validation        | Fixed missing parent variables (e.g., `Location`, `Authentication_Method`)         | ✅ Done         |
| 📊 **CPT Generation**                | Discrete Probabilities        | Used `CPT_Generator.py` with Laplace smoothing to estimate CPTs                    | ✅ Done         |
| 🔍 **Inference**                     | Exact Probabilistic Reasoning | Implemented and tested **Inference by Enumeration** (`BayesNetInference.py`)       | ✅ Done         |
| 🧪 **Heart Evaluation**              | Model Testing                 | Evaluated Heart BN — stable inference with solid results                           | ✅ Done         |
| 💳 **Fraud Evaluation**              | Model Testing                 | Evaluated Fraud BN — handled high-dimensional discrete inputs                      | ✅ Done         |
| 📈 **Metrics Computation**           | Performance Evaluation        | Balanced Accuracy, F1, AUC, Brier, KL, ECL, Inference Time                         | ✅ Done         |
| 🔄 **Cross-Validation (CV)**         | K-Fold Validation             | Implement automated **5-fold CV** using `run_cv_pipeline.py`                       | 🟡 In Progress |
| 🧮 **Gaussian Bayesian Network**     | Continuous Data Modelling     | Use `PDF_Generator.py` for continuous variable inference                           | 🟡 Next        |
| 🤖 **Gaussian Process (Optional)**   | GP Classifier Baseline        | Compare GP vs BN performance (using GPyTorch or sklearn)                           | ⚪ Optional     |
| 🧩 **Structure Learning (Optional)** | Automated BN Learning         | Use `bnlearn`’s `hillclimbsearch` (BIC/BDeu) for learned structure                 | ⚪ Optional     |
| 📉 **Results Aggregation**           | Combine Fold Metrics          | Compute mean ± std for all metrics, output summary tables                          | 🟡 Next        |
| 📊 **Visualisations**                | Graphs & Plots                | BN structures, ROC curves, calibration plots                                       | 🟡 Next        |
| 🧾 **Report Writing**                | IEEE-Style Report (~4 pages)  | Include Intro, Methods, Results, Discussion, Conclusion                            | 🟡 Next        |
| 📦 **Submission Package**            | Final Deliverables            | Repo + report + results CSVs + figures                                             | ⚪ Pending      |
