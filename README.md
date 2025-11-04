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

## 📁 Project Structure

advanced-ai-probabilistic-reasoning/
│
├── BayesNetReader.py # Reads and tokenises BN configuration files
├── BayesNetUtil.py # Utility functions for BN inference
├── BayesNetInference.py # Exact inference by enumeration
│
├── CSV_DataReader.py # Reads and parses training/test CSV data
├── CPT_Generator.py # Generates CPTs for discrete variables
├── PDF_Generator.py # Generates PDFs for continuous variables
│
├── ModelEvaluator.py # Computes predictive performance metrics
├── discretize_data.py # Discretises continuous data for CPT generation
│
├── config/ # Configuration files (.txt, .pkl)
├── data/ # Training and test datasets (.csv)
└── structures/ # Learned network structures and visualisations
