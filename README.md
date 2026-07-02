## Overview

This project implements a **Genetic Algorithm (GA)** to solve the Multi-Skilled Resource-Constrained Project Scheduling Problem (MS-RCPSP), enhanced with a **Machine Learning model (XGBoost)** that predicts optimal GA parameters based on instance characteristics.

The goal is to improve:
- Solution quality (makespan)
- Computational efficiency
- Automatic parameter tuning

---


## Machine Learning Models

This project contains two different ML-assisted optimization strategies:

### 1. Discrete ML-Assisted GA (`model_AG.py`)

- Parameters are selected from a predefined discrete set
- ML model predicts the best configuration among fixed candidates
- Acts as a classification-style hyperparameter selector

Example parameter space:
- Population size: {50, 100, 150, 200}
- Mutation rate: {0.05, 0.1, 0.2}
- Crossover rate: fixed candidate values

---

### 2. Continuous ML-Assisted GA (`model_continue.py`)

- Parameters are sampled from continuous intervals
- XGBoost evaluates large candidate populations (10,000+ samples)
- Local refinement is applied around elite solutions

Parameter ranges:
- Population size: [20, 200]
- Crossover rate: [0.5, 0.99]
- Mutation rate: [0.01, 0.25]
- Generations: [50, 1000]

This approach behaves as:
> regression-guided hyperparameter optimization with local search refinement

## Features

- MS-RCPSP instance parsing
- Feature extraction from scheduling instances
- Genetic Algorithm implementation:
  - precedence-based encoding
  - resource-constrained decoding
  - tournament selection
  - crossover and mutation operators
- Machine Learning model (XGBoost):
  - Predicts optimal GA parameters
  - Predicts makespan and runtime
- Automatic GA parameter recommendation
- Hybrid ML + GA optimization pipeline

---

## Workflow

1. Load MS-RCPSP instance
2. Extract structural features
3. Train / load XGBoost models
4. Predict best GA parameters
5. Run Genetic Algorithm
6. Output optimal schedule and makespan

---

## How to Run

### Run ML-assisted GA (recommended)

```bash
python model_continue.py

Run Parametres  GA
python model_AG.py

Run classical GA
python model_AG.py

Input Format

The project uses .msrcp benchmark files located in:

MSLIB4/MSLIB4/

Each file contains:

Project module
Activity precedence graph
Skill requirements
Resource availability
Output

The program outputs:

Best GA parameters (ML predicted)
Best schedule found
Final makespan
Model evaluation metrics:
MAE
RMSE
MAPE
Machine Learning Model

Two XGBoost models are trained:

1. Makespan Prediction Model

Predicts the expected quality of a GA configuration.

2. Execution Time Model

Predicts computational cost of a configuration.

GA Parameters Tuned
Population size
Crossover rate
Mutation rate
Number of generations
Dataset

Training data is stored in:

GA_Optimization_Results_with_Time.xlsx

It contains historical GA runs with:

instance features
GA parameters
achieved makespan
execution time
Dependencies

Install required packages:

pip install -r requirements.txt
Author

This project was developed as part of a research study on:
Machine Learning assisted optimization for MS-RCPSP

License

For academic and research use only.


---

# ⚡ Done

You now only need to:

1. create `README.md`
2. create `requirements.txt`
3. run:

```bash
pip install -r requirements.txt