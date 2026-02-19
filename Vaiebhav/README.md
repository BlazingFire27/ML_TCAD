# ML_TCAD: Physics-Informed Neural Network for Silicon Oxidation

A modular Physics-Informed Neural Network (PINN) framework for predicting oxygen concentration profiles during silicon oxidation : one of the most critical processes in semiconductor fabrication.

This project combines:

• Data-driven learning
• Physics constraints
• Hyperparameter optimization
<br>
to model oxidation dynamics using TCAD simulation data.

---

## Project Overview

The goal is to predict oxygen concentration inside silicon as a function of:

* Position (x)
* Time (t)
* Oxygen flow
* Nitrogen flow
* Temperature
<br>
Instead of relying purely on data, the neural network is guided by physical laws through PINN constraints — improving generalization across unseen process conditions.

---

## Project Structure

The project follows a modular research-friendly architecture:
<br>
```
project/
│
├── src/
│   ├── data/
│   │   └── loader.py
│   │
│   ├── physics/
│   │   └── oxidation_problem.py
│   │
│   ├── model/
│   │   └── network.py
│   │
│   ├── training/
│   │   ├── objective.py
│   │   └── trainer.py
│   │
│   ├── evaluation/
│   │   └── metrics.py
│   │
│   └── config.py
│
├── main.py
├── requirements.txt
└── README.md
```

### What each module does

| Module        | Role                                                   |
| ------------- | ------------------------------------------------------ |
| `data/`       | Loads and preprocesses TCAD CSV simulation data        |
| `physics/`    | Defines oxidation as a PINN-compatible physics problem |
| `model/`      | Builds the neural network architecture                 |
| `training/`   | Handles training + Optuna hyperparameter search        |
| `evaluation/` | Computes performance metrics                           |
| `main.py`     | Entry point for experiments                            |

---

## Installation

### Requirements

* Python 3.8+
* PyTorch
* PINA
* Optuna
* NumPy
* Pandas
* Scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

Place all TCAD CSV files inside:

```
data/
```

Expected CSV columns:

```
X, Y, Time, O2 Flow, N2 Flow, Temperature
```

The loader automatically:

* Cleans numeric columns
* Removes invalid values
* Converts oxygen concentration to log-scale

---

## Usage

Run the full pipeline:

```bash
python main.py
```

This will:

• Load dataset
• Construct physics-informed problem
• Build neural network
• Run Optuna hyperparameter optimization
• Train PINN model

---

## Model Inputs

The neural network receives:

| Feature | Description      |
| ------- | ---------------- |
| x       | Spatial position |
| t       | Time             |
| o2      | Oxygen flow      |
| n2      | Nitrogen flow    |
| temp    | Temperature      |

Output:

```
log10(Oxygen Concentration)
```

---

## Hyperparameter Optimization

Optuna is used to search for:

* Number of layers
* Neurons per layer
* Activation function
* Learning rate

This avoids manual tuning and improves convergence.

---

## Why PINNs?

Standard neural networks learn patterns.

PINNs learn patterns **and obey physics**.

That means:

Better extrapolation
Fewer data requirements
Physically meaningful predictions

Especially important in semiconductor process modeling.

---

## GPU Support

Training automatically uses:

1. CUDA (NVIDIA GPU)
2. CPU fallback