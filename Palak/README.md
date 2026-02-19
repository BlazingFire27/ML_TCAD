# ML_TCAD: Physics-Informed Neural Network for Silicon Oxidation

A machine learning project that uses Physics-Informed Neural Networks (PINNs) to predict oxygen concentration profiles during silicon oxidation, a critical process in semiconductor manufacturing.

## Project Overview

This project models the oxidation process of silicon wafers using a neural network trained on TCAD (Technology Computer-Aided Design) simulation data. The model predicts oxygen concentration (O₂) as a function of position, pressure, oxygen concentration, nitrogen concentration, temperature, and time.

**Key Features:**
- Physics-informed neural network architecture with tanh activations for smooth derivatives
- Normalized input features and output predictions
- Held-out test set from unseen simulations for robust evaluation
- GPU acceleration support (MPS, CUDA, or CPU fallback)

## Project Structure

### Core Files

| File | Description |
|------|-------------|
| [`model.py`](model.py) | Defines the PINN architecture (feedforward network with 6 hidden layers, size 256, tanh activations) |
| [`train.py`](train.py) | Training script with validation, early stopping, and checkpoint saving |
| [`evaluate.py`](evaluate.py) | Evaluation script that loads the best model and generates predictions on held-out test samples |
| [`data_loader.py`](data_loader.py) | Data loading utilities for reading and preprocessing CSV files |
| [`save_dataset.py`](save_dataset.py) | Script to load raw data, normalize features, and create train/val/test caches |

### Data Files

| File | Description |
|------|-------------|
| `Data/` | Directory containing 80+ CSV files with TCAD simulation data |
| `dataset_cache.pt` | Cached training and validation tensors (inputs, targets, normalization stats) |
| `test_cache.pt` | Cached test set from held-out simulations |

### Output Files

| File | Description |
|------|-------------|
| `best_model.pth` | Best trained model checkpoint (saved during training based on validation loss) |
| `predictions.csv` | Model predictions on test set with true values and errors |
| `test_predictions.csv` | Alternative test predictions file |
| `profile_plot.png` | Visualization of concentration profiles |

### Utility Files

| File | Description |
|------|-------------|
| `plot_profile.py` | Plotting utilities for visualizing concentration profiles |
| `main.py` | Main script with data exploration and statistics |

## Installation

### Requirements
- Python 3.8+
- PyTorch (with GPU support optional)
- NumPy, Pandas

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Crystal-Blaze13/ML_TCAD.git
cd ML_TCAD
```

2. Install dependencies:
```bash
pip install torch numpy pandas
```

## Usage

### 1. Prepare the Dataset

If starting from scratch, create the train/val/test caches from raw CSV files:

```bash
python save_dataset.py
```

This will:
- Load all CSV files from the `Data/` directory
- Normalize features to [0, 1]
- Split into train (70%), validation (15%), and test (15%) sets
- Save caches to `dataset_cache.pt` and `test_cache.pt`

### 2. Train the Model

```bash
python train.py
```

This will:
- Load the cached dataset
- Initialize a PINN with 6 layers, 256 hidden units
- Train with Adam optimizer and learning rate scheduling
- Use early stopping based on validation loss
- Save the best model to `best_model.pth`
- Display training progress and final metrics

**Terminal Output:**
```
Using MPS (Apple Silicon GPU)
Loading cached dataset...
Train: 12345 samples
Val:   2654 samples
Model has 345,600 parameters
Epoch 1/100 | Train Loss: 0.4523 | Val Loss: 0.4102 | LR: 0.001
...
Early stopping triggered at epoch 45
Best validation loss: 0.0234
```

### 3. Evaluate the Model

```bash
python evaluate.py
```

This will:
- Load the best trained model
- Run predictions on the held-out test set
- Compute error metrics (MAE, RMSE, R²)
- Save predictions to `predictions.csv`
- Display per-temperature and per-O₂ performance breakdowns

**Output includes:**
- Overall metrics (MAE, RMSE, correlation)
- Error statistics by temperature
- Error statistics by O₂ concentration
- Prediction CSV with actual vs. predicted values

### 4. Explore Data

```bash
python main.py
```

This displays:
- Dataset statistics (min, max, mean, median of log_y)
- Distribution of oxygen concentrations
- Individual simulation samples

## Model Architecture

**Input Layer:** 6 features (normalized to [0, 1])
- Position (x)
- Pressure
- O₂ concentration
- N₂ concentration
- Temperature
- Time

**Hidden Layers:** 6 layers of 256 units each with tanh activations

**Output Layer:** 1 value (log₁₀ oxygen concentration, normalized to [0, 1])

**Why tanh?** Unlike ReLU, tanh has smooth second derivatives needed for physics-based loss computation (diffusion equation).

## Key Hyperparameters

- **Batch size:** 2048
- **Learning rate:** 0.001 (with exponential decay: 0.95/epoch)
- **Optimizer:** Adam
- **Early stopping:** patience=15 epochs
- **Train/Val/Test split:** 70% / 15% / 15%

## Performance

The model achieves strong predictions on unseen simulations:
- Mean Absolute Error (MAE) on test set typically < 0.5 (in log₁₀ scale)
- Predictions maintain physical accuracy across temperature and O₂ ranges

## File Formats

### Input CSV Format
```
x,pres,o2,n2,temp,time,y
0.5,1.0,3.5,78,920,3600,1e-6
...
```

### Predictions CSV Format
```
actual,predicted,error,temp,o2
11.5,11.4,0.1,920,3.5
...
```

## GPU Support

The code automatically detects and uses the best available device:
1. **MPS** (Apple Silicon / M-series chips)
2. **CUDA** (NVIDIA GPUs)
3. **CPU** (fallback)

Check device detection in terminal output during training/evaluation.

## Troubleshooting

**Issue:** `dataset_cache.pt` not found
- **Solution:** Run `python save_dataset.py` first

**Issue:** CUDA out of memory
- **Solution:** Reduce batch size in `train.py` or `evaluate.py`

**Issue:** Slow training on CPU
- **Solution:** Install PyTorch with GPU support for your hardware

## Future Improvements

- [ ] Add physics-based loss terms (diffusion equation constraints)
- [ ] Implement uncertainty quantification
- [ ] Optimize hyperparameters with grid/random search
- [ ] Support for different oxidation materials
- [ ] Export model to ONNX for production use

## License

[Add your license here]

## Author

Created for TCAD simulation acceleration and silicon oxidation modeling.
