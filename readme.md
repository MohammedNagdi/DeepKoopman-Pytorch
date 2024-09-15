# Deep Learning for Universal Linear Embeddings of Nonlinear Dynamics

This repository implements the approach from the paper: [Deep Learning for Universal Linear Embeddings of Nonlinear Dynamics](https://www.nature.com/articles/s41467-018-07210-0) in PyTorch. It replicates the experiments for the Pendulum, Discrete Spectrum, and Fluid Flow on Attractor, and adds additional experiments on the Lorenz System and the Repressilator Model.

## Features

- **Multiple Experiments**:
  - Pendulum
  - Discrete Spectrum
  - Fluid Flow on Attractor
  - Lorenz System
  - Repressilator Model

- **Flexible Architectures**:
  - Supports multiple architectural variations for the Koopman Autoencoder and Koopman Operator.
  - Choose between MLP or KAN networks for either or both components.

- **Koopman Operator Variations**:
  1. Koopman operator with a combination of complex conjugate eigenvalues and real values.
  2. Koopman operator with only complex conjugate eigenvalues or real values.
  3. Ensemble of Koopman operators with complex conjugate and real operators.

- **Koopman Implementation**:
  - Two variations of KAN networks: "Fast KAN" and "Efficient KAN".

- **Dataset**:
  - Includes generated datasets for the three main experiments.
  - Dataset generator is provided for additional experiments.

## Running Experiments

### 1. Setup

Before running any experiment, create a YAML configuration file that defines the required hyperparameters, including learning rates, architecture options, and experiment-specific parameters.

### 2. Training

Use `train.py` to run the training for your experiment. Specify the path to the YAML file with the hyperparameters.

- **Logs**: During training, logs will display the Koopman loss for both training and validation datasets, along with the prediction loss for the validation dataset.
- **Visualization**: Training will also visualize the Koopman loss and prediction loss over time.

### 3. Evaluation

After training, use `evaluate.py` to evaluate the model on the test dataset. This script will visualize the results for the specified time series of the system being tested.

## File Descriptions

- **`train.py`**: Script to run the training, specifying the YAML file with hyperparameters and experiment details.
- **`evaluate.py`**: Script to evaluate the trained model on the test dataset and visualize the results.
- **Data Files**: Contains the generated datasets for the experiments and a generator to create custom datasets.
