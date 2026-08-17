# Diffusion Spatial-Temporal WaveNet for Short-term Uncertainty Prediction of Metro Demand during Non-recurrent Events (DiffSTWN)

This repository provides the official implementation of **DiffSTWN** for short-term uncertainty prediction of metro origin-destination (OD) demand during non-recurrent events.

---

## Repository Structure

```text
├── Data/                 # Directory containing empirical datasets and graphs
├── Model/                # Implementation of network modules and DiffSTWN architecture
├── save_model/           # Directory to store trained checkpoints
├── utils/                # Utility scripts (metrics, data loading, preprocessing)
├── main_train.py         # Main execution script for model training and validation
├── main_predict.py       # Script for testing and evaluating model checkpoints
└── README.md

```

---

## Dataset Description

The `Data/` directory contains the processed data required for model execution:

* **Traffic-flow time-series data**: OD sequence data with dimensions $\vert{}V\vert{} \times T$, and Inflow sequence data with dimensions $\vert{}S\vert{} \times T$, where $\vert{}S\vert{}$ is the number of metro stations, $\vert{}V\vert{}$ denotes the set of directed OD pairs ($\vert{}V\vert{} = \vert{}S\vert{} \times \vert{}S\vert{}$), and $T$ is the number of time intervals.

* **OD-pair-based graph data**: Origin- and destination-based graph matrices with dimensions $\vert{}V\vert{} \times \vert{}V\vert{}$

* **External-factor data**: Contextual features with dimensions $5 \times T$, incorporating five types of external conditions (e.g., time of day, day of week, holiday attributes, weather conditions).


---

## Requirements

* Python >= 3.8
* PyTorch >= 1.12.0
* NumPy
* Pandas
* SciPy

---

## Usage

### 1. Model Training & Hyperparameter Configuration

Train the model by running `main_train.py`. All training parameters, diffusion steps, and loss weights can be configured directly inside this script:

```bash
python main_train.py

```

### 2. Model Evaluation & Testing

To evaluate the forecasting performance on the test set, load a trained checkpoint using `main_predict.py`:

```bash
python main_predict.py

```
