# Transformers For Multivariate Energy Forecast

**Type:** Seminar Paper

**Authors:** Ekaterina Basova, Emircan Ince , Yash Chougule

**Supervisor:** Georg Velev

## Table of Content

- [Summary](#summary)
- [Working with the repo](#Working-with-the-repo)
  - [Dependencies](#Dependencies)
  - [Setup](#Setup)
- [Reproducing results](#Reproducing-results)
  - [Training code](#Training-code)
  - [Pretrained models](#Pretrained-models)
- [Results](#Results)
- [Project structure](-Project-structure)

## Summary

Empirical evaluation of recently published transformer-based models for multivariate energy time series forecasting.

## Working with the repo

Python version - 3.12.2

### Setup

1. Clone this repository

2. Create an virtual environment and activate it

```bash
python -m venv transf_mv_forecast
source transf_mv_forecast/bin/activate
```

3. Install requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproducing results

To reproduce the results please clone the repository, insert data file into data folder, install the requirements and run the notebook code.

### Training code

Main files of transformer based models are called using an unification code.

### Pretrained models

The only pre trained model used in the analysis is Chronos.

## Results

**Model** **RMSE**

0 Linear Regression 12.29
1 LSTM 14.72
2 Chronos 33.28
3 Nonstationary Autoformer 0.14
4 Basisformer 0.15
5 iTransformer 0.12

## Project structure

```bash
├── Basisformer                                     -- Basisformer model files
    ├── records                                     -- error metrics from trial different runs
    ├── evaluate_tool.py                            -- code for defining evaluation metrics
    ├── main.py                                     -- code for train, test and arguments
    ├── pyplot.py                                   -- code for plotting functions
    └── utils.py                                    -- supporting functions
├── checkpoints                                     -- iTransformer and nsAutoformer checkpoints
├── data                                            -- empty folder, insert data file there
├── iTransformer                                    -- iTransformer model files
    ├── layers                                      -- python files for encoder-decoder, embedding and attention
    ├── utils                                       -- supporting functions
    ├── Model.py                                    -- iTransformer model
    ├── exp_basic.py                                -- Skeleton for implementing the model, training, and testing methods
    └── experiment.py                               -- Inherited from exp_basic with code implemented for model training and testing
├── ns_Autoformer                                   -- nsAutoformer model files
    ├── layers                                      -- python files for encoder-decoder, embedding and attention
    ├── ns_layers                                   -- python files for nonstationary encoder-decoder, embedding and attention
    ├── utils                                       -- supporting functions
    ├── main.py                                     -- code for train, test and arguments
    └── ns_Autoformer.py                            -- code for nonstationary autoformer model
├── records                                         -- Basisformer plots and results
├── results                                         -- iTransformer and nsAutoformer train results
├── supporting_files_chronos                        -- synthetic data simulation supporting functions
├── supporting_files_dynotears                      -- causal graph supporting visualisations files
├── results                                         -- iTransformer and nsAutoformer test results
├── Notebook.ipynb                                  -- notebook file
├── README.md
└── requirements.txt                                -- required libraries
```
