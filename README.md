# Estimating Economic Activity from Nighttime Satellite Imagery

This project investigates whether nighttime satellite imagery can be used to predict county-level poverty rates in the mainland United States using machine learning and deep learning approaches. The project uses 2024 VIIRS nighttime light imagery and county-level poverty statistics from the SAIPE dataset. :contentReference[oaicite:0]{index=0}

---

## Overview

Traditional economic indicators such as poverty rates are typically obtained through surveys and official statistical processes, which can be expensive and time-consuming. This project explores whether remotely sensed nighttime-light imagery can serve as a scalable proxy for local economic conditions.

Compared with our original proposal focused on GDP-tier classification, the project was redesigned as a poverty-rate regression task. This formulation preserves continuous variation across counties and avoids dependence on arbitrary classification thresholds. 

---

## Project Goals

The main objectives of this project are:

- Construct a county-level nighttime-light image dataset
- Predict county-level poverty rates using satellite imagery
- Compare statistical regression and deep learning approaches
- Evaluate whether spatial nighttime-light patterns contain predictive economic information beyond simple brightness summaries

---

## Dataset

### Nighttime Satellite Imagery
- Source: VIIRS Annual Nighttime Lights (2024)
- Resolution: County-level cropped images normalized to 64×64
- Coverage: Mainland United States (excluding Alaska and Hawaii)

### Poverty Labels
- Source: Small Area Income and Poverty Estimates (SAIPE)
- Target Variable: County-level poverty rate (%)

### Geographic Data
- County boundaries obtained from TIGER/Line shapefiles

References:
- VIIRS: https://eogdata.mines.edu/products/vnl/
- SAIPE: https://www.census.gov/data-tools/demo/saipe/
- TIGER/Line: https://www.census.gov/cgi-bin/geo/shapefiles/index.php

---

## Methodology

The project compares three modeling approaches:

### 1. Statistical Regression Baseline
Uses handcrafted nighttime-light features such as:
- Mean brightness
- Standard deviation
- Percentile statistics
- Lit-pixel ratio

Models:
- Linear Regression
- Random Forest Regression

---

### 2. CNN-based End-to-End Regression
A convolutional neural network directly predicts county poverty rates from 64×64 nighttime-light images.

This approach allows the model to learn:
- Spatial light distributions
- Urban structure patterns
- Regional illumination characteristics

---

### 3. Hybrid CNN + Regression Model
This hybrid approach:
1. Extracts embeddings from a trained CNN
2. Uses those embeddings as input to a regression model

The goal is to evaluate whether learned spatial representations improve prediction beyond handcrafted features. :contentReference[oaicite:2]{index=2}

---

## Data Processing Pipeline

The preprocessing pipeline includes:

- Downloading 2024 VIIRS nighttime-light data
- Loading county polygons from TIGER/Line shapefiles
- Aligning county FIPS codes with SAIPE poverty labels
- Cropping county-level nighttime-light regions
- Normalizing images to 64×64 resolution
- Constructing image-label pairs for training

At this stage, the full dataset construction and preprocessing pipeline has been completed. :contentReference[oaicite:3]{index=3}

---

## Example

The project analyzes variation in poverty rates across counties despite similar nighttime brightness levels.

Example counties from the Twin Cities region include:
- Hennepin County
- Ramsey County
- Dakota County
- Washington County

The report highlights that bright urban centers may still exhibit high poverty rates, indicating that brightness alone may not fully explain socioeconomic conditions. :contentReference[oaicite:4]{index=4}

---

## Evaluation Metrics

The models will be evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

The primary research question is whether spatial image patterns improve prediction accuracy compared with simple summary statistics. :contentReference[oaicite:5]{index=5}

---

## Current Progress

Completed:
- Data collection
- County image extraction
- Image normalization
- Label alignment
- Preprocessing pipeline

In Progress:
- Feature engineering
- CNN training
- Hybrid model training
- Model evaluation
- Error analysis

---

## Repository Structure

```text
project/
│
├── data/
│   ├── viirs/
│   ├── saipe/
│   ├── tiger/
│   └── processed/
│
├── models/
│   ├── regression/
│   ├── cnn/
│   └── hybrid/
│
├── notebooks/
│
├── results/
│
├── figures/
│
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── train_regression.py
│   ├── train_cnn.py
│   ├── hybrid_model.py
│   └── evaluation.py
│
└── README.md
