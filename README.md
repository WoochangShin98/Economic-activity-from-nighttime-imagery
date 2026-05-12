# Estimating County-Level Poverty Rates from Nighttime Satellite Imagery

This project investigates whether nighttime satellite imagery can be used to predict county-level poverty rates in the mainland United States using machine learning and deep learning approaches. The project uses 2024 VIIRS nighttime-light imagery and county-level poverty statistics from the SAIPE dataset.

---

## Overview

Traditional economic indicators such as poverty rates are typically obtained through surveys and official statistical processes, which can be expensive and time-consuming. This project explores whether remotely sensed nighttime-light imagery can serve as a scalable proxy for local economic conditions.

Compared with the original proposal focused on GDP-tier classification, the project was redesigned as a poverty-rate regression task. This regression formulation preserves continuous socioeconomic variation across counties while avoiding dependence on arbitrary classification thresholds.

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
- Coverage: Mainland United States, excluding Alaska and Hawaii

### Poverty Labels
- Source: Small Area Income and Poverty Estimates (SAIPE)
- Target variable: County-level poverty rate (%)

### Geographic Data
- County boundaries obtained from TIGER/Line shapefiles

References:
- VIIRS: https://eogdata.mines.edu/products/vnl/
- SAIPE: https://www.census.gov/data-tools/demo/saipe/
- TIGER/Line: https://www.census.gov/cgi-bin/geo/shapefiles/index.php

---

## Methodology

The project compares three modeling approaches.

### 1. Statistical Regression Baseline

This baseline uses handcrafted nighttime-light features such as:

- Mean brightness
- Standard deviation
- Percentile statistics
- Lit-pixel ratio

Models:

- Linear Regression
- Random Forest Regression

---

### 2. CNN-based End-to-End Regression

A convolutional neural network directly predicts county-level poverty rates from 64×64 nighttime-light images.

This approach allows the model to learn:

- Spatial light distributions
- Urban structure patterns
- Regional illumination characteristics

---

### 3. Hybrid CNN + Regression Model

This hybrid approach:

1. Extracts learned spatial embeddings from a trained CNN
2. Uses those embeddings as input features for a downstream regression model

The goal is to evaluate whether learned spatial representations improve prediction beyond handcrafted brightness features.

---

## Data Processing Pipeline

The preprocessing pipeline includes:

- Downloading 2024 VIIRS nighttime-light data
- Loading county polygons from TIGER/Line shapefiles
- Aligning county FIPS codes with SAIPE poverty labels
- Cropping county-level nighttime-light regions
- Normalizing images to 64×64 resolution
- Constructing image-label pairs for training

At this stage, the dataset construction and preprocessing pipeline have been completed.

---

## Preprocessing Options

Two preprocessing versions are included for comparison.

### Option 1: County-only Images

- County-level nighttime-light images
- Original county proportions preserved
- County-only extraction
- Grayscale nighttime imagery
- Border cropping applied
- Some files were removed due to GitHub file size limitations

### Option 2: Dark Background Images

- Alternative preprocessing version
- Dark background style
- County border included
- Grayscale nighttime imagery

The final preferred preprocessing setup uses a dark background, county borders, grayscale imagery, original proportional county sizes, and border cropping. Option 1 was selected as the primary preprocessing approach because it better preserves nighttime-light intensity information and county-scale variation. Option 2 is retained for comparison and preprocessing experiments.

---

## Example

The project analyzes variation in poverty rates across counties despite similar nighttime brightness levels.

Example counties from the Twin Cities region include:

- Hennepin County
- Ramsey County
- Dakota County
- Washington County

The report highlights that bright urban centers may still exhibit high poverty rates, indicating that brightness alone may not fully explain socioeconomic conditions.

---

## Evaluation Metrics

The models will be evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

The primary research question is whether spatial image patterns improve prediction accuracy compared with simple summary statistics.

---

## Current Progress

Completed:

- Data collection
- County image extraction
- Image normalization
- Label alignment
- Preprocessing pipeline
- Uploading reduced/sample processed data to GitHub

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
│   ├── README.md
│   ├── opt1_county_only.zip
│   └── opt2_county_dark.zip
│
├── src/
│   └── CNN_preprocessing_data.ipynb
│
├── results/
│
├── figures/
│
└── README.md
