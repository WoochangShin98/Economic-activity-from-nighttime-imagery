# Estimating County-Level Poverty Rates from Nighttime Satellite Imagery

This project investigates whether nighttime satellite imagery can be used to predict county-level poverty rates across the mainland United States using machine learning and deep learning approaches.

The project uses:

- 2024 VIIRS nighttime-light imagery
- TIGER/Line county polygons
- 2024 SAIPE county-level poverty rates

Compared with the original proposal focused on GDP-tier classification, the task was reformulated as a continuous poverty-rate regression problem.

---

## Models

### Model 1 — Statistical Brightness Regression

Handcrafted brightness features with:

- Linear Regression
- Random Forest Regression

---

### Model 2 — CNN Hybrid Embedding Regression

Pipeline:

County Image → CNN Feature Extractor → Embedding Extraction → Regression Model

CNN embeddings were used with:

- Ridge Regression
- Random Forest Regression

---

### Model 3 — Improved Hybrid CNN Embedding Regression

Additional improvements:

- Logarithmic brightness transformation (`log1p`)
- Grayscale background compositing
- Illumination normalization
- Proportional county scaling
- Border-aware preprocessing
- Tuned Random Forest regression
- Improved embedding stability

---

## Results

| Model | R² |
|---|---:|
| Linear Regression | 0.087 |
| Random Forest | 0.056 |
| End-to-End CNN | 0.067 |
| Ridge on CNN Embeddings | 0.100 |
| RF on CNN Embeddings | 0.115 |
| Improved Ridge on CNN Embeddings | 0.088 |
| Improved RF on CNN Embeddings | **0.141** |

---

## Best-Performing Model

The best-performing model was Random Forest regression on CNN embeddings.

- Best R²: **0.141**

---

## Main Findings

- CNN embedding hybrid models outperformed handcrafted statistical baselines.
- End-to-end CNN regression performed worse than hybrid embedding models.
- Spatial nighttime-light patterns contain socioeconomic information beyond simple brightness statistics.
- Improved preprocessing and hybrid embedding regression produced the strongest predictive performance.
