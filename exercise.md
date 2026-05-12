# Spectral Classification of SDSS Astronomical Sources

## Overview

In this assignment, you will explore the classification of astronomical spectra from the Sloan Digital Sky Survey (SDSS) using machine learning techniques.

You will work with real spectroscopic data from SDSS Data Release 16 and investigate how different types of astronomical objects can be distinguished based on their spectral signatures.

Rather than following a strict recipe, you are encouraged to explore different preprocessing strategies, visualization choices, and modeling approaches. The goal is to develop both a working classification pipeline and an understanding of the underlying data.

---

## Learning Objectives

By completing this assignment, you should be able to:

- Load and explore high-dimensional scientific datasets
- Visualize and interpret astronomical spectra
- Apply preprocessing and scaling techniques appropriately
- Use PCA for dimensionality reduction and exploratory analysis
- Build end-to-end machine learning pipelines with scikit-learn
- Train at least one supervised classification model
- Evaluate model performance using standard metrics
- Interpret results using confusion matrices and classification reports

---

## Dataset

You are provided with SDSS spectra stored as NumPy arrays:

- `data.npy`  
  Spectral flux values  
  Shape: `(N_samples, N_wavelength_bins)`

- `labels.npy`  
  Integer class labels

- `wavelengths.npy`  
  Wavelength grid in Ångström

The dataset includes three classes:

| Label | Class  |
|------:|--------|
| 0     | AGN    |
| 1     | galaxy |
| 2     | QSO    |

---

## Background

The Sloan Digital Sky Survey (SDSS) is one of the largest astronomical surveys ever conducted, providing spectra for millions of celestial objects.

Astronomical spectra encode physical information such as:
- chemical composition
- temperature
- redshift
- emission and absorption line structure

In this assignment, you will investigate how such information can be used for automatic classification.

---

## Suggested Workflow

You are free to design your own approach, but your final solution should include the following components:

### 1 — Data Exploration

Load and inspect the dataset.

You may want to:
- examine shapes and class distribution
- check for imbalance
- visualize example spectra

---

### 2 — Spectral Visualization

Explore the data visually.

Possible directions include:
- plotting individual spectra from different classes
- comparing mean spectra across classes
- visualizing variability (e.g. standard deviation bands)

Try to develop an intuition for how spectra differ between classes.

---

### 3 — Preprocessing and PCA

Experiment with preprocessing techniques such as:
- normalization or scaling

Then apply PCA to:
- reduce dimensionality
- visualize the dataset in 2D or 3D
- inspect eigen-spectra and explained variance

You may choose preprocessing and PCA settings freely.

---

### 4 — Classification Model

Build a machine learning pipeline that includes:
- preprocessing step(s)
- optional dimensionality reduction
- a supervised classifier

You are required to train at least one model. You may explore any of the following:
- Logistic Regression
- Random Forest
- Gradient Boosting (e.g. XGBoost)
- or other scikit-learn compatible classifiers

Hyperparameter tuning and cross-validation are optional but encouraged.

---

### 5 — Evaluation

Evaluate your final model on a held-out test set.

Report:
- accuracy
- macro F1-score
- confusion matrix
- classification report

---

## Final Goal

Your final submission should include:

- a trained classification pipeline
- a PCA-based visualization of the dataset
- evaluation results on a test set
- a brief interpretation of your findings

---

## Optional Extensions (Not Required)

If you want to go further, you may explore:
- different scaling strategies (StandardScaler vs Normalizer)
- nonlinear dimensionality reduction (t-SNE, UMAP)
- feature importance analysis
- comparison of multiple models
- error analysis on misclassified spectra

---

## References

- Sloan Digital Sky Survey (SDSS): https://www.sdss.org/
- scikit-learn documentation: https://scikit-learn.org/
- XGBoost documentation: https://xgboost.readthedocs.io/
- Yip et al. (2004), *Spectral Classification of Quasars in SDSS: Eigenspectra, Redshift, and Luminosity Effects*
