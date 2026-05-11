# Spectral Classification of SDSS Astronomical Sources

## Overview

In this assignment you will build a machine learning pipeline to classify astronomical spectra from the Sloan Digital Sky Survey (SDSS).

You will work with spectroscopic observations from SDSS Data Release 16 and train models to distinguish between different classes of astronomical objects using their spectra.

The assignment combines:

* exploratory data analysis
* dimensionality reduction
* supervised machine learning
* model evaluation
* scientific interpretation

You are encouraged to think critically about both the machine learning methodology and the astrophysical meaning of the results.

---

# Learning Objectives

By the end of this assignment you should be able to:

* Load and inspect astronomical spectral data
* Visualize high-dimensional spectra
* Apply preprocessing and normalization techniques
* Use PCA for dimensionality reduction and interpretation
* Build reproducible ML pipelines using scikit-learn
* Perform hyperparameter optimization with cross-validation
* Evaluate multiclass classification models
* Interpret confusion matrices and classification metrics

---

# Dataset

The dataset contains SDSS spectra stored as NumPy arrays:

* `data.npy`

  * spectral flux values
  * shape: `(N_samples, N_wavelength_bins)`

* `labels.npy`

  * integer class labels

* `wavelengths.npy`

  * wavelength values in Ångström

The dataset contains the following classes:

| Label | Class  |
| ----- | ------ |
| 0     | AGN    |
| 1     | galaxy |
| 2     | QSO    |

---

# Background

The Sloan Digital Sky Survey (SDSS) is one of the largest astronomical surveys ever conducted.

Astronomical spectra contain information about:

* chemical composition
* temperature
* velocity
* redshift
* emission and absorption features

Machine learning methods can classify astronomical sources directly from their spectra.

---

# Part 1 — Data Loading and Inspection

## Tasks

1. Load the dataset using NumPy.
2. Print:

   * data shape
   * label shape
   * wavelength shape
3. Compute the number of samples in each class.
4. Verify whether the dataset is balanced.

---

# Part 2 — Spectral Visualization

## Tasks

1. Plot one random spectrum from each class.
2. Label:

   * x-axis
   * y-axis
   * title
3. Compute and plot:

   * mean spectrum per class
   * ±1 standard deviation region
4. Plot the global mean spectrum across the full dataset.

## Questions

1. Do the classes appear visually separable?
2. Which wavelength regions appear most different between classes?
3. What are some limitations of visual inspection in high-dimensional datasets?

---

# Part 3 — Exploratory PCA Analysis

## Tasks

1. Split the dataset into:

   * training set
   * independent test set

2. Normalize the spectra using L2 normalization.

3. Apply PCA with:

   * `n_components=5`

4. Visualize:

   * projection onto the first two principal components
   * explained variance ratio
   * first two eigenspectra

## Questions

1. What fraction of variance is explained by the first components?
2. Are the classes separated in PCA space?
3. What physical properties might the first principal component represent?
4. Why is PCA useful for high-dimensional spectral data?

---

# Part 4 — Machine Learning Pipeline

## Tasks

Build a scikit-learn pipeline containing:

* spectral normalization
* classifier

You must evaluate at least two of the following models:

1. Logistic Regression
2. Random Forest
3. XGBoost

---

# Part 5 — Model Evaluation

## Tasks

Evaluate the best model on the untouched test set.

Compute:

* accuracy
* macro F1-score
* confusion matrix
* classification report

## Questions

1. Which classes are easiest to classify?
2. Which classes are most frequently confused?

---

# References

* The Sloan Digital Sky Survey: [SDSS Official Website](https://www.sdss.org/?utm_source=chatgpt.com)
* scikit-learn documentation: [scikit-learn](https://scikit-learn.org/stable/?utm_source=chatgpt.com)
* XGBoost documentation: [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/?utm_source=chatgpt.com)
* Yip et al. (2004), *Spectral Classification of Quasars in the Sloan Digital Sky Survey*

