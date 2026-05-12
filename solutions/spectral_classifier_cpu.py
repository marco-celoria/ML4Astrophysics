"""
Supervised classification of astronomical spectra from the Sloan Digital Sky Survey (SDSS).
This script performs data loading, visualization, PCA dimensionality reduction, 
hyperparameter tuning across multiple models (Logistic Regression, Random Forest, XGBoost), 
and saves the outputs and plots to dedicated directories.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib
import scipy.stats as stats
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# =============================================================================
# 1. DIRECTORY SETUP
# =============================================================================
PLOT_DIR = "plots_cpu"
MODEL_DIR = "models_cpu"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================================
# 2. DATA LOADING
# =============================================================================
data_path = "../data/spectra/"

print("Loading data...")
data = np.load(data_path + "data.npy")
labels = np.load(data_path + "labels.npy")
wavelengths = np.load(data_path + "wavelengths.npy")
class_names = ["AGN", "galaxy", "QSO"]

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)
print("Wavelength bins:", wavelengths.shape)

unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {class_names[u]}: {c} samples")

# =============================================================================
# 3. DATA VISUALIZATION
# =============================================================================
print("\nGenerating exploratory plots...")

# Plot 1: Random spectrum from each class
f, axs = plt.subplots(1, 3, figsize=(12, 3))
rng = np.random.default_rng(12345)
for i, cls in enumerate(np.unique(labels)):
    idx = np.where(labels == cls)[0]
    n = rng.choice(idx)
    axs[i].plot(wavelengths, data[n], label=f"{class_names[labels[n]]}")
    axs[i].set_xlabel('wavelength(Å)')
    axs[i].set_title(f"{n} - {class_names[labels[n]]}")

axs[0].set_ylabel('flux (10-17 erg/s/cm2/Å)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "random_spectra.png"))
plt.close()

# Plot 2: Mean spectrum and std deviation for each class
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs = axs.ravel()
colors = ["tab:blue", "tab:red", "tab:green"]
for i in (0, 1, 2):
    idx = np.where(labels == i)[0]
    mu = data[idx].mean(0)
    std = data[idx].std(0)
    axs[i].plot(wavelengths, mu, label=f"{class_names[i]}", color=colors[i])
    axs[i].fill_between(wavelengths, mu - std, mu + std, color=colors[i], alpha=0.3)
    axs[i].set_xlabel('wavelength (Å)')
    axs[i].set_ylim(-40, 80)
    axs[i].set_title(f"{class_names[i]}")

axs[0].set_ylabel('flux (10-17 erg/s/cm2/Å)')
axs[2].set_ylabel('flux (10-17 erg/s/cm2/Å)')
axs[-1].plot(wavelengths, data.mean(0), label="mean", color="black")
axs[-1].fill_between(wavelengths, data.mean(0) - data.std(0), data.mean(0) + data.std(0), color="grey", alpha=0.3)
axs[-1].set_xlabel('wavelength (Å)')
axs[-1].set_ylim(-40, 80)
axs[-1].set_title("mean")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "mean_spectra.png"))
plt.close()

# Plot 3: Class Distribution
plt.figure()
mapped_labels = [class_names[i] for i in labels]
sns.countplot(x=mapped_labels, order=class_names)
plt.title("Class Distribution")
plt.savefig(os.path.join(PLOT_DIR, "class_distribution.png"))
plt.close()

# =============================================================================
# 4. DATA SPLIT & PCA ANALYSIS
# =============================================================================
print("\nSplitting data and running PCA...")
X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels.astype("int32"),
    test_size=0.1,
    stratify=labels,
    random_state=42,
    shuffle=True
)

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# PCA Pipeline
pipeline_pca = Pipeline([
    ("normalize", Normalizer(norm="l2")),
    ("pca", PCA(n_components=5, random_state=42))
])

X_proj = pipeline_pca.fit_transform(X_train)

# Plot 4: PCA Projection
fig, ax = plt.subplots(figsize=(6, 5))
scatter = ax.scatter(
    X_proj[:, 0],
    X_proj[:, 1],
    c=y_train,
    s=4,
    cmap="viridis",
    alpha=0.8
)
cbar = fig.colorbar(scatter, ax=ax, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(class_names)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA Projection of SDSS Spectra")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "pca_projection.png"))
plt.close()

# Plot 5: PCA Eigenspectra
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[1].plot(wavelengths, pipeline_pca.named_steps["pca"].mean_, label="Mean")
for i in range(2):
    axs[1].plot(wavelengths, pipeline_pca.named_steps["pca"].components_[i], label=f"Component {(i + 1)}")

axs[1].set_xlabel('Wavelength (Å)')
axs[1].set_ylabel('Scaled flux')
axs[1].set_title('Mean Spectrum and Eigen-spectra')
axs[1].legend()
axs[0].bar(np.arange(1, 6), pipeline_pca.named_steps["pca"].explained_variance_ratio_)
axs[0].set_xlabel("Principal Component")
axs[0].set_ylabel("Explained Variance Ratio")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "pca_eigenspectra.png"))
plt.close()

# =============================================================================
# 5. MODEL PIPELINE & SELECTION
# =============================================================================
pipeline = Pipeline([
    ("preprocess", "passthrough"),
    ("pca", "passthrough"),
    ("classifier", LogisticRegression())
])

models = {
    "logit": (
        LogisticRegression(max_iter=10000, solver="lbfgs"),
        {
            "preprocess": [Normalizer(norm="l2")],
            "pca": [
                "passthrough",
                PCA(n_components=0.68, random_state=42),
                PCA(n_components=0.95, random_state=42),
            ],
            "classifier__C": stats.loguniform(1e-1, 1e4),
            "classifier__class_weight": [None, "balanced"]
        }
    ),
    "rf": (
        RandomForestClassifier(random_state=42, n_jobs=1),
        {
            "preprocess": [Normalizer(norm="l2")],
            "pca": [
                "passthrough",
                PCA(n_components=0.68, random_state=42),
                PCA(n_components=0.95, random_state=42),
            ],
            "classifier__n_estimators": stats.randint(100, 400),
            "classifier__max_depth": [5, 10, 20, None],
            "classifier__min_samples_leaf": [1, 2, 5],
            "classifier__max_features": ["sqrt", "log2"],
            "classifier__class_weight": [None, "balanced"]
        }
    ),
    "xgb": (
        xgb.XGBClassifier(eval_metric="mlogloss", objective="multi:softprob", random_state=42, n_jobs=1),
        {
            "preprocess": [Normalizer(norm="l2")],
            "pca": [
                "passthrough",
                PCA(n_components=0.68, random_state=42),
                PCA(n_components=0.95, random_state=42),
            ],
            "classifier__n_estimators": stats.randint(100, 500),
            "classifier__learning_rate": stats.loguniform(1e-3, 1e-1),
            "classifier__max_depth": [3, 4, 5, 6],
            "classifier__subsample": [0.6, 0.8, 1.0],
            "classifier__colsample_bytree": [0.5, 0.8, 1.0],
            "classifier__min_child_weight": [1, 3, 5]
        }
    )
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

results = []
trained_models = {}
best_overall_model = None
best_overall_score = -np.inf
best_model_name = None

for name, (model, params) in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}")
    print(f"{'='*50}")

    pipeline.set_params(classifier=model)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=params,
        n_iter=10,
        cv=cv,
        scoring="f1_macro",
        random_state=42,
        verbose=1,
        n_jobs=-1,
        return_train_score=True
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    trained_models[name] = search
    best_model = search.best_estimator_
    pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")

    print("\nBest Parameters:", search.best_params_)
    print(f"Best CV Score: {search.best_score_:.4f}")
    print(f"Test Accuracy: {acc:.4f} | Test F1 Macro: {f1:.4f}")

    results.append({
        "model": name,
        "accuracy": acc,
        "f1_macro": f1,
        "best_cv_score": search.best_score_,
        "fit_time_sec": elapsed,
        "best_params": search.best_params_
    })

    if search.best_score_ > best_overall_score:
        best_overall_score = search.best_score_
        best_overall_model = best_model
        best_model_name = name

results_df = pd.DataFrame(results).sort_values(by="best_cv_score", ascending=False)
print("\nFinal Results:\n", results_df)

print("\n" + "="*60)
print("BEST OVERALL MODEL")
print("="*60)
print(f"Model: {best_model_name}")
print(f"CV F1 Macro: {best_overall_score:.4f}")
print("\nPipeline:\n", best_overall_model)

# Save best model
model_path = os.path.join(MODEL_DIR, "best_spectrum_classifier_cpu.pkl")
joblib.dump(best_overall_model, model_path)
print(f"\nModel saved to: {model_path}")

# =============================================================================
# 6. EVALUATION METRICS
# =============================================================================
def get_multiclass_metrics(y_actual, y_pred, classes):
    cm = confusion_matrix(y_actual, y_pred)
    for i, cls in enumerate(classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - TP - FP - FN

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        print(f"{cls:10s} Sensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f}")

def report(y_actual, y_pred, classes, title, filename):
    print(classification_report(y_actual, y_pred, target_names=classes))
    print("\nPer-class metrics:")
    get_multiclass_metrics(y_actual, y_pred, classes)

    cm = confusion_matrix(y_actual, y_pred, normalize="true")

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="viridis", 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# =============================================================================
# 7. FINAL MODEL EVALUATION
# =============================================================================
print("\nRunning final evaluation on test set...")
final_model = joblib.load(model_path)
test_predictions = final_model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_predictions)
test_f1 = f1_score(y_test, test_predictions, average="macro")

print(f"Final Test Accuracy: {test_accuracy:.4f}")
print(f"Final Test F1: {test_f1:.4f}")

report_img_path = os.path.join(PLOT_DIR, "final_test_confusion_matrix.png")
report(y_test, test_predictions, class_names, title="Final Test Confusion Matrix", filename=report_img_path)
print(f"\nConfusion matrix saved to: {report_img_path}")
