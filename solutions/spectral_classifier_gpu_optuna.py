import os
import time
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# GPU Stack
import cudf
import cupy as cp
import cuml
import xgboost as xgb

from cuml.pipeline import Pipeline
from cuml.preprocessing import Normalizer
from cuml.decomposition import PCA
from cuml.linear_model import LogisticRegression
from cuml.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)

# --- Configuration & Directory Setup ---
DATA_PATH = "../data/spectra/"
PLOT_DIR = "plots_gpu"
MODEL_DIR = "models_gpu"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_NAMES = ["AGN", "galaxy", "QSO"]

def save_plot(filename):
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path)
    print(f"Saved plot to {path}")
    plt.close()

# --- Helper Functions ---
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
    """Refactored report to save to file instead of blocking execution."""
    print(f"\n--- {title} ---")
    print(classification_report(y_actual, y_pred, target_names=classes))
    print("Per-class metrics:")
    get_multiclass_metrics(y_actual, y_pred, classes)
    
    cm = confusion_matrix(y_actual, y_pred, normalize="true")
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="viridis", 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    save_plot(filename)

# --- 1. Load Dataset ---
print("Loading datasets...")
data = np.load(os.path.join(DATA_PATH, "data.npy"))
labels = np.load(os.path.join(DATA_PATH, "labels.npy"))
wavelengths = np.load(os.path.join(DATA_PATH, "wavelengths.npy"))
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)
print("Wavelength bins:", wavelengths.shape)

# EDA: Visualizing Spectra
f, axs = plt.subplots(1, 3, figsize=(12,3))
rng = np.random.default_rng(12345)
for i,cls in enumerate(np.unique(labels)):
    idx = np.where(labels == cls)[0]
    n = rng.choice(idx)
    axs[i].plot(wavelengths, data[n], label=f"{CLASS_NAMES[labels[n]]}")
    axs[i].set_xlabel('wavelength')
    axs[i].set_title(f"{n} - {CLASS_NAMES[labels[n]]}")
    
axs[0].set_ylabel('flux)')
save_plot("random_spectra.png")

fig, axs = plt.subplots(2, 2, figsize=(12,12))
axs = axs.ravel()
colors = ["tab:blue", "tab:red", "tab:green"]
for i in (0, 1, 2):
    idx = np.where(labels == i)[0]
    mu = data[idx].mean(0)
    std = data[idx].std(0)
    l = axs[i].plot(wavelengths, mu, label=f"{CLASS_NAMES[i]}", color=colors[i])
    axs[i].fill_between(wavelengths, mu - std, mu + std, color=colors[i], alpha=0.3)
    axs[i].set_xlabel('wavelength')
    axs[i].set_ylim(-40, 80)
    axs[i].set_title(f"{CLASS_NAMES[i]}")
    
axs[0].set_ylabel('flux')
axs[2].set_ylabel('flux')
axs[-1].plot(wavelengths, data.mean(0), label=f"mean", color="black")
axs[-1].fill_between(wavelengths, data.mean(0) - data.std(0), data.mean(0) + data.std(0), color="grey", alpha=0.3)
axs[-1].set_xlabel('wavelength')
axs[-1].set_ylim(-40, 80)
axs[-1].set_title("mean")
save_plot("mean_spectra.png")

# Maps the integer labels to their actual string names for the plot
mapped_labels = [CLASS_NAMES[i] for i in labels]
sns.countplot(x=mapped_labels, order=CLASS_NAMES)
plt.title("Class Distribution")
save_plot("class_distribution.png")

# --- 2. Data Splitting & GPU Transfer ---
X_train, X_test, y_train, y_test = train_test_split(
    data, labels.astype("int32"), test_size=0.1, 
    stratify=labels, random_state=42, shuffle=True
)
print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

print("Moving data to GPU...")
X_train_gdf = cudf.DataFrame(X_train.astype(np.float32))
y_train_gdf = cudf.Series(y_train.astype(np.int32))
X_test_gdf = cudf.DataFrame(X_test.astype(np.float32))
y_test_gdf = cudf.Series(y_test.astype(np.int32))

# --- 3. PCA & Eigen-analysis ---
pipeline_pca = Pipeline([
    ("normalize", Normalizer(norm="l2")),
    ("pca", PCA(n_components=5))
])
X_proj = pipeline_pca.fit_transform(X_train_gdf)

# GPU-to-CPU for plotting
X_proj_cpu = X_proj.to_numpy()
plt.figure(figsize=(6, 5))
fig, ax = plt.subplots(figsize=(6, 5))
# Now you can use NumPy slicing
scatter = ax.scatter(
    X_proj_cpu[:, 0], 
    X_proj_cpu[:, 1],
    c=y_train,   # Assuming y_train is a cupy array, .get() moves it to CPU
    s=4,
    cmap="viridis",
    alpha=0.8
)

cbar = fig.colorbar(scatter, ax=ax, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(CLASS_NAMES)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA Projection of SDSS Spectra (GPU Accelerated)")
save_plot("pca_projection.png")

# 1. Extract the PCA object from the pipeline
pca_step = pipeline_pca.named_steps["pca"]

# 2. Convert GPU attributes to CPU NumPy arrays
# cuML attributes are often cuDF DataFrames or cupy arrays
pca_mean = pca_step.mean_
if hasattr(pca_mean, "to_numpy"):
    pca_mean = pca_mean.to_numpy() # if it's a cuDF Series
elif hasattr(pca_mean, "get"):
    pca_mean = pca_mean.get()      # if it's a cupy array

pca_components = pca_step.components_
if hasattr(pca_components, "to_numpy"):
    pca_components = pca_components.to_numpy()
elif hasattr(pca_components, "get"):
    pca_components = pca_components.get()

pca_variance = pca_step.explained_variance_ratio_
if hasattr(pca_variance, "to_numpy"):
    pca_variance = pca_variance.to_numpy()
elif hasattr(pca_variance, "get"):
    pca_variance = pca_variance.get()

# 3. Plot using the CPU-converted arrays
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Right Plot: Mean and Eigen-spectra
axs[1].plot(wavelengths, pca_mean, label="Mean", color='black', alpha=0.5)
for i in range(2):
    # Now pca_components[i] will be a 1D NumPy array of length 3522
    axs[1].plot(wavelengths, pca_components[i], label=f"Component {i + 1}")

axs[1].set_xlabel('Wavelength (Å)')
axs[1].set_ylabel('Scaled flux')
axs[1].set_title('Mean Spectrum and Eigen-spectra')
axs[1].legend()

# Left Plot: Explained Variance
axs[0].bar(np.arange(1, 6), pca_variance[:5]) # Ensure we only plot the first 5
axs[0].set_xlabel("Principal Component")
axs[0].set_ylabel("Explained Variance Ratio")
save_plot("pca_eigenspectra.png")

# --- 4. Optimization Objective ---
def objective(trial, X, y, cv, model_type):

    # PCA configuration
    use_pca = trial.suggest_categorical("use_pca", [True, False])

    if use_pca:
        # cuML PCA does not reliably support sklearn-style
        # explained variance ratios such as 0.95
        # therefore we optimize integer component counts
        pca_n = trial.suggest_int("pca_n_components",10,1000)
        pca = PCA(n_components=pca_n,)
    else:
        pca = "passthrough"

    # Model selection

    if model_type == "logit":
        clf = LogisticRegression(
            C=trial.suggest_float(
                "logit_C",
                1e-1,
                1e4,
                log=True
            ),
            max_iter=10000,
            solver="qn"
        )

    elif model_type == "rf":

        clf = RandomForestClassifier(
            n_estimators=trial.suggest_int(
                "rf_n_estimators",
                100,
                400
            ),

            max_depth=trial.suggest_categorical(
                "rf_max_depth",
                [5, 10, 20, None]
            ),

            max_features=trial.suggest_categorical(
                "rf_max_features",
                ["sqrt", "log2"]
            ),

            min_samples_leaf=trial.suggest_categorical(
                "rf_min_samples_leaf",
                [1, 2, 5]
            ),

        )

    elif model_type == "xgb":

        clf = xgb.XGBClassifier(

            objective="multi:softprob",
            eval_metric="mlogloss",

            tree_method="hist",
            device="cuda",

            n_estimators=trial.suggest_int(
                "xgb_n_estimators",
                100,
                500
            ),

            learning_rate=trial.suggest_float(
                "xgb_learning_rate",
                1e-3,
                1e-1,
                log=True
            ),

            max_depth=trial.suggest_categorical(
                "xgb_max_depth",
                [3, 4, 5, 6]
            ),

            subsample=trial.suggest_categorical(
                "xgb_subsample",
                [0.6, 0.8, 1.0]
            ),

            colsample_bytree=trial.suggest_categorical(
                "xgb_colsample_bytree",
                [0.5, 0.8, 1.0]
            ),

            min_child_weight=trial.suggest_categorical(
                "xgb_min_child_weight",
                [1, 3, 5]
            )
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Pipeline

    steps = [("normalize", Normalizer(norm="l2"))]

    if pca != "passthrough":
        steps.append(("pca", pca))

    steps.append(("classifier", clf))

    pipe = Pipeline(steps)

    # Cross-validation

    scores = []

    # sklearn splitter operates on CPU arrays
    for train_idx, val_idx in cv.split(X.to_pandas(), y.to_pandas()):

        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        # fit
        pipe.fit(X_train_fold, y_train_fold)
        # predict
        preds = pipe.predict(X_val_fold)
        # move to CPU for sklearn metrics
        # Convert ground truth safely
        y_true = y_val_fold.to_numpy()
        # Convert predictions safely
        if hasattr(preds, "to_numpy"):
            y_pred = preds.to_numpy()
        elif isinstance(preds, cp.ndarray):
            y_pred = cp.asnumpy(preds)
        else:
            y_pred = np.asarray(preds)
        
        score = f1_score(y_true, y_pred, average="macro")
        scores.append(score)

    return float(np.mean(scores))

# --- 5. Hyperparameter Search ---
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

all_results = {}

best_overall_score = -np.inf
best_overall_model = None
best_model_name = None

for model_name in ["logit", "rf", "xgb"]:

    print("\n" + "=" * 60)
    print(f"Optimizing {model_name}")
    print("=" * 60)

    study = optuna.create_study(
        direction="maximize"
    )

    start = time.time()

    study.optimize(
        lambda trial: objective(
            trial,
            X_train_gdf,
            y_train_gdf,
            cv,
            model_name
        ),
        n_trials=20
    )

    elapsed = time.time() - start

    print("\nBest CV F1:")
    print(f"{study.best_value:.4f}")

    print("\nBest Parameters:")
    print(study.best_params)

    all_results[model_name] = {
        "best_score": study.best_value,
        "best_params": study.best_params,
        "fit_time_sec": elapsed
    }

    # track best overall model
    if study.best_value > best_overall_score:

        best_overall_score = study.best_value
        best_model_name = model_name

# --- 6. Final Training & Persistence ---

best_params = all_results[best_model_name]["best_params"]

print("\nBest overall model:")
print(best_model_name)
steps = [("normalize", Normalizer(norm="l2"))]
# PCA
if best_params.get("use_pca", False):
    steps.append((
        "pca", PCA(n_components=best_params["pca_n_components"],)
    ))

# Classifier
if best_model_name == "logit":

    clf = LogisticRegression(
        C=best_params["logit_C"],
        max_iter=10000,
        solver="qn"
    )

elif best_model_name == "rf":

    clf = RandomForestClassifier(
        n_estimators=best_params["rf_n_estimators"],
        max_depth=best_params["rf_max_depth"],
        max_features=best_params["rf_max_features"],
        min_samples_leaf=best_params["rf_min_samples_leaf"],
    )

elif best_model_name == "xgb":

    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda",
        n_estimators=best_params["xgb_n_estimators"],
        learning_rate=best_params["xgb_learning_rate"],
        max_depth=best_params["xgb_max_depth"],
        subsample=best_params["xgb_subsample"],
        colsample_bytree=best_params["xgb_colsample_bytree"],
        min_child_weight=best_params["xgb_min_child_weight"]
    )

steps.append(("classifier", clf))

best_model = Pipeline(steps)
best_model.fit(X_train_gdf, y_train_gdf)

model_file = os.path.join(MODEL_DIR, "best_spectrum_classifier_gpu.pkl")
joblib.dump(best_model, model_file)
print(f"Model saved to {model_file}")

# --- 7. Evaluation ---
# We load the best refitted model
final_model = joblib.load(model_file)

test_predictions = final_model.predict(X_test_gdf)
# test_predictions is a <class 'numpy.ndarray'>
# move back to CPU
y_test_cpu = y_test_gdf.to_numpy()
pred_cpu = test_predictions 

test_accuracy = accuracy_score(y_test_cpu, pred_cpu)

test_f1 = f1_score(y_test_cpu, pred_cpu, average="macro")

print(f"Final Test Accuracy: {test_accuracy:.4f}")
print(f"Final Test F1 Macro: {test_f1:.4f}")

print(
    classification_report(
        y_test_cpu,
        pred_cpu,
        target_names=CLASS_NAMES
    )
)

report(y_test_cpu, y_pred_cpu, CLASS_NAMES,
       title="Final Test Confusion Matrix",
       filename="confusion_matrix_final.png")

print("\nPipeline Complete.")

