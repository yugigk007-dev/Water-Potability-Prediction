import os
import joblib
import numpy as np
import pandas as pd

from scipy.stats import randint, loguniform, uniform

from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    RandomizedSearchCV,
    cross_validate,
    cross_val_predict
)
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


# ==========================================
# 1. LOAD PREPROCESSED DATA
# ==========================================

X_train = pd.read_csv("processed/X_train.csv")
X_test = pd.read_csv("processed/X_test.csv")

y_train = pd.read_csv("processed/y_train.csv").squeeze()
y_test = pd.read_csv("processed/y_test.csv").squeeze()

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

if X_train.isnull().any().any() or X_test.isnull().any().any():
    raise ValueError("Missing values found. Fix data.py and rerun it.")

# ==========================================
# 2. CROSS-VALIDATION SETUP
# ==========================================

search_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

repeated_cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)

# ==========================================
# 3. DEFINE MODELS AND SEARCH SPACES
# ==========================================

searches = {
    "Extra Trees": (
        ExtraTreesClassifier(
            random_state=42,
            n_jobs=1
        ),
        {
            "n_estimators": randint(200, 801),
            "max_depth": [None, 10, 15, 20, 30],
            "min_samples_split": randint(2, 16),
            "min_samples_leaf": randint(1, 7),
            "max_features": ["sqrt", "log2", None],
            "class_weight": [None, "balanced"]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(
            random_state=42,
            n_jobs=1
        ),
        {
            "n_estimators": randint(200, 801),
            "max_depth": [None, 10, 15, 20, 30],
            "min_samples_split": randint(2, 16),
            "min_samples_leaf": randint(1, 7),
            "max_features": ["sqrt", "log2", None],
            "class_weight": [None, "balanced", "balanced_subsample"]
        }
    ),

    "XGBoost": (
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1
        ),
        {
            "n_estimators": randint(200, 801),
            "max_depth": randint(2, 9),
            "learning_rate": loguniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "min_child_weight": randint(1, 11),
            "reg_lambda": loguniform(0.1, 20)
        }
    ),

    "CatBoost": (
        CatBoostClassifier(
            verbose=0,
            random_seed=42,
            allow_writing_files=False,
            thread_count=1
        ),
        {
            "iterations": randint(300, 1001),
            "depth": randint(4, 10),
            "learning_rate": loguniform(0.01, 0.3),
            "l2_leaf_reg": loguniform(1, 20)
        }
    ),

    "LightGBM": (
        LGBMClassifier(
            random_state=42,
            verbosity=-1,
            n_jobs=1
        ),
        {
            "n_estimators": randint(200, 801),
            "max_depth": [-1, 4, 6, 8, 10],
            "learning_rate": loguniform(0.01, 0.3),
            "num_leaves": randint(7, 51),
            "min_child_samples": randint(10, 51),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_lambda": loguniform(0.1, 20)
        }
    )
}

# ==========================================
# 4. INCLUDE YOUR ORIGINAL EXTRA TREES
# ==========================================

best_models = {
    "Extra Trees Baseline": ExtraTreesClassifier(
        n_estimators=661,
        max_depth=15,
        max_features=None,
        min_samples_leaf=1,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )
}

# ==========================================
# 5. HYPERPARAMETER TUNING
# ==========================================

tuning_scores = {}

for name, (model, params) in searches.items():
    print(f"\nTuning {name}...")

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=30,
        scoring="roc_auc",
        cv=search_cv,
        n_jobs=-1,
        random_state=42,
        verbose=1,
        refit=True
    )

    search.fit(X_train, y_train)

    best_models[name] = search.best_estimator_
    tuning_scores[name] = search.best_score_

    print(f"\n{name} best CV ROC-AUC:", round(search.best_score_, 4))
    print("Best parameters:", search.best_params_)

# ==========================================
# 6. REPEATED CROSS-VALIDATION COMPARISON
# ==========================================

print("\nREPEATED CROSS-VALIDATION")

cv_results = []

for name, model in best_models.items():
    print(f"\nEvaluating {name}...")

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=repeated_cv,
        scoring=["accuracy", "roc_auc"],
        n_jobs=1
    )

    result = {
        "Model": name,
        "CV Accuracy": scores["test_accuracy"].mean(),
        "Accuracy Std": scores["test_accuracy"].std(),
        "CV ROC-AUC": scores["test_roc_auc"].mean(),
        "ROC-AUC Std": scores["test_roc_auc"].std()
    }

    cv_results.append(result)

    print("Accuracy:", round(result["CV Accuracy"], 4),
          "+/-", round(result["Accuracy Std"], 4))
    print("ROC-AUC:", round(result["CV ROC-AUC"], 4),
          "+/-", round(result["ROC-AUC Std"], 4))

cv_df = pd.DataFrame(cv_results).sort_values(
    "CV ROC-AUC",
    ascending=False
)

print("\nCROSS-VALIDATION COMPARISON")
print(cv_df.to_string(index=False))

# ==========================================
# 7. THRESHOLD ANALYSIS USING OOF PREDICTIONS
# ==========================================

threshold_results = {}

for name, model in best_models.items():
    print(f"\nThreshold analysis: {name}")

    oof_prob = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=search_cv,
        method="predict_proba",
        n_jobs=1
    )[:, 1]

    rows = []

    for threshold in np.arange(0.30, 0.71, 0.05):
        pred = (oof_prob >= threshold).astype(int)
        cm = confusion_matrix(y_train, pred, labels=[0, 1])

        rows.append({
            "Threshold": round(threshold, 2),
            "Accuracy": accuracy_score(y_train, pred),
            "Potable Recall": recall_score(
                y_train, pred, pos_label=1, zero_division=0
            ),
            "False-safe": cm[0, 1],
            "False-unsafe": cm[1, 0]
        })

    threshold_df = pd.DataFrame(rows)
    threshold_results[name] = threshold_df

    print(threshold_df.to_string(index=False))

# ==========================================
# 8. TEST SET EVALUATION
# ==========================================

print("\nTEST SET EVALUATION")

test_results = []

for name, model in best_models.items():
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    # Keep the default threshold for the unbiased baseline
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    result = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Potable Precision": precision_score(
            y_test, y_pred, pos_label=1, zero_division=0
        ),
        "Potable Recall": recall_score(
            y_test, y_pred, pos_label=1, zero_division=0
        ),
        "Potable F1": f1_score(
            y_test, y_pred, pos_label=1, zero_division=0
        ),
        "False-safe": cm[0, 1],
        "Threshold": threshold
    }

    test_results.append(result)

    print(f"\n{name}")
    print(pd.Series(result).to_string())

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Non-potable", "Potable"],
        zero_division=0
    ))

test_df = pd.DataFrame(test_results).sort_values(
    "ROC-AUC",
    ascending=False
)

print("\nFINAL TEST COMPARISON")
print(test_df.to_string(index=False))

# ==========================================
# 9. SELECT AND SAVE FINAL MODEL
# ==========================================

best_name = cv_df.iloc[0]["Model"]
best_model = best_models[best_name]

# Refit the selected model on the complete training split
best_model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(0.5, "models/threshold.pkl")
joblib.dump(list(X_train.columns), "models/feature_names.pkl")

print("\nSelected model:", best_name)
print("Selection metric: repeated CV ROC-AUC")
print("Model and threshold saved successfully.")
#FINAL