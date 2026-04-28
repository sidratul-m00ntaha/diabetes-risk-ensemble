# =============================================================
#  DIABETES RISK CLASSIFICATION — COMPLETE ML PIPELINE
#  Target: diabetes_risk_category (Low Risk / Prediabetes / High Risk)
#  Excludes: Patient_ID (identifier), diabetes_risk_score (data leakage)
# =============================================================

# ── 0. IMPORTS ────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    RandomizedSearchCV, GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, roc_curve,
    accuracy_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Run: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Run: pip install shap")

# ── 1. LOAD DATA ───────────────────────────────────────────────
print("=" * 65)
print("  DIABETES RISK CLASSIFICATION — ML PIPELINE")
print("=" * 65)

# ── AUTO-DETECT ENVIRONMENT & SET PATHS ──────────────────────
if os.path.exists("/kaggle/working"):
    OUTPUT_DIR = "/kaggle/working"
    DATA_PATH  = "/kaggle/input/diabetes-dataset/diabetes_risk_dataset.csv"
    print("  Environment: Kaggle detected")
else:
    OUTPUT_DIR = "."
    DATA_PATH  = "diabetes_dataset.csv"
    print("  Environment: Local/Colab detected")

try:
    df = pd.read_csv(DATA_PATH)
    print(f"\n✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print(f"\n⚠ File '{DATA_PATH}' not found.")
    print("  Generating synthetic sample from the 20 provided rows...")

    # Synthetic demo data matching your schema
    np.random.seed(42)
    n = 300
    ages      = np.random.randint(18, 85, n)
    genders   = np.random.choice(["Male", "Female"], n)
    bmis      = np.round(np.random.uniform(18, 50, n), 1)
    bp        = np.random.randint(100, 200, n)
    fg        = np.random.randint(70, 200, n)
    insulin   = np.round(np.random.uniform(2, 45, n), 1)
    hba1c     = np.round(np.random.uniform(4.5, 9.0, n), 1)
    chol      = np.random.randint(150, 300, n)
    trig      = np.random.randint(60, 280, n)
    pa        = np.random.choice(["Low", "Moderate", "High"], n)
    cals      = np.random.randint(1500, 3500, n)
    sugar     = np.round(np.random.uniform(10, 200, n), 1)
    sleep     = np.round(np.random.uniform(4, 10, n), 1)
    stress    = np.random.randint(1, 11, n)
    fam_hist  = np.random.choice(["Yes", "No"], n)
    waist     = np.round(np.random.uniform(55, 155, n), 1)

    # Simple rule-based risk for synthetic labels
    risk_score = (
        (fg > 126).astype(int) * 3 +
        (hba1c > 6.5).astype(int) * 3 +
        (bmis > 30).astype(int) * 2 +
        (fam_hist == "Yes").astype(int) * 1 +
        (pa == "Low").astype(int) * 1
    )
    cats = np.where(risk_score >= 5, "High Risk",
           np.where(risk_score >= 2, "Prediabetes", "Low Risk"))

    df = pd.DataFrame({
        "Patient_ID": [f"{i:05d}" for i in range(1, n+1)],
        "age": ages, "gender": genders, "bmi": bmis,
        "blood_pressure": bp, "fasting_glucose_level": fg,
        "insulin_level": insulin, "HbA1c_level": hba1c,
        "cholesterol_level": chol, "triglycerides_level": trig,
        "physical_activity_level": pa, "daily_calorie_intake": cals,
        "sugar_intake_grams_per_day": sugar, "sleep_hours": sleep,
        "stress_level": stress, "family_history_diabetes": fam_hist,
        "waist_circumference_cm": waist,
        "diabetes_risk_score": np.random.uniform(0, 100, n).round(1),
        "diabetes_risk_category": cats
    })
    print(f"  Synthetic dataset created: {df.shape[0]} rows")


# ── 2. BASIC EDA ───────────────────────────────────────────────
print("\n── 2. EXPLORATORY DATA ANALYSIS ──────────────────────────")
print(f"Shape          : {df.shape}")
print(f"Missing values :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nTarget distribution:")
print(df["diabetes_risk_category"].value_counts())
print(f"\nClass balance (%):")
print((df["diabetes_risk_category"].value_counts(normalize=True) * 100).round(1))


# ── 3. FEATURE ENGINEERING & PREPROCESSING ────────────────────
print("\n── 3. FEATURE ENGINEERING ────────────────────────────────")

# Drop leakage columns and identifier
DROP_COLS = ["Patient_ID", "diabetes_risk_score"]
TARGET    = "diabetes_risk_category"

df_model = df.drop(columns=DROP_COLS, errors="ignore")

# ── 3a. Useful ratio features (backed by literature) ──
df_model["bmi_age_ratio"]       = df_model["bmi"] / df_model["age"]
df_model["glucose_insulin_ratio"] = (
    df_model["fasting_glucose_level"] / (df_model["insulin_level"] + 1e-5)
)
df_model["waist_bmi_ratio"]     = (
    df_model["waist_circumference_cm"] / df_model["bmi"]
)
df_model["calorie_activity"]    = df_model["daily_calorie_intake"] / (
    df_model["physical_activity_level"].map({"Low": 1, "Moderate": 2, "High": 3})
)
print("  ✓ Ratio features added: bmi_age_ratio, glucose_insulin_ratio,")
print("                          waist_bmi_ratio, calorie_activity")

# ── 3b. Identify feature types ──
X = df_model.drop(columns=[TARGET])
y = df_model[TARGET]

NUMERIC_FEATURES = X.select_dtypes(include=[np.number]).columns.tolist()
CATEGORICAL_FEATURES = X.select_dtypes(include=["object"]).columns.tolist()
print(f"\n  Numeric features    ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")
print(f"  Categorical features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")

# ── 3c. Encode target ──
CLASS_ORDER = ["Low Risk", "Prediabetes", "High Risk"]
label_enc   = LabelEncoder()
label_enc.classes_ = np.array(CLASS_ORDER)
y_enc = pd.Categorical(y, categories=CLASS_ORDER).codes
print(f"\n  Target encoding: {dict(zip(CLASS_ORDER, range(3)))}")


# ── 4. TRAIN / TEST SPLIT (stratified) ────────────────────────
print("\n── 4. TRAIN/TEST SPLIT ───────────────────────────────────")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"  Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print(f"  Train class dist: {np.bincount(y_train)}")
print(f"  Test  class dist: {np.bincount(y_test)}")


# ── 5. PREPROCESSING PIPELINE ─────────────────────────────────
print("\n── 5. PREPROCESSING PIPELINE ─────────────────────────────")

num_pipeline = Pipeline([
    ("imputer",    SimpleImputer(strategy="median")),
    ("scaler",     StandardScaler()),
])

cat_pipeline = Pipeline([
    ("imputer",  SimpleImputer(strategy="most_frequent")),
    ("encoder",  OrdinalEncoder(
        categories=[
            ["Low", "Moderate", "High"],   # physical_activity_level
            ["No", "Yes"],                  # family_history_diabetes
            ["Male", "Female"],             # gender
        ],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, NUMERIC_FEATURES),
    ("cat", cat_pipeline, CATEGORICAL_FEATURES),
])
print("  ✓ Numeric  → median imputation + StandardScaler")
print("  ✓ Categorical → most-frequent imputation + OrdinalEncoder")
print("    (physical_activity: Low=0, Moderate=1, High=2)")
print("    (family_history: No=0, Yes=1 | gender: Male=0, Female=1)")


# ── 6. MODEL DEFINITIONS ──────────────────────────────────────
print("\n── 6. DEFINING MODELS ────────────────────────────────────")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, solver="lbfgs"
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes":         GaussianNB(),
    "Decision Tree":       DecisionTreeClassifier(
        max_depth=8, random_state=42
    ),
    "SVM (RBF)":           SVC(
        kernel="rbf", probability=True, random_state=42
    ),
    "Random Forest":       RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting":   GradientBoostingClassifier(
        n_estimators=200, random_state=42
    ),
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), max_iter=500,
        random_state=42, early_stopping=True
    ),
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, use_label_encoder=False,
        eval_metric="mlogloss", random_state=42, n_jobs=-1
    )

print(f"  ✓ {len(models)} models registered")


# ── 7. CROSS-VALIDATION BENCHMARK ─────────────────────────────
print("\n── 7. CROSS-VALIDATION BENCHMARK (10-fold, stratified) ───")

CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])
    scores = cross_val_score(
        pipe, X_train, y_train,
        cv=CV, scoring="f1_weighted", n_jobs=-1
    )
    cv_results[name] = scores
    print(f"  {name:<25} F1={scores.mean():.4f} ± {scores.std():.4f}")

cv_df = pd.DataFrame({
    name: scores for name, scores in cv_results.items()
})


# ── 8. TRAIN ALL MODELS ON FULL TRAIN SET ─────────────────────
print("\n── 8. TRAINING ALL MODELS ────────────────────────────────")

trained_pipes = {}
test_results  = []

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)
    trained_pipes[name] = pipe

    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1w  = f1_score(y_test, y_pred, average="weighted")
    auc  = roc_auc_score(
        y_test, y_proba,
        multi_class="ovr", average="weighted"
    )
    test_results.append({
        "Model": name, "Accuracy": acc,
        "F1 Weighted": f1w, "AUC-ROC (OvR)": auc,
        "CV F1 Mean": cv_results[name].mean(),
        "CV F1 Std":  cv_results[name].std(),
    })
    print(f"  {name:<25} Acc={acc:.4f} | F1={f1w:.4f} | AUC={auc:.4f}")

results_df = pd.DataFrame(test_results).sort_values(
    "AUC-ROC (OvR)", ascending=False
).reset_index(drop=True)

print(f"\n  Best model by AUC: {results_df.iloc[0]['Model']}")


# ── 9. ENSEMBLE: VOTING CLASSIFIER (top 3 by AUC) ─────────────
print("\n── 9. ENSEMBLE — SOFT VOTING (top 3 by AUC) ──────────────")

top3_names  = results_df.head(3)["Model"].tolist()
top3_models = [(n, trained_pipes[n]["clf"]) for n in top3_names]
print(f"  Ensemble members: {top3_names}")

voting_clf = Pipeline([
    ("prep", preprocessor),
    ("clf",  VotingClassifier(estimators=top3_models, voting="soft"))
])
voting_clf.fit(X_train, y_train)

y_pred_v  = voting_clf.predict(X_test)
y_proba_v = voting_clf.predict_proba(X_test)
acc_v  = accuracy_score(y_test, y_pred_v)
f1_v   = f1_score(y_test, y_pred_v, average="weighted")
auc_v  = roc_auc_score(y_test, y_proba_v, multi_class="ovr", average="weighted")

print(f"  Voting Ensemble → Acc={acc_v:.4f} | F1={f1_v:.4f} | AUC={auc_v:.4f}")

ensemble_row = {
    "Model": "Voting Ensemble (top 3)",
    "Accuracy": acc_v, "F1 Weighted": f1_v,
    "AUC-ROC (OvR)": auc_v, "CV F1 Mean": np.nan, "CV F1 Std": np.nan
}
results_df = pd.concat(
    [results_df, pd.DataFrame([ensemble_row])], ignore_index=True
).sort_values("AUC-ROC (OvR)", ascending=False).reset_index(drop=True)


# ── 10. STACKING CLASSIFIER ────────────────────────────────────
print("\n── 10. ENSEMBLE — STACKING ────────────────────────────────")

base_estimators = [
    ("rf",  RandomForestClassifier(n_estimators=100, random_state=42)),
    ("gb",  GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ("svm", SVC(probability=True, random_state=42)),
]
if XGBOOST_AVAILABLE:
    base_estimators.append(
        ("xgb", XGBClassifier(
            n_estimators=100, use_label_encoder=False,
            eval_metric="mlogloss", random_state=42
        ))
    )

stacking_clf = Pipeline([
    ("prep", preprocessor),
    ("clf",  StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, passthrough=False
    ))
])
stacking_clf.fit(X_train, y_train)

y_pred_s  = stacking_clf.predict(X_test)
y_proba_s = stacking_clf.predict_proba(X_test)
acc_s  = accuracy_score(y_test, y_pred_s)
f1_s   = f1_score(y_test, y_pred_s, average="weighted")
auc_s  = roc_auc_score(y_test, y_proba_s, multi_class="ovr", average="weighted")
print(f"  Stacking Ensemble → Acc={acc_s:.4f} | F1={f1_s:.4f} | AUC={auc_s:.4f}")

stack_row = {
    "Model": "Stacking Ensemble",
    "Accuracy": acc_s, "F1 Weighted": f1_s,
    "AUC-ROC (OvR)": auc_s, "CV F1 Mean": np.nan, "CV F1 Std": np.nan
}
results_df = pd.concat(
    [results_df, pd.DataFrame([stack_row])], ignore_index=True
).sort_values("AUC-ROC (OvR)", ascending=False).reset_index(drop=True)


# ── 11. HYPERPARAMETER TUNING — BEST MODEL ────────────────────
print("\n── 11. HYPERPARAMETER TUNING (RandomizedSearchCV) ─────────")

BEST_MODEL_NAME = results_df.iloc[0]["Model"]
print(f"  Tuning: {BEST_MODEL_NAME}")

if "Random Forest" in BEST_MODEL_NAME or BEST_MODEL_NAME == "Random Forest":
    param_dist = {
        "clf__n_estimators":      [100, 200, 300, 500],
        "clf__max_features":      ["sqrt", "log2", None],
        "clf__max_depth":         [None, 10, 20, 30],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf":  [1, 2, 4],
    }
    tune_model = RandomForestClassifier(random_state=42, n_jobs=-1)

elif "XGBoost" in BEST_MODEL_NAME and XGBOOST_AVAILABLE:
    param_dist = {
        "clf__n_estimators":  [100, 200, 300],
        "clf__max_depth":     [3, 5, 7],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__subsample":     [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 1.0],
    }
    tune_model = XGBClassifier(
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1
    )

elif "Gradient Boosting" in BEST_MODEL_NAME:
    param_dist = {
        "clf__n_estimators":  [100, 200, 300],
        "clf__max_depth":     [3, 5, 7],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__subsample":     [0.7, 0.8, 1.0],
    }
    tune_model = GradientBoostingClassifier(random_state=42)
else:
    param_dist = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth":    [None, 10, 20],
    }
    tune_model = RandomForestClassifier(random_state=42, n_jobs=-1)

tune_pipe = Pipeline([("prep", preprocessor), ("clf", tune_model)])
rnd_search = RandomizedSearchCV(
    tune_pipe, param_dist,
    n_iter=20, cv=5,
    scoring="f1_weighted",
    random_state=42, n_jobs=-1, verbose=0
)
rnd_search.fit(X_train, y_train)

print(f"  Best params: {rnd_search.best_params_}")
print(f"  Best CV F1:  {rnd_search.best_score_:.4f}")

y_pred_tuned  = rnd_search.predict(X_test)
y_proba_tuned = rnd_search.predict_proba(X_test)
acc_t  = accuracy_score(y_test, y_pred_tuned)
f1_t   = f1_score(y_test, y_pred_tuned, average="weighted")
auc_t  = roc_auc_score(y_test, y_proba_tuned, multi_class="ovr", average="weighted")
print(f"  Tuned Test → Acc={acc_t:.4f} | F1={f1_t:.4f} | AUC={auc_t:.4f}")

tuned_row = {
    "Model": f"{BEST_MODEL_NAME} (Tuned)",
    "Accuracy": acc_t, "F1 Weighted": f1_t,
    "AUC-ROC (OvR)": auc_t,
    "CV F1 Mean": rnd_search.best_score_, "CV F1 Std": np.nan
}
results_df = pd.concat(
    [results_df, pd.DataFrame([tuned_row])], ignore_index=True
).sort_values("AUC-ROC (OvR)", ascending=False).reset_index(drop=True)


# ── 12. FINAL MODEL SELECTION ─────────────────────────────────
print("\n── 12. FINAL MODEL SELECTION ──────────────────────────────")
print(results_df[["Model","Accuracy","F1 Weighted","AUC-ROC (OvR)",
                   "CV F1 Mean","CV F1 Std"]].to_string(index=False))

FINAL_MODEL_NAME = results_df.iloc[0]["Model"]
print(f"\n  ★ FINAL MODEL: {FINAL_MODEL_NAME}")

# Select the final pipeline
if "Tuned" in FINAL_MODEL_NAME:
    final_model = rnd_search.best_estimator_
elif "Stacking" in FINAL_MODEL_NAME:
    final_model = stacking_clf
elif "Voting" in FINAL_MODEL_NAME:
    final_model = voting_clf
else:
    final_model = trained_pipes[FINAL_MODEL_NAME]

y_pred_final  = final_model.predict(X_test)
y_proba_final = final_model.predict_proba(X_test)


# ── 13. DETAILED EVALUATION — FINAL MODEL ────────────────────
print("\n── 13. DETAILED EVALUATION ────────────────────────────────")
print("\n  Classification Report:")
print(classification_report(
    y_test, y_pred_final,
    target_names=CLASS_ORDER
))


# ── 14. FEATURE IMPORTANCE (RF / XGB) ────────────────────────
print("\n── 14. FEATURE IMPORTANCE ─────────────────────────────────")

# Use best single RF or XGB for feature importance
best_single = results_df[
    ~results_df["Model"].str.contains("Ensemble|Tuned", na=False)
].iloc[0]["Model"]

fi_pipe = trained_pipes.get(
    best_single,
    trained_pipes.get("Random Forest",
    trained_pipes.get("Gradient Boosting", None))
)

if fi_pipe is not None:
    clf_step = fi_pipe["clf"]
    if hasattr(clf_step, "feature_importances_"):
        prep_step = fi_pipe["prep"]
        # Reconstruct feature names after ColumnTransformer
        num_names = NUMERIC_FEATURES
        cat_names = CATEGORICAL_FEATURES
        all_names = num_names + cat_names
        importances = clf_step.feature_importances_
        fi_series = pd.Series(importances, index=all_names[:len(importances)])
        fi_series = fi_series.sort_values(ascending=False)
        print("  Top 10 features:")
        print(fi_series.head(10).to_string())
    else:
        fi_series = None
        print("  Feature importance not available for this model type.")
else:
    fi_series = None


# ── 15. VISUALIZATIONS ────────────────────────────────────────
print("\n── 15. GENERATING VISUALIZATIONS ─────────────────────────")

COLORS    = ["#2563EB", "#10B981", "#F59E0B", "#EF4444",
             "#8B5CF6", "#EC4899", "#14B8A6", "#F97316", "#6366F1"]
CLASS_CLR = ["#10B981", "#F59E0B", "#EF4444"]

fig = plt.figure(figsize=(22, 28))
fig.patch.set_facecolor("#0F172A")
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Model AUC comparison ──
ax1 = fig.add_subplot(gs[0, :2])
plot_df = results_df.dropna(subset=["AUC-ROC (OvR)"]).copy()
bars = ax1.barh(plot_df["Model"], plot_df["AUC-ROC (OvR)"],
                color=COLORS[:len(plot_df)], edgecolor="none", height=0.65)
ax1.set_xlim(0.5, 1.02)
ax1.axvline(x=0.9, color="white", linewidth=0.8, linestyle="--", alpha=0.4)
for bar, val in zip(bars, plot_df["AUC-ROC (OvR)"]):
    ax1.text(val + 0.003, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", ha="left", color="white",
             fontsize=9, fontweight="bold")
ax1.set_title("AUC-ROC (One-vs-Rest) — All Models",
              color="white", fontsize=13, pad=10, fontweight="bold")
ax1.set_facecolor("#1E293B")
ax1.tick_params(colors="white", labelsize=9)
ax1.spines[:].set_visible(False)
ax1.xaxis.label.set_color("white")

# ── Plot 2: Target distribution ──
ax2 = fig.add_subplot(gs[0, 2])
counts = pd.Series(y_test).map(dict(enumerate(CLASS_ORDER))).value_counts()
wedges, texts, autotexts = ax2.pie(
    counts, labels=counts.index, autopct="%1.1f%%",
    colors=CLASS_CLR, startangle=90,
    wedgeprops=dict(edgecolor="#0F172A", linewidth=2)
)
for t in texts + autotexts:
    t.set_color("white"); t.set_fontsize(9)
ax2.set_title("Test Set Class Distribution",
              color="white", fontsize=11, fontweight="bold")
ax2.set_facecolor("#1E293B")

# ── Plot 3: Confusion Matrix ──
ax3 = fig.add_subplot(gs[1, 0])
cm  = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
            ax=ax3, linewidths=0.5, linecolor="#0F172A",
            cbar_kws={"shrink": 0.8})
ax3.set_title(f"Confusion Matrix\n{FINAL_MODEL_NAME}",
              color="white", fontsize=10, fontweight="bold")
ax3.set_xlabel("Predicted", color="white", fontsize=9)
ax3.set_ylabel("Actual",    color="white", fontsize=9)
ax3.tick_params(colors="white", labelsize=8)
ax3.set_facecolor("#1E293B")

# ── Plot 4: CV F1 Box Plot ──
ax4 = fig.add_subplot(gs[1, 1])
cv_plot_data = [cv_results[m] for m in cv_results]
cv_plot_names = list(cv_results.keys())
bp_dict = ax4.boxplot(cv_plot_data, patch_artist=True, notch=False,
                      medianprops=dict(color="white", linewidth=2))
for patch, color in zip(bp_dict["boxes"], COLORS[:len(cv_plot_data)]):
    patch.set_facecolor(color); patch.set_alpha(0.8)
for element in ["whiskers","caps","fliers"]:
    for item in bp_dict[element]:
        item.set_color("#94A3B8")
ax4.set_xticks(range(1, len(cv_plot_names)+1))
ax4.set_xticklabels(
    [n.replace(" ", "\n") for n in cv_plot_names],
    fontsize=6.5, color="white"
)
ax4.set_title("10-Fold CV F1 Score Distribution",
              color="white", fontsize=11, fontweight="bold")
ax4.set_ylabel("F1 Score", color="white", fontsize=9)
ax4.set_facecolor("#1E293B")
ax4.tick_params(colors="white")
ax4.spines[:].set_color("#334155")

# ── Plot 5: ROC Curve (OvR per class) ──
ax5 = fig.add_subplot(gs[1, 2])
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
for i, (cls, clr) in enumerate(zip(CLASS_ORDER, CLASS_CLR)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_final[:, i])
    auc_cls = roc_auc_score(y_test_bin[:, i], y_proba_final[:, i])
    ax5.plot(fpr, tpr, color=clr, lw=2, label=f"{cls} (AUC={auc_cls:.3f})")
ax5.plot([0,1],[0,1], "w--", lw=0.8, alpha=0.4)
ax5.set_title("ROC Curves (One-vs-Rest)\nFinal Model",
              color="white", fontsize=10, fontweight="bold")
ax5.set_xlabel("False Positive Rate", color="white", fontsize=8)
ax5.set_ylabel("True Positive Rate",  color="white", fontsize=8)
ax5.legend(fontsize=7.5, labelcolor="white",
           facecolor="#1E293B", edgecolor="#334155")
ax5.set_facecolor("#1E293B")
ax5.tick_params(colors="white")
ax5.spines[:].set_color("#334155")

# ── Plot 6: Feature Importance ──
ax6 = fig.add_subplot(gs[2, :2])
if fi_series is not None:
    top_fi = fi_series.head(15)
    bars6  = ax6.barh(top_fi.index[::-1], top_fi.values[::-1],
                      color=COLORS[0], edgecolor="none", height=0.7)
    for bar, val in zip(bars6, top_fi.values[::-1]):
        ax6.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", color="white", fontsize=8)
    ax6.set_title("Top 15 Feature Importances",
                  color="white", fontsize=12, fontweight="bold")
    ax6.set_facecolor("#1E293B")
    ax6.tick_params(colors="white", labelsize=9)
    ax6.spines[:].set_visible(False)
else:
    ax6.text(0.5, 0.5, "Feature importance\nnot available",
             ha="center", va="center", color="white", fontsize=12)
    ax6.set_facecolor("#1E293B")

# ── Plot 7: Accuracy vs F1 Scatter ──
ax7 = fig.add_subplot(gs[2, 2])
plot_df2 = results_df.dropna(subset=["Accuracy","F1 Weighted"]).copy()
for i, row in plot_df2.iterrows():
    ax7.scatter(row["Accuracy"], row["F1 Weighted"],
                color=COLORS[i % len(COLORS)], s=120, zorder=3)
    ax7.annotate(row["Model"].replace(" ", "\n"),
                 (row["Accuracy"], row["F1 Weighted"]),
                 textcoords="offset points", xytext=(5, 3),
                 fontsize=6, color="white")
ax7.set_xlabel("Accuracy",   color="white", fontsize=9)
ax7.set_ylabel("F1 Weighted", color="white", fontsize=9)
ax7.set_title("Accuracy vs F1 Score",
              color="white", fontsize=11, fontweight="bold")
ax7.set_facecolor("#1E293B")
ax7.tick_params(colors="white")
ax7.spines[:].set_color("#334155")

# ── Plot 8: Summary Table ──
ax8 = fig.add_subplot(gs[3, :])
ax8.set_facecolor("#1E293B")
ax8.axis("off")
tbl_data = results_df[
    ["Model","Accuracy","F1 Weighted","AUC-ROC (OvR)","CV F1 Mean"]
].copy()
tbl_data["Accuracy"]     = tbl_data["Accuracy"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
tbl_data["F1 Weighted"]  = tbl_data["F1 Weighted"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
tbl_data["AUC-ROC (OvR)"]= tbl_data["AUC-ROC (OvR)"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
tbl_data["CV F1 Mean"]   = tbl_data["CV F1 Mean"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")

table = ax8.table(
    cellText=tbl_data.values,
    colLabels=tbl_data.columns,
    cellLoc="center", loc="center",
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2563EB")
        cell.set_text_props(color="white", fontweight="bold")
    elif row == 1:
        cell.set_facecolor("#10B981")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#1E3A5F")
        cell.set_text_props(color="white")
    else:
        cell.set_facecolor("#1E293B")
        cell.set_text_props(color="#CBD5E1")
    cell.set_edgecolor("#334155")

ax8.set_title("Complete Model Comparison Summary  (★ Top = Best Model)",
              color="white", fontsize=12, fontweight="bold", pad=10)

fig.suptitle("Diabetes Risk Classification — ML Model Evaluation Dashboard",
             color="white", fontsize=16, fontweight="bold", y=0.98)

plt.savefig(os.path.join(OUTPUT_DIR, "diabetes_model_results.png"),
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("  ✓ Dashboard saved: diabetes_model_results.png")
plt.close()


# ── 16. SAVE RESULTS CSV ──────────────────────────────────────
results_df.to_csv(os.path.join(OUTPUT_DIR, "diabetes_model_comparison.csv"), index=False)
print("  ✓ Results CSV saved: diabetes_model_comparison.csv")


# ── 17. SAVE FINAL MODEL ─────────────────────────────────────
import joblib
joblib.dump(final_model, os.path.join(OUTPUT_DIR, "diabetes_best_model.pkl"))
print("  ✓ Model saved: diabetes_best_model.pkl")


# ── 18. USAGE GUIDE ───────────────────────────────────────────
print("""
── HOW TO USE THE SAVED MODEL ─────────────────────────────────
  import joblib, pandas as pd
  model = joblib.load("diabetes_best_model.pkl")

  # New patient row (no Patient_ID, no diabetes_risk_score)
  new_patient = pd.DataFrame([{
      "age": 55, "gender": "Female", "bmi": 32.0,
      "blood_pressure": 145, "fasting_glucose_level": 115,
      "insulin_level": 12.0, "HbA1c_level": 6.1,
      "cholesterol_level": 220, "triglycerides_level": 180,
      "physical_activity_level": "Low", "daily_calorie_intake": 2400,
      "sugar_intake_grams_per_day": 85.0, "sleep_hours": 6.5,
      "stress_level": 7, "family_history_diabetes": "Yes",
      "waist_circumference_cm": 98.0,
      # engineered features
      "bmi_age_ratio": 32.0/55,
      "glucose_insulin_ratio": 115/12.0,
      "waist_bmi_ratio": 98.0/32.0,
      "calorie_activity": 2400/1
  }])
  pred  = model.predict(new_patient)
  proba = model.predict_proba(new_patient)
  CLASS_ORDER = ["Low Risk", "Prediabetes", "High Risk"]
  print("Prediction :", CLASS_ORDER[pred[0]])
  print("Probability:", dict(zip(CLASS_ORDER, proba[0].round(3))))
""")

print("=" * 65)
print("  PIPELINE COMPLETE")
print("=" * 65)
