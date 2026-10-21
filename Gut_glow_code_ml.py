
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    average_precision_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("default")

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = "./gutglow_final_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SUBJECTS = 50
N_DAYS = 90
FLARE_PREVALENCE = 0.6

all_data = []

# ===============================
# DATA SIMULATION
# ===============================
for subj in range(N_SUBJECTS):
    baseline_steps = max(500, np.random.normal(8000, 2000))
    baseline_temp = np.random.normal(36.4, 0.2)
    baseline_flushes = np.random.poisson(3)
    baseline_sleep = np.random.uniform(75, 90)
    baseline_crp = np.random.lognormal(1.2, 0.6)

    has_flares = np.random.random() < FLARE_PREVALENCE
    flare_days = set()

    if has_flares:
        for _ in range(np.random.poisson(2)):
            start = np.random.randint(10, N_DAYS - 20)
            duration = np.random.randint(5, 12)
            flare_days.update(
                range(start, min(start + duration, N_DAYS))
            )

    for day in range(N_DAYS):
        is_flare = day in flare_days

        has_infection = np.random.random() < 0.05
        sleep_deprived = np.random.random() < 0.1
        high_activity = np.random.random() < 0.1

        if is_flare:
            steps = baseline_steps * np.random.uniform(0.5, 0.8)
            temp = baseline_temp + np.random.uniform(0.2, 0.6)
            flushes = baseline_flushes * np.random.uniform(1.8, 3.5)
            sleep = np.random.uniform(50, 75)
            crp = baseline_crp * np.random.uniform(2, 5)
            fcp = np.random.uniform(300, 1200)
            cdai = np.random.uniform(180, 300)
        else:
            steps = baseline_steps * np.random.uniform(0.8, 1.2)
            temp = baseline_temp + np.random.uniform(-0.2, 0.2)
            flushes = baseline_flushes * np.random.uniform(0.5, 1.5)
            sleep = np.random.uniform(75, 95)
            crp = baseline_crp * np.random.uniform(0.5, 1.5)
            fcp = np.random.uniform(20, 200)
            cdai = np.random.uniform(50, 180)

        if has_infection:
            temp += np.random.uniform(0.5, 1.2)
            crp *= np.random.uniform(1.5, 3.0)
            sleep -= np.random.uniform(10, 20)

        if sleep_deprived:
            sleep -= np.random.uniform(15, 25)

        if high_activity:
            steps *= np.random.uniform(1.3, 1.7)

        steps = max(0, steps + np.random.normal(0, 400))
        temp += np.random.normal(0, 0.05)
        flushes = max(0, flushes + np.random.normal(0, 0.4))
        sleep = np.clip(sleep + np.random.normal(0, 4), 0, 100)

        all_data.append({
            "subject_id": subj,
            "day": day,
            "is_flare": int(is_flare),
            "daily_steps": steps,
            "skin_temp_c": temp,
            "bathroom_visits": flushes,
            "sleep_efficiency_pct": sleep,
            "crp": crp,
            "fcp": fcp,
            "cdai": cdai,
        })

df = pd.DataFrame(all_data)

# ===============================
# SUBJECT-WISE ROLLING FEATURES
# ===============================
df = df.sort_values(["subject_id", "day"]).reset_index(drop=True)

feature_columns = [
    "daily_steps",
    "skin_temp_c",
    "bathroom_visits",
    "sleep_efficiency_pct",
]

# Trailing rolling average: uses current and previous days only.
for col in feature_columns:
    df[f"{col}_rolling"] = (
        df.groupby("subject_id")[col]
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )

rolling_columns = [f"{col}_rolling" for col in feature_columns]

# ===============================
# SUBJECT-LEVEL TRAIN/TEST SPLIT
# ===============================
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=SEED,
)

train_idx, test_idx = next(
    splitter.split(df, df["is_flare"], groups=df["subject_id"])
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

# ===============================
# CONSISTENCY SCORE
# Thresholds are derived from training data only.
# ===============================
steps_low = train_df["daily_steps_rolling"].quantile(0.25)
sleep_low = train_df["sleep_efficiency_pct_rolling"].quantile(0.25)
temp_high = train_df["skin_temp_c_rolling"].quantile(0.75)
visits_high = train_df["bathroom_visits_rolling"].quantile(0.75)


def add_consistency_score(data):
    data = data.copy()

    data["consistency_score"] = (
        (data["daily_steps_rolling"] < steps_low).astype(int)
        + (data["sleep_efficiency_pct_rolling"] < sleep_low).astype(int)
        + (data["skin_temp_c_rolling"] > temp_high).astype(int)
        + (data["bathroom_visits_rolling"] > visits_high).astype(int)
    )

    return data


train_df = add_consistency_score(train_df)
test_df = add_consistency_score(test_df)

model_features = rolling_columns + ["consistency_score"]

X_train = train_df[model_features]
y_train = train_df["is_flare"]

X_test = test_df[model_features]
y_test = test_df["is_flare"]

# ===============================
# SCALE USING TRAINING DATA ONLY
# Random Forest does not require scaling, but this
# preserves a leakage-safe scaling pipeline.
# ===============================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# MODEL TRAINING
# ===============================
model = RandomForestClassifier(
    n_estimators=120,
    max_depth=6,
    random_state=SEED,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train_scaled, y_train)

# ===============================
# EVALUATION
# ===============================
y_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

roc_auc = auc(*roc_curve(y_test, y_prob)[:2])
pr_auc = average_precision_score(y_test, y_prob)

tn, fp, fn, tp = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
).ravel()

sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
specificity = tn / (tn + fp) if (tn + fp) else 0.0

print("\n===== GUTGLOW RESULTS (SUBJECT-HELD-OUT) =====")
print(f"Train subjects: {train_df['subject_id'].nunique()}")
print(f"Test subjects:  {test_df['subject_id'].nunique()}")
print(f"Train rows:     {len(train_df)}")
print(f"Test rows:      {len(test_df)}")
print(f"ROC-AUC:        {roc_auc:.3f}")
print(f"PR-AUC:         {pr_auc:.3f}")
print(f"Sensitivity:    {sensitivity:.3f}")
print(f"Specificity:    {specificity:.3f}")
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
)

# ===============================
# SAVE PREDICTIONS AND METRICS
# ===============================
predictions = test_df[
    ["subject_id", "day", "is_flare", "crp", "fcp", "cdai"]
].copy()

predictions["predicted_flare"] = y_pred
predictions["flare_probability"] = y_prob

predictions.to_csv(
    os.path.join(OUTPUT_DIR, "held_out_subject_predictions.csv"),
    index=False,
)

metrics_df = pd.DataFrame([{
    "train_subjects": train_df["subject_id"].nunique(),
    "test_subjects": test_df["subject_id"].nunique(),
    "train_rows": len(train_df),
    "test_rows": len(test_df),
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "sensitivity": sensitivity,
    "specificity": specificity,
    "true_negative": tn,
    "false_positive": fp,
    "false_negative": fn,
    "true_positive": tp,
}])

metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "metrics.csv"),
    index=False,
)

df.to_csv(
    os.path.join(OUTPUT_DIR, "simulated_gutglow_data.csv"),
    index=False,
)

# ===============================
# ROC CURVE
# ===============================
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("GutGlow ROC Curve — Held-Out Subjects")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "roc_curve.png"),
    dpi=200,
)
plt.close()

# ===============================
# CRP VS FCP
# ===============================
plt.figure(figsize=(7, 5))
plt.scatter(df["crp"], df["fcp"], alpha=0.5)
plt.xlabel("Simulated CRP")
plt.ylabel("Simulated FCP")
plt.title("Simulated CRP vs FCP")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "crp_fcp_scatter.png"),
    dpi=200,
)
plt.close()

# ===============================
# STEPS VS CDAI
# ===============================
plt.figure(figsize=(7, 5))
plt.scatter(df["daily_steps"], df["cdai"], alpha=0.5)
plt.xlabel("Simulated Daily Steps")
plt.ylabel("Simulated CDAI")
plt.title("Simulated Daily Steps vs CDAI")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "steps_cdai_scatter.png"),
    dpi=200,
)
plt.close()

# ===============================
# SLEEP DISTRIBUTION
# ===============================
plt.figure(figsize=(7, 5))
plt.hist(df["sleep_efficiency_pct"], bins=25)
plt.xlabel("Simulated Sleep Efficiency (%)")
plt.ylabel("Number of Subject-Days")
plt.title("Simulated Sleep Efficiency Distribution")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "sleep_efficiency_histogram.png"),
    dpi=200,
)
plt.close()

# ===============================
# DESCRIPTIVE CORRELATIONS
# These use the full simulated dataset and are
# descriptive, not held-out predictive evaluations.
# ===============================
correlations = [
    ("Sleep efficiency vs CDAI", "sleep_efficiency_pct", "cdai"),
    ("Bathroom visits vs FCP", "bathroom_visits", "fcp"),
    ("Daily steps vs CDAI", "daily_steps", "cdai"),
    ("Skin temperature vs CRP", "skin_temp_c", "crp"),
]

print("\n===== DESCRIPTIVE CORRELATIONS (SIMULATED DATA) =====")

correlation_results = []

for label, x_col, y_col in correlations:
    rho, p_value = spearmanr(df[x_col], df[y_col])

    print(f"{label}: rho={rho:.3f}, p={p_value:.3e}")

    correlation_results.append({
        "comparison": label,
        "spearman_rho": rho,
        "p_value": p_value,
    })

pd.DataFrame(correlation_results).to_csv(
    os.path.join(OUTPUT_DIR, "descriptive_correlations.csv"),
    index=False,
)

print(f"\nAll outputs saved in: {os.path.abspath(OUTPUT_DIR)}")
print("Files generated:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(f" - {filename}")
