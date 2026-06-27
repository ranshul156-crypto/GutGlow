import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

OUTPUT_DIR = './gutglow_final_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SUBJECTS = 50
N_DAYS = 90
FLARE_PREVALENCE = 0.6

all_data = []

# ===============================
# DATA SIMULATION
# ===============================
for subj in range(N_SUBJECTS):
    baseline_steps = np.random.normal(8000, 2000)
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
            flare_days.update(range(start, min(start + duration, N_DAYS)))

    for day in range(N_DAYS):
        is_flare = day in flare_days

        # Artefacts
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

        # Artefact injection
        if has_infection:
            temp += np.random.uniform(0.5, 1.2)
            crp *= np.random.uniform(1.5, 3.0)
            sleep -= np.random.uniform(10, 20)

        if sleep_deprived:
            sleep -= np.random.uniform(15, 25)

        if high_activity:
            steps *= np.random.uniform(1.3, 1.7)

        # Noise
        steps = max(0, steps + np.random.normal(0, 400))
        temp += np.random.normal(0, 0.05)
        flushes = max(0, flushes + np.random.normal(0, 0.4))
        sleep = np.clip(sleep + np.random.normal(0, 4), 0, 100)

        all_data.append({
            'subject_id': subj,
            'day': day,
            'is_flare': is_flare,
            'daily_steps': steps,
            'skin_temp_c': temp,
            'bathroom_visits': flushes,
            'sleep_efficiency_pct': sleep,
            'crp_mg_l': crp,
            'fcp_ug_g': fcp,
            'cdai_score': cdai
        })

df = pd.DataFrame(all_data)

# ===============================
# SMOOTHING
# ===============================
for col in ['daily_steps', 'skin_temp_c', 'bathroom_visits', 'sleep_efficiency_pct']:
    df[col + '_smooth'] = df.groupby('subject_id')[col].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

# ===============================
# CONSISTENCY SCORE
# ===============================
df['consistency_score'] = (
    (df['skin_temp_c_smooth'] > df['skin_temp_c_smooth'].mean()).astype(int) +
    (df['bathroom_visits_smooth'] > df['bathroom_visits_smooth'].mean()).astype(int) +
    (df['sleep_efficiency_pct_smooth'] < df['sleep_efficiency_pct_smooth'].mean()).astype(int) +
    (df['daily_steps_smooth'] < df['daily_steps_smooth'].mean()).astype(int)
)

df['flare'] = df['is_flare'].astype(int)

# ===============================
# CORRELATIONS (True Spearman Rank)
# ===============================
pairs = {
    'Sleep efficiency vs. CDAI': ('sleep_efficiency_pct', 'cdai_score'),
    'Bowel frequency vs. FCP': ('bathroom_visits', 'fcp_ug_g'),
    'Daily steps vs. CDAI': ('daily_steps', 'cdai_score'),
    'Skin temperature vs. CRP': ('skin_temp_c', 'crp_mg_l')
}

corr_results = []
for pair_name, (col1, col2) in pairs.items():
    rho, p_val = spearmanr(df[col1], df[col2])
    corr_results.append({
        'biomarker_pair': pair_name,
        'spearman_rho': round(rho, 3),
        'p_value': f"{p_val:.2e}"
    })

corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(f'{OUTPUT_DIR}/correlations.csv', index=False)

# ===============================
# MODEL
# ===============================
features = [
    'daily_steps_smooth',
    'skin_temp_c_smooth',
    'bathroom_visits_smooth',
    'sleep_efficiency_pct_smooth',
    'consistency_score'
]

X = df[features]
y = df['flare']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=120, max_depth=6, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# ===============================
# GRAPHS
# ===============================

# 1. ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(f'{OUTPUT_DIR}/roc_curve.png')
plt.close()

# 2. CRP vs FCP
plt.figure()
plt.scatter(df['crp_mg_l'], df['fcp_ug_g'], alpha=0.4)
plt.xlabel("CRP")
plt.ylabel("Fecal Calprotectin")
plt.title("CRP vs FCP")
plt.savefig(f'{OUTPUT_DIR}/crp_fcp.png')
plt.close()

# 3. Steps vs CDAI
plt.figure()
plt.scatter(df['daily_steps'], df['cdai_score'], alpha=0.4)
plt.xlabel("Daily Steps")
plt.ylabel("CDAI Score")
plt.title("Steps vs CDAI")
plt.savefig(f'{OUTPUT_DIR}/steps_cdai.png')
plt.close()

# 4. Sleep Distribution
plt.figure()
df[df['flare'] == 1]['sleep_efficiency_pct'].hist(alpha=0.5, label='Flare')
df[df['flare'] == 0]['sleep_efficiency_pct'].hist(alpha=0.5, label='No Flare')
plt.legend()
plt.title("Sleep Distribution")
plt.savefig(f'{OUTPUT_DIR}/sleep_distribution.png')
plt.close()

# ===============================
# SAVE DATA
# ===============================
df.to_csv(f'{OUTPUT_DIR}/dataset.csv', index=False)

# ===============================
# OUTPUT
# ===============================
print("\n===== FINAL RESULTS =====")
print(f"AUC: {roc_auc:.3f}")
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
print("\nCorrelations:")
print(corr_df)
print(f"\nAll outputs saved in: {OUTPUT_DIR}")
