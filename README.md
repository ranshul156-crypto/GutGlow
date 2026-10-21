# GutGlow: An Artefact-Aware Simulation Framework for Continuous Crohn's Disease Activity Estimation

## Abstract
Crohn's disease activity is monitored episodically and invasively. GutGlow is a simulation and
evaluation framework that estimates flare state continuously from four telemetry streams — daily
steps, skin temperature, bathroom visits and sleep efficiency — using 3-day trailing rolling means
plus a four-signal consistency vote. Fecal calprotectin, CRP and CDAI are simulated alongside these
signals but deliberately withheld from the model: predicting activity from the indices that define
it would be circular. No human subjects, clinical records or biomarker assays were used; nothing
here is a medical device or a clinical recommendation.

## Results
Held-out-subject evaluation (`GroupShuffleSplit` on `subject_id`, 30% of the 50 simulated patients
never seen in training):

| ROC-AUC | PR-AUC | Sensitivity | Specificity | Test set |
|---|---|---|---|---|
| 0.980 | 0.862 | 0.767 (66/86 flare-days) | 0.984 (20 FP / 1,264) | 15 subjects, 1,350 subject-days |

Reproduce with `python3 Gut_glow_code_ml.py`; `metrics.csv` is that run's output. Sensitivity is
reported rather than the headline AUC, because the 14.4% positive rate of the held-out split is what
a deployed alarm would live or die on. The high absolute AUC reflects how sharply the generator
separates flare from non-flare days, not clinical difficulty — that is the study's main limitation.

## Artefact behaviour
Against clean non-flare days (1.6% false positives), modelled febrile illness raises the rate to
5.4% and sleep deprivation to 10.3%. The simulator perturbs the same channels in the same direction
as a flare, so the consistency vote reduces but does not resolve the confusion. Separating artefact
from flare needs a generator that labels them differently, or a model trained to consume that label.

## Limitations
- Fully synthetic; effect sizes are my assumptions, not literature values. 50 subjects, 90 days, no
  external validation and no real wearable data.
- One held-out-subject draw; multi-seed means with confidence intervals are the next step.
- `bathroom_visits` is self-reported, so the input set is wearable **plus** patient-reported, not
  purely sensor-derived.
- Simulated flare episodes never start before day 10 or after day 74, making day index weakly
  predictive of the label.
- `simulated_gutglow_data.csv` is the pre-engineering table; rolling and consistency features are
  rebuilt inside the script so that all thresholds derive from training rows only.

## Reproduction
git clone https://github.com/ranshul156-crypto/GutGlow.git
cd GutGlow && python3 Gut_glow_code_ml.py   # writes ./gutglow_final_output/
