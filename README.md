## Abstract

Crohn's disease activity is currently monitored episodically and invasively. GutGlow is a
simulation and evaluation framework for estimating flare state continuously from four
wearable/self-reported telemetry streams — daily steps, skin temperature, bathroom visits and
sleep efficiency — using 3-day rolling means plus a four-signal consistency vote. Fecal calprotectin,
CRP and CDAI are simulated alongside these signals but are deliberately excluded from the model,
since predicting activity from the laboratory indices that define it would be circular.

The framework was evaluated on a synthetic longitudinal dataset of 50 subjects × 90 days
(4,500 subject-days, 60% of subjects with simulated flares) containing injected confounding
artefacts. All results below are from that simulator; **no human subjects, clinical records or
biomarker assays were used, so no ethics approval was required. Nothing here is a medical device
or a clinical recommendation.**

## Results

| Evaluation | AUC-ROC | Sensitivity | Specificity |
|---|---|---|---|
| Subject-day split (test subjects also appear in training) | 0.984 | 0.905 | 0.973 |
| **Held-out-subject split (5 seeds, mean ± SD)** | **0.971 ± 0.007** | 0.829 | 0.969 |

The first row is reported only to document the inflation; the second is the primary result.
Average precision on the held-out-subject split is 0.840.

The high absolute AUC reflects how sharply the generator separates flare from non-flare days, not
clinical difficulty. This is the main limitation of the study.

### Artefact behaviour

Against clean non-flare days (1.6% false positives), modelled febrile illness raises the false-positive
rate to 5.4% and sleep deprivation to 10.3%. Because the simulator perturbs the same channels in the
same direction as a flare, the current consistency score reduces but does not resolve this confusion.
Disambiguating artefacts from flares requires either an artefact label in the generator or a model
that consumes one, which is the next piece of work here.

## Limitations

- Fully synthetic data; the generator's effect sizes are my own assumptions, not literature values.
- 50 subjects, 90 days; no external validation, no real wearable data.
- The consistency score compares smoothed values against means computed over the whole dataset,
  including test rows — a minor leakage path that should be refit on training data only.

To reproduce
git clone https://github.com/ranshul156-crypto/GutGlow.git && cd GutGlow
python3 Gut_glow_code_ml.py     # prints AUC / sensitivity / specificity and writes ./gutglow_final_output/
- Symptom and biomarker channels are excluded from the model, so the task is sensor-only by design.
- Simulated flare episodes never start before day 10 or after day 74, which makes day index weakly
  predictive of the label.
