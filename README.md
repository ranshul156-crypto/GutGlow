# GutGlow: An Artefact-Aware Multimodal Wearable Framework for Continuous Crohn’s Disease Activity Monitoring

This repository contains the simulation engine, machine learning pipeline, and analytical code for the **GutGlow** framework, designed to track continuous Crohn's Disease (CD) activity using multimodal wearable telemetry while remaining robust to external physiological artefacts.

## Abstract
Existing monitoring methods for Crohn's Disease (CD) are episodic and invasive. GutGlow utilizes multi-signal consistency scoring and temporal rolling-window smoothing within a Random Forest architecture to estimate flare states continuously. Tested on a synthetic 4,500 subject-day longitudinal dataset featuring simulated confounding artefacts (e.g., febrile illnesses, behavioral sleep deprivation), the framework achieves an AUC-ROC of 0.984, a sensitivity of 0.905, and a specificity of 0.973.

## Project Structure
- `Gut_glow_code_ml.py`: The complete executable script that handles data simulation, feature engineering (smoothing and consistency scoring), Random Forest training, evaluation, and plotting.
- `requirements.txt`: Specifies Python library dependencies.

## Installation & Environment Setup

To ensure computational reproducibility, follow these steps to set up a local environment:

1. Clone the repository:
git clone https://github.com/ranshul156-crypto/GutGlow.git
cd GutGlow
