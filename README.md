# In Silico Trial Simulation with Artificial Intelligence-Generated Synthetic Control Cohorts Reproduces Results of a Randomized Controlled Trial in Acute Myeloid Leukemia

**Codebase for generating synthetic AML patients**


# Overview
The framework focuses on generating synthetic patient-level data while preserving clinically relevant relationships and survival characteristics. It also provides a comprehensive evaluation suite covering:

- Synthetic data generation using TabPFN generative modeling
- Risk-score based patient selection using Cox Proportional Hazards modeling
- Privacy assessment
- Statistical fidelity
- Survival analysis preservation

The repository consists of independent Python modules implementing individual metrics and workflows, along with a notebook demonstrating the complete implementation pipeline.

---

# Example Workflow

1. Train TabPFN on the original dataset.
2. Generate synthetic patient records.
3. Apply risk score matching.
   - Generate Cox Proportional Hazards model
   - Patient specific risk-score calculation
   - Select Synthetic patients with similar risk-score as real patient
4. Evaluate fidelity:
   - Stastical Similarity
   - Synthcity Metrics
   - TabSynDex Metrics
5. Evaluate survival utility:
   - Log-Rank Test
   - KM Divergence
   - Optimism
   - Shortsightedness
6. Evaluate privacy:
   - Exact Match
   - Distance to Closest Record (DCR)
   - Nearest Neighbour Distance Ratio (NNDR)
   - Membership inference attack (MIA)

---

# Intended Use

This repository is intended for:

- Synthetic clinical data generation
- Survival analysis research
- Benchmarking synthetic data quality

---

# Notebook

The notebook implementation.ipynb demonstrates the complete workflow from synthetic data generation through privacy, fidelity, and survival evaluation.

---

# References
- TabPFN (https://github.com/PriorLabs/TabPFN)
- TabSynDex (https://github.com/vikram2000b/tabsyndex)
- Synthcity  (https://github.com/vanderschaarlab/synthcity)
