# Sequential Recommendation / Offline RL Evaluation Sources

Research notes prepared 2026-09-13 for a strict chronological split and offline-policy-evaluation protocol.

## Primary sources

- Ji, Sun, Zhang & Li, *A Critical Study on Data Leakage in Recommender System Offline Evaluation* (2020): https://arxiv.org/abs/2010.11060
  - Per-user leave-last-one-out can expose globally future interactions and affect ranking conclusions; a global timeline split avoids that leakage.
- Klenitskiy & Vasilev, *Time to Split: Exploring Data Splitting Strategies for Offline Evaluation of Sequential Recommenders* (RecSys 2025): https://doi.org/10.1145/3705328.3748164
  - Compares splitting strategies and reports temporal leakage risks for common leave-one-out methods.
- Jiang & Li, *Doubly Robust Off-policy Value Evaluation for Reinforcement Learning* (ICML 2016): https://proceedings.mlr.press/v48/jiang16.html
  - Primary doubly-robust OPE formulation for evaluation from data collected by a different policy.
- Le Paine et al., *Hyperparameter Selection for Offline Reinforcement Learning* (2020): https://arxiv.org/abs/2007.09055
  - Offline policy and hyperparameter selection must not rely on execution in the target environment.
- Tang & Wiens, *Model Selection for Offline Reinforcement Learning: Practical Considerations for Healthcare Settings* (MLHC 2021): https://proceedings.mlr.press/v149/tang21a.html
  - Provides a train/validation OPE model-selection pipeline.
- Gülçehre et al., *RL Unplugged: A Suite of Benchmarks for Offline Reinforcement Learning* (NeurIPS 2020): https://arxiv.org/abs/2006.13888
  - Fixed logged datasets with reproducible evaluation protocols.
- Fu et al., *D4RL: Datasets for Deep Data-Driven Reinforcement Learning* (2020): https://arxiv.org/abs/2004.07219
  - Reference for fixed offline datasets and evaluation protocol design.
- Ie et al., *RecSim: A Configurable Simulation Platform for Recommender Systems* (2019): https://arxiv.org/abs/1909.04847
  - Defines configurable sequential-recommendation simulation; simulator findings require sim-to-real caution.
- Shi et al., *Virtual-Taobao: Virtualizing Real-world Online Retail Environment for Reinforcement Learning* (2018): https://arxiv.org/abs/1805.10000
  - A data-driven simulator using GAN-SD and imitation learning, with online validation beyond simulation.

## Protocol synthesis for the V7 Agent

Use global chronological train/validation/test windows, with a complete session or trajectory as the atomic split unit. Construct features and candidate sets only from information available at the decision timestamp. Place a purge gap at least as long as the attribution horizon, and seal the final test set.

Offline policy evaluation requires logged behavior-policy probabilities (propensities) and action support. If they are missing, treat any causal claim about a different intervention policy as a limitation. Fit a GAN or user simulator only on the training split, calibrate it on validation, and never make final performance claims from the same simulator used for policy training.
