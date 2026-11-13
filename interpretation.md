# Practical Session 4 — Overfitting, Data Leakage and Model Generalization

All numbers below come from the corrected notebook `tp4_corrected.ipynb` (seed 42, chronological split: first 960 rows for training, last 240 for testing).

## General introduction
The goal is to see how evaluation mistakes (leakage) and model complexity (overfitting) change the measured performance of regression models predicting `final_cgpa`. The dataset has 1,200 survey responses and 36 columns. The target is the midpoint of the declared CGPA band, so it only takes 4 values. All preprocessing (median imputation, scaling, encoding) is inside scikit-learn Pipelines, model selection uses `TimeSeriesSplit` on the training set, and the test set is used only in D.4.

---
# Part A — Data inspection

## A.1 Overview and missing values
1,200 rows, 36 columns, no exact duplicates. Thirteen columns have missing answers, between 2.3 % (`energy_level`) and 4.8 % (`events_participation`); among the model features: `sleep_hours` 4.6 %, `routine_rating` 3.4 %, `stress_level` 2.9 %, `daily_productivity` 2.8 %, `revision_frequency` 2.8 %. They are imputed with the **training** median inside the pipeline.

## A.2 Duplicates
0 duplicates with all columns, 0 without `student_id`, 0 without `student_id` and `timestamp`. Duplicates are therefore introduced artificially in B.2.

## A.3 Target and correlations
Target distribution: 5.95 → 236, 7.70 → 420, 8.95 → 362, 9.75 → 182 (no missing target).
Correlation with `final_cgpa`: strongest are `effort_index` (0.38), `attendance_rate` (0.37), `study_consistency` (0.32) and `study_hours` (0.30). Practically zero: `submission_day` (0.02), `stress_level` (0.02), `sleep_hours` (−0.01), `age` (−0.02). All correlations are moderate at best, which already suggests a modest R² later.

## A.4 `performance_risk_level` vs `cgpa_category`
Row proportions: 5.0–6.9 → 80 % High Risk, 0 % Low; 9.5–10.0 → 46 % Low, 2 % High. The label follows the CGPA band closely, so it carries information about the target. It is also an outcome-type label (we would not know it before knowing the performance). It cannot be a feature: it would leak the answer, the model would rely on it, and it would fail on new students. It is excluded.

## A.5 `cohort_code`
42 categories, largest 212 students, smallest 3. **12 categories have fewer than 10 students** (70 students in total). Small groups can be memorised, which is a source of overfitting.

## A.6 Temporal order
The file is **not** in chronological order. After sorting, the 7-day rolling mean stays between 7.91 and 8.14 (after the unstable first week), so there is no visible trend or drift in `final_cgpa`.

## Risks found
| Risk | Possible problem |
|---|---|
| `performance_risk_level` linked to the target | target leakage |
| Missing values | leakage if the median is computed on all data |
| High-cardinality `cohort_code` with 12 rare categories | overfitting by memorising small groups |
| Possible repeated submissions | same student in train and test |
| File not in time order, `submission_day` grows with time | random split mixes past and future; test values fall outside the training range |

---
# Part B — Leakage and evaluation strategy

## B.1 Random vs chronological split (unrestricted decision tree)
| Split | Train RMSE | Test RMSE | Baseline RMSE |
|---|---:|---:|---:|
| Random | ≈ 0 | 1.588 | 1.338 |
| Chronological | ≈ 0 | 1.953 | 1.236 |

Both trees memorise the training set (train RMSE ≈ 0) and are **worse than the baseline** on the test set: strong overfitting.
The chronological test error is clearly higher. The cause is `submission_day`: training days are 0–24 while test days are 24–30, so every test row is beyond the range the tree saw. Without this feature the gap almost disappears (chronological 1.610 vs random 1.548). The random split looks better because it lets the model see days around each test row; the chronological split is the honest estimate of predicting the future.
(Random-split scores move by about ±0.07 with the seed, so only the large difference with `submission_day` is meaningful.)

## B.2 Duplicate leakage (chronological split)
15 % of the rows (180) were submitted twice with a new random timestamp.

| Strategy | Features | Train RMSE | Test RMSE | Test rows also in train |
|---|---|---:|---:|---:|
| Split first, duplicates kept | with `submission_day` | ≈ 0 | 1.713 | 52 |
| Remove duplicates, then split | with `submission_day` | ≈ 0 | 1.691 | 0 |
| Split first, duplicates kept | without `submission_day` | 0.074 | **1.281** | 52 |
| Remove duplicates, then split | without `submission_day` | 0.079 | **1.579** | 0 |

A copy has a random timestamp, so the original can fall in the training part and the copy in the test part: 52 of the 276 test rows (19 %) have an identical twin in the training set. A model that memorises its training rows predicts these rows almost perfectly, so the test RMSE drops from 1.579 to 1.281 without any real improvement. With `submission_day` the effect is hidden (1.713 vs 1.691) because the copy has a different day value and the tree may route it elsewhere. Duplicates must be removed **before** splitting.

## B.3 Preprocessing leakage (KNN, k = 10)
| Approach | Test RMSE |
|---|---:|
| Median and scaler fitted on all data | 1.1496 |
| Pipeline (train only) | 1.1454 |

The gap is tiny because the features are discrete and the test set is a small part of the data. The principle still matters: statistics (scaling, median) must come from the training data only, otherwise test information enters the model and the evaluation is no longer an honest simulation of new data. The Pipeline guarantees this automatically.

---
# Part C — Overfitting

## C.1 Tree depth
| max_depth | Train RMSE | Test RMSE | Gap | CV validation RMSE |
|---|---:|---:|---:|---:|
| 1 | 1.179 | 1.159 | −0.020 | 1.180 |
| 2 | 1.139 | 1.127 | −0.013 | 1.162 |
| 3 | 1.102 | 1.068 | −0.034 | 1.161 |
| 5 | 1.010 | 1.290 | 0.280 | 1.242 |
| 10 | 0.531 | 1.875 | 1.344 | 1.586 |
| 20 | ≈ 0 | 1.953 | 1.953 | 1.573 |
| None | ≈ 0 | 1.953 | 1.953 | 1.573 |

Training error keeps falling while test and validation errors are lowest at depth 3 and then rise. **Overfitting starts at depth 5** (gap 0.28) and is severe from depth 10. Depths 20 and None are identical because the tree is already fully grown. A slightly negative gap for shallow trees means the test period is a little easier than the training period (the baseline shows the same).

## C.2 Polynomial degree of `effort_index` (5-fold TimeSeriesSplit on the training set)
| Degree | CV validation RMSE |
|---|---:|
| 1 | 1.177 |
| 2 | 1.178 |
| 3 | 1.179 |
| 5 | 1.185 |
| 9 | 1.188 |
| 15 | 1.188 |

Validation error rises slowly with the degree, so the simplest model is already the best. The effect (about 0.01) is far smaller than in C.1 (more than 0.4) because the model has only one input feature with few distinct values, so even degree 15 has little room to memorise individual students.

---
# Part D — Improving generalization

## D.1 `cohort_code` (tree, max_depth = 5)
| Approach | Train RMSE | Validation RMSE |
|---|---:|---:|
| Rare-category grouping | 0.871 | **1.221** |
| Feature removal | 0.879 | 1.242 |
| Target encoding | 0.964 | 1.260 |

**Selected: rare-category grouping.** The improvement over removing the feature is small (0.02), so `cohort_code` is at best weakly informative. Target encoding is worst, probably because it adds noise from small cohorts.

## D.2 `submission_day`
| Model | Validation RMSE with day | without day |
|---|---:|---:|
| Linear Regression | 1.126 | 1.118 |
| Decision Tree (depth 5) | 1.242 | 1.202 |
| Gradient Boosting | 1.129 | 1.127 |

**Decision: remove it.** Removing it lowers the validation RMSE for all three models; its correlation with the target is 0.02, and in a chronological split every test value lies outside the training range.

## D.3 Regularisation (degree-5 polynomial of `effort_index`)
Best validation RMSE: Linear 1.185, Ridge (α = 10) 1.178, Lasso (α = 0.01) 1.177.
Coefficients: Linear has max |coef| = 66.2; Ridge shrinks it to 0.39 (no coefficient is zero); Lasso sets **4 of 5** coefficients exactly to zero. Regularisation tames the unstable high-degree coefficients, but the validation gain is small (0.007) because this model hardly overfits.

## D.4 Final comparison (feature set: Part B features without `submission_day`, plus `cohort_code` with rare-category grouping)
| Model | Training RMSE | Testing RMSE | Gap | Testing R² |
|---|---:|---:|---:|---:|
| Baseline (training mean) | 1.269 | 1.236 | −0.032 | −0.003 |
| Linear Regression | 1.072 | 1.085 | 0.014 | 0.227 |
| Lasso (α = 0.1) | 1.106 | 1.072 | −0.034 | 0.246 |
| Decision Tree (depth 3) | 1.102 | 1.068 | −0.034 | 0.252 |
| Gradient Boosting (135 trees) | 0.955 | 1.059 | 0.104 | 0.264 |

---
# Final discussion
1. **Best generalisation:** gradient boosting has the lowest test RMSE (1.059, R² = 0.264), but the depth-3 tree (1.068) and Lasso (1.072) are within 0.013, so the differences are small. All beat the baseline by about 0.17.
2. **Largest gap:** among the final models, gradient boosting (+0.104). The unrestricted tree of B.1/C.1 is far worse (train ≈ 0, test 1.95).
3. **Most effective techniques:** limiting tree depth (1.95 → 1.07), removing `submission_day`, and early stopping for boosting. Ridge/Lasso shrank coefficients strongly (66 → 0.4) but changed validation RMSE by very little.
4. **Validation, not training:** training error rewards memorisation (an unlimited tree has RMSE ≈ 0 and is worse than the baseline); validation error shows how the model behaves on rows it did not fit.
5. **Untouched test set:** each decision made after looking at the test score lets test information into the model, so the final score becomes an optimistic estimate.
6. **Modest R²:** the answers are self-reported, coarse ranges and the target has only four values, so the features explain about a quarter of the variance. A high score is not always good: leakage (`performance_risk_level`, duplicates, preprocessing on all data) can inflate it without real predictive power.

# Overall conclusion
Evaluation choices change the measured performance as much as the model does. Duplicates and preprocessing on all data make scores look better than they are; a feature that grows with time (`submission_day`) breaks a chronological split; and an unrestricted tree memorises the training set and is worse than predicting the mean. Controlling complexity (depth, regularisation, early stopping) and choosing everything on validation data gives modest but honest results (test R² ≈ 0.25).
