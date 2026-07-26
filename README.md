# Hard Drive Failure Predictor

## Quick Start

**[Try the deployed app ](https://hard-drive-failure-predictor-pdastyxwwkr3pjd9ugch2g.streamlit.app/)**

[Download sample CSV](app/test_smart_data.csv)

Upload a CSV of SMART telemetry (multiple days per drive) and get failure-risk
predictions instantly, with a downloadable results table. 

**[Full analysis notebook](notebook/hard-drive-failure-predictor.ipynb)** — data
cleaning, corruption discovery/fix, model tuning, SHAP explainability.

---

Predictive maintenance system built on real telemetry data from Backblaze's data centers.
The goal is to predict whether a hard drive will fail within the next 30 days using daily 
SMART diagnostic attributes.

## Problem Statement

Hard drive failures in data centers cause data loss and costly downtime. Backblaze publishes
daily SMART telemetry for every drive in their fleet. Using this data, we can build a model
that flags at-risk drives before they fail — giving engineers time to act.

## Dataset

- **Source:** Backblaze Hard Drive Test Data (2016), via Kaggle
- **Size:** 3.1 million rows, 95 raw columns
- **Failures:** 215 out of 3.1 million drive-days (0.006%)
- **Features:** Daily SMART diagnostic attributes per drive

## Data Preprocessing

### Column Reduction
Dropped 63 columns:
- 36 columns with >90% null values — entirely unreported attributes
- 27 normalized SMART columns — Backblaze research confirms raw values
  are more predictive than manufacturer-normalized equivalents

Final shape after dropping: 3.1M rows × 32 columns

### Null Handling
Remaining nulls were filled with 0 rather than median or mean.

SMART attributes are counters and flags that increment only when a fault occurs.
A null value indicates the attribute was never triggered by that drive — 
functionally identical to 0. Filling with median would incorrectly imply 
the drive had an average fault count.

### dtype Correction
12 SMART columns were incorrectly read as object type due to mixed string/float 
values in the raw CSV. Converted to float64 using pd.to_numeric with errors='coerce', 
then filled resulting nulls with 0.

### Duplicate Removal
120,497 duplicate drive-date combinations were found and removed, keeping the first 
occurrence. These were true duplicates with identical SMART values — a data collection 
artifact in the raw dataset.

### Corrupted Column
`capacity_bytes` was dropped entirely. All values were corrupted near-zero floats 
(e.g. 1.48e-311) rather than real capacity values (expected ~1e12 for TB-range drives).

## Evaluation Strategy

Accuracy is not a suitable metric for this dataset. A model predicting 
'no failure' for every row achieves 99.91% accuracy while missing every 
failure entirely.

Instead we optimize for **Recall** — minimizing missed failures — because:
- Missing a failure = data loss, downtime, costly emergency replacement
- False alarm = unnecessary but cheap drive inspection

Primary metric: **Precision-Recall AUC**
Secondary metric: **Recall at fixed Precision threshold**

## Feature Engineering

Three categories of features were engineered from raw SMART attributes:

### Drive Age
Days since the drive's first appearance in the dataset, calculated per drive 
using groupby + transform. Captures the bathtub curve effect — older drives 
are more failure-prone.

### Delta Features
Day-over-day change in Backblaze's top 5 failure-predictive SMART attributes:
smart_5, smart_187, smart_188, smart_197, smart_198. 
#### smart_5: Reallocated sectors count
Tracks number of bad sectors that have been replaced by spare sectors.

#### smart_187: Reported uncorrectable errors
Counts uncorrectable read errors using hardware ECC. If this value > 0, then the drive is scheduled for proactive replacement.

#### smart_188: Command Timeout
Counts number of aborted operations because the hard drive failed to respond to a system command within its alloted time budget

#### smart_197: Current pending sector count
Drives use this to flag sectors that are unstable and waiting for a successful read or write.

#### smart_198: Offline uncorrectable sector count
Counts errors when attempting to read/write a sector, often indicating physical degradation.

A sudden spike in reallocated sectors is more alarming than a stable high value.
First row per drive filled with 0 (no previous reading available).

### Rolling Averages (7-day)
7-day rolling mean of the same 5 SMART attributes. Smooths daily noise and 
captures sustained trends rather than transient spikes. min_periods=1 ensures 
drives with fewer than 7 days of history are still included.

## Handling Class Imbalance

Class weighting was chosen over SMOTE and undersampling because:
- Undersampling would discard 99.9% of majority class data
- SMOTE generates synthetic rows that don't represent realistic 
  drive readings in a time-series context
- Class weighting penalizes misclassification of failures during 
  training without modifying the dataset

sklearn's class_weight='balanced' assigns weight ~554x higher to 
failure rows, reflecting the 1:554 class ratio in the dataset.

## Corrupted SMART Columns (dtype Reinterpretation Bug)

During evaluation, three model families (Logistic Regression, Random Forest, XGBoost)
all failed to rank true failures above healthy drives — a symptom of missing feature
signal rather than a modeling or threshold issue. Investigating the raw SMART columns
surfaced a second, more subtle instance of the corruption pattern already seen in
`capacity_bytes`.

**Symptom:** `smart_7`, `smart_188`, `smart_240`, `smart_241`, and `smart_242` showed
`std == 0.0` alongside mean values on the order of `1e-314` to `1e-316` — subnormal
floats at the edge of float64 precision, functionally indistinguishable from `0.0` to
any model or statistical test.

**Root cause:** These values are the result of integer bytes being bit-reinterpreted
as float64 (a `.view()`-style cast) rather than properly converted (`.astype()`),
upstream in the source file — before this dataset was ever loaded. Confirmed via
reversal: `column.values.view(np.int64)` on affected rows recovers plausible integer
SMART readings (e.g. `4.940656e-324` → `1`, matching the smallest possible subnormal
double reinterpreted from the integer `1`).

**Why this broke the models:** Unlike ordinary outlier-heavy SMART attributes (e.g.
`smart_197`, `smart_198`, which are legitimately zero-inflated with rare large spikes),
these columns collapsed *every* value — healthy and failed drives alike — into the same
near-zero neighborhood. This erased real variance rather than representing it, making
the affected columns statistically inert regardless of algorithm.

**Fix:**
- Detected affected columns programmatically via the signature `std == 0.0` and
  `0 < |mean| < np.finfo(np.float64).tiny`.
- Recovered true integer values via NaN-safe bit reinterpretation: non-null rows are
  passed through `.view(np.int64)`; true nulls (sensor never reported) are left as
  missing rather than zero-filled, to avoid re-introducing the same signal-flattening
  problem via a different mechanism.
- Rebuilt `smart_188_raw_delta` and `smart_188_raw_roll_avg`, since these engineered
  features were originally computed on corrupted values and inherited `std == 0.0`
  themselves.
- `smart_7`, `smart_240`, `smart_241`, `smart_242` were not included in delta/rolling
  feature engineering (outside Backblaze's top-5 predictive attributes used for that
  step), so recovery is limited to the raw columns for those four.


## Results: Before vs. After Corruption Fix

| Model | PR-AUC (Before) | PR-AUC (After) | Recall @ 0.5 (Before) | Recall @ 0.5 (After) |
|---|---|---|---|---|
| Logistic Regression | 0.0079 | 0.0077 | 0.07 | 0.07 |
| Random Forest | 0.0140] | 0.0106 | 0% | 0% |
| XGBoost | 0.0042 | 0.0061 | 0% | 0.01 |

The fix produced a real but modest improvement in ranking ability (XGBoost PR-AUC
+45%), confirming the corrupted columns were suppressing genuine signal — but XGBoost
still underperforms the simpler Logistic Regression baseline, indicating the dtype
bug was one contributing factor, not the sole cause of weak recall. Hyperparameters
(`max_depth`, `learning_rate` for XGBoost; `n_estimators` for RF) remain at defaults
and are the next lever to pull.

### Threshold Sweep (Random Forest, Post-Fix)

Default `.predict()` uses a 0.5 cutoff, which is uninformative at this level of class
imbalance (~0.09% positive rate) — RF flags zero drives as failures at 0.5 despite a
PR-AUC of 0.0106, the best of the three models. Lowering the threshold surfaces the
real precision-recall tradeoff:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.01 | 0.02 | 0.19 | 0.03 |
| 0.05 | 0.09 | 0.04 | 0.06 |
| 0.10 | 0.09 | 0.02 | 0.03 |

Given the project's stated priority — a missed failure (data loss, downtime) is far
costlier than a false alarm (a cheap inspection) — the 0.01 threshold, trading precision
for substantially higher recall, is the more defensible operating point for this use case.

**Status:** Corruption fixed and verified. Threshold tuning applied to Random Forest;
XGBoost threshold sweep pending. Next: hyperparameter tuning on Random Forest and
XGBoost, since PR-AUC gains from the data fix alone did not close the gap to a
production-usable recall level.

## Choosing an Operating Threshold

PR-AUC measures ranking quality across all thresholds, but deploying a model requires
picking one specific cutoff — and the "best" recall number in isolation can be
misleading. At threshold 0.1, the tuned XGBoost model achieves 75% recall, but flags
1,132,009 of 1,850,717 test rows (61% of the fleet) as at-risk to catch 1,474 true
failures. Inspecting 61% of a data center's fleet is not operationally different from
inspecting nearly everything — the model would provide little practical triage value
despite its high recall figure.

| Threshold | Recall | Drives Flagged | % of Fleet | True Failures Caught |
|---|---|---|---|---|
| 0.1 | 75.2% | 1,132,009 | 61.2% | 1,474 |
| 0.2 | 57.8% | 456,758 | 24.7% | 1,133 |
| 0.3 | 44.8% | 228,915 | 12.4% | 878 |
| 0.4 | 33.7% | 106,198 | 5.7% | 660 |
| 0.5 | 22.9% | 43,921 | 2.4% | 449 |

Recommended operating point: **[0.3 or 0.4]**, trading some recall for an inspection
volume a data center engineering team could realistically act on. The right threshold
ultimately depends on the real-world cost ratio between one inspection and one missed
failure — a business decision this model informs but does not make unilaterally.

## Final Model Configuration

Hyperparameters were tuned using `RandomizedSearchCV` with `TimeSeriesSplit` (5 folds)
rather than standard K-Fold, since random shuffling would let rows from the same drive
leak across train/validation splits given the temporal structure of the data. Search
was scored on `average_precision` (PR-AUC) rather than accuracy or F1 — PR-AUC is
threshold-independent, avoiding the distortion that comes from evaluating candidates
at a fixed 0.5 cutoff on data this imbalanced (~0.09% positive rate).

**Best parameters found (20 iterations, 3-8 minute runtime):**

| Parameter | Value | Rationale |
|---|---|---|
| `learning_rate` | 0.01 | Small, careful updates per tree — a high default learning rate overcorrects toward the dominant majority pattern before the model can refine around the rare failure signal |
| `n_estimators` | 200 | More trees compensate for the smaller learning rate, giving the model enough iterations to converge |
| `max_depth` | 5 | Close to XGBoost's default (6); tree depth was not the main lever for this dataset |
| `min_child_weight` | 1 | Allows splits on small leaf sizes, needed given how rare positive examples are |
| `scale_pos_weight` | 1512.89 | Computed directly from the training set's class ratio (negative/positive), matching the manually-derived value used in the original run |

**Result:** PR-AUC improved from 0.0061 (post-corruption-fix, default hyperparameters)
to 0.0237 on the held-out test set — a ~3.9x improvement, and the best PR-AUC across
all models and phases of this project.

**Operating threshold:** See "Choosing an Operating Threshold" above. Recommended
threshold of [0.3 or 0.4] balances recall against a realistically actionable
inspection volume, rather than optimizing recall in isolation.

## Final Model Selection

Both XGBoost and Random Forest were tuned via `RandomizedSearchCV` (`TimeSeriesSplit`,
scored on `average_precision`). Despite boosting's theoretical advantage of sequential
error-correction — expected to help more on a signal this rare — **tuned Random Forest
outperformed tuned XGBoost on every held-out metric**:

| Model | PR-AUC | ROC-AUC | Best Params |
|---|---|---|---|
| XGBoost (tuned) | 0.0237 | 0.6871 | learning_rate=0.01, n_estimators=200, max_depth=5 |
| **Random Forest (tuned)** | **0.0421** | **0.7633** | max_depth=3, n_estimators=100 |

This is a useful reminder that architecture-based intuition doesn't always hold under
extreme class imbalance — RF's shallow, averaged trees generalized better here than
XGBoost's sequential refinement, possibly because boosting's sequential corrections
had more opportunity to overfit to noise in the small number of positive examples,
while RF's independently-trained, heavily regularized (max_depth=3) trees were more
robust to that same sparsity.

**Final model: Random Forest**, operating threshold **0.5**.

| Threshold | Precision | Recall | Drives Flagged | % of Fleet | Failures Caught |
|---|---|---|---|---|---|
| 0.5 | 0.73% | 39.8% | 106,706 | 5.8% | 781 / 1,961 |

**Lift:** Targeting the top 1% highest-risk drives by predicted probability catches
34.17x more failures than random selection (670 failures in the top 1% of the fleet).

## Drive Age at Failure: A Bathtub Curve

To validate `drive_age_days` as a predictive feature, the distribution of drive age at
the moment of failure was plotted against a fitted Weibull distribution.

![Drive Age at Failure vs Weibull Distribution](images/weibull_drive_age.png)

The fit itself is a poor match (shape ≈ 0.99, roughly flat/random-looking) — but the
underlying histogram tells a more specific story than a single Weibull curve can
capture: failures cluster in two distinct groups, early (0-25 days) and late
(90-120 days), with a quiet gap in between. This is consistent with a **bathtub
curve** pattern from reliability engineering — early failures reflecting
manufacturing defects or DOA units ("infant mortality"), and late failures
reflecting genuine wear-out from age, rather than one continuous failure process.

A single Weibull distribution can't represent two competing failure mechanisms at
once, which is exactly why the fit looks unconvincing — the flat shape parameter is
effectively averaging across two different regimes rather than describing either one
well. This bimodality further supports `drive_age_days` as a meaningful feature: age
alone doesn't predict failure risk uniformly, it predicts risk in two separate,
identifiable windows.

## SHAP: A Bathtub Curve?

The Weibull/age-histogram analysis (above) showed failures clustering at both young
(0-25 days) and old (90-120 days) drive ages. A natural next question: does
`drive_age_days` explain both clusters equally in the model itself?

The SHAP summary plot ranks `drive_age_days` among the top 6 most influential features
overall, and its dots show a striking pattern — young (blue) values dominate the
positive-SHAP side (pushing toward failure), while red (old) values lean toward the
negative side. Taken at face value, this looked contradictory: the actual failure data
skews toward *old* ages (median 99 days), yet the model's age feature seemed to favor
young ages when pushing predictions toward failure.

![SHAP summary plot](images/shap_summary.png)

This discrepancy was worth digging into rather than dismissing — see below.

## Digging Deeper: Decomposing the Bathtub Curve

A SHAP dependence plot for `drive_age_days` alone resolves the apparent contradiction.

![SHAP dependence: drive_age_days](images/shap_dependence_age.png)

`drive_age_days`'s own SHAP contribution is **positive** for young drives (up to +0.01)
but **negative** for old drives — the opposite sign from what a simple "age drives both
clusters" story would predict. Coloring the plot by `smart_197_raw` (Current Pending
Sector Count) explains why: within the old-age cluster, drives with high
`smart_197_raw` show the most negative age-SHAP values, while `smart_197_raw` itself
carries a strong positive SHAP contribution in the overall summary plot.

**Interpretation:** the model attributes old-age failure risk primarily to
`smart_197_raw`'s accumulated damage rather than to `drive_age_days` directly. Since
sector damage naturally accumulates over time, the two features are correlated — this
is a statement about how the model splits credit between correlated features, not a
definitive causal claim that age is irrelevant to old-age failures. It does suggest,
however, that the young-age and old-age failure clusters are driven by meaningfully
different mechanisms in the model: age itself for early-life failures, physical wear
indicators for late-life failures.


