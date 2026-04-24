# Electricity Demand Forecasting — Project README

**Model:** LightGBM Gradient Boosting Regressor  
**Target:** Next-hour electricity demand (MW) for Bangladesh (PGCB grid)  
**Horizon:** 1-step-ahead (t+1) hourly forecast  
**Evaluation Metric:** MAPE (Mean Absolute Percentage Error)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Sources](#2-data-sources)
3. [Data Preparation](#3-data-preparation)
   - 3.1 Half-Hourly Timestamp Handling
   - 3.2 Missing Value Imputation
   - 3.3 Anomaly Detection & Correction
4. [External Feature Integration](#4-external-feature-integration)
   - 4.1 Weather Features
   - 4.2 Economic Indicators
5. [Feature Engineering](#5-feature-engineering)
   - 5.1 Calendar Features
   - 5.2 Cyclical Encoding
   - 5.3 Lag Features
   - 5.4 Rolling Window Features
6. [Model Architecture](#6-model-architecture)
7. [Training & Evaluation Strategy](#7-training--evaluation-strategy)
8. [Feature Importance](#8-feature-importance)
9. [Design Decisions Summary](#9-design-decisions-summary)
10. [Dependencies](#10-dependencies)

---

## 1. Project Overview

This project builds a short-term electricity demand forecasting system using historical grid data from Bangladesh's Power Grid Company of Bangladesh (PGCB), enriched with weather observations and macroeconomic indicators. The model predicts the demand (in megawatts) for the **next hour** based on the current state of the grid and a wide set of engineered features.

The core modelling approach is a **LightGBM regressor** — a gradient-boosted decision tree framework that handles tabular, heterogeneous feature sets efficiently, tolerates missing values gracefully, and provides built-in feature importance rankings.

---

## 2. Data Sources

| Dataset | File | Granularity | Key Columns |
|---|---|---|---|
| Demand & Generation | `PGCB_date_power_demand.xlsx` | Hourly (with some half-hourly records) | `datetime`, `demand_mw`, `generation_mw`, `load_shedding`, `gas`, `liquid_fuel`, `coal`, `hydro` |
| Weather | `weather_data.xlsx` | Hourly | `temperature_2m`, `relative_humidity_2m`, `apparent_temperature`, `precipitation`, `cloud_cover`, `sunshine_duration` |
| Economic Indicators | `economic_full_1.csv` | Annual (World Bank format) | GDP growth, GDP per capita, electricity access %, urban population %, industry % of GDP |

---

## 3. Data Preparation

### 3.1 Half-Hourly Timestamp Handling

**Problem:** The raw demand dataset contains a mix of on-the-hour readings (e.g., `14:00`) and half-hourly readings (e.g., `14:30`). A straightforward drop of the `:30` rows would discard real information and leave the hourly series unrepresentative.

**Solution — Weighted Blend:**

Rather than dropping the half-hourly records, the pipeline blends them back into the preceding hour's row using a **60/40 weighted average**:

```
blended_value = 0.6 × hourly_value + 0.4 × half_hourly_value
```

This logic is applied to all numeric generation and demand columns: `demand_mw`, `generation_mw`, `load_shedding`, `gas`, `liquid_fuel`, `coal`, `hydro`.

**Why 60/40?** The `:00` reading represents the dominant state for the full hour. The `:30` reading captures intra-hour variation. A 60/40 blend retains the primacy of the on-the-hour reading while incorporating the mid-hour signal, rather than ignoring it entirely.

After blending, all `:30` rows are removed, yielding a clean hourly series.

### 3.2 Missing Value Imputation

After blending and re-indexing to a perfect hourly grid (`asfreq('H')`), gaps appear where no reading existed at all (e.g., data dropout periods). The imputation is applied in a **cascade of fallbacks**, from most semantically accurate to most pragmatic:

**Step 1 — Same Hour Last Week (shift by 168 hours)**

Electricity demand is strongly periodic: 2 PM on a Tuesday closely resembles 2 PM on the previous Tuesday. This fill preserves the weekly seasonal pattern and is the most meaningful substitute.

```python
df['demand_mw'].fillna(df['demand_mw'].shift(168))
```

**Step 2 — Same Hour Two Weeks Ago (shift by 336 hours)**

If Step 1 fails (the same slot from one week ago was also missing — e.g., during a prolonged outage spanning multiple weeks), the pipeline looks back another week. The dataset's longest known gap is ~517 hours, so two-week look-back covers most cases.

**Step 3 — Forward Fill / Backward Fill**

For the very first rows in the dataset (where no prior week exists to reference) and any remaining isolated gaps, a forward fill (`ffill`) followed by a backward fill (`bfill`) is applied as a final catch-all. This is semantically weak but affects only a negligible number of rows.

### 3.3 Anomaly Detection & Correction

**Problem:** Raw power grid data frequently contains sensor errors, transcription mistakes, or brief outages that register as extreme spikes or drops in the demand series. If left uncorrected, these corrupt both the target variable and the lag features derived from it.

**Method — Rolling Z-Score with Median Replacement:**

A centered rolling window of **168 hours (1 week)** is used to compute a local mean and standard deviation for each point. A **Z-score threshold of 3.5** is applied:

```
z = |value − rolling_mean| / rolling_std
anomaly = z > 3.5
```

Any value whose z-score exceeds 3.5 is classified as an anomaly and replaced with the **rolling median** of its neighborhood (also computed over the same 168-hour window).

**Why rolling rather than global statistics?**  
Electricity demand has a strong seasonal pattern — summer peaks are structurally higher than winter troughs. A global z-score would flag seasonal highs as anomalies. Rolling statistics adapt to the local context, so only true outliers relative to the recent past are flagged.

**Why median replacement rather than interpolation?**  
The median is robust to the very outliers being corrected, unlike the mean. It is guaranteed to produce a plausible value from the actual distribution of nearby observations.

---

## 4. External Feature Integration

### 4.1 Weather Features

Weather is one of the strongest exogenous drivers of electricity demand. The following variables are merged from the weather dataset on a matching `datetime` key (left join on the cleaned demand dataframe):

| Feature | Physical Rationale |
|---|---|
| `temperature_2m` | Air conditioning and heating load are directly driven by ambient temperature. This is typically the single most predictive weather variable for electricity demand. |
| `apparent_temperature` | The "feels like" temperature accounts for humidity and wind, which affect human thermal comfort and therefore cooling/heating decisions more accurately than dry-bulb temperature alone. |
| `relative_humidity_2m` | High humidity increases perceived heat (see apparent temperature) and also affects industrial processes that require humidity control. |
| `precipitation` | Rain correlates with reduced solar irradiance, lower temperatures, and changes in human activity patterns, all of which shift demand. |
| `cloud_cover` | Directly modulates solar irradiance, relevant for daytime demand patterns and, in grids with solar generation, supply-side effects. |
| `sunshine_duration` | A direct proxy for solar radiation intensity; also correlates with outdoor activity and building cooling load. |

Variables excluded from the final model (`dew_point_2m`, `soil_temperature`, `wind_direction_10m`) were dropped as they are largely redundant with the retained variables or have weaker causal links to grid demand.

Any sensor dropouts in the weather data are filled with **forward fill followed by backward fill**, since short weather dropouts do not represent true meteorological changes.

### 4.2 Economic Indicators

Macroeconomic factors shape the structural, long-run trend in electricity consumption — industrialisation, urbanisation, and income growth all increase baseline demand. Five World Bank indicators are integrated at annual granularity:

| Indicator | Column Name | Rationale |
|---|---|---|
| GDP growth (annual %) | `gdp_growth` | Economic acceleration increases industrial and commercial energy use. A fast-growing economy can shift baseline demand significantly year-on-year. |
| GDP per capita (current US$) | `gdp_per_capita` | Per-capita income drives appliance ownership (air conditioners, refrigerators) and residential electricity consumption. |
| Access to electricity (% of population) | `electricity_access_pct` | Bangladesh has been rapidly extending grid coverage. Rising access directly translates to rising demand even at constant per-capita usage. |
| Urban population (% of total) | `urban_pop_pct` | Urban households and commercial establishments consume significantly more electricity than rural ones. |
| Industry (% of GDP) | `industry_pct_gdp` | Heavier industrial composition means higher energy intensity per unit of economic output. |

**Handling annual granularity:** Annual values are forward-filled per year (each hourly row for a given calendar year receives that year's economic values). Linear interpolation is applied within the economic series to fill any missing years between survey points before the yearly join.

---

## 5. Feature Engineering

### 5.1 Calendar Features

Time-of-day, day-of-week, and seasonality are the dominant drivers of demand shape. Seven raw calendar features are extracted from the `datetime` column:

| Feature | Description |
|---|---|
| `hour` | Hour of day (0–23). Captures the diurnal demand curve. |
| `day_of_week` | Day of the week (0=Monday, 6=Sunday). Captures weekday vs. weekend demand differences. |
| `month` | Month of year (1–12). Captures annual seasonality (monsoon, winter, summer). |
| `quarter` | Calendar quarter (1–4). A coarser seasonal grouping useful for economic seasonality. |
| `day_of_year` | Day number within the year (1–365/366). Captures the full annual cycle at daily resolution. |
| `week_of_year` | ISO calendar week number. Useful alongside `day_of_week` for holiday-period patterns. |
| `is_weekend` | Binary flag (1 if Saturday or Sunday). Demand on weekends is structurally lower due to reduced industrial and commercial activity. |

### 5.2 Cyclical Encoding

**Problem with raw integer calendar features:** If `hour` is passed as a raw integer (0–23), the model treats hour 0 and hour 23 as maximally distant, but they are actually adjacent in the cyclical daily rhythm. Similarly, December and January are consecutive months but are represented by integers 12 and 1, with a discontinuity of 11. Standard tree models can partially compensate via splits, but cyclical encoding makes the periodicity explicit and is more sample-efficient.

**Solution — Sine/Cosine Transformation:**

Each cyclical variable is projected onto the unit circle:

```python
feature_sin = sin(2π × raw_value / period)
feature_cos = cos(2π × raw_value / period)
```

| Raw Feature | Period | Encoded As |
|---|---|---|
| `hour` | 24 | `hour_sin`, `hour_cos` |
| `month` | 12 | `month_sin`, `month_cos` |
| `day_of_week` | 7 | `dow_sin`, `dow_cos` |

The sine/cosine pair together uniquely identify any point on the cycle while preserving its correct distance to neighbours. For example, hour 23 and hour 0 both map to points near (1, 0) on the unit circle, correctly expressing their proximity.

### 5.3 Lag Features

Lag features provide the model with the **recent history of the target variable**, enabling it to learn autoregressive patterns (e.g., "demand one hour ago is a strong predictor of demand now"). They also serve as a safety net: the strongest single predictor of next-hour demand is current-hour demand.

**Demand lags:**

| Feature | Shift | Rationale |
|---|---|---|
| `lag_1h` | 1 hour | Immediate prior value — strongest direct autoregressive signal |
| `lag_2h` | 2 hours | Captures demand momentum |
| `lag_3h` | 3 hours | Early ramp-up/ramp-down context |
| `lag_6h` | 6 hours | Half-day context (e.g., morning demand at prediction time) |
| `lag_12h` | 12 hours | Semi-diurnal; captures the corresponding AM/PM slot |
| `lag_24h` | 24 hours | Same hour yesterday — captures the diurnal pattern |
| `lag_48h` | 48 hours | Same hour two days ago |
| `lag_168h` | 168 hours (1 week) | Same hour last week — the primary weekly seasonal reference |
| `lag_336h` | 336 hours (2 weeks) | Same hour two weeks ago — reinforces weekly seasonality |

**Generation mix lags** (for `gas`, `coal`, `hydro`, `load_shedding`):

1-hour and 24-hour lags are also created for key generation and supply-constraint variables. These capture supply-side regime changes: a plant outage or fuel shortage that began yesterday will still influence today's grid state and therefore demand-served patterns.

### 5.4 Rolling Window Features

Rolling features summarise the **statistical behaviour of the recent past**, giving the model a dynamic picture of demand volatility, trend, and deviation from norms.

**Important:** All rolling features are computed on `demand_mw.shift(1)` — the series shifted back by one step. This ensures no data leakage: the rolling window at time `t` only looks at values up to `t-1`.

**Rolling mean and standard deviation:**

Computed over windows of 3, 6, 12, 24, and 168 hours:

| Feature | Interpretation |
|---|---|
| `roll_mean_3h` | Very short-term demand trend |
| `roll_mean_6h` | Quarter-day average |
| `roll_mean_12h` | Half-day average |
| `roll_mean_24h` | Full daily average |
| `roll_mean_168h` | Weekly average — captures longer regime shifts |
| `roll_std_*` | Volatility at each window; higher values indicate unstable conditions (load shedding events, grid stress) |

**Range feature:**

```python
roll_range_24h = rolling_max(24h) − rolling_min(24h)
```

Captures the peak-to-trough swing over the past 24 hours — a proxy for demand variability driven by weather extremes or event days.

**Deviation from daily average:**

```python
demand_vs_daily_avg = demand[t-1] − roll_mean_24h[t-1]
```

Encodes whether the most recent observation is above or below the day's running average, helping the model understand where in the daily cycle the current reading sits.

**Temperature rolling mean:**

```python
temp_roll_mean_6h = temperature_2m.shift(1).rolling(6).mean()
```

A short-term smoothed temperature signal, as the effect of a temperature change on demand (e.g., people switching on air conditioning) builds up over several hours rather than instantaneously.

---

## 6. Model Architecture

**Algorithm:** LightGBM (`LGBMRegressor`)

LightGBM is chosen for this task for several reasons: it handles large feature sets with mixed types (continuous, binary, cyclical) efficiently; it is robust to outliers via its leaf-wise tree growth; it supports early stopping to prevent overfitting; and it natively supports missing values, which is valuable when some lag features are unavailable at the start of the series.

**Hyperparameter Configuration:**

| Parameter | Value | Purpose |
|---|---|---|
| `n_estimators` | 2000 | Maximum number of boosting rounds; actual rounds controlled by early stopping |
| `learning_rate` | 0.03 | Small step size for fine-grained learning; pairs with high `n_estimators` |
| `num_leaves` | 127 | Controls tree complexity; 127 allows moderately deep trees while limiting overfitting |
| `min_child_samples` | 30 | Minimum samples per leaf; prevents splits on noise in small segments |
| `subsample` | 0.8 | Row subsampling per tree; introduces stochastic diversity and prevents overfitting |
| `colsample_bytree` | 0.8 | Feature subsampling per tree; similar benefit, also speeds training |
| `reg_alpha` | 0.1 | L1 regularisation; encourages sparse feature weights |
| `reg_lambda` | 0.1 | L2 regularisation; smooths weight distribution |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Parallel training using all CPU cores |

**Early Stopping:** Training halts if the validation MAPE does not improve for 100 consecutive rounds, preventing overfitting and making `n_estimators=2000` a ceiling rather than a fixed count.

---

## 7. Training & Evaluation Strategy

**Temporal Split (no shuffling):**

- **Training set:** All records with `year ≤ 2023`
- **Test set:** All records with `year == 2024`

A strict temporal split is mandatory for time-series models. Random cross-validation or shuffled splits would allow future information to leak into the training set via lag features (e.g., if a 2024 row is in training, its lag_168h value draws from a 2023 row that might appear in the test set, invalidating the evaluation).

**Excluded Columns:**

The following columns are explicitly excluded from the feature matrix, for the reasons noted:

| Column | Reason for Exclusion |
|---|---|
| `datetime` | Non-numeric identifier; information is captured via calendar features |
| `demand_mw` | The source of the target variable; including it would be direct data leakage |
| `target` | The label itself |
| `year` | Information captured by economic features and calendar features |
| `remarks` | Free-text annotation column; not machine-readable |
| `solar`, `wind` | Predominantly missing across the dataset; including sparse columns degrades model performance |
| `india_adani`, `nepal`, `india_bheramara_hvdc`, `india_tripura` | Cross-border interconnect flows; highly sparse or missing for most of the date range |

**Evaluation Metrics:**

| Metric | Formula | Interpretation |
|---|---|---|
| MAPE | mean(\|actual − predicted\| / actual) × 100 | Scale-independent; industry standard for demand forecasting |
| MAE | mean(\|actual − predicted\|) | Absolute error in MW; operationally interpretable |
| RMSE | √mean((actual − predicted)²) | Penalises large errors more heavily; sensitive to outlier predictions |

**Benchmark thresholds:**
- MAPE < 5%: Industry-grade accuracy
- MAPE 5–10%: Acceptable for operational short-term forecasting
- MAPE > 10%: Requires review (possible data leakage, insufficient features, or data quality issues)

---

## 8. Feature Importance

After training, LightGBM's built-in **split-count importance** is extracted and plotted for the top 25 features. Split-count importance counts how many times each feature is used as a split point across all trees — features that are used more frequently are more consistently informative.

**Expected top-ranked features and reasoning:**

- **`lag_1h`, `lag_24h`, `lag_168h`** — Autoregressive demand history is consistently the strongest predictor class. One-hour lag captures immediate momentum; 24-hour and 168-hour lags capture diurnal and weekly patterns.
- **`roll_mean_24h`, `roll_mean_168h`** — Smoothed demand baselines give the model a reference level against which to interpret the current situation.
- **`temperature_2m`, `apparent_temperature`** — Weather is the dominant exogenous driver, particularly during peak summer hours.
- **`hour_sin`, `hour_cos`** — Time of day is a fundamental demand shape driver.
- **`is_weekend`** — Weekend demand is structurally lower due to reduced industrial and commercial load.
- **`roll_std_24h`** — Demand volatility is predictive: high-variance periods (e.g., load-shedding events, weather extremes) require different model behaviour.
- **`demand_vs_daily_avg`** — Deviation from the running daily average captures where in the intraday cycle the grid currently sits.

The feature importance plot is saved as `feature_importance.png`.

---

## 9. Design Decisions Summary

| Decision | Choice Made | Rationale |
|---|---|---|
| Half-hourly blending | 60/40 weighted average into the hour | Retains intra-hour information without disrupting hourly granularity |
| Missing value imputation | Cascaded same-week → two-week → ffill | Prioritises seasonal pattern preservation over statistical smoothing |
| Anomaly detection | Centered rolling Z-score (window=168, threshold=3.5) | Adapts to local seasonality; robust to structural level shifts |
| Anomaly correction | Replace with rolling median | Robust estimator unaffected by the outlier being replaced |
| Cyclical encoding | Sine/cosine transformation | Preserves true cyclical distance for tree models |
| Lag window selection | 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h, 336h | Covers immediate autocorrelation, diurnal, and weekly seasonality |
| Rolling windows | 3h, 6h, 12h, 24h, 168h | Multi-scale volatility and trend features |
| Train/test split | Temporal (≤2023 train, 2024 test) | Prevents look-ahead bias inherent in time-series data |
| Model choice | LightGBM | Fast, handles heterogeneous features, robust to missing values, interpretable importance |

---

## 10. Dependencies

```
pandas
numpy
matplotlib
seaborn
lightgbm
scikit-learn
openpyxl       # for reading .xlsx files
warnings
```

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn lightgbm scikit-learn openpyxl
```


