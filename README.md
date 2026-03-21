# NASA Li-ion Battery Aging Dataset — Data Cleaning & Preprocessing Report

**Course:** CSE445.4 Machine Learning  
**Assignment:** Dataset Plotting and Preprocessing  
**Notebook:** `analysis.ipynb`  
**Dataset Source:** [Kaggle — NASA Battery Dataset by patrickfleith](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset)  
**Original Source:** NASA Ames Prognostics Center of Excellence (PCoE) — DASHLINK_133  
**Date Processed:** March 21, 2026  

---

## Table of Contents

1. [Dataset Background](#1-dataset-background)
2. [Original File Structure](#2-original-file-structure)
3. [Libraries Used](#3-libraries-used)
4. [Phase 1 — Metadata Exploration and Cleaning](#4-phase-1--metadata-exploration-and-cleaning)
5. [Phase 2 — Loading Raw Cycle CSV Files](#5-phase-2--loading-raw-cycle-csv-files)
6. [Phase 3 — Charge Dataframe Cleaning](#6-phase-3--charge-dataframe-cleaning)
7. [Phase 4 — Discharge Dataframe Cleaning](#7-phase-4--discharge-dataframe-cleaning)
8. [Phase 5 — Impedance Dataframe Cleaning](#8-phase-5--impedance-dataframe-cleaning)
9. [Final Shapes and Validation](#9-final-shapes-and-validation)
10. [Cleaned File Structure](#10-cleaned-file-structure)
11. [Column Descriptions](#11-column-descriptions)
12. [Key Decisions and Justifications](#12-key-decisions-and-justifications)
13. [How to Reload the Cleaned Data](#13-how-to-reload-the-cleaned-data)
14. [Saving a Notebook Snapshot](#14-saving-a-notebook-snapshot)

---

## 1. Dataset Background

This dataset was collected at the **NASA Ames Prognostics Center of Excellence (PCoE)**. Commercially available Li-ion 18650 rechargeable batteries were repeatedly cycled through charge, discharge, and impedance operations under controlled lab conditions until they reached End-of-Life (EOL).

**Experimental Setup:**
- Batteries subjected to 3 operation types: **charge**, **discharge**, and **electrochemical impedance spectroscopy (EIS)**
- Tests conducted at ambient temperatures of **4 degrees C and 24 degrees C**
- Discharges carried out at different current load levels until battery voltage fell to preset thresholds
- Some thresholds set below OEM recommendation (2.7V) to induce **deep discharge aging effects**
- **EOL criterion:** 30% capacity fade — rated capacity drops from **2.0 Ah to 1.4 Ah**
- Data acquisition rate: approximately **10 Hz**
- Equipment: programmable DC load/supply, voltmeter, ammeter, thermocouple, EIS equipment, environmental chamber, PXI DAQ system

**Scientific Purpose:**
The dataset enables development of **Remaining Useful Life (RUL)** prediction algorithms and study of battery degradation patterns. No two cells have the same State-of-Life (SOL) at the same cycle index due to differences in depth-of-discharge, rest periods, and intrinsic variability.

**Kaggle vs NASA Original:**
The Kaggle version (used in this project) is a community re-upload by Patrick Fleith where the original MATLAB `.mat` files have been converted into CSV format. The underlying experimental data is identical to the NASA original (DASHLINK_133).

---

## 2. Original File Structure

```
data/
├── data/                  <- ~7,500 individual CSV files (one per cycle)
│   ├── 00001.csv
│   ├── 00002.csv
│   └── ... (up to 07565.csv)
├── extra_infos/           <- Supplementary files
└── metadata.csv           <- Master index: one row per cycle (830 KB)
```

**metadata.csv** is the index file — each row maps one experimental cycle to its battery, operation type, timestamp, and corresponding CSV filename.

**Individual CSV files** contain raw time-series sensor readings for that single cycle. The column structure differs by cycle type:

| Cycle Type | Columns in CSV |
|------------|----------------|
| charge | Voltage_measured, Current_measured, Temperature_measured, Current_charge, Voltage_charge, Time |
| discharge | Voltage_measured, Current_measured, Temperature_measured, Current_load, Voltage_load, Time |
| impedance | Sense_current, Battery_current, Current_ratio, Battery_impedance, Rectified_Impedance |

Note: charge and discharge share 4 common columns but differ in the load vs charge current/voltage columns. Impedance has a completely separate structure with complex-valued measurements.

---

## 3. Libraries Used

```python
import pandas as pd        # v3.0.1  — data loading, cleaning, manipulation
import numpy as np         # v2.4.3  — numerical operations
import matplotlib.pyplot   # v3.10.8 — base plotting backend
import seaborn             # v0.13.2 — statistical visualizations
import scipy               # required for seaborn cumulative KDE plots
import os                  # file path and directory management
```

`sns.set_theme()` was used instead of the deprecated `sns.set()` for consistent plot styling throughout the notebook.

---

## 4. Phase 1 — Metadata Exploration and Cleaning

### 4.1 Initial Load

```python
df = pd.read_csv('data/metadata.csv')
```

**Result:** Shape `(7565, 10)` — 7,565 cycles, 10 columns.

**Columns:**
```
['type', 'start_time', 'ambient_temperature', 'battery_id',
 'test_id', 'uid', 'filename', 'Capacity', 'Re', 'Rct']
```

### 4.2 Data Type Problem — Capacity, Re, Rct loaded as strings

Upon inspection of dtypes:

```
type                     object (str)
start_time               object (str)
ambient_temperature      int64
battery_id               object (str)
test_id                  int64
uid                      int64
filename                 object (str)
Capacity                 object (str)   <- PROBLEM: should be float
Re                       object (str)   <- PROBLEM: should be float
Rct                      object (str)   <- PROBLEM: should be float
```

`Capacity`, `Re`, and `Rct` should be `float64` but were read as strings due to inconsistent value formatting in the raw CSV. This prevented any mathematical operations or plotting on these columns.

**Fix applied:**
```python
df['Capacity'] = pd.to_numeric(df['Capacity'], errors='coerce')
df['Re']       = pd.to_numeric(df['Re'],       errors='coerce')
df['Rct']      = pd.to_numeric(df['Rct'],      errors='coerce')
```

`errors='coerce'` safely converts any unparseable string to NaN rather than raising an error.

### 4.3 Missing Value Analysis

After dtype conversion, missing values per column:

| Column | Missing Count | Missing % | Classification |
|--------|--------------|-----------|----------------|
| type | 0 | 0.00% | Clean |
| start_time | 0 | 0.00% | Clean |
| ambient_temperature | 0 | 0.00% | Clean |
| battery_id | 0 | 0.00% | Clean |
| test_id | 0 | 0.00% | Clean |
| uid | 0 | 0.00% | Clean |
| filename | 0 | 0.00% | Clean |
| Capacity | 4796 | 63.40% | Mostly structural |
| Re | 5618 | 74.26% | Mostly structural |
| Rct | 5618 | 74.26% | Mostly structural |

**Cycle type distribution:**
```
charge       2815 rows
discharge    2794 rows
impedance    1956 rows
```

**Accounting for structural NaNs:**

Capacity NaNs: charge (2815) + impedance (1956) = expected 4771, actual 4796 → **25 genuinely missing discharge rows**

Re/Rct NaNs: charge (2815) + discharge (2794) = expected 5609, actual 5618 → **9 genuinely missing impedance rows**

### 4.4 Truly Missing Rows Identified

**25 discharge rows with no Capacity:**
- Batteries **B0050** (4 rows) and **B0052** (21 rows)
- All 3 measurement columns NaN simultaneously
- Cause: data acquisition failure during those test sessions

**9 impedance rows with no Re/Rct:**
- Batteries **B0049** (8 rows) and **B0051** (1 row)
- Cause: acquisition failure during EIS measurement

### 4.5 Action — Drop the 34 Truly Missing Rows

Decision: **Drop** rather than fill.

Justification:
- These rows had zero measurements — filling with mean/median is scientifically dishonest
- Capacity changes continuously over a battery's life — a mean value has no physical meaning for a specific cycle
- 34 rows out of 7,565 = only 0.45% data loss

```python
df = df.drop(missing_capacity.index)
df = df.drop(missing_impedance.index)
```

**Shape after dropping:** `(7531, 10)`

### 4.6 Duplicate Check

```
Duplicate rows:      0
Duplicate UIDs:      0
Duplicate filenames: 0
```

No duplicates found.

---

## 5. Phase 2 — Loading Raw Cycle CSV Files

All ~7,500 individual CSV files were loaded and merged into 3 separate dataframes by cycle type. Metadata columns were attached to each row to preserve battery and cycle context.

```python
DATA_PATH = 'data/data/'

for _, row in df.iterrows():
    filepath = os.path.join(DATA_PATH, row['filename'])
    temp = pd.read_csv(filepath)

    temp['battery_id']          = row['battery_id']
    temp['cycle_type']          = row['type']
    temp['ambient_temperature'] = row['ambient_temperature']
    temp['test_id']             = row['test_id']
    temp['uid']                 = row['uid']

    if row['type'] == 'charge':
        charge_dfs.append(temp)
    elif row['type'] == 'discharge':
        discharge_dfs.append(temp)
    elif row['type'] == 'impedance':
        impedance_dfs.append(temp)
```

**Shapes after concatenation:**

| Dataframe | Rows | Columns | Each row represents |
|-----------|------|---------|---------------------|
| charge_df | 6,512,876 | 11 | One ~6-second time-step during a charge cycle |
| discharge_df | 764,674 | 11 | One time-step during a discharge cycle |
| impedance_df | 93,456 | 10 | One EIS frequency sweep data point |

The large size of charge_df (6.5M rows) is expected: approximately 2,815 charge cycles multiplied by hundreds of time-steps per cycle.

---

## 6. Phase 3 — Charge Dataframe Cleaning

**Initial shape:** `(6,512,876, 11)`

### Missing Values

| Column | Missing Count | Missing % |
|--------|--------------|-----------|
| Voltage_measured | 257 | 0.004% |
| Current_measured | 257 | 0.004% |
| Temperature_measured | 257 | 0.004% |
| All other columns | 0 | 0.00% |

257 rows had all 3 sensor readings missing simultaneously — sensor dropout during a cycle.

**Action: Drop**

```python
charge_df = charge_df.dropna(
    subset=['Voltage_measured', 'Current_measured', 'Temperature_measured']
)
```

257 out of 6.5M rows = 0.004% — negligible loss.

**Duplicates:** 0

**Final shape:** `(6,512,619, 11)`

---

## 7. Phase 4 — Discharge Dataframe Cleaning

**Initial shape:** `(764,674, 11)`

**Missing values:** 0 across all columns — perfectly clean.

**Duplicates:** 0

**No action required.**

**Final shape:** `(764,674, 11)` — unchanged.

---

## 8. Phase 5 — Impedance Dataframe Cleaning

**Initial shape:** `(93,456, 10)`

This was the most complex cleaning phase due to complex-valued measurements and significant outliers.

### Step 1 — Convert Complex Strings to Python Complex Type

All 5 measurement columns contained complex numbers stored as plain strings, for example:
```
"(0.190217+0.079140j)"
```

These were converted to Python native complex type:

```python
for col in ['Sense_current', 'Battery_current', 'Current_ratio',
            'Battery_impedance', 'Rectified_Impedance']:
    impedance_df[col] = impedance_df[col].apply(
        lambda x: complex(x) if pd.notna(x) else x
    )
```

### Step 2 — Detect and Remove Extreme Outliers

Statistical inspection revealed extreme outliers despite realistic median values:

| Statistic | Battery_impedance real | Rectified_Impedance real |
|-----------|------------------------|--------------------------|
| min | -1.94e+15 Ohms | -4.02e+15 Ohms |
| 25% | 0.190 Ohms | 0.070 Ohms |
| median | 0.214 Ohms | 0.086 Ohms |
| 75% | 0.258 Ohms | 0.115 Ohms |
| max | 1.79e+15 Ohms | 4.18e+15 Ohms |
| mean | 3.31e+10 Ohms | 6.67e+09 Ohms |

The median (0.21 Ohms) is physically realistic for Li-ion batteries. Values of 10^15 Ohms are clearly data acquisition errors — the mean was being completely distorted by a handful of extreme rows.

**Threshold:** Realistic Li-ion impedance = **within ±2.0 Ohms**

```
Battery_impedance outside ±2 Ohms:     683 rows (0.73%)
Rectified_Impedance outside ±2 Ohms:   525 rows (0.56%)
```

```python
valid_mask = (batt_real.abs() <= 2.0) & (rect_real.abs() <= 2.0)
impedance_df = impedance_df[valid_mask].reset_index(drop=True)
```

**After outlier removal:** 75,376 rows (removed 18,080 total including outliers and previously NaN rows)

### Step 3 — Extract Real and Imaginary Parts as Separate Float Columns

Complex numbers cannot be used directly in ML models or most plotting functions, so real and imaginary parts were split:

```python
impedance_df['Battery_impedance_real']   = impedance_df['Battery_impedance'].apply(lambda x: x.real)
impedance_df['Battery_impedance_imag']   = impedance_df['Battery_impedance'].apply(lambda x: x.imag)
impedance_df['Rectified_Impedance_real'] = impedance_df['Rectified_Impedance'].apply(lambda x: x.real if isinstance(x, complex) else None)
impedance_df['Rectified_Impedance_imag'] = impedance_df['Rectified_Impedance'].apply(lambda x: x.imag if isinstance(x, complex) else None)
```

Columns grew from 10 to **14**.

### Step 4 — Investigate Battery_impedance vs Rectified_Impedance

A scatter plot comparing real and imaginary parts of both columns revealed:
- Battery_impedance real part spreads widely: -1.5 to 2.0 Ohms
- Rectified_Impedance real part is tightly constrained: 0.05 to 0.25 Ohms

**Conclusion:** Rectified_Impedance is a heavily smoothed and calibrated version of Battery_impedance. They cannot be substituted for each other — the two columns represent fundamentally different processed versions of the same raw signal.

### Step 5 — Fill Remaining Missing Rectified_Impedance with Median

Since Rectified_Impedance is tightly clustered, the **median** is a statistically sound imputation:

```python
median_real = 0.086359   # median of Rectified_Impedance_real
median_imag = -0.001695  # median of Rectified_Impedance_imag

impedance_df['Rectified_Impedance_real'].fillna(median_real, inplace=True)
impedance_df['Rectified_Impedance_imag'].fillna(median_imag, inplace=True)
```

**Missing after fill: 0**

**Duplicates:** 0

**Final shape:** `(75,376, 14)`

---

## 9. Final Shapes and Validation

### Shapes
```
metadata (df):   (7531,    10)
charge_df:       (6512619, 11)
discharge_df:    (764674,  11)
impedance_df:    (75376,   14)
```

### Missing Values — Final State
```
df:            15,930  <- structural NaNs by design (Capacity/Re/Rct)
charge_df:          0
discharge_df:       0
impedance_df:       0
```

The 15,930 missing values in metadata are **structural by design** — Capacity is only recorded for discharge cycles, Re and Rct only for impedance cycles. These are not errors and should not be filled.

### Duplicates — Final State
```
df:            0
charge_df:     0
discharge_df:  0
impedance_df:  0
```

---

## 10. Cleaned File Structure

```
data/cleaned_data/
├── metadata_cleaned.csv              (7,531 rows  x 10 cols)
├── discharge_cleaned.csv             (764,674 rows x 11 cols)
├── impedance_cleaned.csv             (75,376 rows  x 14 cols)
├── charge_chunks/
│   ├── charge_part_01_of_05.csv      (1,600,000 rows)
│   ├── charge_part_02_of_05.csv      (1,600,000 rows)
│   ├── charge_part_03_of_05.csv      (1,600,000 rows)
│   ├── charge_part_04_of_05.csv      (1,600,000 rows)
│   └── charge_part_05_of_05.csv      (112,619 rows)
└── README.md                         <- this file
```

charge_df was split into 5 chunks of approximately 1,600,000 rows each (~150 MB per file) for practical file management. The data is identical to a single merged dataframe.

---

## 11. Column Descriptions

### metadata_cleaned.csv

| Column | Type | Description |
|--------|------|-------------|
| type | str | Cycle operation type: charge, discharge, or impedance |
| start_time | str | Cycle start time as MATLAB date vector [year month day hour min sec] |
| ambient_temperature | int | Ambient temperature during test in degrees C (4 or 24) |
| battery_id | str | Battery identifier e.g. B0047, B0055 |
| test_id | int | Cycle index within that battery's lifetime starting from 0 |
| uid | int | Unique cycle ID across the entire dataset (1 to 7531) |
| filename | str | Source CSV filename in data/data/ e.g. 00001.csv |
| Capacity | float | Battery discharge capacity in Ah — discharge rows only, NaN otherwise |
| Re | float | Electrolyte resistance in Ohms — impedance rows only, NaN otherwise |
| Rct | float | Charge transfer resistance in Ohms — impedance rows only, NaN otherwise |

### discharge_cleaned.csv and charge_chunks

| Column | Type | Description |
|--------|------|-------------|
| Voltage_measured | float | Battery terminal voltage in Volts |
| Current_measured | float | Battery output current in Amps |
| Temperature_measured | float | Battery temperature in degrees C |
| Current_load (discharge only) | float | Current measured at load in Amps |
| Voltage_load (discharge only) | float | Voltage measured at load in Volts |
| Current_charge (charge only) | float | Current measured at charger in Amps |
| Voltage_charge (charge only) | float | Voltage measured at charger in Volts |
| Time | float | Elapsed time within this cycle in seconds |
| battery_id | str | Battery identifier |
| cycle_type | str | Always discharge or charge |
| ambient_temperature | int | Ambient temperature in degrees C |
| test_id | int | Cycle number within battery lifetime |
| uid | int | Unique cycle ID linking back to metadata |

### impedance_cleaned.csv

| Column | Type | Description |
|--------|------|-------------|
| Sense_current | complex | Current in sense branch in Amps |
| Battery_current | complex | Current in battery branch in Amps |
| Current_ratio | complex | Ratio of sense to battery current |
| Battery_impedance | complex | Raw battery impedance in Ohms |
| Rectified_Impedance | complex | Calibrated and smoothed battery impedance in Ohms |
| Battery_impedance_real | float | Real part of Battery_impedance |
| Battery_impedance_imag | float | Imaginary part of Battery_impedance |
| Rectified_Impedance_real | float | Real part of Rectified_Impedance — median-filled where missing |
| Rectified_Impedance_imag | float | Imaginary part of Rectified_Impedance — median-filled where missing |
| battery_id | str | Battery identifier |
| cycle_type | str | Always impedance |
| ambient_temperature | int | Ambient temperature in degrees C |
| test_id | int | Cycle number within battery lifetime |
| uid | int | Unique cycle ID linking back to metadata |

---

## 12. Key Decisions and Justifications

| Decision | What was done | Why |
|----------|---------------|-----|
| Drop 34 metadata rows | Dropped 25 discharge and 9 impedance rows with fully missing measurements | All measurement columns were NaN simultaneously indicating data acquisition failure. Only 0.45% of data. Filling would be scientifically invalid |
| Three separate dataframes | charge_df, discharge_df, impedance_df kept separate | The 3 cycle types have incompatible column structures and cannot be merged without creating massive structural NaN columns throughout |
| Drop 257 charge rows | Removed rows where Voltage, Current, and Temperature were all simultaneously missing | Sensor dropout pattern. Only 0.004% of charge data — negligible |
| Impedance outlier threshold 2 Ohms | Removed rows where real part exceeded plus or minus 2 Ohms | Realistic Li-ion impedance is 0.01 to 0.5 Ohms. Values of 10^15 Ohms are instrument errors. Only 0.73% of rows |
| Median fill for Rectified_Impedance | Filled missing real part with 0.086359 and imaginary part with -0.001695 | Rectified_Impedance has a tight compact distribution making median a safe representative value. Substituting raw Battery_impedance was confirmed invalid by scatter plot analysis |
| Charge split into 5 chunks | Files of approximately 1,600,000 rows each at roughly 150 MB per file | Purely for practical file management. A single 600 MB CSV is impractical to open or share |
| Keep structural NaNs in metadata | Did not fill Capacity, Re, or Rct NaNs in metadata | These are by experimental design — each column only applies to one cycle type. Filling them would produce meaningless values |

---

## 13. How to Reload the Cleaned Data

```python
import pandas as pd
import glob

# Metadata
df = pd.read_csv('data/cleaned_data/metadata_cleaned.csv')

# Discharge
discharge_df = pd.read_csv('data/cleaned_data/discharge_cleaned.csv')

# Impedance
impedance_df = pd.read_csv('data/cleaned_data/impedance_cleaned.csv')

# Charge — reload all chunks and merge
charge_files = sorted(glob.glob('data/cleaned_data/charge_chunks/*.csv'))
charge_df = pd.concat([pd.read_csv(f) for f in charge_files], ignore_index=True)

print("metadata:   ", df.shape)
print("charge:     ", charge_df.shape)
print("discharge:  ", discharge_df.shape)
print("impedance:  ", impedance_df.shape)
```

Note: Reloading all charge chunks will take a few minutes due to 6.5M rows.

---

## 14. Saving a Notebook Snapshot

To permanently save the notebook with all outputs, printed results, and rendered graphs intact, run one of these commands from the terminal inside the project directory with your .venv activated:

### Option 1 — HTML (recommended — preserves all outputs and graphs)
```bash
jupyter nbconvert --to html analysis.ipynb --output analysis_snapshot.html
```

### Option 2 — PDF (requires LaTeX)
```bash
jupyter nbconvert --to pdf analysis.ipynb --output analysis_snapshot.pdf
```

### Option 3 — Execute fresh and export in one command
```bash
jupyter nbconvert --to html --execute analysis.ipynb --output analysis_snapshot.html
```

The HTML file is fully self-contained and captures the complete run state of the notebook including all tables, statistics, and matplotlib/seaborn figures. This is the recommended format for submission and archiving.