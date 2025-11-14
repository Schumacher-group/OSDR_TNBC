# OSDR Workflow Implementation Analysis

## Overview
This is a codebase implementing the **One-Shot tissue Dynamics Reconstruction (OSDR)** method validation pipeline developed by Gabin Rousseau (MSc Bioinformatics). The project applies OSDR to Triple Negative Breast Cancer (TNBC) biopsy data.

**Key Paper Reference:** Somer et al. (preprint included in repo)

---

## 1. SPATIAL SIMULATION CODE

### Location
**Primary Implementation:** `/home/user/OSDR_TNBC/notebooks/validation_simulations_alt.ipynb`
- **Better validated version** using logistic regression for division probability
- Alternative version: `validation_simulations.ipynb` (uses simpler logistic ODE: dX/dt = aX - bX²)

### Cell Placement
**Function:** `random_tissue()` (Cell 27 in validation_simulations_alt.ipynb)
- **Tissue dimensions:** 2500 × 2500 microns
- **Cell types:** F (Fibroblasts) and M (Myoepithelial) - two types
- **Initial distribution:** Random uniform placement
  - F cells: randomly sampled between 250-9750 cells
  - M cells: randomly sampled between 250-9750 cells
- **Storage:** Dataframe columns: `Tissue_ID`, `Cell_Type`, `X`, `Y`
- **Initial neighborhood counts computed** immediately after placement

### Neighborhood Counting
**Data Structure (after proliferation):**
```
Dataframe columns:
- Tissue_ID: tissue replicate identifier
- Cell_Type: "F" or "M"
- X, Y: spatial coordinates (microns)
- #F_neighbours: count of F cells in radius (for all cells)
- #F_neighbours_log2: log2 of above
- #M_neighbours: count of M cells in radius (for all cells)
- #M_neighbours_log2: log2 of above
- Time_Step: recording timepoint
```

**Radius parameter:** `r = 80 microns` (same as Somer et al.)

**Computation Method:** scipy.spatial.cKDTree
```python
tree = cKDTree(tissue[tissue['Cell_Type'] == ctype][['X', 'Y']].values)
neighbours = [len(tree.query_ball_point(coords, r)) - (1 if cat == ctype else 0)
              for coords, cat in zip(coordinates, tissue['Cell_Type'])]
```
- Uses ball point query at each cell location
- **Subtracts 1 for self-neighbors** (removes self from count when same cell type)
- Recomputed after EVERY step (cell division/death)

### Division Events
**Division Probability Formula (logistic regression):**
```python
# Cell type-specific
p_div_F = 1/(1 + exp(-(intercept_F + a_F * #F_neighbours)))
p_div_M = 1/(1 + exp(-(intercept_M + a_M * #M_neighbours)))

# Known model parameters (same for both types for simplicity):
a = -0.120
intercept = -2.456
```

**Division mechanics:**
1. New cell placed at **distance r_disp = 80 microns** from parent
2. Direction: random uniform in 2D
3. **Boundary handling:** prevents placement outside [0, 2500]²
   - If x+rx < 0: resample from [0, r]
   - If x+rx > 2500: resample from [-r, 0]
   - Same for y

**Cell division creates new row** in dataframe with same tissue_id but different row index

### Death Events
**Death Probability:**
```python
p_death = b = 1 / (1 + exp(-(intercept + a*16)))
```
- **Constant death rate** independent of neighborhood
- Calculated at steady state (16 cells) → approximately 0.104

**Death mechanics:**
- Dead cells **removed from dataframe** via row deletion
- Dead cell IDs stored and dropped after step completes

### Tissue Boundaries
**Explicit boundary handling:**
- Hard boundaries at x=[0, 2500], y=[0, 2500] microns
- Prevents proliferation outside boundaries with conditional resampling
- **No explicit edge correction for neighborhood counts** (cells at boundaries have reduced neighborhoods naturally)

**Time stepping:**
- Single step = all cells get stochastic event sample → division/death/stay
- Neighborhoods recomputed after every step
- Optional intermediate recording at time intervals (parameter `t`)

---

## 2. NEIGHBORHOOD COMPUTATION

### Method: scipy.spatial.cKDTree
**Implementation in `tissue_proliferation()` and `random_tissue()`:**
```python
from scipy.spatial import cKDTree

# For each cell type separately
tree = cKDTree(tissue[tissue['Cell_Type'] == ctype][['X', 'Y']].values)

# Query all cells
neighbours = [len(tree.query_ball_point(coords, r)) - (1 if cat == ctype else 0)
              for coords, cat in zip(coordinates, tissue['Cell_Type'])]
```

### Edge Correction: NO EXPLICIT IMPLEMENTATION
- **No edge correction function present**
- Boundary effects handled implicitly:
  - Cells near edges have naturally fewer neighbors (boundary truncation)
  - No radius correction or density weighting applied
  - No Kaplan-Meier or other statistical corrections

### Features:
- **Spatial index:** KDTree on (X, Y) coordinates
- **Query radius:** r = 80 microns (Euclidean distance)
- **Self-counting:** Explicitly removes self from count with `-(1 if cat == ctype else 0)`
- **Per-cell-type:** Computed separately for F and M cell types
- **All cells queried:** Every cell gets a count, even with 0 neighbors

---

## 3. KI67 PROCESSING

### Current Status: NOT IMPLEMENTED IN VALIDATION CODE
**Findings:**
- ✓ Real dataset has Ki67 columns: `Ki-67`, `Ki-67_nuclear` 
- ✓ Referenced in TNBC exploration notebook (`tnbc_exploration.ipynb`)
- ✗ **NO Ki67 normalization or threshold selection code found**
- ✗ Simulated tissue generation does NOT use Ki67

**Activity log note (Week 1):**
> "Authors should provide some information with how ki67 relates to the final sampling step described under 'Validating OSDR by simulating known dynamical models'."

**Current validation approach:**
- Uses **direct neighborhood counts** as division probability features
- Does NOT infer Ki67 from data
- Simulations purely based on mathematical model

---

## 4. LOGISTIC REGRESSION INFERENCE

### Location
**Function:** `tissue_regression()` (Cell 44 in validation_simulations_alt.ipynb)

### Workflow:

**Step 1: Sample cells from simulated tissue**
```python
main_sample, k1_df, k5_df, k10_df, k25_df = cell_sampler(post_df, t, seed)
```
- Samples: 50k total, 5k per tissue, from 100 tissue replicates
- Creates subsamples: 1k, 5k, 10k, 25k cells for testing sample size effects

**Step 2: Generate synthetic division observations**
```python
# For each cell, sample stochastic event based on KNOWN model parameters:
p_div = 1/(1 + exp(-(intercept + a * #F_neighbours)))
p_death = b
p_stay = 1 - p_div - p_death

# Sample: r ~ U(0,1)
if r <= p_div:
    y = 1  # Division observed
else:
    y = 0  # No division (death or stay)
```

**Step 3: Fit logistic regression per cell type**
```python
from sklearn.linear_model import LogisticRegression

X_f = df[["#F_neighbours"]].loc[df["Cell_Type"]=="F"]  # Feature: F-neighborhood count
y_f = df["Division_Observation"].loc[df["Cell_Type"]=="F"]  # Label: division yes/no

model_f = LogisticRegression(random_state=seed).fit(X_f, y_f)

# Extract parameters
intercept_f = model_f.intercept_[0]
coef_f = model_f.coef_[0][0]  # slope
```

**Step 4: Evaluate inferred division probability**
```python
# At steady state (X=16):
z = intercept + 16 * coef
p_plus = 1 / (1 + exp(-z))

# Compare to known
p_death = mean(y)  # empirical death rate from sampled events
```

### Return Values
```python
return pplus_f, pplus_m, pminus_f, pminus_m
```
- `pplus_f[i]`: [intercept, slope] for F cells from sample i
- `pminus_f[i]`: scalar death rate for F cells from sample i
- Same for M cells
- Dictionary keys 0-3: correspond to 1k, 5k, 10k, 25k samples

### Key Features:
- ✓ **Univariate** features (one neighborhood type per cell type)
- ✓ **Per-cell-type fitting** (separate models for F and M)
- ✓ **Logistic form preserved** from known model
- ✓ **No edge corrections applied** to feature data
- ✓ **Binary classification:** division vs. no-division

---

## 5. CODE ORGANIZATION

### Notebooks vs. Modules

#### Main Validation Notebooks (in `/notebooks/`)

**1. `validation_simulations_alt.ipynb` (RECOMMENDED)**
- **Status:** Latest, validated version
- **Model type:** Logistic regression form for p+ (neighborhood-dependent)
  - `p+ = 1/(1+exp(-(intercept+a*neighbors)))`
  - `p- = b (constant)`
- **Key sections:**
  - Cell 6: Imports (scipy.spatial.cKDTree, sklearn)
  - Cell 9: `random_tissue()` - initial tissue generation
  - Cell 27: `simulate_model()` - master orchestrator
  - Cell 41: `cell_sampler()` - stratified cell sampling
  - Cell 44: `tissue_regression()` - logistic regression inference
  - Cell 62: `div_death_known()`, `div_death_inferred()` - model comparison
  - Cell 31: Execution call
  - Output: `simulated_tissues_alt.csv`, `simulated_tissues_post_alt.csv`

**2. `validation_simulations.ipynb` (EARLIER VERSION)**
- **Model type:** Simple logistic ODE
  - `dX/dt = aX - bX²` (autocatalytic)
  - Parameters: a=48e-4, b=3e-4, SS=16
  - No explicit neighborhood dependence in equation (though spatial sim still uses neighborhoods)
- Functionally similar but less faithful to paper's methodology

#### Python Modules (in `/notebooks/osdr_validation/`)

**1. `phase_portrait_alt.py`
- Uses logistic regression form from alt notebook
- Parameters: a=-0.120, intercept=-2.456, b≈0.104
- Functions:
  - `ODE_system()`: dX/dt equations for phase portrait
  - `nullclines()`: Find F and M nullclines
  - `find_fixed_points()`: Locate equilibria
  - `is_stable()`, `is_unstable()`: Jacobian-based stability via autograd
  - `plot_phase_portrait()`: Visualization with streamlines

**2. `phase_portrait.py`
- Uses simpler logistic form from original notebook
- Parameters: F_a=48e-4, F_b=3e-4, M_a=48e-4, M_b=3e-4
- Same structure as phase_portrait_alt.py

**3. `inferred_portraits.py`
- **Currently empty** (stub file)
- Intended for: plotting phase portraits from inferred parameters

**4. `__init__.py`
- Empty; modules imported directly

### Function Call Hierarchy

```
simulate_model(replicates=100, n=1000, t=100)
├── random_tissue(seed, cells, tissue_id)  [parallelized via Pool]
│   ├── Generate random (X,Y) positions
│   ├── Build cKDTree per cell type
│   └── Compute initial neighbors
│
├── tissue_proliferation(seed, tissue, n, t)  [parallelized via Pool]
│   └── For step in 1..n:
│       ├── cell_action(cell_id, t, seed)  [parallelized via starmap]
│       │   ├── Sample event (div/death/stay) from probabilities
│       │   └── Return new_cell or death_id
│       │
│       ├── Remove dead cells from dataframe
│       │
│       └── Recompute neighbors for all cells
│           └── Build cKDTree per type
│               └── query_ball_point for all cells
│
└── [Output: simulated tissues at final + intermediate timepoints]

tissue_regression(post_df, t, seed)
├── cell_sampler(post_df, t, seed)
│   └── Stratified sampling (50k total from 100 tissues)
│
├── For each sample size (1k, 5k, 10k, 25k):
│   ├── Generate synthetic division labels
│   │   └── Sample events using KNOWN model
│   │
│   ├── LogisticRegression.fit(X, y) [per cell type]
│   │   ├── X: neighborhood counts
│   │   └── y: division observations
│   │
│   └── Extract [intercept, slope] parameters
│
└── Return: pplus_f, pplus_m, pminus_f, pminus_m dicts
```

### Computation Flow in validate_simulations_alt.ipynb

1. **Tissue initialization** (parallelized)
   - 100 replicates × random_tissue()
   - Each ~5000 cells per type
   - Time: ~1 min

2. **Tissue proliferation** (parallelized, intensive)
   - 100 replicates × 1000 steps
   - Each step: stochastic events + neighborhood recomputation
   - Time: ~10 min

3. **Sampling and regression** (sequential)
   - Extract 50k cells from post-proliferation tissues
   - Fit logistic models (4 sample sizes)
   - Compare inferred vs. known parameters
   - Time: ~1-2 min

---

## 6. KEY PARAMETERS AND CONSTANTS

### Tissue-scale
| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| X bounds | [0, 2500] | μm | Notebook constant |
| Y bounds | [0, 2500] | μm | Notebook constant |
| Neighborhood radius | 80 | μm | Somer et al. |
| Cell types | F, M | - | Fibroblasts, Myoepithelial |
| Initial cell count | ~5,000-10,000 per type | cells | Stochastic, per tissue |

### Division/Death (Logistic Regression Model)
| Parameter | Value | Meaning |
|-----------|-------|---------|
| a | -0.120 | Slope (log odds per neighbor) |
| intercept | -2.456 | Intercept (log odds at 0 neighbors) |
| b | ~0.104 | Constant death rate |
| Steady state (p+=p-) | 16 | Neighbors (log2=4) |

### Simulation-scale
| Parameter | Value | Notes |
|-----------|-------|-------|
| Division displacement | r=80 μm | Distance from parent cell |
| Simulation steps | 1000 | Default n |
| Recording interval | 100 | Time steps between recordings |
| Replicates | 100 | Number of independent tissues |

### Logistic Regression
| Parameter | Value | Notes |
|-----------|-------|-------|
| Solver | sklearn default | (LBFGS) |
| max_iter | default (100) | Usually converges <20 |
| Sample sizes tested | 1k, 5k, 10k, 25k | To test convergence |
| Features per cell type | 1 | (#neighbors of same type) |

---

## 7. DATA STRUCTURES

### Tissue DataFrame (in-memory working format)
```
Index | Tissue_ID | Cell_Type | X    | Y    | #F_neighbours | #F_neighbours_log2 | #M_neighbours | #M_neighbours_log2 | Time_Step
------|-----------|-----------|------|------|----------------|-------------------|----------------|-------------------|----------
0     | 0         | F         | 450  | 1200 | 8              | 3.00              | 3              | 1.58              | 100
1     | 0         | M         | 2100 | 900  | 2              | 1.00              | 12             | 3.58              | 100
...   | ...       | ...       | ...  | ...  | ...            | ...               | ...            | ...               | ...
```

### Sampled Data for Regression
```
Index | Tissue_ID | Cell_Type | X   | Y   | #F_neighbours | #M_neighbours | Division_Observation
------|-----------|-----------|-----|-----|----------------|----------------|--------------------
0     | 5         | F         | 800 | 600 | 14             | 8              | 1
1     | 5         | M         | 2200| 100 | 4              | 11             | 0
...   | ...       | ...       | ... | ... | ...            | ...            | ...
```

### Inferred Parameters
```python
pplus_f = {
    0: [intercept_1k, coef_1k],
    1: [intercept_5k, coef_5k],
    2: [intercept_10k, coef_10k],
    3: [intercept_25k, coef_25k]
}

pminus_f = {
    0: death_rate_1k,
    1: death_rate_5k,
    2: death_rate_10k,
    3: death_rate_25k
}
# Same structure for pplus_m and pminus_m
```

---

## 8. NOTABLE DESIGN DECISIONS

### Why Two Notebooks?
1. **validation_simulations.ipynb**: Uses logistic ODE form, less faithful to paper
2. **validation_simulations_alt.ipynb**: Uses logistic regression form, recommended

Activity log (Week 3, Friday):
> "Tried curve fitting on different versions of the model...results weren't very conclusive. I should probably first focus on model estimation to pinpoint what's the best way to define the model."

### Why Logistic Regression Over Simple Equation?
The OSDR method specifically uses logistic regression to infer `p+` from neighborhood data. The validation notebooks must:
1. Simulate tissue under **known** p+ formula
2. Sample cells and treat division as binary outcome
3. Use logistic regression to **recover** p+ parameters
4. Compare inferred vs. actual

This tests whether OSDR can correctly estimate the underlying model from noisy spatial data.

### Why Recompute Neighborhoods Every Step?
Cell populations are dynamic (divisions/deaths). Neighborhood membership changes constantly. Recomputing ensures:
- Accurate representation of current tissue state
- Next event probabilities use current neighbors (not stale data)

### Why Subsample Multiple Sizes?
- **1k samples:** Can model infer with very limited data?
- **5k, 10k:** Intermediate convergence
- **25k:** Near-asymptotic inference
- Tests **sample size effects on inference accuracy**

---

## 9. MISSING OR TODO ITEMS

From activity log and code inspection:

1. ✗ **Ki67 integration**: Not yet implemented (acknowledged in activity log)
2. ✗ **Edge correction**: No Kaplan-Meier or boundary-aware corrections
3. ✗ **inferred_portraits.py**: Empty (intended for future visualization)
4. ✗ **Real data application**: Validation notebooks are simulation-only
5. ✗ **Multi-feature regression**: Currently uses only same-type neighborhood; could include opposite type
6. ✗ **Time-dependent parameters**: Death rate `b` is constant; could be made neighborhood-dependent

---

## 10. SUMMARY TABLE: CODE LOCATIONS

| Functionality | Location | Key Functions |
|---|---|---|
| **Spatial simulation** | validation_simulations_alt.ipynb (Cell 27) | `simulate_model()`, `tissue_proliferation()`, `random_tissue()` |
| **Neighborhood computation** | Both notebooks (Cell 15/19 for proliferation) | cKDTree, query_ball_point |
| **Edge handling** | Both notebooks (lines 1338+) | Conditional resampling in `tissue_proliferation()` |
| **Ki67 processing** | None found | N/A - not implemented |
| **Logistic regression** | validation_simulations_alt.ipynb (Cell 44) | `tissue_regression()`, sklearn.LogisticRegression |
| **Phase portraits** | osdr_validation/phase_portrait_alt.py | `ODE_system()`, `plot_phase_portrait()` |
| **Parameter comparison** | validation_simulations_alt.ipynb (Cell 62) | `div_death_known()`, `div_death_inferred()`, `compare_likelihoods()` |
| **Utilities** | validation_simulations_alt.ipynb (Cell 41) | `cell_sampler()` |

