# OSDR_TNBC AI Coding Agent Instructions

## Project Overview
This repository implements the One-Shot tissue Dynamics Reconstruction (OSDR) method for analyzing Triple Negative Breast Cancer (TNBC) biopsy data. The workflow validates OSDR by simulating tissues with known dynamical models, performing model inference, and comparing inferred vs. ground truth phase portraits.

## Architecture & Key Components

### Workflow: Ground Truth → Simulation → Inference → Validation
1. **Ground Truth Model** (`notebooks/osdr_validation/phase_portrait*.py`): Define ODE systems with known steady states
2. **Tissue Simulation** (notebooks): Generate synthetic tissues, simulate stochastic proliferation using multiprocessing
3. **Model Inference**: Sample cells, compute division/death events, fit logistic regression to infer dynamics
4. **Phase Portrait Comparison**: Plot streamlines, nullclines, and fixed points for validation

### Critical Mathematical Context
- **ODE Systems**: `dX/dt = X(p⁺(X) - p⁻(X))` where X is cell density (log₂ scale)
  - Two model variants: `phase_portrait.py` (simple logistic: `dX/dt = aX - bX²`) and `phase_portrait_alt.py` (logistic division: `p⁺ = 1/(1+exp(-(intercept+a*X)))`)
  - Steady state targeted at **X=16 cells** in neighbourhoods
- **Neighbourhood Dynamics**: Cells labeled by type (F/M), spatial neighbours computed via `scipy.spatial.cKDTree`
- **Phase Portraits**: Plot on log₂ scale, use `autograd` for Jacobian eigenvalue stability analysis

## Development Patterns

### Notebook-Centric Workflow
- Primary analysis in `notebooks/validation_simulations*.ipynb` and `validation_simulations_alt_1_ground_truth_model.ipynb`
- Refactor reusable functions to `notebooks/osdr_validation/*.py` modules
- Import pattern: `from osdr_validation.phase_portrait import plot_phase_portrait`

### Performance Optimizations
- **Parallelization**: Use `multiprocessing.Pool().starmap()` with seeded RNG for tissue simulations
  - Example: 100 tissues @ 1000 time steps takes ~2 hours (see `activitylog.md` Week 4)
- **Progress tracking**: Use `tqdm` for loops (multiprocessing complicates this)
- **Data structure**: Pandas DataFrames with columns like `cell_type`, `F_neighbours`, `M_neighbours`, `time_step`

### Model Inference Specifics
- Sample workflow: 50k cell parent sample → 4 child samples (1k, 5k, 10k, 25k cells)
- **Critical insight**: Sample size matters! 10k+ samples show better model convergence (Week 8 findings)
- **Division events**: Binary labels (0/1), computed via RNG with known model rates
- Use `sklearn.linear_model.LogisticRegression` to fit `p⁺` as function of neighbourhood densities
- Death rate `p⁻` is mean of non-division events

### Phase Portrait Plotting
- **Fixed points**: Use `scipy.optimize.fsolve` with targeted guesses (e.g., `[0,0], [16,0], [0,16], [16,16]`)
- **Stability**: Eigenvalues of Jacobian (all real < 0 → stable; any > 0 → unstable; otherwise semi-stable)
- **Streamlines**: Scale rates by `2**exp_X` for log₂ coordinates
- **Gotcha**: Avoid large lattice searches with fsolve—targeted guesses based on expected steady states work best

## Data & Environment

### Key Files
- `data/cell_table_bothneighbourhoods.csv`: Real TNBC biopsy data (loaded via soft link)
- `data/simulated_tissues*.csv`: Generated synthetic tissue states
- `environment.yml`: Conda environment with Python 3.13, scipy, pandas, matplotlib, seaborn, autograd, scikit-learn

### Remote Development Setup
- Workstation accessed via VS Code Remote-SSH
- Use `screen` for long-running processes (100-tissue simulations)
- Jupyter Lab runs remotely with port forwarding

## Common Pitfalls

1. **Log₂ Scale Confusion**: KDE plots on log-transformed data; all density interpretations are log-scale specific
2. **Inference Failures**: If no divisions predicted in sample, LogisticRegression fitting fails (try different RNG seed)
3. **Positive Coefficients**: Inferred models with positive regression coefficients produce reversed dynamics—only negative coefficients indicate correct cell density/division relationships
4. **Steady State Sampling**: Data too clustered near steady state impairs inference; record states at multiple time steps
5. **Time Column Indexing**: Use `.loc` with column labels after adding `time_step`, not `.iloc` (Week 7 note)

## Testing & Validation
- Visual inspection: Convergent streamlines toward central fixed point indicate good inference
- Quantitative: Compare inferred parameters across multiple RNG seeds to assess reliability
- Reference: Target S2H figure from Somer et al. for neighbourhood dynamics patterns

## Debugging Strategy
- Check `activitylog.md` for detailed problem-solving history (e.g., parallelization issues Week 3, model formulation Week 8-9)
- Plot `p⁺ - p⁻` vs. X with hexbins to visualize data density and model fit quality
- For phase portrait issues: Verify fixed points with `fsolve`, inspect Jacobian eigenvalues

## Key References
- Base code adapted from `https://github.com/Schumacher-group/cellcircuits`
- Method validation targets Figure S2H from Somer et al. (bioRxiv 2024.04.22.590503)
