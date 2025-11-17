# OSDR Implementation Roadmap

**Project:** One-Shot tissue Dynamics Reconstruction (OSDR) for Triple Negative Breast Cancer (TNBC)
**Author:** Gabin Rousseau (MSc Bioinformatics)
**Last Updated:** 2025-11-17

---

## Executive Summary

This roadmap outlines the path from completed v1 validation simulations to:
1. Completing any remaining v1 implementation tasks
2. Applying OSDR v1 to the TNBC dataset
3. Implementing OSDR v2 (cell state transitions)

**Status Overview:**
- ✅ **V1 Validation (Simulations):** COMPLETE - Successfully validated OSDR 1.0 with synthetic data
- 🔄 **V1 Application (TNBC):** IN PROGRESS - Dataset explored, Ki67 processing needed
- 📋 **V2 Implementation:** PLANNED - Based on Shalom et al. 2025 preprint

**Key Documents:**
- This roadmap: High-level implementation plan and timeline
- `OSDR_GAP_ANALYSIS.md`: Detailed gap analysis comparing implementation to Somer et al. 2024 paper
- `OSDR_WORKFLOW_ANALYSIS.md`: Technical analysis of current v1 implementation
- `activitylog.md`: Weekly progress log (through Week 10)

---

## Phase 1: Complete V1 Implementation

### 1.1 Ki67 Processing Pipeline 🔴 CRITICAL
**Status:** NOT IMPLEMENTED
**Priority:** HIGH (Critical blocker for real data)
**Reference:** OSDR_GAP_ANALYSIS.md Section 2, OSDR_WORKFLOW_ANALYSIS.md Section 3

The validation simulations used direct division probability without Ki67 normalization. Real data requires implementing the full Ki67 pipeline based on Uxa et al. (2021) as specified in Somer et al.

**Background:**
- Current validation uses **synthetic division observations** from known model
- Real TNBC dataset has `Ki-67` and `Ki-67_nuclear` columns (unused)
- Gap analysis identifies this as **critical blocker** for real data application

**Tasks:**
- [ ] **1.1.1** Review Ki67 marker distribution in TNBC dataset
  - File: `tnbc_exploration.ipynb` already has preliminary exploration
  - Columns: `Ki-67`, `Ki-67_nuclear`
  - Visualize distribution per cell type
  - Check for outliers and data quality issues

- [ ] **1.1.2** Implement Ki67 normalization function (following Somer et al.)
  - **Step 1 - Noise threshold:**
    - `Tn = 0.5 × mean(isotopic counts)`
    - Filters experimental noise in IMC data
  - **Step 2 - Normalization:**
    - Select Ki67 values > Tn
    - Subtract Tn
    - Divide by standard deviation
    - Creates comparable distributions across cell types
  - **Step 3 - Division threshold:**
    - Cell dividing if `normalized Ki67 > Td`
    - Test Td ∈ [0, 1] (results should be robust per paper)
  - **Step 4 - Division rate:**
    - `p+ = (# Ki67+ cells) / (total cells × dt)`
    - dt = 1 time unit (few hours, Ki67 marker persistence window)
  - Create module: `notebooks/osdr_validation/ki67_processing.py`

- [ ] **1.1.3** Validate Ki67 processing
  - Compare resulting division rates to literature values for breast cancer
  - Test threshold robustness (vary Td, check stability of results)
  - Compare distribution of inferred division probabilities with validation simulations
  - Visualize Ki67+ cells spatially to check biological plausibility

- [ ] **1.1.4** Create Ki67 processing notebook
  - Document threshold selection methodology
  - Show visualizations of Ki67 distribution, normalization, and thresholding
  - Provide example usage for TNBC dataset

**Deliverables:**
- `ki67_processing.py` module with all 4 processing steps
- Notebook: `notebooks/5_ki67_processing_pipeline.ipynb`
- Validation report comparing to literature values
- Spatial visualization of Ki67+ cells

**Estimated Time:** 1-2 weeks

**Code Template (from gap analysis):**
```python
def process_ki67(df, Td=0.5):
    # 1. Noise threshold
    Tn = 0.5 * df['Ki-67'].mean()

    # 2. Normalization
    ki67_filtered = df['Ki-67'][df['Ki-67'] > Tn]
    ki67_norm = (ki67_filtered - Tn) / ki67_filtered.std()

    # 3. Division classification
    division_label = (ki67_norm > Td).astype(int)

    return division_label
```

---

### 1.2 Edge Correction Implementation 🟡 IMPORTANT
**Status:** NOT IMPLEMENTED
**Priority:** MEDIUM for validation, HIGH for real data
**Reference:** OSDR_GAP_ANALYSIS.md Section 1.2, OSDR_WORKFLOW_ANALYSIS.md Section 2

**Background:**
- Current implementation: Hard boundaries with conditional resampling (prevents cells from leaving tissue)
- **No rescaling** of neighborhood counts for cells near edges
- Gap analysis assessment:
  - **Low impact for validation**: Large tissues (2500×2500 μm) vs small radius (80 μm)
  - **High impact for real data**: TNBC biopsies ~500×500 μm → significant edge effects
  - Edge cells systematically undercount neighbors → biased division probability estimates

**Paper Specification (Somer et al.):**
> "When cells near the tissue edge divide, daughter cells may be placed outside tissue bounds. We rescale cell counts for cells near the edge by the 'neighborhood fraction within the tissue'. This provides an unbiased estimator of neighborhood composition."

**Tasks:**
- [ ] **1.2.1** Assess current edge effect magnitude
  - Quantify what % of cells in validation simulations are near edges (within 80 μm)
  - Quantify what % would be near edges in typical TNBC FOV (~500 μm)
  - Visualize edge cells vs interior cells in existing data

- [ ] **1.2.2** Implement edge correction following Somer et al.
  - Calculate fraction of neighborhood circle within tissue bounds
  - For each cell, compute: `fraction_inside = area_of_circle_in_tissue / area_of_full_circle`
  - Rescale neighborhood counts: `corrected_count = observed_count / fraction_inside`
  - Integrate into neighborhood computation in `tissue_simulation.py`

- [ ] **1.2.3** Create edge correction utility module
  - Function: `calculate_edge_fraction(cell_x, cell_y, tissue_bounds, radius)`
  - Function: `correct_neighborhood_counts(tissue_df, tissue_bounds, radius)`
  - Add to `osdr_validation/neighborhood_utils.py` (new module)

- [ ] **1.2.4** Evaluate impact on inference
  - Re-run validation simulations with edge correction
  - Compare inferred parameters with/without correction
  - Quantify improvement in model fit (AUC, coefficient accuracy)
  - Document impact assessment

**Deliverables:**
- `osdr_validation/neighborhood_utils.py` with edge correction functions
- Comparison notebook: edge correction impact on validation
- Decision document: use correction for real data? (recommendation: YES for TNBC)

**Estimated Time:** 1-2 weeks

**Decision Point:** If impact is minimal for validation, **DEFER** to real data phase. If moderate/high impact, implement before finalizing v1 validation.

**Code Template (from gap analysis):**
```python
def edge_correction(cell_pos, tissue_bounds, r=80):
    """
    Calculate fraction of neighborhood circle within tissue bounds.
    Rescale cell counts by 1/fraction for unbiased density estimate.
    """
    # Calculate fraction of neighborhood circle within tissue
    # See Somer et al. methods for full algorithm
    # Return: corrected neighborhood fraction
    pass
```

---

### 1.3 Multivariate Logistic Regression 🟡 RECOMMENDED
**Status:** NOT IMPLEMENTED (currently univariate)
**Priority:** LOW for validation, MEDIUM for real data
**Reference:** OSDR_GAP_ANALYSIS.md Section 3.1

**Background:**
- Current implementation: **Univariate** regression (F cells use only #F_neighbours, M cells use only #M_neighbours)
- Paper specification: **Multivariate** regression with all cell types + interaction terms
- Gap analysis assessment:
  - **Low impact for validation**: Known model only depends on same-type neighbors (intentional simplification)
  - **Medium impact for real data**: May miss cross-type regulation (e.g., macrophage suppressing T cell division)

**Paper Specification (Somer et al.):**
```
p+(Ni(x)) = 1 / (1 + exp(-(β0 + Σt β_t × N_t(x))))
```
where N_t(x) = count of type t cells in neighborhood, for **all cell types t**

- Interaction terms added (selected via cross-validation)
- Feature selection via cross-validation

**Tasks:**
- [ ] **1.3.1** Implement multivariate regression option
  - Modify `model_inference.py` to accept feature configuration
  - Support both univariate (current) and multivariate modes
  - Example: `X_f = df[["#F_neighbours", "#M_neighbours"]]` (multivariate)

- [ ] **1.3.2** Add interaction terms
  - Compute pairwise interactions: `#F × #M`, `#F²`, `#M²`
  - Add to feature matrix

- [ ] **1.3.3** Implement cross-validation for feature selection
  - Use sklearn's RFE (Recursive Feature Elimination) or Lasso
  - Test multiple feature combinations
  - Select features that maximize cross-validated AUC

- [ ] **1.3.4** Compare univariate vs multivariate on validation data
  - Re-run inference with multivariate features
  - Compare model fit quality (AUC, p-values, coefficient stability)
  - Assess whether added complexity improves prediction

**Deliverables:**
- Updated `model_inference.py` with multivariate regression support
- Feature selection pipeline with cross-validation
- Comparison analysis: univariate vs multivariate

**Estimated Time:** 1 week

**Decision Point:**
- **For validation:** DEFER - univariate is appropriate for 2-cell symmetric model
- **For real data:** IMPLEMENT - likely important for capturing cross-type interactions in complex TNBC microenvironment

**Code Change (from gap analysis):**
```python
# Current (univariate):
X_f = df[["#F_neighbours"]].loc[df["Cell_Type"] == "F"]

# Proposed (multivariate):
X_f = df[["#F_neighbours", "#M_neighbours", "#F_neighbours * #M_neighbours"]].loc[df["Cell_Type"] == "F"]
# Add cross-validation for feature selection
```

---

### 1.4 Code Refactoring & Organization
**Status:** PARTIALLY COMPLETE
**Priority:** MEDIUM

The validation notebooks have been split into 4 focused modules (from git log), but additional organization may be needed.

**Tasks:**
- [ ] **1.3.1** Review current module structure
  - Check `notebooks/osdr_validation/` directory
  - Ensure clean separation of concerns
  - Verify all reusable code is modularized

- [ ] **1.3.2** Complete `inferred_portraits.py`
  - Currently empty stub file (OSDR_WORKFLOW_ANALYSIS.md Section 5)
  - Should contain functions for plotting phase portraits from inferred parameters
  - Extract relevant code from validation notebooks

- [ ] **1.3.3** Create comprehensive test suite
  - Unit tests for key functions:
    - Neighborhood computation
    - Division/death probability calculation
    - Logistic regression fitting
  - Integration tests for full pipeline

- [ ] **1.3.4** Documentation
  - Add docstrings to all functions
  - Create module-level documentation
  - Update README with module descriptions

**Deliverables:**
- Complete `inferred_portraits.py` module
- Test suite with >80% code coverage
- Updated documentation

**Estimated Time:** 1 week

---

## Phase 2: Apply OSDR V1 to TNBC Dataset

### 2.1 Data Preparation
**Status:** EXPLORATION COMPLETE, PROCESSING PENDING
**Priority:** HIGH

**Tasks:**
- [ ] **2.1.1** Load and validate TNBC dataset
  - File: `data/cell_table_bothneighbourhoods.csv` (symbolic link)
  - Verify all required columns present:
    - Spatial coordinates (X, Y)
    - Cell types
    - Ki67 markers
    - Patient/FOV identifiers
  - Check data quality (missing values, outliers)

- [ ] **2.1.2** Preprocess spatial data
  - Convert coordinates to consistent units (microns)
  - Verify tissue boundaries
  - Identify and handle edge cases (cells at FOV boundaries)

- [ ] **2.1.3** Cell type annotation
  - Verify cell type classifications
  - Document cell type mapping
  - Ensure consistency with validation simulation cell types

- [ ] **2.1.4** Patient/FOV stratification
  - Organize data by patient
  - Organize by field of view (FOV)
  - Plan analysis strategy (per-patient, pooled, etc.)

**Deliverables:**
- Data preprocessing notebook: `notebooks/5_tnbc_data_preparation.ipynb`
- Clean, validated dataset ready for OSDR analysis
- Data quality report

**Estimated Time:** 1 week

---

### 2.2 Neighborhood Computation for TNBC Data
**Status:** PENDING
**Priority:** HIGH

**Tasks:**
- [ ] **2.2.1** Adapt neighborhood computation for real data
  - Use same radius: r = 80 microns (consistent with Somer et al.)
  - Apply cKDTree method from validation code
  - Handle multi-FOV data (compute neighborhoods within FOVs only)

- [ ] **2.2.2** Compute neighborhood features
  - For each cell, count neighbors of each type
  - Log2 transform neighborhood counts
  - Store in analysis-ready format

- [ ] **2.2.3** Apply edge correction (if implemented in 1.2)
  - Correct for FOV boundaries
  - Document correction methodology

- [ ] **2.2.4** Visualize neighborhood distributions
  - Recreate plots similar to validation simulations
  - Compare real data neighborhoods to simulation expectations
  - Identify any unexpected patterns

**Deliverables:**
- Neighborhood computation applied to full TNBC dataset
- Visualization notebook comparing real vs simulated neighborhoods
- Analysis-ready dataset with neighborhood features

**Estimated Time:** 1 week

---

### 2.3 Model Inference on TNBC Data
**Status:** PENDING
**Priority:** HIGH

**Tasks:**
- [ ] **2.3.1** Apply Ki67 processing (from 1.1)
  - Convert Ki67 markers to division observations
  - Create binary division labels for logistic regression

- [ ] **2.3.2** Fit logistic regression models
  - Per-cell-type division probability models
  - Features: neighborhood composition (cell type counts)
  - Labels: division observations (from Ki67)
  - Use sklearn.LogisticRegression (consistent with validation)

- [ ] **2.3.3** Estimate death rates
  - Calculate p⁻ (removal rate) per cell type
  - Compare to validation simulation parameters

- [ ] **2.3.4** Evaluate model quality
  - Check regression coefficients for biological plausibility
  - Compare AUC, accuracy, precision/recall
  - Assess convergence of fits

- [ ] **2.3.5** Multi-patient analysis
  - Fit models per patient or pooled across patients
  - Compare parameter consistency across patients
  - Identify patient-specific vs universal dynamics

**Deliverables:**
- Inferred OSDR v1 parameters for TNBC dataset
- Model quality metrics and validation
- Patient-stratified analysis

**Estimated Time:** 2 weeks

---

### 2.4 Phase Portrait Analysis for TNBC
**Status:** PENDING
**Priority:** HIGH

**Tasks:**
- [ ] **2.4.1** Generate phase portraits from TNBC-inferred parameters
  - Use `phase_portrait_alt.py` and `inferred_portraits.py`
  - Create ODE system from inferred coefficients
  - Compute nullclines and fixed points

- [ ] **2.4.2** Stability analysis
  - Jacobian-based stability classification
  - Identify stable/unstable equilibria
  - Compare to validation simulation phase portraits

- [ ] **2.4.3** Biological interpretation
  - Map fixed points to biological steady states
  - Identify feedback loops and cell-cell interactions
  - Compare to literature on TNBC tumor microenvironment

- [ ] **2.4.4** Visualization
  - Multi-panel phase portraits (per patient or cell type pair)
  - Overlay streamlines showing trajectory flow
  - Annotate fixed points and stability

**Deliverables:**
- Phase portrait figures for TNBC data
- Biological interpretation document
- Comparison with Somer et al. results

**Estimated Time:** 1-2 weeks

---

### 2.5 Temporal Simulations
**Status:** PENDING
**Priority:** MEDIUM

**Tasks:**
- [ ] **2.5.1** Simulate TNBC tissue dynamics forward in time
  - Starting condition: actual TNBC spatial snapshot
  - Use inferred p⁺ and p⁻ parameters
  - Run simulations for clinically relevant timescales (weeks to months)

- [ ] **2.5.2** Validate simulations
  - Check that simulations reach expected equilibria
  - Verify biological plausibility of trajectories
  - Compare cell type proportions over time

- [ ] **2.5.3** Sensitivity analysis
  - Vary parameters within confidence intervals
  - Assess robustness of predictions
  - Identify key parameters driving dynamics

**Deliverables:**
- Temporal simulation code for TNBC
- Trajectory visualizations
- Sensitivity analysis report

**Estimated Time:** 1-2 weeks

---

### 2.6 Clinical Relevance Analysis
**Status:** PENDING
**Priority:** HIGH

**Tasks:**
- [ ] **2.6.1** Correlate dynamics with patient outcomes
  - Link inferred parameters to clinical metadata (if available)
  - Identify prognostic signatures in phase portraits
  - Compare dynamics in responders vs non-responders (if data available)

- [ ] **2.6.2** Tumor microenvironment characterization
  - Identify dominant cell-cell interactions
  - Map feedback circuits (e.g., macrophage-fibroblast, T-B cell)
  - Compare to known TNBC biology

- [ ] **2.6.3** Treatment implications
  - Simulate response to perturbations (e.g., removing cell types)
  - Predict impact of immunotherapy on dynamics
  - Generate hypotheses for therapeutic intervention

**Deliverables:**
- Clinical correlation analysis
- TME characterization report
- Treatment prediction simulations

**Estimated Time:** 2 weeks

---

## Phase 3: Implement OSDR V2 (Cell State Transitions)

**Reference:** Shalom et al. 2025 preprint (file: `2025.09.02.673328.Shalom-et-al.xml`)

### 3.1 Understand OSDR 2.0 Methodology
**Status:** PENDING
**Priority:** HIGH

**Tasks:**
- [ ] **3.1.1** Read and annotate Shalom et al. 2025 preprint
  - Focus on methods section
  - Extract mathematical formulations
  - Identify algorithmic differences from v1

- [ ] **3.1.2** Identify key innovations in v2
  - Cell state transition modeling
  - Separation of timescales (state transitions vs population changes)
  - Logistic regression for state probabilities
  - Integration of state transitions into simulations

- [ ] **3.1.3** Map v2 requirements to TNBC dataset
  - Identify cell types with multiple states:
    - T cells: PD1⁺ vs PD1⁻
    - Fibroblasts: CAF vs resting
  - Check if required markers are available in dataset
  - Plan marker selection strategy

- [ ] **3.1.4** Review v2 code (if available)
  - Check if authors released code for OSDR 2.0
  - Adapt to local codebase if available
  - Otherwise, implement from scratch following methods

**Deliverables:**
- Annotated preprint with implementation notes
- Gap analysis (v1 → v2)
- TNBC dataset compatibility assessment

**Estimated Time:** 1 week

---

### 3.2 Cell State Definition and Annotation
**Status:** PENDING
**Priority:** HIGH

**Tasks:**
- [ ] **3.2.1** Define cell states for TNBC analysis
  - **T cells:**
    - PD1⁺ (exhausted) vs PD1⁻ (functional)
    - Check for PD1/PD-L1 markers in dataset
  - **Fibroblasts:**
    - CAF (cancer-associated) vs resting
    - Identify CAF markers (α-SMA, FAP, etc.)
  - **Other cell types:** Identify additional state transitions if relevant

- [ ] **3.2.2** Implement cell state classification
  - Threshold-based classification from marker expression
  - Validate classifications against literature
  - Handle ambiguous/intermediate states

- [ ] **3.2.3** Annotate TNBC dataset with cell states
  - Add state labels to each cell
  - Compute state proportions per patient/FOV
  - Visualize spatial distribution of states

**Deliverables:**
- Cell state classification pipeline
- TNBC dataset with state annotations
- State distribution analysis

**Estimated Time:** 1-2 weeks

---

### 3.3 State Transition Probability Modeling
**Status:** PENDING
**Priority:** HIGH

**Tasks:**
- [ ] **3.3.1** Implement logistic regression for state probabilities
  - Following Shalom et al. methodology:
    - P(state α | neighborhood composition)
    - Features: counts of cell TYPES in neighborhood (not states)
    - Labels: cell states
  - Per-cell-type models (e.g., T cell state model, fibroblast state model)

- [ ] **3.3.2** Train state probability models on TNBC data
  - Fit models for each cell type with multiple states
  - Extract regression coefficients
  - Interpret which neighborhood cell types drive state transitions

- [ ] **3.3.3** Validate state probability models
  - Check model fit quality (AUC, p-values)
  - Assess biological plausibility of coefficients
  - Compare to shuffled control (as in Shalom et al. Fig 2)

- [ ] **3.3.4** Visualize state-neighborhood relationships
  - Recreate plots similar to Shalom et al. Fig 2:
    - Predicted vs observed state probabilities
    - Regression coefficient plots
  - Identify key neighborhood drivers of each state

**Deliverables:**
- State probability models for TNBC
- Model validation metrics
- Visualization of state-neighborhood associations

**Estimated Time:** 2 weeks

---

### 3.4 Integrate State Transitions into Simulations
**Status:** PENDING
**Priority:** HIGH

**Tasks:**
- [ ] **3.4.1** Implement OSDR 2.0 simulation algorithm
  - Following Shalom et al. algorithm:
    1. Initialize cells with types and states in space
    2. **Population dynamics step:**
       - Evaluate division and removal based on neighborhood
       - Update cell populations
    3. **State dynamics step (quasi-steady-state):**
       - Recompute neighborhoods after population update
       - Adjust cell states according to new neighborhood probabilities
    4. Repeat for temporal trajectory

- [ ] **3.4.2** Implement separation of timescales
  - Ensure state transitions occur after each population update
  - Validate that timescales are appropriately separated
  - Document assumptions and limitations

- [ ] **3.4.3** Test simulation implementation
  - Run on synthetic data first (to verify correctness)
  - Compare to OSDR 1.0 simulations (state transitions should add new dynamics)
  - Check for numerical stability

**Deliverables:**
- OSDR 2.0 simulation code
- Validation on synthetic data
- Comparison with v1 simulations

**Estimated Time:** 2-3 weeks

---

### 3.5 Apply OSDR 2.0 to TNBC Data
**Status:** PENDING
**Priority:** HIGH

**Tasks:**
- [ ] **3.5.1** Run OSDR 2.0 simulations on TNBC dataset
  - Starting condition: TNBC spatial snapshot with states
  - Use inferred division/removal rates (from v1)
  - Use inferred state transition probabilities (from 3.3)
  - Simulate forward in time

- [ ] **3.5.2** Track population and state dynamics
  - Record cell type abundances over time
  - Record state proportions within each cell type over time
  - Visualize trajectories

- [ ] **3.5.3** Compare v1 vs v2 predictions
  - Do state transitions alter population dynamics significantly?
  - Identify feedback between states and populations
  - Assess added predictive value of v2

**Deliverables:**
- OSDR 2.0 simulation results for TNBC
- Population + state trajectory plots
- V1 vs V2 comparison analysis

**Estimated Time:** 1-2 weeks

---

### 3.6 Treatment Response Prediction (if data available)
**Status:** PENDING
**Priority:** HIGH (if treatment data available)

**Tasks:**
- [ ] **3.6.1** Assess treatment response data availability
  - Check if TNBC dataset includes:
    - Pre/post-treatment biopsies
    - Treatment type (chemotherapy, immunotherapy)
    - Response labels (responder/non-responder)
  - If not available: consider obtaining Wang et al. 2023 dataset (used in Shalom et al.)

- [ ] **3.6.2** Train OSDR 2.0 on treatment cohorts
  - Separate responders vs non-responders
  - Fit separate models for each group
  - Compare inferred dynamics between groups

- [ ] **3.6.3** Simulate treatment response
  - Starting from early post-treatment biopsy
  - Simulate dynamics forward to final timepoint
  - Compare predicted vs observed outcomes

- [ ] **3.6.4** Evaluate predictive accuracy
  - Recreate analysis from Shalom et al. (Fig 4-5)
  - Compute classification metrics (AUC, accuracy, etc.)
  - Compare OSDR 2.0 vs OSDR 1.0 predictive power

**Deliverables:**
- Treatment response prediction analysis
- Comparison of v1 vs v2 for clinical prediction
- Figures demonstrating predictive accuracy

**Estimated Time:** 2-3 weeks (dependent on data availability)

---

### 3.7 Advanced OSDR 2.0 Features (Optional)
**Status:** PENDING
**Priority:** LOW (future work)

**Tasks:**
- [ ] **3.7.1** Multi-state transitions
  - Extend beyond binary states (e.g., multiple T cell states)
  - Implement multi-class logistic regression

- [ ] **3.7.2** Spatial heterogeneity analysis
  - Identify spatial domains with distinct dynamics
  - Cluster regions by inferred parameters
  - Link spatial structure to function

- [ ] **3.7.3** Perturbation simulations
  - Simulate therapeutic interventions:
    - PD1 blockade (modulate T cell states)
    - CAF targeting (modulate fibroblast states)
  - Predict optimal intervention strategies

**Deliverables:**
- Extended OSDR 2.0 capabilities
- Perturbation prediction framework

**Estimated Time:** 3-4 weeks

---

## Phase 4: Validation, Documentation & Dissemination

### 4.1 Comprehensive Validation
**Status:** PENDING
**Priority:** HIGH

**Tasks:**
- [ ] **4.1.1** Cross-validation
  - Split TNBC dataset into train/test sets
  - Validate model generalization
  - Report cross-validated performance metrics

- [ ] **4.1.2** Biological validation
  - Compare inferred parameters to literature values
  - Consult with domain experts on plausibility
  - Identify novel predictions for experimental validation

- [ ] **4.1.3** Computational validation
  - Verify numerical stability of simulations
  - Check parameter sensitivity
  - Ensure reproducibility (set random seeds, document versions)

**Deliverables:**
- Validation report
- Cross-validation metrics
- Sensitivity analysis

**Estimated Time:** 1-2 weeks

---

### 4.2 Code Quality & Reproducibility
**Status:** PENDING
**Priority:** HIGH

**Tasks:**
- [ ] **4.2.1** Finalize code organization
  - Clean up all notebooks
  - Ensure modular structure
  - Remove dead code and deprecated functions

- [ ] **4.2.2** Documentation
  - Complete docstrings for all functions
  - Create user guide for running analysis
  - Document parameter choices and assumptions

- [ ] **4.2.3** Reproducibility
  - Update `environment.yml` with all dependencies
  - Create example notebooks demonstrating full pipeline
  - Add data provenance tracking

- [ ] **4.2.4** Testing
  - Expand test suite
  - Add integration tests for full pipeline
  - Set up continuous integration (optional)

**Deliverables:**
- Clean, well-documented codebase
- Reproducible analysis pipeline
- Test suite with high coverage

**Estimated Time:** 1-2 weeks

---

### 4.3 Dissertation & Publication
**Status:** PENDING
**Priority:** HIGH

**Tasks:**
- [ ] **4.3.1** Dissertation writing
  - Introduction: OSDR background, TNBC biology
  - Methods: v1 and v2 implementation details
  - Results: validation, TNBC analysis, treatment prediction
  - Discussion: biological insights, limitations, future work
  - Use Quarto for writing (already set up per activity log)

- [ ] **4.3.2** Figure preparation
  - High-quality figures for all key results
  - Multi-panel figures following publication standards
  - Figure legends with comprehensive descriptions

- [ ] **4.3.3** Supplementary materials
  - Supplementary figures and tables
  - Code availability statement
  - Data availability statement

- [ ] **4.3.4** Publication preparation (optional, post-MSc)
  - Identify target journal
  - Adapt dissertation to manuscript format
  - Coordinate with supervisor on authorship

**Deliverables:**
- Complete MSc dissertation
- Publication-ready figures
- (Optional) Manuscript draft

**Estimated Time:** 4-6 weeks

---

## Timeline & Milestones

### Estimated Total Duration: 18-24 weeks

**Milestone 1: V1 Complete** (Weeks 1-4)
- Ki67 processing implemented
- Edge correction evaluated
- Code refactored and tested

**Milestone 2: TNBC Analysis with V1** (Weeks 5-10)
- TNBC data processed
- OSDR v1 applied to TNBC
- Phase portraits and simulations complete
- Clinical interpretation documented

**Milestone 3: V2 Implementation** (Weeks 11-16)
- OSDR 2.0 methodology understood
- Cell states defined and annotated
- State transition models trained
- OSDR 2.0 simulations running

**Milestone 4: Treatment Prediction & Validation** (Weeks 17-20)
- Treatment response analysis (if data available)
- Comprehensive validation complete
- V1 vs V2 comparison finalized

**Milestone 5: Dissertation & Publication** (Weeks 21-24)
- Dissertation complete
- Code published and documented
- (Optional) Manuscript submitted

---

## Resource Requirements

### Computational Resources
- **Hardware:** Workstation with multi-core CPU (for parallelization)
- **Memory:** 16+ GB RAM (for large spatial datasets)
- **Storage:** 100+ GB (for simulation outputs)
- **Software:**
  - Python 3.12+
  - Conda environment (see `environment.yml`)
  - Jupyter Lab
  - Quarto (for dissertation)

### Data Requirements
- **Current:** TNBC spatial proteomics data (symbolic link in `data/`)
- **Additional (for v2):**
  - Treatment response data (check Wang et al. 2023 availability)
  - Additional state markers (PD1, CAF markers)

### External Support
- **Supervisor meetings:** Weekly (Tuesdays 1pm per activity log)
- **Domain expertise:** Consult on biological interpretation
- **Code review:** Peer review of critical implementations

---

## Risk Assessment & Mitigation

### High-Priority Risks

**Risk 1: Ki67 processing methodology unclear**
- **Impact:** Cannot apply v1 to real data
- **Mitigation:** Review Somer et al. supplementary methods; consult supervisor
- **Contingency:** Use multiple threshold strategies, compare results

**Risk 2: TNBC dataset lacks required markers for v2**
- **Impact:** Cannot implement full OSDR 2.0
- **Mitigation:** Early assessment of available markers (Task 3.1.3)
- **Contingency:** Obtain Wang et al. 2023 dataset; or focus on v1 with deeper analysis

**Risk 3: Treatment response data unavailable**
- **Impact:** Cannot validate v2 clinical predictions
- **Mitigation:** Contact Wang et al. for data access
- **Contingency:** Focus on state dynamics characterization without treatment prediction

**Risk 4: V2 implementation more complex than expected**
- **Impact:** Timeline overruns, incomplete v2
- **Mitigation:** Phased implementation, early prototyping
- **Contingency:** Prioritize core v2 features, defer advanced features to future work

### Medium-Priority Risks

**Risk 5: Edge correction significantly impacts results**
- **Impact:** Need to revisit v1 validation simulations
- **Mitigation:** Early evaluation of edge correction impact (Task 1.2.3)
- **Contingency:** Document limitation; apply correction to v2 only

**Risk 6: Computational performance issues with large simulations**
- **Impact:** Slow iteration, difficulty with sensitivity analysis
- **Mitigation:** Leverage existing parallelization; optimize code
- **Contingency:** Subsample data for exploration; run full simulations overnight

---

## Open Questions & Decision Points

### Technical Questions
1. **Ki67 threshold:** What is the optimal threshold for binary division classification?
   - Decision by: End of Phase 1.1
   - Method: Compare multiple thresholds, validate against literature

2. **Edge correction:** Should we apply edge correction? Does it improve results?
   - Decision by: End of Phase 1.2
   - Method: Quantitative comparison of model fit with/without correction

3. **Cell state definitions:** How should we define CAF vs resting fibroblasts in our data?
   - Decision by: End of Phase 3.2
   - Method: Literature review + marker expression analysis

4. **Simulation timescale:** What is the appropriate simulation duration for TNBC?
   - Decision by: During Phase 2.5
   - Method: Match to clinical timescales (weeks to months)

### Strategic Questions
1. **V1 vs V2 focus:** How much effort to invest in v1 vs v2?
   - Recommendation: Complete v1 thoroughly before v2; v1 is foundation
   - Flexibility: If v2 data unavailable, deepen v1 analysis

2. **Publication strategy:** Aim for publication during MSc or after?
   - Decision by: After Phase 2 (TNBC v1 results)
   - Depends on: Result quality, timeline, supervisor input

3. **Dataset scope:** Focus on single dataset or analyze multiple?
   - Current: TNBC dataset primary
   - Opportunity: If time permits, apply to other cancer types for generalization

---

## References

### Key Papers
1. **Somer et al. 2024** - OSDR 1.0 methodology (preprint in repo)
2. **Shalom et al. 2025** - OSDR 2.0 with state transitions (preprint in repo: `2025.09.02.673328.Shalom-et-al.xml`)
3. **Wang et al. 2023** - TNBC spatial proteomics with treatment response (potential data source)

### Internal Documentation
- `OSDR_WORKFLOW_ANALYSIS.md` - Detailed analysis of v1 implementation
- `activitylog.md` - Weekly progress log (through Week 10)
- `notebooks/validation_simulations_alt.ipynb` - V1 validation reference
- `notebooks/tnbc_exploration.ipynb` - TNBC dataset exploration

---

## Version History

- **v1.0** (2025-11-17): Initial roadmap created, covering v1 completion, TNBC application, and v2 implementation

---

## Notes

This roadmap is a living document and should be updated as:
- Tasks are completed
- New challenges emerge
- Priorities shift
- New opportunities arise

Review and update weekly during supervisor meetings.
