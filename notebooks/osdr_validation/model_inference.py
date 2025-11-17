"""
Model inference utilities for OSDR validation.

Functions for sampling cells from simulated tissues and performing logistic
regression to infer tissue dynamics.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Model parameters (alternative model with logistic division)
A = -0.120
INTERCEPT = -2.456
B = 1 / (1 + np.exp(-(INTERCEPT + A * 16)))


def cell_sampler(post_df, t=None, seed=0):
    """
    Sample cells from post-proliferation tissue data.
    
    Creates a 50k cell parent sample (500 cells/tissue across 100 tissues),
    then draws 4 child samples without replacement (1k, 5k, 10k, 25k cells).
    
    Parameters
    ----------
    post_df : pd.DataFrame
        Post-proliferation tissue dataset
    t : int or None
        Time step to filter data. If None, uses all data
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple of pd.DataFrame
        (main_sample, k1_df, k5_df, k10_df, k25_df)
    """
    # isolate desired time step data
    if t is not None and type(t) == int:
        post_df = post_df.loc[post_df["Time_Step"] == t].copy()
        print(f"\nCell Sampler : Now processing data at t={t}.\n")
    
    # use a seeded number generator
    rng = np.random.default_rng(seed)

    sampled_cells = []

    # sample with replacement 50k cells evenly across tissues (500 cells/tissue)
    for tissue_id in range(0, 100):
        cell_ids = post_df.index[post_df["Tissue_ID"] == tissue_id].tolist()
        if len(cell_ids) == 0:
            continue
        startid = min(cell_ids)
        stopid = max(cell_ids)
        for i in range(0, 500):
            sampled_cells.append(int(rng.integers(startid, stopid + 1)))

    main_sample = post_df.loc[sampled_cells].copy()

    print(f"Done sampling cell data. {len(main_sample)} cells sampled.")

    # make smaller samples from the patchwork cell data
    idrange = range(0, len(main_sample))

    k1_ids = rng.choice(idrange, size=1000, replace=False)
    k5_ids = rng.choice(idrange, size=5000, replace=False)
    k10_ids = rng.choice(idrange, size=10000, replace=False)
    k25_ids = rng.choice(idrange, size=25000, replace=False)
    
    print("Done choosing sampled cells for child dataframes.")
    k1_df = main_sample.iloc[k1_ids].copy()
    print(f"Child 1: {len(k1_df)} cells.")
    k5_df = main_sample.iloc[k5_ids].copy()
    print(f"Child 2: {len(k5_df)} cells.")
    k10_df = main_sample.iloc[k10_ids].copy()
    print(f"Child 3: {len(k10_df)} cells.")
    k25_df = main_sample.iloc[k25_ids].copy()
    print(f"Child 4: {len(k25_df)} cells.")
    
    sample_dfs = (main_sample, k1_df, k5_df, k10_df, k25_df)
    print("Done generating child dataframes.")
    return sample_dfs


def tissue_regression(post_df, t=None, seed=0, broadcast=False, 
                      a=A, intercept=INTERCEPT, b=B):
    """
    Perform model inference using multivariate logistic regression.
    
    Given sampled tissue data, computes division observations using the known
    model, then fits logistic regression models to infer division probability
    parameters for each cell type.
    
    Parameters
    ----------
    post_df : pd.DataFrame
        Post-proliferation tissue dataset
    t : int or None
        Time step to filter data
    seed : int
        Random seed for reproducibility
    broadcast : bool
        If True, makes y_dict a global variable for visualization functions
    a, intercept, b : float
        Known model parameters
        
    Returns
    -------
    tuple of dict
        (pplus_f, pplus_m, pminus_f, pminus_m) - parameters for F and M cells
        Each dict has keys 0-3 corresponding to sample sizes (1k, 5k, 10k, 25k)
    """
    rng = np.random.default_rng(seed)

    main_sample, k1_df, k5_df, k10_df, k25_df = cell_sampler(post_df, t, seed)
    k_dfs = [k1_df, k5_df, k10_df, k25_df]

    if broadcast:  # make y_dict global for visualization functions
        global y_dict
    y_dict = {}
    
    for i, df in enumerate(k_dfs):
        y_dict[i] = []  # storing sample-specific labels
        for cell_id in range(0, len(df)):
            # current cell
            cell = df.iloc[cell_id]

            # event probabilities from known model
            if cell.iloc[1] == "F":
                p_div = 1 / (1 + np.exp(-(intercept + a * cell.iloc[4])))
            else:
                p_div = 1 / (1 + np.exp(-(intercept + a * cell.iloc[6])))
            p_death = b
            p_stay = 1 - p_div - p_death

            if p_stay < 0:
                print("Error with probability space")
                print(p_div, p_death, p_stay, p_div + p_death + p_stay)
                return
                
            # sample random event uniformly
            r1 = rng.uniform(0, 1)

            # register sampled event as a binary observation for division
            if r1 <= p_div:  # division
                y_dict[i].append(1)
            elif r1 <= p_div + p_death:  # death
                y_dict[i].append(0)
            elif r1 <= p_div + p_death + p_stay:  # nothing happens
                y_dict[i].append(0)
            else:
                print("Unexpected random number.")
                return
    
    print(f"Completed generation of label y dictionary, with {len(y_dict)} lists stored.")
    print(f"k1 div count: {y_dict[0].count(1)}, p+ all ~= {y_dict[0].count(1)/len(y_dict[0])}")
    print(f"k5 div count: {y_dict[1].count(1)}, p+ all ~= {y_dict[1].count(1)/len(y_dict[1])}")
    print(f"k10 div count: {y_dict[2].count(1)}, p+ all ~= {y_dict[2].count(1)/len(y_dict[2])}")
    print(f"k25 div count: {y_dict[3].count(1)}, p+ all ~= {y_dict[3].count(1)/len(y_dict[3])}")
    
    # update dfs with y
    for i, df in enumerate(k_dfs):
        df["Division_Observation"] = y_dict[i]

    # Set parameter dictionaries, one entry per k_df
    pplus_f = {}
    pplus_m = {}
    pminus_f = {}
    pminus_m = {}
    print("\n#####INFERENCE#####")

    for i, df in enumerate(k_dfs):
        # fit multivariate logistic regression model for each cell type
        X_f = df[["#F_neighbours"]].loc[df["Cell_Type"] == "F"].copy()
        y_f = df["Division_Observation"].loc[df["Cell_Type"] == "F"].copy()
        X_m = df[["#M_neighbours"]].loc[df["Cell_Type"] == "M"].copy()
        y_m = df["Division_Observation"].loc[df["Cell_Type"] == "M"].copy()
    
        model_f = LogisticRegression(random_state=seed).fit(X_f, y_f)
        model_m = LogisticRegression(random_state=seed).fit(X_m, y_m)

        # Get model parameters
        coefficients_f = model_f.coef_
        intercept_f = model_f.intercept_
        coefficients_m = model_m.coef_
        intercept_m = model_m.intercept_

        # Print out some summary values
        print(f"Coefficients of df {i} (F/M):", coefficients_f[0], "/", coefficients_m[0])
        print(f"Intercept of df {i} (F/M):", intercept_f, "/", intercept_m)

        X_testf = {"#F_neighbours": [16]}
        X_testf = pd.DataFrame(X_testf)
        z_f = intercept_f + 16 * coefficients_f[0][0]
        p_div_f = 1 / (1 + np.exp(-z_f))
        print(f"p+ at known steady state for F, df{i}: {model_f.predict_proba((X_testf))[0][1]}")
        print(f"...manual computation with logistic function gives: {p_div_f}\n")

        X_testm = {"#M_neighbours": [16]}
        X_testm = pd.DataFrame(X_testm)
        z_m = intercept_m + 16 * coefficients_m[0][0]
        p_div_m = 1 / (1 + np.exp(-z_m))
        print(f"p+ at known steady state for M, df{i}: {model_m.predict_proba((X_testm))[0][1]}")
        print(f"...manual computation with logistic function gives: {p_div_m}\n\n")

        # Store parameters
        pplus_f[i] = [intercept_f[0], coefficients_f[0][0]]
        pplus_m[i] = [intercept_m[0], coefficients_m[0][0]]
        pminus_f[i] = y_f.mean()
        pminus_m[i] = y_m.mean()

    print("\n\nReturning parameter dictionaries pplus_f, pplus_m, pminus_f, pminus_m.")
    print("Each corresponds to the dataset of origin, e.g. 0 for 1K cells sample, 1 for 5k")
    return pplus_f, pplus_m, pminus_f, pminus_m


def parameter_checker(post_df, srange, t=1000):
    """
    Check frequency of models with appropriate (negative) parameter signs.
    
    Negative regression coefficients indicate correct relationship between
    cell density and division probability.
    
    Parameters
    ----------
    post_df : pd.DataFrame
        Post-proliferation tissue dataset
    srange : list of int
        Range of random seeds to test
    t : int
        Time step to analyze
        
    Returns
    -------
    tuple
        (total_models, freq_good) where freq_good is list of frequencies
        for each sample size
    """
    total = 0
    gc_dict = {0: 0, 1: 0, 2: 0, 3: 0}

    for s in srange:
        print(f"Checking seed {s}")
        total += 1
        pplus_f, pplus_m, pminus_f, pminus_m = tissue_regression(post_df, t, seed=s)
        for sample in range(0, 4):
            if pplus_f[sample][1] < 0 and pplus_m[sample][1] < 0:
                gc_dict[sample] += 1
    
    freq_good = [round(gc_dict[i] / total, 2) for i in gc_dict.keys()]
    print("Checker completed, returning total and frequency of properly signed models per sample size")
    return total, freq_good
