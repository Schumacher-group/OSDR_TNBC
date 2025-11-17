"""
Visualization utilities for OSDR validation.

Functions for plotting phase portraits, tissue statistics, and model fit comparisons.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.optimize import fsolve
import autograd.numpy as anp
from autograd import jacobian

# Model parameters
A = -0.120
INTERCEPT = -2.456
B = 1 / (1 + np.exp(-(INTERCEPT + A * 16)))


def plot_average_neighbourhoods(initial_df, post_df):
    """
    Plot cell counts and average neighbourhood densities across tissues.
    
    Parameters
    ----------
    initial_df : pd.DataFrame
        Initial tissue states
    post_df : pd.DataFrame
        Post-proliferation tissue states
    """
    # Calculate the count of rows for each category
    count_i_df = pd.crosstab(initial_df['Tissue_ID'], initial_df['Cell_Type'])
    count_p_df = pd.crosstab(post_df['Tissue_ID'], post_df['Cell_Type'])

    # log2 adjustment for dfs
    log2_i_df = initial_df[np.isfinite(initial_df["#F_neighbours_log2"]) & 
                           np.isfinite(initial_df["#M_neighbours_log2"])]
    log2_p_df = post_df[np.isfinite(post_df["#F_neighbours_log2"]) & 
                        np.isfinite(post_df["#M_neighbours_log2"])]
    
    # Calculate the average of neighbourhood counts
    average_i_df = log2_i_df.groupby('Tissue_ID')[['#F_neighbours_log2', 
                                                     '#M_neighbours_log2']].mean().reset_index()
    average_p_df = log2_p_df.groupby('Tissue_ID')[['#F_neighbours_log2', 
                                                     '#M_neighbours_log2']].mean().reset_index()
    melted_ai_df = average_i_df.melt(id_vars='Tissue_ID', var_name='variable', value_name='average')
    melted_ap_df = average_p_df.melt(id_vars='Tissue_ID', var_name='variable', value_name='average')

    # Plot count I
    count_i_df.plot(kind='bar', stacked=True, figsize=(20, 8), colormap='viridis')
    plt.xlabel('Tissue_ID')
    plt.ylabel('Number of Cells')
    plt.title('Cell type count per initiated tissue')
    plt.legend(title='Type')
    plt.tight_layout()
    plt.show()

    # Plot count P
    count_p_df.plot(kind='bar', stacked=True, figsize=(20, 8), colormap='viridis')
    plt.xlabel('Tissue_ID')
    plt.ylabel('Number of Cells')
    plt.title('Cell type count per proliferated tissue')
    plt.legend(title='Type')
    plt.tight_layout()
    plt.show()

    # Plot average I
    plt.figure(figsize=(20, 8))
    sns.barplot(data=melted_ai_df, x='Tissue_ID', y='average', hue='variable')
    plt.xticks(rotation=90)
    plt.xlabel('Tissue_ID')
    plt.ylabel('Average log2 neighbour density')
    plt.title('Average of F and M log2 neighbour densities per tissue')
    plt.tight_layout()
    plt.show()

    # Plot average P
    plt.figure(figsize=(20, 8))
    sns.barplot(data=melted_ap_df, x='Tissue_ID', y='average', hue='variable')
    plt.xticks(rotation=90)
    plt.xlabel('Tissue_ID')
    plt.ylabel('Average log2 neighbour density')
    plt.title('Average of F and M log2 neighbour densities per tissue')
    plt.tight_layout()
    plt.show()


def neighbourhoods_over_time(initial_df, post_df, tissue_id=0):
    """
    Plot cell counts and neighbourhood densities over time for a single tissue.
    
    Parameters
    ----------
    initial_df : pd.DataFrame
        Initial tissue states
    post_df : pd.DataFrame
        Post-proliferation tissue states with Time_Step column
    tissue_id : int
        ID of tissue to plot
    """
    # Select tissue from dataframe
    initial_tissue = initial_df.loc[initial_df["Tissue_ID"] == tissue_id].copy()
    post_tissue = post_df.loc[post_df["Tissue_ID"] == tissue_id].copy()

    # log2 adjustment for dfs
    log2_i_df = initial_tissue[np.isfinite(initial_tissue["#F_neighbours_log2"]) & 
                               np.isfinite(initial_tissue["#M_neighbours_log2"])]
    log2_p_df = post_tissue[np.isfinite(post_tissue["#F_neighbours_log2"]) & 
                            np.isfinite(post_tissue["#M_neighbours_log2"])]

    # x axis container
    time_steps = [0]

    # Containers for counts and neighbourhood averages
    counts_F = []
    counts_M = []
    neighmean_F = []
    neighmean_M = []
    
    # at t=0
    counts_F.append((initial_tissue["Cell_Type"] == "F").sum())
    counts_M.append((initial_tissue["Cell_Type"] == "M").sum())
    neighmean_F.append((log2_i_df["#F_neighbours_log2"]).mean())
    neighmean_M.append((log2_i_df["#M_neighbours_log2"]).mean())

    # at all time steps
    for step in post_tissue["Time_Step"].unique():
        df = post_tissue.loc[post_tissue["Time_Step"] == step]
        log2_df = log2_p_df.loc[log2_p_df["Time_Step"] == step]

        time_steps.append(step)
        
        counts_F.append((df["Cell_Type"] == "F").sum())
        counts_M.append((df["Cell_Type"] == "M").sum())
        
        neighmean_F.append((log2_df["#F_neighbours_log2"]).mean())
        neighmean_M.append((log2_df["#M_neighbours_log2"]).mean())

    # Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Overall cell count
    ax1.plot(time_steps, counts_F, color='orange', linestyle='-', label='F')
    ax1.plot(time_steps, counts_M, color='blue', linestyle='-', label='M')
    ax1.set_title('Cell counts')
    ax1.set_xlabel('Time steps')
    ax1.set_ylabel('Count')
    ax1.legend()

    # Neighbourhood plot
    ax2.plot(time_steps, neighmean_F, color='green', linestyle='-', label='F neighbourhood content')
    ax2.plot(time_steps, neighmean_M, color='purple', linestyle='-', label='M neighbourhood content')
    ax2.set_title('Neighbourhood densities')
    ax2.set_xlabel('Time steps')
    ax2.set_ylabel('Average neighbourhood density (log2)')
    ax2.legend()
 
    # Global title
    fig.suptitle(f'Cell counts and average neighbourhood densities in tissue {tissue_id}', fontsize=16)

    # Display plots
    plt.tight_layout()
    plt.show()


def div_death_known(state, a=A, intercept=INTERCEPT, b=B):
    """Calculate p+ - p- for known model."""
    F, M = state

    pp_f = 1 / (1 + np.exp(-(intercept + a * F)))
    pm_f = b

    pp_m = 1 / (1 + np.exp(-(intercept + a * M)))
    pm_m = b
    
    return [pp_f - pm_f, pp_m - pm_m]


def div_death_inferred(state, params, t_id):
    """Calculate p+ - p- for inferred model."""
    F, M = state

    z_f = params["pplus_f"][t_id][0] + F * params["pplus_f"][t_id][1]
    pp_f = 1 / (1 + np.exp(-z_f))
    pm_f = params["pminus_f"][t_id]

    z_m = params["pplus_m"][t_id][0] + M * params["pplus_m"][t_id][1]
    pp_m = 1 / (1 + np.exp(-z_m))
    pm_m = params["pminus_m"][t_id]

    return [pp_f - pm_f, pp_m - pm_m]


def compare_likelihoods(dfs, params, t_id=0):
    """
    Compare known vs inferred model p+ - p- over neighbourhood densities.
    
    Parameters
    ----------
    dfs : dict
        Dictionary of sampled dataframes {0: k1_df, 1: k5_df, ...}
    params : dict
        Inferred model parameters
    t_id : int
        Sample size index (0=1k, 1=5k, 2=10k, 3=25k)
    """
    # get the chosen dataframe and separate F and M cell rows
    df = dfs[t_id]
    df_F = df.loc[df["Cell_Type"] == "F"]
    df_M = df.loc[df["Cell_Type"] == "M"]

    # counts of cell X in cell X's neighbourhood
    F_known = []
    F_infer = []

    M_known = []
    M_infer = []

    # likewise, but p+ minus p-
    divdeath_kf = []
    divdeath_if = []

    divdeath_km = []
    divdeath_im = []

    # fill known lists, F models then M models
    for i in range(len(df_F)):
        cell = df_F.iloc[i]
        count = int(cell["#F_neighbours"])
        F_known.append(count)
        divdeath_kf.append(div_death_known([count, count])[0])

    for i in range(len(df_M)):
        cell = df_M.iloc[i]
        count = int(cell["#M_neighbours"])
        M_known.append(count)
        divdeath_km.append(div_death_known([count, count])[1])

    # same with lists for inferred data
    for i in range(max(F_known) + 10):
        F_infer.append(i)
        divdeath_if.append(div_death_inferred([i, i], params, t_id)[0])
    
    for i in range(max(M_known) + 10):
        M_infer.append(i)
        divdeath_im.append(div_death_inferred([i, i], params, t_id)[1])
    
    # plotting 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # First plot using hexbin
    ax1.hexbin(F_known, divdeath_kf, gridsize=40, cmap='Blues', mincnt=1, alpha=1)
    ax1.plot(F_infer, divdeath_if, color='orange', linestyle='-', label='Inferred model')
    ax1.set_title('F models')
    ax1.set_xlabel('F')
    ax1.set_ylabel('Proliferation')
    ax1.legend()

    # Second plot using hexbin
    ax2.hexbin(M_known, divdeath_km, gridsize=40, cmap='Greens', mincnt=1, alpha=1)
    ax2.plot(M_infer, divdeath_im, color='purple', linestyle='-', label='Inferred model')
    ax2.set_title('M models')
    ax2.set_xlabel('M')
    ax2.set_ylabel('Proliferation')
    ax2.legend()
 
    # Global title
    fig.suptitle(f'Known vs Inference | p+ minus p- | df {t_id}', fontsize=16)

    # Display plots
    plt.tight_layout()
    plt.show()


def logistic_p(type, count, params, t_id):
    """Calculate logistic division probability for visualization."""
    z = params[f"pplus_{type}"][t_id][0] + count * params[f"pplus_{type}"][t_id][1]
    pp = 1 / (1 + np.exp(-z))
    return pp


def visualise_logfit(dfs, params, y_dict, t_id=0):
    """
    Visualize logistic regression fits with division observations.
    
    Parameters
    ----------
    dfs : dict
        Dictionary of sampled dataframes
    params : dict
        Inferred model parameters
    y_dict : dict
        Division observation labels
    t_id : int
        Sample size index
    """
    # get the chosen dataframe and separate F and M cell rows
    df = dfs[t_id].copy()
    df["Division_Observation"] = y_dict[t_id]
    
    df_F = df.loc[df["Cell_Type"] == "F"].copy()
    df_M = df.loc[df["Cell_Type"] == "M"].copy()

    # Prepare data points for logistic plots
    range_f = np.linspace(min(df_F['#F_neighbours']), max(df_F['#F_neighbours']), 200)
    range_m = np.linspace(min(df_M['#M_neighbours']), max(df_M['#M_neighbours']), 200)
    fitY_f = [logistic_p("f", x, params, t_id) for x in range_f]
    fitY_m = [logistic_p("m", x, params, t_id) for x in range_m]

    ### PLOTTING 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # Plots (normless) to extract counts
    hb1 = ax1.hexbin(df_F["#F_neighbours"], df_F["Division_Observation"], gridsize=40, 
                     cmap='magma', mincnt=1, alpha=1, label='Observations')
    hb2 = ax2.hexbin(df_M["#M_neighbours"], df_M["Division_Observation"], gridsize=40, 
                     cmap='inferno', mincnt=1, alpha=1, label='Observations')
    
    # Extract the count arrays
    counts1 = hb1.get_array()
    counts2 = hb2.get_array()

    # Calculate global min and max from both counts
    global_min = min(counts1.min(), counts2.min())
    global_max = max(counts1.max(), counts2.max())

    # Define shared normalization
    norm = Normalize(vmin=global_min, vmax=global_max)

    # Clear and replot with shared norm
    ax1.cla()
    ax2.cla()
    hb1 = ax1.hexbin(df_F["#F_neighbours"], df_F["Division_Observation"], gridsize=40, 
                     cmap='magma', mincnt=1, alpha=1, label='Observations', norm=norm)
    hb2 = ax2.hexbin(df_M["#M_neighbours"], df_M["Division_Observation"], gridsize=40, 
                     cmap='inferno', mincnt=1, alpha=1, label='Observations', norm=norm)                 
    
    # Create second axes for the regression
    ax1b = ax1.twinx()
    ax1b.plot(range_f, fitY_f, color='blue', linestyle='-', label='Inferred probabilities')

    ax2b = ax2.twinx()
    ax2b.plot(range_m, fitY_m, color='blue', linestyle='-', label='Inferred probabilities')

    # Global colorbar
    fig.colorbar(hb1, ax=[ax1, ax2], orientation='vertical', label="Observation counts")
    
    # Axes formatting
    ax1.set_title('F model')
    ax1.set_xlabel('F')
    ax1.set_ylabel('Division observation')
    ax1.set_yticks([0, 1])

    ax1b.tick_params(axis='y', labelcolor='blue')
    ax1b.set_ylim(min(fitY_f), max(fitY_f))
    ax1b.set_yticks(np.linspace(0, ax1b.get_ylim()[1], num=5))  

    ax2.set_title('M model')
    ax2.set_xlabel('M')
    ax2.set_yticks([0, 1])

    ax2b.set_ylabel('Inferred division probability', color="blue")
    ax2b.tick_params(axis='y', labelcolor='blue')
    ax2b.set_ylim(min(fitY_m), max(fitY_m))
    ax2b.set_yticks(np.linspace(0, ax2b.get_ylim()[1], num=5)) 

    fig.suptitle(f'OSDR Logistic fit | df {t_id}', fontsize=16)

    plt.show()


# Phase portrait functions for inferred models

def ODE_system(state, params, t_id):
    """ODE system for inferred model."""
    F, M = state

    z_f = params["pplus_f"][t_id][0] + F * params["pplus_f"][t_id][1]
    pp_f = 1 / (1 + np.exp(-z_f))
    pm_f = params["pminus_f"][t_id]

    z_m = params["pplus_m"][t_id][0] + M * params["pplus_m"][t_id][1]
    pp_m = 1 / (1 + np.exp(-z_m))
    pm_m = params["pminus_m"][t_id]

    dF_dt = F * (pp_f - pm_f)
    dM_dt = M * (pp_m - pm_m)
    return [dF_dt, dM_dt]


def ODE_system_np(state, params, t_id):
    """ODE system using autograd numpy."""
    F, M = state

    z_f = params["pplus_f"][t_id][0] + F * params["pplus_f"][t_id][1]
    pp_f = 1 / (1 + anp.exp(-z_f))
    pm_f = params["pminus_f"][t_id]

    z_m = params["pplus_m"][t_id][0] + M * params["pplus_m"][t_id][1]
    pp_m = 1 / (1 + anp.exp(-z_m))
    pm_m = params["pminus_m"][t_id]

    dF_dt = F * (pp_f - pm_f)
    dM_dt = M * (pp_m - pm_m)
    
    return anp.array([dF_dt, dM_dt])


def streamlines(exp_F, exp_M, params, t_id):
    """Generate rates for streamlines adapted to log2 scale."""
    F = 2**exp_F
    M = 2**exp_M
    dF_dt, dM_dt = ODE_system([F, M], params, t_id)
    return dF_dt, dM_dt


def find_fixed_points(params, t_id):
    """Find fixed points using fsolve."""
    expected = [[0, 0], [16, 0], [0, 16], [16, 16]]
    fixed_points = [fsolve(ODE_system, expectation, args=(params, t_id)) 
                    for expectation in expected]
    return fixed_points


def compute_jacobian(params, t_id):
    """Compute Jacobian for stability analysis."""
    system_func = lambda fp: ODE_system_np(fp, params, t_id)
    return jacobian(system_func)


def is_stable(fp, params, t_id):
    """Check if fixed point is stable."""
    jac = compute_jacobian(params, t_id)
    J = jac(fp)
    eigenvalues = anp.linalg.eigvals(J)
    return all(e.real < 0 for e in eigenvalues)


def is_unstable(fp, params, t_id):
    """Check if fixed point is unstable."""
    jac = compute_jacobian(params, t_id)
    J = jac(fp)
    eigenvalues = anp.linalg.eigvals(J)
    return any(e.real > 0 for e in eigenvalues)


def plot_inferred_portrait(params, t_id=0, srange=None):
    """
    Plot phase portrait for inferred model.
    
    Parameters
    ----------
    params : dict or pd.DataFrame from tissue_regression
        If srange is provided, params should be post_df for recomputing
    t_id : int
        Sample size index (0=1k, 1=5k, 2=10k, 3=25k)
    srange : list of int or None
        If provided, overlays fixed points from multiple seeds
    """
    from .model_inference import tissue_regression
    
    if srange is None:
        srange = [0]
        
    # Get parameters for first seed (for streamlines)
    if not isinstance(params, dict):
        # params is post_df, need to compute
        pplus_f, pplus_m, pminus_f, pminus_m = tissue_regression(params, 1000, seed=srange[0])
        params_dict = {"pplus_f": pplus_f, "pplus_m": pplus_m, 
                       "pminus_f": pminus_f, "pminus_m": pminus_m}
    else:
        params_dict = params
     
    # Parameters for meshgrid
    exp_F_mesh = np.linspace(0, 8, 30)
    exp_M_mesh = np.linspace(0, 8, 30)
    exp_F, exp_M = np.meshgrid(exp_F_mesh, exp_M_mesh)

    # Calculate the growth rates for the streamlines
    F_rate, M_rate = streamlines(exp_F, exp_M, params_dict, t_id)
    F_rate_scaled = F_rate / (2**exp_F)
    M_rate_scaled = M_rate / (2**exp_M)
    
    plt.figure()

    # Streamplot
    plt.streamplot(exp_F, exp_M, F_rate_scaled, M_rate_scaled,
                   color="black", linewidth=0.5)

    label_added = {'Stable': False, 'Unstable': False, 'Semi-stable': False}
    
    for s in srange:
        # Get params for this seed
        if not isinstance(params, dict):
            pplus_f, pplus_m, pminus_f, pminus_m = tissue_regression(params, 1000, seed=s)
            params_s = {"pplus_f": pplus_f, "pplus_m": pplus_m, 
                        "pminus_f": pminus_f, "pminus_m": pminus_m}
        else:
            params_s = params_dict
        
        # Fixed points
        fixed_points = find_fixed_points(params_s, t_id)
        print(fixed_points)
        
        for fp in fixed_points:
            x = (np.log2(fp[0])) if fp[0] != 0 else fp[0]
            y = (np.log2(fp[1])) if fp[1] != 0 else fp[1]
            
            stability = ''
            if is_stable(fp, params_s, t_id):
                stability = 'Stable'
            elif is_unstable(fp, params_s, t_id):
                stability = 'Unstable'
            else:
                stability = 'Semi-stable'
            print(stability)
            
            fcolor = 'black' if stability == 'Stable' or stability == 'Semi-stable' else 'white'
            ecolor = 'black' if stability == 'Stable' or stability == 'Unstable' else 'red'
            
            if label_added[stability] == False:
                plt.scatter(x, y, s=50, edgecolors=ecolor, facecolors=fcolor, 
                            label=f'{stability} Fixed Point', zorder=2)
                label_added[stability] = True
            else:
                plt.scatter(x, y, s=50, edgecolors=ecolor, facecolors=fcolor, zorder=2)

    # Labels and legend
    plt.xlabel('log2(F)')
    plt.ylabel('log2(M)')
    plt.xlim(0, 8)
    plt.ylim(0, 8)
    plt.legend(loc='upper left')
    plt.title('Phase Portrait')
    plt.grid(True)
    plt.show()
