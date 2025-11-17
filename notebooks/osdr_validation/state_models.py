"""
State probability models for OSDR v2.0.

Implements logistic regression models for cell state transitions as a function
of neighborhood composition.

Author: Based on Shalom et al. 2025 OSDR v2.0 methodology
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional


# =============================================================================
# STATE PROBABILITY REGRESSION (OSDR v2.0 CORE)
# =============================================================================

def fit_state_probability_model(df: pd.DataFrame,
                               cell_type: str,
                               state_column: str,
                               positive_state: str,
                               neighbor_columns: List[str],
                               random_state: int = 42) -> Tuple[LogisticRegression, Dict]:
    """
    Fit logistic regression model for state probability.

    P(state=positive | neighborhood) = σ(β0 + Σ β_i × N_i)

    Parameters
    ----------
    df : pd.DataFrame
        Cell data with neighborhoods and states classified
    cell_type : str
        Cell type to model (e.g., 'CD4_T', 'CD8_T', 'Fibroblast')
    state_column : str
        Column containing state labels (e.g., 'PD1_State', 'CAF_State')
    positive_state : str
        The "positive" state label (e.g., 'PD1+', 'CAF')
    neighbor_columns : list
        List of neighborhood count columns to use as features
        Example: ['#CD4_T_neighbours', '#CD8_T_neighbours', '#Macrophage_neighbours', ...]
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    model : LogisticRegression
        Fitted model
    stats_dict : dict
        Model statistics and diagnostics
    """
    # Filter to cells of interest
    cell_mask = df['Cell_Type'] == cell_type
    state_mask = df[state_column] != 'N/A'
    valid_mask = cell_mask & state_mask

    if valid_mask.sum() == 0:
        raise ValueError(f"No valid cells found for {cell_type} with {state_column}")

    # Prepare features and labels
    X = df.loc[valid_mask, neighbor_columns].values
    y = (df.loc[valid_mask, state_column] == positive_state).astype(int).values

    # Check class balance
    n_positive = y.sum()
    n_negative = len(y) - n_positive
    class_balance = min(n_positive, n_negative) / max(n_positive, n_negative)

    print(f"\n{cell_type} {state_column} Model:")
    print(f"  Training samples: {len(y)}")
    print(f"  {positive_state}: {n_positive} ({100*n_positive/len(y):.1f}%)")
    print(f"  Other: {n_negative} ({100*n_negative/len(y):.1f}%)")
    print(f"  Class balance: {class_balance:.3f}")

    # Fit model
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        solver='lbfgs'
    )
    model.fit(X, y)

    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

    # Model statistics
    stats_dict = {
        'cell_type': cell_type,
        'state_column': state_column,
        'positive_state': positive_state,
        'n_samples': len(y),
        'n_positive': n_positive,
        'n_negative': n_negative,
        'class_balance': class_balance,
        'intercept': model.intercept_[0],
        'coefficients': dict(zip(neighbor_columns, model.coef_[0])),
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'training_accuracy': model.score(X, y)
    }

    print(f"  Training accuracy: {stats_dict['training_accuracy']:.3f}")
    print(f"  Cross-val AUC: {stats_dict['cv_auc_mean']:.3f} ± {stats_dict['cv_auc_std']:.3f}")

    return model, stats_dict


def fit_all_state_models(df: pd.DataFrame,
                         neighbor_columns: List[str],
                         states_config: Optional[Dict] = None,
                         random_state: int = 42) -> Dict:
    """
    Fit state probability models for all available states.

    Parameters
    ----------
    df : pd.DataFrame
        Fully classified cell data
    neighbor_columns : list
        Neighborhood count columns
    states_config : dict, optional
        Configuration for which states to model. If None, uses defaults.
        Format: {
            'state_name': {
                'cell_types': ['CD4_T', 'CD8_T'],
                'state_column': 'PD1_State',
                'positive_state': 'PD1+'
            },
            ...
        }
    random_state : int
        Random seed

    Returns
    -------
    dict
        Dictionary of fitted models and statistics
    """
    if states_config is None:
        # Default configuration
        states_config = {
            'PD1_T': {
                'cell_types': ['CD4_T', 'CD8_T', 'T_other'],
                'state_column': 'PD1_State',
                'positive_state': 'PD1+'
            },
            'CAF': {
                'cell_types': ['Fibroblast'],
                'state_column': 'CAF_State',
                'positive_state': 'CAF'
            },
            'Cytotoxic': {
                'cell_types': ['CD8_T'],
                'state_column': 'Cytotoxic_State',
                'positive_state': 'Active'
            },
            'Macrophage_M2': {
                'cell_types': ['Macrophage'],
                'state_column': 'Macrophage_State',
                'positive_state': 'M2-like'
            },
            'Treg': {
                'cell_types': ['CD4_T'],
                'state_column': 'Treg_State',
                'positive_state': 'Treg'
            }
        }

    models = {}

    for state_name, config in states_config.items():
        print(f"\n{'='*70}")
        print(f"Fitting models for {state_name}")
        print(f"{'='*70}")

        models[state_name] = {}

        for cell_type in config['cell_types']:
            try:
                model, stats = fit_state_probability_model(
                    df,
                    cell_type=cell_type,
                    state_column=config['state_column'],
                    positive_state=config['positive_state'],
                    neighbor_columns=neighbor_columns,
                    random_state=random_state
                )

                models[state_name][cell_type] = {
                    'model': model,
                    'stats': stats
                }

            except ValueError as e:
                print(f"  Skipping {cell_type}: {e}")
                continue

    return models


# =============================================================================
# MODEL VALIDATION AND COMPARISON
# =============================================================================

def likelihood_ratio_test(df: pd.DataFrame,
                         model: LogisticRegression,
                         X: np.ndarray,
                         y: np.ndarray,
                         n_permutations: int = 100) -> Dict:
    """
    Likelihood ratio test comparing real model vs shuffled control.

    Tests whether state-neighborhood association is significant.

    Parameters
    ----------
    df : pd.DataFrame
        Cell data
    model : LogisticRegression
        Fitted model on real data
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        True labels
    n_permutations : int
        Number of permutation tests

    Returns
    -------
    dict
        Test statistics and p-value
    """
    # Real model log-likelihood
    real_ll = model.score(X, y) * len(y)

    # Shuffled controls
    shuffled_lls = []

    for i in range(n_permutations):
        # Shuffle labels
        y_shuffled = np.random.permutation(y)

        # Fit model on shuffled data
        model_shuffled = LogisticRegression(
            random_state=i,
            max_iter=1000,
            solver='lbfgs'
        )
        model_shuffled.fit(X, y_shuffled)

        # Calculate log-likelihood
        shuffled_ll = model_shuffled.score(X, y_shuffled) * len(y)
        shuffled_lls.append(shuffled_ll)

    shuffled_lls = np.array(shuffled_lls)

    # P-value: fraction of shuffled models with LL >= real LL
    p_value = (shuffled_lls >= real_ll).mean()

    return {
        'real_log_likelihood': real_ll,
        'shuffled_log_likelihoods': shuffled_lls,
        'p_value': p_value,
        'z_score': (real_ll - shuffled_lls.mean()) / shuffled_lls.std()
    }


def visualize_state_model(df: pd.DataFrame,
                         model: LogisticRegression,
                         stats: Dict,
                         neighbor_columns: List[str]) -> None:
    """
    Visualize state probability model coefficients and predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Cell data
    model : LogisticRegression
        Fitted model
    stats : dict
        Model statistics
    neighbor_columns : list
        Neighborhood column names
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Regression coefficients
    coefficients = stats['coefficients']
    cell_types = [col.replace('#', '').replace('_neighbours', '')
                  for col in neighbor_columns]
    coef_values = [coefficients[col] for col in neighbor_columns]

    colors = ['red' if c > 0 else 'blue' for c in coef_values]
    ax1.barh(cell_types, coef_values, color=colors, alpha=0.7)
    ax1.axvline(0, color='black', linewidth=1, linestyle='--')
    ax1.set_xlabel('Coefficient (log odds)')
    ax1.set_title(f'{stats["cell_type"]} {stats["positive_state"]} State Model\n'
                  f'Intercept: {stats["intercept"]:.3f}')
    ax1.grid(axis='x', alpha=0.3)

    # Add interpretation labels
    ax1.text(0.02, 0.98, 'Positive: Favors state\nNegative: Inhibits state',
             transform=ax1.transAxes, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Predicted probabilities vs actual states
    cell_mask = df['Cell_Type'] == stats['cell_type']
    state_mask = df[stats['state_column']] != 'N/A'
    valid_mask = cell_mask & state_mask

    X = df.loc[valid_mask, neighbor_columns].values
    y_true = (df.loc[valid_mask, stats['state_column']] == stats['positive_state']).astype(int).values

    # Predict probabilities
    y_prob = model.predict_proba(X)[:, 1]

    # Histogram of predicted probabilities by true state
    ax2.hist(y_prob[y_true == 0], bins=50, alpha=0.6, label='Negative state', color='blue')
    ax2.hist(y_prob[y_true == 1], bins=50, alpha=0.6, label='Positive state', color='red')
    ax2.set_xlabel('Predicted probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Predicted State Probabilities')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Add performance metrics
    auc_text = f"AUC: {stats['cv_auc_mean']:.3f} ± {stats['cv_auc_std']:.3f}\n"
    auc_text += f"Accuracy: {stats['training_accuracy']:.3f}"
    ax2.text(0.02, 0.98, auc_text,
             transform=ax2.transAxes, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    plt.show()


# =============================================================================
# STATE EQUILIBRATION (SIMULATION INTEGRATION)
# =============================================================================

def equilibrate_states(tissue: pd.DataFrame,
                      state_models: Dict,
                      neighbor_columns: List[str],
                      random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Update cell states based on current neighborhoods (OSDR v2.0 Step 2).

    This implements the fast dynamics assumption: states instantly equilibrate
    to new neighborhood composition after population changes.

    Parameters
    ----------
    tissue : pd.DataFrame
        Current tissue state with neighborhoods computed
    state_models : dict
        Dictionary of fitted state probability models
        Format: {state_name: {cell_type: {'model': model, 'stats': stats}}}
    neighbor_columns : list
        Neighborhood count columns
    random_state : int, optional
        Random seed for stochastic state sampling

    Returns
    -------
    pd.DataFrame
        Tissue with updated state columns
    """
    tissue = tissue.copy()
    rng = np.random.default_rng(random_state)

    # For each state model
    for state_name, cell_type_models in state_models.items():
        for cell_type, model_dict in cell_type_models.items():
            model = model_dict['model']
            stats = model_dict['stats']

            # Get cells of this type
            mask = tissue['Cell_Type'] == cell_type

            if mask.sum() == 0:
                continue

            # Extract features
            X = tissue.loc[mask, neighbor_columns].values

            # Predict state probabilities
            P_state = model.predict_proba(X)[:, 1]

            # Sample states stochastically
            new_states = rng.binomial(1, P_state)

            # Update state column
            state_column = stats['state_column']
            positive_state = stats['positive_state']

            # Determine negative state label
            if 'PD1' in state_column:
                negative_state = 'PD1-'
            elif 'CAF' in state_column:
                negative_state = 'Resting'
            elif 'Cytotoxic' in state_column:
                negative_state = 'Inactive'
            elif 'Macrophage' in state_column:
                negative_state = 'M1-like'
            elif 'Treg' in state_column:
                negative_state = 'Conventional'
            else:
                negative_state = 'Negative'

            # Assign states
            tissue.loc[mask & (new_states == 1), state_column] = positive_state
            tissue.loc[mask & (new_states == 0), state_column] = negative_state

    return tissue


# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

def compare_state_probabilities(df: pd.DataFrame,
                                models: Dict,
                                neighbor_columns: List[str],
                                cell_type: str,
                                state_name: str) -> pd.DataFrame:
    """
    Compare predicted state probabilities across different neighborhoods.

    Shows how state probability varies with microenvironment composition.

    Parameters
    ----------
    df : pd.DataFrame
        Cell data
    models : dict
        Fitted state models
    neighbor_columns : list
        Neighborhood columns
    cell_type : str
        Cell type to analyze
    state_name : str
        State model name

    Returns
    -------
    pd.DataFrame
        DataFrame with neighborhood compositions and predicted probabilities
    """
    if state_name not in models or cell_type not in models[state_name]:
        raise ValueError(f"Model not found for {state_name} {cell_type}")

    model_dict = models[state_name][cell_type]
    model = model_dict['model']
    stats = model_dict['stats']

    # Get cells of this type
    cell_mask = df['Cell_Type'] == cell_type
    state_mask = df[stats['state_column']] != 'N/A'
    valid_mask = cell_mask & state_mask

    # Extract features
    X = df.loc[valid_mask, neighbor_columns].values

    # Predict probabilities
    P_state = model.predict_proba(X)[:, 1]

    # Create result dataframe
    result_df = df.loc[valid_mask, neighbor_columns].copy()
    result_df['Predicted_Probability'] = P_state
    result_df['True_State'] = df.loc[valid_mask, stats['state_column']].values

    return result_df


def print_model_summary(models: Dict) -> None:
    """
    Print summary of all fitted state models.

    Parameters
    ----------
    models : dict
        Dictionary of fitted models
    """
    print("\n" + "="*80)
    print("STATE PROBABILITY MODELS SUMMARY")
    print("="*80)

    for state_name, cell_type_models in models.items():
        print(f"\n{state_name}:")
        print("-" * 80)

        for cell_type, model_dict in cell_type_models.items():
            stats = model_dict['stats']

            print(f"\n  {cell_type}:")
            print(f"    Samples: {stats['n_samples']}")
            print(f"    Positive: {stats['n_positive']} ({100*stats['n_positive']/stats['n_samples']:.1f}%)")
            print(f"    AUC: {stats['cv_auc_mean']:.3f} ± {stats['cv_auc_std']:.3f}")
            print(f"    Accuracy: {stats['training_accuracy']:.3f}")
            print(f"    Intercept: {stats['intercept']:.3f}")

            print(f"    Top positive coefficients:")
            coef_items = sorted(stats['coefficients'].items(), key=lambda x: x[1], reverse=True)
            for col, coef in coef_items[:3]:
                cell_type_name = col.replace('#', '').replace('_neighbours', '')
                print(f"      {cell_type_name:20s}: {coef:+.4f}")

            print(f"    Top negative coefficients:")
            for col, coef in coef_items[-3:]:
                cell_type_name = col.replace('#', '').replace('_neighbours', '')
                print(f"      {cell_type_name:20s}: {coef:+.4f}")

    print("\n" + "="*80)
