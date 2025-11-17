"""
Cell state classification for OSDR v2.0.

This module implements cell type identification and functional state classification
based on protein marker expression from IMC/spatial proteomics data.

Author: Based on Shalom et al. 2025 OSDR v2.0 methodology
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.mixture import GaussianMixture
from typing import Dict, Tuple, Optional, List


# =============================================================================
# THRESHOLD DETERMINATION
# =============================================================================

def determine_threshold_percentile(marker_values: pd.Series, percentile: float = 50) -> float:
    """
    Simple percentile-based threshold.

    Parameters
    ----------
    marker_values : pd.Series
        Marker intensity values
    percentile : float
        Percentile to use (default 50 = median split)

    Returns
    -------
    float
        Threshold value
    """
    return np.percentile(marker_values[marker_values > 0], percentile)


def determine_threshold_gmm(marker_values: pd.Series,
                            n_components: int = 2,
                            plot: bool = False) -> float:
    """
    Gaussian Mixture Model for bimodal distributions.

    Best for markers with clear positive/negative populations.

    Parameters
    ----------
    marker_values : pd.Series
        Marker intensity values
    n_components : int
        Number of Gaussian components (typically 2 for +/-)
    plot : bool
        Whether to visualize the fit

    Returns
    -------
    float
        Optimal threshold between components
    """
    # Remove zeros and prepare data
    data = marker_values[marker_values > 0].values.reshape(-1, 1)

    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data)

    # Get means and sort
    means = gmm.means_.flatten()
    means_sorted = np.sort(means)

    # Threshold at midpoint between two lowest means
    threshold = (means_sorted[0] + means_sorted[1]) / 2

    if plot:
        plt.figure(figsize=(10, 4))
        plt.hist(data, bins=100, alpha=0.6, density=True, label='Data')

        # Plot GMM components
        x_range = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
        responsibilities = gmm.predict_proba(x_range)
        pdf = np.exp(gmm.score_samples(x_range))

        plt.plot(x_range, pdf, 'k-', linewidth=2, label='GMM fit')
        for i in range(n_components):
            plt.axvline(means[i], linestyle='--', alpha=0.5,
                       label=f'Component {i+1} mean')

        plt.axvline(threshold, color='red', linewidth=2, label='Threshold')
        plt.xlabel('Marker intensity')
        plt.ylabel('Density')
        plt.legend()
        plt.title(f'GMM-based threshold determination')
        plt.show()

    return threshold


def determine_threshold_otsu(marker_values: pd.Series) -> float:
    """
    Otsu's method for automatic threshold determination.

    Maximizes between-class variance. Good for markers with
    distinct positive/negative populations.

    Parameters
    ----------
    marker_values : pd.Series
        Marker intensity values

    Returns
    -------
    float
        Optimal threshold
    """
    data = marker_values[marker_values > 0].values

    # Create histogram
    hist, bin_edges = np.histogram(data, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize histogram
    hist = hist.astype(float) / hist.sum()

    # Calculate cumulative sums and means
    cum_sum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * bin_centers)

    # Global mean
    global_mean = cum_mean[-1]

    # Between-class variance
    between_class_variance = (global_mean * cum_sum - cum_mean) ** 2 / (
        cum_sum * (1 - cum_sum) + 1e-10
    )

    # Find threshold that maximizes variance
    threshold_idx = np.argmax(between_class_variance)
    threshold = bin_centers[threshold_idx]

    return threshold


def visualize_marker_distribution(df: pd.DataFrame,
                                  marker: str,
                                  threshold: Optional[float] = None,
                                  cell_type_filter: Optional[str] = None) -> None:
    """
    Visualize marker distribution with optional threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Cell data with marker columns
    marker : str
        Marker column name
    threshold : float, optional
        Threshold to plot
    cell_type_filter : str, optional
        Filter to specific cell type
    """
    data = df[df[marker] > 0][marker]

    if cell_type_filter and 'Cell_Type' in df.columns:
        data = df[(df[marker] > 0) & (df['Cell_Type'] == cell_type_filter)][marker]

    plt.figure(figsize=(12, 4))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(data, bins=100, alpha=0.7, edgecolor='black')
    if threshold is not None:
        plt.axvline(threshold, color='red', linewidth=2,
                   label=f'Threshold = {threshold:.2f}')
        pos_count = (data > threshold).sum()
        neg_count = (data <= threshold).sum()
        plt.title(f'{marker} Distribution\n'
                 f'Positive: {pos_count} ({100*pos_count/len(data):.1f}%), '
                 f'Negative: {neg_count} ({100*neg_count/len(data):.1f}%)')
    else:
        plt.title(f'{marker} Distribution')
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    plt.legend()

    # Log scale
    plt.subplot(1, 2, 2)
    plt.hist(np.log10(data + 1), bins=100, alpha=0.7, edgecolor='black')
    if threshold is not None:
        plt.axvline(np.log10(threshold + 1), color='red', linewidth=2,
                   label=f'Threshold')
    plt.xlabel('log10(Intensity + 1)')
    plt.ylabel('Count')
    plt.title(f'{marker} Distribution (Log Scale)')
    plt.legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# CELL TYPE IDENTIFICATION
# =============================================================================

def identify_cell_types(df: pd.DataFrame,
                       thresholds: Dict[str, float]) -> pd.DataFrame:
    """
    Identify major cell types based on marker combinations.

    Parameters
    ----------
    df : pd.DataFrame
        Cell data with marker columns
    thresholds : dict
        Dictionary of marker thresholds {marker_name: threshold_value}

    Returns
    -------
    pd.DataFrame
        Input dataframe with added 'Cell_Type' column
    """
    df = df.copy()
    df['Cell_Type'] = 'Unknown'

    # Priority order (more specific first)

    # 1. Cancer cells: Pan-Keratin+
    cancer_mask = df['Pan-Keratin'] > thresholds.get('Pan-Keratin', 0)
    df.loc[cancer_mask, 'Cell_Type'] = 'Cancer'

    # 2. T cells: CD3+ (subdivide by CD4/CD8)
    t_mask = df['CD3'] > thresholds.get('CD3', 0)
    cd4_mask = df['CD4'] > thresholds.get('CD4', 0)
    cd8_mask = df['CD8a'] > thresholds.get('CD8a', 0)

    df.loc[t_mask & cd4_mask, 'Cell_Type'] = 'CD4_T'
    df.loc[t_mask & cd8_mask, 'Cell_Type'] = 'CD8_T'
    df.loc[t_mask & ~cd4_mask & ~cd8_mask, 'Cell_Type'] = 'T_other'

    # 3. Macrophages: CD68+
    macro_mask = df['CD68'] > thresholds.get('CD68', 0)
    df.loc[macro_mask & ~cancer_mask, 'Cell_Type'] = 'Macrophage'

    # 4. Fibroblasts: Vimentin+ Pan-Keratin-
    vim_mask = df['Vimentin'] > thresholds.get('Vimentin', 0)
    fib_mask = vim_mask & ~cancer_mask & ~t_mask & ~macro_mask
    df.loc[fib_mask, 'Cell_Type'] = 'Fibroblast'

    # 5. Endothelial: CD31+ (if available)
    if 'CD31' in df.columns:
        endo_mask = df['CD31'] > thresholds.get('CD31', 0)
        df.loc[endo_mask & ~cancer_mask, 'Cell_Type'] = 'Endothelial'

    return df


# =============================================================================
# STATE CLASSIFICATION (OSDR v2.0 CORE)
# =============================================================================

def classify_pd1_states(df: pd.DataFrame,
                       pd1_threshold: float,
                       only_t_cells: bool = True) -> pd.DataFrame:
    """
    Classify PD1+ (exhausted) vs PD1- (functional) T cell states.

    This is the primary v2.0 feature with highest predictive power (AUC 0.94-0.96).

    Parameters
    ----------
    df : pd.DataFrame
        Cell data with 'Cell_Type' and 'PD1' columns
    pd1_threshold : float
        Threshold for PD1 positivity
    only_t_cells : bool
        Only classify T cells (recommended)

    Returns
    -------
    pd.DataFrame
        Input dataframe with added 'PD1_State' column
    """
    df = df.copy()
    df['PD1_State'] = 'N/A'

    if only_t_cells:
        t_cell_types = ['CD4_T', 'CD8_T', 'T_other']
        t_mask = df['Cell_Type'].isin(t_cell_types)
    else:
        t_mask = df.index.to_series().astype(bool)  # All cells

    # Classify states
    df.loc[t_mask & (df['PD1'] > pd1_threshold), 'PD1_State'] = 'PD1+'
    df.loc[t_mask & (df['PD1'] <= pd1_threshold), 'PD1_State'] = 'PD1-'

    return df


def classify_caf_states(df: pd.DataFrame,
                       sma_threshold: float,
                       vim_threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Classify CAF (Cancer-Associated Fibroblast) vs resting fibroblast states.

    Simplified version using Alpha-SMA only (missing PDGFRB, PDPN from paper).

    Parameters
    ----------
    df : pd.DataFrame
        Cell data with 'Cell_Type' and 'Alpha-SMA' columns
    sma_threshold : float
        Threshold for Alpha-SMA positivity
    vim_threshold : float, optional
        Additional Vimentin filter for fibroblast confirmation

    Returns
    -------
    pd.DataFrame
        Input dataframe with added 'CAF_State' column
    """
    df = df.copy()
    df['CAF_State'] = 'N/A'

    # Only classify fibroblasts
    fib_mask = df['Cell_Type'] == 'Fibroblast'

    if vim_threshold is not None:
        fib_mask = fib_mask & (df['Vimentin'] > vim_threshold)

    # Classify states
    df.loc[fib_mask & (df['Alpha-SMA'] > sma_threshold), 'CAF_State'] = 'CAF'
    df.loc[fib_mask & (df['Alpha-SMA'] <= sma_threshold), 'CAF_State'] = 'Resting'

    return df


def classify_cytotoxic_states(df: pd.DataFrame,
                              gzmb_threshold: float) -> pd.DataFrame:
    """
    Classify cytotoxic T cell activity states (GranzymeB+ active vs inactive).

    Novel state beyond original v2.0 paper.

    Parameters
    ----------
    df : pd.DataFrame
        Cell data with 'Cell_Type' and 'Granzyme-B' columns
    gzmb_threshold : float
        Threshold for Granzyme-B positivity

    Returns
    -------
    pd.DataFrame
        Input dataframe with added 'Cytotoxic_State' column
    """
    df = df.copy()
    df['Cytotoxic_State'] = 'N/A'

    # Only CD8+ T cells (primary cytotoxic population)
    cd8_mask = df['Cell_Type'] == 'CD8_T'

    # Classify states
    df.loc[cd8_mask & (df['Granzyme-B'] > gzmb_threshold), 'Cytotoxic_State'] = 'Active'
    df.loc[cd8_mask & (df['Granzyme-B'] <= gzmb_threshold), 'Cytotoxic_State'] = 'Inactive'

    return df


def classify_macrophage_states(df: pd.DataFrame,
                               cd163_threshold: float) -> pd.DataFrame:
    """
    Classify macrophage polarization (CD163+ M2-like vs CD163- M1-like).

    Novel state beyond original v2.0 paper.

    Parameters
    ----------
    df : pd.DataFrame
        Cell data with 'Cell_Type' and 'CD163' columns
    cd163_threshold : float
        Threshold for CD163 positivity

    Returns
    -------
    pd.DataFrame
        Input dataframe with added 'Macrophage_State' column
    """
    df = df.copy()
    df['Macrophage_State'] = 'N/A'

    # Only macrophages
    macro_mask = df['Cell_Type'] == 'Macrophage'

    # Classify states
    df.loc[macro_mask & (df['CD163'] > cd163_threshold), 'Macrophage_State'] = 'M2-like'
    df.loc[macro_mask & (df['CD163'] <= cd163_threshold), 'Macrophage_State'] = 'M1-like'

    return df


def classify_treg_states(df: pd.DataFrame,
                        foxp3_threshold: float) -> pd.DataFrame:
    """
    Classify regulatory T cell states (FOXP3+ Tregs vs conventional).

    Novel state beyond original v2.0 paper.

    Parameters
    ----------
    df : pd.DataFrame
        Cell data with 'Cell_Type' and 'FOXP3' columns
    foxp3_threshold : float
        Threshold for FOXP3 positivity

    Returns
    -------
    pd.DataFrame
        Input dataframe with added 'Treg_State' column
    """
    df = df.copy()
    df['Treg_State'] = 'N/A'

    # Only CD4+ T cells
    cd4_mask = df['Cell_Type'] == 'CD4_T'

    # Classify states
    df.loc[cd4_mask & (df['FOXP3'] > foxp3_threshold), 'Treg_State'] = 'Treg'
    df.loc[cd4_mask & (df['FOXP3'] <= foxp3_threshold), 'Treg_State'] = 'Conventional'

    return df


# =============================================================================
# MASTER CLASSIFICATION PIPELINE
# =============================================================================

def classify_all_states(df: pd.DataFrame,
                       thresholds: Dict[str, float],
                       states_to_classify: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Master function to classify all cell types and states.

    Parameters
    ----------
    df : pd.DataFrame
        Cell data with marker columns
    thresholds : dict
        Dictionary of all marker thresholds
    states_to_classify : list, optional
        Which states to classify. If None, classifies all available.
        Options: ['PD1', 'CAF', 'Cytotoxic', 'Macrophage', 'Treg']

    Returns
    -------
    pd.DataFrame
        Fully classified dataframe
    """
    if states_to_classify is None:
        states_to_classify = ['PD1', 'CAF', 'Cytotoxic', 'Macrophage', 'Treg']

    # Step 1: Identify cell types
    df = identify_cell_types(df, thresholds)

    # Step 2: Classify states
    if 'PD1' in states_to_classify and 'PD1' in thresholds:
        df = classify_pd1_states(df, thresholds['PD1'])

    if 'CAF' in states_to_classify and 'Alpha-SMA' in thresholds:
        df = classify_caf_states(df, thresholds['Alpha-SMA'],
                                thresholds.get('Vimentin'))

    if 'Cytotoxic' in states_to_classify and 'Granzyme-B' in thresholds:
        df = classify_cytotoxic_states(df, thresholds['Granzyme-B'])

    if 'Macrophage' in states_to_classify and 'CD163' in thresholds:
        df = classify_macrophage_states(df, thresholds['CD163'])

    if 'Treg' in states_to_classify and 'FOXP3' in thresholds:
        df = classify_treg_states(df, thresholds['FOXP3'])

    return df


# =============================================================================
# THRESHOLD DETERMINATION PIPELINE
# =============================================================================

def auto_determine_thresholds(df: pd.DataFrame,
                              markers: List[str],
                              method: str = 'gmm',
                              plot: bool = True) -> Dict[str, float]:
    """
    Automatically determine thresholds for all markers.

    Parameters
    ----------
    df : pd.DataFrame
        Cell data with marker columns
    markers : list
        List of marker column names
    method : str
        Method to use: 'gmm', 'otsu', 'median', or 'percentile'
    plot : bool
        Whether to visualize results

    Returns
    -------
    dict
        Dictionary of {marker: threshold}
    """
    thresholds = {}

    for marker in markers:
        if marker not in df.columns:
            print(f"Warning: {marker} not found in dataframe")
            continue

        print(f"\nDetermining threshold for {marker}...")

        if method == 'gmm':
            threshold = determine_threshold_gmm(df[marker], plot=plot)
        elif method == 'otsu':
            threshold = determine_threshold_otsu(df[marker])
        elif method == 'median':
            threshold = determine_threshold_percentile(df[marker], percentile=50)
        elif method == 'percentile':
            threshold = determine_threshold_percentile(df[marker], percentile=75)
        else:
            raise ValueError(f"Unknown method: {method}")

        thresholds[marker] = threshold

        if plot:
            visualize_marker_distribution(df, marker, threshold)

    return thresholds


# =============================================================================
# VALIDATION AND QUALITY CONTROL
# =============================================================================

def validate_state_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for state classifications.

    Parameters
    ----------
    df : pd.DataFrame
        Classified cell data

    Returns
    -------
    pd.DataFrame
        Summary table with counts and percentages
    """
    summary_data = []

    # Cell type distribution
    if 'Cell_Type' in df.columns:
        type_counts = df['Cell_Type'].value_counts()
        for cell_type, count in type_counts.items():
            summary_data.append({
                'Category': 'Cell Type',
                'Type': cell_type,
                'Count': count,
                'Percentage': 100 * count / len(df)
            })

    # PD1 states (within T cells)
    if 'PD1_State' in df.columns:
        pd1_counts = df[df['PD1_State'] != 'N/A']['PD1_State'].value_counts()
        total_t = (df['PD1_State'] != 'N/A').sum()
        for state, count in pd1_counts.items():
            summary_data.append({
                'Category': 'PD1 State (T cells)',
                'Type': state,
                'Count': count,
                'Percentage': 100 * count / total_t if total_t > 0 else 0
            })

    # CAF states (within fibroblasts)
    if 'CAF_State' in df.columns:
        caf_counts = df[df['CAF_State'] != 'N/A']['CAF_State'].value_counts()
        total_fib = (df['CAF_State'] != 'N/A').sum()
        for state, count in caf_counts.items():
            summary_data.append({
                'Category': 'CAF State (Fibroblasts)',
                'Type': state,
                'Count': count,
                'Percentage': 100 * count / total_fib if total_fib > 0 else 0
            })

    # Cytotoxic states (within CD8 T cells)
    if 'Cytotoxic_State' in df.columns:
        cyto_counts = df[df['Cytotoxic_State'] != 'N/A']['Cytotoxic_State'].value_counts()
        total_cd8 = (df['Cytotoxic_State'] != 'N/A').sum()
        for state, count in cyto_counts.items():
            summary_data.append({
                'Category': 'Cytotoxic State (CD8 T)',
                'Type': state,
                'Count': count,
                'Percentage': 100 * count / total_cd8 if total_cd8 > 0 else 0
            })

    # Macrophage states
    if 'Macrophage_State' in df.columns:
        macro_counts = df[df['Macrophage_State'] != 'N/A']['Macrophage_State'].value_counts()
        total_macro = (df['Macrophage_State'] != 'N/A').sum()
        for state, count in macro_counts.items():
            summary_data.append({
                'Category': 'Macrophage State',
                'Type': state,
                'Count': count,
                'Percentage': 100 * count / total_macro if total_macro > 0 else 0
            })

    # Treg states (within CD4 T cells)
    if 'Treg_State' in df.columns:
        treg_counts = df[df['Treg_State'] != 'N/A']['Treg_State'].value_counts()
        total_cd4 = (df['Treg_State'] != 'N/A').sum()
        for state, count in treg_counts.items():
            summary_data.append({
                'Category': 'Treg State (CD4 T)',
                'Type': state,
                'Count': count,
                'Percentage': 100 * count / total_cd4 if total_cd4 > 0 else 0
            })

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def plot_state_distributions(df: pd.DataFrame,
                             save_path: Optional[str] = None) -> None:
    """
    Visualize all state distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Classified cell data
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Cell types
    if 'Cell_Type' in df.columns:
        type_counts = df['Cell_Type'].value_counts()
        axes[0].bar(range(len(type_counts)), type_counts.values)
        axes[0].set_xticks(range(len(type_counts)))
        axes[0].set_xticklabels(type_counts.index, rotation=45, ha='right')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Cell Type Distribution')

    # PD1 states
    if 'PD1_State' in df.columns:
        pd1_counts = df[df['PD1_State'] != 'N/A']['PD1_State'].value_counts()
        axes[1].bar(range(len(pd1_counts)), pd1_counts.values, color=['red', 'blue'])
        axes[1].set_xticks(range(len(pd1_counts)))
        axes[1].set_xticklabels(pd1_counts.index)
        axes[1].set_ylabel('Count')
        axes[1].set_title('PD1 States (T cells)')

    # CAF states
    if 'CAF_State' in df.columns:
        caf_counts = df[df['CAF_State'] != 'N/A']['CAF_State'].value_counts()
        axes[2].bar(range(len(caf_counts)), caf_counts.values, color=['orange', 'green'])
        axes[2].set_xticks(range(len(caf_counts)))
        axes[2].set_xticklabels(caf_counts.index)
        axes[2].set_ylabel('Count')
        axes[2].set_title('CAF States (Fibroblasts)')

    # Cytotoxic states
    if 'Cytotoxic_State' in df.columns:
        cyto_counts = df[df['Cytotoxic_State'] != 'N/A']['Cytotoxic_State'].value_counts()
        axes[3].bar(range(len(cyto_counts)), cyto_counts.values, color=['darkred', 'pink'])
        axes[3].set_xticks(range(len(cyto_counts)))
        axes[3].set_xticklabels(cyto_counts.index)
        axes[3].set_ylabel('Count')
        axes[3].set_title('Cytotoxic States (CD8 T)')

    # Macrophage states
    if 'Macrophage_State' in df.columns:
        macro_counts = df[df['Macrophage_State'] != 'N/A']['Macrophage_State'].value_counts()
        axes[4].bar(range(len(macro_counts)), macro_counts.values, color=['purple', 'yellow'])
        axes[4].set_xticks(range(len(macro_counts)))
        axes[4].set_xticklabels(macro_counts.index)
        axes[4].set_ylabel('Count')
        axes[4].set_title('Macrophage Polarization')

    # Treg states
    if 'Treg_State' in df.columns:
        treg_counts = df[df['Treg_State'] != 'N/A']['Treg_State'].value_counts()
        axes[5].bar(range(len(treg_counts)), treg_counts.values, color=['brown', 'cyan'])
        axes[5].set_xticks(range(len(treg_counts)))
        axes[5].set_xticklabels(treg_counts.index)
        axes[5].set_ylabel('Count')
        axes[5].set_title('Treg States (CD4 T)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
