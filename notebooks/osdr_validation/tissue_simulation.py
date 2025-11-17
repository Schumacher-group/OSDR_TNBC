"""
Tissue simulation utilities for OSDR validation.

Functions for generating random tissues and simulating stochastic proliferation
with known dynamical models.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count

# Global parameters
CELLS = ["F", "M"]  # cell type list
X_BOUNDARIES = (0, 2500)  # tissue size x boundaries in microns
Y_BOUNDARIES = (0, 2500)  # tissue size y boundaries in microns
R = 80  # neighbourhood radius in microns, same as in Somer et al.

# Model parameters (alternative model with logistic division)
A = -0.120
INTERCEPT = -2.456
B = 1 / (1 + np.exp(-(INTERCEPT + A * 16)))


def random_tissue(seed, cells, tissue_id, x_boundaries=X_BOUNDARIES, 
                  y_boundaries=Y_BOUNDARIES, r=R):
    """
    Generate a random tissue with two cell types.
    
    Parameters
    ----------
    seed : int or np.random.SeedSequence
        Random seed for reproducibility
    cells : list of str
        Cell type labels (e.g., ["F", "M"])
    tissue_id : int
        Identifier for this tissue
    x_boundaries : tuple
        (min, max) x-coordinates for tissue bounds
    y_boundaries : tuple
        (min, max) y-coordinates for tissue bounds
    r : float
        Neighbourhood radius in microns
        
    Returns
    -------
    pd.DataFrame
        Tissue dataframe with columns: Tissue_ID, Cell_Type, X, Y,
        #F_neighbours, #F_neighbours_log2, #M_neighbours, #M_neighbours_log2
    """
    rng = np.random.default_rng(seed)

    idcol = []
    cell_type = []
    position_x = []
    position_y = []
    
    for cell in cells:
        cell_num = int(rng.integers(250, 9750))
        for i in range(0, cell_num):
            idcol.append(tissue_id)
            cell_type.append(f"{cell}")
            position_x.append(rng.uniform(x_boundaries[0], x_boundaries[1]))
            position_y.append(rng.uniform(y_boundaries[0], y_boundaries[1]))
            
    tissue = pd.DataFrame(
        list(zip(idcol, cell_type, position_x, position_y)), 
        columns=["Tissue_ID", "Cell_Type", "X", "Y"]
    )
    
    # Get all spatial coordinates as array for query
    coordinates = tissue[['X', 'Y']].values
    
    for cell in cells:
        # Build KD-Trees for each cell
        tree = cKDTree(tissue[tissue['Cell_Type'] == cell][['X', 'Y']].values)
        # Search neighbourhood contents of the current cell type for all cells
        neighbours = [len(tree.query_ball_point(coords, r)) - (1 if cat == cell else 0)
                      for coords, cat in zip(coordinates, tissue['Cell_Type'])]
        column_name = f"#{cell}_neighbours"
        tissue[column_name] = neighbours
        with np.errstate(divide='ignore', invalid='ignore'):
            tissue[column_name + "_log2"] = np.log2(neighbours)
     
    return tissue


def tissue_proliferation(seed, tissue, n=100, t=None, cells=CELLS, r=R,
                         a=A, intercept=INTERCEPT, b=B,
                         x_boundaries=X_BOUNDARIES, y_boundaries=Y_BOUNDARIES):
    """
    Simulate stochastic tissue proliferation with known dynamical model.
    
    Parameters
    ----------
    seed : int or np.random.SeedSequence
        Random seed for reproducibility
    tissue : pd.DataFrame
        Initial tissue state
    n : int
        Number of time steps to simulate
    t : int or None
        Interval to record tissue states. If None, only record final state
    cells : list of str
        Cell type labels
    r : float
        Neighbourhood radius
    a, intercept, b : float
        Model parameters for logistic division probability
    x_boundaries, y_boundaries : tuple
        Tissue spatial boundaries
        
    Returns
    -------
    pd.DataFrame
        Proliferated tissue states with Time_Step column
    """
    if t is None:  # if t is not given, only record the final time step
        t = n
    
    rng = np.random.default_rng(seed)

    div_counter = 0
    death_counter = 0
    stay_counter = 0

    t_tracker = 0  # when it hits the set t, record data and reset tracker

    proliferation_results = pd.DataFrame(columns=tissue.columns)
    proliferation_results['Time_Step'] = pd.Series(dtype='int')
    
    for step in range(0, n):
        t_tracker += 1

        next_tissue = tissue.copy()
        dead_ids = []

        for cell_id in range(0, len(tissue)):
            # get current cell
            cell = tissue.iloc[cell_id]

            # get event probabilities
            if cell.iloc[1] == "F":
                p_div = 1 / (1 + np.exp(-(intercept + a * cell.iloc[4])))
            else:
                p_div = 1 / (1 + np.exp(-(intercept + a * cell.iloc[6])))
            p_death = b
            p_stay = 1 - p_div - p_death

            if p_stay < 0:
                print("Error with probability space")
                print(p_div, p_death, p_stay, p_div + p_death + p_stay)
                break

            # sample random event uniformly
            r1 = rng.uniform(0, 1)

            # act on sampled event
            if r1 <= p_div:  # division
                rx = rng.uniform(-r, r)
                ry = rng.uniform(-r, r)

                # prevent out of bounds and excessive proliferation on the edges
                if cell.iloc[2] + rx < 0:
                    rx = rng.uniform(0, r)
                elif cell.iloc[2] + rx > x_boundaries[1]:
                    rx = rng.uniform(-r, 0)
                if cell.iloc[3] + ry < 0:
                    ry = rng.uniform(0, r)
                elif cell.iloc[3] + ry > y_boundaries[1]:
                    ry = rng.uniform(-r, 0)
                
                new_cell = [cell.iloc[0], cell.iloc[1], cell.iloc[2] + rx, 
                            cell.iloc[3] + ry, 2, 1, 2, 1]

                next_tissue.loc[len(next_tissue)] = new_cell
                
                div_counter += 1
            elif r1 <= p_div + p_death:  # death
                dead_ids.append(cell_id)
                death_counter += 1
            elif r1 <= p_div + p_death + p_stay:  # nothing happens
                stay_counter += 1
                pass
            else:  # if triggered, I made a mistake in the handling/definition of the events
                print("Unexpected random number.")
                break
        
        # remove chosen dead cells
        next_tissue = next_tissue.drop(dead_ids)
        next_tissue = next_tissue.reset_index(drop=True)

        # recompute neighbours
        coordinates = next_tissue[['X', 'Y']].values
        for ctype in cells:
            # Build KD-Trees for each cell type
            tree = cKDTree(next_tissue[next_tissue['Cell_Type'] == ctype][['X', 'Y']].values)
            # Search neighbourhood contents of the current cell type for all cells
            neighbours = [len(tree.query_ball_point(coords, r)) - (1 if cat == ctype else 0)
                          for coords, cat in zip(coordinates, next_tissue['Cell_Type'])]
            column_name = f"#{ctype}_neighbours"
            next_tissue[column_name] = neighbours
            with np.errstate(divide='ignore', invalid='ignore'):
                next_tissue[column_name + "_log2"] = np.log2(neighbours)

        tissue = next_tissue.copy()

        # record tissue at appropriate time steps
        if step == n - 1:
            recorded_tissue = tissue.copy()
            recorded_tissue["Time_Step"] = step + 1
            proliferation_results = pd.concat([proliferation_results, recorded_tissue], 
                                              ignore_index=True)
        elif t_tracker == t:
            recorded_tissue = tissue.copy()
            recorded_tissue["Time_Step"] = step + 1
            proliferation_results = pd.concat([proliferation_results, recorded_tissue], 
                                              ignore_index=True)
            t_tracker = 0

    return proliferation_results


def wrapper1(args):
    """Wrapper for random_tissue to use with multiprocessing."""
    return random_tissue(*args)


def wrapper2(args):
    """Wrapper for tissue_proliferation to use with multiprocessing."""
    return tissue_proliferation(*args)


def simulate_model(replicates=100, n=1000, t=100, cells=CELLS):
    """
    Simulate multiple tissue replicates in parallel.
    
    Parameters
    ----------
    replicates : int
        Number of tissue replicates to simulate
    n : int
        Number of proliferation time steps
    t : int
        Interval to record tissue states
    cells : list of str
        Cell type labels
        
    Returns
    -------
    tuple of pd.DataFrame
        (initial_tissues, proliferated_tissues)
    """
    start_time = time.time()
    
    # Prepare controlled random seeding for the parallelised tasks
    master_seed = np.random.SeedSequence()  # new every run
    child_seeds = master_seed.spawn(replicates)
    arguments = [(seed, cells, t_id) for seed, t_id in zip(child_seeds, range(0, replicates))]
    
    print("Beginning tissue initialisation...")
    # Initialise random tissues
    with Pool(processes=cpu_count()) as pool:
        results1 = list(tqdm(pool.imap(wrapper1, arguments), total=len(arguments)))
            
    end_time = time.time()
    duration = end_time - start_time
    print(f"Done... This took about {round(duration)} second(s).")
    tissues = pd.concat(results1, ignore_index=True)
    
    # Get a new set of seeds for proliferation
    start_time = time.time()
    master_seed = np.random.SeedSequence()  # new every run
    child_seeds = master_seed.spawn(replicates)
    arguments = [(seed, tissue, n, t) for seed, tissue in zip(child_seeds, results1)]
    
    print("Beginning tissue proliferation...")
    # Tissue proliferation
    with Pool(processes=cpu_count()) as pool:
        results2 = list(tqdm(pool.imap(wrapper2, arguments), total=len(arguments)))
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Done... This took about {round(duration)} second(s).")
    proliferated_tissues = pd.concat(results2, ignore_index=True)

    return tissues, proliferated_tissues
