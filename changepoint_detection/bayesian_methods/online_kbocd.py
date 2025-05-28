#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fully Nonparametric Bayesian Online Change Point Detection Using Kernel Density Estimation
with Nonparametric Hazard Function (r, a) tracking and Improved Sheather-Jones algorithm for bandwidth selection,
plus some modifications to reduce early false-positives and capture later major shifts more reliably.
"""

import numpy as np
import warnings
from math import log, exp, sqrt, pi, isinf
from typing import Tuple, List, Dict, Optional
from KDEpy import FFTKDE
from functools import lru_cache
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

###############################################################################
# Log-sum-exp helper
###############################################################################
def log_sum_exp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    if x.size == 0:
        return -np.inf
    xmax = np.max(x)
    if np.isneginf(xmax):
        return -np.inf
    return xmax + np.log(np.sum(np.exp(x - xmax)))

###############################################################################
# KDE predictive probability with caching
###############################################################################
@lru_cache(maxsize=1024)
def compute_bandwidth_isj(data_key: Tuple[float, ...]) -> float:
    """
    Compute ISJ bandwidth for a given data segment.
    Falls back to Silverman's rule only if ISJ fails.
    """
    data = np.array(data_key)
    
    try:
        # Use ISJ for bandwidth selection
        kde = FFTKDE(kernel='gaussian', bw='ISJ')
        kde.fit(data[:, np.newaxis])
        return kde.bw
    except Exception as e:
        # Simple fallback to Silverman's rule if ISJ fails
        std = np.std(data)
        n = len(data)
        return 1.06 * std * (n ** (-1/5))

def kde_predictive_prob(x_new: float, data_seg: np.ndarray) -> float:
    """
    Gaussian KDE-based predictive probability using ISJ bandwidth selection.
    """
    data_seg = np.asarray(data_seg, dtype=float)
    n = len(data_seg)
    
    # Minimum sample size check
    if n < 4:
        mean = np.mean(data_seg)
        std = max(np.std(data_seg), 1e-12)
        return max(1e-12, (1.0 / (std * np.sqrt(2*pi))) * 
                  np.exp(-0.5 * ((x_new - mean) / std)**2))
    
    try:
        # Fit KDE using ISJ bandwidth selection
        kde = FFTKDE(kernel='gaussian', bw='ISJ')
        kde.fit(data_seg[:, np.newaxis])
        
        # Evaluate at the new point
        density = kde.evaluate(np.array([[x_new]]))
        return max(density[0], 1e-12)
        
    except Exception as e:
        # Simple fallback using normal approximation
        mean = np.mean(data_seg)
        std = max(np.std(data_seg), 1e-12)
        return max(1e-12, (1.0 / (std * np.sqrt(2*pi))) * 
                  np.exp(-0.5 * ((x_new - mean) / std)**2))

###############################################################################
# Main KBOCD w/ Nonparametric Hazard (r,a) Tracking
###############################################################################
def kbocd_nonparametric_hazard(
    data: np.ndarray,
    detect_threshold: float = 0.95,
    burn_in: int = 15,
    min_distance: int = 20
) -> Tuple[np.ndarray, List[int], np.ndarray, np.ndarray]:
    """
    A fully nonparametric Bayesian online change point detection method.
    Uses ISJ bandwidth for both predictive probability and hazard rate.
    """
    print("\nInitializing KBOCD...")
    
    # Input validation and preprocessing
    data = np.asarray(data, dtype=float)
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or Inf values")
    
    T = len(data)
    print(f"Data length: {T}")
    
    if T < 2:
        print("Warning: Data too short for meaningful analysis")
        return np.zeros((1, T)), [], np.ones(T), np.zeros((T+1, T+1, T))

    # Initialize run length distribution
    log_R = np.full((T+1, T), -np.inf)  # Run length distribution
    log_R[0, 0] = 0.0  # Start with r=0 at t=0
    
    # Storage for evidence and change points
    log_evidence = np.zeros(T)
    change_points = []
    
    # Main loop with progress bar
    print("\nStarting main detection loop...")
    with tqdm(total=T-1, desc="Processing time steps", unit="step") as pbar:
        prev_max_prob = 0.0  # Track previous maximum probability
        for t in range(1, T):
            x_t = data[t]
            
            # Compute predictive probabilities
            pred_probs = np.full(t + 1, -np.inf)
            for r in range(t + 1):
                # Get data segment
                if r == 0:
                    pred_data = data[max(0, t-10):t]
                else:
                    pred_data = data[t-r:t]
                
                if len(pred_data) > 0:
                    try:
                        p_x = kde_predictive_prob(x_t, pred_data)
                        pred_probs[r] = log(max(p_x, 1e-12))
                    except Exception:
                        continue
            
            # Message passing
            log_growth_prob = np.full(t + 1, -np.inf)
            
            # Compute growth probabilities
            for r in range(t):
                if isinf(log_R[r, t-1]):
                    continue
                
                # Get data segment for hazard computation
                data_segment = data[max(0, t-r):t]
                
                # Compute hazard rate from ISJ bandwidth
                try:
                    if len(data_segment) >= 10:
                        bw = compute_bandwidth_isj(tuple(data_segment))
                        h = min(max(bw, 0.05), 0.5)  # Use bandwidth directly as hazard
                    else:
                        h = 0.1
                except Exception:
                    h = 0.1
                
                # Growth probability: P(r_t = r+1 | r_{t-1} = r)
                if not isinf(pred_probs[r+1]):
                    log_growth_prob[r+1] = (log_R[r, t-1] + 
                                          pred_probs[r+1] + 
                                          log(1.0 - h))
                
                # Change point probability: P(r_t = 0 | r_{t-1} = r)
                if not isinf(pred_probs[0]):
                    log_growth_prob[0] = np.logaddexp(
                        log_growth_prob[0],
                        log_R[r, t-1] + pred_probs[0] + log(h)
                    )
            
            # Update run length distribution
            log_R[:t+1, t] = log_growth_prob[:t+1]
            
            # Normalize in log space
            log_norm = log_sum_exp(log_R[:t+1, t])
            if not isinf(log_norm):
                log_R[:t+1, t] -= log_norm
                log_evidence[t] = log_norm
            
            # Get current maximum probability
            current_probs = np.exp(log_R[:t+1, t])
            current_max_prob = np.max(current_probs)
            
            # Detect change points based on probability drops
            if t >= burn_in:
                # Check if we had a high probability that suddenly dropped
                if prev_max_prob > detect_threshold and current_max_prob < 0.5:  # Significant drop
                    far_enough = not change_points or (t - change_points[-1]) >= min_distance
                    
                    if far_enough:
                        # Verify change using ISJ bandwidth
                        pre_window = data[max(0, t-10):t]
                        post_window = data[t:min(t+10, T)]
                        
                        if len(pre_window) >= 4 and len(post_window) >= 4:
                            try:
                                # Use ISJ bandwidth as a direct measure of distribution complexity
                                bw = compute_bandwidth_isj(tuple(np.concatenate([pre_window, post_window])))
                                if bw > 0.5:  # Significant distribution change
                                    change_points.append(t)
                                    print(f"\nChange point detected at t={t}")
                                    print(f"Probability drop: {prev_max_prob:.3f} -> {current_max_prob:.3f}")
                                    print(f"ISJ Bandwidth: {bw:.2f}")
                                    pbar.set_postfix({"Change Points": len(change_points)})
                            except Exception:
                                continue
            
            prev_max_prob = current_max_prob  # Update previous maximum probability
            pbar.update(1)
    
    # Convert to probabilities and store history
    run_length_prob = np.exp(log_R)
    log_joint_prob_history = log_R  # Store the full run length distribution
    
    # Calculate change point probabilities for visualization
    cp_probs = np.zeros(T)
    max_cp_prob = 0
    for t in range(T):
        if t == 0:
            cp_probs[t] = 0
        else:
            # Get maximum probability across all run lengths at time t
            cp_probs[t] = np.max(run_length_prob[:t+1, t])
            max_cp_prob = max(max_cp_prob, cp_probs[t])
            
            # Also track which run length had the maximum probability
            max_rl = np.argmax(run_length_prob[:t+1, t])
            if cp_probs[t] > detect_threshold:
                print(f"t={t}: max prob {cp_probs[t]:.3f} at run length {max_rl}")

    print(f"\nChange Point Analysis:")
    print(f"Maximum probability: {max_cp_prob:.3f}")
    print(f"Number of high probability points > {detect_threshold}: {np.sum(cp_probs > detect_threshold)}")
    print(f"Mean probability: {np.mean(cp_probs):.3f}")
    print(f"Std probability: {np.std(cp_probs):.3f}")

    # Create custom colormap
    colors = [(0, 0, 0.5), (0, 0, 1), (1, 1, 0)]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    # Set up the figure with three subplots
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(3, 1, height_ratios=[1, 1, 1.5], hspace=0.3)

    # Top plot: Data and change points
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data, 'k-', linewidth=1, alpha=0.7, label="Time Series")
    
    # Add change point vertical lines with annotations
    for cp in change_points:
        ax1.axvline(x=cp, color='r', linestyle='--', alpha=0.4, linewidth=1.5)
        ax1.annotate(f'CP: {cp}', 
                    xy=(cp, ax1.get_ylim()[1]),
                    xytext=(cp, ax1.get_ylim()[1] + 0.5),
                    ha='center',
                    va='bottom',
                    rotation=0,
                    fontsize=8,
                    arrowprops=dict(arrowstyle='->',
                                  connectionstyle="arc3,rad=0",
                                  alpha=0.6))

    ax1.set_title("Time Series with Change Points", pad=20, fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_ylabel("Value", fontsize=10)

    # Middle plot: Maximum probabilities
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(cp_probs, 'b-', linewidth=1.5, alpha=0.7, label="Max Probability")
    ax2.axhline(y=detect_threshold, color='r', linestyle='--', alpha=0.5, 
                label=f'Threshold ({detect_threshold})')
    
    # Shade the burn-in period
    ax2.axvspan(0, burn_in, color='gray', alpha=0.2, label='Burn-in Period')
    
    # Add vertical lines for detected change points
    for cp in change_points:
        ax2.axvline(x=cp, color='r', linestyle='--', alpha=0.4)
        ax2.plot(cp, cp_probs[cp], 'ro', alpha=0.6)  # Add red dot at detection point

    ax2.set_title("Maximum Run Length Probabilities", pad=20, fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_ylabel("Probability", fontsize=10)
    ax2.set_ylim(-0.05, 1.05)  # Add some padding to y-axis

    # Bottom plot: Run length distribution
    ax3 = fig.add_subplot(gs[2])
    im = ax3.imshow(run_length_prob,
                    aspect='auto',
                    origin='lower',
                    cmap=cmap,
                    extent=[0, T, 0, T//2],
                    interpolation='nearest')

    # Add vertical lines for change points in heatmap
    for cp in change_points:
        ax3.axvline(x=cp, color='r', linestyle='--', alpha=0.4)

    ax3.set_title("Run Length Distribution", pad=20, fontsize=12)
    ax3.set_xlabel("Time", fontsize=10)
    ax3.set_ylabel("Run Length", fontsize=10)
    ax3.grid(True, alpha=0.3, color='white', linestyle='-', linewidth=0.5)

    # Add colorbar
    plt.colorbar(im, ax=ax3, label='Probability')

    plt.tight_layout()
    plt.show()

    # Print additional statistics
    print("\nDetection Results:")
    print("-" * 50)
    print(f"Total time points: {T}")
    print(f"Number of change points detected: {len(change_points)}")
    print("\nChange point locations:")
    for i, cp in enumerate(change_points, 1):
        if i > 1:
            distance = cp - change_points[i-2]
            print(f"CP {i}: {cp} (distance from previous: {distance})")
        else:
            print(f"CP {i}: {cp}")
    print(f"\nFinal model evidence: {evidence[-1]:.2e}")
    
    # Print run length distribution statistics
    print("\nRun Length Distribution Statistics:")
    print("-" * 50)
    print(f"Max run length probability: {np.max(run_length_prob):.3f}")
    print(f"Mean run length probability: {np.mean(run_length_prob):.3f}")
    print(f"Number of active hypotheses: {np.sum(run_length_prob > 1e-12)}")
    
    return run_length_prob, change_points, np.exp(log_evidence), log_R


###############################################################################
# Demo / Example Usage
###############################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Generate synthetic dataset with more pronounced changes
    np.random.seed(42)
    data_segment1 = np.random.normal(loc=0.0, scale=1.0, size=30)
    data_segment2 = np.random.normal(loc=8.0, scale=1.2, size=30)
    data_segment3 = np.random.normal(loc=-5.0, scale=0.8, size=30)
    data = np.concatenate([data_segment1, data_segment2, data_segment3])
    T = len(data)

    # Algorithm parameters
    detect_threshold = 0.95
    burn_in = 15
    min_distance = 20

    print("\nDataset Statistics:")
    print(f"Total length: {T}")
    print(f"Segment means: {np.mean(data_segment1):.2f}, {np.mean(data_segment2):.2f}, {np.mean(data_segment3):.2f}")
    print(f"Segment stds: {np.std(data_segment1):.2f}, {np.std(data_segment2):.2f}, {np.std(data_segment3):.2f}")
    print(f"\nParameters:")
    print(f"Detection threshold: {detect_threshold}")
    print(f"Burn-in period: {burn_in}")
    print(f"Minimum distance: {min_distance}")

    # Run detection
    run_length_prob, change_points, evidence, log_joint_prob = kbocd_nonparametric_hazard(
        data,
        detect_threshold=detect_threshold,
        burn_in=burn_in,
        min_distance=min_distance
    )

    # Calculate probabilities for visualization
    cp_probs = np.zeros(T)
    max_cp_prob = 0
    for t in range(T):
        if t == 0:
            cp_probs[t] = 0
        else:
            # Get maximum probability across all run lengths at time t
            probs = run_length_prob[:t+1, t]
            max_idx = np.argmax(probs)
            cp_probs[t] = probs[max_idx]
            max_cp_prob = max(max_cp_prob, cp_probs[t])
            
            # Print high probability points
            if cp_probs[t] > 0.8:  # Lower threshold for informative printing
                print(f"t={t}: max prob {cp_probs[t]:.3f} at run length {max_idx}")

    print(f"\nProbability Analysis:")
    print(f"Maximum probability: {max_cp_prob:.3f}")
    print(f"Number of high probability points > {detect_threshold}: {np.sum(cp_probs > detect_threshold)}")
    print(f"Mean probability: {np.mean(cp_probs):.3f}")
    print(f"Std probability: {np.std(cp_probs):.3f}")

    # Create custom colormap
    colors = [(0, 0, 0.5), (0, 0, 1), (1, 1, 0)]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    # Set up the figure with three subplots
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(3, 1, height_ratios=[1, 1, 1.5], hspace=0.3)

    # Top plot: Data and change points
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data, 'k-', linewidth=1, alpha=0.7, label="Time Series")
    
    # Add change point vertical lines with annotations
    for cp in change_points:
        ax1.axvline(x=cp, color='r', linestyle='--', alpha=0.4, linewidth=1.5)
        ax1.annotate(f'CP: {cp}', 
                    xy=(cp, ax1.get_ylim()[1]),
                    xytext=(cp, ax1.get_ylim()[1] + 0.5),
                    ha='center',
                    va='bottom',
                    rotation=0,
                    fontsize=8,
                    arrowprops=dict(arrowstyle='->',
                                  connectionstyle="arc3,rad=0",
                                  alpha=0.6))

    ax1.set_title("Time Series with Change Points", pad=20, fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_ylabel("Value", fontsize=10)

    # Middle plot: Maximum probabilities
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(cp_probs, 'b-', linewidth=1.5, alpha=0.7, label="Max Probability")
    ax2.axhline(y=detect_threshold, color='r', linestyle='--', alpha=0.5, 
                label=f'Threshold ({detect_threshold})')
    
    # Shade the burn-in period
    ax2.axvspan(0, burn_in, color='gray', alpha=0.2, label='Burn-in Period')
    
    # Add vertical lines for detected change points
    for cp in change_points:
        ax2.axvline(x=cp, color='r', linestyle='--', alpha=0.4)
        ax2.plot(cp, cp_probs[cp], 'ro', alpha=0.6)  # Add red dot at detection point

    ax2.set_title("Maximum Run Length Probabilities", pad=20, fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_ylabel("Probability", fontsize=10)
    ax2.set_ylim(-0.05, 1.05)  # Add some padding to y-axis

    # Bottom plot: Run length distribution
    ax3 = fig.add_subplot(gs[2])
    im = ax3.imshow(run_length_prob,
                    aspect='auto',
                    origin='lower',
                    cmap=cmap,
                    extent=[0, T, 0, T//2],
                    interpolation='nearest')

    # Add vertical lines for change points in heatmap
    for cp in change_points:
        ax3.axvline(x=cp, color='r', linestyle='--', alpha=0.4)

    ax3.set_title("Run Length Distribution", pad=20, fontsize=12)
    ax3.set_xlabel("Time", fontsize=10)
    ax3.set_ylabel("Run Length", fontsize=10)
    ax3.grid(True, alpha=0.3, color='white', linestyle='-', linewidth=0.5)

    # Add colorbar
    plt.colorbar(im, ax=ax3, label='Probability')

    plt.tight_layout()
    plt.show()

    # Print additional statistics
    print("\nDetection Results:")
    print("-" * 50)
    print(f"Total time points: {T}")
    print(f"Number of change points detected: {len(change_points)}")
    print("\nChange point locations:")
    for i, cp in enumerate(change_points, 1):
        if i > 1:
            distance = cp - change_points[i-2]
            print(f"CP {i}: {cp} (distance from previous: {distance})")
        else:
            print(f"CP {i}: {cp}")
    print(f"\nFinal model evidence: {evidence[-1]:.2e}")
    
    # Print run length distribution statistics
    print("\nRun Length Distribution Statistics:")
    print("-" * 50)
    print(f"Max run length probability: {np.max(run_length_prob):.3f}")
    print(f"Mean run length probability: {np.mean(run_length_prob):.3f}")
    print(f"Number of active hypotheses: {np.sum(run_length_prob > 1e-12)}")