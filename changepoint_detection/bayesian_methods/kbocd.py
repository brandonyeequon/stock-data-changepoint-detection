#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fully Nonparametric Bayesian Online Change Point Detection Using Kernel Density Estimation
with Nonparametric Hazard Function (r, a) tracking and Silverman's rule for bandwidth selection,
plus some modifications to reduce early false-positives and capture later major shifts more reliably.
"""

import numpy as np
import warnings
from math import log, exp, sqrt, pi, isinf
from typing import Tuple, List

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
# KDE predictive probability
###############################################################################
def kde_predictive_prob(x_new: float, data_seg: np.ndarray, bandwidth: float) -> float:
    """
    Gaussian KDE-based predictive probability at x_new, given data_seg.
    """
    data_seg = np.asarray(data_seg, dtype=float)
    n = len(data_seg)
    if n < 2:
        # If segment is too small, can't reliably do KDE
        return 1e-12

    z = (x_new - data_seg) / bandwidth
    kvals = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return max(1e-12, np.mean(kvals) / bandwidth)

###############################################################################
# Main KBOCD w/ Nonparametric Hazard (r,a) Tracking
###############################################################################
def kbocd_nonparametric_hazard(
    data: np.ndarray,
    detect_threshold: float = 0.3,
    burn_in: int = 20,
    min_distance: int = 30
) -> Tuple[np.ndarray, List[int], np.ndarray, np.ndarray]:
    """
    A fully nonparametric Bayesian online change point detection method.
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

    # Calculate bandwidth using Silverman's rule
    init_segment = data[: min(50, T)]
    print(f"Initial segment stats: mean={np.mean(init_segment):.3f}, std={np.std(init_segment):.3f}")
    
    std_init = np.std(init_segment)
    if std_init < 1e-12:
        kernel_bw = 1.0
        print("Warning: Very small standard deviation, using default bandwidth=1.0")
    else:
        iqr_init = np.percentile(init_segment, 75) - np.percentile(init_segment, 25)
        if iqr_init < 1e-12:
            iqr_init = std_init
            print("Warning: Small IQR, falling back to standard deviation")
        kernel_bw = 0.9 * min(std_init, iqr_init / 1.34) * (len(init_segment) ** (-1 / 5))
        print(f"Using Silverman's rule bandwidth: {kernel_bw:.3f}")

    # Initialize with more informative debug output
    log_joint_prob = np.full((T+1, T+1, T), -np.inf, dtype=float)
    log_joint_prob[0, 0, 0] = log(0.1)
    log_joint_prob[1, 0, 0] = log(0.9)
    print(f"Initialized with continue prob: {0.9:.2f}, change prob: {0.1:.2f}")

    change_points = []
    log_evidence = np.full(T, -np.inf, dtype=float)
    log_evidence[0] = log(1.0)

    # Add debug counters
    total_hypotheses = 0
    valid_hypotheses = 0
    
    # Modified predictive probability with better error handling
    def compute_pred_prob(x_t: float, seg_data: np.ndarray, t: int) -> float:
        try:
            if len(seg_data) < 2:
                window_size = min(50, t)
                recent_data = data[max(0, t-window_size):t]
                if len(recent_data) > 1:
                    mu = np.mean(recent_data)
                    sigma = max(np.std(recent_data), 1e-12)  # Ensure non-zero std
                    z = (x_t - mu) / sigma
                    return max(1e-12, np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * pi)))
                return 1e-12
            return kde_predictive_prob(x_t, seg_data, kernel_bw)
        except Exception as e:
            print(f"Warning: Error in pred_prob calculation: {e}")
            return 1e-12

    # Modified hazard rate calculation
    def compute_hazard(r_prev: int, a_prev: int, t: int) -> float:
        """More conservative hazard rate calculation."""
        if t < burn_in:
            return 0.01  # Very low hazard during burn-in
            
        base_h = float(a_prev + 1) / float(t + 1)
        
        # More conservative run length factor
        run_factor = min(1.0, np.sqrt(r_prev + 1) / (t + 1))
        
        # Increase hazard for very long runs
        long_run_factor = max(0, (r_prev - 100) / 100) if r_prev > 100 else 0
        
        h = base_h * (1.0 + run_factor + long_run_factor)
        
        # More conservative bounds
        return min(max(h, 0.01), 0.5)

    # Main loop with progress tracking
    print("\nStarting main detection loop...")
    for t in range(T):
        if t % 50 == 0:  # Progress update every 50 steps
            print(f"Processing time step {t}/{T}")
        
        x_t = data[t]
        if t == 0:
            continue

        current_valid_hypotheses = 0
        
        # Process hypotheses
        for r_prev in range(t+1):
            for a_prev in range(t+1):
                total_hypotheses += 1
                
                old_lp = log_joint_prob[r_prev, a_prev, t-1]
                if isinf(old_lp):
                    continue

                current_valid_hypotheses += 1
                valid_hypotheses += 1

                # Compute probabilities with error checking
                try:
                    seg_start = max(0, t - r_prev)
                    seg_data = data[seg_start:t]
                    p_x = compute_pred_prob(x_t, seg_data, t)
                    log_p_x = log(max(p_x, 1e-12))  # Ensure valid log

                    h = compute_hazard(r_prev, a_prev, t)
                    
                    # Growth probabilities
                    r_new = r_prev + 1
                    if r_new <= T:
                        lp_growth = old_lp + log_p_x + log(1.0 - h)
                        log_joint_prob[r_new, a_prev, t] = np.logaddexp(
                            log_joint_prob[r_new, a_prev, t],
                            lp_growth
                        )

                    # Change point probabilities
                    r_new = 0
                    a_new = a_prev + 1
                    if a_new <= T:
                        lp_cp = old_lp + log_p_x + log(h)
                        log_joint_prob[r_new, a_new, t] = np.logaddexp(
                            log_joint_prob[r_new, a_new, t],
                            lp_cp
                        )
                except Exception as e:
                    print(f"Warning: Error processing hypothesis at t={t}, r={r_prev}, a={a_prev}: {e}")
                    continue

        if current_valid_hypotheses == 0:
            print(f"Warning: No valid hypotheses at t={t}")

        # Normalize
        finite_vals = log_joint_prob[:, :, t][~np.isinf(log_joint_prob[:, :, t])]
        if finite_vals.size > 0:
            ls = log_sum_exp(finite_vals)
            log_evidence[t] = ls
            log_joint_prob[:, :, t] -= ls

        # Improved change point detection with minimum distance enforcement
        if t >= burn_in:
            cp_slice = log_joint_prob[0, :, t]
            cp_slice_finite = cp_slice[~np.isinf(cp_slice)]
            if cp_slice_finite.size > 0:
                cp_prob = exp(log_sum_exp(cp_slice_finite))
                
                # Check if we're far enough from the last change point
                far_enough = True
                if change_points:
                    far_enough = (t - change_points[-1]) >= min_distance
                
                # Additional magnitude check
                if far_enough and cp_prob > detect_threshold:
                    # Check if the change is significant enough
                    pre_window = data[max(0, t-20):t]
                    post_window = data[t:min(T, t+20)]
                    if len(pre_window) > 0 and len(post_window) > 0:
                        pre_mean = np.mean(pre_window)
                        post_mean = np.mean(post_window)
                        pre_std = np.std(pre_window)
                        post_std = np.std(post_window)
                        
                        # Require significant difference in mean or variance
                        mean_diff = abs(post_mean - pre_mean)
                        std_diff = abs(post_std - pre_std)
                        if (mean_diff > 1.5 * max(pre_std, post_std) or 
                            std_diff > 0.5 * max(pre_std, post_std)):
                            change_points.append(t)

    # Final statistics
    print("\nAlgorithm completed:")
    print(f"Total hypotheses considered: {total_hypotheses}")
    print(f"Valid hypotheses processed: {valid_hypotheses}")
    print(f"Change points detected: {len(change_points)}")
    
    # Convert to run length probabilities
    run_length_prob = np.zeros((T+1, T), dtype=float)
    for t in range(T):
        for r in range(T+1):
            slice_ = log_joint_prob[r, :, t]
            finite_slice = slice_[~np.isinf(slice_)]
            if finite_slice.size > 0:
                run_length_prob[r, t] = exp(log_sum_exp(finite_slice))

    evidence = np.exp(log_evidence)
    return run_length_prob, change_points, evidence, log_joint_prob


###############################################################################
# Demo / Example Usage
###############################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Generate synthetic dataset with more pronounced changes
    np.random.seed(42)
    data_segment1 = np.random.normal(loc=0.0, scale=1.0, size=80)
    data_segment2 = np.random.normal(loc=8.0, scale=1.2, size=80)
    data_segment3 = np.random.normal(loc=-5.0, scale=0.8, size=80)
    data = np.concatenate([data_segment1, data_segment2, data_segment3])
    T = len(data)

    # Algorithm parameters
    detect_threshold = 0.3
    burn_in = 20
    min_distance = 30

    print("\nDataset Statistics:")
    print(f"Total length: {T}")
    print(f"Segment means: {np.mean(data_segment1):.2f}, {np.mean(data_segment2):.2f}, {np.mean(data_segment3):.2f}")
    print(f"Segment stds: {np.std(data_segment1):.2f}, {np.std(data_segment2):.2f}, {np.std(data_segment3):.2f}")
    print(f"\nParameters:")
    print(f"Detection threshold: {detect_threshold}")
    print(f"Burn-in period: {burn_in}")
    print(f"Minimum distance: {min_distance}")

    # Run detection with updated unpacking
    run_length_prob, change_points, evidence, log_joint_prob = kbocd_nonparametric_hazard(
        data,
        detect_threshold=detect_threshold,
        burn_in=burn_in,
        min_distance=min_distance
    )

    # Calculate change point probabilities for visualization
    cp_probs = np.zeros(T)
    max_cp_prob = 0
    for t in range(T):
        if t == 0:
            cp_probs[t] = 0
        else:
            cp_slice = log_joint_prob[0, :, t]
            cp_slice_finite = cp_slice[~np.isinf(cp_slice)]
            if cp_slice_finite.size > 0:
                cp_probs[t] = np.exp(log_sum_exp(cp_slice_finite))
                max_cp_prob = max(max_cp_prob, cp_probs[t])

    print(f"\nChange Point Analysis:")
    print(f"Maximum CP probability: {max_cp_prob:.3f}")
    print(f"Number of probabilities > {detect_threshold}: {np.sum(cp_probs > detect_threshold)}")
    print(f"Mean CP probability: {np.mean(cp_probs):.3f}")
    print(f"Std CP probability: {np.std(cp_probs):.3f}")

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

    # Middle plot: Change point probabilities
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(cp_probs, 'b-', linewidth=1.5, alpha=0.7, label="CP Probability")
    ax2.axhline(y=detect_threshold, color='r', linestyle='--', alpha=0.5, 
                label=f'Threshold ({detect_threshold})')
    
    # Shade the burn-in period
    ax2.axvspan(0, burn_in, color='gray', alpha=0.2, label='Burn-in Period')
    
    # Add vertical lines for detected change points
    for cp in change_points:
        ax2.axvline(x=cp, color='r', linestyle='--', alpha=0.4)
        ax2.plot(cp, cp_probs[cp], 'ro', alpha=0.6)  # Add red dot at detection point

    ax2.set_title("Change Point Probabilities", pad=20, fontsize=12)
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

    # Adjust layout
    plt.tight_layout()

    # Add text box with statistics
    stats_text = (f"Number of change points: {len(change_points)}\n"
                 f"Change points: {', '.join(map(str, change_points))}\n"
                 f"Final evidence: {evidence[-1]:.2e}")
    
    fig.text(0.02, 0.02, stats_text,
             fontsize=8,
             family='monospace',
             bbox=dict(facecolor='white',
                      alpha=0.8,
                      edgecolor='none',
                      boxstyle='round,pad=0.5'))

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