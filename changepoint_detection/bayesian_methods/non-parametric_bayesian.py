#!/usr/bin/env python3
"""
Hierarchical KBOCD with Log-Binned r1, r2 + Threshold-Pruned (a1, a2)
=====================================================================
- r1: top-level run length    => log2 bin => b1
- r2: data-level run length   => log2 bin => b2
- a1: top-level CP count      => up to T (but pruned)
- a2: data-level CP count     => up to T (but pruned)

At each time step:
  1) Message passing to compute new_w.
  2) Zero out new_w below prune_threshold (full-array).
  3) Sum over (b1,b2) for each (a1,a2). If that sum < prune_threshold, zero out that slice.
  4) Renormalize again.
  5) Store hazard_post[t] = new_w
  6) Summarize run_posterior_data[t, b2] = sum_{b1,a1,a2} hazard_post[t,b1,a1,b2,a2].

A KDE-based predictive distribution is used (still expensive vs. parametric).
"""

import numpy as np
import math
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# 1) Gaussian KDE + Bandwidth
# ============================================================================
def gaussian_kernel(u):
    return np.exp(-0.5 * u*u) / math.sqrt(2.0 * math.pi)

def kde_predictive_prob(x_new, data_seg, bw):
    data_seg = np.asarray(data_seg)
    n = len(data_seg)
    if n < 1:
        return 1e-12  # fallback
    if bw < 1e-12:
        bw = 1e-3
    diffs = (x_new - data_seg) / bw
    vals = gaussian_kernel(diffs)
    return vals.sum() / (n * bw)

def improved_sheather_jones_bandwidth(data, max_iter=100, tol=1e-7):
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return 1.0
    std_data = np.std(data)
    if std_data < 1e-15:
        return 1e-3

    x = (data - np.mean(data)) / std_data
    z = 0.9  # initial guess

    def derivative_moment_approx(_x, lam):
        return 1.0 + 0.1 * np.mean(_x**2)

    def fixed_point(z_current):
        # Very rough approximation
        k = 5
        denom = derivative_moment_approx(x, z_current)
        return 0.9 * (1.0 / denom)**(2.0 / (3 + 2*k))

    for _ in range(max_iter):
        z_next = fixed_point(z)
        if abs(z_next - z) < tol:
            z = z_next
            break
        z = z_next

    return z * std_data

# ============================================================================
# 2) Beta Hazard Functions
# ============================================================================
def hazard_prob_cp_level1(a1_prev, t, alpha1, beta1):
    """
    Probability top-level hazard changes at time t:
      h^(1)_t = (a1_prev + alpha1) / ((t-1) + alpha1 + beta1)
    """
    denom = (t - 1) + alpha1 + beta1
    if denom < 1e-12:
        return 1e-12
    return float(a1_prev + alpha1) / float(denom)

def hazard_prob_cp_level2(a2_prev, t, alpha2, beta2):
    """
    Probability data-level hazard changes at time t:
      h^(2)_t = (a2_prev + alpha2) / ((t-1) + alpha2 + beta2)
    """
    denom = (t - 1) + alpha2 + beta2
    if denom < 1e-12:
        return 1e-12
    return float(a2_prev + alpha2) / float(denom)

# ============================================================================
# 3) Threshold Pruning
# ============================================================================
def prune_by_threshold(arr, threshold=1e-12):
    """
    Zero out entries < threshold, then renormalize. Returns new total mass.
    """
    mask = (arr < threshold)
    arr[mask] = 0.0
    total_mass = arr.sum()
    if total_mass > 0:
        arr /= total_mass
    else:
        # fallback => reset to a single nonzero entry
        arr[...] = 0.0
        arr.flat[0] = 1.0
        total_mass = 1.0
    return total_mass

def prune_a1_a2_slices(arr, t, threshold=1e-12):
    """
    For each (a1, a2) in [0..t] x [0..t], sum over (b1,b2). 
    If that sum < threshold, zero out that entire (b1, b2) slice.

    arr shape = (nbins1, T+1, nbins2, T+1)
      index 0 => b1
      index 1 => a1
      index 2 => b2
      index 3 => a2
    """
    # Summation over (b1, b2)
    mass_per_a1a2 = arr.sum(axis=(0,2))  # shape => (T+1, T+1)

    for a1_idx in range(t+1):
        for a2_idx in range(t+1):
            if mass_per_a1a2[a1_idx, a2_idx] < threshold:
                arr[:, a1_idx, :, a2_idx] = 0.0

# ============================================================================
# 4) Log-Bin Utilities (for r1, r2)
# ============================================================================
def log_bin_index(r):
    """Return floor(log2(r)) if r>0, else 0 for r=0."""
    if r < 1:
        return 0
    return int(np.floor(np.log2(r)))

def build_bin_map(max_r_length):
    """
    Precompute bin indices for r in [0..max_r_length].
    Also define a 'representative' run length for each bin.
    """
    bin_map = []
    for r in range(max_r_length+1):
        b = log_bin_index(r)
        bin_map.append(b)
    bin_map = np.array(bin_map, dtype=int)

    max_bin = bin_map.max()
    nbins = max_bin + 1

    # representative run length for each bin = midpoint of [2^b, 2^(b+1)-1]
    # (capped at max_r_length)
    rep_r = np.zeros(nbins, dtype=float)
    for b in range(nbins):
        low = 2**b
        high = 2**(b+1) - 1
        if high > max_r_length:
            high = max_r_length
        rep_r[b] = 0.5 * (low + high)
    return bin_map, rep_r

# ============================================================================
# 5) Main Hierarchical KBOCD with Binned r1 & r2 + (a1,a2) pruning
# ============================================================================
def kbocd_hierarchical_logbin_r1_r2_prune_a1a2(
    data,
    alpha1=0.5, beta1=5.0,   # top-level hazard Beta
    alpha2=0.5, beta2=5.0,   # data-level hazard Beta
    detect_threshold=0.7,
    skip_first=5,
    use_isj=True,
    const_bw=0.5,
    max_r_length=None,
    prune_threshold=1e-12
):
    """
    Hierarchical KBOCD with:
    - Log-binned r1 => b1
    - Log-binned r2 => b2
    - Threshold pruning of entire array
    - Additional threshold pruning of (a1,a2) slices
    
    hazard_post[t, b1, a1, b2, a2]:
      b1 => bin index for r1
      a1 => top-level CP count
      b2 => bin index for r2
      a2 => data-level CP count
    
    We'll interpret b2=0 as r2 in [0..1], so p_r2_0 is approx the mass in that bin.
    """
    data = np.asarray(data)
    T = len(data)
    if max_r_length is None:
        max_r_length = T

    # (A) Build bin maps for r1, r2
    bin_map1, rep_r1 = build_bin_map(max_r_length)  # top-level
    nbins1 = len(rep_r1)
    bin_map2, rep_r2 = build_bin_map(max_r_length)  # data-level
    nbins2 = len(rep_r2)

    # (B) hazard_post[t, b1, a1, b2, a2]
    hazard_post = np.zeros((T, nbins1, T+1, nbins2, T+1), dtype=float)

    # run_posterior_data[t, b2]
    run_posterior_data = np.zeros((T, nbins2))

    # Initialization at t=0
    hazard_post[0,0,0,0,0] = 1.0

    #----------------------------------------
    # MAIN LOOP
    #----------------------------------------
    for t in range(T):
        if t == 0:
            continue

        x_t = data[t]
        new_w = np.zeros_like(hazard_post[t])  # shape => (nbins1, T+1, nbins2, T+1)

        #----------------------------------------
        # (1) Message Passing
        #----------------------------------------
        for b1_prev in range(nbins1):
            for a1_prev in range(t+1):
                for b2_prev in range(nbins2):
                    for a2_prev in range(t+1):
                        w_val = hazard_post[t-1, b1_prev, a1_prev, b2_prev, a2_prev]
                        if w_val < 1e-30:
                            continue

                        # Effective run lengths
                        r1_eff = int(rep_r1[b1_prev])
                        r2_eff = int(rep_r2[b2_prev])

                        # Predictive prob from data-level run length
                        seg_start = (t - 1) - r2_eff
                        if seg_start < 0:
                            seg_start = 0
                        seg = data[seg_start : t]

                        if use_isj:
                            bw = improved_sheather_jones_bandwidth(seg)
                        else:
                            bw = const_bw
                        p_x = kde_predictive_prob(x_t, seg, bw)

                        # Hazards
                        pc1 = hazard_prob_cp_level1(a1_prev, t, alpha1, beta1)
                        pg1 = 1.0 - pc1
                        pc2 = hazard_prob_cp_level2(a2_prev, t, alpha2, beta2)
                        pg2 = 1.0 - pc2

                        # Next states
                        # (a) no top-level CP, no data-level CP
                        r1_eff_new = r1_eff + 1
                        b1_new = bin_map1[min(r1_eff_new, max_r_length)]
                        r2_eff_new = r2_eff + 1
                        b2_new = bin_map2[min(r2_eff_new, max_r_length)]
                        a1_new = a1_prev
                        a2_new = a2_prev
                        if a1_new <= T and a2_new <= T:
                            val = w_val * pg1 * pg2 * p_x
                            new_w[b1_new, a1_new, b2_new, a2_new] += val

                        # (b) no top-level CP, data-level CP => b2=0
                        a1_new = a1_prev
                        a2_new = a2_prev + 1
                        if a1_new <= T and a2_new <= T:
                            val = w_val * pg1 * pc2 * p_x
                            new_w[b1_new, a1_new, 0, a2_new] += val

                        # (c) top-level CP, no data-level CP => b1=0
                        r1_new = 0
                        a1_new = a1_prev + 1
                        a2_new = a2_prev
                        if a1_new <= T and a2_new <= T:
                            val = w_val * pc1 * pg2 * p_x
                            new_w[0, a1_new, b2_new, a2_new] += val

                        # (d) top-level CP, data-level CP => b1=0, b2=0
                        a1_new = a1_prev + 1
                        a2_new = a2_prev + 1
                        if a1_new <= T and a2_new <= T:
                            val = w_val * pc1 * pc2 * p_x
                            new_w[0, a1_new, 0, a2_new] += val

        #----------------------------------------
        # (2) Prune + Normalize
        #----------------------------------------
        # (2a) Full array threshold
        prune_by_threshold(new_w, threshold=prune_threshold)

        # (2b) Threshold prune (a1,a2) slices
        #      If sum_{b1,b2} < prune_threshold, zero them out
        prune_a1_a2_slices(new_w, t, threshold=prune_threshold)

        # (2c) Renormalize again (in case we just zeroed out slices)
        prune_by_threshold(new_w, threshold=prune_threshold)

        hazard_post[t] = new_w

        #----------------------------------------
        # (3) Summarize data-level bin => run_post_data[t, b2]
        #----------------------------------------
        # sum over (b1, a1, a2) => leftover dimension is b2
        run_posterior_data[t] = new_w.sum(axis=(0,1,3))

    # Probability r2~[0..1] => b2=0
    p_r2_bin0 = run_posterior_data[:,0]

    # CP detection rule
    detected_cps = []
    for ti in range(skip_first, T):
        if p_r2_bin0[ti] > detect_threshold:
            detected_cps.append(ti)

    return hazard_post, run_posterior_data, detected_cps, p_r2_bin0

# ============================================================================
# 6) Demo + Visualization
# ============================================================================
if __name__ == "__main__":
    np.random.seed(123)

    # Synthetic data with multiple distinct segments
    N1, N2, N3 = 50, 50, 50
    seg1 = np.random.normal(0, 0.5, N1)          # Normal(0, 0.5)
    seg2 = np.random.exponential(scale=2.0, size=N2) + 5  # Exp + shift
    seg3 = np.random.normal(10, 2.0, N3)         # Normal(10, 2.0)
    data = np.concatenate([seg1, seg2, seg3])
    true_cps = [N1, N1+N2]  # [50, 100]

    # Hierarchical KBOCD with log-binned r1, r2, plus (a1,a2) pruning
    alpha1, beta1 = 0.05, 5
    alpha2, beta2 = 0.05, 5
    detect_threshold = 0.9
    skip_first = 10

    hazard_post, run_post_data, detected_cps, p_r2_bin0 = kbocd_hierarchical_logbin_r1_r2_prune_a1a2(
        data,
        alpha1=alpha1, beta1=beta1,
        alpha2=alpha2, beta2=beta2,
        detect_threshold=detect_threshold,
        skip_first=skip_first,
        use_isj=True,
        const_bw=1.0,
        max_r_length=None,
        prune_threshold=1e-12
    )

    print("Detected changes:", detected_cps)
    print("True CP indices:", true_cps)

    # ~~~ Visualization ~~~
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # (1) Plot data + CP lines
    axs[0].plot(data, color='C0', label="Data")
    for cp in true_cps:
        axs[0].axvline(cp, color='g', linestyle='--',
                       label="True CP" if cp == true_cps[0] else None)
    for cp in detected_cps:
        axs[0].axvline(cp, color='r', linestyle=':',
                       label="Detected CP" if cp == detected_cps[0] else None)
    axs[0].set_title("Time Series (Hierarchical, Log-binned r1 & r2, Pruned a1,a2)")
    axs[0].legend(loc='upper right')

    # (2) Heatmap of data-level bin posterior
    # run_post_data[t, b2], i.e. bin index for r2
    # Show up to bin=6 => covers r2 up to 63
    max_b2_plot = 6
    im = axs[1].imshow(
        run_post_data[:, :max_b2_plot].T,
        origin='lower',
        aspect='auto',
        cmap='viridis',
        extent=[0, len(data), 0, max_b2_plot]
    )
    axs[1].set_title("Data-Level Run-length Posterior (Binned) P(b2)")
    axs[1].set_ylabel("bin index for r2 (log2 scale)")
    fig.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.05)

    # (3) Probability r2 in bin=0 => run lengths ~ [0..1]
    axs[2].plot(p_r2_bin0, color='C2', label="P(r2-bin=0)")
    axs[2].axhline(detect_threshold, color='gray', linestyle='--',
                   label=f"Threshold={detect_threshold:.2f}")
    for cp in detected_cps:
        axs[2].axvline(cp, color='r', linestyle=':', alpha=0.7)
    axs[2].set_title("Probability data-level run-length in [0..1]")
    axs[2].set_xlabel("Time index t")
    axs[2].set_ylabel("P(bin2=0)")
    axs[2].legend()

    plt.tight_layout()
    plt.show()
