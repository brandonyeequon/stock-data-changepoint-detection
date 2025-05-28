#!/usr/bin/env python3
"""
Fully Nonparametric Bayesian Online Change Point Detection
===========================================================
This script implements the single-level, fully nonparametric change point detection
method described in the referenced paper. It incorporates:

  1) Kernel density estimation (KDE) as the underlying predictive model (UPM),
     removing the need to assume any parametric data distribution.
  2) Adaptive bandwidth selection via the Improved Sheather-Jones algorithm.
  3) A nonparametric hazard rate estimator (see Wilson et al. [26]) so no fixed
     hazard parameter is required.

Compared to a classical (parametric) BOCD, we now track two indices at each time t:
  • r(t)  : run length  = # of time steps since the last change point
  • a(t)  : count of how many change points have occurred so far

Hence, at each time t we maintain weights w[t, r, a]. The hazard for each run length
r is computed as a Bernoulli parameter (a+1)/(t+1). If a change is declared, the next
time step transitions to run length = 0, with a(t+1) = a(t)+1.

References from the paper:

• Adams, R. P., & MacKay, D. J. C. (2007). Bayesian online changepoint detection.
• Botev, Z. I., Grotowski, J. F., & Kroese, D. P. (2010). Kernel density estimation via diffusion.
• Wilson, A. G., Saatci, Y., & Williams, C. K. I. (2010). Prediction with Gaussian processes:
  a sparse representation and fast algorithms.

Note: In practice, for large T, this O(T^2) approach can be slow. More intricate
data structures (e.g. segment trees, log-binning, hierarchical expansions, etc.) can
be used to optimize. This code aims to stay close to the exact formulation outlined
in the paper.
"""
import pandas as pd
import numpy as np
import math
import warnings
from scipy.stats import gaussian_kde
from scipy.sparse import lil_matrix
from multiprocessing import Pool
from functools import lru_cache
from numba import jit
import seaborn as sns
from tqdm import tqdm
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------------------------------------
# 1) Gaussian Kernel and Kernel Density Estimation
# ----------------------------------------------------------------------------------

# Pre-compute constant for better performance
SQRT_2PI = 2.506628274631000  # Pre-computed √2π

#@jit(nopython=True)
def gaussian_kernel(u):
    """
    Compute the standard Gaussian kernel: (1/√2π) * exp(-0.5 * u²)
    
    Parameters:
    -----------
    u : array-like
        Standardized distances ((x - x_i) / bandwidth)
    
    Returns:
    --------
    array-like
        Kernel weights for each point
    """
    return np.exp(-0.5 * u * u) / SQRT_2PI

#@jit(nopython=True, parallel=True)
def kde_predictive_prob(x_new, data_seg, bw):
    """
    Calculate predictive probability p(x_new | data_seg) using kernel density estimation
    with the given bandwidth bw.
    """
    data_seg = np.asarray(data_seg)
    n = len(data_seg)
    
    # Special handling for small segments
    if n <= 2:  # Changed from n < 3 to be more conservative
        if n == 0:
            return 1e-12
        if n == 1:
            return float(gaussian_kernel((x_new - data_seg[0]) / 1.0))
        if n == 2:
            mean = np.mean(data_seg)
            std = max(np.std(data_seg), 0.1)  # Force minimum std for n=2
            return float(gaussian_kernel((x_new - mean) / std) / std)
    
    # For larger segments, use standard KDE
    bw = max(bw, 0.1)  # Much more aggressive minimum bandwidth
    
    try:
        diffs = (x_new - data_seg) / bw
        weights = gaussian_kernel(diffs)
        result = float(np.sum(weights) / (n * bw))
        return max(result, 1e-12)
    except:
        print("error")
        return 1e-12


# ----------------------------------------------------------------------------------
# 2) Improved Sheather-Jones Bandwidth Selection
# ----------------------------------------------------------------------------------
def improved_sheather_jones_bandwidth(data, max_iter=100, tol=None):
    """
    Improved Sheather-Jones algorithm to compute bandwidth for KDE
    in a fully nonparametric way, not assuming data is from a known distribution.
    
    Parameters:
    -----------
    data : array-like
        Input data for bandwidth estimation
    max_iter : int, default=100
        Maximum number of iterations
    tol : float, optional
        Convergence tolerance. If None, uses machine epsilon
        
    Returns:
    --------
    float
        Optimal bandwidth value
        
    References:
    -----------
    Botev, Z. I., Grotowski, J. F., & Kroese, D. P. (2010). 
    Kernel density estimation via diffusion.
    """
    data = np.asarray(data)
    data = data[np.isfinite(data)]  # Remove non-finite values
    N = len(data)
    
    if N < 2:
        return 1.0  # fallback for very small samples
    
    # Standardize data
    data_mean = np.mean(data)
    data_std = np.std(data)
    if data_std < 1e-15:
        return 1e-3  # fallback for nearly constant data
    
    x = (data - data_mean) / data_std
    # Constants from the paper
    xi = 0.907069048686
    l = 7  # we use 7th derivative as in the paper
    
    # Initial bandwidth (Silverman's rule-of-thumb)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    silverman_bw = 0.9 * min(np.std(x), iqr/1.34) * np.power(N, -0.2)
    z_n = silverman_bw
    
    # Convergence tolerance
    eps = np.finfo(float).eps if tol is None else tol
    
    def estimate_derivative_norm(x_in, k, bw):
        """Estimate the L2 norm of k-th derivative."""
        # Build a grid for numerical integration
        grid = np.linspace(x_in.min(), x_in.max(), 512)
        kde = gaussian_kde(x_in, bw_method=bw)
        dens = kde.evaluate(grid)
        
        # Compute k-th derivative via repeated numerical gradient
        derivative = dens.copy()
        for _ in range(k):
            derivative = np.gradient(derivative, grid)
            
        # Return squared L2 norm using trapezoid rule
        return np.trapezoid(derivative**2, grid)
    
    def gamma_l(k, NN, f_deriv_norm):
        """Compute γ[l](k) from the paper."""
        num_1 = 1.0 + np.power(0.5, k + 0.5)
        num_2 = np.prod(np.arange(1, 2 * k, 2))  # 1×3×5×...×(2k-1)
        denom = 3.0 * NN * np.sqrt(np.pi / 2.0) * f_deriv_norm
        return (num_1 * num_2) / denom
    
    # Iterative solution
    for _ in range(max_iter):
        bw_current = z_n
        dnorm = estimate_derivative_norm(x, l, bw_current)
        
        if dnorm <= 0 or np.isnan(dnorm):
            print(f"Warning: Derivative norm is non-positive or NaN: {dnorm}")
            break
            
        gamma_val = gamma_l(l, N, dnorm)
        z_next = xi * np.power(gamma_val, 1.0 / (2*l + 3))
        
        if abs(z_next - z_n) < eps:
            z_n = z_next
            break
            
        z_n = z_next
    
    # Rescale back to original data scale
    h = z_n * data_std
    '''
    if h < 1e-3:
        #return silverman's rule
        print(silverman_bw)
        return silverman_bw
    '''
    #return max(h, 1e-3)  # Ensure positive bandwidth
    return h

# ----------------------------------------------------------------------------------
# 3) Nonparametric Hazard Rate BOCD
# ----------------------------------------------------------------------------------

@lru_cache(maxsize=10000)
def cached_bandwidth(data_key):
    """
    Cached version of bandwidth calculation.
    Uses lru_cache for automatic cache management.
    
    Parameters:
    -----------
    data_key : bytes
        Bytes representation of data array
        
    Returns:
    --------
    float
        Computed bandwidth
    """
    data = np.frombuffer(data_key, dtype=np.float64)
    return improved_sheather_jones_bandwidth(data)

def kbocd_nonparametric(
    data,
    detect_threshold=0.5,
    skip_first=10,
    prune_threshold=1e-8,
    max_run_length=1000,
    dynamic_prune_threshold=1e-5,
    hazard_constant=0.99,
    cool_down_period=20  # New parameter for cool-down
):
    data = np.asarray(data)
    T = len(data)
    max_run_length = min(max_run_length, T)
    
    # Pre-allocate arrays
    w = [lil_matrix((max_run_length+1, max_run_length+1)) for _ in range(T)]
    active_states = [set() for _ in range(T)]
    w[0][0, 0] = 1.0
    
    p_r0 = np.zeros(T)
    p_r0[0] = 1.0
    detected_cps = []
    
    # Pre-compute bandwidths
    bw_dict = {}
    
    # Track metrics for progress bar
    max_weight = 0.0
    total_pruned = 0
    last_change_point = -np.inf  # Initialize to an invalid index
    
    # Add progress bar with more metrics
    pbar = tqdm(range(1, T), desc="Processing time steps", unit="steps")
    
    for t in pbar:
        x_t = data[t]
        
        # Add debug info
        if not np.isfinite(x_t):
            print(f"Warning: Non-finite x_t at t={t}: {x_t}")
            x_t = 0.0  # fallback value
        
        new_w = lil_matrix((max_run_length+1, max_run_length+1))
        new_active = set()
        pruned_this_step = 0
        
        # Process active states and their transitions
        for r_prev, a_prev in active_states[t-1]:
            w_val = w[t-1][r_prev, a_prev]
            if w_val < dynamic_prune_threshold:
                pruned_this_step += 1
                continue
            
            # Update max weight when we find a larger value
            max_weight = max(max_weight, w_val)
            
            # Rest of the existing code...
            hazard = float(a_prev + 1) / float(t + hazard_constant)
            if t < skip_first:
                hazard *= 0.5
                
            growth = 1.0 - hazard
            
            # Get segment and compute KDE with more checks
            seg_start = max(0, (t-1) - r_prev)
            seg = data[seg_start:t]
            
            seg_key = seg.tobytes()
            
            if seg_key not in bw_dict:
                bw = improved_sheather_jones_bandwidth(seg)
                if bw < 1e-6 or not np.isfinite(bw):
                    bw = 1e-3
                bw_dict[seg_key] = bw
            bw = bw_dict[seg_key]
            
            # Extra safety check
            if bw == 0:
                print(f"Warning: Zero bandwidth at t={t}, using fallback")
                bw = 1.0
            
            try:
                p_x = kde_predictive_prob(x_t, seg, bw)
            except Exception as e:
                print(f"Error at t={t}: {str(e)}")
                print(f"Debug info: x_t={x_t}, seg_len={len(seg)}, bw={bw}")
                p_x = 1e-12
            
            # Growth transition
            r_new = r_prev + 1
            if r_new <= max_run_length:
                new_weight = w_val * growth * p_x
                max_weight = max(max_weight, new_weight)
                if new_weight >= dynamic_prune_threshold:
                    new_w[r_new, a_prev] = new_weight
                    new_active.add((r_new, a_prev))
            
            # Change point transition
            if a_prev + 1 <= max_run_length:
                new_weight = w_val * hazard * p_x
                max_weight = max(max_weight, new_weight)
                if new_weight >= dynamic_prune_threshold:
                    new_w[0, a_prev + 1] = new_weight
                    new_active.add((0, a_prev + 1))
        
        total_pruned += pruned_this_step
        
        # Normalize weights
        total_mass = new_w.sum()
        if total_mass > prune_threshold:
            new_w = new_w.tocsr()
            new_w.data /= total_mass
            active_states[t] = new_active
        else:
            new_w = lil_matrix((max_run_length+1, max_run_length+1))
            new_w[0, 0] = 1.0
            active_states[t] = {(0, 0)}
        
        w[t] = new_w
        p_r0[t] = new_w[0, :].sum()
        
        # Change point detection with cool-down
        if (
            t >= skip_first
            and p_r0[t] > detect_threshold
            and (t - last_change_point >= cool_down_period)  # Check cool-down
        ):
            detected_cps.append(t)
            last_change_point = t  # Update last change point
        
        # Update progress bar with current metrics
        pbar.set_postfix({
            'Change Points': len(detected_cps),
            'Active States': len(active_states[t]),
            'Pruned States': total_pruned
        })
    
    pbar.close()
    return w, p_r0, detected_cps


# Helper function for parallel KDE computation
def parallel_kde_compute(args):
    """Helper function for parallel KDE computation."""
    x_t, seg, bw = args
    return kde_predictive_prob(x_t, seg, bw)

def create_synthetic_data():
    """Create synthetic data with known change points"""
    np.random.seed(42)
    N1, N2, N3 = 100, 100, 100
    seg1 = np.random.normal(-1, 1, N1)
    seg2 = np.random.normal(5, 1, N2) 
    seg3 = np.random.normal(-3, 1, N3)
    return np.concatenate([seg1, seg2, seg3])

def prepare_stock_data(filepath, use_log_returns, start_date, end_date):
    """
    Load and prepare stock data with robust normalization.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing stock data
    use_log_returns : bool
        If True, compute log returns instead of using raw prices

    
    Returns:
    --------
    tuple
        (raw_data, normalized_data)
    """
    # Load and prepare stock data
    data = pd.read_csv(filepath)
    # Convert timestamp to datetime and set as index
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    # Resample to daily frequency using last price of each day
    data = data.resample('D').last()
    
    # Remove days with no data
    data = data.dropna()
    
    # Filter data between start and end dates if provided
    # Find nearest available dates if exact dates not found
    start_idx = data.index.searchsorted(start_date)
    end_idx = data.index.searchsorted(end_date)
    
    # Adjust indices to nearest valid dates
    start_idx = min(max(0, start_idx), len(data.index) - 1)
    end_idx = min(max(0, end_idx), len(data.index) - 1)
    
    data = data.iloc[start_idx:end_idx + 1]
        
    data_for_visualization = data

    # Get the close prices
    data = data['close']
    # Convert to numpy array and handle NaN values
    raw_data = data.values
    raw_data = raw_data[~np.isnan(raw_data)]  # Remove NaN values
    
    if use_log_returns:
        # Calculate log returns with epsilon to prevent division by zero
        epsilon = 1e-8
        log_returns = np.log((raw_data[1:] + epsilon) / (raw_data[:-1] + epsilon))
        
        # Robust standardization for log returns
        returns_mean = np.mean(log_returns)
        returns_std = np.std(log_returns)
        if returns_std < epsilon:
            returns_std = 1.0  # Fallback if standard deviation is too small
            
        scaled_data = (log_returns - returns_mean) / (returns_std + epsilon)
    else:
        # Add small noise to raw prices to prevent identical values
        epsilon = np.std(raw_data) * 1e-6  # Scale noise with data
        noisy_data = raw_data + np.random.normal(0, epsilon, len(raw_data))
        
        # Robust standardization for prices
        price_mean = np.mean(noisy_data)
        price_std = np.std(noisy_data)
        if price_std < epsilon:
            price_std = 1.0  # Fallback if standard deviation is too small
            
        scaled_data = (noisy_data - price_mean) / (price_std + epsilon)
    
    return scaled_data, data_for_visualization

def visualize_change_points(data, stockdata, data_for_visualization, w, p_r0, detected_cps, use_log_returns, actual_max_run_length):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.dates import DateFormatter
    import pandas as pd
    
    # Set seaborn style
    sns.set_theme(style="whitegrid", context="notebook")
    
    colors = {
        'data': '#2E86C1',
        'change_point': '#E74C3C',
        'probability': '#27AE60',
        'price': '#8E44AD'
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(3, 1, height_ratios=[1, 1.5, 2], hspace=0.3)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    
    # Plot 1: Probability
    sns.lineplot(data=pd.Series(p_r0), ax=ax1, color=colors['probability'], 
                linewidth=1.5, label='P(run length=0)')
    ax1.fill_between(range(len(p_r0)), p_r0, alpha=0.2, color=colors['probability'])
    ax1.set_title("Change Point Probability", fontsize=12, pad=10)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.set_ylim(0, 1.1)
    sns.despine(ax=ax1)
    
    # Plot 2: Heatmap
    max_len = len(data)
    heatmap_data = np.zeros((max_len, actual_max_run_length + 1))
    for t in range(max_len):
        run_length_dist = np.zeros(actual_max_run_length + 1)
        w_t = w[t].tocsr()
        for r in range(w_t.shape[0]):
            run_length_dist[r] = w_t[r,:].sum()
        heatmap_data[t, :] = run_length_dist
    
    # Create heatmap with improved aesthetics
    sns.heatmap(heatmap_data.T, 
                ax=ax2,
                cmap='viridis',
                xticklabels=50,
                yticklabels=50,
                cbar_kws={'label': 'Probability'},
                rasterized=True)
    
    ax2.set_xlabel('Time', fontsize=10)
    ax2.set_ylabel('Run Length', fontsize=10)
    ax2.set_title('Run Length Distribution Over Time', fontsize=12, pad=10)
    
    # Plot 3: Stock price with change points
    df_plot = data_for_visualization.copy()
    if use_log_returns:
        df_plot = df_plot.iloc[1:]
    
    # Plot stock price
    sns.lineplot(data=df_plot, x=df_plot.index, y='close', ax=ax3,
                color=colors['price'], linewidth=1.5, label='Closing Price')
    
    # Add change points
    for cp in detected_cps:
        if cp < len(df_plot):
            cp_timestamp = df_plot.index[cp]
            cp_price = df_plot['close'].iloc[cp]
            
            # Vertical line
            ax3.axvline(cp_timestamp, color=colors['change_point'], 
                       linestyle='--', alpha=0.7, linewidth=1.5)
            
            # Point marker
            ax3.scatter(cp_timestamp, cp_price, 
                       color=colors['change_point'], 
                       s=100, zorder=5, alpha=0.7)
            
            try:
                # Add annotation with date
                ax3.annotate(
                    cp_timestamp.strftime('%Y-%m-%d'),
                    xy=(cp_timestamp, cp_price),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(
                        boxstyle='round,pad=0.5',
                        fc='#FEF9E7',
                        ec=colors['change_point'],
                        alpha=0.8
                    ),
                    arrowprops=dict(
                        arrowstyle='->',
                        connectionstyle='arc3,rad=0',
                        color=colors['change_point']
                    ),
                    fontsize=8,
                    rotation=45
                )
            except Exception as e:
                print(f"Warning: Could not add annotation for change point at {cp_timestamp}: {e}")
    
    ax3.set_title("Stock Price with Change Points", fontsize=12, pad=10)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Price ($)', fontsize=10)
    ax3.legend(frameon=True, fancybox=True, shadow=True)
    
    # Format x-axis dates
    ax3.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add overall title
    fig.suptitle('Bayesian Online Change Point Detection Analysis', 
                 fontsize=14, y=0.95, weight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    try:
        plt.show()
    except Exception as e:
        print(f"Warning: Error displaying plot: {e}")
        try:
            plt.savefig('change_point_analysis.png', dpi=300, bbox_inches='tight')
            print("Plot saved as 'change_point_analysis.png'")
        except Exception as e:
            print(f"Error saving plot: {e}")

def main():
    """
    Main function to run the change point detection analysis.
    """
    # Create synthetic data for comparison
    synthetic_data = create_synthetic_data()
    print("Synthetic data stats:")
    print(f"Mean: {np.mean(synthetic_data):.3f}")
    print(f"Std: {np.std(synthetic_data):.3f}")
    print(f"Range: [{np.min(synthetic_data):.3f}, {np.max(synthetic_data):.3f}]")

    # Load and prepare stock data
    use_synthetic_data = False
    use_log_returns = True
    start_date = '2021-07-01'
    end_date = '2023-07-01'
    stockdata, data_for_visualization = prepare_stock_data(
        'data_warehouse/15_min_interval_stocks/Technology/AAPL.csv', 
        use_log_returns, 
        start_date,
        end_date
    )

    print("\nStock data stats:")
    print(f"Mean: {np.mean(stockdata):.3f}")
    print(f"Std: {np.std(stockdata):.3f}")
    print(f"Range: [{np.min(stockdata):.3f}, {np.max(stockdata):.3f}]")

    # Choose between synthetic and real data
  
    data = synthetic_data if use_synthetic_data else stockdata

    # Configure and run change point detection
    actual_max_run_length = min(100, len(data) - 1)
    w, p_r0, detected_cps = kbocd_nonparametric(
        data,
        detect_threshold=0.5,
        skip_first=50,
        prune_threshold=1e-99,
        max_run_length=actual_max_run_length,
        cool_down_period=20,
        dynamic_prune_threshold=1e-99
    )

    print("Detected change points:", detected_cps)
    
    # Create visualization
    visualize_change_points(
        data=data,
        stockdata=stockdata,
        data_for_visualization=data_for_visualization,
        w=w,
        p_r0=p_r0,
        detected_cps=detected_cps,
        use_log_returns=use_log_returns,
        actual_max_run_length=actual_max_run_length
    )

if __name__ == "__main__":
    main()