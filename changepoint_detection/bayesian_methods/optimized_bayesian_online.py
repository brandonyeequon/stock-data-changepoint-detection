import numpy as np
import pandas as pd
import warnings

from scipy.stats import gaussian_kde
from scipy.sparse import lil_matrix
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------------------------------------
# 1) Gaussian Kernel & Simple KDE
# ----------------------------------------------------------------------------------
SQRT_2PI = 2.506628274631000  # Pre-computed √(2π)

def gaussian_kernel(u):
    """Standard Gaussian kernel = (1 / √(2π)) * exp(-0.5 * u^2)."""
    return np.exp(-0.5 * u * u) / SQRT_2PI

def kde_predictive_prob(x_new, data_seg, bw):
    """
    Calculate p(x_new | data_seg) via Gaussian KDE with bandwidth bw.
    data_seg is only past data (no future info).
    """
    data_seg = np.asarray(data_seg)
    n = len(data_seg)
    
    # For very small segments
    if n <= 2:
        if n == 0:
            return 1e-12
        if n == 1:
            return float(gaussian_kernel((x_new - data_seg[0]) / 1.0))
        if n == 2:
            mean = np.mean(data_seg)
            std = max(np.std(data_seg), 0.1)
            return float(gaussian_kernel((x_new - mean) / std) / std)
    
    # Ensure a minimum bandwidth
    bw = max(bw, 0.1)
    
    try:
        diffs = (x_new - data_seg) / bw
        weights = gaussian_kernel(diffs)
        result = float(np.sum(weights) / (n * bw))
        return max(result, 1e-12)
    except:
        return 1e-12

# ----------------------------------------------------------------------------------
# 2) Improved Sheather-Jones Bandwidth
# ----------------------------------------------------------------------------------
def improved_sheather_jones_bandwidth(data, max_iter=100, tol=None):
    """
    Nonparametric bandwidth selection (Sheather-Jones-like).
    Uses only data in 'data' (which is presumably up to current time).
    """
    from scipy.stats import gaussian_kde
    
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    N = len(data)
    if N < 2:
        return 1.0
    
    data_mean = np.mean(data)
    data_std = np.std(data)
    if data_std < 1e-15:
        return 1e-3
    
    x = (data - data_mean) / (data_std + 1e-15)
    xi = 0.907069048686
    l = 7
    
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    silverman_bw = 0.9 * min(np.std(x), iqr/1.34) * np.power(N, -0.2)
    z_n = silverman_bw
    
    eps = np.finfo(float).eps if tol is None else tol
    
    def estimate_derivative_norm(x_in, k, bw):
        grid = np.linspace(x_in.min(), x_in.max(), 512)
        kde = gaussian_kde(x_in, bw_method=bw)
        dens = kde.evaluate(grid)
        
        # repeatedly take numerical derivative
        derivative = dens.copy()
        for _ in range(k):
            derivative = np.gradient(derivative, grid)
        return np.trapz(derivative**2, grid)
    
    def gamma_l(k, NN, f_deriv_norm):
        num_1 = 1.0 + np.power(0.5, k + 0.5)
        num_2 = np.prod(np.arange(1, 2 * k, 2))  # 1,3,5,...(2k-1)
        denom = 3.0 * NN * np.sqrt(np.pi / 2.0) * f_deriv_norm
        return (num_1 * num_2) / denom
    
    for _ in range(max_iter):
        bw_current = z_n
        dnorm = estimate_derivative_norm(x, l, bw_current)
        if dnorm <= 0 or np.isnan(dnorm):
            break
        gamma_val = gamma_l(l, N, dnorm)
        z_next = xi * np.power(gamma_val, 1.0 / (2*l + 3))
        if abs(z_next - z_n) < eps:
            z_n = z_next
            break
        z_n = z_next
    
    return z_n * data_std

# ----------------------------------------------------------------------------------
# 3) Single-Step KB-OCD Update
# ----------------------------------------------------------------------------------
def kbocd_update(
    x_t,                 # current *standardized* data point
    data_history,        # entire data array up to t, used for segment extraction
    w_prev,              # previous posterior matrix
    active_states_prev,  # previous active states
    t,                   # current time index
    skip_first=10,
    prune_threshold=1e-8,
    max_run_length=1000,
    dynamic_prune_threshold=1e-5,
    hazard_constant=0.99,
    bw_dict=None
):
    """
    Perform a single KB-OCD update step for time t.
    data_history: The original *unstandardized* data up to index t
                  if we need to build segments.
    x_t: The *standardized* current point (but we'll still use data_history
         to get segments for the kernel PDE).
    
    Returns: (w_curr, active_states_curr, p_r0_t)
    """
    if bw_dict is None:
        bw_dict = {}
    
    new_w = lil_matrix((max_run_length+1, max_run_length+1))
    new_active = set()
    w_prev = w_prev.tolil()  # ensure modifiable
    
    # For each active state
    for (r_prev, a_prev) in active_states_prev:
        w_val = w_prev[r_prev, a_prev]
        if w_val < dynamic_prune_threshold:
            continue
        
        # Hazard
        hazard = float(a_prev + 1) / float(t + hazard_constant)
        if t < skip_first:
            hazard *= 0.5
        hazard = max(min(hazard, 1.0 - 1e-15), 1e-15)  # clamp
        growth = 1.0 - hazard
        
        # Build segment for kernel PDE: data[t-r_prev : t]
        seg_start = max(0, t - r_prev)
        seg = data_history[seg_start:t]  # up to but NOT including x_t
        # get bandwidth from cache or compute
        seg_key = seg.tobytes()
        if seg_key not in bw_dict:
            bw_val = improved_sheather_jones_bandwidth(seg)
            if not np.isfinite(bw_val) or bw_val < 1e-12:
                bw_val = 1e-3
            bw_dict[seg_key] = bw_val
        bw = bw_dict[seg_key]
        
        # Evaluate p_x
        p_x = kde_predictive_prob(data_history[t], seg, bw)
        
        # 1) Growth transition
        r_new = r_prev + 1
        if r_new <= max_run_length:
            new_weight = w_val * growth * p_x
            if new_weight >= dynamic_prune_threshold:
                new_w[r_new, a_prev] += new_weight
                # use += in case multiple transitions might point here
                # (rare, but possible). Or do = if you prefer overwriting.
                new_active.add((r_new, a_prev))
        
        # 2) Change transition
        if a_prev + 1 <= max_run_length:
            new_weight = w_val * hazard * p_x
            if new_weight >= dynamic_prune_threshold:
                new_w[0, a_prev + 1] += new_weight
                new_active.add((0, a_prev + 1))
    
    # Normalize
    total_mass = new_w.sum()
    if total_mass > prune_threshold:
        new_w = new_w.tocsr()
        new_w.data /= total_mass
    else:
        new_w = lil_matrix((max_run_length+1, max_run_length+1))
        new_w[0, 0] = 1.0
        new_w = new_w.tocsr()
        new_active = {(0,0)}
    
    # Probability run-length=0
    p_r0_t = new_w[0,:].sum()
    
    return new_w, new_active, p_r0_t


# ----------------------------------------------------------------------------------
# 4) Online Simulation with Initialization Stage
# ----------------------------------------------------------------------------------
def simulate_online_with_init(
    data,
    init_window_size=300,
    detect_threshold=0.5,
    skip_first=10,
    prune_threshold=1e-8,
    max_run_length=1000,
    dynamic_prune_threshold=1e-5,
    hazard_constant=0.99,
    cooldown_period=20
):
    """
    1) Use the first `init_window_size` points to initialize the model (still "online," but 
       we allow full updates up to t=init_window_size-1).
    2) After that, for t >= init_window_size, we do one-step updates, each time 
       standardizing x_t with a rolling window of the last init_window_size data points.
    3) We keep track of detected change points with a cooldown.
    """
    N = len(data)
    if N == 0:
        return [], [], []
    
    # Posterior structures
    w_list = [None] * N
    active_states_list = [None] * N
    p_r0 = np.zeros(N)
    detected_cps = []
    last_cp_time = -np.inf
    
    # Initialize w_0
    w_0 = lil_matrix((max_run_length+1, max_run_length+1))
    w_0[0, 0] = 1.0
    w_0 = w_0.tocsr()
    w_list[0] = w_0
    active_states_list[0] = {(0,0)}
    p_r0[0] = 1.0
    
    # We'll keep a cache for bandwidth to speed up repeated segments
    bw_dict = {}
    
    # --------------------------------------
    # (A) Initialization Stage (t in [1, init_window_size-1])
    # --------------------------------------
    init_stage_end = min(init_window_size, N)  # handle case if data < 300
    for t in tqdm(range(1, init_stage_end), desc="Initialization Stage", unit="pts"):
        # Standardize x_t with data up to t (the standard approach for partial "online")
        slice_for_std = data[:t]  # only up to but not including x_t
        mean_t = np.mean(slice_for_std)
        std_t = np.std(slice_for_std)
        if std_t < 1e-12:
            std_t = 1.0
        x_std = (data[t] - mean_t) / std_t
        
        w_prev = w_list[t-1]
        active_prev = active_states_list[t-1]
        
        w_t, active_t, p_r0_t = kbocd_update(
            x_std, 
            data,            # pass entire data, but kbocd_update only uses data[:t]
            w_prev,
            active_prev,
            t,
            skip_first=skip_first,
            prune_threshold=prune_threshold,
            max_run_length=max_run_length,
            dynamic_prune_threshold=dynamic_prune_threshold,
            hazard_constant=hazard_constant,
            bw_dict=bw_dict
        )
        
        w_list[t] = w_t
        active_states_list[t] = active_t
        p_r0[t] = p_r0_t
        
        # Check detection with cooldown
        if (
            t >= skip_first 
            and p_r0_t > detect_threshold
            and (t - last_cp_time >= cooldown_period)
        ):
            detected_cps.append(t)
            last_cp_time = t
    
    # --------------------------------------
    # (B) Main Online Stage (t in [init_window_size, N))
    # --------------------------------------
    for t in tqdm(range(init_stage_end, N), desc="Main Online Stage", unit="pts"):
        # Rolling standardization over last init_window_size points
        # If t < init_window_size, the window is smaller anyway
        window_start = max(0, t - init_window_size)
        rolling_segment = data[window_start:t]
        
        mean_t = np.mean(rolling_segment)
        std_t = np.std(rolling_segment)
        if std_t < 1e-12:
            std_t = 1.0
        
        x_std = (data[t] - mean_t) / std_t
        
        w_prev = w_list[t-1]
        active_prev = active_states_list[t-1]
        
        w_t, active_t, p_r0_t = kbocd_update(
            x_std,
            data,
            w_prev,
            active_prev,
            t,
            skip_first=skip_first,
            prune_threshold=prune_threshold,
            max_run_length=max_run_length,
            dynamic_prune_threshold=dynamic_prune_threshold,
            hazard_constant=hazard_constant,
            bw_dict=bw_dict
        )
        
        w_list[t] = w_t
        active_states_list[t] = active_t
        p_r0[t] = p_r0_t
        
        # Check detection with cooldown
        if (
            t >= skip_first 
            and p_r0_t > detect_threshold
            and (t - last_cp_time >= cooldown_period)
        ):
            detected_cps.append(t)
            last_cp_time = t
    
    return w_list, p_r0, detected_cps

# ----------------------------------------------------------------------------------
# 5) Example Usage / Main
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    def create_synthetic_data():
        """
        Synthesize data with two known shifts:
        seg1 ~ N(-1,1), seg2 ~ N(5,1), seg3 ~ N(-3,1).
        """
        np.random.seed(42)
        N1, N2, N3 = 100, 100, 100
        seg1 = np.random.normal(-1, 1, N1)
        seg2 = np.random.normal(5, 1, N2)
        seg3 = np.random.normal(-3, 1, N3)
        return np.concatenate([seg1, seg2, seg3])
    
    data = create_synthetic_data()
    print(f"Total data points: {len(data)}")

    # We'll do a 300-pt initialization, then purely online
    w_list, p_r0, detected_cps = simulate_online_with_init(
        data,
        init_window_size=300,
        detect_threshold=0.5,
        skip_first=10,
        prune_threshold=1e-8,
        max_run_length=200,
        dynamic_prune_threshold=1e-5,
        hazard_constant=0.99,
        cooldown_period=20
    )

    print("Detected change points (indices):", detected_cps)
    # E.g., you might see ~ 100 and ~ 200, matching the known shifts in synthetic data.
