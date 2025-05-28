import sys
import os
from multiprocessing import Pool, cpu_count, freeze_support

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tueplots import bundles

from bocpd import bocpd
from hazard import ConstantHazard
from models import DSMGaussian
from omega_estimator import OmegaEstimatorGaussian
from utils.find_cp import find_cp
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
def process_dataset(args):
    data_scaled, m, grad_m, mu0, Sigma0 = args
    
    estimator = OmegaEstimatorGaussian(data_scaled[100:200], m, grad_m, mu0, Sigma0)
    omegas, costs = estimator.omega(0.1, lr=1e-11, niter=2000, prior_parameters=[1, 1, 100, 100])
    omega = omegas[-1]
    
    model_DSM = DSMGaussian(data_scaled, m, grad_m, np.round(omega,5), mu0, Sigma0, b=40)
    hazard = ConstantHazard(100)
    
    np.random.seed(100)
    R_DSM = bocpd(data_scaled, hazard, model_DSM, K=50, verbose=True)
    cps_DSM = find_cp(R_DSM)
    
    return omegas, costs, R_DSM, cps_DSM
def m(x):
    m = np.eye(1)
    m[0,0] = (1+x[0]**2)**(-1/2)
    return m

def grad_m(x):
    m1 = np.zeros((1,1))
    m1[0,0] = -x[0]/((1+x[0]**2)**(3/2))
    return np.expand_dims(m1, axis=0)

def visualize_results(data, log_returns, R_DSM, cps_DSM, R_DSM_log, cps_DSM_log):
    x_indices = np.arange(len(data)-1)
    T = len(log_returns)

    with plt.rc_context(bundles.icml2022(ncols=2)):
        plt.rcParams['text.usetex'] = False
        fig, ax = plt.subplots(2, 1, sharex=True, 
                              gridspec_kw={'height_ratios': [1, 1]}, 
                              dpi=200, figsize=(3.25, 3))
        
        ax[0].plot(x_indices, data[:-1,0], c='black', lw=0.8)
        y_lims = ax[0].get_ylim()
        ax[0].set_xlim([0, len(data)-1])
        
        ax[1].set_ylim([0,T])
        ax[1].imshow(np.rot90(R_DSM), aspect='auto', cmap='gray_r', 
                     norm=LogNorm(vmin=0.0001, vmax=1), 
                     extent=[0, len(data)-1, 0, T])
        ax[1].plot(x_indices, np.argmax(R_DSM[1:], axis=1), 
                   c=CB_color_cycle[0], alpha=1, lw=1)
        
        for cp in cps_DSM:
            ax[0].axvline(cp, c=CB_color_cycle[0], ls='dotted', lw=1.5, label='DSM (Price)' if cp == cps_DSM[0] else "")
            
        for cp in cps_DSM_log:
            ax[0].scatter(cp, y_lims[0], marker="^", 
                         c=CB_color_cycle[2], alpha=1, lw=1, label='DSM (Log Returns)' if cp == cps_DSM_log[0] else "")
        
        ax[0].legend(loc='upper right', fontsize=4)
        
        ax[1].set_ylabel('run-length \n [robust]', size=6)
        ax[0].set_ylabel('Price', size=6)
        ax[1].set_xlabel('time')
        
        fig.subplots_adjust(hspace=0)
        plt.show()

def get_parameters(data_type="price"):
    mean_mu0, var_mu0 = 0, 1  
    mean_Sigma0, var_Sigma0 = 10, 1  

    mu0 = np.array([[mean_mu0/var_mu0], [1/var_mu0]])
    Sigma0 = np.eye(2)
    Sigma0[0,0] = mean_Sigma0/var_Sigma0
    Sigma0[1,1] = 1/var_Sigma0
    
    Sigma0 += 1e-6 * np.eye(2)
    
    return mu0, Sigma0

def main():
    FILE_PATH_1 = "/Users/brandonyeequon/Syukufuku/data_warehouse/15_min_interval_stocks/Technology/ADEA.csv"
    FILE_PATH_2 = '/Users/brandonyeequon/Syukufuku/data_warehouse/15_min_interval_stocks/Technology/ADBE.csv'
    
    start_date = '2024-06-01'
    end_date = '2024-07-10'
    
    test_file_1 = pd.read_csv(FILE_PATH_1)
    test_file_1 = test_file_1[(test_file_1['timestamp']>=start_date) & (test_file_1['timestamp']<=end_date)]
    data_1 = test_file_1['close'].values.reshape(-1, 1)
    log_returns_1 = np.log(data_1[1:]) - np.log(data_1[:-1])
    
    test_file_2 = pd.read_csv(FILE_PATH_2)
    test_file_2 = test_file_2[(test_file_2['timestamp']>=start_date) & (test_file_2['timestamp']<=end_date)]
    data_2 = test_file_2['close'].values.reshape(-1, 1)
    log_returns_2 = np.log(data_2[1:]) - np.log(data_2[:-1])

    scaler = preprocessing.StandardScaler()  
    data_scaled_1 = scaler.fit_transform(data_1)[:-1]
    log_returns_scaled_1 = scaler.fit_transform(log_returns_1)
    data_scaled_2 = scaler.fit_transform(data_2)[:-1]
    log_returns_scaled_2 = scaler.fit_transform(log_returns_2)

    mu0_price_1, Sigma0_price_1 = get_parameters([data_scaled_1])
    mu0_price_2, Sigma0_price_2 = get_parameters([data_scaled_2])
    mu0_log_1, Sigma0_log_1 = get_parameters([log_returns_scaled_1])
    mu0_log_2, Sigma0_log_2 = get_parameters([log_returns_scaled_2])
    
    args_list = [
        (data_scaled_1, m, grad_m, mu0_price_1, Sigma0_price_1),
        (log_returns_scaled_1, m, grad_m, mu0_log_1, Sigma0_log_1),
        (data_scaled_2, m, grad_m, mu0_price_2, Sigma0_price_2),
        (log_returns_scaled_2, m, grad_m, mu0_log_2, Sigma0_log_2)
    ]

    with Pool(processes=4) as pool:
        results = pool.map(process_dataset, args_list)

    stock1_price_results = results[0]
    stock1_returns_results = results[1]
    stock2_price_results = results[2]
    stock2_returns_results = results[3]

    print("\nGenerating Stock 1 plots...")
    visualize_results(data_1, log_returns_1, 
                     stock1_price_results[2], stock1_price_results[3],
                     stock1_returns_results[2], stock1_returns_results[3])
    print("\nGenerating Stock 2 plots...")
    visualize_results(data_2, log_returns_2, 
                     stock2_price_results[2], stock2_price_results[3],
                     stock2_returns_results[2], stock2_returns_results[3])
if __name__ == '__main__':
    freeze_support()
    main()