from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import logsumexp
from bayesian_changepoint_detection.hazard_functions import constant_hazard
from bayesian_changepoint_detection.bayesian_models import online_changepoint_detection
import bayesian_changepoint_detection.online_likelihoods as online_ll
from bayesian_changepoint_detection.generate_data import generate_normal_time_series
import matplotlib.cm as cm


partition, data = generate_normal_time_series(7, 50, 200)


df = pd.read_csv('data_warehouse/15_min_interval_stocks/Technology/AAPL.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df = df.tail(2000)
close_values = df['close'].values
data = np.log(close_values[1:] / close_values[:-1])
data = (data - np.mean(data)) / np.std(data)


hazard_function = partial(constant_hazard, 250)
R, maxes = online_changepoint_detection(
    data, hazard_function, online_ll.StudentT(alpha=0.1, beta=.01, kappa=1, mu=0)
)


epsilon = 1e-7
fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
ax[0].plot(data)
sparsity = 5  # only plot every fifth data for faster display
density_matrix = -np.log(R[0:-1:sparsity, 0:-1:sparsity]+epsilon)
ax[1].pcolor(np.array(range(0, len(R[:,0]), sparsity)), 
          np.array(range(0, len(R[:,0]), sparsity)), 
          density_matrix, 
          cmap=cm.Greys, vmin=0, vmax=density_matrix.max(),
            shading='auto')
Nw=10
ax[2].plot(R[Nw,Nw:-1])
plt.show()