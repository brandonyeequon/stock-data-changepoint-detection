import pandas as pd
import numpy as np
import fastdtw
import numba
from numba import float64, njit
import time


class Processor:
    def __init__(self):
        pass

    @staticmethod
    def euclidean_distance(x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def DTW(self, target_df, other_df) -> float:
        # Input validation
        if len(target_df) < 2 or len(other_df) < 2:
            raise ValueError("Input DataFrames must have at least 2 rows")

        # Cache frequently accessed values and ensure dtype
        target_close = target_df['close'].values.astype(np.float64)
        other_close = other_df['close'].values.astype(np.float64)
        
        # Calculate sizes for preallocation
        target_size = len(target_close) - 1
        other_size = len(other_close) - 1
        
        # Pre-allocate all arrays we'll need
        target_returns = np.empty(target_size, dtype=np.float64)
        other_returns = np.empty(other_size, dtype=np.float64)
        target_z = np.empty(target_size, dtype=np.float64)
        other_z = np.empty(other_size, dtype=np.float64)
        
        # Calculate log returns with preallocated arrays
        np.log(target_close[1:] / target_close[:-1], out=target_returns)
        np.log(other_close[1:] / other_close[:-1], out=other_returns)

        # Calculate z-scores with preallocated arrays
        target_mean = np.mean(target_returns)
        target_std = np.std(target_returns)
        other_mean = np.mean(other_returns)
        other_std = np.std(other_returns)
        
        # Calculate z-scores
        np.subtract(target_returns, target_mean, out=target_z)
        np.divide(target_z, target_std, out=target_z)
        np.subtract(other_returns, other_mean, out=other_z)
        np.divide(other_z, other_std, out=other_z)
        
        # Calculate DTW distance
        distance, _ = fastdtw.fastdtw(
            target_z,
            other_z,
            radius=15,
            dist=Processor.euclidean_distance
        )
        
        return distance
    



processor = Processor()

start_time = time.time()
score = processor.DTW(pd.read_csv('data/listener/tests.csv'), pd.read_csv('data/listener/tests2.csv'))
end_time = time.time()
print(f"Time taken to calculate DTW: {end_time - start_time} seconds")
print(score)
