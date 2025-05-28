# Stock Data Generation & Changepoint Detection

A comprehensive toolkit for generating raw stock market data with alphavantage. It also contains many experimental changepoint detection algorithms I have been experimenting with.

## Overview

This repository contains two main components:

1. **Raw Stock Data Generation**: Tools for fetching and managing 15-minute interval stock market data from AlphaVantage API
2. **Changepoint Detection**: Implementation of various Bayesian changepoint detection algorithms, including Dynamic State Model BOCD (DSM-BOCD)

## Features

### Stock Data Generation
- ✅ AlphaVantage API integration for 15-minute interval data
- ✅ Rate limiting and error handling
- ✅ Data validation and quality checks
- ✅ Bulk data fetching by month/year or ticker
- ✅ CSV output format organized by sectors
- ✅ Data management utilities

### Changepoint Detection
- ✅ Dynamic State Model Bayesian Online Changepoint Detection (DSM-BOCD)
- ✅ Standard Bayesian Online Changepoint Detection (BOCD)
- ✅ Nonparametric Bayesian changepoint detection with KDE
- ✅ Online and batch processing modes
- ✅ Multiple statistical models (Gaussian, DSM-Gaussian)
- ✅ Configurable hazard functions

## Installation

```bash
git clone <repository-url>
cd stock-data-changepoint-detection
pip install -r requirements.txt
```

### Dependencies

```bash
pip install numpy pandas scipy matplotlib seaborn
pip install scikit-learn
pip install requests  # For AlphaVantage API
```

## Quick Start

### 1. Stock Data Generation

```python
from data_clients.pull_by_ticker import fetch_stock_data

# Fetch data for a specific ticker
fetch_stock_data("AAPL", "2024", "01", api_key="your_alphavantage_key")
```

### 2. Changepoint Detection

```python
from changepoint_detection.dsm_bocd.bocpd import BOCPD
from changepoint_detection.dsm_bocd.models import DSMGaussian
from changepoint_detection.dsm_bocd.hazard import ConstantHazard

# Initialize DSM-BOCD
model = DSMGaussian(mean0=0, var0=1, varx=1)
hazard = ConstantHazard(lam=250)
bocd = BOCPD(model=model, hazard=hazard)

# Detect changepoints in your data
for x in data:
    bocd.update(x)
    
# Get changepoint probabilities
changepoint_probs = bocd.get_changepoint_probabilities()
```

## Repository Structure

```
stock-data-changepoint-detection/
├── data_clients/                    # Stock data fetching
│   ├── pull_by_month.py            # Fetch data by month/year
│   ├── pull_by_ticker.py           # Fetch data by ticker symbol
│   ├── replace_bad_data.py         # Data quality management
│   └── validate_set2.py            # Data validation
├── data_managers/                   # Data management utilities
│   ├── stock_data_manager.py       # Data warehouse management
│   ├── clean_tickers.py            # Ticker list cleaning
│   └── combine.py                  # Data combination utilities
├── changepoint_detection/
│   ├── dsm_bocd/                   # DSM-BOCD implementation
│   │   ├── bocpd.py               # Main BOCD algorithm
│   │   ├── dsm-bocd_realworld.py  # Real-world application
│   │   ├── models.py              # Statistical models
│   │   ├── hazard.py              # Hazard functions
│   │   ├── omega_estimator.py     # Parameter estimation
│   │   └── utils/                 # Utility functions
│   │       ├── find_cp.py         # Changepoint detection
│   │       └── generate_data.py   # Data generation
│   └── bayesian_methods/           # Alternative implementations
│       ├── optimized_bayesian.py  # Nonparametric BOCD
│       ├── online_kbocd.py        # Kernel-based BOCD
│       └── ...                    # Other experimental methods
├── utils/                          # General utilities
│   └── processor.py               # Data processing
├── examples/                       # Example scripts
│   └── test_DSM-bocd.py           # DSM-BOCD example
└── README.md                      # This file
```

## Algorithms

### Dynamic State Model BOCD (DSM-BOCD)

DSM-BOCD extends traditional BOCD by incorporating a dynamic state model that adapts to changing data characteristics. Key features:

- **Adaptive Parameters**: Model parameters evolve over time
- **Improved Accuracy**: Better performance on non-stationary data
- **Real-time Processing**: Online algorithm suitable for streaming data

### Nonparametric Bayesian Methods

Alternative implementations using kernel density estimation:

- **Kernel-based BOCD**: Uses KDE for predictive modeling
- **Adaptive Bandwidth**: ISJ bandwidth selection for optimal performance
- **Fully Nonparametric**: No distributional assumptions

## Data Sources

### AlphaVantage API

The stock data generation tools use the AlphaVantage API to fetch:
- 15-minute interval OHLCV data
- Historical data dating back to 1999
- Coverage of 6,500+ stock tickers
- Organized by market sectors

**API Key Required**: Get your free API key at [AlphaVantage](https://www.alphavantage.co/)

## Usage Examples

### Complete Pipeline Example

```python
# 1. Fetch stock data
from data_clients.pull_by_ticker import main as fetch_data
fetch_data()  # Fetches data for configured tickers

# 2. Load and process data
import pandas as pd
data = pd.read_csv("path/to/your/stock_data.csv")
prices = data['close'].values

# 3. Detect changepoints
from changepoint_detection.dsm_bocd.bocpd import BOCPD
from changepoint_detection.dsm_bocd.models import DSMGaussian
from changepoint_detection.dsm_bocd.hazard import ConstantHazard

model = DSMGaussian(mean0=prices[0], var0=1, varx=1)
hazard = ConstantHazard(lam=250)  # Expected run length
bocd = BOCPD(model=model, hazard=hazard)

changepoints = []
for i, price in enumerate(prices):
    bocd.update(price)
    if i > 0:
        cp_prob = bocd.get_most_recent_changepoint_prob()
        if cp_prob > 0.5:  # Threshold for changepoint detection
            changepoints.append(i)

print(f"Detected {len(changepoints)} changepoints")
```

## Configuration

### AlphaVantage API Configuration

Set up your API configuration in the data client files:

```python
# In data_clients/pull_by_ticker.py
API_KEY = "your_alphavantage_api_key_here"
BASE_URL = "https://www.alphavantage.co/query"
```

### DSM-BOCD Parameters

Key parameters for tuning DSM-BOCD performance:

- `mean0`: Initial mean estimate
- `var0`: Initial variance estimate  
- `varx`: Observation noise variance
- `lam`: Expected run length (hazard parameter)
- `omega`: State evolution parameters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{stock-changepoint-detection,
    title={Stock Data Generation and Changepoint Detection Toolkit},
    author={Brandon Yee Quon},
    year={2025},
    url={https://github.com/brandonyeequon/stock-data-changepoint-detection}
}
```

## Acknowledgments

- DSM-BOCD algorithm implementation based on research in Bayesian changepoint detection
- AlphaVantage for providing stock market data API
- Contributors to the original BOCD research and implementations

## Support

For questions, issues, or contributions, please open an issue on GitHub or contact me.