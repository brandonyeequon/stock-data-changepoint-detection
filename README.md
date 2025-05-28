# Comprehensive Stock Market Dataset & Generation Tools

A massive stock market dataset and toolkit for generating high-frequency financial data, featuring 6,500+ tickers with 15-minute interval data organized by market sectors.

## 📊 Dataset Overview

This repository provides access to one of the most comprehensive stock market datasets available, featuring:

- **🔢 6,564 individual stock tickers** across all major US exchanges
- **📈 15-minute interval OHLCV data** (Open, High, Low, Close, Volume)
- **📅 Historical coverage** from 1999 to present
- **🏢 Complete sector organization** across 12 major market sectors
- **💾 Multi-gigabyte dataset** with validated, clean data
- **🔗 Available on Hugging Face**: [brandonyeequon/stock-market-data-warehouse](https://huggingface.co/datasets/brandonyeequon/stock-market-data-warehouse)

### Sector Coverage

The dataset is organized across **12 major market sectors**:

- **Basic Materials** (146 tickers): Mining, chemicals, forestry, paper
- **Consumer Discretionary** (845 tickers): Retail, media, automotive, leisure  
- **Consumer Staples** (168 tickers): Food, beverages, household products
- **Energy** (244 tickers): Oil, gas, renewable energy, utilities
- **Financials** (923 tickers): Banks, insurance, real estate, investment
- **Healthcare** (789 tickers): Pharmaceuticals, biotech, medical devices
- **Industrials** (768 tickers): Aerospace, defense, construction, transportation
- **Technology** (1,347 tickers): Software, hardware, semiconductors, telecommunications
- **Telecommunications** (83 tickers): Telecom services and infrastructure
- **Utilities** (158 tickers): Electric, gas, water utilities
- **Real Estate** (267 tickers): REITs and real estate companies
- **Other** (826 tickers): Miscellaneous and hybrid sector companies

### Data Format

Each CSV file contains standardized columns:
- `timestamp`: ISO format datetime (15-minute intervals)
- `open`: Opening price for the interval
- `high`: Highest price during the interval  
- `low`: Lowest price during the interval
- `close`: Closing price for the interval
- `volume`: Trading volume for the interval

## 🛠️ Data Generation Tools

This repository includes the complete toolkit used to generate and manage this massive dataset:

### AlphaVantage API Integration
- **Rate-limited fetching**: Respects API limits with intelligent retry logic
- **Bulk data collection**: Download by ticker, month, or year ranges
- **Error handling**: Robust validation and bad data replacement
- **Sector organization**: Automatic filing into appropriate sector directories

### Data Management Utilities
- **Quality validation**: Automated data integrity checks
- **Ticker management**: Clean and maintain ticker lists
- **Data combination**: Merge and analyze datasets across tickers
- **Storage optimization**: Efficient CSV storage with sector organization

## 🚀 Quick Start

### Using Pre-Generated Dataset

```python
import pandas as pd

# Load data for any ticker (example: Apple)
df = pd.read_csv("data_warehouse/15_min_interval_stocks/Technology/AAPL.csv")
print(f"AAPL data: {len(df)} rows from {df['timestamp'].min()} to {df['timestamp'].max()}")
```

### Generating New Data

```python
from data_clients.pull_by_ticker import fetch_stock_data

# Generate data for additional tickers
fetch_stock_data("NVDA", "2024", "01", api_key="your_alphavantage_key")
```

## 📁 Repository Structure

```
stock-data-changepoint-detection/
├── data_clients/           # AlphaVantage API data fetching tools
│   ├── pull_by_month.py   # Fetch data by time periods
│   ├── pull_by_ticker.py  # Fetch data by stock symbols
│   └── validate_set2.py   # Data validation utilities
├── data_managers/          # Dataset management and analysis
│   ├── stock_data_manager.py  # Data warehouse operations
│   ├── clean_tickers.py       # Ticker list maintenance
│   └── combine.py             # Data aggregation tools
├── changepoint_detection/  # Bonus: Changepoint detection algorithms
│   ├── dsm_bocd/          # DSM-BOCD implementation
│   └── bayesian_methods/   # Alternative Bayesian methods
└── examples/              # Usage examples and tutorials
```

## 💾 Dataset Access

### Hugging Face Hub
The complete dataset is available on Hugging Face:

**🔗 [brandonyeequon/stock-market-data-warehouse](https://huggingface.co/datasets/brandonyeequon/stock-market-data-warehouse)**

```python
from datasets import load_dataset

# Load specific sector data
dataset = load_dataset("brandonyeequon/stock-market-data-warehouse", 
                      data_files="15_min_interval_stocks/Technology/*.csv")
```

### Local Generation
Use the included tools to generate your own dataset:

```bash
# Install dependencies
pip install -r requirements.txt

# Configure your AlphaVantage API key
export ALPHAVANTAGE_API_KEY="your_key_here"

# Generate data for specific tickers
python data_clients/pull_by_ticker.py
```

## 📊 Dataset Statistics

- **Total Files**: 6,564 CSV files
- **Data Points**: Billions of individual price records
- **Time Span**: 25+ years of market data
- **Update Frequency**: 15-minute intervals
- **Coverage**: Complete US stock market representation
- **Quality**: Validated, cleaned, and error-corrected

## 🔧 API Configuration

### AlphaVantage Setup

1. Get your free API key: [AlphaVantage](https://www.alphavantage.co/)
2. Configure in the data clients:

```python
# In data_clients/pull_by_ticker.py
API_KEY = "your_alphavantage_api_key_here"
```

## 📈 Use Cases

This dataset is perfect for:

- **Quantitative Finance Research**: High-frequency trading strategies
- **Machine Learning**: Price prediction and pattern recognition  
- **Market Analysis**: Sector performance and correlation studies
- **Academic Research**: Financial markets and economic studies
- **Algorithm Development**: Testing trading strategies and indicators
- **Risk Management**: Portfolio optimization and risk assessment

## 🎯 Bonus: Changepoint Detection

As a bonus feature, this repository includes experimental changepoint detection algorithms:

- **DSM-BOCD**: Dynamic State Model Bayesian Online Changepoint Detection
- **Nonparametric methods**: KDE-based changepoint detection
- **Real-time processing**: Online algorithms for streaming data

Perfect for detecting regime changes, market crashes, and volatility shifts in the financial data.

## 📝 Installation

```bash
git clone https://github.com/brandonyeequon/stock-data-changepoint-detection.git
cd stock-data-changepoint-detection
pip install -r requirements.txt
```

## 🤝 Contributing

Contributions welcome! Please feel free to:
- Add new data sources or APIs
- Improve data validation and cleaning
- Enhance the changepoint detection algorithms
- Add new analysis tools and utilities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AlphaVantage** for providing comprehensive stock market data API
- **Hugging Face** for hosting the massive dataset
- **Contributors** to the financial data and research community

## 📞 Support

For questions, issues, or contributions, please open an issue on GitHub or contact me.