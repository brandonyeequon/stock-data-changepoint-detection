import requests
from datetime import datetime
from typing import Optional
import os
import pandas as pd
import requests
import time
import logging
#from dotenv import load_dotenv
api_key = 'JX07Y9KM8GI7BS1M'
class AlphaVantageClient:
    def __init__(self, csv_dir: str = "data_warehouse/15_min_interval_stocks"):
        self.base_url = "https://www.alphavantage.co/query"
        # self.h5_path = "data_warehouse/HDF5_file/stock_data.h5"  # Commented out
        self.csv_dir = csv_dir
        # Create directories if they don't exist
        os.makedirs(self.csv_dir, exist_ok=True)
        # os.makedirs(os.path.dirname(self.h5_path), exist_ok=True)  # Commented out
    
    
    def save_stock_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Save stock data to both HDF5 and CSV formats
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        try:
            # Define paths
            csv_path = f'data_warehouse/validation_data/{symbol}.csv'
            
            # If data exists, merge with new data
            if self.data_exists(symbol):
                # Merge CSV data
                try:
                    # Read existing CSV
                    existing_csv = pd.read_csv(csv_path)
                    existing_csv['timestamp'] = pd.to_datetime(existing_csv['timestamp'])
                    
                    # Get the month and year from the new data
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    month = df['timestamp'].dt.month[0]  # Get month from first row
                    year = df['timestamp'].dt.year[0]    # Get year from first row
                    
                    # Remove all data from the same month and year in existing_csv
                    existing_csv = existing_csv[
                        ~((existing_csv['timestamp'].dt.month == month) & 
                          (existing_csv['timestamp'].dt.year == year))
                    ]
                    
                    # Concatenate and sort
                    df = pd.concat([existing_csv, df])
                    df = df.sort_values('timestamp')
                    
                except FileNotFoundError:
                    logging.warning(f"No existing CSV data for {symbol}")
              
            
            # Ensure all numeric columns are the correct type before saving
            df = df.astype({
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'int64'
            })
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            
            logging.info(f"Saved {len(df)} rows for {symbol}")
            
        except Exception as e:
            logging.error(f"Error saving data for {symbol}: {str(e)}")
            raise Exception(f"Failed to save data for {symbol}: {str(e)}")

    def fetch_stock_data(self, symbol: str, month, year) -> pd.DataFrame:
        """Fetch and save stock data, return the DataFrame"""
        url = f'https://www.alphavantage.co/query?' \
            f'function=TIME_SERIES_INTRADAY&' \
            f'symbol={symbol}&' \
            f'interval=15min&' \
            f'outputsize=full&' \
            f'datatype=json&' \
            f'month={year}-{month}&' \
            f'apikey=JX07Y9KM8GI7BS1M'
         
        # Get the response
        response = requests.get(url)
        
        # Convert response to JSON
        data = response.json()
        
        # Check for various error responses
        if "Error Message" in data:
            raise Exception(f"API Error: {data['Error Message']}")
        
        if "Note" in data:  # Rate limit message
            raise Exception(f"API Rate Limit: {data['Note']}")
        
        if "Information" in data:  # Sometimes returns informational messages as errors
            raise Exception(f"API Information: {data['Information']}")
        
        # Check if data is empty or malformed
        if len(data) == 0 or "Time Series (15min)" not in data:
            raise Exception(f"No valid data returned for {symbol}")

        # Extract time series data
        time_series = data.get('Time Series (15min)', {})
        
        if len(time_series) == 0:
            raise Exception(f"Empty time series data for {symbol}")
        
        # Create DataFrame from time series
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Clean column names (remove prefixes like "1. ", "2. ", etc.)
        try:
            df.columns = [col.split('. ')[1] for col in df.columns]
            #print(f"Cleaned columns: {df.columns.tolist()}")
        except Exception as e:
            #print(f"Raw data: {data}")  # Print the raw response
            raise Exception(f"Invalid column format for {symbol}")
        
        # Convert types with error handling
        try:
            df = df.astype({
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': int
            })
        except KeyError as e:
            raise Exception(f"Missing required columns for {symbol}")
        
        df.index = pd.to_datetime(df.index)
        df = df.reset_index().rename(columns={'index': 'timestamp'})
        df = df.sort_values('timestamp')
        
        self.save_stock_data(symbol, df)
        
        return df
    
    def data_exists(self, symbol: str) -> bool:
        """Check if data already exists for this symbol"""
        csv_path = f"{self.csv_dir}/{symbol}.csv"
        
        if os.path.exists(csv_path):
            return True
        
        return False
    

    def data_month_exists(self, symbol: str, month: str, year: str) -> bool:
        """
        Check if data for the specified month and year already exists for the given symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            month: Month as a string in MM format (e.g., '12')
            year: Year as a string in YYYY format (e.g., '2023')
            
        Returns:
            bool: True if data for the month and year exists, False otherwise
        """
        csv_path = f"{self.csv_dir}/{symbol}.csv"

        if not os.path.exists(csv_path):
            # CSV file doesn't exist, so data for the month can't exist
            return False

        try:
            # Read the CSV file and parse 'timestamp' column as datetime
            # change nrows depending on last starting point
            df = pd.read_csv(csv_path, parse_dates=['timestamp'], nrows=10000)
        except Exception as e:
            logging.error(f"Error reading CSV for {symbol}: {str(e)}")
            return False

        # Create start and end timestamps for the month
        start_date = pd.Timestamp(year=int(year), month=int(month), day=1)
        end_date = (start_date + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)

        # Check if any timestamps in the DataFrame fall within the specified month
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        
        if mask.any():
            # Data for the specified month and year exists
            return True
        else:
            # No data found for the specified month and year
            return False

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='stock_data_fetch_1999.log'
    )
    
    
    # Read ticker list
    try:
        with open('logs/full_tickers.txt', 'r') as f:
            symbols = [line.strip() for line in f]
    except FileNotFoundError:
        logging.error("stock_tickers.txt not found!")
        return
     
    # Initialize client
    client = AlphaVantageClient() 
    months = ['11', '10']
    # Calculate delay to achieve 75 iterations per minute
    delay = 60 / 75  # 0.8 seconds between iterations
    completed = 0
    years = ['2024']
    total_requests = len(symbols) * len(months) * len(years)

    for year in years:
        for month in months:
            logging.info(f"Starting month: {year}-{month}")
            for i, symbol in enumerate(symbols, 1):
                start_time = time.time()
                
                # Check if data for this symbol and month/year already exists
                '''
                if client.data_month_exists(symbol, month, year):
                    logging.info(f"Data for {symbol} in {year}-{month} already exists, skipping...")
                    print(f"Data for {symbol} in {year}-{month} already exists, skipping...")
                    continue  # Skip to the next symbol
                '''
                try:                    
                    df = client.fetch_stock_data(symbol, month, year)

                    # Rate limiting
                    elapsed = time.time() - start_time
                    if elapsed < delay and (completed < total_requests):
                        time.sleep(delay - elapsed)
                    logging.info(f"Successfully fetched {len(df)} records for {symbol} - {year}-{month}")
                    print(f"Successfully fetched {len(df)} records for {symbol} - {year}-{month}")
                    
                except Exception as e:
                    error_msg = str(e)
                    logging.error(f"Error processing |{symbol}| for {year}-{month}: {error_msg}")
                    print(error_msg)
                    continue

                completed += 1

                    
            logging.info(f"Completed month {year}-{month}, {completed}/{total_requests} requests processed")
    
    # Log summary
    logging.info(f"Process completed:")
    logging.info(f"- Total months processed: {len(months)}")
    logging.info(f"- Total requests: {completed}")

if __name__ == "__main__":
    main()
    print("Done")


 




   
