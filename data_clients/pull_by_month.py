import requests
from datetime import datetime
from typing import Optional
import os
import pandas as pd
import time
import logging
#from dotenv import load_dotenv
api_key = 'JX07Y9KM8GI7BS1M'
class AlphaVantageClient:
    def __init__(self):
        self.base_url = "https://www.alphavantage.co/query"
        # self.h5_path = "data_warehouse/HDF5_file/stock_data.h5"  # Commented out
        self.csv_dir = "data_warehouse/15_min_interval_stocks"
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
            csv_path = f'data_warehouse/15_min_interval_stocks/{symbol}.csv'
            
            # If data exists, merge with new data
            existing_csv = pd.read_csv(csv_path)
            existing_csv['timestamp'] = pd.to_datetime(existing_csv['timestamp'])
            df = pd.concat([existing_csv, df])
            df = df.drop_duplicates(subset=['timestamp'])
            df = df.sort_values('timestamp')

            
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
        
        # Save to HDF5
        self.save_stock_data(symbol, df)
        
        return df
    

    def data_month_exists(self, symbol: str, month: str, year: str, df: pd.DataFrame) -> bool:
        """
        Check if data for the specified month and year already exists for the given symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            month: Month as a string in MM format (e.g., '12')
            year: Year as a string in YYYY format (e.g., '2023')
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
        Returns:
            bool: True if data for the month and year exists, False otherwise
        """

        # Create start and end timestamps for the month
        start_date = pd.Timestamp(year=int(year), month=int(month), day=1)
        end_date = (start_date + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)


        # Create a mask for the specified month
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        
        # Debugging: Log the number of entries found
        logging.debug(f"Checking data for {symbol} in {year}-{month}: {mask.sum()} entries found.")

        return mask.any()

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='validate_data_set_2.log'
    )
    
    
    # Read ticker list
    try:
        with open('logs/full_tickers.txt', 'r') as f:
            symbols = [line.strip() for line in f]
    except FileNotFoundError:
        logging.error("full_tickers.txt not found!")
        return
     
    # Initialize client
    client = AlphaVantageClient() 
    months = ['12', '11', '10', '09', '08', '07', '06', '05', '04', '03', '02', '01']
    completed = 0
    years = ['2024','2023', '2022', '2021', '2020', '2019', '2018', '2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010', '2009', '2008', '2007', '2006', '2005', '2004', '2003', '2002', '2001', '2000']
    failed_attempts = 0
     
    for i, symbol in enumerate(symbols, 1):
        try:
            df = pd.read_csv(f'data_warehouse/15_min_interval_stocks/{symbol}.csv', parse_dates=['timestamp'])
        except FileNotFoundError:
            df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
        failed_attempts = 0  # Reset counter for each new symbol
        
        for year in years:
            for month in months:
                '''
                if client.data_month_exists(symbol, month, year, df):
                    print(f"Data for {symbol} in {year}-{month} already exists, skipping...")
                    logging.info(f"Data for {symbol} in {year}-{month} already exists, skipping...")
                    continue  # Skip to the next month
                '''
                try:                    
                    # Fetch data
                    client.fetch_stock_data(symbol, month, year)
                    logging.info(f"Successfully fetched data for {symbol} - {year}-{month}")
                    print(f"Successfully fetched data for {symbol} - {year}-{month}")
                    failed_attempts = 0  # Reset counter after successful fetch
                    completed += 1
                    
                    # Reload df from the CSV to include the newly saved data
                    df = pd.read_csv(f'data_warehouse/15_min_interval_stocks/{symbol}.csv', parse_dates=['timestamp'])
                    
                except Exception as e:
                    error_msg = str(e)
                    logging.error(f"Error processing |{symbol}| for {year}-{month}: {error_msg}")
                    print(error_msg)
                    failed_attempts += 1
                    
                    if failed_attempts >= 1:
                        logging.error(f"Failed {failed_attempts} attempts for {symbol}, skipping to next symbol...")
                        print(f"Failed {failed_attempts} attempts for {symbol}, skipping to next symbol...")
                        break  # Break out of months loop
                
            if failed_attempts >= 1:
                break  # Break out of years loop to move to next symbol
    
    # Log summary
    logging.info(f"Process completed:")
    logging.info(f"- Total months processed: {len(months)}")
    logging.info(f"- Total requests: {completed}")

if __name__ == "__main__":
    main()
    print("Done")


 




   
