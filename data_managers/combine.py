import pandas as pd
import os
'''
This script combines comm sector stocks into one dataframe
'''
def combine_comm_sector_stocks():
    """Combines communication sector stock data into a single DataFrame with closing prices."""
    combined_df = pd.DataFrame()
    
    # Read and combine each file
    for file in os.listdir(os.path.join('data_warehouse', '15_min_interval_stocks')):
        # Read the CSV
        df = pd.read_csv(os.path.join('data_warehouse', '15_min_interval_stocks', file))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        # Extract stock symbol from filename (assuming format like "AAPL.csv")
        stock_symbol = file.split('.')[0]
        
        # Only keep the 'close' column and rename it to the stock symbol
        close_prices = df['close'].rename(stock_symbol)
        
        # Add to combined DataFrame
        if combined_df.empty:
            combined_df = pd.DataFrame(close_prices)
        else:
            # Join the new stock's close prices with the existing dataframe
            # Using outer join to keep all dates even if some prices are missing
            combined_df = combined_df.join(close_prices, how='outer')
    
    # Sort by date in descending order (most recent first)
    combined_df.sort_index(ascending=False, inplace=True)
    
    # Save combined DataFrame
    combined_df.to_csv('combined_comm_stocks.csv')
    return combined_df

if __name__ == '__main__':
    print('beginning')
    combine_comm_sector_stocks()
    print('done')


