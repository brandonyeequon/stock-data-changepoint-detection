import os
from pathlib import Path

# Define paths
log_path = Path('stock_data_fetch_2002-2000.log')
error_path = Path('stock_data_fetch_errors.log')

print(f"Reading from: {log_path}")
print(f"Writing to: {error_path}")

# Check if source file exists
if not log_path.exists():
    print(f"Source file not found: {log_path}")
    exit()

# iterates through stock_data_fetch.log and appends each log with "ERROR" to a new file
with open(log_path, 'r') as file:
    error_count = 0
    for line in file:
        if 'ERROR' in line.upper():  # Case-insensitive check
            with open(error_path, 'a') as error_file:
                error_file.write(line)
            error_count += 1

print(f"Found {error_count} errors")
print(f"Error log saved to: {error_path}")
print(f"File exists: {error_path.exists()}")

bad_tickers = []
with open(error_path, 'r') as file:
    for line in file:
        line_list = line.split(' ')
        if line_list[7].endswith('|'):
            bad_tickers.append(line_list[7][1:-1])
        else:
            bad_tickers.append(line_list[7])
print(len(bad_tickers))
# Read current tickers
ticker_path = Path('stock_tickers.txt')
with open(ticker_path, 'r') as f:
    tickers = [line.strip() for line in f]

print(f"Original ticker count: {len(tickers)}")

# Remove problematic tickers 
clean_tickers = [ticker for ticker in tickers if ticker not in bad_tickers]

print(f"Clean ticker count: {len(clean_tickers)}")
print(f"Removed {len(tickers) - len(clean_tickers)} tickers")

# Save clean tickers back to file
with open(ticker_path, 'w') as f:
    f.write('\n'.join(clean_tickers))

print("Done! Updated stock_tickers.txt")

# Optional: Save removed tickers to a separate file for reference
with open('removed_tickers.txt', 'w') as f:
    f.write('\n'.join(sorted(bad_tickers)))
print("Saved removed tickers to removed_tickers.txt")
