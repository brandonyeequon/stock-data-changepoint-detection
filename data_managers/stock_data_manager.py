import os

class StockDataManager:
    def __init__(self):
        self.csv_dir = "data_warehouse/15_min_interval_stocks"
        os.makedirs(self.csv_dir, exist_ok=True)
    
    def check_data_length(self) -> int:
        total_length = 0
        for file in os.listdir(self.csv_dir):
            if file.endswith('.csv'):
                csv_file_path = os.path.join(self.csv_dir, file)
                with open(csv_file_path, 'r') as f:
                    # Count the lines and subtract 1 for the header
                    row_count = sum(1 for _ in f) - 1
                    total_length += row_count
        
        print(f"Total length of all CSV files: {total_length}")
        return total_length

if __name__ == "__main__":
    manager = StockDataManager()
    manager.check_data_length()
