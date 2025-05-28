from alpha_vantage_client import AlphaVantageClient

client = AlphaVantageClient()
df = client.fetch_stock_data("A", "11", "2024")
df.to_csv("data_warehouse/confirmed_data/A_2024-11.csv", index=False)
