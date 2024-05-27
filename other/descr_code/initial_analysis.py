import pandas as pd
# csv into a dataframe
df = pd.read_csv('data/european_wholesale_electricity_price_data_daily-5.csv')

# first few rows of the df to get a sence og how data looks like
print("First few rows of the DataFrame:")
print(df.head())

# summary stats
print("\nSummary statistics for numeric column price:")
print(df.describe())

#missing values
print("\nCheck for missing values:")
print(df.isnull().sum())
