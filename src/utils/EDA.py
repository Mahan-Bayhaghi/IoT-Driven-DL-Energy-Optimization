import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

matplotlib.use('TkAgg')
SHOW = True

pd.set_option("display.max_colwidth", None)

df = pd.read_csv("../../data/raw/MER_T02_01A.csv")

mapping = (
    df.groupby("Column_Order")["Description"]
    .unique()
    .reset_index()
)

print(mapping)

# what matters based on the mapping
# Index       Column_Order                        Description
# 4              5                    [Total Energy Consumed by the Residential Sector]
# 9             10                    [Total Energy Consumed by the Commercial Sector]
# 14            15                    [Total Energy Consumed by the Industrial Sector]

# Filter for Residential (5) and Commercial (10) sectors
df_filtered = df[df['Column_Order'].isin([5, 10])].copy()

df_filtered['Value'] = pd.to_numeric(df_filtered['Value'], errors='coerce')

# Convert YYYYMM to datetime
# dataset uses '13' for annual totals -> we should drop those for time-series
df_filtered = df_filtered[df_filtered['YYYYMM'].astype(str).str[-2:] != '13']
df_filtered['Date'] = pd.to_datetime(df_filtered['YYYYMM'], format='%Y%m')

df_pivot = df_filtered.pivot(index='Date', columns='Description', values='Value')
print(df_pivot.describe())

print(f"Missing values:\n{df_pivot.isnull().sum()}")

df_pivot.plot(figsize=(12, 6), title="Energy Consumption: Residential vs Commercial")
plt.ylabel("Energy Units")
# plt.savefig("../../results/plots/EDA/Energy_Consumption_Residential_vs_Commercial.png")
if SHOW:
    plt.show()

# seasonal decomposition for Residential Sector
result = seasonal_decompose(df_pivot.iloc[:, 0].dropna(), model='additive', period=12)
result.plot()
# plt.savefig("../../results/plots/EDA/Seasonal_Decomposition_of_Energy Consumption.png")
if SHOW:
    plt.show()


def create_sequences(data, window_size):
    # prepares data for the hybrid model input
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i: i + window_size])
    return sequences


window_size = 12
sample_seq = create_sequences(df_pivot.iloc[:, 0].values, window_size)
print(f"Number of sequences created: {len(sample_seq)}")


# ACF plot to identify lag patterns
plot_acf(df_pivot.iloc[:, 0].dropna(), lags=24)
plt.title("Autocorrelation - Residential Sector")
# plt.savefig("../../results/plots/EDA/Autocorrelation_Residential_Sector.png")
if SHOW:
    plt.show()

plt.close()
