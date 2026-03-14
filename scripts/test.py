import pandas as pd

file1 = "datasets/collocation.csv"
file2 = "datasets/data_points_phi.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# вывод первых 10 строк
print(f"First 10 rows of {file1}:")
print(df1.head(10))

print(f"\nFirst 10 rows of {file2}:")
print(df2.head(10))