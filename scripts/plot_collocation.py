import pandas as pd
import matplotlib.pyplot as plt

df_1000 = pd.read_csv("datasets/collocation.csv", nrows=2000)

plt.figure(figsize=(6, 6))
plt.scatter(df_1000["x"][1000:], df_1000["y"][1000:], s=5)  # s — размер точки
plt.xlabel("x")
plt.ylabel("y")
plt.title("First 1000 points")
plt.axis("equal")  # чтобы масштаб по осям был одинаковый
plt.show()