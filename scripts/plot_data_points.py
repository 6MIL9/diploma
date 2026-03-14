import pandas as pd
import matplotlib.pyplot as plt

file = "datasets/data_points_phi.csv"

df = pd.read_csv(file)

# выбрать нужный момент времени
t_target = 2.0
df_t = df[df["t"] == t_target]

x = df_t["x"].values
y = df_t["y"].values

plt.figure(figsize=(5,8))
plt.scatter(x, y, s=5)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Sampled points at t={t_target}")
plt.gca().set_aspect("equal")

plt.show()