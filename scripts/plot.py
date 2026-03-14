import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = "data_phi_new/data_100.csv"

def load_fields(file):
    df = pd.read_csv(file)

    x = df["Points:0"].values
    y = df["Points:1"].values
    phi = df["phi"].values
    alpha = df["alpha"].values

    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))

    X, Y = np.meshgrid(x_unique, y_unique)

    Z_phi = phi.reshape(len(y_unique), len(x_unique))
    Z_alpha = alpha.reshape(len(y_unique), len(x_unique))

    # δ = 1 − tanh²(80 * alpha)
    Z_delta = 1 - np.tanh(80 * Z_alpha) ** 2

    return X, Y, Z_phi, Z_alpha, Z_delta

X, Y, Z_phi, Z_alpha, Z_delta = load_fields(file)

# явные границы для alpha
alpha_min = Z_alpha.min()
alpha_max = Z_alpha.max()

fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                         sharex=True, sharey=True,
                         constrained_layout=True)

# phi
im_phi = axes[0].pcolormesh(X, Y, Z_phi, shading="auto")
axes[0].set_title("phi")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
fig.colorbar(im_phi, ax=axes[0], label="phi")

# alpha с фиксированными границами
im_alpha = axes[1].pcolormesh(
    X, Y, Z_alpha,
    shading="auto",
    vmin=alpha_min,
    vmax=alpha_max
)
axes[1].set_title(f"alpha [{alpha_min:.3g}, {alpha_max:.3g}]")
axes[1].set_xlabel("x")
fig.colorbar(im_alpha, ax=axes[1], label="alpha")

# delta (0–1)
im_delta = axes[2].pcolormesh(
    X, Y, Z_delta,
    shading="auto",
    vmin=0,
    vmax=1
)
axes[2].set_title(r"$\delta = 1 - \tanh^2(80\,\alpha)$")
axes[2].set_xlabel("x")
fig.colorbar(im_delta, ax=axes[2], label="delta")

plt.show()