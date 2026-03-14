"""
Проверка качества построения phi_data из interFoam alpha.water (VOF).

Ожидаемый вход: CSV с колонками:
x, y, t, alpha_water
(названия можно поменять в CONFIG ниже)

Что проверяем:
1) phi_data в [-1,1], знак фаз корректный
2) интерфейс (phi≈0) совпадает с alpha≈0.5
3) если строим SDF d: ||grad d||≈1 в полосе около интерфейса
4) “толщина” интерфейса согласуется с параметром s в tanh(s*d)
5) визуализации среза на выбранном времени

Зависимости: numpy pandas scipy matplotlib
pip install numpy pandas scipy matplotlib
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt


# ---------------- CONFIG ----------------
@dataclass
class Config:
    csv_path: str = "data_phi_tanh/data_0.csv"
    col_x: str = "Points:0"
    col_y: str = "Points:1"
    col_t: str = "Time"
    col_a: str = "alpha.water"

    # какое время проверять (будет выбран ближайший снапшот)
    t_check: float = 0.0

    # параметр "резкости" для tanh(s*...)
    s: float = 50.0

    # полоса около интерфейса для проверки ||grad d|| (в метрах)
    band_m: float = 0.03

    # уровни для оценки "толщины" интерфейса
    phi_lo: float = -0.9
    phi_hi: float = +0.9

CFG = Config()
# ----------------------------------------


def nearest_time(df: pd.DataFrame, t_check: float, col_t: str) -> float:
    ts = np.sort(df[col_t].unique())
    idx = np.argmin(np.abs(ts - t_check))
    return float(ts[idx])


def is_regular_grid(x: np.ndarray, y: np.ndarray, tol: float = 1e-10) -> bool:
    ux = np.unique(np.round(x / tol) * tol)
    uy = np.unique(np.round(y / tol) * tol)
    # регулярная, если точки полностью образуют декартово произведение
    return len(ux) * len(uy) == len(x)


def to_grid_regular(x, y, v):
    ux = np.unique(x)
    uy = np.unique(y)
    ux.sort(); uy.sort()
    nx, ny = len(ux), len(uy)

    # маппинг в 2D
    xi = np.searchsorted(ux, x)
    yi = np.searchsorted(uy, y)
    grid = np.full((ny, nx), np.nan, dtype=float)
    grid[yi, xi] = v
    return ux, uy, grid


def build_sdf_from_alpha(alpha_grid: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Signed distance до изолинии alpha=0.5:
      d > 0 в воде (alpha>0.5)
      d < 0 в воздухе
    """
    mask = alpha_grid > 0.5
    d_water = distance_transform_edt(mask, sampling=(dy, dx))
    d_air   = distance_transform_edt(~mask, sampling=(dy, dx))
    d = d_water - d_air
    return d


def grad_mag(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    # np.gradient returns [d/dy, d/dx]
    fy, fx = np.gradient(field, dy, dx)
    return np.sqrt(fx*fx + fy*fy)


def interface_thickness_along_vertical(phi_grid, xg, yg, x0=0.5, phi_lo=-0.9, phi_hi=0.9):
    """
    Оценка толщины интерфейса по вертикальной линии x≈x0:
    берём y, где phi пересекает phi_lo и phi_hi (прибл. через интерполяцию).
    """
    ix = int(np.argmin(np.abs(xg - x0)))
    col = phi_grid[:, ix]
    y = yg

    # найти область пересечения: предполагаем монотонный переход около интерфейса.
    # Для устойчивости ищем точки с (phi_lo < phi < phi_hi)
    mask = np.isfinite(col)
    y = y[mask]
    col = col[mask]
    if len(col) < 10:
        return np.nan, ix

    # пересечение уровней по линейной интерполяции
    def crossing(level):
        sgn = col - level
        idxs = np.where(np.sign(sgn[:-1]) * np.sign(sgn[1:]) <= 0)[0]
        if len(idxs) == 0:
            return np.nan
        i = idxs[0]
        y0, y1 = y[i], y[i+1]
        f0, f1 = col[i], col[i+1]
        if f1 == f0:
            return y0
        return y0 + (level - f0) * (y1 - y0) / (f1 - f0)

    y_lo = crossing(phi_lo)
    y_hi = crossing(phi_hi)
    if np.isnan(y_lo) or np.isnan(y_hi):
        return np.nan, ix
    return abs(y_hi - y_lo), ix


def diagnostics(alpha_grid, phi_data, d_sdf, xg, yg, title_suffix, s, band_m):
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])

    # 1) базовая статистика
    def stats(name, arr):
        finite = np.isfinite(arr)
        return {
            "name": name,
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "nan_frac": float(1.0 - finite.mean()),
        }

    st_alpha = stats("alpha", alpha_grid)
    st_phi   = stats("phi_data", phi_data)

    # 2) интерфейс: где |phi| маленькое, alpha должна быть около 0.5
    eps_d = 2.0 * max(dx, dy)
    band = np.isfinite(d_sdf) & (np.abs(d_sdf) < eps_d)

    if band.any():
        a_in_band = alpha_grid[band]
        interface_alpha_mean = float(np.nanmean(a_in_band))
        interface_alpha_std  = float(np.nanstd(a_in_band))
    else:
        interface_alpha_mean = np.nan
        interface_alpha_std  = np.nan

    # 3) проверка SDF: ||grad d||≈1 в полосе |d|<band_m
    grad_d = grad_mag(d_sdf, dx, dy)
    band_d = np.isfinite(d_sdf) & (np.abs(d_sdf) < band_m)
    if band_d.any():
        g_mean = float(np.nanmean(grad_d[band_d]))
        g_med  = float(np.nanmedian(grad_d[band_d]))
        g_p10  = float(np.nanpercentile(grad_d[band_d], 10))
        g_p90  = float(np.nanpercentile(grad_d[band_d], 90))
    else:
        g_mean = g_med = g_p10 = g_p90 = np.nan

    # 4) оценка толщины интерфейса vs ожидаемая (по tanh)
    thickness, ix = interface_thickness_along_vertical(
        phi_data, xg, yg, x0=0.5, phi_lo=CFG.phi_lo, phi_hi=CFG.phi_hi
    )
    # теоретическая толщина между phi_lo и phi_hi для tanh(s*d):
    # d = atanh(phi)/s => thickness_d = (atanh(phi_hi)-atanh(phi_lo))/s
    # если phi_lo=-phi_hi => thickness_d = 2*atanh(phi_hi)/s
    if abs(CFG.phi_lo + CFG.phi_hi) < 1e-12:
        thickness_expected = float(2.0 * np.arctanh(CFG.phi_hi) / s)
    else:
        thickness_expected = float((np.arctanh(CFG.phi_hi) - np.arctanh(CFG.phi_lo)) / s)

    # ---- print summary ----
    print("\n=== Diagnostics", title_suffix, "===")
    print("alpha stats:", st_alpha)
    print("phi_data stats:", st_phi)
    print(f"Interface check (|d|<{eps_d}): mean(alpha)={interface_alpha_mean:.5f}, std={interface_alpha_std:.5f}")
    print(f"SDF grad check (|d|<{band_m} m): mean|∇d|={g_mean:.3f}, median={g_med:.3f}, p10={g_p10:.3f}, p90={g_p90:.3f}")
    print(f"Interface thickness along x≈0.5: measured={thickness:.5f} m, expected≈{thickness_expected:.5f} m (from tanh, s={s})")

    # ---- plots ----
    Xg, Yg = np.meshgrid(xg, yg)

    plt.figure()
    plt.title(f"alpha.water {title_suffix}")
    plt.pcolormesh(Xg, Yg, alpha_grid, shading="auto")
    plt.colorbar()
    plt.contour(Xg, Yg, alpha_grid, levels=[0.5])  # интерфейс
    plt.xlabel("x"); plt.ylabel("y")

    plt.figure()
    plt.title(f"phi_data {title_suffix}")
    plt.pcolormesh(Xg, Yg, phi_data, shading="auto")
    plt.colorbar()
    plt.contour(Xg, Yg, phi_data, levels=[0.0])
    plt.xlabel("x"); plt.ylabel("y")

    plt.figure()
    plt.title(f"SDF d(x,y) {title_suffix}")
    plt.pcolormesh(Xg, Yg, d_sdf, shading="auto")
    plt.colorbar()
    plt.contour(Xg, Yg, d_sdf, levels=[0.0])
    plt.xlabel("x"); plt.ylabel("y")

    plt.figure()
    plt.title(f"|∇d| {title_suffix} (target≈1 near interface)")
    plt.pcolormesh(Xg, Yg, grad_d, shading="auto")
    plt.colorbar()
    plt.contour(Xg, Yg, d_sdf, levels=[-band_m, 0.0, band_m])
    plt.xlabel("x"); plt.ylabel("y")

    plt.figure()
    plt.title(f"Histogram phi_data {title_suffix}")
    vals = phi_data[np.isfinite(phi_data)].ravel()
    plt.hist(vals, bins=80)
    plt.xlabel("phi"); plt.ylabel("count")

    plt.show()


def main(cfg: Config):
    df = pd.read_csv(cfg.csv_path)
    for c in [cfg.col_x, cfg.col_y, cfg.col_t, cfg.col_a]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in CSV. Found: {list(df.columns)}")

    t0 = nearest_time(df, cfg.t_check, cfg.col_t)
    print(f"Using snapshot time t={t0} (nearest to t_check={cfg.t_check})")

    dft = df[df[cfg.col_t] == t0].copy()
    x = dft[cfg.col_x].to_numpy(dtype=float)
    y = dft[cfg.col_y].to_numpy(dtype=float)
    a = dft[cfg.col_a].to_numpy(dtype=float)

    xg, yg, alpha_grid = to_grid_regular(x, y, a)

    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    d_sdf = build_sdf_from_alpha(alpha_grid, dx=dx, dy=dy)
    phi_B = np.tanh(cfg.s * d_sdf)

    diagnostics(alpha_grid, phi_B, d_sdf, xg, yg, title_suffix=f"(t={t0}) VARIANT B: phi=tanh(s*SDF)", s=cfg.s, band_m=cfg.band_m)

if __name__ == "__main__":
    main(CFG)
