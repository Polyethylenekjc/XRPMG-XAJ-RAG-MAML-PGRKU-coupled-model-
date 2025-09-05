
"""
xaj_demo.py — Example usage of xaj_model.py

This demo creates synthetic rainfall and E0, a synthetic "observed" hydrograph,
runs the model, calibrates parameters by random search, and plots results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xaj_model import XAJParameters, XAJState, XAJModel, random_search_calibration, nse, rmse, kge

np.random.seed(0)

# --- Synthetic inputs ---
n = 200
dates = pd.date_range("2000-01-01", periods=n, freq="D")
P = np.maximum(0, np.random.gamma(2.0, 2.0, size=n) - 2)  # intermittent rain
E0 = np.full(n, 2.5) + 0.5*np.sin(np.linspace(0, 10*np.pi, n))  # seasonal-ish

# Create a synthetic "observed" Q by running the model with a hidden parameter set + noise
true_params = XAJParameters(WM=180, B=0.45, CU=40, CM=70, CL=70, IMP=0.03,
                            CS=0.25, CI=0.35, KS=0.55, KI=0.35, KG=0.12,
                            EM_red_m=0.8, EM_red_l=0.5)

model_true = XAJModel(true_params, XAJState(), use_muskingum=False)
out_true = model_true.run(P, E0)
Qobs = out_true["Q"] + np.random.normal(0, 0.2, size=n)  # add noise

# --- Split into calibration and validation ---
split = int(0.7*n)
P_cal, E0_cal, Qobs_cal = P[:split], E0[:split], Qobs[:split]
P_val, E0_val, Qobs_val = P[split:], E0[split:], Qobs[split:]

# --- Calibrate ---
best_p, best_score, sim_cal = random_search_calibration(
    P_cal, E0_cal, Qobs_cal,
    n_iter=300, seed=123, use_muskingum=False, score="NSE"
)

# --- Validate ---
model_best = XAJModel(best_p, XAJState(), use_muskingum=False)
out_val = model_best.run(P_val, E0_val)
sim_val = out_val["Q"]

# --- Metrics ---
nse_cal = nse(sim_cal, Qobs_cal)
nse_val = nse(sim_val, Qobs_val)
rmse_cal = rmse(sim_cal, Qobs_cal)
rmse_val = rmse(sim_val, Qobs_val)
kge_cal = kge(sim_cal, Qobs_cal)
kge_val = kge(sim_val, Qobs_val)

print("Best parameters:", best_p)
print(f"Calibration NSE={nse_cal:.3f}, RMSE={rmse_cal:.3f}, KGE={kge_cal:.3f}")
print(f"Validation   NSE={nse_val:.3f}, RMSE={rmse_val:.3f}, KGE={kge_val:.3f}")

# --- Save a CSV template ---
df = pd.DataFrame({"date": dates, "P": P, "E0": E0, "Qobs": Qobs})
csv_path = Path("/mnt/data/xaj_template.csv")
df.to_csv(csv_path, index=False)

# --- Plot observed vs simulated (validation) ---
plt.figure()
plt.plot(dates[split:], Qobs_val, label="Observed Q")
plt.plot(dates[split:], sim_val, label="Simulated Q")
plt.title("Observed vs Simulated Discharge (Validation)")
plt.xlabel("Date")
plt.ylabel("Discharge (mm/timestep)")
plt.legend()
plt.tight_layout()
fig_path = Path("/mnt/data/xaj_validation_plot.png")
plt.savefig(fig_path, dpi=160)

# --- Also write calibration/validation results to CSV ---
res = pd.DataFrame({
    "date": dates,
    "P": P, "E0": E0, "Qobs": Qobs,
    "Qsim": np.concatenate([sim_cal, sim_val])
})
res_path = Path("/mnt/data/xaj_results.csv")
res.to_csv(res_path, index=False)

print("Files saved:")
print(csv_path)
print(fig_path)
print(res_path)
