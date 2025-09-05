import pandas as pd
import matplotlib.pyplot as plt
from xaj_model import pso_calibration, XAJModel

# ============ 1. Read Data ============
# Please replace your_data.csv with your own filename
# If it's tab-separated, keep sep="\t"; if it's comma-separated, remove the sep parameter
data = pd.read_csv("/home/fifth/code/Python/XAJ/data/taohe_TAD.csv")

# Observed runoff
Qobs = data["runoffs"].values
Qobs = (Qobs-Qobs.min())/(Qobs.max()-Qobs.min())

# Driving data: Precipitation (P) and potential evaporation (E0)
P = data["tp"].values
E0 = data["e"].values

# ============ 2. Perform Calibration ============
best_params, best_score, sim_cal = pso_calibration(
    P, E0, Qobs,
    n_iter=50,   # Number of search iterations, can be increased for better accuracy
    score="RMSE"   # Options: "NSE", "KGE", "RMSE"
)

print("Optimal RMSE:", best_score)
print("Optimal Parameters:", best_params.__dict__)

# ============ 3. Rerun for the Full Period with Optimal Parameters ============
model = XAJModel(best_params)
out = model.run(P, E0)
Qsim = out["Q"]

# Save results
df = pd.DataFrame({"Obs": Qobs, "Sim": Qsim})
df.to_csv("xaj_results.csv", index=False)
print("Simulation results have been saved to xaj_results.csv")

# ============ 4. Save Optimal Parameters to Excel ============
# Convert parameters to a dictionary and add the objective function value
params_dict = best_params.__dict__.copy()
params_dict['objective_function'] = best_score
params_dict['objective_name'] = 'RMSE'

# Create DataFrame and save to Excel
params_df = pd.DataFrame([params_dict])
params_df.to_excel("xaj_best_parameters.xlsx", index=False)
print("Optimal parameters have been saved to xaj_best_parameters.xlsx")

# ============ 5. Visualize Comparison ============
plt.figure(figsize=(12,5))
plt.plot(Qobs, label="Observed")
plt.plot(Qsim, label="Simulated")
plt.legend()
plt.title("Xinanjiang Model Calibration Result")
plt.xlabel("Time step")
plt.ylabel("Runoff (mm)")
plt.tight_layout()
plt.savefig("xaj_calibration_plot.png", dpi=150)
plt.show()
print("Result plot has been saved as xaj_calibration_plot.png")