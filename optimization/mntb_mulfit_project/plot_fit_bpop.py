# plot_fit_bluepyopt.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config_bpop

from neuron_model import create_neuron
from simulation_bpop import run_simulation
from data_loader import load_heka_data

# Load best parameters
param_file = os.path.join(config_bpop.output_dir, "optimized_parameters_bluepyopt.csv")
df = pd.read_csv(param_file)
params = df['Value'].values

# Load experimental data
voltage, time, stim, labels = load_heka_data(
    config_bpop.full_path_to_file,
    config_bpop.group_idx,
    config_bpop.series_idx,
    config_bpop.channel_idx
)

n_sweeps = len(voltage)

# Create neuron model
soma, axon, dend = create_neuron()

# Simulate with best parameters
t_sim_list, v_sim_list = run_simulation(soma, axon, dend, params)

# Ensure output directory exists
os.makedirs(config_bpop.output_dir, exist_ok=True)

# --- Plot 1: Overlay Experimental vs Simulated ---
plt.figure(figsize=(14, 8))

for idx in range(n_sweeps):
    v_exp = voltage[idx] * 1000
    t_exp = time[idx] * 1000
    v_sim = v_sim_list[idx]
    t_sim = t_sim_list[idx]

    # Interpolate simulation to experimental time
    v_sim_interp = np.interp(t_exp, t_sim, v_sim)

    plt.plot(t_exp, v_exp, color="black", alpha=0.5, label="Experimental" if idx == 0 else "")
    plt.plot(t_exp, v_sim_interp, '--', color="blue", alpha=0.7, label="Simulated" if idx == 0 else "")

plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Overlay: All Experimental and Simulated Sweeps")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
plt.savefig(os.path.join(config_bpop.output_dir, "overlay_exp_sim_traces.png"), dpi=300)
plt.show()

# --- Calculate and Plot 2: Sweep-by-sweep MSE errors ---
sweep_errors = []
for t_exp, v_exp, t_sim, v_sim in zip(time, voltage, t_sim_list, v_sim_list):
    v_exp_mV = v_exp * 1000
    t_exp_ms = t_exp * 1000
    v_sim_interp = np.interp(t_exp_ms, t_sim, v_sim)
    mse = np.mean((v_exp_mV - v_sim_interp)**2)
    sweep_errors.append(mse)

# Plot errors
plt.figure(figsize=(10, 5))
plt.bar(range(len(sweep_errors)), sweep_errors)
plt.xlabel("Sweep Index")
plt.ylabel("Mean Squared Error (mV²)")
plt.title("Sweep-by-Sweep Fitting Errors")
plt.grid(True)
plt.tight_layout()

# Save error plot
plt.savefig(os.path.join(config_bpop.output_dir, "sweep_errors.png"), dpi=300)
plt.show()

# --- Print top 3 worst sweeps ---
worst_sweeps = np.argsort(sweep_errors)[::-1][:3]
print("\nTop 3 Worst Fitting Sweeps (Highest MSE):")
for idx in worst_sweeps:
    print(f"Sweep {idx}: MSE = {sweep_errors[idx]:.2f} mV²")

# --- (Optional) Save sweep errors to CSV ---
errors_df = pd.DataFrame({
    'Sweep': np.arange(len(sweep_errors)),
    'MSE (mV^2)': sweep_errors
})
errors_df.to_csv(os.path.join(config_bpop.output_dir, "sweep_errors.csv"), index=False)
print(f"\nSweep errors saved to {config_bpop.output_dir}/sweep_errors.csv")
