# plot_fit_bluepyopt_overlay.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config_bpop

from neuron_model import create_neuron
from simulation_bpop import run_simulation
from data_loader import load_heka_data

# Load best parameters
param_file = f"{config_bpop.output_dir}/optimized_parameters_bluepyopt.csv"
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

# Simulate with fitted parameters
t_sim_list, v_sim_list = run_simulation(soma, axon, dend, params)

# --- Plot overlay of all sweeps ---
plt.figure(figsize=(14, 8))

for idx in range(n_sweeps):
    v_exp = voltage[idx] * 1000
    t_exp = time[idx] * 1000

    plt.plot(t_exp, v_exp, color="black", alpha=0.4, label='Experimental' if idx == 0 else "")

# Plot simulated traces
for idx in range(n_sweeps):
    plt.plot(t_sim_list[idx], v_sim_list[idx], '--', color="blue", linewidth=1.5, alpha=0.7, label='Simulated' if idx == 0 else "")

plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Overlay: All Experimental Sweeps and Simulated Fit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Calculate and plot sweep-by-sweep fitting errors ---
sweep_errors = []

for t_exp, v_exp, t_sim, v_sim in zip(time, voltage, t_sim_list, v_sim_list):
    v_exp_mV = v_exp * 1000
    t_exp_ms = t_exp * 1000
    v_sim_interp = np.interp(t_exp_ms, t_sim, v_sim)  # interpolate sim onto experimental timebase
    mse = np.mean((v_exp_mV - v_sim_interp)**2)
    sweep_errors.append(mse)

# Plot sweep-by-sweep errors
plt.figure(figsize=(10, 5))
plt.bar(range(len(sweep_errors)), sweep_errors)
plt.xlabel("Sweep index")
plt.ylabel("Mean Squared Error (mV²)")
plt.title("Sweep-by-sweep fitting errors")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Print top 3 worst fitting sweeps ---
worst_sweeps = np.argsort(sweep_errors)[::-1][:3]  # Sort descending and take top 3

print("\nTop 3 worst sweeps (highest MSE):")
for idx in worst_sweeps:
    print(f"Sweep {idx}: MSE = {sweep_errors[idx]:.2f} mV²")
