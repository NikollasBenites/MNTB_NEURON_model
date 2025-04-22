# main.py

import config
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_heka_data, select_sweep
from neuron_model import create_neuron
from fitting import fit_parameters
from plotting import plot_voltage_fit, plot_phase_plane
from simulation import cost_function, run_simulation
from datetime import datetime
from neuron import h

np.random.seed(42)
h.celsius = config.celsius

# Create timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print("Starting fitting pipeline...")
np.random.seed(1)
# Load Data
voltage, time, stim, labels = load_heka_data(config.full_path_to_file, config.group_idx, config.series_idx, config.channel_idx)
v_exp, t_exp, sweep_idx = select_sweep(voltage, time, labels)

# Create Model
soma, axon, dend = create_neuron()

parent = dend.parentseg()
parent1 = axon.parentseg()
if parent is not None:
    print(f"dend is connected to {parent.sec.name()} at {parent.x:.2f}")
else:
    print("dend has no parent (not connected)")
if parent1 is not None:
    print(f"axon is connected to {parent.sec.name()} at {parent1.x:.2f}")
else:
    print("axon has no parent (not connected)")

# Define Cost Function
cost_fn = lambda params: cost_function(params, soma, axon, dend, t_exp, v_exp)

# Run Fit
result = fit_parameters(cost_fn)
params_opt = result.x

# Final Simulation
t_sim, v_sim = run_simulation(soma, axon, dend, params_opt)

param_names = [
    "gna", "gkht", "gklt","gh","gleak",
    "cam", "kam", "cbm", "kbm",
    "cah", "kah", "cbh", "kbh",
    "can", "kan", "cbn", "kbn",
    "cap", "kap", "cbp", "kbp",
    "na_scale", "kht_scale", "klt_scale", "ih_soma","ih_dend", "stim_amp"
]
# Create DataFrame
df = pd.DataFrame({
    'Parameter': param_names,
    'Value': params_opt
})

# Create output directory if it doesn't exist
os.makedirs(config.output_dir, exist_ok=True)

# Save
output_path = os.path.join(config.output_dir, 'optimized_parameters.csv')
df.to_csv(output_path, index=False)

print(f"\nParameters saved to: {output_path}")

# Print each param nicely
for name, value in zip(param_names, params_opt):
    print(f"{name:10s}: {value:.6f}")

# Plot
plot_voltage_fit(t_exp, v_exp, t_sim, v_sim)# Now plot phase plane
plt.figure(figsize=(8,6))  # <<--- create a new figure manually
plot_phase_plane(v_exp, t_exp, label='Experimental')
plot_phase_plane(v_sim, t_sim, label='Simulated')
plt.title('Phase Plane Plot')
plt.xlabel('Voltage (mV)')
plt.ylabel('dV/dt (mV/ms)')
plt.legend()
plt.grid()
plt.tight_layout()

# Save and show (only once)
if config.save_figures:
    plt.savefig(f"{config.output_dir}/phase_plane_plot.png", dpi=300)

if config.save_figures:
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    plt.savefig(f"{config.output_dir}/voltage_fit.png", dpi=300)


if config.show_plots:
    plt.show()
print(f"Optimized stim_amp: {params_opt[-1]:.3f} nA")

print("Optimization completed.")
