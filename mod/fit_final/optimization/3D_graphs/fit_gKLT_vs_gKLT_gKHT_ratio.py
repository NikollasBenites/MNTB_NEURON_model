import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuron import h
from MNTB_PN_fit import MNTB
import MNTB_PN_myFunctions as mFun

# Load NEURON hoc files
h.load_file('stdrun.hoc')
h.celsius = 35

# === Load fitted parameters ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
param_file_path = os.path.join(os.path.dirname(__file__), "..","all_fitted_params.csv")

if os.path.exists(param_file_path):
    params_df = pd.read_csv(param_file_path)
    params_row = params_df.loc[0]

    fixed_params = {
        'gid': 0,
        'somaarea': 25e-6,  # cm² (25 pF assuming 1 µF/cm²)
        'erev': params_row["erev"],
        'gleak': params_row["gleak"],
        'gh': params_row["gh"],
        'gna': params_row["gna"],
        'gklt': params_row["gklt"],
        'gkht': params_row["gkht"],  # this will be updated later
        'gka': params_row["gka"],
        'ena': 62.77,
        'ek': -106.81,
        'cam': params_row["cam"],
        'kam': params_row["kam"],
        'cbm': params_row["cbm"],
        'kbm': params_row["kbm"],
        'cah': params_row["cah"],
        'kah': params_row["kah"],
        'cbh': params_row["cbh"],
        'kbh': params_row["kbh"],
        'can': params_row["can"],
        'kan': params_row["kan"],
        'cbn': params_row["cbn"],
        'kbn': params_row["kbn"],
        'cap': params_row["cap"],
        'kap': params_row["kap"],
        'cbp': params_row["cbp"],
        'kbp': params_row["kbp"],
    }

    print("📥 Parameters loaded successfully.")
else:
    raise FileNotFoundError(f"Parameter file not found at: {param_file_path}")

# === Define ranges ===
gklt_values = np.linspace(1, 50, 100)        # Sodium conductance in nS
ratios = np.linspace(0.01, 1.0, 50)            # gKLT/gKHT ratios

spike_matrix = np.zeros((len(ratios), len(gklt_values)))

# === Simulation parameters ===
stim_start = 10      # ms
stim_end = 310       # ms
stim_amp = 0.2       # nA
threshold = -5       # mV for spike detection

# === Run simulations ===
for i, ratio in enumerate(ratios):
    for j, gklt in enumerate(gklt_values):
        gkht = gklt / ratio

        # Update parameters
        fixed_params['gklt'] = gklt
        fixed_params['gkht'] = gkht

        neuron = MNTB(**fixed_params)

        # Inject current
        stim = h.IClamp(neuron.soma(0.5))
        stim.delay = stim_start
        stim.dur = stim_end - stim_start
        stim.amp = stim_amp

        # Record voltage and time
        v = h.Vector().record(neuron.soma(0.5)._ref_v)
        t = h.Vector().record(h._ref_t)

        mFun.custom_init(-70)
        h.continuerun(510)

        v_np = np.array(v)
        t_np = np.array(t)

        # Detect spikes
        spike_indices = np.where((v_np[:-1] < threshold) & (v_np[1:] >= threshold))[0]
        spike_times = t_np[spike_indices]
        valid_spikes = np.logical_and(spike_times >= stim_start, spike_times <= stim_end)
        spike_count = np.sum(valid_spikes)

        # Store result
        spike_matrix[i, j] = spike_count

# === Plotting ===
plt.figure(figsize=(10, 8))
plt.imshow(spike_matrix, origin='lower', aspect='auto',
           extent=[gklt_values[0], gklt_values[-1], ratios[0], ratios[-1]],
           cmap='viridis')
plt.colorbar(label='Number of Spikes')
plt.xlabel('gKLT (nS)')
plt.ylabel('gKLT / gKHT Ratio')
plt.title('Spike Count vs gKLT and gKLT/gKHT Ratio')
plt.grid(False)
plt.tight_layout()
plt.show()
