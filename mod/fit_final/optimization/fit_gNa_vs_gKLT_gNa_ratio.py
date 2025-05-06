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
param_file_path = os.path.join(os.path.dirname(__file__), "all_fitted_params.csv")

if os.path.exists(param_file_path):
    params_df = pd.read_csv(param_file_path)
    params_row = params_df.loc[0]

    fixed_params = {
        'gid': 0,
        'somaarea': 25e-6,  # cm² (25 pF assuming 1 µF/cm²)
        'erev': params_row["erev"],
        'gleak': params_row["gleak"],
        'gh': params_row["gh"],
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
gna_values = np.linspace(1, 600, 100)        # Sodium conductance in nS
ratios = np.linspace(0.001, 0.1, 100)            # gNa/gKLT ratios

spike_matrix = np.zeros((len(ratios), len(gna_values)))

# === Simulation parameters ===
stim_start = 10      # ms
stim_end = 310       # ms
stim_amp = 0.2       # nA
threshold = -5       # mV for spike detection

# === Run simulations ===
for i, ratio in enumerate(ratios):
    for j, gna in enumerate(gna_values):
        gklt = gna * ratio

        # Update parameters
        fixed_params['gna'] = gna
        fixed_params['gklt'] = gklt

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

classification_map = np.zeros_like(spike_matrix)

classification_map[spike_matrix == 0] = 0                    # Silent
classification_map[(spike_matrix >= 1) & (spike_matrix <= 3)] = 1  # Phasic
classification_map[spike_matrix >= 4] = 2                    # Tonic
# === Plotting ===
plt.figure(figsize=(10, 8))
plt.imshow(spike_matrix, origin='lower', aspect='auto',
           extent=[gna_values[0], gna_values[-1], ratios[0], ratios[-1]],
           cmap='Blues')
plt.colorbar(label='Number of Spikes')
plt.xlabel('gNa (nS)')
plt.ylabel('gNa / gKLT Ratio')
plt.title('Spike Count vs gNa and gNa/gKLT Ratio')
plt.grid(False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8))
im = plt.imshow(classification_map, origin='lower', aspect='auto',
                extent=[gna_values[0], gna_values[-1], ratios[0], ratios[-1]],
                cmap='viridis', vmin=0, vmax=4)

cbar = plt.colorbar(ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['Silent', 'Phasic', 'Tonic'])

plt.xlabel('Max Sodium Conductance (nS)')
plt.ylabel('gKLT / gNa Ratio')
plt.title('Classification of Neuron Firing Behavior')
plt.grid(False)
plt.show()