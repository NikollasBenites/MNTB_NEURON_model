import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuron import h
from MNTB_PN_fit import MNTB
import MNTB_PN_myFunctions as mFun
# Simulation parameters
h.load_file('stdrun.hoc')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
param_file_path = os.path.join(os.path.dirname(__file__), "all_fitted_params.csv")

if os.path.exists(param_file_path):
    params_df = pd.read_csv(param_file_path)
    params_row = params_df.loc[0]

    # Dictionary
    fixed_params = {
        'gid': 0,
        'somaarea': (25e-6),  # 25 pF -> 25e-6 uF; already area in cm^2 assuming 1 uF/cm^2
        'erev': params_row["erev"],
        'gleak': params_row["gleak"],
        'gh': params_row["gh"],
        'gka': params_row["gka"],
        'gkht': params_row["gkht"],
        'ena': 62.77,  # or params_row["ena"] if you add it to CSV
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

    print("📥 Parameters loaded and organized into fixed_params!")
else:
    raise FileNotFoundError(f"Parameter file not found at {param_file_path}")


h.celsius = 35

# === Define ranges to explore ===
gna_values = np.linspace(200, 600, 50)       # Sodium conductance (nS)
erev_values = np.linspace(-90, -50, 50)    # Leak reversal (mV)

# === Spike detection settings ===
detection_threshold = -5  # mV
stim_start = 10           # ms
stim_end = 310            # ms

# === Prepare a matrix to store results ===
spike_matrix = np.zeros((len(erev_values), len(gna_values)))

# === Start simulations ===
for i, erev in enumerate(erev_values):
    for j, gna in enumerate(gna_values):
        # Update parameters for this run
        fixed_params['erev'] = erev
        fixed_params['gna'] = gna

        # Create a new MNTB neuron
        neuron = MNTB(gka, **fixed_params)

        # Apply current injection
        stim = h.IClamp(neuron.soma(0.5))
        stim.delay = stim_start
        stim.dur = stim_end - stim_start
        stim.amp = 0.3    # nA

        # Record
        v = h.Vector().record(neuron.soma(0.5)._ref_v)
        t = h.Vector().record(h._ref_t)

        # Run
        v_init = -70
        mFun.custom_init(v_init)
        h.continuerun(510)

        # Analyze
        v_np = np.array(v)
        t_np = np.array(t)
        spike_indices = np.where((v_np[:-1] < detection_threshold) & (v_np[1:] >= detection_threshold))[0]
        spike_times = t_np[spike_indices]
        valid_spikes = np.logical_and(spike_times >= stim_start, spike_times <= stim_end)

        spike_matrix[i, j] = np.sum(valid_spikes)


classification_map = np.zeros_like(spike_matrix)

# Classify based on spike counts
classification_map[spike_matrix == 0] = 0     # Silent
classification_map[spike_matrix == 1] = 1      # Phasic (1 spike)
classification_map[spike_matrix >= 2] = 2      # Tonic (2 or more spikes)


# === Plotting the Heatmap ===
plt.figure(figsize=(10,8))
plt.imshow(spike_matrix, origin='lower', aspect='auto',
           extent=[gna_values[0], gna_values[-1], erev_values[0], erev_values[-1]],
           cmap='viridis')
plt.colorbar(label='Number of Spikes')
plt.xlabel('Max Sodium Conductance (nS)')
plt.ylabel('Leak Reversal Potential (mV)')
plt.title('Spiking Behavior Depending on g_Na and E_rev (leak)')
plt.grid(False)
plt.show()

plt.figure(figsize=(10,8))
im = plt.imshow(classification_map, origin='lower', aspect='auto',
                extent=[gna_values[0], gna_values[-1], erev_values[0], erev_values[-1]],
                cmap='Set2', vmin=0, vmax=2)

cbar = plt.colorbar(ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['Silent', 'Phasic', 'Tonic'])

plt.xlabel('Max Sodium Conductance (nS)')
plt.ylabel('Max IH Conductance (nS)')
plt.title('Classification of Neuron Firing Behavior')
plt.grid(False)
plt.show()