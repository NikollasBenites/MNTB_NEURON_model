from neuron import h
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from MNTB_PN_fit import MNTB  # Your class
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
        'gklt': params_row["gklt"],
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

gna_values = np.linspace(200, 600, 10)  # 1 nS to 100 nS, adjust as needed
spike_counts = []
v_init = -70
for gna in gna_values:
    fixed_params['gna'] = gna
    neuron = MNTB(**fixed_params)

    stim = h.IClamp(neuron.soma(0.5))
    stim.delay = 10
    stim.dur = 300
    stim.amp = 0.3

    v = h.Vector().record(neuron.soma(0.5)._ref_v)
    t = h.Vector().record(h._ref_t)

    mFun.custom_init(v_init)
    h.continuerun(510)

    v_np = np.array(v)
    t_np = np.array(t)

    spike_indices = np.where((v_np[:-1] < 0) & (v_np[1:] >= 0))[0]
    spike_times = t_np[spike_indices]

    stim_start = 10
    stim_end = 310
    valid_spikes = np.logical_and(spike_times >= stim_start, spike_times <= stim_end)

    spike_counts.append(np.sum(valid_spikes))

# Plot the results
plt.figure(figsize=(8,6))
plt.plot(gna_values, spike_counts, 'o-')
plt.xlabel('Max Sodium Conductance (nS)')
plt.ylabel('Number of Spikes')
plt.title('Firing Behavior vs Sodium Conductance (g_Na)')
plt.grid(True)
plt.show()