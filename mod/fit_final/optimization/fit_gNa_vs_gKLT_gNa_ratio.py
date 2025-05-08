import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neuron import h
from scipy.signal.windows import blackman
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import MNTB_PN_myFunctions as mFun
from MNTB_PN_fit import MNTB

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
        'gna': params_row["gna"],
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
gna_fixed = fixed_params['gna']
print(f"gNa fixed: {gna_fixed}")
gklt_fixed = fixed_params['gklt']
print(f"gKLT fixed: {gklt_fixed}")
ratio_fixed = gklt_fixed / gna_fixed if gna_fixed != 0 else 0.0
# === Define ranges ===
gna_values = np.linspace(50, 300, 50)        # Sodium conductance in nS
ratios = np.linspace(0.0, 0.1, 50)            # gNa/gKLT ratios

spike_matrix = np.zeros((len(ratios), len(gna_values)))

# === Simulation parameters ===
stim_start = 10      # ms
stim_end = 310       # ms
stim_amp = 0.2       # nA
threshold = -15       # mV for spike detection

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
           cmap='viridis', vmin=0, vmax=5)
# === Plot red dot on 2D heatmap ===
plt.scatter(gna_fixed, ratio_fixed, color='red', s=80,
            edgecolor='black', linewidth=1.2, label='Fixed Params')
plt.legend(loc='upper right')
plt.colorbar(label='Number of Spikes')
plt.xlabel('gNa (nS)')
plt.ylabel('gNa / gKLT Ratio')
plt.title('Spike Count vs gNa and gNa/gKLT Ratio')
plt.grid(False)
plt.tight_layout()
plt.show()

# Create meshgrid for gNa and ratios
GNA, RATIO = np.meshgrid(gna_values, ratios)

# 3D plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Normalize color range for colormap
norm = plt.Normalize(vmin=0, vmax=5)
colors = plt.cm.viridis(norm(spike_matrix))

# Plot surface
surf = ax.plot_surface(GNA, RATIO, spike_matrix,
                       facecolors=colors, rstride=1, cstride=1,alpha=0.5,
                       linewidth=0.5, edgecolor='black', antialiased=True)

# Add color bar
mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
mappable.set_array(spike_matrix)
# fig.colorbar(mappable, ax=ax, label='Number of Spikes')


# Labels
ax.set_xlabel('gNa (nS)')
ax.set_ylabel('gKLT / gNa Ratio')
ax.set_zlabel('Spike Count')
ax.set_title('3D Surface of Spike Count vs gNa and gKLT/gNa Ratio')

ax.view_init(elev=30, azim=150,roll=3)

# === Simulate the fixed point directly ===


# Rebuild clean fixed parameter dictionary
fixed_sim_params = fixed_params.copy()
fixed_sim_params['gna'] = gna_fixed
fixed_sim_params['gklt'] = gklt_fixed

neuron_fixed = MNTB(**fixed_sim_params)

stim = h.IClamp(neuron_fixed.soma(0.5))
stim.delay = stim_start
stim.dur = stim_end - stim_start
stim.amp = stim_amp

v_fix = h.Vector().record(neuron_fixed.soma(0.5)._ref_v)
t_fix = h.Vector().record(h._ref_t)

mFun.custom_init(-70)
h.continuerun(510)

v_np_fix = np.array(v_fix)
t_np_fix = np.array(t_fix)

spike_indices_fix = np.where((v_np_fix[:-1] < threshold) & (v_np_fix[1:] >= threshold))[0]
spike_times_fix = t_np_fix[spike_indices_fix]
valid_spikes_fix = np.logical_and(spike_times_fix >= stim_start, spike_times_fix <= stim_end)
spike_fixed = np.sum(valid_spikes_fix)
print(f"gNa fixed: {gna_fixed}")
# === Find closest point on mesh ===
i_closest = (np.abs(ratios - ratio_fixed)).argmin()
j_closest = (np.abs(gna_values - gna_fixed)).argmin()

gna_closest = gna_values[j_closest]
ratio_closest = ratios[i_closest]
spike_closest = spike_matrix[i_closest, j_closest]


from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# === Highlight 3D prism at mesh point ===
dx = gna_values[1] - gna_values[0]
dy = ratios[1] - ratios[0]
x0, x1 = gna_closest - dx / 2, gna_closest + dx / 2
y0, y1 = ratio_closest - dy / 2, ratio_closest + dy / 2
z_base = 0
z_top = spike_closest+5

verts = [
    [(x0, y0, z_base), (x1, y0, z_base), (x1, y1, z_base), (x0, y1, z_base)],
    [(x0, y0, z_top), (x1, y0, z_top), (x1, y1, z_top), (x0, y1, z_top)],
    [(x0, y0, z_base), (x1, y0, z_base), (x1, y0, z_top), (x0, y0, z_top)],
    [(x1, y0, z_base), (x1, y1, z_base), (x1, y1, z_top), (x1, y0, z_top)],
    [(x1, y1, z_base), (x0, y1, z_base), (x0, y1, z_top), (x1, y1, z_top)],
    [(x0, y1, z_base), (x0, y0, z_base), (x0, y0, z_top), (x0, y1, z_top)],
]

prism = Poly3DCollection(verts, facecolor='crimson', edgecolor='black', alpha=0.95, linewidth=0.7)
prism.set_label('Fixed Params (3D Box)')
ax.add_collection3d(prism)

print(f"Tile location: gna={gna_closest}, ratio={ratio_closest}, spike_count={spike_closest}")


ax.text(gna_closest, ratio_closest, z_top + 1,
        f"{int(spike_closest)} spikes",
        fontsize=10, color='black', ha='center', va='bottom')

ax.plot([gna_closest, gna_closest],
        [ratio_closest, ratio_closest],
        [spike_closest, z_top],
        color='gray', linestyle='--', linewidth=1)
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.figure()
plt.plot(t_np_fix, v_np_fix, label="Fixed Params Trace")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Voltage (mV)")
plt.title("Voltage Trace at Fixed Params")
plt.legend()
plt.tight_layout()
plt.show()