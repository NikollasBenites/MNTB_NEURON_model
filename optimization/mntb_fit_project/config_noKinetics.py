# config.py

# ----------------------------------
# File paths
# ----------------------------------
full_path_to_file = r"/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/optimization/mntb_fit_project/10142022_P9_FVB_PunTeTx.dat"

group_idx = 1
series_idx = 3
channel_idx = 0

output_dir = "fit_outputs"

# ----------------------------------
# NEURON Model parameters
# ----------------------------------
celsius = 35
v_init = -77
ena = 62.77
ek = -106.8
erev = -79

#gleak = 12  # nS
#gh = 18.8  # nS

total_capacitance_pF = 25

# ----------------------------------
# Simulation parameters
# ----------------------------------
#stim_amp_nA = 0.4
stim_dur_ms = 300
stim_delay_ms = 10
h_dt = 0.02
steps_per_ms = (1/h_dt)

# ----------------------------------
# Optimization parameters

bounds = [
    (400, 2000),    # gna (axon Na conductance)
    (400, 2000),    # gkht (soma KHT conductance)
    (1, 200),      # gklt (axon KLT conductance)
    (9, 15),      #gh (soma and dendrites)
    (1,20),          #gleak (all)
    (1, 5),   # na_scale (axon Na scale factor)
    (1, 5),   # kht_scale (soma KHT scale factor)
    (1, 5),   # klt_scale (axon KLT scale factor)
    (1, 5),   # ih_soma (IH channel scale factor)
    (0.001, 1),   # ih_dend (IH channel scale factor)
    (0.3, 0.45)   # stim_amp (current injection amplitude, in nA)
]

# Global search (Differential Evolution)
maxiter_global = 30
popsize_global = 15
tol_global = 0.01  # relative tolerance for convergence of DE (0.01 = 1%)

# Local search (L-BFGS-B)
maxiter_local = 200
ftol_local = 1e-6   # tolerance for function value changes
gtol_local = 1e-5   # tolerance for gradient norm (optional)

# ----------------------------------
# Plotting and saving
# ----------------------------------
show_plots = True
save_figures = True
save_fit_results = True
fit_results_filename = "fit_results.csv"
