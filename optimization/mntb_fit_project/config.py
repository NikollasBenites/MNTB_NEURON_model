# config.py

# ----------------------------------
# File paths
# ----------------------------------
full_path_to_file = r"/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/MNTB_Model_Dann/10142022_P9_FVB_PunTeTx.dat"

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

gleak = 12  # nS
gh = 18.8  # nS

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
# # ----------------------------------
# bounds = [
#     (1, 1500),
#     (1, 1500),
#     (1, 50),
#     (1, 200),
#     (0.01, 0.1),
#     (1, 200),
#     (-0.1, -0.01),
#     (1e-5, 0.01),
#     (-0.15, -0.05),
#     (0.1, 5),
#     (0.02, 0.1),
#     (0.1, 0.3),
#     (0.01, 0.04),
#     (0.1, 0.3),
#     (0.0, 0.5),
#     (0.005, 0.008),
#     (-0.3, 0.1),
#     (0.07, 0.1),
#     (0.004, 0.007),
#     (0.001,5),
#     (0.001,5),
#     (0.001, 5),
#     (0.001, 5),
#     (0.03,0.6)
# ]
bounds = [
    (1, 1500),    # gna (axon Na conductance)
    (1, 1500),    # gkht (soma KHT conductance)
    (1, 50),      # gklt (axon KLT conductance)
    (1, 200),     # cam (Na activation)
    (0.01, 0.1),  # kam (Na activation slope)
    (1, 200),     # cbm (Na activation offset)
    (-0.1, -0.01),# kbm (Na activation slope 2)
    (1e-5, 0.01), # cah (Na inactivation)
    (-0.15, -0.05),# kah (Na inactivation slope)
    (0.1, 5),     # cbh (Na inactivation offset)
    (0.02, 0.1),  # kbh (Na inactivation slope 2)
    (0.1, 0.3),   # can (KHT activation)
    (0.01, 0.04), # kan (KHT activation slope)
    (0.1, 0.3),   # cbn (KHT activation offset)
    (0.0, 0.5),   # kbn (KHT activation slope 2)
    (0.005, 0.008), # cap (KHT inactivation)
    (-0.3, 0.1),  # kap (KHT inactivation slope)
    (0.07, 0.1),  # cbp (KHT inactivation offset)
    (0.004, 0.007), # kbp (KHT inactivation slope 2)
    (0.001, 5),   # na_scale (axon Na scale factor)
    (0.001, 5),   # kht_scale (soma KHT scale factor)
    (0.001, 5),   # klt_scale (axon KLT scale factor)
    (0.001, 5),   # ih_scale (IH channel scale factor)
    (0.03, 0.6)   # stim_amp (current injection amplitude, in nA)
]


maxiter_global = 20
popsize_global = 8
maxiter_local = 100

# ----------------------------------
# Plotting and saving
# ----------------------------------
show_plots = True
save_figures = True
save_fit_results = True
fit_results_filename = "fit_results.csv"
