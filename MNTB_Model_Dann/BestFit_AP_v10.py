import numpy as np

np.random.seed(1)
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize, differential_evolution
from neuron import h
import MNTB_PN_myFunctions as mFun
from load_heka_python.load_heka import LoadHeka
from functools import lru_cache

h.load_file('stdrun.hoc')

full_path_to_file = r"/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/MNTB_Model_Dann/10142022_P9_FVB_PunTeTx.dat"

with LoadHeka(full_path_to_file) as hf:
    hf.print_group_names()
    hf.print_series_names(group_idx=1)

    # Load series data
    series = hf.get_series_data(group_idx=1, series_idx=3, channel_idx=0, include_stim_protocol=True)
    voltage = series['data']
    time = series['time']
    stim = series.get('stim', None)
    # Ensure stim is in list-of-arrays format
    if stim is not None:
        if isinstance(stim, np.ndarray):
            if stim.ndim == 1:
                # Single sweep, wrap into a list
                stim = [stim]
            elif stim.ndim == 2:
                # Already multiple sweeps: convert to list of arrays
                stim = [s for s in stim]
            else:
                raise ValueError("Unexpected stim shape:", stim.shape)
        else:
            # Catch any other types
            stim = list(stim)
    labels = series.get('labels', None)

    n_sweeps = len(voltage)

    # Check if labels is a valid list or array
    try:
        label_list = list(labels)
        if len(label_list) != n_sweeps:
            raise ValueError
    except:
        label_list = [None] * n_sweeps

    # Plot all sweeps
    plt.figure(figsize=(12, 6))
    for i in range(n_sweeps):
        stim_label = f"{label_list[i]} pA" if label_list[i] is not None else f"Sweep {i}"
        plt.plot(time[i] * 1000, voltage[i], label=f"Sweep {i}")

    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.title("HEKA Sweeps - Inspect Before Fitting")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
   # plt.show(block=False)

    # Print sweep info
    print("\nAvailable sweeps:")
    for i in range(n_sweeps):
        stim_label = f"{label_list[i]} pA" if label_list[i] is not None else "unknown"
        print(f"  Sweep {i:2d} → {stim_label}")

    # User selects sweep
    sweep_idx = int(input(f"\nSelect sweep index (0 to {n_sweeps - 1}): "))
    v_exp_heka = voltage[sweep_idx]
    t_exp_heka = time[sweep_idx] * 1000


    print(f"\nSelected Sweep {sweep_idx}: Length = {len(v_exp_heka)} samples")

    # Plot selected sweep
    plt.figure()
    plt.plot(t_exp_heka, v_exp_heka, color='black')
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    stim_label = f"{label_list[sweep_idx]} pA" if label_list[sweep_idx] is not None else "unknown"
    plt.title(f"Sweep {sweep_idx} – {stim_label}")
    plt.tight_layout()
    plt.grid(True)
    #plt.show(block=False)

# Load experimental data
# experimental_data = np.genfromtxt('P9_iMNTB_Rheobase_raw.csv', delimiter=',', skip_header=1, dtype=float, filling_values=np.nan)
# t_exp = experimental_data[:,0]*1000  # ms
# t_exp = t_exp - t_exp[0]
# V_exp = experimental_data[:,2]  # mV
V_exp = v_exp_heka*1000 #convert to mV
t_exp = t_exp_heka

h.celsius = 35

#Create a dendrite
dend = h.Section(name='dend')
dend.diam = 3
dend.L = 80
dend.Ra = 100
dend.cm = 1
dend.insert('leak')
dend.insert('IH_dth')

# Create soma section
soma = h.Section(name='soma')
#soma.L = 15  # µm
soma.diam = 15.5  # µm
soma.Ra = 150
soma.cm = 1
v_init = -77
soma.v = v_init
soma.insert('leak')
#soma.insert('LT')
soma.insert('IH_dth')
soma.insert('HT_dth_nmb')
#soma.insert('NaCh')
soma.ek = -106.8
#soma.ena = 62.77
ena = 62.77

# Create axon section
axon = h.Section(name='axon')
axon.L = 25
axon.diam = 3
axon.Ra = 100
axon.cm = 1
axon.nseg = 5
axon.insert('leak')
axon.insert('NaCh_nmb')
#axon.insert('HT')
axon.insert('LT_dth')
#axon.insert('IH')
axon.ek = soma.ek
axon.ena = ena

erev = -79
gleak = 12
# gklt = 161.1
gh = 18.8

axon.connect(soma(1))
dend.connect(soma(0))

totalcap = 25  # Total membrane capacitance in pF
somaarea = (totalcap * 1e-6) / 1  # Convert to cm^2 assuming 1 µF/cm²
axonarea = np.pi * axon.diam * axon.L * 1e-8  # in cm²
def nstomho(x):
    return (1e-9 * x / somaarea)  # Convert conductance to mho/cm²
def nstomho_axon(x):
    return (1e-9 * x / axonarea)

def set_conductances(gna, gkht, gklt, gh, erev, gleak,
                     cam, kam, cbm, kbm,
                     cah, kah, cbh, kbh,
                     can, kan, cbn, kbn,
                     cap, kap, cbp, kbp,
                     axon_scale=2):
    soma.gkhtbar_HT_dth_nmb = nstomho(gkht)
    soma.can_HT_dth_nmb = can
    soma.kan_HT_dth_nmb = kan
    soma.cbn_HT_dth_nmb = cbn
    soma.kbn_HT_dth_nmb = kbn
    soma.cap_HT_dth_nmb = cap
    soma.kap_HT_dth_nmb = kap
    soma.cbp_HT_dth_nmb = cbp
    soma.kbp_HT_dth_nmb = kbp
    soma.ghbar_IH_dth = nstomho(gh) * 0.08
    soma.erev_leak = erev
    soma.g_leak = nstomho(gleak) * 0.2

    axon.gnabar_NaCh_nmb = nstomho_axon(gna)*axon_scale
    axon.cam_NaCh_nmb = cam
    axon.kam_NaCh_nmb = kam
    axon.cbm_NaCh_nmb = cbm
    axon.kbm_NaCh_nmb = kbm
    axon.cah_NaCh_nmb = cah
    axon.kah_NaCh_nmb = kah
    axon.cbh_NaCh_nmb = cbh
    axon.kbh_NaCh_nmb = kbh
    axon.gkltbar_LT_dth = nstomho_axon(gklt)
    axon.erev_leak = erev
    axon.g_leak = nstomho_axon(gleak) * 0.12

    for seg in dend:
        seg.g_leak = nstomho(gleak) * 0.5
        seg.erev_leak = erev
        seg.ghbar_IH_dth = nstomho(gh) * 0.16

def plot_inf_curves_ab(cam, kam, cbm, kbm, cah, kah, cbh, kbh, v_range=(-100, 60)):
    v = np.linspace(*v_range, 300)
    am = cam * np.exp(kam * v)
    bm = cbm * np.exp(kbm * v)
    ah = cah * np.exp(kah * v)
    bh = cbh * np.exp(kbh * v)

    minf = am / (am + bm)
    hinf = ah / (ah + bh)

    plt.figure()
    plt.plot(v, minf, label='m∞')
    plt.plot(v, hinf, label='h∞')
    plt.xlabel('Membrane potential (mV)')
    plt.ylabel('Activation / Inactivation')
    plt.title('Steady-State Na⁺ Gating (Fitted)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
  #  plt.show(block=False)


def extract_features(trace, time):
    dt = time[1] - time[0]
    dV = np.gradient(trace, dt)
    rest = trace[:int(5/dt)].mean()  # average first 5 ms
    peak_idx = np.argmax(trace)
    peak = trace[peak_idx]

    # Only consider dV/dt after 10 ms for threshold detection
    search_start_time = 11  # ms
    search_start_idx = np.searchsorted(time, search_start_time)
    dV_slice = dV[search_start_idx:]
    trace_slice = trace[search_start_idx:]
    time_slice = time[search_start_idx:]

    try:
        relative_thresh_idx = np.where(dV_slice > 45)[0][0]
        thresh_idx = search_start_idx + relative_thresh_idx
        threshold = trace[thresh_idx]
        latency = time[thresh_idx]
    except IndexError:
        threshold = np.nan
        latency = np.nan

    amp = peak - threshold

    # Half width
    half_amp = threshold + 0.5 * amp
    above_half = np.where(trace > half_amp)[0]
    if len(above_half) > 2:
        width = (above_half[-1] - above_half[0]) * dt
    else:
        width = np.nan

    # AHP (min value after peak)
    AHP = np.min(trace[peak_idx:]) if peak_idx < len(trace) else np.nan

    return {
        'rest': rest,
        'peak': peak,
        'amp': amp,
        'threshold': threshold,
        'latency': latency,
        'width': width,
        'AHP': AHP
    }

def feature_cost(sim_trace, exp_trace, time):
    sim_feat = extract_features(sim_trace, time)
    exp_feat = extract_features(exp_trace, time)
    weights = {
        'peak': 1,  # Increase penalty on overshoot
        'amp': 1,
        'width': 1,
        'threshold': 1,  # Strong push toward threshold match
        'latency':1,
        'AHP': 1
    }
    error = 0
    for k in weights:
        if not np.isnan(sim_feat[k]) and not np.isnan(exp_feat[k]):
            error += weights[k] * ((sim_feat[k] - exp_feat[k]) ** 2)
    return error

@lru_cache(maxsize=2048)
def run_simulation(gna, gkht, gklt,
                   cam, kam, cbm, kbm,
                   cah, kah, cbh, kbh,
                   can, kan, cbn, kbn,
                   cap, kap, cbp, kbp,
                   stim_amp=0.320, stim_dur=300):
    set_conductances(gna, gkht, gklt, gh, erev, gleak,
                     cam, kam, cbm, kbm,
                     cah, kah, cbh, kbh,
                     can, kan, cbn, kbn,
                     cap, kap, cbp, kbp)


    stim = h.IClamp(soma(0.5))
    stim.delay = 10
    stim.dur = stim_dur
    stim.amp = stim_amp

    h.dt = 0.02
    h.steps_per_ms = int(1.0 / h.dt)
    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(soma(0.5)._ref_v)
    v_vec_axon = h.Vector().record(axon(0.5)._ref_v)

    h.v_init = v_init
    mFun.custom_init(v_init)
    h.tstop = stim.delay + stim_dur
    h.continuerun(510)

    return np.array(t_vec), np.array(v_vec), np.array(v_vec_axon)

def monitor_cache_size():
    cache_info = run_simulation.cache_info()
    print(f"Cache size: {cache_info.currsize}/{cache_info.maxsize}")
    print(f"Hit ratio: {cache_info.hits/(cache_info.hits + cache_info.misses):.2%}")

def interpolate_simulation(t_neuron, v_neuron, t_exp):
    interp_func = interp1d(t_neuron, v_neuron, kind='cubic', fill_value='extrapolate')
    return interp_func(t_exp)

def penalty_terms(v_sim):
    peak = np.max(v_sim)
    rest = v_sim[0]
    penalty = 0
    if peak < -10 or peak > 10:
        penalty += 100
    if rest > -40 or rest < -80:
        penalty += 1000
    return penalty


def cost_function(params):
    gna, gkht, gklt, cam, kam, cbm, kbm, cah, kah, cbh, kbh, can, kan, cbn, kbn, cap, kap, cbp, kbp = params

    t_sim, v_sim, _ = run_simulation(
        gna, gkht, gklt,
        cam, kam, cbm, kbm,
        cah, kah, cbh, kbh,
        can, kan, cbn, kbn,
        cap, kap, cbp, kbp)


    v_interp = interpolate_simulation(t_sim, v_sim, t_exp)

    features_exp = extract_features(V_exp, t_exp)
    ap_tmin = features_exp['latency']
    ap_tmax = t_exp[-1]
    if not np.isnan(features_exp['AHP']):
        try:
            ahp_idx = np.where(V_exp == features_exp['AHP'])[0][0]
            ap_tmax = t_exp[ahp_idx]
        except IndexError:
            pass
    if np.isnan(ap_tmin) or ap_tmin >= ap_tmax:
        return 1e6

    ap_mask = (t_exp >= ap_tmin) & (t_exp <= ap_tmax)
    if np.sum(ap_mask) < 2:
        return 1e6


    t_ap = t_exp[ap_mask]
    V_ap_exp = V_exp[ap_mask]
    v_ap_interp = v_interp[ap_mask]

    max_dvdt_exp = max_dvdt(V_ap_exp, t_ap)
    max_dvdt_sim = max_dvdt(v_ap_interp, t_ap)
    dvdt_error = 3.0 * (max_dvdt_sim - max_dvdt_exp) ** 2

    dt = t_ap[1] - t_ap[0]
    time_shift = abs(np.argmax(v_ap_interp) - np.argmax(V_ap_exp)) * dt
    time_error = 5.0 * time_shift

    rmse = np.sqrt(np.mean((v_ap_interp - V_ap_exp)**2))
    f_cost = feature_cost(v_ap_interp, V_ap_exp, t_ap)
    penalty = penalty_terms(v_ap_interp)

    alpha = 5
    beta = 0.5

    total_cost = alpha * rmse + beta * f_cost + time_error + dvdt_error + penalty
    return total_cost


def max_dvdt(trace, time):
    dt = time[1] - time[0]
    dVdt = np.gradient(trace, dt)
    return np.max(dVdt)

# t_exp = experimental_data[499:,0]*1000 # in ms, sampled at 50 kHz
# t_exp = t_exp - t_exp[0]  # ensure starts at 0
# V_exp = experimental_data[499:,1]  # in mV
# print(soma.psection())  # or axon.psection(), dend.psection()

# Initial guess and bounds
bounds = [
    (200, 1500),      # gNa
    (200,2500),        # gKHT
    (1,50),           #gKLT
    # (gklt * 0.5, gklt * 1.5),  # gKLT

    (1, 200),        # cam
    (0.01, 0.1),      # kam
    (1, 200),         # cbm
    (-0.1, -0.01),    # kbm

    (1e-5, 0.01),     # cah
    (-0.15, -0.05),   # kah
    (0.1, 5),         # cbh
    (0.02, 0.1),       # kbh

    (0.1, 0.3), #can
	 (0.01,0.04), #kan
	 (0.1,0.3), #cbn
	 (0,0.5), #kbn

	 (0.005,0.008), #cap
	 (-0.3,0.1), #kap
	 (0.07,0.1), #cbp
	 (0.004,0.007) #kbp
]


# result = differential_evolution(cost_function, bounds, strategy='best1bin',
#                                 maxiter=20, popsize=10, polish=True)

# x0 = [350, 350, gklt,gh,erev]  # gNa, gKHT, gKLT, gH
# bounds = [(1e-4, 700), (1e-4, 700),(gklt,gklt),(gh,gh),(erev,erev)]
# result = minimize(cost_function, x0, bounds=bounds, method='L-BFGS-B', options={'maxiter': 200})

result_global = differential_evolution(cost_function, bounds, strategy='best1bin', maxiter=20, popsize=8,polish=False)
result_local = minimize(cost_function, result_global.x, bounds=bounds, method='L-BFGS-B', options={'maxiter': 100})

print(result_local.x)

params_opt = result_local.x
(gna_opt, gkht_opt, gklt_opt,
 cam_opt, kam_opt, cbm_opt, kbm_opt,
 cah_opt, kah_opt, cbh_opt, kbh_opt,
 can_opt, kan_opt, cbn_opt, kbn_opt,
 cap_opt, kap_opt, cbp_opt, kbp_opt) = params_opt
print(f" Optimized gna: {gna_opt:.2f}, gklt: {gklt_opt: .2f}, gkht: {gkht_opt: .2f})")
print(f" Optimized cam: {cam_opt:.2f}, kam: {kam_opt:.3f}, cbm: {cbm_opt:.2f}, kbm: {kbm_opt:.3f}")
print(f" Optimized cah: {cah_opt:.5f}, kah: {kah_opt:.4f}, cbh: {cbh_opt:.2f}, kbh: {kbh_opt:.3f}")


# Final simulation and plot
t_sim, v_sim, v_axon = run_simulation(
    gna_opt, gkht_opt, gklt_opt,
    cam_opt, kam_opt, cbm_opt, kbm_opt,
    cah_opt, kah_opt, cbh_opt, kbh_opt,
    can_opt, kan_opt, cbn_opt, kbn_opt,
    cap_opt, kap_opt, cbp_opt, kbp_opt
)

feat_sim = extract_features(v_sim, t_sim)
print("Simulate Features:")
for k, v in feat_sim.items():
    print(f"{k}: {v:.2f}")

feat_exp = extract_features(V_exp, t_exp)
print("Experimental Features:")
for k, v in feat_exp.items():
    print(f"{k}: {v:.2f}")

lat_soma = extract_features(v_sim, t_sim)['latency']
lat_axon = extract_features(v_axon, t_sim)['latency']
print(f"AIS leads soma by {lat_soma - lat_axon:.3f} ms")

results = {
    "gna_opt": f"{gna_opt:.2f}",
    "gkht_opt": f"{gkht_opt:.2f}",
    "gklt_opt": f"{gklt_opt:.2f}",
    "cam": f"{cam_opt:.2f}", "kam": f"{kam_opt:.3f}", "cbm": f"{cbm_opt:.2f}", "kbm": f"{kbm_opt:.3f}",
    "cah": f"{cah_opt:.5f}", "kah": f"{kah_opt:.4f}", "cbh": f"{cbh_opt:.2f}", "kbh": f"{kbh_opt:.3f}",
    # "latency_soma": f"{lat_soma:.2f}",
    # "latency_axon": f"{lat_axon:.2f}",
    # "AIS_lead_ms": lat_soma - lat_axon,
    **feat_sim
}

pd.DataFrame([results]).to_csv("fit_results.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(t_exp, V_exp, label='Experimental', linewidth=2)
plt.plot(t_sim, v_sim, label='Simulated (fit)', linestyle='--')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.title('Action Potential Fit')
thresh_exp = extract_features(V_exp, t_exp)['latency']
thresh_sim = extract_features(v_sim, t_sim)['latency']
plt.axvline(thresh_exp, color='blue', linestyle=':', label='Exp Threshold')
plt.axvline(thresh_sim, color='orange', linestyle=':', label='Sim Threshold')
# plt.plot(t_sim, v_axon, label='Axon (AIS)', linestyle=':')
plt.tight_layout()

def plot_dvdt(trace, time, label):
    dt = time[1] - time[0]
    dVdt = np.gradient(trace, dt)
    plt.plot(trace, dVdt, label=label)

plot_inf_curves_ab(cam_opt, kam_opt, cbm_opt, kbm_opt, cah_opt, kah_opt, cbh_opt, kbh_opt)

# param_dict = {
#     "gna": gna, "gkht": gkht, "gklt": gklt,
#     "cam": cam, "kam": kam, "cbm": cbm, "kbm": kbm,
#     "cah": cah, "kah": kah, "cbh": cbh, "kbh": kbh
# }
# pd.DataFrame([param_dict]).to_csv("fit_params.csv", index=False)
monitor_cache_size()
plt.figure()
plot_dvdt(V_exp, t_exp, 'Experimental')
plot_dvdt(v_sim, t_sim, 'Simulated')
# plot_dvdt(v_axon, t_sim, 'AIS')
plt.xlabel('Membrane potential (mV)')
plt.ylabel('dV/dt (mV/ms)')
plt.title('Phase Plane Plot')
plt.legend()
plt.grid()

plt.legend()
plt.show()


