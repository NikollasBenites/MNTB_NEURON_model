import numpy as np
np.random.seed(1)
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize, differential_evolution
from neuron import h
import MNTB_PN_myFunctions as mFun
h.load_file('stdrun.hoc')

# Load experimental data
experimentalTrace = np.genfromtxt('../P9_iMNTB_Rheobase_raw.csv', delimiter=',', skip_header=1, dtype=float, filling_values=np.nan)
t_exp = experimentalTrace[:,0]*1000  # ms
t_exp = t_exp - t_exp[0]
V_exp = experimentalTrace[:,2]  # mV
h.celsius = 35
# Create soma section
soma = h.Section(name='soma')
soma.L = 15  # µm
soma.diam = 20  # µm
soma.Ra = 150
soma.cm = 1
v_init = -77
soma.insert('leak')
soma.insert('LT')
soma.insert('IH')
soma.insert('HT')
soma.insert('NaCh')
soma.ek = -106.8
soma.ena = 62.77

# Create axon section
axon = h.Section(name='axon')
axon.L = 15
axon.diam = 1
axon.Ra = 150
axon.cm = 1
axon.nseg = 5
axon.insert('leak')
axon.insert('NaCh')
axon.insert('HT')
#axon.insert('LT')
#axon.insert('IH')
axon.ek = soma.ek
axon.ena = soma.ena

erev = -79
gleak = 12
gklt = 50
gh = 18.8

axon.connect(soma(1))

totalcap = 20  # Total membrane capacitance in pF
somaarea = (totalcap * 1e-6) / 1  # Convert to cm^2 assuming 1 µF/cm²
axonarea = np.pi * axon.diam * axon.L * 1e-8  # in cm²
def nstomho(x):
    return (1e-9 * x / somaarea)  # Convert conductance to mho/cm²
def nstomho_axon(x):
    return (1e-9 * x / axonarea)

def set_conductances(gna, gkht, gklt, gh, erev, gleak, axon_scale = 1.2):
    soma.gnabar_NaCh_nmb = nstomho(gna) * 0.01
    soma.gkhtbar_HT = nstomho(gkht)*0.5
    soma.gkltbar_LT = nstomho(gklt)
    soma.ghbar_IH = nstomho(gh)
    soma.erev_leak = erev
    soma.g_leak = nstomho(gleak)

    axon.gnabar_NaCh_nmb = nstomho_axon(gna) * axon_scale # ~5x soma
    axon.gkhtbar_HT = nstomho_axon(gkht) * 1.5
    #axon.gkltbar_LT = nstomho_axon(gklt)*0.00001
   #axon.ghbar_IH = nstomho_axon(gh)*0.000001
    axon.erev_leak = erev
    axon.g_leak = nstomho_axon(gleak)

def extract_features(trace, time):
    dt = time[1] - time[0]
    dV = np.gradient(trace, dt)
    rest = np.mean(trace[:int(5/dt)])  # average first 5 ms
    peak_idx = np.argmax(trace)
    peak = trace[peak_idx]

    # Threshold = first time where dV/dt > 10 mV/ms
    try:
        thresh_idx = np.where(dV > 40)[0][0]
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
        'peak': 10,  # Increase penalty on overshoot
        'amp': 1,
        'width': 1,
        'threshold': 1,  # Strong push toward threshold match
        'latency':10,
        'AHP': 1
    }
    error = 0
    for k in weights:
        if not np.isnan(sim_feat[k]) and not np.isnan(exp_feat[k]):
            error += weights[k] * ((sim_feat[k] - exp_feat[k]) ** 2)
    return error

def run_simulation(gna, gkht, gklt, stim_amp=0.6, stim_dur=10):
    set_conductances(gna, gkht, gklt, gh, erev, gleak)

    stim = h.IClamp(soma(0.5))
    stim.delay = 10
    stim.dur = stim_dur
    stim.amp = stim_amp

    h.dt = 0.02
    h.steps_per_ms = int(1.0 / h.dt)
    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(soma(0.5)._ref_v)
    v_vec_axon = h.Vector().record(axon(0.5)._ref_v)  # new

    h.v_init = v_init
    mFun.custom_init(v_init)
    h.continuerun(stim.delay + stim_dur)

    return np.array(t_vec), np.array(v_vec), np.array(v_vec_axon)

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
    gna, gkht, gklt = params
    t_sim, v_sim, _ = run_simulation(gna, gkht, gklt)
    v_interp = interpolate_simulation(t_sim, v_sim, t_exp)

    # === Extract dynamic AP window based on experimental trace
    features_exp = extract_features(V_exp, t_exp)
    ap_tmin = features_exp['latency']
    ap_tmax = t_exp[-1]

    if not np.isnan(features_exp['AHP']):
        try:
            ahp_idx = np.where(V_exp == features_exp['AHP'])[0][0]
            ap_tmax = t_exp[ahp_idx]
        except IndexError:
            pass

    # === Create mask for AP time window
    ap_mask = (t_exp >= ap_tmin) & (t_exp <= ap_tmax)
    t_ap = t_exp[ap_mask]
    V_ap_exp = V_exp[ap_mask]
    v_ap_interp = v_interp[ap_mask]

    # === 📈 Rate-of-rise term (after AP window is defined!)
    max_dvdt_exp = max_dvdt(V_ap_exp, t_ap)
    max_dvdt_sim = max_dvdt(v_ap_interp, t_ap)
    dvdt_error = 3.0 * (max_dvdt_sim - max_dvdt_exp) ** 2

    # === Cost calculations
    dt = t_ap[1] - t_ap[0]
    time_shift = abs(np.argmax(v_ap_interp) - np.argmax(V_ap_exp)) * dt
    time_error = 5.0 * time_shift

    mse = np.mean((v_ap_interp - V_ap_exp)**2)
    f_cost = feature_cost(v_ap_interp, V_ap_exp, t_ap)
    penalty = penalty_terms(v_ap_interp)

    alpha = 0.9
    beta = 0.8

    total_cost = alpha * mse + beta * f_cost + time_error + dvdt_error + penalty

    # print(f"gNa: {gna:.2f}, gKHT: {gkht:.2f}, MSE: {mse:.4f}, f_cost: {f_cost:.4f}, shift: {time_shift:.2f}, total: {total_cost:.4f}")
    return total_cost

def max_dvdt(trace, time):
    dt = time[1] - time[0]
    dVdt = np.gradient(trace, dt)
    return np.max(dVdt)

# t_exp = experimentalTrace[499:,0]*1000 # in ms, sampled at 50 kHz
# t_exp = t_exp - t_exp[0]  # ensure starts at 0
# V_exp = experimentalTrace[499:,1]  # in mV

# Initial guess and bounds
bounds = [(1, 400), (300, 400), (gklt*0.1,gklt*1.9)]
# result = differential_evolution(cost_function, bounds, strategy='best1bin',
#                                 maxiter=20, popsize=10, polish=True)

# x0 = [350, 350, gklt,gh,erev]  # gNa, gKHT, gKLT, gH
# bounds = [(1e-4, 700), (1e-4, 700),(gklt,gklt),(gh,gh),(erev,erev)]
# result = minimize(cost_function, x0, bounds=bounds, method='L-BFGS-B', options={'maxiter': 200})

result_global = differential_evolution(cost_function, bounds, strategy='best1bin', maxiter=20, popsize=10, polish=True)
result_local = minimize(cost_function, result_global.x, bounds=bounds, method='L-BFGS-B', options={'maxiter': 200})
opt_gna, opt_gkht, opt_gklt = result_local.x

# opt_gna, opt_gkht, gklt, gh, erev = result.x
print(f"Optimal gNa: {opt_gna:.2f} , Optimal gKHT: {opt_gkht:.2f}, Optimal gKLT: {opt_gklt:.2f}, Set erev: {erev:.2f}")


# Final simulation and plot
t_sim, v_sim, v_axon = run_simulation(opt_gna, opt_gkht, opt_gklt)

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
    "gna_opt": f"{opt_gna: .2f}",
    "gkht_opt": f"{opt_gkht: .2f}",
    "latency_soma": f"{lat_soma: .2f}",
    "latency_axon": f"{lat_axon: .2f}",
    "AIS_lead_ms": lat_soma - lat_axon,
    **feat_sim
}
pd.DataFrame([results]).to_csv("../fit_results.csv", index=False)

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
plt.plot(t_sim, v_axon, label='Axon (AIS)', linestyle=':')
plt.tight_layout()

def plot_dvdt(trace, time, label):
    dt = time[1] - time[0]
    dVdt = np.gradient(trace, dt)
    plt.plot(trace, dVdt, label=label)

plt.figure()
plot_dvdt(V_exp, t_exp, 'Experimental')
plot_dvdt(v_sim, t_sim, 'Simulated')
plot_dvdt(v_axon, t_sim, 'AIS')
plt.xlabel('Membrane potential (mV)')
plt.ylabel('dV/dt (mV/ms)')
plt.title('Phase Plane Plot')
plt.legend()
plt.grid()
plt.show()