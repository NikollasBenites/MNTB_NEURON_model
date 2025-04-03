import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize, differential_evolution
from neuron import h
import MNTB_PN_myFunctions as mFun
h.load_file('stdrun.hoc')

# Define soma parameters
totalcap = 25  # Total membrane capacitance in pF
somaarea = (totalcap * 1e-6) / 1  # Convert to cm^2 assuming 1 µF/cm²

def nstomho(x):
    return (1e-9 * x / somaarea)  # Convert conductance to mho/cm²

# Load experimental data
experimentalTrace = np.genfromtxt('P9_iMNTB_Rheobase_raw.csv', delimiter=',', skip_header=1, dtype=float, filling_values=np.nan)
t_exp = experimentalTrace[:,0]*1000  # ms
t_exp = t_exp - t_exp[0]
V_exp = experimentalTrace[:,2]  # mV

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
erev = -77
gklt = 161.1
gh = 18.87

def set_conductances(gna, gkht, gklt, gh, erev):
    soma.gnabar_NaCh = nstomho(gna)
    soma.gkhtbar_HT = nstomho(gkht)
    soma.gkltbar_LT = nstomho(gklt)
    soma.ghbar_IH = nstomho(gh)
    soma.erev_leak = erev

def extract_features(trace, time):
    dt = time[1] - time[0]
    dV = np.gradient(trace, dt)

    rest = np.mean(trace[:int(5/dt)])  # average first 5 ms
    peak_idx = np.argmax(trace)
    peak = trace[peak_idx]


    # Threshold = first time where dV/dt > 10 mV/ms
    try:
        thresh_idx = np.where(dV > 45)[0][0]
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
        'peak':     10.0,   # Increase penalty on overshoot
        'amp':      1.5,
        'width':    7.0,
        'threshold': 10.0,  # Strong push toward threshold match
        'latency':  1.0,
        'AHP':      1.0
    }
    error = 0
    for k in weights:
        if not np.isnan(sim_feat[k]) and not np.isnan(exp_feat[k]):
            error += weights[k] * ((sim_feat[k] - exp_feat[k]) ** 2)
    return error

def interpolate_simulation(t_neuron, v_neuron, t_exp):
    interp_func = interp1d(t_neuron, v_neuron, kind='cubic', fill_value='extrapolate')
    return interp_func(t_exp)

def run_sim_passive(gklt, gh, cm, stim_amp=0.32, stim_dur=10):
    soma.cm = cm
    soma.gnabar_NaCh = 0
    soma.gkhtbar_HT = 0
    soma.gkltbar_LT = nstomho(gklt)
    soma.ghbar_IH = nstomho(gh)
    soma.erev_leak = erev

    stim = h.IClamp(soma(0.5))
    stim.delay = 10
    stim.dur = stim_dur
    stim.amp = stim_amp

    h.dt = 0.02
    h.steps_per_ms = int(1.0 / h.dt)
    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(soma(0.5)._ref_v)

    h.v_init = v_init
    mFun.custom_init(v_init)
    h.continuerun(stim.delay + stim_dur)

    return np.array(t_vec), np.array(v_vec)

# Extract subthreshold portion from experimental trace
threshold_exp = extract_features(V_exp, t_exp)['threshold']
thresh_idx = np.where(t_exp < extract_features(V_exp, t_exp)['latency'])[0]
t_sub = t_exp[thresh_idx]
V_sub = V_exp[thresh_idx]

def cost_passive(params):
    gklt, gh, cm = params
    t_sim, v_sim = run_sim_passive(gklt, gh, cm)
    v_interp = interpolate_simulation(t_sim, v_sim, t_sub)
    return np.mean((v_interp - V_sub) ** 2)

# Bounds and optimization for passive parameters
bounds_passive = [(gklt*0.5, gklt*1.5), (gh*0.5, gh*1.5), (0.5, 2.5)]  # gKLT, gH, cm
result_passive = differential_evolution(cost_passive, bounds_passive, maxiter=20, popsize=10)
gklt, gh, soma.cm = result_passive.x

print(f"[Stage 1] Passive Fit - gKLT: {gklt:.2f}, gH: {gh:.2f}, cm: {soma.cm:.2f}")

def run_simulation(gna, gkht, gklt, gh, stim_amp=0.32, stim_dur=10):
    set_conductances(gna, gkht, gklt, gh, erev)

    stim = h.IClamp(soma(0.5))
    stim.delay = 10
    stim.dur = stim_dur
    stim.amp = stim_amp

    h.dt = 0.02
    h.steps_per_ms = int(1.0 / h.dt)
    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(soma(0.5)._ref_v)

    h.v_init = v_init
    mFun.custom_init(v_init)
    h.continuerun(stim.delay+stim_dur)

    return np.array(t_vec), np.array(v_vec)








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
    gna, gkht, gklt, gh = params
    t_sim, v_sim = run_simulation(gna, gkht, gklt, gh)
    v_interp = interpolate_simulation(t_sim, v_sim, t_exp)

    # Time shift between peaks
    dt = t_exp[1] - t_exp[0]
    time_shift = abs(np.argmax(v_interp) - np.argmax(V_exp)) * dt
    weight = 5.0  # you can tune this weight
    time_error = weight * time_shift

    mse = np.mean((v_interp - V_exp)**2)
    f_cost = feature_cost(v_interp, V_exp, t_exp)
    penalty = penalty_terms(v_interp)
    peak_penalty = 0
    sim_peak = np.max(v_interp)
    if sim_peak > 5:
        peak_penalty += 10 * (sim_peak - 20)**2


    alpha = 0.9  # weight for MSE
    beta = 0.8  # weight for feature cost

    total_cost = alpha * mse + beta * f_cost + time_error + penalty + peak_penalty

    # print(
    #     f"gNa: {gna:.2f}, gKHT: {gkht:.2f}, MSE: {mse:.4f}, f_cost: {f_cost:.4f}, shift: {time_shift:.2f}, total: {total_cost:.4f}")

    return total_cost
# --- Stage 2: Active fit (with Na and HT) ---
def cost_active(params):
    gna, gkht = params
    t_sim, v_sim = run_simulation(gna, gkht, gklt, gh)
    v_interp = interpolate_simulation(t_sim, v_sim, t_exp)

    dt = t_exp[1] - t_exp[0]
    time_shift = abs(np.argmax(v_interp) - np.argmax(V_exp)) * dt
    time_error = 5.0 * time_shift

    mse = np.mean((v_interp - V_exp) ** 2)
    f_cost = feature_cost(v_interp, V_exp, t_exp)
    penalty = penalty_terms(v_interp)

    return 0.9 * mse + 0.8 * f_cost + time_error + penalty

bounds_active = [(100, 800), (100, 800)]  # gNa, gKHT
result_active = differential_evolution(cost_active, bounds_active, maxiter=20, popsize=10)
gna, gkht = result_active.x

print(f"[Stage 2] Active Fit - gNa: {gna:.2f}, gKHT: {gkht:.2f}")

# t_exp = experimentalTrace[499:,0]*1000 # in ms, sampled at 50 kHz
# t_exp = t_exp - t_exp[0]  # ensure starts at 0
# V_exp = experimentalTrace[499:,1]  # in mV

# Initial guess and bounds
bounds = [(100, 800), (100, 800), (gklt*0.75, gklt*1.25), (gh*0.5, gh*1.5)]
# result = differential_evolution(cost_function, bounds, strategy='best1bin',
#                                 maxiter=20, popsize=10, polish=True)

# x0 = [350, 350, gklt,gh,erev]  # gNa, gKHT, gKLT, gH
# bounds = [(1e-4, 700), (1e-4, 700),(gklt,gklt),(gh,gh),(erev,erev)]
# result = minimize(cost_function, x0, bounds=bounds, method='L-BFGS-B', options={'maxiter': 200})

result_global = differential_evolution(cost_function, bounds, strategy='best1bin', maxiter=20, popsize=10, polish=True)
result_local = minimize(cost_function, result_global.x, bounds=bounds, method='L-BFGS-B', options={'maxiter': 200})
opt_gna, opt_gkht, gklt, gh = result_local.x

# opt_gna, opt_gkht, gklt, gh, erev = result.x
print(f"Optimal gNa: {opt_gna:.2f} , Optimal gKHT: {opt_gkht:.2f}, Set gKLT: {gklt:.2f}, set gH: {gh:.2f}, Set erev: {erev:.2f}")

# Final simulation and plot
t_sim, v_sim = run_simulation(opt_gna, opt_gkht, gklt, gh)
feat_sim = extract_features(v_sim, t_sim)
print("Simulate Features:")
for k, v in feat_sim.items():
    print(f"{k}: {v:.2f}")

feat_exp = extract_features(V_exp, t_exp)
print("Experimental Features:")
for k, v in feat_exp.items():
    print(f"{k}: {v:.2f}")


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
plt.tight_layout()
plt.show()
