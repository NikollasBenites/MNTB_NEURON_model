import numpy as np
import os

from fontTools.merge.tables import headFlagsMergeBitMap

np.random.seed(1)
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize, differential_evolution
from neuron import h
import MNTB_PN_myFunctions as mFun
#from functools import lru_cache
import datetime
h.load_file('stdrun.hoc')

script_dir = os.path.dirname(os.path.abspath(__file__))
param_file_path = os.path.join(script_dir, "best_fit_params.txt")
if not os.path.exists(param_file_path):
    raise FileNotFoundError(f"Passive parameters not found at: {param_file_path}")
with open(param_file_path, "r") as f:
    gleak, gklt, gh, erev, gna, gkht = map(float, f.read().strip().split(","))

def nstomho(x):
    return (1e-9 * x / somaarea)  # Convert conductance to mho/cm²

# === Create Output Folder ===
age = "9_TeNT_1223204"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(os.getcwd(), "results", f"BestFit_P{age}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# Load experimental data
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sweep_20_clipped_50ms_02132024_P9.csv"))
experimentalTrace = np.genfromtxt(data_path, delimiter=',', skip_header=1, dtype=float, filling_values=np.nan)
timeconverter = 1

t_exp = experimentalTrace[:,0]*timeconverter  # ms
t_exp = t_exp - t_exp[0]
V_exp = experimentalTrace[:,1]  # mV

# Define soma parameters
totalcap = 25  # Total membrane capacitance in pF
somaarea = (totalcap * 1e-6) / 1  # Convert to cm^2 assuming 1 µF/cm²

# Create soma section
soma = h.Section(name='soma')
soma.L = 20  # µm
soma.diam = 15  # µm
soma.Ra = 150
soma.cm = 1
v_init = -70

soma.insert('leak')
soma.insert('LT_dth')
soma.insert('IH_dth')
soma.insert('HT_dth_nmb')
soma.insert('NaCh_nmb')
#soma.insert('ka')

soma.ek = -106.1
soma.ena = 62.77
################## sodium kinetics
cam = 100 #76.4
kam = .037
cbm = 1 #6.930852
kbm = -.043

cah = 0.000533
kah = -.0909
cbh = .787
kbh = .0691
################# KHT kinetics
can = .2719
kan = .04
cbn = .1974
kbn = 0

cap = .00713
kap = -.1942
cbp = .0935
kbp = .0058

stim_dur = 40

stim_amp = 0.100
lbamp = 0.9
hbamp = 1.1

lbleak = 0.9
hbleak = 1.1

gkht = 400
lbKht = 0.7
hbKht = 1.9

lbKlt = 0.9
hbKlt = 1.1

# gka = 100
# lbka = 0.9
# hbka = 1.1

lbih = 0.1
hbih = 1.9

gna = 400
lbgNa = 0.9
hbgNa = 1.5

lbcNa = 0.5
hbcNa = 1.5

lbckh = 0.5
hbckh = 1.5

def set_conductances(gna, gkht, gklt, gh, erev, gleak,
                     cam, kam, cbm, kbm,
                     cah, kah, cbh, kbh,
                     can, kan, cbn, kbn,
                     cap, kap, cbp, kbp):
    soma.gnabar_NaCh_nmb = nstomho(gna)
    soma.cam_NaCh_nmb = cam
    soma.kam_NaCh_nmb = kam
    soma.cbm_NaCh_nmb = cbm
    soma.kbm_NaCh_nmb = kbm
    soma.cah_NaCh_nmb = cah
    soma.kah_NaCh_nmb = kah
    soma.cbh_NaCh_nmb = cbh
    soma.kbh_NaCh_nmb = kbh
    soma.gkhtbar_HT_dth_nmb = nstomho(gkht)
    soma.can_HT_dth_nmb = can
    soma.kan_HT_dth_nmb = kan
    soma.cbn_HT_dth_nmb = cbn
    soma.kbn_HT_dth_nmb = kbn
    soma.cap_HT_dth_nmb = cap
    soma.kap_HT_dth_nmb = kap
    soma.cbp_HT_dth_nmb = cbp
    soma.kbp_HT_dth_nmb = kbp
    soma.gkltbar_LT_dth = nstomho(gklt)
    soma.ghbar_IH_dth = nstomho(gh)
    soma.g_leak = nstomho(gleak)
    soma.erev_leak = erev

# def set_conductances2(gna, gkht, gklt, gh, erev, gleak):
#     soma.gnabar_NaCh_nmb = nstomho(gna)
#     soma.gkhtbar_HT_dth_nmb = nstomho(gkht)
#     soma.gkltbar_LT_dth = nstomho(gklt)
#     soma.ghbar_IH_dth = nstomho(gh)
#     soma.g_leak = nstomho(gleak)
#     soma.erev_leak = erev

def extract_features(trace, time):
    dt = time[1] - time[0]
    dV = np.gradient(trace, dt)

    rest = np.mean(trace[:int(9/dt)])  # average first 5 ms
    peak_idx = np.argmax(trace)
    peak = trace[peak_idx]

    # Threshold = first time where dV/dt > 10 mV/ms
    try:
        thresh_idx = np.where(dV > 50)[0][0]
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
        'rest': 5,
        'peak':     5,   # Increase penalty on overshoot
        'amp':      10,
        'width':    7.0,
        'threshold': 10.0,  # Strong push toward threshold match
        'latency':  5.0,
        'AHP':      1.0
    }
    error = 0
    for k in weights:
        if not np.isnan(sim_feat[k]) and not np.isnan(exp_feat[k]):
            error += weights[k] * ((sim_feat[k] - exp_feat[k]) ** 2)
    return error

stim_delay = 10
h.celsius = 35
#@lru_cache(maxsize=None)
def run_simulation(gna, gkht, gklt, gh, gleak,
                   cam, kam, cbm, kbm,
                   cah, kah, cbh, kbh,
                   can, kan, cbn, kbn,
                   cap, kap, cbp, kbp,
                   stim_amp=stim_amp, stim_dur=stim_dur):
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

    h.v_init = v_init
    mFun.custom_init(v_init)
    h.continuerun(stim_delay+stim_dur)
    # h.continuerun(510)
    return np.array(t_vec), np.array(v_vec)

# def run_simulation2(gna, gkht, gklt, gh, stim_amp=0.320, stim_dur=stim_dur):
#     set_conductances2(gna, gkht, gklt, gh, erev, gleak)
#
#     stim = h.IClamp(soma(0.5))
#     stim.delay = 10
#     stim.dur = stim_dur
#     stim.amp = stim_amp
#
#     h.dt = 0.02
#     h.steps_per_ms = int(1.0 / h.dt)
#     t_vec = h.Vector().record(h._ref_t)
#     v_vec = h.Vector().record(soma(0.5)._ref_v)
#
#     h.v_init = v_init
#     mFun.custom_init(v_init)
#     h.continuerun(stim.delay+stim_dur)
#
#     return np.array(t_vec), np.array(v_vec)
# # def monitor_cache_size():
# #     cache_info = run_simulation.cache_info()
# #     print(f"Cache size: {cache_info.currsize}/{cache_info.maxsize}")
# #     print(f"Hit ratio: {cache_info.hits/(cache_info.hits + cache_info.misses):.2%}")

def interpolate_simulation(t_neuron, v_neuron, t_exp):
    interp_func = interp1d(t_neuron, v_neuron, kind='cubic', fill_value='extrapolate')
    return interp_func(t_exp)

def penalty_terms(v_sim):
    peak = np.max(v_sim)
    rest = v_sim[0]
    penalty = 0
    if peak < -10 or peak > 10:
        penalty += 100
    if rest > -55 or rest < -80:
        penalty += 1000
    return penalty

def cost_function(params):
    (gna, gkht, gklt, gh, gleak,
     cam, kam, cbm, kbm,
     cah, kah, cbh, kbh,
     can, kan, cbn, kbn,
     cap, kap, cbp, kbp, stim_amp) = params

    t_sim, v_sim = run_simulation(gna, gkht, gklt, gh, gleak, cam, kam, cbm, kbm, cah, kah, cbh, kbh, can, kan,
                                  cbn, kbn, cap, kap, cbp, kbp, stim_amp=stim_amp, stim_dur=stim_dur)

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

    alpha = 2  # weight for MSE
    beta = 1 # weight for feature cost

    total_cost = alpha * mse + beta * f_cost + time_error + penalty + peak_penalty

    return total_cost

bounds = [
#    (100, 2000),                       # gNa
#    (100, 2000),                       # gKHT
    (gna*lbgNa, gna*hbgNa),             # gNa
    (gkht * lbKht, gkht * hbKht),
    (gklt * lbKlt, gklt * hbKlt),       # gKLT
    (gh * lbih, gh * hbih),             # gIH
    #(gka * lbka, gka * hbka),           # gka
    (gleak * lbleak, gleak * hbleak),     # gleak
    # Na activation (m)
    (cam * lbcNa, cam * hbcNa),    # cam
    (kam * lbcNa, kam * hbcNa),    # kam
    (cbm * lbcNa, cbm * hbcNa),    # cbm
    (kbm * hbcNa, kbm * lbcNa),    # kbm (note: negative slope → flip)

    # Na inactivation (h)
    (cah * lbcNa, cah * hbcNa),    # cah
    (kah * hbcNa, kah * lbcNa),    # kah (note: negative slope → flip)
    (cbh * lbcNa, cbh * hbcNa),    # cbh
    (kbh * lbcNa, kbh * hbcNa),    # kbh

    # KHT activation (n)
    (can * lbckh, can * hbckh),    # can
    (kan * lbckh, kan * hbckh),    # kan
    (cbn * lbckh, cbn * hbckh),    # cbn
    (kbn * lbckh, kbn * hbckh),    # kbn

    # KHT inactivation (p)
    (cap * lbckh, cap * hbckh),    # cap
    (kap * hbckh, kap * lbckh),    # kap (note: negative slope → flip)
    (cbp * lbckh, cbp * hbckh),    # cbp
    (kbp * lbckh, kbp * hbckh),    # kbp

    (stim_amp*lbamp, stim_amp*hbamp)  # stim-amp
]


result_global = differential_evolution(cost_function, bounds, strategy='best1bin', maxiter=20, popsize=10, polish=True)
result_local = minimize(cost_function, result_global.x, bounds=bounds, method='L-BFGS-B', options={'maxiter': 200})
print(result_local.x)
params_opt = result_local.x
#
(gna_opt, gkht_opt, gklt_opt, gh_opt,gleak_opt,
 cam_opt, kam_opt, cbm_opt, kbm_opt,
 cah_opt, kah_opt, cbh_opt, kbh_opt,
 can_opt, kan_opt, cbn_opt, kbn_opt,
 cap_opt, kap_opt, cbp_opt, kbp_opt, opt_stim) = params_opt
print(f"Best stim-amp: {opt_stim:.2f} pA")
print(f" Optimized gna: {gna_opt:.2f}, gklt: {gklt_opt: .2f}, gkht: {gkht_opt: .2f}), gh: {gh_opt:.2f}, gleak: {gleak_opt:.2f}")
print(f" Optimized cam: {cam_opt:.2f}, kam: {kam_opt:.3f}, cbm: {cbm_opt:.2f}, kbm: {kbm_opt:.3f}")
print(f" Optimized cah: {cah_opt:.5f}, kah: {kah_opt:.4f}, cbh: {cbh_opt:.2f}, kbh: {kbh_opt:.3f}")



# Final simulation and plot
t_sim, v_sim = run_simulation(gna_opt, gkht_opt, gklt_opt, gh_opt, gleak_opt, cam_opt, kam_opt, cbm_opt, kbm_opt,
                              cah_opt, kah_opt, cbh_opt, kbh_opt, can_opt, kan_opt, cbn_opt, kbn_opt, cap_opt, kap_opt,
                              cbp_opt, kbp_opt, opt_stim)

feat_sim = extract_features(v_sim, t_sim)
print("Simulate Features:")
for k, v in feat_sim.items():
    print(f"{k}: {v:.2f}")

feat_exp = extract_features(V_exp, t_exp)
print("Experimental Features:")
for k, v in feat_exp.items():
    print(f"{k}: {v:.2f}")

results = {
    "gna_opt": f"{gna_opt:.2f}",
    "gkht_opt": f"{gkht_opt:.2f}",
    "gklt_opt": f"{gklt_opt:.2f}",
    "gh_opt": f"{gh_opt:.2f}",
    "cam": f"{cam_opt:.2f}", "kam": f"{kam_opt:.3f}", "cbm": f"{cbm_opt:.2f}", "kbm": f"{kbm_opt:.3f}",
    "cah": f"{cah_opt:.5f}", "kah": f"{kah_opt:.4f}", "cbh": f"{cbh_opt:.2f}", "kbh": f"{kbh_opt:.3f}",
    "opt_stim": f"{opt_stim:.2f}",
    **feat_sim
}


df = pd.DataFrame([results]).to_csv(os.path.join(output_dir,f"fit_results_{timestamp}.csv"), index=False)
combined_results = {
    "gleak": gleak, "gklt": gklt_opt, "gh": gh_opt, "erev": erev,
    "gna": gna_opt, "gkht": gkht_opt,
    "cam": cam_opt, "kam": kam_opt, "cbm": cbm_opt, "kbm": kbm_opt,
    "cah": cah_opt, "kah": kah_opt, "cbh": cbh_opt, "kbh": kbh_opt,
    "can": can_opt, "kan": kan_opt, "cbn": cbn_opt, "kbn": kbn_opt,
    "cap": cap_opt, "kap": kap_opt, "cbp": cbp_opt, "kbp": kbp_opt,
    "stim_amp": opt_stim
}


pd.DataFrame([combined_results]).to_csv(os.path.join(script_dir, "all_fitted_params.csv"), index=False)
# monitor_cache_size()
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
