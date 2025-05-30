import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize, differential_evolution
from neuron import h
from collections import namedtuple
import MNTB_PN_myFunctions as mFun
import datetime
from MNTB_PN_fit import MNTB
from sklearn.metrics import mean_squared_error, r2_score


ParamSet = namedtuple("ParamSet", [
    "gna", "gkht", "gklt", "gh", "gka", "gleak", "stim_amp"
])

h.load_file('stdrun.hoc')
np.random.seed(42)
script_dir = os.path.dirname(os.path.abspath(__file__))
param_file_path = os.path.join(script_dir, "..","results","_fit_results","best_fit_params.txt")
filename = "sweep_15_clipped_510ms_12172022_P9_FVB_PunTeTx_phasic_iMNTB.csv"

if not os.path.exists(param_file_path):
    raise FileNotFoundError(f"Passive parameters not found at: {param_file_path}")
with open(param_file_path, "r") as f:
    gleak, gklt, gh, erev, gkht, gna, gka = map(float, f.read().strip().split(","))


# === Create Output Folder ===
file = filename.split(".")[0]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(os.getcwd(),"..", "results", f"fit_AP_{file}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# Load experimental data
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", filename))
experimentalTrace = np.genfromtxt(data_path, delimiter=',', skip_header=1, dtype=float, filling_values=np.nan)

t_exp = experimentalTrace[:,0] # ms
t_exp = t_exp - t_exp[0]
samples = np.sum((t_exp >= 0) & (t_exp <= 0.002))
if samples > 2:
    t_exp = t_exp*1000
    print("t_exp converted to ms")

V_exp = experimentalTrace[:,1]  # mV
fp = V_exp[0]
if abs(fp) < 1:
    V_exp *= 1000
    print("V_exp converted to mV")
# Define soma parameters
totalcap = 25  # Total membrane capacitance in pF for the cell (input capacitance)
somaarea = (totalcap * 1e-6) / 1  # pf -> uF,assumes 1 uF/cm2; result is in cm2
h.celsius = 35
ek = -106.81
ena = 62.77
cell = MNTB(0, somaarea, erev, gleak, ena, gna, gh, gka, gklt, gkht, ek)


stim_dur = 300
stim_delay = 10

stim_amp = 0.210
lbamp = 0.999
hbamp = 1.001

gleak = gleak
lbleak = 0.999
hbleak = 1.001

gkht = 150
lbKht = 0.5
hbKht = 1.5

lbKlt = 0.5
hbKlt = 1.5

gka = 100
lbka = 0.3
hbka = 1.7

lbih = 0.999
hbih = 1.001

gna = 150
lbgNa = 0.5
hbgNa = 1.5

lbcNa = 0.9999
hbcNa = 1.0001

lbckh = 0.9999
hbckh = 1.0001


def extract_features(trace, time):
    dt = time[1] - time[0]
    dV = np.gradient(trace, dt)

    rest = np.mean(trace[:int(9/dt)])  # average first 5 ms
    peak_idx = np.argmax(trace)
    peak = trace[peak_idx]

    try:
        thresh_idx = np.where(dV > 50)[0][0]
        threshold = trace[thresh_idx]
        latency = time[thresh_idx]
    except IndexError:
        return {
            'rest': rest, 'peak': peak, 'amp': np.nan,
            'threshold': np.nan, 'latency': np.nan,
            'width': np.nan, 'AHP': np.nan
        }

    amp = peak - threshold
    half_amp = threshold + 0.5 * amp

    # ✅ Limit width to a narrow window around AP
    above_half = np.where(
        (trace > half_amp) &
        (np.arange(len(trace)) > thresh_idx) &
        (np.arange(len(trace)) < peak_idx + int(5/dt))
    )[0]

    width = (above_half[-1] - above_half[0]) * dt if len(above_half) > 1 else np.nan
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
        'rest': 1,
        'peak':     1.0,   # Increase penalty on overshoot
        'amp':      1.0,
        'width':    1.0,
        'threshold': 10.0,  # Strong push toward threshold match
        'latency':  1.0,
        'AHP':      1.0
    }
    error = 0
    for k in weights:
        if not np.isnan(sim_feat[k]) and not np.isnan(exp_feat[k]):
            error += weights[k] * ((sim_feat[k] - exp_feat[k]) ** 2)
    return error



#@lru_cache(maxsize=None)
def run_simulation(p: ParamSet, stim_dur=300, stim_delay=10):
    v_init = -70
    totalcap = 25  # pF
    somaarea = (totalcap * 1e-6) / 1  # cm² assuming 1 µF/cm²

    cell = MNTB(
        gid=0,
        somaarea=somaarea,
        erev=erev,
        gleak=p.gleak,
        ena=ena,
        gna=p.gna,
        gh=p.gh,
        gka=p.gka,
        gklt=p.gklt,
        gkht=p.gkht,
        ek=ek
    )

    stim = h.IClamp(cell.soma(0.5))
    stim.delay = stim_delay
    stim.dur = stim_dur
    stim.amp = p.stim_amp

    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(cell.soma(0.5)._ref_v)

    h.dt = 0.02
    h.steps_per_ms = int(1 / h.dt)
    h.v_init = v_init
    mFun.custom_init(v_init)
    h.continuerun(stim_delay + stim_dur + 200)

    return np.array(t_vec), np.array(v_vec)


def interpolate_simulation(t_neuron, v_neuron, t_exp):
    interp_func = interp1d(t_neuron, v_neuron, kind='cubic', fill_value='extrapolate')
    return interp_func(t_exp)

def penalty_terms(v_sim):
    peak = np.max(v_sim)
    rest = v_sim[0]
    penalty = 0
    if peak < -10 or peak > 10:
        penalty += 1
    if rest > -55 or rest < -80:
        penalty += 1000
    return penalty

def cost_function(params): #no ap window
    assert len(params) == 7, "Mismatch in number of parameters"
    p = ParamSet(*params)

    t_sim, v_sim = run_simulation(p)

    v_interp = interpolate_simulation(t_sim, v_sim, t_exp)
    exp_feat = extract_features(V_exp, t_exp)
    sim_feat = extract_features(v_interp, t_exp)
    # Time shift between peaks
    dt = t_exp[1] - t_exp[0]
    time_shift = abs(np.argmax(v_interp) - np.argmax(V_exp)) * dt
    weight = 50  # you can tune this weight
    time_error = weight * time_shift

    mse = np.mean((v_interp - V_exp)**2)
    f_cost = feature_cost(v_interp, V_exp, t_exp)
    penalty = penalty_terms(v_interp)
    peak_penalty = 0
    # sim_peak = np.max(v_interp)
    # if sim_peak > 5:
    #     peak_penalty += 10 * (sim_peak - 20)**2

    alpha = 1  # weight for MSE
    beta =  2 # weight for feature cost

    total_cost = alpha * mse + beta * f_cost + time_error + penalty + peak_penalty

    return total_cost

def cost_function1(params):
    assert len(params) == 7, "Mismatch in number of parameters"
    p = ParamSet(*params)

    # Run simulation
    t_sim, v_sim = run_simulation(p)

    # Interpolate to match experimental time resolution
    v_interp = interpolate_simulation(t_sim, v_sim, t_exp)

    # === Extract AP region ===
    exp_feat = extract_features(V_exp, t_exp)
    sim_feat = extract_features(v_interp, t_exp)

    # If no AP detected in either, return a large penalty
    if np.isnan(exp_feat['latency']) or np.isnan(sim_feat['latency']):
        return 1e6

    # Define AP window (2 ms before threshold to 30 ms after peak)
    dt = t_exp[1] - t_exp[0]
    try:
        ap_start = max(0, int((exp_feat['latency'] - 5) / dt))
        ap_end = min(len(t_exp), int((exp_feat['latency'] + 20) / dt))
    except Exception:
        return 1e6

    v_interp_ap = v_interp[ap_start:ap_end]
    v_exp_ap = V_exp[ap_start:ap_end]
    t_ap = t_exp[ap_start:ap_end]

    # Feature cost only within AP window
    f_cost = feature_cost(v_interp_ap, v_exp_ap, t_ap)

    # MSE only in AP window
    mse = np.mean((v_interp_ap - v_exp_ap) ** 2)

    # Time shift of peaks (only within window)
    time_shift = abs(np.argmax(v_interp_ap) - np.argmax(v_exp_ap)) * dt
    time_error = 500 * time_shift

    # Resting potential penalty still on full trace
    penalty = penalty_terms(v_interp)

    # Total weighted cost
    alpha = 5     # MSE
    beta =  1     # Feature cost

    total_cost = alpha * mse + beta * f_cost + time_error + penalty

    return total_cost

print("Running optimization...")
bounds = [
#    (100, 2000),                       # gNa
#    (100, 2000),                       # gKHT
    (gna*lbgNa, gna*hbgNa),             # gNa
    (gkht * lbKht, gkht * hbKht),
    (gklt * lbKlt, gklt * hbKlt),       # gKLT
    (gh * lbih, gh * hbih),             # gIH
    (gka * lbka, gka * hbka),           # gka
    (gleak * lbleak, gleak * hbleak),   # gleak
    (stim_amp*lbamp, stim_amp*hbamp)  # stim-amp
]

result_global = differential_evolution(cost_function1, bounds, strategy='best1bin', maxiter=50, popsize=20, polish=True)
result_local = minimize(cost_function1, result_global.x, bounds=bounds, method='L-BFGS-B', options={'maxiter': 400})
params_opt = ParamSet(*result_local.x)
print("Optimized parameters:")
for name, value in params_opt._asdict().items():
    print(f"{name}: {value:.4f}")

#
print(f"Best stim-amp: {params_opt.stim_amp:.2f} pA")
print(f" Optimized gna: {params_opt.gna:.2f}, gklt: {params_opt.gklt: .2f}, gkht: {params_opt.gkht: .2f}), gh: {params_opt.gh:.2f}, gka:{params_opt.gka:.2f}, gleak: {params_opt.gleak:.2f}")

# Final simulation and plot
t_sim, v_sim = run_simulation(params_opt)


# Interpolate simulated trace to match experimental time points
v_interp = interpolate_simulation(t_sim, v_sim, t_exp)

# Compute fit quality metrics
mse = mean_squared_error(V_exp, v_interp)
r2 = r2_score(V_exp, v_interp)
time_shift = abs(np.argmax(v_interp) - np.argmax(V_exp)) * (t_exp[1] - t_exp[0])
feature_error = feature_cost(v_interp, V_exp, t_exp)

# Add fit quality label
fit_quality = 'good' if r2 > 0.95 and time_shift < 0.5 else 'poor'


feat_sim = extract_features(v_sim, t_sim)
print("Simulate Features:")
for k, v in feat_sim.items():
    print(f"{k}: {v:.2f}")

feat_exp = extract_features(V_exp, t_exp)
print("Experimental Features:")
for k, v in feat_exp.items():
    print(f"{k}: {v:.2f}")

results = params_opt._asdict()
results.update(feat_sim)
results = params_opt._asdict()
results.update(feat_sim)
results['mse'] = mse
results['r2'] = r2
results['time_shift'] = time_shift
results['feature_error'] = feature_error
results['fit_quality'] = fit_quality
print(f"\n=== Fit Quality Metrics ===")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²):           {r2:.4f}")
print(f"Time Shift (ms):          {time_shift:.4f}")
print(f"Feature Error:            {feature_error:.2f}")
print(f"Fit Quality:              {fit_quality}")


pd.DataFrame([results]).to_csv(os.path.join(output_dir,f"fit_results_{timestamp}.csv"), index=False)

results_exp = {feat_exp[k]: v for k, v in results.items() if k in feat_exp}
df = pd.DataFrame([results_exp])  # Create DataFrame first
df = pd.DataFrame([results_exp]).to_csv(os.path.join(output_dir,f"fit_results_exp_{timestamp}.csv"), index=False)

combined_results = {
    "gleak": gleak, "gklt": params_opt.gklt, "gh": params_opt.gh, "erev": erev,
    "gna": params_opt.gna, "gkht": params_opt.gkht, "gka": params_opt.gka,
    "stim_amp": params_opt.stim_amp
}

pd.DataFrame([combined_results]).to_csv(os.path.join(script_dir, "..","results","_fit_results", f"all_fitted_params_{file}_{timestamp}.csv"), index=False)
pd.DataFrame([combined_results]).to_csv(os.path.join(script_dir, "..","results","_fit_results", f"all_fitted_params.csv"), index=False) #the last
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

half_amp = feat_sim['threshold'] + 0.5 * feat_sim['amp']
plt.plot(t_sim, v_sim)
plt.axhline(half_amp, color='red', linestyle='--', label='Half amplitude')
plt.title("Check AP width region")
plt.legend()
plt.show()