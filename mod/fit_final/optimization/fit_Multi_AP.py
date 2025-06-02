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

# === Parameters and Setup ===
h.load_file('stdrun.hoc')
np.random.seed(42)
script_dir = os.path.dirname(os.path.abspath(__file__))
param_file_path = os.path.join(script_dir, "..","results","_fit_results","best_fit_params.txt")
fenotype = "P9_iMNTB"
if not os.path.exists(param_file_path):
    raise FileNotFoundError(f"Passive parameters not found at: {param_file_path}")

with open(param_file_path, "r") as f:
    gleak, gklt, gh, erev, gkht, gna, gka = map(float, f.read().strip().split(","))

ek = -106.81
ena = 62.77
stim_dur = 300
stim_delay = 10

ParamSet = namedtuple("ParamSet", [
    "gna", "gkht", "gklt", "gh", "gka", "gleak", "stim_amp"
])
# === Define bounds variables ===
lbgNa = 0.5
hbgNa = 1.5
lbKht = 0.5
hbKht = 1.5
lbKlt = 0.5
hbKlt = 1.5
lbih = 0.999
hbih = 1.001
lbka = 0.1
hbka = 1.9
lbleak = 0.999
hbleak = 1.001
stim_amp = 0.210
lbamp = 0.8
hbamp = 1.2

bounds = [
    (gna * lbgNa, gna * hbgNa),        # gNa
    (gkht * lbKht, gkht * hbKht),      # gKHT
    (gklt * lbKlt, gklt * hbKlt),      # gKLT
    (gh * lbih, gh * hbih),           # gIH
    (gka * lbka, gka * hbka),         # gKA
    (gleak * lbleak, gleak * hbleak), # gLeak
    (stim_amp * lbamp, stim_amp * hbamp)  # stim_amp
]

# === Global results container ===
global_summary = []

def extract_features(trace, time):
    dt = time[1] - time[0]
    dV = np.gradient(trace, dt)
    rest = np.mean(trace[:int(9 / dt)])
    peak_idx = np.argmax(trace)
    peak = trace[peak_idx]
    try:
        thresh_idx = np.where(dV > 50)[0][0]
        threshold = trace[thresh_idx]
        latency = time[thresh_idx]
    except IndexError:
        return {k: np.nan for k in ['rest', 'peak', 'amp', 'threshold', 'latency', 'width', 'AHP']}
    amp = peak - threshold
    half_amp = threshold + 0.5 * amp
    above_half = np.where((trace > half_amp) & (np.arange(len(trace)) > thresh_idx) & (np.arange(len(trace)) < peak_idx + int(5 / dt)))[0]
    width = (above_half[-1] - above_half[0]) * dt if len(above_half) > 1 else np.nan
    AHP = np.min(trace[peak_idx:]) if peak_idx < len(trace) else np.nan
    return {'rest': rest, 'peak': peak, 'amp': amp, 'threshold': threshold, 'latency': latency, 'width': width, 'AHP': AHP}

def feature_cost(sim_trace, exp_trace, time):
    sim_feat = extract_features(sim_trace, time)
    exp_feat = extract_features(exp_trace, time)
    weights = {'rest': 1, 'peak': 1.0, 'amp': 1.0, 'width': 1.0, 'threshold': 10.0, 'latency': 1.0, 'AHP': 1.0}
    error = 0
    for k in weights:
        if not np.isnan(sim_feat[k]) and not np.isnan(exp_feat[k]):
            error += weights[k] * ((sim_feat[k] - exp_feat[k]) ** 2)
    return error

def run_simulation(p: ParamSet):
    v_init = -70
    totalcap = 25
    somaarea = (totalcap * 1e-6) / 1
    cell = MNTB(0, somaarea, erev, p.gleak, ena, p.gna, p.gh, p.gka, p.gklt, p.gkht, ek)
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
    return interp1d(t_neuron, v_neuron, kind='cubic', fill_value='extrapolate')(t_exp)

def penalty_terms(v_sim):
    peak, rest = np.max(v_sim), v_sim[0]
    penalty = 0
    if peak < -10 or peak > 10:
        penalty += 1
    if rest > -55 or rest < -80:
        penalty += 1000
    return penalty

def cost_function1(params):
    p = ParamSet(*params)
    t_sim, v_sim = run_simulation(p)
    v_interp = interpolate_simulation(t_sim, v_sim, t_exp)
    exp_feat, sim_feat = extract_features(V_exp, t_exp), extract_features(v_interp, t_exp)
    if np.isnan(exp_feat['latency']) or np.isnan(sim_feat['latency']):
        return 1e6
    dt = t_exp[1] - t_exp[0]
    ap_start = max(0, int((exp_feat['latency'] - 5) / dt))
    ap_end = min(len(t_exp), int((exp_feat['latency'] + 20) / dt))
    v_interp_ap, v_exp_ap = v_interp[ap_start:ap_end], V_exp[ap_start:ap_end]
    f_cost = feature_cost(v_interp_ap, v_exp_ap, t_exp[ap_start:ap_end])
    mse = np.mean((v_interp_ap - v_exp_ap) ** 2)
    time_shift = abs(np.argmax(v_interp_ap) - np.argmax(v_exp_ap)) * dt
    time_error = 500 * time_shift
    penalty = penalty_terms(v_interp)
    return 5 * mse + f_cost + time_error + penalty

def fit_single_AP(filename):
    global t_exp, V_exp
    data_path = os.path.join(script_dir, "..", "data", filename)
    experimentalTrace = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    t_exp = experimentalTrace[:, 0] - experimentalTrace[0, 0]
    if np.sum((t_exp >= 0) & (t_exp <= 0.002)) > 2:
        t_exp *= 1000
    V_exp = experimentalTrace[:, 1]
    if abs(V_exp[0]) < 1:
        V_exp *= 1000
    file = filename.split(".")[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(script_dir, "..", "results", f"fit_AP_{file}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    result_global = differential_evolution(cost_function1, bounds, strategy='best1bin', maxiter=20, popsize=70, polish=False)
    result_local = minimize(cost_function1, result_global.x, bounds=bounds, method='L-BFGS-B', options={'maxiter': 200})
    params_opt = ParamSet(*result_local.x)
    t_sim, v_sim = run_simulation(params_opt)

    feat_sim, feat_exp = extract_features(v_sim, t_sim), extract_features(V_exp, t_exp)
    v_interp = interpolate_simulation(t_sim, v_sim, t_exp)
    # Fit quality metrics
    mse = mean_squared_error(V_exp, v_interp)
    r2 = r2_score(V_exp, v_interp)
    time_shift = abs(np.argmax(v_interp) - np.argmax(V_exp)) * (t_exp[1] - t_exp[0])
    feature_error = feature_cost(v_interp, V_exp, t_exp)
    fit_quality = 'good' if r2 > 0.95 and time_shift < 0.5 else 'poor'



    results = params_opt._asdict()
    results.update(feat_sim)
    results['filename'] = filename
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

    results_poor = results[results['fit_quality'] == 'poor']
    print(f"Number of poor fits: {len(results_poor)}")
    print(results_poor[['filename', 'r2', 'time_shift', 'feature_error']])

    global_summary.append(results)
    pd.DataFrame([results]).to_csv(os.path.join(output_dir, f"fit_results_{timestamp}.csv"), index=False)
    plt.figure(figsize=(10, 5))
    plt.plot(t_exp, V_exp, label='Experimental', linewidth=2)
    plt.plot(t_sim, v_sim, label='Simulated (fit)', linestyle='--')
    plt.legend()
    plt.title(f"Fit: {file}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fit_plot_{file}.png"))
    plt.close()
    print(f"Finished fitting {file}")


if __name__ == "__main__":
    data_folder = os.path.join(script_dir, "..", "data",f"ap_{fenotype}")
    all_files = [f for f in os.listdir(data_folder) if f.endswith(".csv") and "sweep" in f]
    for fname in sorted(all_files):
        try:
            fit_single_AP(fname)
        except Exception as e:
            print(f"Error in {fname}: {e}")
    # === Save global summary ===
    summary_df = pd.DataFrame(global_summary)
    summary_path = os.path.join(script_dir, "..", "results", "_fit_results", f"summary_all_ap_{fenotype}_fits.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved global summary: {summary_path}")