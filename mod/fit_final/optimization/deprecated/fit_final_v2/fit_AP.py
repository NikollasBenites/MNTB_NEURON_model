# fit_AP.py

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize, differential_evolution
from neuron import h
import MNTB_PN_myFunctions as mFun

h.load_file("stdrun.hoc")
np.random.seed(1)
h.celsius = 35

def load_passive_params(results_dir):
    path = os.path.join(results_dir, "best_fit_params.txt")
    with open(path, "r") as f:
        return map(float, f.read().strip().split(","))

def load_experimental_trace():
    data_path = os.path.abspath(os.path.join("data", "sweep_23_clipped_510ms.csv"))
    trace = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    t_exp = trace[:, 0]
    samples = np.sum((t_exp >= 0) & (t_exp <= 0.02))
    if samples <= 2:
        t_exp *= 1000

    v_exp = trace[:, 1]
    fp = v_exp[0]
    if abs(fp) < 1:
        v_exp *= 1000

    t_exp -= t_exp[0]
    return t_exp, v_exp

def nstomho(x, area_cm2):
    return (1e-9 * x / area_cm2)

def create_soma(params, area_cm2):
    gleak, gklt, gh, gka, erev, gkht, gna = params
    soma = h.Section(name='soma')
    soma.L = 20
    soma.diam = 15
    soma.Ra = 150
    soma.cm = 1
    soma.insert('leak')
    soma.insert('LT_dth')
    soma.insert('IH_dth')
    soma.insert('HT_dth_nmb')
    soma.insert('NaCh_nmb')
    soma.insert('ka')
    soma.ek = -106.1
    soma.ena = 62.77
    soma.g_leak = nstomho(gleak, area_cm2)
    soma.gkltbar_LT_dth = nstomho(gklt, area_cm2)
    soma.ghbar_IH_dth = nstomho(gh, area_cm2)
    soma.gkabar_ka = nstomho(gka, area_cm2)
    soma.erev_leak = erev
    soma.gkhtbar_HT_dth_nmb = nstomho(gkht, area_cm2)
    soma.gnabar_NaCh_nmb = nstomho(gna, area_cm2)
    return soma

def extract_features(trace, time):
    dt = time[1] - time[0]
    dV = np.gradient(trace, dt)
    rest = np.mean(trace[:int(9/dt)])
    peak_idx = np.argmax(trace)
    peak = trace[peak_idx]
    try:
        thresh_idx = np.where(dV > 50)[0][0]
        threshold = trace[thresh_idx]
        latency = time[thresh_idx]
    except IndexError:
        threshold, latency = np.nan, np.nan
    amp = peak - threshold
    half_amp = threshold + 0.5 * amp
    above_half = np.where(trace > half_amp)[0]
    width = (above_half[-1] - above_half[0]) * dt if len(above_half) > 2 else np.nan
    AHP = np.min(trace[peak_idx:]) if peak_idx < len(trace) else np.nan
    return {
        'rest': rest, 'peak': peak, 'amp': amp,
        'threshold': threshold, 'latency': latency,
        'width': width, 'AHP': AHP
    }

def feature_cost(sim_trace, exp_trace, time):
    sim_feat = extract_features(sim_trace, time)
    exp_feat = extract_features(exp_trace, time)
    weights = {'rest': 5, 'peak': 5, 'amp': 5, 'width': 7, 'threshold': 0.001, 'latency': 50, 'AHP': 1}
    return sum(weights[k] * (sim_feat[k] - exp_feat[k])**2 for k in weights if not np.isnan(sim_feat[k]) and not np.isnan(exp_feat[k]))

def interpolate_simulation(t_neuron, v_neuron, t_exp):
    interp_func = interp1d(t_neuron, v_neuron, kind='cubic', fill_value='extrapolate')
    return interp_func(t_exp)

def run_simulation(soma, stim_amp=0.150, stim_dur=300):
    stim = h.IClamp(soma(0.5))
    stim.delay = 10
    stim.dur = stim_dur
    stim.amp = stim_amp
    h.dt = 0.02
    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(soma(0.5)._ref_v)
    v_init = -70
    mFun.custom_init(v_init)
    h.continuerun(510)
    return np.array(t_vec), np.array(v_vec)

def cost_function(params, soma_template, t_exp, v_exp):
    gna, gkht, gklt, gh, gka, gleak, stim_amp = params
    soma = soma_template(gna, gkht, gklt, gh, gka, gleak)
    t_sim, v_sim = run_simulation(soma, stim_amp=stim_amp)
    v_interp = interpolate_simulation(t_sim, v_sim, t_exp)
    mse = np.mean((v_interp - v_exp)**2)
    f_cost = feature_cost(v_interp, v_exp, t_exp)
    return 2 * mse + 1 * f_cost

def fit_AP(results_dir, age):
    gleak, gklt, gh, gka, erev, gkht, gna = load_passive_params(results_dir)
    t_exp, v_exp = load_experimental_trace()
    totalcap = 25
    area = (totalcap * 1e-6)
    soma_template = lambda gna, gkht, gklt, gh, gka, gleak: create_soma(
        (gleak, gklt, gh, gka, erev, gkht, gna), area
    )

    bounds = [
        (gna*0.2, gna*2),
        (gkht*0.2, gkht*2),
        (gklt*0.8, gklt*1.2),
        (gh*0.8, gh*1.2),
        (gka*0.8, gka*1.2),
        (gleak*0.8, gleak*1.2),
        (0.1, 0.4)  # stim amp
    ]
    init = [gna, gkht, gklt, gh, gka, gleak, 0.15]

    result = differential_evolution(
        lambda p: cost_function(p, soma_template, t_exp, v_exp),
        bounds,
        maxiter=20,
        polish=True
    )
    final = minimize(
        lambda p: cost_function(p, soma_template, t_exp, v_exp),
        result.x,
        bounds=bounds,
        method='L-BFGS-B'
    )
    (gna_opt, gkht_opt, gklt_opt, gh_opt, gka_opt, gleak_opt, stim_amp) = final.x

    params = {
        "gleak": gleak_opt, "gka": gklt_opt, "gh": gh_opt, "erev": erev,
        "gka": gka_opt, "gna": gna_opt, "gkht": gkht_opt, "stim_amp": stim_amp
    }

    pd.DataFrame([params]).to_csv(os.path.join(results_dir, "all_fitted_params.csv"), index=False)
    print("✅ Saved active/AP fitted parameters.")

    soma_final = soma_template(gna_opt, gkht_opt, gklt_opt, gh_opt, gka_opt, gleak_opt)
    t_sim, v_sim = run_simulation(soma_final, stim_amp=stim_amp)

    plt.figure(figsize=(10, 5))
    plt.plot(t_exp, v_exp, label='Experimental', linewidth=2)
    plt.plot(t_sim, v_sim, label='Simulated', linestyle='--')
    plt.axvline(x=extract_features(v_exp, t_exp)['latency'], color='blue', linestyle=':', label='Exp Thresh')
    plt.axvline(x=extract_features(v_sim, t_sim)['latency'], color='orange', linestyle=':', label='Sim Thresh')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Action Potential Fit')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "ap_fit.png"), dpi=300)
    print("📊 Saved AP fit plot.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--file", type=str, default="P9")
    args = parser.parse_args()

    fit_AP(args.results_dir, args.age)

if __name__ == "__main__":
    main()
