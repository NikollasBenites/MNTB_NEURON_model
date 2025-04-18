# simulation.py

import numpy as np
from functools import lru_cache
from scipy.interpolate import interp1d
from neuron import h
import config
import MNTB_PN_myFunctions as mFun
from neuron_model import set_conductances
import os



h.load_file('stdrun.hoc')

#@lru_cache(maxsize=2048)
def run_simulation(soma, axon, dend, params):
    # Unpack params
    # (gna, gkht, gklt,
    #  cam, kam, cbm, kbm,
    #  cah, kah, cbh, kbh,
    #  can, kan, cbn, kbn,
    #  cap, kap, cbp, kbp,
    #  na_scale,kht_scale,) = params
    neuron_params = params[:-5]
    na_scale = params[-5]
    kht_scale = params[-4]
    klt_scale = params[-3]
    ih_scale = params[-2]
    stim_amp = params[-1]


    # Set conductances
    set_conductances(soma, axon, dend, neuron_params, na_scale, kht_scale, klt_scale, ih_scale)

    # Insert stimulation
    stim = h.IClamp(soma(0.5))
    stim.delay = config.stim_delay_ms
    stim.dur = config.stim_dur_ms
    stim.amp = stim_amp
    #stim.amp = config.stim_amp_nA

    # Setup recording
    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(soma(0.5)._ref_v)

    h.dt = config.h_dt
    h.steps_per_ms = int(1.0 / h.dt)
    h.v_init = config.v_init
    mFun.custom_init(config.v_init)

    h.tstop = stim.delay + stim.dur
    h.continuerun(510)

    return np.array(t_vec), np.array(v_vec)

def interpolate_simulation(t_neuron, v_neuron, t_exp):
    interp_func = interp1d(t_neuron, v_neuron, kind='cubic', fill_value='extrapolate')
    return interp_func(t_exp)

def extract_features(trace, time):
    dt = time[1] - time[0]
    dV = np.gradient(trace, dt)
    rest = trace[:int(5/dt)].mean()
    peak_idx = np.argmax(trace)
    peak = trace[peak_idx]

    search_start_idx = np.searchsorted(time, 11)
    dV_slice = dV[search_start_idx:]
    trace_slice = trace[search_start_idx:]

    try:
        relative_thresh_idx = np.where(dV_slice > 45)[0][0]
        thresh_idx = search_start_idx + relative_thresh_idx
        threshold = trace[thresh_idx]
        latency = time[thresh_idx]
    except IndexError:
        threshold = np.nan
        latency = np.nan

    amp = peak - threshold

    half_amp = threshold + 0.5 * amp
    above_half = np.where(trace > half_amp)[0]
    width = (above_half[-1] - above_half[0]) * dt if len(above_half) > 2 else np.nan

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
    weights = {'peak': 1, 'amp': 1, 'width': 1, 'threshold': 1, 'latency':1, 'AHP':1}

    error = 0
    for k in weights:
        if not np.isnan(sim_feat[k]) and not np.isnan(exp_feat[k]):
            error += weights[k] * ((sim_feat[k] - exp_feat[k])**2)
    return error

def penalty_terms(v_sim,stim_amp=None):
    peak = np.max(v_sim)
    rest = v_sim[0]
    penalty = 0
    if peak < -10 or peak > 10:
        penalty += 100
    if rest > -50 or rest < -80:
        penalty += 1000
    if stim_amp is not None:
        if stim_amp < 0.1 or stim_amp > 1.5:
            penalty += 500
    return penalty

def max_dvdt(trace, time):
    dt = time[1] - time[0]
    dVdt = np.gradient(trace, dt)
    return np.max(dVdt)

def cost_function(params, soma, axon, dend, t_exp, v_exp):
    t_sim, v_sim = run_simulation(soma, axon, dend, params)

    v_interp = interpolate_simulation(t_sim, v_sim, t_exp)

    # Extract AP window
    features_exp = extract_features(v_exp, t_exp)
    ap_tmin = features_exp['latency']
    ap_tmax = t_exp[-1]
    stim_amp = params[-1]

    if not np.isnan(features_exp['AHP']):
        try:
            ahp_idx = np.where(v_exp == features_exp['AHP'])[0][0]
            ap_tmax = t_exp[ahp_idx]
        except IndexError:
            pass
    if np.isnan(ap_tmin) or ap_tmin >= ap_tmax:
        return 1e6

    ap_mask = (t_exp >= ap_tmin) & (t_exp <= ap_tmax)
    if np.sum(ap_mask) < 2:
        return 1e6

    t_ap = t_exp[ap_mask]
    v_exp_ap = v_exp[ap_mask]
    v_sim_ap = v_interp[ap_mask]

    max_dvdt_exp = max_dvdt(v_exp_ap, t_ap)
    max_dvdt_sim = max_dvdt(v_sim_ap, t_ap)
    dvdt_error = 3.0 * (max_dvdt_sim - max_dvdt_exp)**2

    dt = t_ap[1] - t_ap[0]
    time_shift = abs(np.argmax(v_sim_ap) - np.argmax(v_exp_ap)) * dt
    time_error = 5.0 * time_shift

    rmse = np.sqrt(np.mean((v_sim_ap - v_exp_ap)**2))
    f_cost = feature_cost(v_sim_ap, v_exp_ap, t_ap)
    penalty = penalty_terms(v_sim_ap,stim_amp=stim_amp)

    alpha = 5
    beta = 0.5

    total_cost = alpha * rmse + beta * f_cost + time_error + dvdt_error + penalty
    return total_cost
