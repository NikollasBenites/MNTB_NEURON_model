# simulation.py

import numpy as np
from neuron import h
from scipy.interpolate import interp1d

import MNTB_PN_myFunctions as mFun
import config_bpop
from neuron_model import set_conductances

h.load_file('stdrun.hoc')


def run_simulation(soma, axon, dend, params):
    # Unpack params
    neuron_params = params[:-5]
    na_scale = params[-5]
    kht_scale = params[-4]
    klt_scale = params[-3]
    ih_soma = params[-2]
    ih_dend = params[-1]

    # 1. Set conductances
    set_conductances(soma, axon, dend, neuron_params, na_scale, kht_scale, klt_scale, ih_soma, ih_dend)

    # 2. Setup fixed time step
    h.dt = config_bpop.h_dt
    h.steps_per_ms = int(1.0 / h.dt)

    # 3. Initial voltage
    h.v_init = config_bpop.v_init
    #mFun.custom_init_multicompartment(v_init=config_bpop.v_init, relax_time_ms=config_bpop.relax_time_ms)
    mFun.custom_init(config_bpop.v_init)
    # 4. Relaxation phase
    # v_relax, t_relax = mFun.relax_to_steady_state(soma, config_bpop.relax_time_ms)
    # if v_relax is None or t_relax is None or len(v_relax) == 0 or len(t_relax) == 0:
    #     print("⚠️ Empty relaxation detected — returning penalty signal.")
    #     return None, None
    # mFun.check_relaxation_stability(v_relax, t_relax, threshold_mV=0.1, last_ms=50)

    # 5. Save steady state
    steady_state = h.batch
    steady_state.save()

    # 6. Now simulate each sweep
    t_sim_list = []
    v_sim_list = []

    for sweep_idx in range(25):
        steady_state.restore()

        stim = h.IClamp(soma(0.5))
        stim.delay = config_bpop.stim_delay_ms
        stim.dur = config_bpop.stim_dur_ms

        stim_current_pA = -100 + sweep_idx * 20  # pA
        stim_current_nA = stim_current_pA / 1000.0  # nA
        stim.amp = stim_current_nA

        # 📌 Setup recording here INSIDE the loop
        t_vec = h.Vector().record(h._ref_t)
        v_vec = h.Vector().record(soma(0.5)._ref_v)

        h.t = 0
        h.frecord_init()
        h.tstop = stim.delay + stim.dur

        h.continuerun(510)  # Always run a bit longer to settle

        # Convert and store
        t_sim = np.array(t_vec.to_python())
        v_sim = np.array(v_vec.to_python())

        t_sim_list.append(t_sim.copy())
        v_sim_list.append(v_sim.copy())

    return t_sim_list, v_sim_list



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
    weights = {'peak': 10, 'amp': 10, 'width': 1, 'threshold': 1, 'latency':1, 'AHP':1}

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
    if rest > -70 or rest < -80:
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

    # 🔥 INSERT: only use stim window for point-to-point RMSE
    stim_start = 0
    stim_end = 300
    mask = (t_exp >= stim_start) & (t_exp <= stim_end)

    t_exp_fit = t_exp[mask]
    v_exp_fit = v_exp[mask]
    v_sim_fit = v_interp[mask]

    if len(t_exp_fit) < 2:
        return 1e6  # Penalty if no points to compare

    # ✅ Now calculate RMSE ONLY during stimulation
    rmse = np.sqrt(np.mean((v_sim_fit - v_exp_fit)**2))

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

    alpha = 1
    beta = 1

    total_cost = alpha * rmse + beta * f_cost + time_error + dvdt_error + penalty
    return total_cost

def cost_function_all(params, soma, axon, dend, t_exp_list, v_exp_list):
    """Sum cost over all sweeps."""
    total_cost = 0
    t_sim_list, v_sim_list = run_simulation(soma, axon, dend, params)

    if t_sim_list is None or v_sim_list is None:
        return 1e6  # Apply a huge penalty if relaxation failed

    for t_exp, v_exp, t_sim, v_sim in zip(t_exp_list, v_exp_list, t_sim_list, v_sim_list):
        if len(t_sim) == 0 or len(v_sim) == 0:
            continue  # Skip empty simulations to avoid interpolation error


        v_interp = interpolate_simulation(t_sim, v_sim, t_exp)

        # 🔥 INSERT: Select only stimulation window for point-to-point
        stim_start = 0
        stim_end = 300
        mask = (t_exp >= stim_start) & (t_exp <= stim_end)

        t_exp_fit = t_exp[mask]
        v_exp_fit = v_exp[mask]
        v_sim_fit = v_interp[mask]

        if len(t_exp_fit) < 2:
            continue  # Skip if no valid points

        rmse = np.sqrt(np.mean((v_sim_fit - v_exp_fit)**2))

        # Same AP window extraction
        features_exp = extract_features(v_exp, t_exp)
        ap_tmin = features_exp['latency']
        ap_tmax = t_exp[-1]

        if not np.isnan(features_exp['AHP']):
            try:
                ahp_idx = np.where(v_exp == features_exp['AHP'])[0][0]
                ap_tmax = t_exp[ahp_idx]
            except IndexError:
                pass
        if np.isnan(ap_tmin) or ap_tmin >= ap_tmax:
            continue

        ap_mask = (t_exp >= ap_tmin) & (t_exp <= ap_tmax)
        if np.sum(ap_mask) < 2:
            continue

        t_ap = t_exp[ap_mask]
        v_exp_ap = v_exp[ap_mask]
        v_sim_ap = v_interp[ap_mask]

        max_dvdt_exp = max_dvdt(v_exp_ap, t_ap)
        max_dvdt_sim = max_dvdt(v_sim_ap, t_ap)
        dvdt_error = 3.0 * (max_dvdt_sim - max_dvdt_exp) ** 2

        dt = t_ap[1] - t_ap[0]
        time_shift = abs(np.argmax(v_sim_ap) - np.argmax(v_exp_ap)) * dt
        time_error = 5.0 * time_shift

        rmse = np.sqrt(np.mean((v_sim_ap - v_exp_ap) ** 2))
        f_cost = feature_cost(v_sim_ap, v_exp_ap, t_ap)

        alpha = 1
        beta = 1

        this_cost = alpha * rmse + beta * f_cost + time_error + dvdt_error

        total_cost += this_cost
    if total_cost == 0:
        return 1e6  # Huge penalty if no sweep was valid
    return total_cost

