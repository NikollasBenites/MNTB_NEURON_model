import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from neuron import h
import MNTB_PN_myFunctions as mFun
h.load_file('stdrun.hoc')
# Define soma parameters
totalcap = 20  # Total membrane capacitance in pF
somaarea = (totalcap * 1e-6) / 1  # Convert to cm^2 assuming 1 µF/cm²


def nstomho(x):
    return (1e-9 * x / somaarea)  # Convert conductance to mho/cm²

experimentalTrace = np.genfromtxt('../P9_iMNTB_Rheobase.csv', delimiter=',', skip_header=1, dtype=float, filling_values=np.nan)
# Load your experimental data here

#t_exp = np.load("t_exp.npy")
#V_exp = np.load("V_exp.npy")


t_exp = experimentalTrace[:,0]*1000 # in ms, sampled at 50 kHz
t_exp = t_exp - t_exp[0]  # ensure starts at 0
V_exp = experimentalTrace[:,1]  # in mV

# Create soma section
soma = h.Section(name='soma')
soma.L = 15  # Length in µm
soma.diam = 15  # Diameter in µm
soma.Ra = 150  # Axial resistance (Ohm*cm)
soma.cm = 1  # Membrane capacitance (µF/cm²)
soma.v = -77  # Initial membrane potential (mV)

# Insert channels to fit in the simulation
soma.insert('leak')

soma.insert('LT')  # Kv1 Potassium channel

soma.insert('IH')  # HCN channel

# Insert active conductances (Mainen & Sejnowski 1996)
soma.insert('HT')  # Kv3 Potassium channel


soma.insert('NaCh')  # Sodium channel
#soma.gnabar_NaCh = nstomho(300)

soma.ek = -106.8
soma.ena = 62.77
erev = -77
#v_init = -70
def set_conductances(gna, gkht,gklt,gh,erev):
    #v_init = mFun.custom_init(v_init)
    soma.gnabar_NaCh_nmb = nstomho(gna)
    soma.gkhtbar_HT = nstomho(gkht)
    soma.gkltbar_LT = nstomho(gklt)
    soma.ghbar_IH = nstomho(gh)
    soma.erev_leak = erev

def extract_features(trace, time):
    peak = np.max(trace)
    peak_time = time[np.argmax(trace)]
    rest = trace[0]
    amp = peak - rest
    width = np.sum(trace > rest + 0.5 * amp) * (time[1] - time[0])  # rough FWHM
    return {'peak': peak, 'amp': amp, 'width': width}

def feature_cost(sim_trace, exp_trace, time):
    sim_feat = extract_features(sim_trace, time)
    exp_feat = extract_features(exp_trace, time)
    error = 0
    for k in sim_feat:
        error += ((sim_feat[k] - exp_feat[k]) ** 2)
    return error

def run_simulation(gna, gkht, gklt, gh, stim_amp=0.320, stim_dur=20):
    #v_init = mFun.custom_init(v_init)
    set_conductances(gna, gkht, gklt, gh, erev)

    stim = h.IClamp(soma(0.5))
    stim.delay = 10
    stim.dur = stim_dur
    stim.amp = stim_amp

    h.dt = 0.02
    h.steps_per_ms = int(1.0 / h.dt)
    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(soma(0.5)._ref_v)

    h.finitialize(-77)
    h.continuerun(t_exp[-1] + 1)

    return np.array(t_vec), np.array(v_vec)

def interpolate_simulation(t_neuron, v_neuron, t_exp):
    interp_func = interp1d(t_neuron, v_neuron, kind='cubic', fill_value='extrapolate')
    return interp_func(t_exp)

def penalty_terms(v_sim):
    peak = np.max(v_sim)
    rest = v_sim[0]
    penalty = 0
    if peak < 0 or peak > 60:
        penalty += 1000
    if rest > -40 or rest < -80:
        penalty += 1000
    return penalty

def cost_function(params):
    gna, gkht, gklt, gh= params
    t_sim, v_sim = run_simulation(gna, gkht, gklt, gh)
    v_interp = interpolate_simulation(t_sim, v_sim, t_exp)

    mse = np.mean((v_interp - V_exp)**2)
    f_cost = feature_cost(v_interp, V_exp, t_exp)
    penalty = penalty_terms(v_interp)

    alpha = 1.0  # weight for MSE
    beta = 0.5   # weight for feature cost

    return alpha * mse + beta * f_cost + penalty


# Initial guess and bounds
x0 = [200, 200, 50, 30]  # gNa, gKHT, gKLT, gH, erev
bounds = [(1e-4, 400), (1e-4, 400),(1e-4,100),(1e-4,60)]

result = minimize(cost_function, x0, bounds=bounds, method='L-BFGS-B', options={'maxiter': 200})
opt_gna, opt_gkht, opt_gklt, opt_gh = result.x
print(f"Optimal gNa: {opt_gna:.2f} , Optimal gKHT: {opt_gkht:.2f}, Optimal gKLT: {opt_gklt: .2f}, Optimal gH {opt_gh:.2f}, Set erev: {erev: .2f}")

# Final simulation and plot
t_sim, v_sim = run_simulation(opt_gna, opt_gkht, opt_gklt, opt_gh)

plt.figure(figsize=(10, 5))
plt.plot(t_exp, V_exp, label='Experimental', linewidth=2)
plt.plot(t_sim, v_sim, label='Simulated (fit)', linestyle='--')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.title('Action Potential Fit')
plt.tight_layout()
plt.show()
