import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from neuron import h
h.load_file('stdrun.hoc')

# Define soma parameters
totalcap = 20  # Total membrane capacitance in pF
somaarea = (totalcap * 1e-6) / 1  # Convert to cm^2 assuming 1 µF/cm²


def nstomho(x):
    return (1e-9 * x / somaarea)  # Convert conductance to mho/cm²

# Load your experimental data here
t_exp = np.load("t_exp.npy")  # in ms, sampled at 50 kHz
t_exp = t_exp - t_exp[0]  # ensure starts at 0
V_exp = np.load("V_exp.npy")  # in mV

# Create soma section
soma = h.Section(name='soma')
soma.L = 15  # Length in µm
soma.diam = 15  # Diameter in µm
soma.Ra = 150  # Axial resistance (Ohm*cm)
soma.cm = 1  # Membrane capacitance (µF/cm²)
soma.v = -70  # Initial membrane potential (mV)

# Insert channels to fit in the simulation
soma.insert('leak')
soma.insert('LT')  # Kv1 Potassium channel
soma.insert('IH')  # HCN channel

# Insert active conductances (Mainen & Sejnowski 1996)
soma.insert('HT')  # Kv3 Potassium channel
soma.gkhtbar_HT = nstomho(300)

soma.insert('NaCh')  # Sodium channel
soma.gnabar_NaCh = nstomho(300)

soma.ek = -106.8
soma.ena = 62.77

def set_conductances(gna, gkht):
    soma.gnabar_NaCh = nstomho(gna)
    soma.gkhtbar_HT = nstomho(gkht)

def run_simulation(gna, gkht, stim_amp=0.2, stim_dur=2.0):
    set_conductances(gna, gkht)

    stim = h.IClamp(soma(0.5))
    stim.delay = 1
    stim.dur = stim_dur
    stim.amp = stim_amp

    h.dt = 0.02
    h.steps_per_ms = int(1.0 / h.dt)
    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(soma(0.5)._ref_v)

    h.finitialize(-70)
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
    gna, gkht = params
    t_sim, v_sim = run_simulation(gna, gkht)
    v_interp = interpolate_simulation(t_sim, v_sim, t_exp)
    mse = np.mean((v_interp - V_exp)**2)
    penalty = penalty_terms(v_interp)
    return mse + penalty

# Initial guess and bounds
x0 = [0.1, 0.05]  # gNa, gKHT
bounds = [(1e-4, 1.0), (1e-4, 1.0)]

result = minimize(cost_function, x0, bounds=bounds, method='L-BFGS-B', options={'maxiter': 200})
opt_gna, opt_gkht = result.x
print(f"Optimal gNa: {opt_gna:.4f} S/cm^2, Optimal gKHT: {opt_gkht:.4f} S/cm^2")

# Final simulation and plot
t_sim, v_sim = run_simulation(opt_gna, opt_gkht)

plt.figure(figsize=(10, 5))
plt.plot(t_exp, V_exp, label='Experimental', linewidth=2)
plt.plot(t_sim, v_sim, label='Simulated (fit)', linestyle='--')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.title('Action Potential Fit')
plt.tight_layout()
plt.show()
