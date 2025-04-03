import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize, differential_evolution
from neuron import h
import MNTB_PN_myFunctions as mFun
import BestFit_Functions as bf
h.load_file('stdrun.hoc')

# Define soma parameters
totalcap = 20  # Total membrane capacitance in pF
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
#soma.cm = 1
v_init = -77

soma.insert('leak')
soma.insert('LT')
soma.insert('IH')
soma.insert('HT')
soma.insert('NaCh')

soma.ek = -106.8
soma.ena = 62.77
erev = -79
gklt = 161.1
gh = 18.87

# t_exp = experimentalTrace[499:,0]*1000 # in ms, sampled at 50 kHz
# t_exp = t_exp - t_exp[0]  # ensure starts at 0
# V_exp = experimentalTrace[499:,1]  # in mV

# Initial guess and bounds
bounds = [(100, 800), (100, 800), (gklt*0.75, gklt*1.25), (gh*0.5, gh*1.5),(0.1,3)]

result_global = differential_evolution(cost_function, bounds, strategy='best1bin', maxiter=20, popsize=10, polish=True)
result_local = minimize(cost_function, result_global.x, bounds=bounds, method='L-BFGS-B', options={'maxiter': 200})
opt_gna, opt_gkht, gklt, gh, opt_cm = result_local.x

# opt_gna, opt_gkht, gklt, gh, erev = result.x
print(f"Optimal gNa: {opt_gna:.2f} , Optimal gKHT: {opt_gkht:.2f}, Set gKLT: {gklt:.2f}, set gH: {gh:.2f}, Set erev: {erev:.2f}, Optimal cm: {opt_cm:.2f}")

# Final simulation and plot
t_sim, v_sim = bf.run_simulation(opt_gna, opt_gkht, gklt, gh, opt_cm)
feat_sim = bf.extract_features(v_sim, t_sim)
print("Simulate Features:")
for k, v in feat_sim.items():
    print(f"{k}: {v:.2f}")

feat_exp = bf.extract_features(V_exp, t_exp)
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
