from neuron import h
h.load_file("stdrun.hoc")
from neuron.units import ms, mV, µm
import numpy as np
import matplotlib.pyplot as plt


totalcap = 20 #Total membrane capacitance in pF for the cell
somaarea = (totalcap * 1e-6)/1 #pf -> uF,assumes 1 uF/cm2; result is in cm2
#lstd = 1e4 * (np.sqrt(somaarea/np.pi)) #convert from cm to um

def nstomho(x):
    return (1e-9 * x/somaarea)

# create model
soma = h.Section(name='soma')
soma.L = 15      #Length of soma (um)
soma.diam = 15   #Diameter of soma (um)

#soma.insert('pas')
soma.Ra = 150            #Membrane axial resistance (Ohm/cm^2)
soma.cm = 1             #Membrane capacitance
#soma.g_pas = 1.0/25370.0   #Leak maximal conductance

soma.v = -70
soma.insert('leak')   # add passive properties
soma.g_leak = nstomho(12.2) # set the specific membrane resistance to 10000 ohm*cm^2
soma.erev_leak = -73.0

# add active conductances (the channels [mod files] are from Mainen and Sejnowski 1996)
soma.insert('HT') # add potassium channel
soma.gkhtbar_HT = nstomho(300)# set the Kv3 potassium conductance. Units are microSiemens
soma.ek = -106.8

soma.insert('LT') # add potassium channel
#soma.gkltbar_LT = nstomho(0) # set the Kv1 potassium conductance
soma.gkltbar_LT_dth = nstomho(36.28) # set the Kv1 potassium conductance
soma.ek = -106.8

soma.insert('NaCh') # add sodium channel
soma.gnabar_NaCh_nmb = nstomho(300) # set the sodium conductance
soma.ena = 62.77

soma.insert('IH') # add HCN channel
soma.ghbar_IH_dth = nstomho(32.29) # set Ih conductance
#soma.eh = -45

#Create Current-Clamp
st = h.IClamp(0.5)  # Choose where in the soma to point-stimulate

st.dur = 300  # Stimulus duration (ms)
st.delay = 10  # Stimulus delay (ms)
st.amp = 0  # Stimulus amplitude (nA)
h.tstop = 1000  # stop the simulation (ms)

v_vec = h.Vector()   #Membrane potential vector
t_vec = h.Vector()   #Time stamp vector
stim_current = h.Vector()

v_vec.record(soma(0.5)._ref_v)  #Recording from the soma the desired quantities
t_vec.record(h._ref_t)
stim_current.record(st._ref_i)

h.v_init = -70  # Set initializing simulation voltage (mV) at t0
h.finitialize(-70)  # Set initializing voltage for all mechanisms in the section
h.run()

plt.figure(figsize=(12, 7))
plt.plot(t_vec, v_vec, 'k')
plt.ylabel('Membrane voltage (mV)', fontsize=16)
plt.xlabel('Time (ms)', fontsize=16)


for i in np.arange(-0.1, 0.32, 0.02):
    st.amp = i
    h.tstop = 1000  # set the simulation time
    h.v_init = -70  # Set initializing simulation voltage (mV) at t0
    h.finitialize(-70)  # Set initializing voltage for all mechanisms in the section
    h.run()
    plt.plot(t_vec, v_vec, 'k')

plt.show()

plt.figure(figsize=(12, 7))
plt.plot(t_vec, stim_current, 'k')
plt.ylabel('Current (nA)', fontsize=16)
plt.xlabel('Time (ms)', fontsize=16)


for i in np.arange(-0.1, 0.2, 0.02):
    st.amp = i
    h.tstop = 510  # set the simulation time
    h.v_init = -70  # Set initializing simulation voltage (mV) at t0
    h.finitialize(-70)  # Set initializing voltage for all mechanisms in the section
    h.run()
    plt.plot(t_vec, stim_current, 'k')

plt.show()