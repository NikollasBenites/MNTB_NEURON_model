from neuron import h, rxd
from neuron.units import ms, mV, µm
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from input_resistance_calculator import avg_soma_values
h.load_file("stdrun.hoc")

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))
# Change the working directory to the script's directory
os.chdir(script_directory)
print("Current working directory:", os.getcwd())

def nstomho(x):
    return (1e-9 * x/somaarea)

totalcap = 30 #Total membrane capacitance in pF for the cell (input capacitance)
somaarea = (totalcap * 1e-6)/1 #pf -> uF,assumes 1 uF/cm2; result is in cm2
#lstd = 1e4 * (np.sqrt(somaarea/np.pi)) #convert from cm to um

#### variables that will be used in model
#file: int = 2
###
revleak: int = -70
v_rest = 0 #set 0 do not initialize the model with v_rest
###
revk: int = -80
revna: int = 50
reveh = -45
###
leakg: int = 3      #3
nag: int = 210      #210
kltg: int = 5       #5
khtg: int = 150     #150
ihg: int = 35       #35
kag: int = 3        #3

amps = np.round(np.arange(-0.1,0.32,0.020),3) #stimulus (first, last, step) in nA


stimdelay: int = 100
stimdur: int = 300

filename: str = f'Ek_{revk}_L{leakg}_Na{nag}_IH{ihg}_KLT{kltg}_KHT{khtg}_KA_{kag}.png' #change the name according

#### variables to turn on (1) and off (something) blocks of code
annotation: int = 1     #annotate the variables in the fig 1
currentclamp: int = 1   #current stim -
vclamp: int = 0         #do voltage clamp (work in progress)
insetccfig: int = 1     #cc stim inset in the fig1
apcountPython: int = 1        #use voltage threshold and python aproach - depends currentclamp = 1
apcountNEURON: int = 1         #use net con from NEURON - depends currentclamp = 1

plottracesfig: int = 1  #plot traces fig
plotstimfig: int = 1    #plot the CC stim in different figure
plot_Rin_vs_current = 0 #plot the input resistance vs current graph
plotapcountpy: int = 1
plotapcountN: int = 1

savetracesfile: int = 0       #save the model figure file
savestimfile: int = 0   #save the stim figure file

####
####
class MNTB:
    def __init__(self, gid):
        self._gid = gid
        self._setup_morphology()
        self._setup_biophysics()


    #def_temperature(self):
    #    self.h.celsius = 35
    def _setup_morphology(self):
        self.soma = h.Section(name = 'soma', cell = self)
        self.soma.L = 20
        self.soma.diam = 20
    def _setup_biophysics(self ):
        # self.h.celsius = 35
        self.soma.Ra = 150 #axcn axial resistance (Ohm/cm^2)
        self.soma.cm = 1 #Membrane capacitance
        #self.soma.v = -70

        self.soma.insert('leak') #g = .001	(mho/cm2)
        self.soma.insert('NaCh') #gbar = .05 (S/cm2)
        self.soma.insert('IH') #ghbar = 0.0037 (mho/cm2)
        self.soma.insert('LT') #gbar = .002 (S/cm2)
        self.soma.insert('HT') #gbar = .015 (S/cm2)
        self.soma.insert('ka') #gkabar 0.00477

        for seg in self.soma:
        #
            seg.leak.g = nstomho(leakg)
            seg.leak.erev = revleak
        # # #
            seg.NaCh.gnabar = nstomho(nag)
            seg.ena = revna
        # # #
            seg.IH.ghbar = nstomho(ihg)
            seg.IH.eh = -45
        # # #
            seg.LT.gkltbar = nstomho(kltg)

        # # #
            seg.HT.gkhtbar = nstomho(khtg)

        # # #
            seg.ka.gkabar = nstomho(kag)
            seg.ek = revk
    def __repr__(self):
        return 'MNTB [{}]'.format(self._gid)
my_cell = MNTB(0)

#CURRENT CLAMP
stim = h.IClamp(my_cell.soma(0.5))
stim_traces = h.Vector().record(stim._ref_i)
soma_v = h.Vector().record(my_cell.soma(0.5)._ref_v)
t = h.Vector().record(h._ref_t)

ax1: object
fig1, ax1 = plt.subplots()

if insetccfig == 1:
    axin = ax1.inset_axes([0.6, 0.1, 0.2, 0.2]) # Create inset of current stimulation in the voltage plot
    ax1.set_xlabel('t (ms)')
    ax1.set_ylabel('v (mV)')
    #axin.set_xlabel('t (ms)')
    axin.set_ylabel('I (nA)')
    axin.grid(False)

if plotstimfig == 1:
    ax2: object
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('t (ms)')
    ax2.set_ylabel('I (nA)')

# average_soma_values = np.array([])

if currentclamp == 1:
    for amp in amps:
        if v_rest != 0:
            h.finitialize(v_rest)  # Reinitialize for each amplitude
        else:
            h.finitialize()
        stim.amp = amp
        stim.delay = stimdelay
        stim.dur = stimdur
        h.continuerun(1000)  # Run the simulation for 1000 ms
        h.tstop = 1000
        stim_values = np.array(stim_traces.to_python())
        soma_values = np.array(soma_v.to_python())  # Extract recorded soma voltage as a numpy array
        t_values = np.array(t.to_python())
        average_soma_values = avg_soma_values(t_values, stimdelay, stimdur, soma_values, amp)
        ax1.plot(t_values, soma_values, color='red', linewidth=0.5)
        if insetccfig == 1:
            axin.plot(t_values, stim_values, color='black', linewidth=0.5)
        if plotstimfig == 1:  ########### to do another fig with the stim
            ax2.plot(t_values, stim_values, color='black', linewidth=0.5)
            ax2.plot(t_values, stim_values, color='black', linewidth=0.5)
            ax2.spines['right'].set_color('black')
            ax2.spines['left'].set_color('black')
            ax2.yaxis.label.set_color('black')
            ax2.tick_params(axis='y', colors='black')
# slopes = []
# for i in range(1, len(amps)):
#     delta_amp = amps[i] - amps[i - 1]
#     delta_soma = average_soma_values[i] - average_soma_values[i - 1]
#     slope = np.round((delta_soma / delta_amp) / 1000, 3)
#     slopes.append(slope)

# Find the slope between the currents you want (default: -0.02 and 0)
# slope_range_index = np.where((amps[:-1] == -0.02) & (amps[1:] == 0))[0]
# if len(slope_range_index) > 0:
#     input_resistance = slopes[slope_range_index[0]]
# else:
#     input_resistance = None


# Arguments: text, xy (point to annotate), xytext (position of the text)
if annotation == 1:
    annotation_text = \
f"""
gLeak: {leakg}nS
gNa: {nag}nS
gIH: {ihg}nS
gKLT: {kltg}nS
gKHT: {khtg}nS
gKA: {kag}nS
ELeak: {revleak}mV
Ek: {revk}mV
ENa: {revna}mV"""
    ax1.annotate(
        annotation_text,
        xy=(600, -80),                 # Point to annotate (x, y)
        xytext=(600, -50),             # Position of the text (x, y)
        #arrowprops=dict(facecolor='black', shrink=0.05),  # Arrow style
        fontsize = 10,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightyellow")  #Add a box around the text
    )
# Print the average soma values for each amplitude and the slopes
for amp, avg in zip(amps, average_soma_values):
    print(f"Amplitude: {amp}, Average Soma Value: {avg}")

# print("Slopes between consecutive amps and avg_soma_values:")
# for i, slope in enumerate(slopes):
#     print(f"Slope between amps {amps[i]} and {amps[i+1]}: {slope} GOhms")

if plot_Rin_vs_current == 1:
    amps_mid_points = (amps[:-1] + amps[1:]) / 2

    # Plot the slopes against the mid-points of amps
    ax3: object
    fig3, ax3 = plt.subplots()
    plt.plot(amps_mid_points, slopes, marker='o', linestyle='-', color='k')
    plt.xlabel('Amplitude (nA)')
    plt.ylabel('Input Resistance (GΩ)')
    plt.title('Input Resistance vs Amplitude')
    plt.grid(True)

#######Save the plot to a file
if savetracesfile == 1:
    file_path = fr'C:\Users\nikol\PycharmProjects\MNTB_neuron\MNTB_Neuron_images\Experiments_Ek_Changes\{filename}'
    fig1.savefig(file_path, dpi=1200, bbox_inches='tight')
if savestimfile == 1:
    file_path = fr'C:\Users\nikol\PycharmProjects\MNTB_neuron\MNTB_Neuron_images\Experiments_Ek_Changes\Stim_{filename}'
    fig2.savefig(file_path, dpi=1200, bbox_inches='tight')

# if vclamp == 1: #work in progress
#     h.finitialize(-50 * mV)  # Reinitialize for each amplitude
#     stim.amp = 0
#     stim.delay = 50
#     stim.dur = 200
#     h.continuerun(250 * ms)  # Run the simulation for 25 ms
#
#     soma_values = np.array(soma_v.to_python())  # Extract recorded soma voltage as a numpy array
#     t_values = np.array(t.to_python())
#
#     ax2.plot(stim_values, color='black', linewidth=1)

#########################
if apcountPython == 1:
    ap_counts_py = []
    # Define a function to count spikes based on a threshold
    def count_spikes(v_trace, threshold=-10):
        spikes = np.where((v_trace[:-1] < threshold) & (v_trace[1:] >= threshold))[0]
        return len(spikes)

# Plot the results
if plotapcountpy == 1:
    ax4: object
    fig4, ax4 = plt.subplots()

    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')

    for amp in amps:
        if v_rest != 0:
            h.finitialize(v_rest)  # Reinitialize for each amplitude
        else:
            h.finitialize()  # Reinitialize for each amplitude

        stim.amp = amp
        h.continuerun(1000)  # Run the simulation for 1000 ms

        # Count spikes in the voltage trace
        soma_values_apcpy = np.array(soma_v.to_python())
        t_values_apcpy = np.array(t.to_python())
        num_spikes_py = count_spikes(soma_values_apcpy)
        ap_counts_py.append(num_spikes_py)
        #plot
        ax4.plot(t_values_apcpy, soma_values_apcpy, label=f'Stim: {amp} nA')



    # Display AP counts for each amplitude
    for amp, count in zip(amps, ap_counts_py):
        print(f"Amplitude: {amp} nA, AP Count py.based: {count}")





    # Convert the AP counts to numpy array for further analysis if needed
    ap_counts_array = np.array(ap_counts_py)

if apcountNEURON == 1:
    #Arrays to store AP counts and times

    ap_counts = []
    ap_times = []
    trace_data = []


    # Create a NetCon object to detect spikes
    netcon = h.NetCon(my_cell.soma(0.5)._ref_v, None, sec=my_cell.soma)
    netcon.threshold = -10  # Set the threshold for spike detection

    # List to store spike times
    spike_times = h.Vector()
    netcon.record(spike_times)

    first_trace_detected = False
    first_trace_data = None
    plt.figure()
    for amp in amps:
        if v_rest != 0:
            h.finitialize(v_rest)  # Reinitialize for each amplitude
        else:
            h.finitialize()  # Reinitialize for each amplitude

        stim.amp = amp
        h.continuerun(1000)  # Run the simulation for 1000 ms
        h.tstop = 1000
        # Count spikes detected by NetCon
        num_spikes = sum(stimdelay <= time <= stimdelay + stimdur for time in spike_times)
        ap_counts.append(num_spikes)
        ap_times.append(list(spike_times))

        # Extract recorded data
        soma_values_apc = np.array(soma_v.to_python())
        t_values_apc = np.array(t.to_python())
        trace_data.append((t_values_apc, soma_values_apc,amp,num_spikes))

        # Store the first trace with an AP
        if not first_trace_detected and num_spikes > 0:
            first_trace_data = (t_values_apc, soma_values_apc,amp)
            first_trace_detected = True
        # Clear spike times for the next run
        spike_times.clear()

    # Plot all red traces first

    for t_values_apc, soma_values_apc, amp, num_spikes in trace_data:
        if num_spikes == 0 or (first_trace_data is not None and (t_values_apc == first_trace_data[0]).all()):
            plt.plot(t_values_apc, soma_values_apc, color='red', linewidth=0.5)

    # Plot the first trace with an AP in black
    if first_trace_data is not None:
        t_values_apc, soma_values_apc, amp = first_trace_data
        plt.plot(t_values_apc, soma_values_apc, color='black', label=f'Rheobase {amp*1000} pA', linewidth=0.5)

    # Display AP counts and times for each amplitude
    for amp, count in zip(amps, ap_counts):
        print(f"Amplitude: {amp} nA, AP Count NetCon: {count}")

    # Plot the results
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.legend()


    # Convert the AP counts and times to numpy arrays for further analysis if needed
    ap_counts_array = np.array(ap_counts)
    ap_times_array = np.array(ap_times, dtype=object)

    # Plot number of APs vs. stimulus amplitude
    if plotapcountN == 1:
        ax6: object
        fig6, ax6 = plt.subplots()
        plt.plot(amps*1000, ap_counts, marker='o', linestyle='-', color='b')

        # Set the x-axis tick distance

        ax6.xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', steps=[1, 2, 5, 10]))
        ticks = np.arange(min(amps*1000), max(amps*1000) + 20, 20)
        ax6.set_xticks(ticks)

        # Ensure grid lines follow the ticks
        ax6.grid(which='both', axis='both', linestyle='-', linewidth=0.5, c ='lightgrey')


        plt.xlabel('Stimulus Amplitude (pA)')
        plt.ylabel('Number of APs')
        plt.title('Number of APs vs. Stimulus Amplitude')
        plt.grid(True)

plt.show()