import os

import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import find_peaks
from matplotlib.ticker import MaxNLocator
from neuron import h

import MNTB_PN_myFunctions as mFun
from MNTB_PN import MNTB


h.load_file("stdrun.hoc")

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))
# Change the working directory to the script's directory
os.chdir(script_directory)
print("Current working directory:", os.getcwd())

totalcap = 30  # Total membrane capacitance in pF for the cell (input capacitance)
somaarea = (totalcap * 1e-6) / 1  # pf -> uF,assumes 1 uF/cm2; result is in cm2
# lstd = 1e4 * (np.sqrt(somaarea/np.pi)) #convert from cm to um

################################################# variables that will be used in model

### reversal potentials
revleak: int = -70
revk: int = -80
revna: int = 50
reveh: int = -45

### AGE
age: int = 4

### Type of experiment
leak_exp: int = 0
Na_exp: int = 0
KLT_exp: int = 0
KHT_exp: int = 0
IH_exp: int = 0
KA_exp: int = 0
Sierksma_exp: int = 0

savetracesfile: int = 0  # save the simulation fig1 file
savestimfile: int = 0  # save the stim fig2 file

################################## channel conductances (Sierkisma P4 age is default) ##################################
leakg = 2.8         #2.8     Leak
nag: int = 210      #210     NaV
kltg: int = 20      #20      LVA
khtg: int = 80      #80      HVA
ihg: int = 0       #37      IH
kag: int = 20        #3       Kv A

############################################## stimulus amplitude ######################################################
amps = np.round(np.arange(-0.08, 0.2, 0.020), 3)  # stimulus (first, last, step) in nA
################################### setup the current-clamp stimulus protocol
stimdelay: int = 100
stimdur: int = 200
totalrun: int = 1000
v_init: int = -70  # if use with custom_init() the value is not considered, but must be close the expected rmp

################################### where to pick the values up the voltages traces to average
t_min = stimdelay + stimdur - 60
t_max = stimdelay + stimdur - 10
average_soma_values = np.array([])  # Array with the average_soma_values in different amplitudes

rmp = None  ######### resting membrane potential measure at steady state when amp = 0 pA

################################ variables to turn on (1) and off (0) blocks of code ###################################
annotation: int = 1  # annotate the variables in the fig 1
insetccfig: int = 1  # cc stim inset in the fig1
apcount: int = 1  # use netcon from NEURON to count APs - it depends currentclamp = 1

plottracesfig: int = 1  # plot traces fig1
plotstimfig: int = 0  # plot the CC stim in fig2
plot_Rin_vs_current = 1  # plot the input resistance vs current graph
plotapcountN: int = 1  # plot the AP counting vs current (NetCon approach)

AP_Rheo: int = 1
AP_Rheo_plot: int = 1

############################################# MNTB_PN file imported ####################################################
my_cell = MNTB(0, somaarea, revleak, leakg, revna, nag, ihg, kltg, khtg, kag, revk)

############################################### CURRENT CLAMP setup ####################################################
stim = h.IClamp(my_cell.soma(0.5))
stim_traces = h.Vector().record(stim._ref_i)
soma_v = h.Vector().record(my_cell.soma(0.5)._ref_v)
t = h.Vector().record(h._ref_t)

##################################### arrays to count APs and detect the rheobase ######################################
ap_counts = []
ap_times = []
trace_data_apc = []

########################################### NEURON approach to detect APs ##############################################
netcon = h.NetCon(my_cell.soma(0.5)._ref_v, None, sec=my_cell.soma)
netcon.threshold = -10  # Set the threshold for spike detection

### List to store spike times
spike_times = h.Vector()
netcon.record(spike_times)
num_spikes = []
first_trace_detected = False
first_trace_data = None

################################################## FIGURES 1 AND 2 #####################################################
### fig1 - the simulation
ax1: object
fig1, ax1 = plt.subplots()
#mngr = plt.get_current_fig_manager()
#mngr.window.setGeometry(50, 100, 640, 545)
### inset with the current stim in fig1
if insetccfig == 1:
    axin = ax1.inset_axes([0.6, 0.1, 0.2, 0.2])  # Create inset of current stimulation in the voltage plot
    ax1.set_xlabel('t (ms)')
    ax1.set_ylabel('v (mV)')
    # axin.set_xlabel('t (ms)')
    axin.set_ylabel('I (nA)')
    axin.grid(False)

### fig2 - the stimulus traces fig2
if plotstimfig == 1:
    ax2: object
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('t (ms)')
    ax2.set_ylabel('I (nA)')
#    mngr.window.setGeometry(700, 100, 640, 545)
############################################## current clamp simulation ################################################
for amp in amps:
    v_init = mFun.custom_init(v_init)  # default -70mV
    soma_values, stim_values, t_values = mFun.run_simulation(amp, stim, soma_v, t, totalrun, stimdelay, stimdur,
                                                             stim_traces)
    soma_values_range, t_values_range, average_soma_values = mFun.avg_ss_values(soma_values, t_values, t_min, t_max,
                                                                                average_soma_values)
    num_spikes,spike_times,ap_counts,ap_times = mFun.count_spikes(num_spikes, stimdelay, stimdur, spike_times,ap_counts,
                                                                  ap_times)

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

### pick the rmp value
for amp, avg in zip(amps, average_soma_values):
    if 0 == amp:
        rmp = avg

# Calculate the slope for each step in amps relative to the avg_soma_values
# average_soma_values_array = np.array(avg_soma_values)
slopes = np.array([])
for i in range(1, len(amps)):
    delta_amp = amps[i] - amps[i - 1]
    delta_soma = average_soma_values[i] - average_soma_values[i - 1]
    # delta_soma = average_soma_values_array[i] - average_soma_values_array[i-1]
    slope = np.round((delta_soma / delta_amp) / 1000, 3)
    slopes = np.append(slopes, slope)

# Find the slope between the currents you want (default: -0.02 and 0)
slope_range_index = np.where((amps[:-1] == -0.02) & (amps[1:] == 0))[0]
if len(slope_range_index) > 0:
    input_resistance = slopes[slope_range_index[0]]
else:
    input_resistance = None

############################# Arguments: text, xy (point to annotate), xytext (position of the text)
if annotation == 1:
    annotation_text = \
        f"""RMP: {rmp}mV
Rin: {input_resistance} GOhms
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
        xy=(600, -80),  # Point to annotate (x, y)
        xytext=(600, -50),  # Position of the text (x, y)
        # arrowprops=dict(facecolor='black', shrink=0.05),  # Arrow style
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightyellow")  # Add a box around the text
    )


############################### Print the average soma values for each amplitude and the slopes
for amp, avg in zip(amps, average_soma_values):
    print(f"Amplitude: {amp}, Average Soma Value: {avg}")

print("Slopes between consecutive amps and avg_soma_values:")
for i, slope in enumerate(slopes):
    print(f"Slope between amps {amps[i]} and {amps[i + 1]}: {slope} GOhms")

    amps_mid_points = (amps[:-1] + amps[1:]) / 2
#####################################################################################################################
if plot_Rin_vs_current == 1:
    # Plot the slopes against the mid-points of amps
    ax3: object
    fig3, ax3 = plt.subplots()
    plt.plot(amps_mid_points, slopes, marker='o', linestyle='-', color='k')
#    mngr.window.setGeometry(1350, 100, 640, 545)
    plt.xlabel('Current (nA)')
    plt.ylabel('Input Resistance (GΩ)')
    plt.title('Input Resistance vs Current Injection')
    plt.grid(True)
#####################################################################################################################
if apcount == 1:
    # Arrays to store AP counts and times
    ap_counts = []
    ap_times = []
    trace_data_apc = []
    # Create a NetCon object to detect spikes
    netcon = h.NetCon(my_cell.soma(0.5)._ref_v, None, sec=my_cell.soma)
    netcon.threshold = -10  # Set the threshold for spike detection

    # List to store spike times
    spike_times = h.Vector()
    netcon.record(spike_times)

    first_trace_detected = False
    first_trace_data = []
    plt.figure()

    for amp in amps:
        v_init = mFun.custom_init(v_init)
        soma_values_apc, t_values_apc = mFun.run_simulation(amp, stim, soma_v, t, totalrun, stimdelay, stimdur)

        # Count spikes detected by NetCon
        num_spikes = sum(stimdelay <= time <= stimdelay + stimdur for time in spike_times)
        ap_counts.append(num_spikes)
        ap_times.append(list(spike_times))

        # Extract recorded data
        soma_values_apc = np.array(soma_v.to_python())
        t_values_apc = np.array(t.to_python())
        trace_data_apc.append((t_values_apc, soma_values_apc, amp, num_spikes))

        # Store the first trace with an AP
        if not first_trace_detected and num_spikes > 0:
            first_trace_data = (t_values_apc, soma_values_apc, amp)
            first_trace_detected = True
            if AP_Rheo == 1:
               ap_data = mFun.analyze_AP(t_values_apc,soma_values_apc)

        # Clear spike times for the next run
        spike_times.clear()

    # Plot all red traces first

    for t_values_apc, soma_values_apc, amp, num_spikes in trace_data_apc:
        if num_spikes == 0 or (first_trace_data is not None and (t_values_apc == first_trace_data[0]).all()):
            plt.plot(t_values_apc, soma_values_apc, color='red', linewidth=0.5)
            #mngr.window.setGeometry(50, 700, 640, 545)
 ################################# Plot the first trace with an AP in black ############################################
    if first_trace_data is not None:
        t_values_apc, soma_values_apc, amp = first_trace_data
        plt.plot(t_values_apc, soma_values_apc, color='black', label=f'Rheobase {amp * 1000} pA', linewidth=0.5)
        #mngr.window.setGeometry(700, 700, 640, 545)
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

##################################### Plot number of APs vs. stimulus amplitude #######################################
    if plotapcountN == 1:
        ax4: object
        fig4, ax4 = plt.subplots()
        plt.plot(amps * 1000, ap_counts, marker='o', linestyle='-', color='b')
        #mngr.window.setGeometry(1350, 700, 640, 545)  # Sixth window (bottom-right)
        # Set the x-axis tick distance

        ax4.xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', steps=[1, 2, 5, 10]))
        ticks = np.arange(min(amps * 1000), max(amps * 1000) + 20, 20)
        ax4.set_xticks(ticks)

        # Ensure grid lines follow the ticks
        ax4.grid(which='both', axis='both', linestyle='-', linewidth=0.5, c='lightgrey')

        plt.xlabel('Stimulus Amplitude (pA)')
        plt.ylabel('Number of APs')
        plt.title('Number of APs vs. Stimulus Amplitude')
        plt.grid(True)

######################### AP Analysis #################################################################################
if AP_Rheo == 1:
    if ap_data:
        print("First AP Analysis:")
        for key, value in ap_data.items():
            print(f"{key}: {value:.2f}")
        if AP_Rheo_plot == 1:
            # Plot the trace with AP features
            plt.figure()
            #mngr.window.setGeometry(1350, 700, 640, 545)  # Sixth window (bottom-right)
            plt.plot(t_values_apc, soma_values_apc, label="Voltage Trace", color='black')
            plt.scatter(ap_data["spike time"], ap_data["peak"], color='red', label="Peak", zorder=3)
            plt.scatter(t_values_apc[np.where(soma_values_apc == ap_data["threshold"])[0][0]], ap_data["threshold"], color='blue',
                        label="Threshold", zorder=3)
            plt.axhline(ap_data["AHP"], color='purple', linestyle="--", label="AHP", alpha=0.7)

            plt.xlabel("Time (ms)")
            plt.ylabel("Voltage (mV)")
            plt.legend()
            plt.title("Action Potential Analysis")
    else:
        print("No AP detected in this trace.")

############################################## Save the plot to a file #################################################
if savetracesfile == 1:
    if leak_exp == 1:
        file_path = fr"C:\Users\nikol\PycharmProjects\MNTB_neuron\figures\Experiments_Leak_Changes\{filename}"
        fig1.savefig(file_path, dpi=1200, bbox_inches='tight')
    if Na_exp == 1:
        file_path = fr'C:\Users\nikol\PycharmProjects\MNTB_neuron\figures\Experiments_Na_Changes\{filename}'
        fig1.savefig(file_path, dpi=1200, bbox_inches='tight')
    if KLT_exp == 1:
        file_path = fr'C:\Users\nikol\PycharmProjects\MNTB_neuron\figures\Experiments_KLT_Changes\{filename}'
        fig1.savefig(file_path, dpi=1200, bbox_inches='tight')
    if KHT_exp == 1:
        file_path = fr'C:\Users\nikol\PycharmProjects\MNTB_neuron\figures\Experiments_KHT_Changes\{filename}'
        fig1.savefig(file_path, dpi=1200, bbox_inches='tight')
    if IH_exp == 1:
        file_path = fr'C:\Users\nikol\PycharmProjects\MNTB_neuron\figures\Experiments_IH_Changes\{filename}'
        fig1.savefig(file_path, dpi=1200, bbox_inches='tight')
    if KA_exp == 1:
        file_path = fr'C:\Users\nikol\PycharmProjects\MNTB_neuron\figures\Experiments_KA_Changes\{filename}'
        fig1.savefig(file_path, dpi=1200, bbox_inches='tight')
    if Sierksma_exp == 1:
        file_path = fr'C:\Users\nikol\PycharmProjects\MNTB_neuron\figures\Sierskma_Values\{filename}'
        fig1.savefig(file_path, dpi=1200, bbox_inches='tight')
if savestimfile == 1:
    file_path = fr'{file_path}\Stim_{filename}'
    fig2.savefig(file_path, dpi=1200, bbox_inches='tight')

#################################### filename with important values to save ############################################

if leak_exp == 0:
    filename: str = f'Leak_{ihg}_P{age}_Na{nag}_KLT{kltg}_KHT{khtg}_IH{ihg}_KA{kag}.png'
if Na_exp == 0:
    filename: str = f'Na_{ihg}_P{age}_L{leakg}_KLT{kltg}_KHT{khtg}_IH{ihg}_KA{kag}.png'
if KLT_exp == 0:
    filename: str = f'KLT_{ihg}_P{age}_L{leakg}_Na{nag}_KHT{khtg}_IH{ihg}_KA{kag}.png'
if KHT_exp == 0:
    filename: str = f'KHT_{ihg}_P{age}_L{leakg}_Na{nag}_KLT{kltg}_IH{ihg}_KA{kag}.png'
if IH_exp == 0:
    filename: str = f'IH_{ihg}_P{age}_L{leakg}_Na{nag}_KLT{kltg}_KHT{khtg}_KA{kag}.png'
if KA_exp == 0:
    filename: str = f'KA_{ihg}_P{age}_L{leakg}_Na{nag}_KLT{kltg}_KHT{khtg}_IH{kag}.png'
if Sierksma_exp == 0:
    filename = f'P{age}_L{leakg}_Na{nag}_KLT{kltg}_KHT{khtg}_IH{ihg}_KA_{kag}.png'

# Print results

plt.show()
