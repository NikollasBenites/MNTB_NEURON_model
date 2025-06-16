# neuron_model.py

import numpy as np
from neuron import h
import MNTB_PN_myFunctions as mFun
import config_noKinetics


h.load_file('stdrun.hoc')

def create_neuron():
    soma = h.Section(name='soma')
    axon = h.Section(name='axon')
    dend = h.Section(name='dend')

    soma.diam = 15.5
    soma.Ra = 150
    soma.cm = 1
    soma.insert('leak')
    soma.insert('IH_dth')
    soma.insert('HT_dth')
    soma.ek = config_noKinetics.ek
    soma.v = config_noKinetics.v_init

    axon.L = 25
    axon.diam = 3
    axon.Ra = 100
    axon.cm = 1
    axon.insert('leak')
    axon.insert('NaCh_dth')
    axon.insert('LT_dth')
    axon.ek = config_noKinetics.ek
    axon.ena = config_noKinetics.ena
    axon.v = config_noKinetics.v_init

    dend.L = 80
    dend.diam = 3
    dend.Ra = 100
    dend.cm = 1
    dend.insert('leak')
    dend.insert('IH_dth')
    dend.v = config_noKinetics.v_init

    axon.connect(soma(1))
    dend.connect(soma(0))

    return soma, axon, dend

def nstomho(x, area):
    return (1e-9 * x / area)

def set_conductances(soma, axon, dend, neuron_params, na_scale, kht_scale, klt_scale, ih_soma,ih_dend, erev=config_noKinetics.erev):
    # Unpack parameters
    (gna, gkht, gklt,gh,gleak) = neuron_params
     #na_scale, kht_scale) = params

    # Calculate areas
    totalcap = config_noKinetics.total_capacitance_pF
    soma_area = (totalcap * 1e-6) / 1  # cm², assuming 1 uF/cm²
    axon_area = np.pi * axon.diam * axon.L * 1e-8  # cm²

    # Helper functions
    def nstomho_soma(x):
        return (1e-9 * x / soma_area)

    def nstomho_axon(x):
        return (1e-9 * x / axon_area)

    # Set soma conductances
    soma.gkhtbar_HT_dth_nmb = nstomho_soma(gkht) * kht_scale
    soma.ghbar_IH_dth = nstomho_soma(gh)*ih_soma
    soma.erev_leak = erev
    soma.g_leak = nstomho_soma(gleak)

    # Set axon conductances
    axon.gnabar_NaCh_dth = nstomho_axon(gna)*na_scale
    axon.gkltbar_LT_dth = nstomho_axon(gklt)*klt_scale
    axon.erev_leak = erev
    axon.g_leak = nstomho_axon(gleak)

    # Set dendrite conductances
    for seg in dend:
        seg.g_leak = nstomho_soma(gleak)
        seg.erev_leak = erev
        seg.ghbar_IH_dth = nstomho_soma(gh)*ih_dend
