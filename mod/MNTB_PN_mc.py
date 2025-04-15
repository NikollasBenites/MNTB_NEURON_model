from neuron import h
import numpy as np
h.load_file("stdrun.hoc")

def nstomho(x, area):
    return (1e-9 * x / area)

class PN:
    def __init__(self, gid, somaarea, AISarea, dendarea, erev, ena, ek, leakg, gna, gh, gklt, gkht):
        self._gid = gid
        self.somaarea = somaarea
        self.AISarea = AISarea
        self.dendarea = dendarea

        self.erev = erev
        self.ena = ena
        self.ek = ek

        self.leakg = leakg
        self.gna = gna
        self.gh = gh
        self.kltg = gklt
        self.khtg = gkht

        self._setup_morphology()
        self._setup_biophysics()

    def _setup_morphology(self):
        self.soma = h.Section(name='soma', cell=self)
        self.dend = h.Section(name='dend', cell=self)
        self.AIS = h.Section(name='AIS', cell=self)

        self.soma.diam = 20
        self.soma.L = 20

        self.dend.diam = 3
        self.dend.L = 80

        self.AIS.diam = 2
        self.AIS.L = 25
        self.AIS.nseg = 5

        self.AIS.connect(self.soma(1))
        self.dend.connect(self.soma(0))

    def _setup_biophysics(self):
        self.soma.Ra = 150
        self.soma.cm = 1
        self.soma.insert('leak')
        self.soma.insert('IH')
        self.soma.insert('HT')
        for seg in self.soma:
            seg.leak.g = nstomho((self.leakg*0.17), self.somaarea)
            seg.leak.erev = self.erev
            seg.IH.ghbar = nstomho((self.gh*0.08), self.somaarea)
            seg.HT.gkhtbar = nstomho((self.khtg*2.5), self.somaarea)
            seg.ek = self.ek

        self.dend.Ra = 100
        self.dend.cm = 1
        self.dend.insert('leak')
        self.dend.insert('IH')
        for seg in self.dend:
            seg.leak.g = nstomho((self.leakg*0.5), self.dendarea)
            seg.leak.erev = self.erev
            seg.IH.ghbar = nstomho((self.gh*0.16), self.dendarea)

        self.AIS.Ra = 100
        self.AIS.cm = 1
        self.AIS.insert('leak')
        self.AIS.insert('NaCh')
        self.AIS.insert('LT')
        for seg in self.AIS:
            seg.leak.g = nstomho((self.leakg*0.12), self.AISarea)
            seg.leak.erev = self.erev
            seg.NaCh.gnabar = nstomho((self.gna*5), self.AISarea)
            seg.LT.gkltbar = nstomho((self.kltg), self.AISarea)
            seg.ena = self.ena
            seg.ek = self.ek

    def __repr__(self):
        return 'PN [{}]'.format(self._gid)
