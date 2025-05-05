from neuron import h
h.load_file("stdrun.hoc")
def nstomho(x, somaarea):
    return (1e-9 * x / somaarea)

class MNTB:
    def __init__(self, gid, somaarea, revleak, leakg, revna, nag,kag,ihg, kltg, khtg, revk):
        self._gid = gid
        self.somaarea = somaarea
        self.revleak = revleak
        self.leakg = leakg
        self.revna = revna
        self.nag = nag
        self.ihg = ihg
        self.kltg = kltg
        self.khtg = khtg
        self.kag = kag
        self.revk = revk
        self._setup_morphology()
        self._setup_biophysics()

    def _setup_morphology(self):
        self.soma = h.Section(name='soma', cell=self)
        self.soma.L = 20
        self.soma.diam = 15

    def _setup_biophysics(self):
        self.soma.Ra = 150
        self.soma.cm = 1
        self.soma.insert('leak')
        self.soma.insert('NaCh_nmb')
        self.soma.insert('IH_dth')
        self.soma.insert('LT_dth')
        self.soma.insert('HT_dth_nmb')
        self.soma.insert('ka')

        for seg in self.soma:
            seg.leak.g = nstomho(self.leakg, self.somaarea)
            seg.leak.erev = self.revleak
            seg.NaCh_nmb.gnabar = nstomho(self.nag, self.somaarea)
            seg.ena = self.revna
            seg.IH_dth.ghbar = nstomho(self.ihg, self.somaarea)
            #seg.IH.eh = -45
            seg.LT_dth.gkltbar = nstomho(self.kltg, self.somaarea)
            seg.HT_dth_nmb.gkhtbar = nstomho(self.khtg, self.somaarea)
            seg.ka.gkabar = nstomho(self.kag, self.somaarea)
            seg.ek = self.revk

    def __repr__(self):
        return 'MNTB [{}]'.format(self._gid)
