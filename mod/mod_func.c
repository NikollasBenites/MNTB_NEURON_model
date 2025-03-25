#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern void _Iclamp2_reg();
extern void _ih_reg();
extern void _ka_reg();
extern void _kht_reg();
extern void _klt_reg();
extern void _leak_reg();
extern void _nach_reg();

void modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," Iclamp2.mod");
fprintf(stderr," ih.mod");
fprintf(stderr," ka.mod");
fprintf(stderr," kht.mod");
fprintf(stderr," klt.mod");
fprintf(stderr," leak.mod");
fprintf(stderr," nach.mod");
fprintf(stderr, "\n");
    }
_Iclamp2_reg();
_ih_reg();
_ka_reg();
_kht_reg();
_klt_reg();
_leak_reg();
_nach_reg();
}
