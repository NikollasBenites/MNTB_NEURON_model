#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern void _ih_reg();
extern void _ih_nmb_reg();
extern void _ka_reg();
extern void _kht_reg();
extern void _kht_dth_reg();
extern void _kht_nmb_reg();
extern void _klt_reg();
extern void _klt_nmb_reg();
extern void _leak_reg();
extern void _nach_reg();
extern void _nach_nmb_reg();

void modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," ih.mod");
fprintf(stderr," ih_nmb.mod");
fprintf(stderr," ka.mod");
fprintf(stderr," kht.mod");
fprintf(stderr," kht_dth.mod");
fprintf(stderr," kht_nmb.mod");
fprintf(stderr," klt.mod");
fprintf(stderr," klt_nmb.mod");
fprintf(stderr," leak.mod");
fprintf(stderr," nach.mod");
fprintf(stderr," nach_nmb.mod");
fprintf(stderr, "\n");
    }
_ih_reg();
_ih_nmb_reg();
_ka_reg();
_kht_reg();
_kht_dth_reg();
_kht_nmb_reg();
_klt_reg();
_klt_nmb_reg();
_leak_reg();
_nach_reg();
_nach_nmb_reg();
}
