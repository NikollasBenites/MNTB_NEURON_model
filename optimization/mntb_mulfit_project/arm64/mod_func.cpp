#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _Iclamp2_reg(void);
extern void _ih_reg(void);
extern void _ih_dth_reg(void);
extern void _ih_nmb_reg(void);
extern void _ka_reg(void);
extern void _kht_reg(void);
extern void _kht_dth_reg(void);
extern void _kht_dth_nmb_reg(void);
extern void _kht_nmb_reg(void);
extern void _klt_reg(void);
extern void _klt_dth_reg(void);
extern void _klt_nmb_reg(void);
extern void _leak_reg(void);
extern void _nach_reg(void);
extern void _nach_dth_reg(void);
extern void _nach_nmb_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"mod//Iclamp2.mod\"");
    fprintf(stderr, " \"mod//ih.mod\"");
    fprintf(stderr, " \"mod//ih_dth.mod\"");
    fprintf(stderr, " \"mod//ih_nmb.mod\"");
    fprintf(stderr, " \"mod//ka.mod\"");
    fprintf(stderr, " \"mod//kht.mod\"");
    fprintf(stderr, " \"mod//kht_dth.mod\"");
    fprintf(stderr, " \"mod//kht_dth_nmb.mod\"");
    fprintf(stderr, " \"mod//kht_nmb.mod\"");
    fprintf(stderr, " \"mod//klt.mod\"");
    fprintf(stderr, " \"mod//klt_dth.mod\"");
    fprintf(stderr, " \"mod//klt_nmb.mod\"");
    fprintf(stderr, " \"mod//leak.mod\"");
    fprintf(stderr, " \"mod//nach.mod\"");
    fprintf(stderr, " \"mod//nach_dth.mod\"");
    fprintf(stderr, " \"mod//nach_nmb.mod\"");
    fprintf(stderr, "\n");
  }
  _Iclamp2_reg();
  _ih_reg();
  _ih_dth_reg();
  _ih_nmb_reg();
  _ka_reg();
  _kht_reg();
  _kht_dth_reg();
  _kht_dth_nmb_reg();
  _kht_nmb_reg();
  _klt_reg();
  _klt_dth_reg();
  _klt_nmb_reg();
  _leak_reg();
  _nach_reg();
  _nach_dth_reg();
  _nach_nmb_reg();
}

#if defined(__cplusplus)
}
#endif
