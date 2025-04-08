#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _ih_reg(void);
extern void _ih_nmb_reg(void);
extern void _ka_reg(void);
extern void _kht_reg(void);
extern void _kht_nmb_reg(void);
extern void _klt_reg(void);
extern void _klt_nmb_reg(void);
extern void _leak_reg(void);
extern void _nach_reg(void);
extern void _nach_nmb_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"ih.mod\"");
    fprintf(stderr, " \"ih_nmb.mod\"");
    fprintf(stderr, " \"ka.mod\"");
    fprintf(stderr, " \"kht.mod\"");
    fprintf(stderr, " \"kht_nmb.mod\"");
    fprintf(stderr, " \"klt.mod\"");
    fprintf(stderr, " \"klt_nmb.mod\"");
    fprintf(stderr, " \"leak.mod\"");
    fprintf(stderr, " \"nach.mod\"");
    fprintf(stderr, " \"nach_nmb.mod\"");
    fprintf(stderr, "\n");
  }
  _ih_reg();
  _ih_nmb_reg();
  _ka_reg();
  _kht_reg();
  _kht_nmb_reg();
  _klt_reg();
  _klt_nmb_reg();
  _leak_reg();
  _nach_reg();
  _nach_nmb_reg();
}

#if defined(__cplusplus)
}
#endif
