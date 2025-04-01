#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _Iclamp2_reg(void);
extern void _ih_reg(void);
extern void _ka_reg(void);
extern void _kht_reg(void);
extern void _klt_reg(void);
extern void _leak_reg(void);
extern void _nach_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"mod//Iclamp2.mod\"");
    fprintf(stderr, " \"mod//ih.mod\"");
    fprintf(stderr, " \"mod//ka.mod\"");
    fprintf(stderr, " \"mod//kht.mod\"");
    fprintf(stderr, " \"mod//klt.mod\"");
    fprintf(stderr, " \"mod//leak.mod\"");
    fprintf(stderr, " \"mod//nach.mod\"");
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

#if defined(__cplusplus)
}
#endif
