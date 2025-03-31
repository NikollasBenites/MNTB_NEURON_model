#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _ih_reg(void);
extern void _kht_reg(void);
extern void _klt_reg(void);
extern void _leak_reg(void);
extern void _nach_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"MNTB_Model_Dann//ih.mod\"");
    fprintf(stderr, " \"MNTB_Model_Dann//kht.mod\"");
    fprintf(stderr, " \"MNTB_Model_Dann//klt.mod\"");
    fprintf(stderr, " \"MNTB_Model_Dann//leak.mod\"");
    fprintf(stderr, " \"MNTB_Model_Dann//nach.mod\"");
    fprintf(stderr, "\n");
  }
  _ih_reg();
  _kht_reg();
  _klt_reg();
  _leak_reg();
  _nach_reg();
}

#if defined(__cplusplus)
}
#endif
