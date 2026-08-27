/* test_uds.cpp -- local unit test harness for us_mdl_rings.cpp
   Builds natively on Linux (POSIX shim path). Traces rays through the
   UDS entry point exactly as OpticStudio would (case 3, 5, 7) and
   prints results for comparison with the Python reference model. */
#include <cstdio>
#include <cstring>
#include <cmath>

#include "usersurf.h"

extern "C" int UserDefinedSurface(USER_DATA *UD, FIXED_DATA *FD);

int main(int argc, char **argv)
{
    USER_DATA UD;
    FIXED_DATA FD;

    memset(&FD, 0, sizeof(FD));
    FD.param[1] = 1.0;   /* File # 1 */
    FD.param[2] = 1.0;   /* scale    */
    FD.param[3] = 1.0;   /* z sign   */
    FD.n1 = 1.6300;      /* inside resist  */
    FD.n2 = 1.0;         /* into air       */
    FD.surf = 2;

    /* case 3: sag samples */
    printf("# sag samples (rho_mm sag_mm)\n");
    for (double rho = 0.0; rho <= 5.0; rho += 0.37) {
        memset(&UD, 0, sizeof(UD));
        UD.x = rho; UD.y = 0.0;
        FD.type = 3;
        UserDefinedSurface(&UD, &FD);
        printf("SAG %.6f %.9f\n", rho, UD.sag1);
    }

    /* case 5: real rays -- axial-parallel bundle plus tilted rays */
    printf("# ray traces: in(x,y,z,l,m,n) -> out(x,y,z,l,m,n,path)\n");
    double tests[][6] = {
        {0.2000, 0.0, 0.0, 0.0, 0.0, 1.0},
        {1.7345, 0.5, 0.0, 0.0, 0.0, 1.0},
        {3.3000, 1.1, 0.0, 0.05, -0.02, 0.0},   /* n filled below */
        {4.9000, 0.0, 0.0, -0.10, 0.00, 0.0},
    };
    for (int i = 0; i < 4; ++i) {
        memset(&UD, 0, sizeof(UD));
        UD.x = tests[i][0]; UD.y = tests[i][1]; UD.z = tests[i][2];
        UD.l = tests[i][3]; UD.m = tests[i][4];
        UD.n = sqrt(1.0 - UD.l * UD.l - UD.m * UD.m);
        FD.type = 5;
        int rv = UserDefinedSurface(&UD, &FD);
        printf("RAY %d rv=%d  pos %.9f %.9f %.9f  dir %.9f %.9f %.9f"
               "  path %.9f\n",
               i, rv, UD.x, UD.y, UD.z, UD.l, UD.m, UD.n, UD.path);
    }
    return 0;
}
