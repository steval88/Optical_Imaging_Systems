/* test_od.cpp -- local unit test for us_mdl_rings_od.cpp
   Traces rays and checks focal behavior of selected orders against
   the analytic harmonic-lens expectations. */
#include <cstdio>
#include <cstring>
#include <cmath>

#include "usersurf.h"

extern "C" int UserDefinedSurface(USER_DATA *UD, FIXED_DATA *FD);

static double focus_of(double r_mm, double lam, double m_ord,
                       double P, double lam0)
{
    USER_DATA UD;
    FIXED_DATA FD;
    memset(&FD, 0, sizeof(FD));
    memset(&UD, 0, sizeof(UD));
    FD.param[1] = 2.0;      /* mdl_rings_2.txt (S3 verbatim) */
    FD.param[2] = m_ord;
    FD.param[3] = P;
    FD.param[4] = lam0;
    FD.param[5] = 1.0;
    FD.param[6] = 1.0;
    FD.n1 = 1.0;            /* air-to-air phase surface for the test */
    FD.n2 = 1.0;
    FD.surf = 2;
    FD.wavelength = lam;
    FD.type = 5;
    UD.x = r_mm; UD.y = 0.0; UD.z = 0.0;
    UD.l = 0.0; UD.m = 0.0; UD.n = 1.0;
    int rv = UserDefinedSurface(&UD, &FD);
    if (rv != 0) return -1.0;
    /* axial crossing distance for the bent ray */
    double zc = -UD.x * UD.n / UD.l;    /* x + l/n * z = 0 */
    printf("  r=%5.2f lam=%.2f m=%2.0f: dir=(%.6f, 0, %.6f) "
           "T=%.4f  z_cross=%8.3f mm\n",
           r_mm, lam, m_ord, UD.l, UD.n, UD.rel_surf_tran, zc);
    return zc;
}

int main()
{
    printf("# 550 nm, order 17 (design order): expect z ~ 50.94 mm\n");
    focus_of(1.0, 0.55, 17, 17, 0.55);
    focus_of(3.0, 0.55, 17, 17, 0.55);
    focus_of(5.0, 0.55, 17, 17, 0.55);
    printf("# 850 nm, order 11: expect z ~ 50.9 mm (resonant refocus)\n");
    focus_of(3.0, 0.85, 11, 17, 0.55);
    printf("# 850 nm, order 17: expect z ~ 50.94*0.55/0.85 = 32.96 mm\n");
    focus_of(3.0, 0.85, 17, 17, 0.55);
    printf("# 700 nm, order 13 and 14 (no resonance): z ~ 52.3 / 48.6\n");
    focus_of(3.0, 0.70, 13, 17, 0.55);
    focus_of(3.0, 0.70, 14, 17, 0.55);
    printf("# order 0: expect straight ray (z_cross = inf/negative)\n");
    focus_of(3.0, 0.55, 0, 17, 0.55);
    return 0;
}
