/**********************************************************************
 us_mdl_rings_od.cpp
 ===================

 Zemax OpticStudio User-Defined Surface DLL:
 Multilevel Diffractive Lens -- ORDER-DECOMPOSITION representation.

 Companion to us_mdl_rings.cpp (zone decomposition). Same ring-table
 files, opposite modeling philosophy: instead of tracing the physical
 staircase (rays undeviated, focusing lives in the OPD), this surface
 treats the MDL as a phase DOE and bends rays according to the local
 grating equation for ONE selected diffraction order m -- exactly like
 the built-in Binary 2 approach. This makes the ray-based system
 analyses meaningful: longitudinal aberration, chromatic focal shift,
 spot diagrams, solves, optimization -- one configuration per order.

 Theory
 ------
 The staircase h(rho) is the design OPD folded at P*lam0 (P = harmonic
 order of the design, lam0 = design wavelength). At load time the table
 is UNWRAPPED to the smooth design OPD

     Ou(rho) = (n0(lam0) - 1) * h_unfolded(rho)          [lens units]

 The DOE transmission decomposes as  t = SUM_m c_m exp(i m PHI(rho)/P),
 PHI = (2 pi / lam0) * Ou. Order m therefore imparts the wavelength-
 INDEPENDENT phase  phi_m(rho) = (m/P)(2 pi/lam0) Ou(rho)  whose OPL
 contribution  phi_m * lam / (2 pi)  scales with lam -- the classic
 diffractive dispersion (V ~ -3.45 behavior). Ray bending:

     n2 sin(theta') = n1 sin(theta) + (m/P)(lam/lam0) dOu/drho

 Order efficiency (optional, scalar): ideal-sawtooth model with material
 dispersion,  |c_m|^2 = sinc^2( m - alpha(lam) ),
 alpha(lam) = (n(lam)-1) h_fold / lam,  n(lam) = AZ4562 Cauchy
 (1.594 + 0.01152/lam^2, lam in um) -- edit N_CAUCHY_* below for other
 materials. Applied through the relative surface transmission.

 LDE parameters
 --------------
     Par 1 : File #     ring table mdl_rings_<ID>.txt next to this DLL
     Par 2 : Order m    diffraction order to trace (17 -> 550 nm focus,
                        11 -> 850 nm). 0 = AUTO: use the dominant
                        (blazed) order at each wavelength -- one config
                        shows the multi-order achromat (paper Fig. 4)
     Par 3 : Design P   fold harmonic of the design (S3: 17 @ 0.55)
     Par 4 : Lam0 um    design wavelength in MICRONS
     Par 5 : Add OPL    1 = add the diffractive phase OPL to the ray
                        path (OPD fans meaningful), 0 = bend only
     Par 6 : Use eff    1 = apply sinc^2 order efficiency as relative
                        transmission, 0 = unit transmission

 The surface substrate is a PLANE (sag = 0); place it where the relief
 face sits, typically after the substrate glass. Multi-configuration
 over Par 2 gives the paper-style multi-order picture.

 Build:  cl /LD /O2 us_mdl_rings_od.cpp   (or the mingw line as in
 us_mdl_rings.cpp). Requires the same usersurf.h.
**********************************************************************/

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else  /* POSIX build for local unit testing only ---------------------- */
#  include <pthread.h>
#  include <unistd.h>
#  include <limits.h>
#  define MAX_PATH PATH_MAX
#  define APIENTRY
#  define __declspec(x)
   typedef pthread_mutex_t CRITICAL_SECTION;
   typedef void *HMODULE;
   typedef void *HINSTANCE;
   typedef void *LPVOID;
   typedef unsigned long DWORD;
   typedef int BOOL;
#  define WINAPI
#  define TRUE 1
#  define DLL_PROCESS_ATTACH 1
#  define DLL_PROCESS_DETACH 0
   static void InitializeCriticalSection(CRITICAL_SECTION *cs)
   { pthread_mutex_init(cs, NULL); }
   static void DeleteCriticalSection(CRITICAL_SECTION *cs)
   { pthread_mutex_destroy(cs); }
   static void EnterCriticalSection(CRITICAL_SECTION *cs)
   { pthread_mutex_lock(cs); }
   static void LeaveCriticalSection(CRITICAL_SECTION *cs)
   { pthread_mutex_unlock(cs); }
   static DWORD GetModuleFileNameA(HMODULE, char *buf, DWORD sz)
   { ssize_t r = readlink("/proc/self/exe", buf, sz - 1);
     if (r < 0) { r = 0; } buf[r] = '\0'; return (DWORD)r; }
#endif /* ------------------------------------------------------------- */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "usersurf.h"

/* AZ4562 Cauchy (lam in um) -- used ONLY for unfolding threshold and
   the optional efficiency model                                        */
static const double N_CAUCHY_A = 1.594;
static const double N_CAUCHY_B = 0.01152;
static double n_resist(double lam_um)
{ return N_CAUCHY_A + N_CAUCHY_B / (lam_um * lam_um); }

/* ------------------------------------------------------------------ */
static HMODULE          g_hModule = NULL;
static CRITICAL_SECTION g_cs;
static int              g_cs_init = 0;

/* The unfolded design OPD is represented by a GLOBAL smooth carrier:
   a 9-term Chebyshev series in x = 2(r/R)^2 - 1, fitted at load time.
   Rationale: the optimized ring table contains deliberate sub-fold
   phase detunings (not fold resets); any local slope estimator is
   corrupted by them, while the global fit recovers the design carrier
   (residual RMS ~0.4 um for the S3 design; slope accurate to <1% for
   r > 0.2 R, few % near the axis where the profile genuinely
   deviates). This mirrors the polynomial-phase philosophy of the
   built-in Binary 2 surface.                                          */
#define OD_NCOEF 9

struct OdTable {
    int     file_id;   /* -1 = empty                                   */
    double  P;         /* design fold harmonic the unfold used         */
    double  lam0;      /* design wavelength [um] the unfold used       */
    int     n;         /* rings                                        */
    double  delta;     /* ring width [mm]                              */
    double  R;         /* table radius n*delta [mm]                    */
    double  h_fold;    /* fold height [mm]                             */
    double  coef[OD_NCOEF]; /* Chebyshev coefficients of Ou(x) [mm]    */
    double  F0;        /* paraxial focal of Ou at (m=P, lam0) [mm]     */
};

/* Chebyshev value and derivative w.r.t. x by recurrence               */
static void cheb_eval(const double *c, int nc, double x,
                      double *val, double *der)
{
    double T0 = 1.0, T1 = x, D0 = 0.0, D1 = 1.0;
    double v = c[0] + c[1] * x;
    double d = c[1];
    for (int j = 2; j < nc; ++j) {
        double T2 = 2.0 * x * T1 - T0;
        double D2 = 2.0 * T1 + 2.0 * x * D1 - D0;
        v += c[j] * T2;
        d += c[j] * D2;
        T0 = T1; T1 = T2; D0 = D1; D1 = D2;
    }
    *val = v; *der = d;
}

/* Ou and dOu/dr from the fitted carrier at radius r [mm]              */
static void carrier_at(const OdTable *t, double r, double *Ou,
                       double *slope)
{
    double u = (r / t->R); u *= u;
    double x = 2.0 * u - 1.0;
    double v, d;
    cheb_eval(t->coef, OD_NCOEF, x, &v, &d);
    *Ou = v;
    *slope = d * 4.0 * r / (t->R * t->R);   /* d/dr = d/dx * 4r/R^2   */
}

#define MAX_TABLES 8
static OdTable g_tab[MAX_TABLES];

BOOL WINAPI DllMain(HINSTANCE hinst, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH) {
        g_hModule = (HMODULE)hinst;
        if (!g_cs_init) { InitializeCriticalSection(&g_cs); g_cs_init = 1; }
        for (int i = 0; i < MAX_TABLES; ++i)
            g_tab[i].file_id = -1;
    }
    if (reason == DLL_PROCESS_DETACH) {
        if (g_cs_init) { DeleteCriticalSection(&g_cs); g_cs_init = 0; }
    }
    return TRUE;
}

/* solve the small SPD system A c = b by Gaussian elimination          */
static int solve_lin(double A[OD_NCOEF][OD_NCOEF], double *b, double *c)
{
    int n = OD_NCOEF;
    for (int k = 0; k < n; ++k) {
        int piv = k;
        for (int i = k + 1; i < n; ++i)
            if (fabs(A[i][k]) > fabs(A[piv][k])) piv = i;
        if (fabs(A[piv][k]) < 1e-300) return 0;
        if (piv != k) {
            for (int j = 0; j < n; ++j) {
                double tmp = A[k][j]; A[k][j] = A[piv][j];
                A[piv][j] = tmp;
            }
            double tmp = b[k]; b[k] = b[piv]; b[piv] = tmp;
        }
        for (int i = k + 1; i < n; ++i) {
            double f = A[i][k] / A[k][k];
            for (int j = k; j < n; ++j) A[i][j] -= f * A[k][j];
            b[i] -= f * b[k];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j) s -= A[i][j] * c[j];
        c[i] = s / A[i][i];
    }
    return 1;
}

/* load + unfold "mdl_rings_<id>.txt"; P and lam0 come from the LDE     */
static OdTable *get_table(int file_id, double P, double lam0_um)
{
    if (file_id < 0 || P <= 0.0 || lam0_um <= 0.0) return NULL;

    for (int i = 0; i < MAX_TABLES; ++i)
        if (g_tab[i].file_id == file_id &&
            g_tab[i].P == P && g_tab[i].lam0 == lam0_um)
            return &g_tab[i];

    EnterCriticalSection(&g_cs);
    for (int i = 0; i < MAX_TABLES; ++i)
        if (g_tab[i].file_id == file_id &&
            g_tab[i].P == P && g_tab[i].lam0 == lam0_um) {
            LeaveCriticalSection(&g_cs);
            return &g_tab[i];
        }

    char path[MAX_PATH];
    GetModuleFileNameA(g_hModule, path, MAX_PATH);
    char *slash = strrchr(path, '\\');
    if (!slash) slash = strrchr(path, '/');
    if (slash) *(slash + 1) = '\0'; else path[0] = '\0';
    char fname[MAX_PATH];
    snprintf(fname, MAX_PATH, "%smdl_rings_%d.txt", path, file_id);

    OdTable *slot = NULL;
    FILE *fp = fopen(fname, "rt");
    if (fp) {
        int n = 0; double delta = 0.0;
        if (fscanf(fp, "%d %lf", &n, &delta) == 2 &&
            n > 2 && n < 10000000 && delta > 0.0) {
            double *h = (double *)malloc(sizeof(double) * n);
            int ok = (h != NULL);
            for (int i = 0; ok && i < n; ++i)
                ok = (fscanf(fp, "%lf", &h[i]) == 1);
            if (ok) {
                double n0 = n_resist(lam0_um);
                /* fold height in mm (lam0 um -> mm)                   */
                double h_fold = P * (lam0_um * 1.0e-3) / (n0 - 1.0);
                double R = n * delta;

                /* unwrap: big jumps are fold resets; smaller jumps are
                   design detunings and stay in the residual           */
                double carry = 0.0, Ou_prev = (n0 - 1.0) * h[0];

                /* accumulate Chebyshev normal equations on the fly    */
                double A[OD_NCOEF][OD_NCOEF];
                double bvec[OD_NCOEF], coef[OD_NCOEF];
                memset(A, 0, sizeof(A));
                memset(bvec, 0, sizeof(bvec));
                for (int i = 0; i < n; ++i) {
                    if (i > 0) {
                        double dh = h[i] - h[i - 1];
                        if (dh >  0.5 * h_fold) carry -= h_fold;
                        if (dh < -0.5 * h_fold) carry += h_fold;
                    }
                    double Ou_i = (n0 - 1.0) * (h[i] + carry);
                    Ou_prev = Ou_i;
                    double r = (i + 0.5) * delta;
                    double u = (r / R) * (r / R);
                    double x = 2.0 * u - 1.0;
                    double T[OD_NCOEF];
                    T[0] = 1.0; T[1] = x;
                    for (int j = 2; j < OD_NCOEF; ++j)
                        T[j] = 2.0 * x * T[j - 1] - T[j - 2];
                    for (int a = 0; a < OD_NCOEF; ++a) {
                        bvec[a] += T[a] * Ou_i;
                        for (int b = a; b < OD_NCOEF; ++b)
                            A[a][b] += T[a] * T[b];
                    }
                }
                (void)Ou_prev;
                for (int a = 0; a < OD_NCOEF; ++a)
                    for (int b = 0; b < a; ++b)
                        A[a][b] = A[b][a];

                if (solve_lin(A, bvec, coef)) {
                    slot = &g_tab[0];
                    for (int i = 0; i < MAX_TABLES; ++i)
                        if (g_tab[i].file_id < 0) { slot = &g_tab[i];
                                                    break; }
                    slot->n = n; slot->delta = delta; slot->R = R;
                    slot->h_fold = h_fold;
                    slot->P = P; slot->lam0 = lam0_um;
                    memcpy(slot->coef, coef, sizeof(coef));
                    /* paraxial focal from the carrier slope at 0.1 R  */
                    double r0 = 0.1 * R, Ov, sl;
                    slot->F0 = 0.0;
                    carrier_at(slot, r0, &Ov, &sl);
                    if (sl != 0.0) slot->F0 = -r0 / sl;
                    slot->file_id = file_id;
                }
            }
            free(h);
        }
        fclose(fp);
    }
    LeaveCriticalSection(&g_cs);
    return slot;
}

static double sincf(double x)
{
    if (fabs(x) < 1e-12) return 1.0;
    double px = 3.14159265358979323846 * x;
    return sin(px) / px;
}

extern "C" {

int __declspec(dllexport) APIENTRY
UserDefinedSurface(USER_DATA *UD, FIXED_DATA *FD)
{
    switch (FD->type) {

    case 0:
        strcpy(UD->string, "MDL Rings Order");
        break;

    case 1:
        switch (FD->numb) {
        case 1: strcpy(UD->string, "File #");   break;
        case 2: strcpy(UD->string, "Order m");  break;
        case 3: strcpy(UD->string, "Design P"); break;
        case 4: strcpy(UD->string, "Lam0 um");  break;
        case 5: strcpy(UD->string, "Add OPL");  break;
        case 6: strcpy(UD->string, "Use eff");  break;
        default: UD->string[0] = '\0';          break;
        }
        break;

    case 2:
        UD->string[0] = '\0';
        break;

    case 3:
        /* the substrate of the phase surface is a plane */
        UD->sag1 = 0.0;
        UD->sag2 = 0.0;
        break;

    case 4: {
        /* paraxial: thin element of order-m power at this wavelength  */
        OdTable *t = get_table((int)(FD->param[1] + 0.5),
                               FD->param[3], FD->param[4]);
        double power = 0.0;
        if (t && t->F0 != 0.0 && FD->param[3] > 0.0 && FD->param[4] > 0.0) {
            double m_par = FD->param[2];
            if (m_par == 0.0) {
                double alpha = (n_resist(FD->wavelength) - 1.0)
                               * (t->h_fold * 1.0e3) / FD->wavelength;
                m_par = floor(alpha + 0.5);
            }
            power = (m_par / FD->param[3])
                    * (FD->wavelength / FD->param[4]) / t->F0;
        }
        if (UD->n != 0.0) {
            UD->l = UD->l / UD->n;
            UD->m = UD->m / UD->n;
            UD->l = (FD->n1 * UD->l - UD->x * power) / FD->n2;
            UD->m = (FD->n1 * UD->m - UD->y * power) / FD->n2;
            UD->n = sqrt(1.0 / (1.0 + UD->l * UD->l + UD->m * UD->m));
            UD->l *= UD->n;
            UD->m *= UD->n;
        }
        UD->ln = 0.0; UD->mn = 0.0; UD->nn = -1.0;
        break; }

    case 5: {
        OdTable *t = get_table((int)(FD->param[1] + 0.5),
                               FD->param[3], FD->param[4]);
        if (!t) return -1;
        double m_ord = FD->param[2];
        double P     = FD->param[3];
        double lam0  = FD->param[4];        /* um                      */
        double lam   = FD->wavelength;      /* um                      */
        /* AUTO-ORDER: m = 0 selects the dominant (blazed) order at the
           current wavelength, m* = round(alpha(lam)) -- one config then
           reproduces the multi-order achromat the camera sees.        */
        if (m_ord == 0.0) {
            double alpha = (n_resist(lam) - 1.0)
                           * (t->h_fold * 1.0e3) / lam;
            m_ord = floor(alpha + 0.5);
        }

        /* plane substrate: the tangent plane IS the surface           */
        UD->path = 0.0;
        UD->ln = 0.0; UD->mn = 0.0; UD->nn = -1.0;

        double r = sqrt(UD->x * UD->x + UD->y * UD->y);
        double slope = 0.0, Ou = 0.0;
        if (r <= t->R) carrier_at(t, r, &Ou, &slope);
        /* outside the ring table: no phase, plain refraction          */

        /* order-m, wavelength-scaled phase-gradient factor            */
        double s = (m_ord / P) * (lam / lam0);

        /* grating equation on the transverse wavevector               */
        double ux = 0.0, uy = 0.0;
        if (r > 0.0) { ux = UD->x / r; uy = UD->y / r; }
        double Tx = FD->n1 * UD->l + s * slope * ux;
        double Ty = FD->n1 * UD->m + s * slope * uy;
        double rad = FD->n2 * FD->n2 - Tx * Tx - Ty * Ty;
        if (rad <= 0.0) return FD->surf;     /* evanescent order       */
        UD->l = Tx / FD->n2;
        UD->m = Ty / FD->n2;
        UD->n = sqrt(1.0 - UD->l * UD->l - UD->m * UD->m);

        /* diffractive phase OPL = phi * lam / 2pi = s * Ou  [mm];
           OpticStudio multiplies path by n1 -> divide it out          */
        if (FD->param[5] > 0.5 && FD->n1 != 0.0)
            UD->path = s * Ou / FD->n1;

        /* scalar order efficiency (ideal sawtooth + dispersion)       */
        if (FD->param[6] > 0.5) {
            double alpha = (n_resist(lam) - 1.0)
                           * (t->h_fold * 1.0e3) / lam;  /* mm -> um   */
            double c = sincf(m_ord - alpha);
            UD->rel_surf_tran = c * c;
        } else {
            UD->rel_surf_tran = 1.0;
        }
        break; }

    case 6:
        UD->index = FD->n2;
        UD->dndx = UD->dndy = UD->dndz = 0.0;
        break;

    case 7:
        FD->param[1] = 1.0;   /* File #   */
        FD->param[2] = 1.0;   /* Order m  */
        FD->param[3] = 1.0;   /* Design P */
        FD->param[4] = 0.55;  /* Lam0 um  */
        FD->param[5] = 1.0;   /* Add OPL  */
        FD->param[6] = 1.0;   /* Use eff  */
        break;

    default:
        return -1;
    }
    return 0;
}

} /* extern "C" */
