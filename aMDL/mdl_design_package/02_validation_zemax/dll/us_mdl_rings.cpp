/**********************************************************************
 us_mdl_rings.cpp
 =================

 Zemax OpticStudio User-Defined Surface DLL:
 Multilevel Diffractive Lens (MDL) as a stepped ring relief
 ("zone decomposition").

 v4 PHASE MODEL (2026-08-31) ------------------------------------------
 Zemax's UDS OPD accounting was CALIBRATED by experiment on the s3
 softmin design (batch ray trace, null-subtracted against a flat
 reference):
   v1 (physical staircase intercept at z=+h, UD->path = step):
       imprinted OPD = -1.000*h, wavelength-FLAT -- the CONJUGATE of
       the sag, no index factor: neither the substrate index nor
       UD->path entered.
   v2 (flat intercept, phase injected via UD->path = OPL/n1):
       imprinted OPD = 0 exactly -- UD->path is IGNORED by the OPD
       accounting.
 Unique law consistent with both (SIGN CALIBRATED against the
 canonical ring table on the v3 run of 2026-08-31: traced profile
 equaled -(h - h0) to 1.7e-11 um when dz carried the minus sign):
 OPD_contribution = +n2 * z_intercept (UD->path ignored).

 v4 therefore injects the transmitted phase THROUGH THE INTERCEPT
 POSITION: the ray is intercepted at

     z = +zsign * scale * (n(lam) - 1) * h(rho) / n2

 so the position-based accounting yields exactly the design phase
 +zsign*scale*(n(lam)-1)*h, with the DLL's own exact Cauchy (same fit
 as mdl_core.n_az4562 and the od DLL -- keep the three in sync). The
 um-scale displacement is geometrically negligible (straight rays,
 mm-scale gaps). For straight rays at normal incidence this is
 mathematically identical to the traced staircase -- the zone model's
 entire content is the OPD. The sag (case 3) still reports the
 physical staircase for drawings/cross-sections. Every v1 OPD-based
 result (FFT PSFs included) traced the conjugate phase -h/lam and is
 retro-invalidated; ray-geometry checks (err=0) were always fine.

 Surface model
 -------------
 Concentric annular rings of uniform width DELTA; ring i (i = 0..N-1)
 spans radius [i*DELTA, (i+1)*DELTA) and has constant height h_i, so
 the sag is piecewise constant (a staircase):

     z(rho) = sign * scale * h( floor(rho/DELTA) )

 The ring table is loaded from a text file that sits NEXT TO THIS DLL:

     mdl_rings_<ID>.txt      with <ID> = integer value of Parameter 1

 File format (plain ASCII):
     line 1:  N  DELTA_mm            (ring count, ring width in mm)
     line 2+: h_i in mm, one value per line (N lines)

 LDE parameters
 --------------
     Par 1 : File #        integer ID of the ring table file
     Par 2 : Height scale  multiplies all heights (default 1.0;
                           0 = flat null reference for OPD tests)
     Par 3 : Z sign        +1: relief toward +z, -1: toward -z
                           (flips the sign of the injected phase)
     Par 4 : Parax f       paraxial focal length in lens units
                           (0 = treat as plane parallel plate)

 Build (Visual Studio):
     cl /LD /O2 us_mdl_rings.cpp /Fe:us_mdl_rings.dll
 Build (mingw-w64 cross compile):
     x86_64-w64-mingw32-g++ -shared -static -O2 -o us_mdl_rings.dll \
         us_mdl_rings.cpp
 Install: copy the DLL and mdl_rings_*.txt into
     {Documents}\Zemax\DLL\Surfaces\
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
     if (r < 0) r = 0; buf[r] = '\0'; return (DWORD)r; }
#endif /* ------------------------------------------------------------- */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "usersurf.h"

/* AZ4562 Cauchy (lam in um) -- the SAME fit as mdl_core.n_az4562 and
   us_mdl_rings_od.cpp; keep the three in sync if the material changes */
static const double N_CAUCHY_A = 1.594;
static const double N_CAUCHY_B = 0.01152;
static double n_resist(double lam_um)
{ return N_CAUCHY_A + N_CAUCHY_B / (lam_um * lam_um); }

/* ------------------------------------------------------------------ */
/* module state: ring table cache (per file ID), thread safe          */
/* ------------------------------------------------------------------ */
static HMODULE           g_hModule = NULL;
static CRITICAL_SECTION  g_cs;
static int               g_cs_init = 0;

struct RingTable {
    int     file_id;   /* -1 = empty slot                              */
    int     n;         /* number of rings                              */
    double  delta;     /* ring width [lens units, mm]                  */
    double *h;         /* heights [mm], length n                       */
};

#define MAX_TABLES 8
static RingTable g_tab[MAX_TABLES];

BOOL WINAPI DllMain(HINSTANCE hinst, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH) {
        g_hModule = (HMODULE)hinst;
        if (!g_cs_init) { InitializeCriticalSection(&g_cs); g_cs_init = 1; }
        for (int i = 0; i < MAX_TABLES; ++i) {
            g_tab[i].file_id = -1; g_tab[i].h = NULL;
            g_tab[i].n = 0; g_tab[i].delta = 0.0;
        }
    }
    if (reason == DLL_PROCESS_DETACH) {
        for (int i = 0; i < MAX_TABLES; ++i) free(g_tab[i].h);
        if (g_cs_init) { DeleteCriticalSection(&g_cs); g_cs_init = 0; }
    }
    return TRUE;
}

/* load "mdl_rings_<id>.txt" from the DLL's own folder; returns slot or
   NULL. Caller must NOT free. */
static RingTable *get_table(int file_id)
{
    if (file_id < 0) return NULL;

    /* fast path: already cached */
    for (int i = 0; i < MAX_TABLES; ++i)
        if (g_tab[i].file_id == file_id && g_tab[i].h) return &g_tab[i];

    EnterCriticalSection(&g_cs);
    /* re-check under the lock */
    for (int i = 0; i < MAX_TABLES; ++i)
        if (g_tab[i].file_id == file_id && g_tab[i].h) {
            LeaveCriticalSection(&g_cs);
            return &g_tab[i];
        }

    /* build path next to the DLL */
    char path[MAX_PATH];
    GetModuleFileNameA(g_hModule, path, MAX_PATH);
    char *slash = strrchr(path, '\\'); if (!slash) slash = strrchr(path, '/');
    if (slash) *(slash + 1) = '\0'; else path[0] = '\0';
    char fname[MAX_PATH];
    snprintf(fname, MAX_PATH, "%smdl_rings_%d.txt", path, file_id);

    RingTable *slot = NULL;
    FILE *fp = fopen(fname, "rt");
    if (fp) {
        int n = 0; double delta = 0.0;
        if (fscanf(fp, "%d %lf", &n, &delta) == 2 &&
            n > 0 && n < 10000000 && delta > 0.0) {
            double *h = (double *)malloc(sizeof(double) * n);
            int ok = (h != NULL);
            for (int i = 0; ok && i < n; ++i)
                ok = (fscanf(fp, "%lf", &h[i]) == 1);
            if (ok) {
                /* find a slot (evict first if full) */
                slot = &g_tab[0];
                for (int i = 0; i < MAX_TABLES; ++i)
                    if (g_tab[i].file_id < 0) { slot = &g_tab[i]; break; }
                free(slot->h);
                slot->h = h; slot->n = n;
                slot->delta = delta; slot->file_id = file_id;
            } else {
                free(h);
            }
        }
        fclose(fp);
    }
    LeaveCriticalSection(&g_cs);
    return slot;
}

/* staircase sag at radial coordinate rho (>=0), heights in lens units */
static double stair_sag(const RingTable *t, double rho,
                        double scale, double zsign)
{
    int i = (int)(rho / t->delta);
    if (i < 0) i = 0;
    if (i >= t->n) return 0.0;          /* outside the lens: substrate  */
    return zsign * scale * t->h[i];
}

/* classic Snell refraction from us_stand.c; returns -1 on TIR */
static int Refract(double thisn, double nextn, double *l, double *m,
                   double *n, double ln, double mn, double nn)
{
    double nr, cosi, cosi2, rad, cosr, gamma;
    if (thisn != nextn) {
        nr = thisn / nextn;
        cosi = fabs((*l) * ln + (*m) * mn + (*n) * nn);
        cosi2 = cosi * cosi;
        if (cosi2 > 1) cosi2 = 1;
        rad = 1 - ((1 - cosi2) * (nr * nr));
        if (rad < 0) return -1;
        cosr = sqrt(rad);
        gamma = nr * cosi - cosr;
        (*l) = (nr * (*l)) + (gamma * ln);
        (*m) = (nr * (*m)) + (gamma * mn);
        (*n) = (nr * (*n)) + (gamma * nn);
    }
    return 0;
}

extern "C" {

int __declspec(dllexport) APIENTRY
UserDefinedSurface(USER_DATA *UD, FIXED_DATA *FD)
{
    switch (FD->type) {

    case 0:
        /* general information: surface name */
        strcpy(UD->string, "MDL Rings");
        break;

    case 1:
        /* parameter names */
        switch (FD->numb) {
        case 1: strcpy(UD->string, "File #");       break;
        case 2: strcpy(UD->string, "Height scale"); break;
        case 3: strcpy(UD->string, "Z sign");       break;
        case 4: strcpy(UD->string, "Parax f");      break;
        default: UD->string[0] = '\0';              break;
        }
        break;

    case 2:
        /* extra data names: none used */
        UD->string[0] = '\0';
        break;

    case 3: {
        /* sag at UD->x, UD->y: the physical staircase (drawings and
           cross-sections; the traced PHASE is injected in case 5) */
        UD->sag1 = 0.0;
        UD->sag2 = 0.0;
        RingTable *t = get_table((int)(FD->param[1] + 0.5));
        if (t) {
            double rho = sqrt(UD->x * UD->x + UD->y * UD->y);
            UD->sag1 = stair_sag(t, rho, FD->param[2], FD->param[3]);
            UD->sag2 = UD->sag1;
        }
        break; }

    case 4: {
        /* paraxial ray trace: thin element of power 1/f_par (0 = flat) */
        double power = 0.0;
        if (FD->param[4] != 0.0) power = 1.0 / FD->param[4];
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
        /* real ray trace, v2: TEA phase injection at the tangent
           plane (see header "v2 PHASE MODEL"). The surface is traced
           as a PLANE (no geometric displacement -- identical to the
           staircase for normal incidence, where the whole model
           content is the OPD) and the transmitted phase is added as
           optical path:

               OPL = zsign * scale * (n(lam) - 1) * h(rho)   [mm]

           via UD->path. OpticStudio multiplies path by n1, so the
           value is divided by n1 -- the injection mechanism verified
           in us_mdl_rings_od.cpp (od OPD analyses match congruence
           theory exactly). n(lam) is the DLL's own exact Cauchy, so
           the phase no longer depends on how the LDE approximates the
           substrate material.                                        */
        RingTable *t = get_table((int)(FD->param[1] + 0.5));
        double scale = FD->param[2];
        double zsign = FD->param[3];

        if (!t) return -1;                       /* table missing      */

        /* v3 MEASURED OPD LAW (2026-08-31, batch-trace calibration on
           the s3 softmin design): OpticStudio IGNORES UD->path in its
           OPD accounting for UDS surfaces and derives the surface
           contribution from the INTERCEPT POSITION instead:

               OPD_contribution = -n2 * z_intercept

           (v1, intercept at z=+h: imprinted -1.000*h, lam-flat --
           the CONJUGATE phase; v2, path-injection at z=0: imprinted
           exactly 0). The phase is therefore injected through the
           position: displace the intercept to

               z = +zsign*scale*(n(lam)-1)*h(rho) / n2   (law: +n2*z)

           so the position-based accounting yields exactly
           +zsign*scale*(n(lam)-1)*h -- the design phase, with the
           DLL's own exact Cauchy. The um-scale displacement is
           geometrically negligible (straight rays, mm gaps).        */
        UD->ln = 0.0; UD->mn = 0.0; UD->nn = -1.0;

        double rho = sqrt(UD->x * UD->x + UD->y * UD->y);
        int i = (int)(rho / t->delta);
        double h = (i >= 0 && i < t->n) ? t->h[i] : 0.0;   /* mm      */

        double dz = 0.0;
        if (FD->n2 != 0.0) {
            double nl = n_resist(FD->wavelength);   /* lam in um      */
            /* SIGN: OPD law is +n2*z (calibrated to 1.7e-11 um
               against the canonical table, 2026-08-31)             */
            dz = zsign * scale * (nl - 1.0) * h / FD->n2;
        }
        double tstep = (UD->n != 0.0) ? (dz - UD->z) / UD->n : 0.0;
        UD->x += tstep * UD->l;
        UD->y += tstep * UD->m;
        UD->z  = dz;
        UD->path = tstep;      /* geometric step (ignored by the OPD
                                  accounting -- measured -- but kept
                                  truthful for any engine that uses it) */

        if (Refract(FD->n1, FD->n2, &UD->l, &UD->m, &UD->n,
                    UD->ln, UD->mn, UD->nn) == -1)
            return FD->surf;                     /* TIR                */
        break; }

    case 6:
        /* GRIN: not a gradient surface */
        UD->index = FD->n2;
        UD->dndx = UD->dndy = UD->dndz = 0.0;
        break;

    case 7:
        /* safe/default parameter values */
        FD->param[1] = 1.0;   /* File #       */
        FD->param[2] = 1.0;   /* Height scale */
        FD->param[3] = 1.0;   /* Z sign       */
        FD->param[4] = 0.0;   /* Parax f      */
        break;

    default:
        return -1;
    }
    return 0;
}

} /* extern "C" */
