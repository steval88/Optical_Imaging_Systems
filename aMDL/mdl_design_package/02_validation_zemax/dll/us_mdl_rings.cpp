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
     Par 5 : Sub ideal     1 = inject only the RESIDUAL design-minus-
                           ideal-lens phase (zero paraxial power); the
                           focusing then comes from a separate Paraxial
                           surface (POP hybrid, 2026-09-07)
     Par 6 : Avg cell      POP grid pitch [mm]; > 0 (with Sub ideal = 1)
                           replaces the point sample of the residual by
                           its cell mean (phase = arg<u>, transmission =
                           |<u>|^2). POP ONLY -- 0 for ray analyses.
     Par 7 : OPD law       0 = intercept displacement read by the batch
                           trace as +n2*z (calibrated 2026-08-31; rz);
                           1 = PHYSICAL path (n1-n2)*z for POP, which
                           ignores the bookkeeping (measured 2026-09-07):
                           needs a real index step at the surface

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

/* cell-average cache (defined below, initialised in DllMain) */
#define MAX_AVG 16
struct AvgCache {
    int     file_id;                        /* -1 = empty              */
    double  lam, c, F, scale, zsign;        /* key                     */
    int     n;                              /* fine samples            */
    double *sre, *sim;                      /* prefix sums, length n+1 */
    double  rmax;                           /* lens radius [mm]        */
    volatile int ready;
};
static AvgCache g_avg[MAX_AVG];
static int      g_avg_next = 0;

/* Par 8 "Debug log" > 0: append the first LOG_MAX calls (any type) to
   us_mdl_rings_log.txt next to the DLL -- what OpticStudio actually
   passes (type, wavelength, n1, n2, ray position/direction, params)
   and what the DLL returned. Added 2026-09-07 to learn how POP
   evaluates a User Defined Surface instead of guessing. */
#define LOG_MAX 400
static int  g_log_count = 0;
static void log_call(const FIXED_DATA3 *FD, const USER_DATA *UD,
                     double dz, double tran, const char *what)
{
    if (FD->param[8] < 0.5 || g_log_count >= LOG_MAX) return;
    EnterCriticalSection(&g_cs);
    if (g_log_count < LOG_MAX) {
        char path[MAX_PATH];
        GetModuleFileNameA(g_hModule, path, MAX_PATH);
        char *slash = strrchr(path, '\\');
        if (!slash) slash = strrchr(path, '/');
        if (slash) *(slash + 1) = '\0'; else path[0] = '\0';
        char fname[MAX_PATH];
        snprintf(fname, MAX_PATH, "%sus_mdl_rings_log.txt", path);
        FILE *fp = fopen(fname, g_log_count == 0 ? "wt" : "at");
        if (fp) {
            if (g_log_count == 0)
                fprintf(fp, "# call type numb surf wave lam n1 n2 x y z "
                            "l m n dz tran p1 p2 p3 p4 p5 p6 p7 what\n");
            fprintf(fp, "%d %d %d %d %d %.6f %.6f %.6f %.6f %.6f %.6f "
                        "%.6f %.6f %.6f %.6e %.6f %g %g %g %g %g %g %g "
                        "%s\n",
                    g_log_count, FD->type, FD->numb, FD->surf, FD->wave,
                    FD->wavelength, FD->n1, FD->n2, UD->x, UD->y, UD->z,
                    UD->l, UD->m, UD->n, dz, tran, FD->param[1],
                    FD->param[2], FD->param[3], FD->param[4],
                    FD->param[5], FD->param[6], FD->param[7], what);
            fclose(fp);
        }
        g_log_count++;
    }
    LeaveCriticalSection(&g_cs);
}

BOOL WINAPI DllMain(HINSTANCE hinst, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH) {
        g_hModule = (HMODULE)hinst;
        if (!g_cs_init) { InitializeCriticalSection(&g_cs); g_cs_init = 1; }
        for (int i = 0; i < MAX_TABLES; ++i) {
            g_tab[i].file_id = -1; g_tab[i].h = NULL;
            g_tab[i].n = 0; g_tab[i].delta = 0.0;
        }
        for (int i = 0; i < MAX_AVG; ++i) {
            g_avg[i].file_id = -1; g_avg[i].ready = 0;
            g_avg[i].sre = g_avg[i].sim = NULL; g_avg[i].n = 0;
        }
    }
    if (reason == DLL_PROCESS_DETACH) {
        for (int i = 0; i < MAX_TABLES; ++i) free(g_tab[i].h);
        for (int i = 0; i < MAX_AVG; ++i) { free(g_avg[i].sre); free(g_avg[i].sim); }
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

/* ------------------------------------------------------------------ */
/* CELL-AVERAGED RESIDUAL (Par 6 "Avg cell", 2026-09-07) -- POP only  */
/*                                                                    */
/* POP samples a surface ONCE per grid cell (pitch c = 12-142 um on   */
/* a 4096^2 grid). The residual design-minus-ideal phase changes on   */
/* the ring (2 um) and fold (46 um at the rim) scales, so a point     */
/* sample per cell ALIASES the fold structure into the focal window   */
/* (radial model, synthetic s3-like lens, 2026-09-07: corr vs the     */
/* full-resolution PSF 0.67 at c = 141.8 um, 0.99 at 11.7 um), while  */
/* the CELL MEAN of the complex transmission                          */
/*                                                                    */
/*     <u>(rho) = (1/c) Int_{rho-c/2}^{rho+c/2} exp(i k phi_res) drho */
/*                                                                    */
/* reproduces it (corr 1.0000 at every c tested, 141.8 / 11.7 /       */
/* 2.9 um). Physics: light diffracted by structure finer than the     */
/* cell lands at |r| > lam F / (2c) (>= 108 um at c = 141.8 um) --    */
/* outside the +-30 um window -- and a POP grid cannot carry it       */
/* anyway; the cell mean keeps exactly the part of the field that     */
/* reaches the window (the local order-efficiency phasor: |<u>|^2 is  */
/* the local efficiency, arg<u> the local phase error). Applied as    */
/*     phase   : arg<u>  -> intercept displacement (OPD law +n2*z)    */
/*     |<u>|^2 : UD->rel_surf_tran (POP applies it to the irradiance, */
/*               measured 2026-09-04 on the od DLL: plateau 0.85)     */
/* Only active with Sub ideal = 1 and Avg cell > 0. The mean is a     */
/* 1-D box in rho (the POP cell is a square of side c; the radial box */
/* is the model that was validated). Ray-based analyses must keep     */
/* Avg cell = 0.                                                      */
/*                                                                    */
/* Cache: one fine radial grid (step AVG_DR) of prefix sums of        */
/* exp(i k phi_res) per (file, lam, c, F, scale, zsign); POP calls    */
/* the DLL once per grid point, so the query must be O(1).            */
/* ------------------------------------------------------------------ */
static const double AVG_DR = 1.25e-4;      /* mm = 0.125 um (16/ring) */

static AvgCache *get_avg(const RingTable *t, int file_id, double lam,
                         double c, double F, double scale, double zsign)
{
    for (int i = 0; i < MAX_AVG; ++i) {
        AvgCache *a = &g_avg[i];
        if (a->ready && a->file_id == file_id && a->lam == lam &&
            a->c == c && a->F == F && a->scale == scale &&
            a->zsign == zsign) return a;
    }
    EnterCriticalSection(&g_cs);
    for (int i = 0; i < MAX_AVG; ++i) {        /* re-check under lock  */
        AvgCache *a = &g_avg[i];
        if (a->ready && a->file_id == file_id && a->lam == lam &&
            a->c == c && a->F == F && a->scale == scale &&
            a->zsign == zsign) { LeaveCriticalSection(&g_cs); return a; }
    }
    AvgCache *a = &g_avg[g_avg_next];
    g_avg_next = (g_avg_next + 1) % MAX_AVG;
    a->ready = 0;
    free(a->sre); free(a->sim); a->sre = a->sim = NULL;
    double rmax = t->n * t->delta;
    int n = (int)(rmax / AVG_DR) + 1;
    double *sre = (double *)malloc(sizeof(double) * (n + 1));
    double *sim = (double *)malloc(sizeof(double) * (n + 1));
    if (!sre || !sim) { free(sre); free(sim);
                        LeaveCriticalSection(&g_cs); return NULL; }
    double nl = n_resist(lam);
    double k  = 2.0 * 3.14159265358979323846 / (lam * 1.0e-3); /* rad/mm */
    sre[0] = sim[0] = 0.0;
    for (int j = 0; j < n; ++j) {
        double rho = (j + 0.5) * AVG_DR;
        int i = (int)(rho / t->delta);
        double h = (i >= 0 && i < t->n) ? t->h[i] : 0.0;
        double opl = (nl - 1.0) * h + sqrt(rho * rho + F * F) - F;
        double ph = k * zsign * scale * opl;
        sre[j + 1] = sre[j] + cos(ph);
        sim[j + 1] = sim[j] + sin(ph);
    }
    a->file_id = file_id; a->lam = lam; a->c = c; a->F = F;
    a->scale = scale; a->zsign = zsign; a->n = n; a->rmax = rmax;
    a->sre = sre; a->sim = sim;
    a->ready = 1;
    LeaveCriticalSection(&g_cs);
    return a;
}

/* mean of exp(i phi_res) over [rho-c/2, rho+c/2] clipped to the lens */
static void avg_phasor(const AvgCache *a, double rho, double *re, double *im)
{
    double lo = rho - 0.5 * a->c, hi = rho + 0.5 * a->c;
    if (lo < 0.0) lo = 0.0;
    if (hi > a->rmax) hi = a->rmax;
    int jlo = (int)(lo / AVG_DR), jhi = (int)(hi / AVG_DR);
    if (jlo < 0) jlo = 0;
    if (jhi > a->n) jhi = a->n;
    if (jhi <= jlo) {                      /* cell entirely outside    */
        *re = 1.0; *im = 0.0; return;
    }
    double cnt = (double)(jhi - jlo);
    *re = (a->sre[jhi] - a->sre[jlo]) / cnt;
    *im = (a->sim[jhi] - a->sim[jlo]) / cnt;
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

/* ------------------------------------------------------------------ */
/* Intercept displacement dz(rho) [mm] and surface transmission for   */
/* the ray trace (case 5) -- and, with Sub ideal = 1, ALSO the sag     */
/* reported by case 3, so that an engine evaluating the SAG (rather    */
/* than the traced intercept) sees the same physical surface: a relief */
/* of height dz between media n1 / n2 whose path difference            */
/* (n1 - n2) dz is the residual OPL. Added 2026-09-07 after POP, with  */
/* the physical law and an index step, still returned neither the     */
/* phased nor the phase-free PSF (peak off-axis at 550 nm) -- which    */
/* route POP takes is unknown; making both routes agree removes the   */
/* question.                                                          */
/* ------------------------------------------------------------------ */
static double intercept_dz(const RingTable *t, const FIXED_DATA3 *FD,
                           double rho, double *tran)
{
    double scale = FD->param[2];
    double zsign = FD->param[3];
    int i = (int)(rho / t->delta);
    double h = (i >= 0 && i < t->n) ? t->h[i] : 0.0;   /* mm      */
    *tran = 1.0;

    /* Par 7 (OPD law, 2026-09-07). MEASURED: POP derives the phase of a
       User Defined Surface from the PHYSICAL optical path, not from the
       batch-trace bookkeeping (+n2*z) calibrated on 2026-08-31: with
       air on both sides a displaced intercept adds no path and POP
       rung 4 in point mode returned an exact Airy pattern (FWHM/Airy
       1.002-1.006) -- the residual phase was silently ZERO; only
       rel_surf_tran got through. Law 1 places the surface between two
       DIFFERENT media and displaces the intercept by dz = OPL/(n1-n2)
       so the physical path difference IS the design OPL, whatever
       glass carries it. Law 0 keeps the calibrated law for rz.       */
    double law_den = FD->n2;
    if (FD->param[7] > 0.5 && fabs(FD->n1 - FD->n2) > 1.0e-9)
        law_den = FD->n1 - FD->n2;
    if (law_den == 0.0) return 0.0;

    double nl = n_resist(FD->wavelength);   /* lam in um      */
    /* SIGN: OPD law is +n2*z (calibrated to 1.7e-11 um against the
       canonical table, 2026-08-31)                                   */
    double opl = (nl - 1.0) * h;                /* design OPL [mm] */
    /* Par 5 (Sub ideal, 2026-09-07): subtract the PERFECT-lens OPL
       -(sqrt(rho^2+F^2) - F), F = Par 4, so the injected phase is the
       RESIDUAL design - ideal = k*(P lam0) + quantization error: flat
       within a fold, jumps only at the folds. Used by the POP hybrid
       (Paraxial surface f=F + this surface): POP's pilot beam is built
       from REAL rays and a phase-only surface leaves it collimated
       (measured 2026-09-04), so the focusing must come from a
       ray-bending surface while the diffractive structure rides here. */
    if (FD->param[5] > 0.5 && FD->param[4] != 0.0) {
        double F = FD->param[4];
        opl += sqrt(rho * rho + F * F) - F;     /* - OPL_ideal   */
    }
    double dz = zsign * scale * opl / law_den;
    /* Par 6 (Avg cell > 0, with Sub ideal): replace the point sample
       by the CELL MEAN of exp(i k phi_res) -- phase from arg<u>, local
       efficiency |<u>|^2 as the surface transmission (AvgCache above) */
    if (FD->param[5] > 0.5 && FD->param[4] != 0.0 &&
        FD->param[6] > 0.0 && FD->wavelength > 0.0) {
        AvgCache *a = get_avg(t, (int)(FD->param[1] + 0.5),
                              FD->wavelength, FD->param[6],
                              FD->param[4], scale, zsign);
        if (a) {
            double re, im;
            avg_phasor(a, rho, &re, &im);
            double lam_mm = FD->wavelength * 1.0e-3;
            double ph = atan2(im, re);          /* (-pi, pi]   */
            dz = ph / (2.0 * 3.14159265358979323846) * lam_mm / law_den;
            *tran = re * re + im * im;
        }
    }
    /* Par 9 (dz gain, 2026-09-07): multiplies the FINAL displacement.
       Diagnostic: the wave engines (POP, Huygens) apply a phase
       proportional to dz with a factor that is neither the batch-trace
       bookkeeping (n2) nor the physical path (n1-n2) -- a gain sweep
       identifies the factor empirically. Default 1 (no effect).      */
    if (FD->param[9] != 0.0) dz *= FD->param[9];
    /* Par 10 (dz offset [mm], 2026-09-07): constant added to the
       displacement so that EVERY intercept lies at z > 0. A constant
       displacement is a constant phase (invisible), but the Huygens
       gain sweep showed gain -1 != gain +1 (corr 0.886 vs 0.990) for
       a rotationally symmetric residual -- impossible for a pure
       phase at focus -- and a monotone energy loss with |gain|: the
       hypothesis is that rays whose intercept falls BEHIND the
       surface vertex (negative step) are not propagated correctly by
       the wave engines. Offset >= lam/(2(n1-n2)) removes negative
       steps for the wrapped cell-mean phase.                        */
    dz += FD->param[10];
    return dz;
}

extern "C" {

/* ENTRY POINT v3 (2026-09-04). Ported from the original
   UserDefinedSurface/FIXED_DATA for two MEASURED reasons:
   (a) POP: OpticStudio decides per DLL whether a User Defined
       Surface may be propagated by physical optics; with the v1
       entry point it forced 'Use Rays To Propagate' on and the
       ray hand-off aborted ('Computation aborted; invalid
       results!'), while its own us_stand.dll (v3) propagated.
   (b) case 0 sub-queries: the v1 code answered EVERY numb with
       the surface name -- including numb 2 ('is this a GRIN
       medium?'), where a NON-EMPTY string means YES. A GRIN
       medium must be ray-propagated: that alone explains the
       forced flag. Now: numb 0 name, 1 '1' (rotationally
       symmetric), 2 '' (not GRIN).
   Also: unknown request types return 0 (us_stand.c behaviour),
   not -1 -- POP issues types the v1 code treated as errors;
   case 7 zeroes all parameters via max_parameter; cases 8/9
   (first/last call) are served. Ray-trace physics (cases 3-6)
   is UNCHANGED; the rz self-check must still read 0.0000. */
int __declspec(dllexport) APIENTRY
UserDefinedSurface3(USER_DATA *UD, FIXED_DATA3 *FD)
{
    switch (FD->type) {

    case 0:
        /* general information -- FD->numb selects the query */
        switch (FD->numb) {
        case 0:  strcpy(UD->string, "MDL Rings"); break;  /* name */
        case 1:  strcpy(UD->string, "1");  break;  /* rotationally symmetric: any char = yes */
        case 2:  UD->string[0] = '\0';    break;  /* GRIN medium: EMPTY = no */
        default: UD->string[0] = '\0';    break;
        }
        break;

    case 1:
        /* parameter names */
        switch (FD->numb) {
        case 1: strcpy(UD->string, "File #");       break;
        case 2: strcpy(UD->string, "Height scale"); break;
        case 3: strcpy(UD->string, "Z sign");       break;
        case 4: strcpy(UD->string, "Parax f");      break;
        case 5: strcpy(UD->string, "Sub ideal");    break;
        case 6: strcpy(UD->string, "Avg cell");     break;
        case 7: strcpy(UD->string, "OPD law");      break;
        case 8: strcpy(UD->string, "Debug log");    break;
        case 9: strcpy(UD->string, "dz gain");      break;
        case 10: strcpy(UD->string, "dz offset");   break;
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
            if (FD->param[5] > 0.5) {
                /* residual mode: the sag IS the traced intercept, so a
                   sag-evaluating engine and the ray trace agree */
                double tran;
                UD->sag1 = intercept_dz(t, FD, rho, &tran);
            } else {
                UD->sag1 = stair_sag(t, rho, FD->param[2], FD->param[3]);
            }
            UD->sag2 = UD->sag1;
            log_call(FD, UD, UD->sag1, 1.0, "sag");
        }
        break; }

    case 4: {
        /* paraxial ray trace: thin element of power 1/f_par (0 = flat) */
        double power = 0.0;
        if (FD->param[4] != 0.0 && FD->param[5] < 0.5)
            power = 1.0 / FD->param[4];
        /* Par 5 (Sub ideal) = 1: the surface carries only the RESIDUAL
           design-minus-ideal phase -> zero paraxial power (a separate
           Paraxial surface supplies the focusing in the POP hybrid) */
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
        log_call(FD, UD, 0.0, 1.0, "paraxial");
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
        double tran = 1.0;
        double dz = intercept_dz(t, FD, rho, &tran);
        UD->rel_surf_tran = tran;
        log_call(FD, UD, dz, tran, "trace");
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

    case 7: {
        /* safe/default values: zero everything OpticStudio exposes, then ours */
        int i;
        for (i = 0; i <= FD->max_parameter && i < 201; i++) FD->param[i] = 0.0;
        for (i = 0; i <= FD->max_extradata && i < 501; i++) FD->xdata[i] = 0.0;
        FD->param[1] = 1.0;   /* File #       */
        FD->param[2] = 1.0;   /* Height scale */
        FD->param[3] = 1.0;   /* Z sign       */
        FD->param[4] = 0.0;   /* Parax f      */
        FD->param[5] = 0.0;   /* Sub ideal    */
        FD->param[6] = 0.0;   /* Avg cell [mm]: 0 = point sample */
        FD->param[7] = 0.0;   /* OPD law: 0 = +n2*z (batch trace), 1 = (n1-n2)*z physical (POP) */
        FD->param[8] = 0.0;   /* Debug log: 1 = write us_mdl_rings_log.txt (first calls) */
        FD->param[9] = 1.0;   /* dz gain (diagnostic multiplier on the displacement) */
        FD->param[10] = 0.0;  /* dz offset [mm]: constant displacement (constant phase) */
        break; }

    case 8:
        /* first call: nothing to pre-allocate (tables load lazily) */
        break;

    case 9:
        /* last call: nothing to release that the process will not */
        break;

    default:
        /* unknown request type: NOT an error (us_stand.c returns 0) */
        log_call(FD, UD, 0.0, 1.0, "unknown-type");
        break;
    }
    return 0;
}

} /* extern "C" */