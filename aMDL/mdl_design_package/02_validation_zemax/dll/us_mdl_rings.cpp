/**********************************************************************
 us_mdl_rings.cpp
 =================

 Zemax OpticStudio User-Defined Surface DLL:
 Multilevel Diffractive Lens (MDL) as a stepped ring relief
 ("zone decomposition" -- the real staircase profile is ray traced, so
 all diffraction orders, their efficiencies and the chromatic behavior
 emerge physically, cf. the Ansys KB article "Realistic modeling of
 relief-type diffractive intraocular lenses using User-Defined Surface
 DLLs").

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
     Par 2 : Height scale  multiplies all heights (default 1.0)
     Par 3 : Z sign        +1: relief toward +z, -1: toward -z
     Par 4 : Parax f       paraxial focal length in lens units
                           (0 = treat as plane parallel plate)

 Ray trace
 ---------
 The staircase intercept is found by fixed-point iteration on the ring
 index (converges in <= 3 steps for realistic rays; vertical side walls
 are not modeled -- standard thin-relief assumption, valid when the ray
 obliquity on the relief is moderate). Refraction is plain Snell's law
 on the locally flat ring top, i.e. the zone-decomposition model.

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
        /* sag at UD->x, UD->y */
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
        /* real ray trace: intersect the staircase, then refract       */
        RingTable *t = get_table((int)(FD->param[1] + 0.5));
        double scale = FD->param[2];
        double zsign = FD->param[3];

        if (!t) return -1;                       /* table missing      */
        if (UD->n == 0.0) return FD->surf;       /* ray parallel to    */
                                                 /* the surface plane  */
        /* fixed-point iteration on the RING INDEX: within one ring the
           surface is the plane z = h_i, whose ray intercept is exact;
           iterate until the assumed ring contains the intercept.
           (Vertical side walls are not modeled: a ray landing exactly
           on a boundary keeps the last assumption -- the standard
           thin-relief / zone-decomposition approximation.)            */
        double rho0 = sqrt(UD->x * UD->x + UD->y * UD->y);
        int    i_as = (int)(rho0 / t->delta);
        double x = UD->x, y = UD->y, zplane = 0.0, tstep = 0.0;
        for (int it = 0; it < 16; ++it) {
            zplane = (i_as >= 0 && i_as < t->n)
                     ? zsign * scale * t->h[i_as] : 0.0;
            tstep = (zplane - UD->z) / UD->n;    /* from tangent plane */
            x = UD->x + tstep * UD->l;
            y = UD->y + tstep * UD->m;
            double rho = sqrt(x * x + y * y);
            int i_new = (int)(rho / t->delta);
            if (i_new >= t->n) i_new = t->n;     /* outside: substrate */
            if (i_new == i_as) break;
            i_as = i_new;
        }
        UD->x = x; UD->y = y; UD->z = zplane;
        UD->path = tstep;

        /* normal of the locally flat ring top                          */
        UD->ln = 0.0; UD->mn = 0.0; UD->nn = -1.0;

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
