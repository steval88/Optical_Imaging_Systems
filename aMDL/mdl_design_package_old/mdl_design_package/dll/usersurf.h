/*
usersurf.h -- Zemax OpticStudio User-Defined Surface (UDS) interface,
classic version-1 structures (UserDefinedSurface export).

NOTE: This is a reconstruction of the standard header shipped with
OpticStudio at {Documents}\Zemax\DLL\Surfaces\usersurf.h (original by
Kenneth Moore, Focus Software / Zemax). The version-1 USER_DATA /
FIXED_DATA layout has been stable for decades and is what virtually all
Code-Exchange surface DLLs compile against. If your installation's
usersurf.h differs, REPLACE THIS FILE with your local copy and rebuild
-- the .cpp only uses documented fields.
*/

#ifndef USERSURF_H
#define USERSURF_H

typedef struct
{
    double x, y, z;          /* the ray coordinates at the surface        */
    double l, m, n;          /* the ray direction cosines                 */
    double ln, mn, nn;       /* the surface normal at the intercept       */
    double path;             /* the physical path length added            */
    double sag1, sag2;       /* the sag and the alternate sag             */
    double index, dndx, dndy, dndz;  /* GRIN index data                   */
    double rel_surf_tran;    /* relative surface transmission             */
    double udreserved1, udreserved2, udreserved3, udreserved4;
    char string[20];         /* string data returned to OpticStudio       */
} USER_DATA;

typedef struct
{
    int type, numb;          /* request type and sub-number               */
    int surf, wave;          /* surface number and wavelength number      */
    double wavelength, pwavelength;  /* current & primary wavelength [um] */
    double n1, n2;           /* index before and after the surface        */
    double cv, thic, sdia, k;/* curvature, thickness, semi-dia, conic     */
    double param[9];         /* parameters 1-8 (param[0] unused)          */
    double xdata[201];       /* extra data 1-200 (xdata[0] unused)        */
    double unit_scale;       /* lens units per mm (1.0 = mm)              */
    int is_a_mirror;         /* 1 if the following media is a mirror      */
    double fdreserved1, fdreserved2, fdreserved3, fdreserved4;
} FIXED_DATA;

#endif /* USERSURF_H */
