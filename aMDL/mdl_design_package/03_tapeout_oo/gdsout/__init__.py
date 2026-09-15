"""
gdsout -- STAGE 3, object-oriented: the MDL ring-height table as a GDSII
layout for maskless grayscale lithography (Heidelberg DWL and the like).

Typed, documented port of ``03_tapeout/export_gds.py`` (2026-09-16); the
legacy script stays untouched, this package is the sibling stage
``03_tapeout_oo``. Same polygons, same layer numbers, same sidecar CSV
(regression: tests/test_against_legacy.py compares the layouts polygon
by polygon).

Module map
----------
    rings.py     RingTable   -- the table the Zemax DLLs read (N, DELTA,
                                h_i), from a file or from a run folder;
                                gray levels m_i = round(h_i / dh) with dh
                                from the run's config.json, given
                                explicitly, or inferred (and echoed)
    encode.py    Encoding    -- how gray levels become layers:
                                IndexEncoding  (layer L = rings at level L)
                                TerraceEncoding(layer L = rings at level >= L)
                 Annulus     -- one merged annulus [r_in, r_out) on a layer
    export.py    GdsExport   -- annuli -> gdstk polygons (chord tolerance,
                                fracturing), the library, the .gds file and
                                the layer-map CSV
    package.py   TapeoutPackage -- <run>/tapeout/<stamp>_<mode>/: the GDS
                                plus the per-ring height map (csv/npz/png),
                                the ring table, m_final.npy, the design
                                config, tapeout_info.json (design numbers,
                                validation summary, SHA256) and a README
    cli.py       main()      -- argparse front end of export_gds.py
"""
__version__ = "2026-09-16.01"

from .encode import Annulus, Encoding, IndexEncoding, TerraceEncoding, encoding_by_name  # noqa: E402
from .export import ExportSettings, GdsExport  # noqa: E402
from .package import TapeoutPackage  # noqa: E402
from .rings import RingTable  # noqa: E402

__all__ = ["Annulus", "Encoding", "IndexEncoding", "TerraceEncoding",
           "encoding_by_name", "ExportSettings", "GdsExport", "RingTable",
           "TapeoutPackage", "__version__"]
