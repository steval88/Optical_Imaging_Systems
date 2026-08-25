"""
tradeoff_gui.py -- STAGE 0 (optional GUI): interactive front-end for the
trade-off exploration of tradeoff_maps.py.

Usage (from the package root):

    python 01_design\\tradeoff_gui.py

What it is / is not
-------------------
* A THIN front-end: all physics lives in mdl_core.py
  (pairwise_bound_matrix / upper_bound_jf) and the authoritative batch
  tool remains tradeoff_maps.py. This window only helps you EXPLORE.
* The live preview (H and D sliders -> pair map + ceiling readout) is
  computed at reduced resolution for responsiveness and is deliberately
  NOT saved anywhere: exploration is transient by design.
* The "Run full study" button hands your current settings to
  tradeoff_maps.main() on a background thread, which produces a normal
  timestamped run folder (config.json, npz data, figures, script
  snapshots) -- anything you keep goes through the standard provenance
  machinery. Progress is printed to the terminal you launched from.

Layout: left column = inputs (band, NA, dh, H/D sliders, preview
resolution); right = the pair map max Re J_w(rho1, rho2) for the
current (H, D), with the alias-free ceiling max J_w(F) in the title.

Computations run on a worker thread; the UI stays responsive and
ignores slider events while a preview is in flight (the latest request
is queued and run last, so the map always ends on the current values).
"""
import os
import queue
import sys
import threading
import time

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# package root = parent of this stage folder; mdl_core.py lives there
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_ROOT)

from mdl_core import pairwise_bound_matrix, upper_bound_jf

# the batch tool provides main() and the preset dictionaries
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "tradeoff_maps", os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "tradeoff_maps.py"))
tradeoff_maps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tradeoff_maps)


class TradeoffGUI:
    PREVIEW_RES = {"fast (96)": 96, "medium (144)": 144, "fine (200)": 200}

    def __init__(self, root):
        self.root = root
        root.title("MDL trade-off explorer (stage 0)")
        self.jobs = queue.Queue()          # preview requests -> worker
        self.results = queue.Queue()       # preview results  -> UI
        self.study_msgs = queue.Queue()    # full-study completion -> UI
        self.busy = False
        self.pending = None                # last request made while busy
        self.study_thread = None

        # ---------------- left column: controls --------------------------
        left = ttk.Frame(root, padding=8)
        left.grid(row=0, column=0, sticky="ns")

        def labeled_entry(parent, row, text, default):
            ttk.Label(parent, text=text).grid(row=row, column=0,
                                              sticky="w")
            var = tk.StringVar(value=str(default))
            ttk.Entry(parent, textvariable=var, width=9).grid(
                row=row, column=1, sticky="w", pady=1)
            return var

        base = tradeoff_maps.SETTINGS
        ttk.Label(left, text="Band / material",
                  font=("", 10, "bold")).grid(row=0, column=0,
                                              columnspan=2, sticky="w")
        self.v_lmin = labeled_entry(left, 1, "lam_min (µm)",
                                    base["lam_min_um"])
        self.v_lmax = labeled_entry(left, 2, "lam_max (µm)",
                                    base["lam_max_um"])
        self.v_na = labeled_entry(left, 3, "NA", base["na"])
        self.v_dh = labeled_entry(left, 4, "dh (µm)", base["dh_um"])

        ttk.Label(left, text="Geometry (preview)",
                  font=("", 10, "bold")).grid(row=5, column=0,
                                              columnspan=2, sticky="w",
                                              pady=(10, 0))
        self.v_h = tk.DoubleVar(value=15.0)
        self.v_d = tk.DoubleVar(value=10.24)
        self.lab_h = ttk.Label(left, text="H = 15.0 µm")
        self.lab_h.grid(row=6, column=0, columnspan=2, sticky="w")
        s_h = ttk.Scale(left, from_=0.5, to=50.0, variable=self.v_h,
                        length=180, command=self._slider_moved)
        s_h.grid(row=7, column=0, columnspan=2, sticky="we")
        self.lab_d = ttk.Label(left, text="D = 10.24 mm")
        self.lab_d.grid(row=8, column=0, columnspan=2, sticky="w")
        s_d = ttk.Scale(left, from_=0.3, to=12.0, variable=self.v_d,
                        length=180, command=self._slider_moved)
        s_d.grid(row=9, column=0, columnspan=2, sticky="we")
        for s in (s_h, s_d):
            s.bind("<ButtonRelease-1>", lambda e: self.request_preview())

        ttk.Label(left, text="Preview resolution").grid(
            row=10, column=0, sticky="w", pady=(10, 0))
        self.v_res = tk.StringVar(value="fast (96)")
        ttk.Combobox(left, textvariable=self.v_res, width=12,
                     values=list(self.PREVIEW_RES),
                     state="readonly").grid(row=10, column=1, sticky="w",
                                            pady=(10, 0))

        ttk.Button(left, text="Recompute preview",
                   command=self.request_preview).grid(
            row=11, column=0, columnspan=2, sticky="we", pady=(8, 2))
        ttk.Button(left, text="Run FULL study → run folder",
                   command=self.run_full_study).grid(
            row=12, column=0, columnspan=2, sticky="we", pady=2)

        self.status = ttk.Label(left, text="ready", wraplength=190,
                                foreground="#444")
        self.status.grid(row=13, column=0, columnspan=2, sticky="w",
                         pady=(8, 0))
        note = ("Preview maps are NOT saved.\nThe full study writes a "
                "normal\nruns/<stamp>_... folder; progress\nprints to "
                "the terminal.")
        ttk.Label(left, text=note, foreground="#777").grid(
            row=14, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # ---------------- right: matplotlib canvas -----------------------
        self.fig = Figure(figsize=(5.6, 5.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.cbar = None
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        # worker thread (daemon: dies with the window)
        threading.Thread(target=self._worker, daemon=True).start()
        self.root.after(100, self._poll_results)
        self.request_preview()

    # ------------------------------------------------------------------ UI
    def _slider_moved(self, _evt=None):
        self.lab_h.config(text="H = %.1f µm" % self.v_h.get())
        self.lab_d.config(text="D = %.2f mm" % self.v_d.get())

    def _read_band(self):
        return (float(self.v_lmin.get()), float(self.v_lmax.get()),
                float(self.v_na.get()), float(self.v_dh.get()))

    def request_preview(self):
        try:
            lmin, lmax, na, dh = self._read_band()
        except ValueError:
            self.status.config(text="invalid band/NA/dh entry")
            return
        req = dict(H=float(self.v_h.get()), D_mm=float(self.v_d.get()),
                   lmin=lmin, lmax=lmax, na=na, dh=dh,
                   n_rho=self.PREVIEW_RES[self.v_res.get()])
        if self.busy:
            self.pending = req          # run after the current one
        else:
            self.busy = True
            self.status.config(text="computing preview...")
            self.jobs.put(req)

    def _worker(self):
        while True:
            req = self.jobs.get()
            t0 = time.time()
            try:
                D_um = req["D_mm"] * 1000.0
                rho_norm, B = pairwise_bound_matrix(
                    D_um, req["na"], req["lmin"], req["lmax"],
                    req["H"], req["dh"], n_rho=req["n_rho"],
                    n_wavelengths=256)
                ceiling = upper_bound_jf(
                    D_um, req["na"], req["lmin"], req["lmax"],
                    req["H"], req["dh"], n_rho=96, n_wavelengths=192)
                self.results.put(("ok", req, B, ceiling,
                                  time.time() - t0))
            except Exception as exc:            # surface, don't crash UI
                self.results.put(("err", req, str(exc), None, None))

    def _poll_results(self):
        try:
            kind, req, B, ceiling, dt = self.results.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_results)
            return
        if kind == "ok":
            self._draw(req, B, ceiling, dt)
            self.status.config(text="preview done (%.1fs)" % dt)
        else:
            self.status.config(text="error: %s" % B)
        self.busy = False
        if self.pending is not None:
            req2, self.pending = self.pending, None
            self.busy = True
            self.status.config(text="computing preview...")
            self.jobs.put(req2)
        self.root.after(100, self._poll_results)

    def _draw(self, req, B, ceiling, dt):
        self.ax.clear()
        im = self.ax.imshow(B.T, origin="lower", extent=[0, 1, 0, 1],
                            cmap="turbo", vmin=0, vmax=1,
                            aspect="equal")
        self.ax.set_xlabel(r"$\rho_1/R$")
        self.ax.set_ylabel(r"$\rho_2/R$")
        self.ax.set_title(
            "H=%.1f µm  D=%.2f mm  |  %.0f–%.0f nm, NA %.3f\n"
            "ceiling max $J_\\omega(F)$ ≈ %.3f"
            % (req["H"], req["D_mm"], req["lmin"] * 1000,
               req["lmax"] * 1000, req["na"], ceiling), fontsize=10)
        if self.cbar is None:
            self.cbar = self.fig.colorbar(im, ax=self.ax,
                                          fraction=0.046, pad=0.04)
            self.cbar.set_label(r"max Re $J_\omega(\rho_1,\rho_2)$")
        else:
            self.cbar.update_normal(im)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ---------------------------------------------------- full batch study
    def run_full_study(self):
        if self.study_thread is not None and self.study_thread.is_alive():
            self.status.config(text="a full study is already running "
                                    "(see terminal)")
            return
        try:
            lmin, lmax, na, dh = self._read_band()
        except ValueError:
            self.status.config(text="invalid band/NA/dh entry")
            return
        H, D_mm = float(self.v_h.get()), float(self.v_d.get())
        cfg = dict(tradeoff_maps.SETTINGS,
                   name="tradeoff_gui",
                   lam_min_um=lmin, lam_max_um=lmax, na=na, dh_um=dh,
                   pair_panels=[{"label": "GUI", "h_max_um": H,
                                 "diameter_um": D_mm * 1000.0}],
                   star_samples=[{"label": "GUI", "h_max_um": H,
                                  "diameter_um": D_mm * 1000.0}])
        self.status.config(text="full study running... progress in the "
                                "terminal; folder path shown on finish")

        def job():
            try:
                run_dir = tradeoff_maps.main(cfg)
                msg = "full study done -> %s" % os.path.relpath(run_dir)
            except Exception as exc:
                msg = "full study FAILED: %s" % exc
            self.study_msgs.put(msg)

        self.study_thread = threading.Thread(target=job, daemon=True)
        self.study_thread.start()
        self.root.after(300, self._poll_study_msgs)

    def _poll_study_msgs(self):
        """Own queue for study completion -- never touches the preview
        traffic on self.results."""
        try:
            self.status.config(text=self.study_msgs.get_nowait())
            return                        # done; stop polling
        except queue.Empty:
            self.root.after(300, self._poll_study_msgs)


def main():
    root = tk.Tk()
    TradeoffGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()