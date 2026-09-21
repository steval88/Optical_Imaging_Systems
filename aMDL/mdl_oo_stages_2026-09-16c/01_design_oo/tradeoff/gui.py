"""TradeoffApp -- the Tkinter front end of the design-space tool.

Left: the specification (diameter in mm OR inch, NA or F-number, band,
relief H, dh, DELTA, material) and the targets; buttons for the three
actions. Right: a matplotlib canvas (pair map / sweep) above a text
panel with the derived quantities and the feasibility verdict.

Nothing is limited by sliders: every field is a free entry validated by
LensSpec, so a 2-, 4- or 8-inch aperture is typed like any other. Long
computations (numeric ceiling, pair map, full study) run in a worker
thread; the window stays responsive and reports progress.
"""
from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .space import (INCH_MM, PAPER_THRESHOLD_CLASS, REFERENCE_ORDER, Ceiling, Feasibility, LensSpec,
                    MATERIALS, reference_spec)
from .study import PRESETS, StudyConfig, StudyFolder, PairMapStudy, SweepStudy, replace


class TradeoffApp:
    """The window. ``pkg_root`` is where runs/ lives."""

    FAST = dict(n_rho=128, n_wavelengths=256)          # feasibility button
    MAP = dict(n_rho=192, n_wavelengths=384)           # pair-map button

    def __init__(self, pkg_root: str) -> None:
        self.pkg_root = pkg_root
        self.root = tk.Tk()
        self.root.title("MDL design space -- trade-offs and feasibility")
        self.q: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self.busy = False
        self._build()
        self.root.after(100, self._poll)

    # -- widgets -----------------------------------------------------------------
    def _build(self) -> None:
        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(self.root, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.v: Dict[str, tk.Variable] = {}

        def field(row: int, label: str, key: str, default: str, unit: str = "") -> None:
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w")
            var = tk.StringVar(value=default)
            self.v[key] = var
            ttk.Entry(left, textvariable=var, width=12).grid(row=row, column=1, sticky="w")
            if unit:
                ttk.Label(left, text=unit).grid(row=row, column=2, sticky="w")

        r = 0
        ttk.Label(left, text="Specification", font=("", 10, "bold")).grid(row=r, column=0,
                                                                          columnspan=3, sticky="w")
        r += 1
        field(r, "diameter", "diameter", "10.24")
        self.v["d_unit"] = tk.StringVar(value="mm")
        ttk.Combobox(left, textvariable=self.v["d_unit"], values=("mm", "inch"), width=5,
                     state="readonly").grid(row=r, column=2, sticky="w")
        r += 1
        self.v["ap_mode"] = tk.StringVar(value="na")
        ttk.Radiobutton(left, text="NA", variable=self.v["ap_mode"], value="na").grid(row=r, column=0, sticky="w")
        ttk.Radiobutton(left, text="F-number", variable=self.v["ap_mode"], value="fnum").grid(row=r, column=1, sticky="w")
        r += 1
        field(r, "NA", "na", "0.1")
        r += 1
        field(r, "F-number", "fnum", "5.0")
        r += 1
        field(r, "band min", "lam_min", "400", "nm")
        r += 1
        field(r, "band max", "lam_max", "1100", "nm")
        r += 1
        field(r, "relief height H", "h_max", "15.0", "um")
        r += 1
        field(r, "height quantum dh", "dh", "0.078", "um")
        r += 1
        field(r, "ring width DELTA", "delta", "auto", "um  (auto = Nyquist lam_min/2NA)")
        r += 1
        ttk.Label(left, text="material").grid(row=r, column=0, sticky="w")
        self.v["material"] = tk.StringVar(value="AZ4562")
        ttk.Combobox(left, textvariable=self.v["material"], values=sorted(MATERIALS),
                     width=10, state="readonly").grid(row=r, column=1, sticky="w")
        r += 1
        ttk.Separator(left).grid(row=r, column=0, columnspan=3, sticky="ew", pady=6)
        r += 1
        ttk.Label(left, text="Requirement", font=("", 10, "bold")).grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1
        self.v["req_mode"] = tk.StringVar(value="paper")
        ttk.Radiobutton(left, text="paper's criterion (S2-6): max J >= 0.2 on Xiao's scale",
                        variable=self.v["req_mode"], value="paper").grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1
        ttk.Radiobutton(left, text="mean focusing efficiency >=", variable=self.v["req_mode"],
                        value="eff").grid(row=r, column=0, sticky="w")
        self.v["target_j"] = tk.StringVar(value="3")
        ttk.Entry(left, textvariable=self.v["target_j"], width=10).grid(row=r, column=1, sticky="w")
        ttk.Label(left, text="%  (band average; = J by Eq. 5)").grid(row=r, column=2, sticky="w")
        r += 1
        ttk.Radiobutton(left, text="paper lens class >=", variable=self.v["req_mode"],
                        value="class").grid(row=r, column=0, sticky="w")
        self.v["target_class"] = tk.StringVar(value="S3")
        ttk.Combobox(left, textvariable=self.v["target_class"], values=list(REFERENCE_ORDER),
                     width=5, state="readonly").grid(row=r, column=1, sticky="w")
        ttk.Label(left, text="S1..S5 = Xiao et al., NA 0.1, 400-1100 nm").grid(row=r, column=2, sticky="w")
        r += 1
        ttk.Radiobutton(left, text="none (informational only)", variable=self.v["req_mode"],
                        value="none").grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1
        self.v["show_refs"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="show where the paper's lenses sit (S1..S5)",
                        variable=self.v["show_refs"]).grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1
        field(r, "achievable fraction of ceiling", "frac", "0.55")
        r += 1
        field(r, "class tolerance (ceiling ratio)", "class_tol", "0.75")
        r += 1
        ttk.Separator(left).grid(row=r, column=0, columnspan=3, sticky="ew", pady=6)
        r += 1
        ttk.Button(left, text="Check feasibility", command=self.on_check).grid(row=r, column=0, columnspan=3, sticky="ew")
        r += 1
        ttk.Button(left, text="Pair map (Fig. 1b/c)", command=self.on_pair).grid(row=r, column=0, columnspan=3, sticky="ew")
        r += 1
        self.v["pair_mode"] = tk.StringVar(value="numeric")
        ttk.Radiobutton(left, text="numeric, Eq. S14", variable=self.v["pair_mode"],
                        value="numeric").grid(row=r, column=0, sticky="w")
        ttk.Radiobutton(left, text="semi-analytic, Eq. S16", variable=self.v["pair_mode"],
                        value="analytic").grid(row=r, column=1, columnspan=2, sticky="w")
        r += 1
        ttk.Label(left, text="full study preset").grid(row=r, column=0, sticky="w")
        self.v["preset"] = tk.StringVar(value="around this spec")
        ttk.Combobox(left, textvariable=self.v["preset"],
                     values=["around this spec"] + sorted(PRESETS), width=16,
                     state="readonly").grid(row=r, column=1, columnspan=2, sticky="w")
        r += 1
        ttk.Button(left, text="Run full study (Fig. 1d sweep -> run folder)",
                   command=self.on_study).grid(row=r, column=0, columnspan=3, sticky="ew")
        r += 1
        self.progress = ttk.Progressbar(left, length=220, mode="determinate")
        self.progress.grid(row=r, column=0, columnspan=3, sticky="ew", pady=4)
        r += 1
        self.status = tk.StringVar(value="ready")
        ttk.Label(left, textvariable=self.status, wraplength=240).grid(row=r, column=0, columnspan=3, sticky="w")

        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(6.4, 4.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=2)
        right.columnconfigure(0, weight=1)
        self.text = tk.Text(right, height=18, width=100, font=("Consolas", 9), wrap="word")
        self.text.grid(row=1, column=0, sticky="nsew")
        sb = ttk.Scrollbar(right, orient="vertical", command=self.text.yview)
        sb.grid(row=1, column=1, sticky="ns")
        self.text.configure(yscrollcommand=sb.set)

    # -- spec from the fields --------------------------------------------------------
    def spec(self) -> LensSpec:
        g = lambda k: float(self.v[k].get())  # noqa: E731
        d_mm = g("diameter") * (INCH_MM if self.v["d_unit"].get() == "inch" else 1.0)
        delta_txt = self.v["delta"].get().strip().lower()
        delta: Optional[float] = None if delta_txt in ("", "auto", "nyquist") else float(delta_txt)
        kw: Dict[str, Any] = dict(diameter_mm=d_mm, lam_min_um=g("lam_min") / 1000.0,
                                  lam_max_um=g("lam_max") / 1000.0, h_max_um=g("h_max"),
                                  dh_um=g("dh"), ring_width_um=delta,
                                  material=self.v["material"].get(), name="gui")
        if self.v["ap_mode"].get() == "na":
            kw["na"] = g("na")
        else:
            kw["fnum"] = g("fnum")
        return LensSpec(**kw)

    def _spec_or_warn(self) -> Optional[LensSpec]:
        try:
            return self.spec()
        except (ValueError, TypeError) as exc:
            messagebox.showerror("specification", str(exc))
            return None

    # -- actions --------------------------------------------------------------------
    def on_check(self) -> None:
        s = self._spec_or_warn()
        if s is None or self.busy:
            return
        mode = self.v["req_mode"].get()
        target: Optional[float] = None
        klass: Optional[str] = None
        if mode == "eff":
            target = float(self.v["target_j"].get().strip().rstrip("%")) / 100.0   # percent -> fraction
        elif mode == "class":
            klass = self.v["target_class"].get()
        paper = mode == "paper"
        frac = float(self.v["frac"].get())
        refs = bool(self.v["show_refs"].get())
        tol = float(self.v["class_tol"].get())
        lines = ["SPECIFICATION"] + ["  " + l for l in s.describe()]
        c0 = Ceiling(s, Ceiling.analytic_of(s))
        lines += ["", "CEILING (analytic, instant)"] + \
                 ["  " + l for l in Feasibility(s, c0, klass, None, class_tolerance=tol,
                                                target_j=target, achievable_fraction=frac,
                                                show_references=refs, paper_threshold=paper).lines()[:2]]
        ref_cls = klass or (PAPER_THRESHOLD_CLASS if paper else None)
        lines += ["", "numeric alias-free ceiling running (%d x %d)%s ..."
                  % (self.FAST["n_rho"], self.FAST["n_wavelengths"],
                     ", spec and %s reference" % ref_cls if ref_cls else "")]
        self._show(lines)
        self._start("numeric ceiling")

        def work() -> None:
            c = Ceiling.compute(s, numeric=True, **self.FAST)
            cr = Ceiling.compute(reference_spec(ref_cls), numeric=True, **self.FAST) if ref_cls else None
            f = Feasibility(s, c, klass, cr, class_tolerance=tol, target_j=target,
                            achievable_fraction=frac, show_references=refs, paper_threshold=paper)
            out = ["SPECIFICATION"] + ["  " + l for l in s.describe()] + ["", "FEASIBILITY"] + \
                  ["  " + l for l in f.lines()]
            def show() -> None:
                self._show(out)
                self._done("feasibility: %s" % f.verdict)
            self.q.put(show)
        threading.Thread(target=work, daemon=True).start()

    def on_pair(self) -> None:
        s = self._spec_or_warn()
        if s is None or self.busy:
            return
        analytic = self.v["pair_mode"].get() == "analytic"
        self._start("pair map (%s)" % ("Eq. S16, instant" if analytic else
                                       "Eq. S14, %d x %d" % (self.MAP["n_rho"], self.MAP["n_wavelengths"])))

        def work() -> None:
            if analytic:
                rho, B = Ceiling.analytic_pair_map(s, self.MAP["n_rho"])
                J = Ceiling.analytic_of(s)
                J_map = Ceiling.analytic_from_map(s)
                how = "Eq. S16 region |r1-r2| <= (n-1)H -> Eq. S17 max J(F) = %.3f (region integral %.3f)" % (J, J_map)
            else:
                c = Ceiling.compute(s, numeric=True, **self.MAP)
                rho, B = c.pair_map(self.MAP["n_rho"], self.MAP["n_wavelengths"])
                J = float(c.numeric) if c.numeric is not None else float("nan")
                how = "Eq. S14 max Re J, alias-free -> Eq. S15 max J(F) <= %.3f" % J

            def draw() -> None:
                self.ax.clear()
                im = self.ax.imshow(B, origin="lower", extent=(0, 1, 0, 1), vmin=0, vmax=1, cmap="jet")
                self.ax.set_xlabel("rho1 / R")
                self.ax.set_ylabel("rho2 / R")
                self.ax.set_title("D %.2f mm, H %.1f um, NA %.3f, %.0f-%.0f nm\n%s"
                                  % (s.diameter_mm, s.h_max_um, s.NA, 1000 * s.lam_min_um,
                                     1000 * s.lam_max_um, how), fontsize=8)
                if not getattr(self, "_cbar", None):
                    self._cbar = self.fig.colorbar(im, ax=self.ax)
                self.canvas.draw()
                self._done("pair map: red area %.1f %% of pairs, ceiling %.4f"
                           % (100 * np.mean(B > 0.9), J))
            self.q.put(draw)
        threading.Thread(target=work, daemon=True).start()

    def on_study(self) -> None:
        s = self._spec_or_warn()
        if s is None or self.busy:
            return
        name = self.v["preset"].get()
        if name in PRESETS:
            cfg = PRESETS[name]
        else:
            d = s.diameter_mm
            cfg = StudyConfig(name="gui_%s" % s.material.lower(), base=s,
                              pair_panels=[(d, s.h_max_um)],
                              sweep_d_mm=list(np.round(np.geomspace(d / 10, 4 * d, 16), 3)),
                              sweep_h_um=list(np.round(np.geomspace(1.0, max(60.0, 3 * s.h_max_um), 16), 3)),
                              stars={"spec": (d, s.h_max_um)})
        folder = StudyFolder(cfg, self.pkg_root)
        self._start("full study '%s' -> %s" % (cfg.name, os.path.relpath(folder.dir, self.pkg_root)))
        log_lines: List[str] = []

        def log(msg: str) -> None:
            log_lines.append(msg)
            def upd(m: str = msg) -> None:
                self.status.set(m[:90])
            self.q.put(upd)

        def progress(k: int, n: int) -> None:
            def upd() -> None:
                self.progress.configure(value=100.0 * k / n)
            self.q.put(upd)

        def work() -> None:
            if cfg.pair_panels:
                PairMapStudy(cfg, folder, log).run()
            sw = SweepStudy(cfg, folder, log, progress)
            sw.run()

            def draw() -> None:
                self.ax.clear()
                D, H = np.meshgrid(cfg.sweep_d_mm, cfg.sweep_h_um)
                cf = self.ax.contourf(D, H, sw.J, levels=np.linspace(0, 1, 21), cmap="jet")
                for nm, (dd, hh) in cfg.stars.items():
                    self.ax.plot(dd, hh, marker="*", ms=11, color="w", mec="k")
                    self.ax.annotate("%s %.3f" % (nm, sw.star_values.get(nm, float("nan"))), (dd, hh),
                                     textcoords="offset points", xytext=(6, 4), fontsize=7, color="w")
                self.ax.set_xscale("log")
                self.ax.set_yscale("log")
                self.ax.set_xlabel("diameter D (mm)")
                self.ax.set_ylabel("relief height H (um)")
                self.ax.set_title("ceiling max J_w(F) -- study '%s'" % cfg.name, fontsize=9)
                if not getattr(self, "_cbar", None):
                    self._cbar = self.fig.colorbar(cf, ax=self.ax)
                self.canvas.draw()
                self._show(["STUDY '%s' written to %s" % (cfg.name, folder.dir), ""] + log_lines)
                self._done("study done: %s" % os.path.relpath(folder.dir, self.pkg_root))
            self.q.put(draw)
        threading.Thread(target=work, daemon=True).start()

    # -- plumbing -------------------------------------------------------------------
    def _show(self, lines: List[str]) -> None:
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, "\n".join(lines))

    def _start(self, what: str) -> None:
        self.busy = True
        self.progress.configure(value=0)
        self.status.set("running: " + what)

    def _done(self, msg: str) -> None:
        self.busy = False
        self.progress.configure(value=100)
        self.status.set(msg)

    def _poll(self) -> None:
        try:
            while True:
                self.q.get_nowait()()
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def run(self) -> None:
        self.root.mainloop()


__all__ = ["TradeoffApp", "replace"]
