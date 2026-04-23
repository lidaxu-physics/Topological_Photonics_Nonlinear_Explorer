"""
NonLinear.py
============
Nonlinear frequency-comb simulation window, opened from the linear
Topological Photonic Lattice Explorer (Linear.py) via the
"Feed to Nonlinear Simulator" button.

Usage (inside Linear.py):
    from NonLinear import NonlinearWindow
    self._nl_win = NonlinearWindow(self.state, parent=self)
    self._nl_win.show()
"""

import os, io, sys, zipfile, datetime, queue, threading
import numpy as np
import scipy.linalg

# Import lattice Hamiltonian builder from Linear.py (same directory)
try:
    import importlib.util as _ilu, sys as _sys
    _base = (sys._MEIPASS if getattr(sys, 'frozen', False)
             else os.path.dirname(os.path.abspath(__file__)))
    _spec = _ilu.spec_from_file_location(
        'Linear', os.path.join(_base, 'Linear.py'))
    _lin = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_lin)
    build_hamiltonian = _lin.build_hamiltonian
except Exception:
    build_hamiltonian = None   # fallback: ψ sweep without precompute

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QDoubleSpinBox, QSpinBox, QComboBox,
    QPushButton, QGroupBox, QSplitter, QStatusBar, QFileDialog,
    QProgressBar, QScrollArea, QSizePolicy, QFrame,
    QDialog, QStackedWidget, QPlainTextEdit, QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui  import QFont

import matplotlib
if matplotlib.get_backend() != 'Qt5Agg':
    matplotlib.use('Qt5Agg')      # must be set before any pyplot import; guard against double-set
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.collections import LineCollection

try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    from functools import partial
    jax.config.update("jax_enable_x64", True)
    # Runtime test: verify XLA can actually JIT-compile and execute — not just import.
    # In a PyInstaller exe, 'import jax' succeeds even when XLA data files (MLIR
    # dialects, kernel libraries) are missing from the bundle.  The crash happens
    # silently the first time jit() tries to lower a function.  Running a trivial
    # jit here at startup catches that failure as a Python exception so we can fall
    # back to NumPy instead of segfaulting mid-simulation when Run is clicked.
    _jax_probe = jit(lambda x: x + 1.0)
    _jax_probe(jnp.array(1.0, dtype=jnp.float64))
    del _jax_probe
    JAX_AVAILABLE = True
except Exception:
    # Covers ImportError, any XLA initialisation error, and jit compile failures.
    JAX_AVAILABLE = False
    jnp = np          # fallback: jnp aliases numpy so non-jit code still runs
#if getattr(sys, 'frozen', False):
#    JAX_AVAILABLE = False
# =====================================================================
# LATTICE GEOMETRY  (mirrored from Linear.py for visualization)
# =====================================================================
def LocationToNumber(X, Y, Nx, Ny):
    return Nx * (Y - 1) + X

def NumberToLocation(N, Nx, Ny):
    X = N % Nx
    if X == 0: X = Nx
    Y = (N - X) // Nx + 1
    return int(X), int(Y)

def superlattice(Nx0, Ny0, m):
    NL = Nx0 * Ny0
    m2 = m % NL
    if m2 == 0: m2 = NL
    return 1 + (m - m2) // NL, m2

def LocationToNumber_AQH_zigzag(X, Y, Nx, Ny):
    if Y % 2 == 1:
        return int(X + (Y - 1) // 2 * (2 * Nx - 1))
    else:
        return int(Nx - 1 + X + (Y - 2) // 2 * (2 * Nx - 1))

def NumberToLocation_AQH_zigzag(N, Nx, Ny):
    flag = 0
    X1 = N % (2 * Nx - 1)
    if X1 == 0: X1 = 2 * Nx - 1
    if X1 > Nx - 1:
        flag = 1
        X1 = X1 - Nx + 1
    Y1 = (N - X1) // (2 * Nx - 1) * 2 + 1 + flag
    return int(X1), int(Y1)

def site_xy(n, h_str, nx0, ny0, nx1, ny1):
    if h_str == 'A_zigzag':
        loc = NumberToLocation_AQH_zigzag(n, nx0, ny0)
        return float(2 * loc[0] - 1 + loc[1] % 2), float(loc[1])
    if h_str in ('IQH_cyl', 'AQH_cyl'):
        mx, my = NumberToLocation(n, nx0, ny0)
        R0    = nx0 / (2 * np.pi)
        theta = (mx - 1) / nx0 * 2 * np.pi
        r     = R0 + (ny0 - my) * 1.2
        return float(r * np.cos(theta)), float(r * np.sin(theta))
    sl1, sl0 = superlattice(nx0, ny0, n)
    x1, y1   = NumberToLocation(sl1, nx1, ny1)
    x0, y0   = NumberToLocation(sl0, nx0, ny0)
    return float((x1 - 1) * nx0 + x0 - 1), float((y1 - 1) * ny0 + y0 - 1)


def compute_flow(field_1d, H_mat, NL, isite, osite, h_str, nx0, ny0, nx1, ny1):
    """Photon current arrows for all lattice types.
    Current from site n to nb: J = -Im(ψ_n* H_{n,nb} ψ_{nb})
    Mirrored from Linear.py with the correct sign convention.
    """
    is_zz  = (h_str == 'A_zigzag')
    is_cyl = (h_str in ('IQH_cyl', 'AQH_cyl'))
    Nxt    = nx0 if (is_zz or is_cyl) else nx0 * nx1
    Nyt    = ny0 if (is_zz or is_cyl) else ny0 * ny1
    Xr, Yr, Ur, Vr, cols = [], [], [], [], []

    if is_zz:
        vr    = [(0, 2), (0, -2), (2, 0), (-2, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        vecs  = [np.array(v, float) for v in vr]
        norms = [v / np.linalg.norm(v) for v in vecs]

        def inlat(xc, yc):
            xc, yc = int(round(xc)), int(round(yc))
            if not 1 <= yc <= 2 * Nyt - 1: return False
            return (2 <= xc <= 2 * Nxt - 2) if yc % 2 == 1 else (1 <= xc <= 2 * Nxt - 1)

        def c2s(xc, yc):
            xc, yc = int(round(xc)), int(round(yc))
            if yc % 2 == 1: return int((yc - 1) // 2 * (2 * Nxt - 1) + xc // 2)
            return int((yc - 2) // 2 * (2 * Nxt - 1) + Nxt - 1 + (xc + 1) // 2)

        for n in range(1, NL + 1):
            xn, yn = site_xy(n, h_str, nx0, ny0, nx1, ny1)
            arr = np.zeros(2)
            for v, u in zip(vecs, norms):
                xb, yb = xn + v[0], yn + v[1]
                if not inlat(xb, yb): continue
                try:
                    nb = c2s(xb, yb)
                    f  = np.imag(field_1d[nb] * np.conj(field_1d[n]) * H_mat[n, nb])
                except Exception:
                    f = 0.
                arr += f * u
            Xr.append(xn); Yr.append(yn); Ur.append(arr[0]); Vr.append(arr[1])
            cols.append('#4a9eff' if n == isite else '#ff4a6e' if n == osite else 'white')

    elif is_cyl:
        sc = {n: site_xy(n, h_str, nx0, ny0, nx1, ny1) for n in range(1, NL + 1)}
        for n in range(1, NL + 1):
            xn, yn = sc[n]
            arr = np.zeros(2)
            for nb in range(1, NL + 1):
                if nb == n or H_mat[n, nb] == 0: continue
                xb, yb = sc[nb]
                diff = np.array([xb - xn, yb - yn])
                norm = np.linalg.norm(diff)
                if norm < 1e-10: continue
                u = diff / norm
                try:
                    f = np.imag(field_1d[nb] * np.conj(field_1d[n]) * H_mat[n, nb])
                except Exception:
                    f = 0.
                arr += f * u
            Xr.append(xn); Yr.append(yn); Ur.append(arr[0]); Vr.append(arr[1])
            cols.append('#4a9eff' if n == isite else '#ff4a6e' if n == osite else 'white')

    else:
        vr    = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
        vecs  = [np.array(v, float) for v in vr]
        norms = [v / np.linalg.norm(v) for v in vecs]
        sc    = {n: site_xy(n, h_str, nx0, ny0, nx1, ny1) for n in range(1, NL + 1)}
        c2n   = {(int(round(v[0])), int(round(v[1]))): k for k, v in sc.items()}
        for n in range(1, NL + 1):
            xn, yn = sc[n]
            arr = np.zeros(2)
            for v, u in zip(vecs, norms):
                nb = c2n.get((int(round(xn + v[0])), int(round(yn + v[1]))))
                if nb is None: continue
                try:
                    f = np.imag(field_1d[nb] * np.conj(field_1d[n]) * H_mat[n, nb])
                except Exception:
                    f = 0.
                arr += f * u
            Xr.append(xn); Yr.append(yn); Ur.append(arr[0]); Vr.append(arr[1])
            cols.append('#4a9eff' if n == isite else '#ff4a6e' if n == osite else 'white')

    U, V = np.array(Ur), np.array(Vr)
    mx   = np.sqrt(U ** 2 + V ** 2).max()
    if mx > 0: U /= mx; V /= mx
    U = -U; V = -V
    return (np.array(Xr), np.array(Yr), U, V, cols)


# =====================================================================
# PHYSICS ENGINE
# =====================================================================

def build_disp_loss(NLattice, one_side_fsr, D2, kin, kex, ISite, OSite):
    NumFSRs = 2*one_side_fsr + 1
    FSRVec  = np.arange(-one_side_fsr, one_side_fsr+1, dtype=float)
    DispMat = np.tile(FSRVec**2*(D2/2.), (NLattice,1)).astype(np.complex128)
    LM      = np.full((NLattice, NumFSRs), kin, dtype=np.complex128)
    LM[ISite-1, :] += kex
    if OSite != ISite: LM[OSite-1, :] += kex
    return DispMat, LM, FSRVec, one_side_fsr, NumFSRs


def build_propagators(H, DispMat, LM, detuning, dt, F, PumpFSR, ISite):
    NLattice = H.shape[0];  NumFSRs = DispMat.shape[1]
    props = np.empty((NumFSRs, NLattice, NLattice), dtype=np.complex128)
    fc    = np.zeros((NumFSRs, NLattice), dtype=np.complex128)
    eye   = np.eye(NLattice, dtype=np.complex128);  cache = {}
    for mu in range(NumFSRs):
        M = np.diag((1j*detuning - 1j*DispMat[:,mu] - LM[:,mu]).astype(np.complex128)) - 1j*H
        P = scipy.linalg.expm(M*dt);  props[mu] = P
        if mu == PumpFSR:
            dP = P - eye
            try:    MidP = np.linalg.solve(M, dP)
            except Exception: MidP = np.linalg.lstsq(M, dP, rcond=None)[0]
            cache['MidP_pump'] = MidP
            fv = np.zeros(NLattice, dtype=np.complex128);  fv[ISite-1] = F
            fc[mu] = MidP @ fv
    return props, fc, cache


# Module-level cache for build_propagators_fast eigendecomposition results.
# Keyed by (H_bytes, shape) so identity is based on content, not object id().
_BPF_EIG_CACHE: dict = {}

def build_propagators_fast(H, DispMat, LM, detuning, dt, F, PumpFSR, ISite):
    """Fast propagator build using eigendecomposition of H_aug.

    Replaces 257 expm calls with 1 eig + 257 vector-exp + 257 matrix multiplies.
    ~20-30x faster than build_propagators when H changes (ψ / heater sweeps).

    H_aug = H - i·kex·diag(IO_mask)  absorbs the site-dependent loss.
    M_μ   = c_μ·I - i·H_aug  where c_μ is a scalar (uniform per mode).
    expm(M_μ·dt) = V · diag(exp((c_μ - i·λ)·dt)) · V⁻¹
    """
    NLattice = H.shape[0]; NumFSRs = DispMat.shape[1]
    FSRVec   = DispMat[0, :] / (DispMat[0, 1] if DispMat[0, 1] != 0 else 1)  # recover mu²·D2/2

    # Extra loss beyond uniform kin (site-dependent part)
    kex_per_site = LM[:, 0] - LM[:, 0].min()   # shape (NL,), real
    H_aug = H - 1j * np.diag(kex_per_site)       # absorb into H_aug
    kin   = LM[:, 0].min().real                   # uniform loss floor

    # Eigendecompose H_aug (non-Hermitian in general).
    # Cache keyed by tobytes() + shape — stable content-based identity,
    # avoids the id()-reuse footgun of the old mutable default arg approach.
    H_key = (H_aug.tobytes(), H_aug.shape)
    if H_key not in _BPF_EIG_CACHE:
        lam, V = scipy.linalg.eig(H_aug)
        Vd = np.linalg.inv(V)
        if len(_BPF_EIG_CACHE) > 8:   # bound memory: keep at most 8 entries
            _BPF_EIG_CACHE.pop(next(iter(_BPF_EIG_CACHE)))
        _BPF_EIG_CACHE[H_key] = (lam, V, Vd)
    lam, V, Vd = _BPF_EIG_CACHE[H_key]

    # Uniform diagonal scalar per mode: c_μ = i·Δ - i·D2·μ²/2 - kin
    # DispMat[m, mu] = D2/2 · FSRVec[mu]^2 (same for all m)
    disp_per_mode = DispMat[0, :]   # (NumFSRs,) — use row 0 (same for all sites)
    c_mus = 1j*detuning - 1j*disp_per_mode - kin   # (NumFSRs,) complex

    # Build all props at once: props[mu] = V @ diag(exp((c_mu - i*lam)*dt)) @ Vd
    # exp_all: (NumFSRs, NL)
    exp_all = np.exp((c_mus[:, np.newaxis] - 1j*lam[np.newaxis, :]) * dt)  # (NumFSRs, NL)
    # props[mu] = (V * exp_all[mu,:]) @ Vd
    props = np.einsum('kl,ml,lj->mkj', V, exp_all, Vd)   # (NumFSRs, NL, NL)

    # Force real symmetry (tiny imaginary artifacts from eig)
    # props = props.real + 1j*props.imag  — keep as-is, already complex

    # Pump forcing fc: same as build_propagators
    fc    = np.zeros((NumFSRs, NLattice), dtype=np.complex128)
    eye   = np.eye(NLattice, dtype=np.complex128); cache = {}
    # Recompute M at pump mode only (for MidP)
    mu_p   = PumpFSR
    M_pump = c_mus[mu_p]*eye - 1j*H_aug
    dP     = props[mu_p] - eye
    try:    MidP = np.linalg.solve(M_pump, dP)
    except Exception: MidP = np.linalg.lstsq(M_pump, dP, rcond=None)[0]
    cache['MidP_pump'] = MidP
    fv = np.zeros(NLattice, dtype=np.complex128); fv[ISite-1] = F
    fc[mu_p] = MidP @ fv

    return props, fc, cache


def update_fc_fast(cache, F, PumpFSR, ISite, NumFSRs, NLattice):
    """Cheapest update: only the pump forcing vector changed (F changed, H and M unchanged)."""
    fc   = np.zeros((NumFSRs, NLattice), dtype=np.complex128)
    MidP = cache.get('MidP_pump')
    if MidP is not None:
        fv = np.zeros(NLattice, dtype=np.complex128);  fv[ISite-1] = F
        fc[PumpFSR] = MidP @ fv
    return fc


_stepper_cache: dict = {}

def get_stepper(NIter: int):
    """Return a compiled JAX stepper if JAX is available, else a plain NumPy loop."""
    if NIter not in _stepper_cache:
        if JAX_AVAILABLE:
            @partial(jit)
            def _run(a_T, props, fc, dt):
                def step(a_T, _):
                    a_T = jnp.exp(1j*(dt*0.5)*jnp.abs(a_T)**2) * a_T
                    a_W = jnp.fft.fft(a_T, axis=1, norm='ortho')
                    a_W = jnp.einsum('mij,jm->im', props, a_W) + fc.T
                    a_T = jnp.fft.ifft(a_W, axis=1, norm='ortho')
                    a_T = jnp.exp(1j*(dt*0.5)*jnp.abs(a_T)**2) * a_T
                    return a_T, None
                a_final, _ = jax.lax.scan(step, a_T, None, length=NIter)
                return a_final
        else:
            def _run(a_T, props, fc, dt):
                """Pure-NumPy fallback stepper (no JIT compilation)."""
                a_T = np.array(a_T, dtype=complex)
                for _ in range(NIter):
                    a_T = np.exp(1j*(dt*0.5)*np.abs(a_T)**2) * a_T
                    a_W = np.fft.fft(a_T, axis=1, norm='ortho')
                    a_W = np.einsum('mij,jm->im', np.array(props), a_W) + np.array(fc).T
                    a_T = np.fft.ifft(a_W, axis=1, norm='ortho')
                    a_T = np.exp(1j*(dt*0.5)*np.abs(a_T)**2) * a_T
                return a_T
        _stepper_cache[NIter] = _run
    return _stepper_cache[NIter]


def make_initial_state(NLattice, NumFSRs, seed=42):
    rng = np.random.default_rng(seed)
    a_W = 1e-8*rng.random((NLattice,NumFSRs))*np.exp(1j*rng.random((NLattice,NumFSRs))*2*np.pi)
    state = np.fft.ifft(a_W, axis=1, norm='ortho')
    return jnp.array(state) if JAX_AVAILABLE else state


_H_KEYS = {'psi'}


class Schedule:
    def __init__(self, n_steps):
        self.n_steps  = int(n_steps)
        self.channels = {}

    def add(self, name, values):
        arr = np.asarray(values, dtype=float)
        if arr.shape != (self.n_steps,):
            raise ValueError(f"Channel '{name}': shape mismatch")
        self.channels[name] = arr;  return self

    def ramp(self, name, start, end):
        return self.add(name, np.linspace(start, end, self.n_steps))

    def fixed(self, name, value):
        return self.add(name, np.full(self.n_steps, float(value)))

    def at(self, gg):
        return {k: float(v[gg]) for k, v in self.channels.items()}

    def _changed(self, gg, key):
        if key not in self.channels: return False
        return gg == 0 or self.channels[key][gg] != self.channels[key][gg-1]

    def needs_H_rebuild(self, gg):
        hk = _H_KEYS | {k for k in self.channels if k.startswith('heater_')}
        return any(self._changed(gg, k) for k in hk)

    def needs_M_rebuild(self, gg):
        return self.needs_H_rebuild(gg) or self._changed(gg, 'detuning')

    def needs_fc_only(self, gg):
        return not self.needs_M_rebuild(gg) and self._changed(gg, 'F')


class SweepRunner:
    def __init__(self, fp):
        self.fp = dict(fp)

    def run(self, schedule, callback=None):
        import time
        fp       = self.fp
        NIter    = int(fp['NIter']);    dt       = float(fp['TStep'])
        ISite    = int(fp['ISite']);    OSite    = int(fp['OSite'])
        NLattice = int(fp['NLattice'])
        H_base   = fp['H_mat'].copy()

        DispMat, LM, FSRVec, PumpFSR, NumFSRs = build_disp_loss(
            NLattice, fp['one_side_fsr'], fp['D2'], fp['kin'], fp['kex'], ISite, OSite)

        stepper = get_stepper(NIter)
        a_T     = make_initial_state(NLattice, NumFSRs)
        H       = H_base.copy()
        props   = fc_arr = cache = props_j = fc_j = None
        a_T_all = np.zeros((NLattice, NumFSRs, schedule.n_steps), dtype=np.complex128)
        t0 = time.time()
        last_gg = -1

        precomp_props = precomp_fc = None   # unused, kept for future reference

        try:
            for gg in range(schedule.n_steps):
                sp       = schedule.at(gg)
                detuning = sp.get('detuning', fp.get('detuning', 1.5))
                F        = sp.get('F',        fp.get('F',        0.3))

                if schedule.needs_H_rebuild(gg):
                    # Rebuild H: apply new ψ and/or heaters
                    psi_val = sp.get('psi', 0.0)
                    if 'h_str' in fp and build_hamiltonian is not None:
                        H = build_hamiltonian(
                            fp['h_str'], fp['nx0'], fp['ny0'], fp['nx1'], fp['ny1'],
                            fp['j1'], fp['phi_iqh0'], fp['phi_iqh1'],
                            fp['phi_aqh0'], fp['phi_aqh1'], psi_val)[1:, 1:]
                    else:
                        H = H_base.copy()
                    for k, v in sp.items():
                        if k.startswith('heater_'):
                            n = int(k.split('_')[1]); H[n-1, n-1] += v
                    # Fast eigendecomposition-based propagator (~20x faster than expm)
                    props, fc_arr, cache = build_propagators_fast(
                        H, DispMat, LM, detuning, dt, F, PumpFSR, ISite)
                    props_j = jnp.array(props); fc_j = jnp.array(fc_arr)

                elif props is None or schedule.needs_M_rebuild(gg):
                    # Detuning changed only — standard build
                    props, fc_arr, cache = build_propagators(
                        H, DispMat, LM, detuning, dt, F, PumpFSR, ISite)
                    props_j = jnp.array(props); fc_j = jnp.array(fc_arr)

                elif schedule.needs_fc_only(gg):
                    # Only F changed — cheapest update
                    fc_arr = update_fc_fast(cache, F, PumpFSR, ISite, NumFSRs, NLattice)
                    fc_j   = jnp.array(fc_arr)

                a_T    = stepper(a_T, props_j, fc_j, dt)
                a_T_np = np.array(a_T); a_T_all[:, :, gg] = a_T_np
                last_gg = gg

                if callback is not None:
                    callback(gg, a_T_np, sp, time.time()-t0)

        except StopIteration:
            pass

        n_done = last_gg + 1
        return dict(a_T=a_T_all[:, :, :n_done], FSRVec=FSRVec,
                    PumpFSR=PumpFSR, schedule=schedule, n_done=n_done)


# =====================================================================
# SECTION 7 – QT APPLICATION
# =====================================================================

DARK_BG   = '#08090d';  PANEL_BG = '#0a0c14';  GRID_COL  = '#1a1f2e'
SPINE_COL = '#1e2230';  TEXT_DIM = '#4a5270';  TEXT_COL  = '#c8d0e7'
ACCENT    = '#00e5ff';  RING_R   = 0.30

STYLE = """
QMainWindow,QWidget{background:#08090d;color:#c8d0e7;font-size:14px;}
QGroupBox{border:1px solid #1e2230;border-radius:6px;margin-top:10px;
          padding:7px;font-size:13px;font-weight:bold;color:#3a4a70;}
QGroupBox::title{subcontrol-origin:margin;left:8px;}
QPushButton{background:#171c2e;border:1px solid #1e2230;border-radius:4px;
            padding:5px 11px;color:#c8d0e7;font-size:13px;}
QPushButton:hover{background:#1e2a40;border-color:#00e5ff;}
QPushButton:pressed,QPushButton:checked{background:#00e5ff;color:#08090d;}
QPushButton:disabled{color:#2a3050;}
QComboBox,QDoubleSpinBox,QSpinBox{background:#171c2e;border:1px solid #1e2230;
    border-radius:4px;padding:4px 6px;color:#c8d0e7;font-size:13px;}
QSlider::groove:horizontal{height:4px;background:#1e2230;border-radius:2px;}
QSlider::handle:horizontal{background:#00e5ff;width:14px;height:14px;
    margin:-5px 0;border-radius:7px;}
QProgressBar{background:#171c2e;border:1px solid #1e2230;border-radius:4px;
             color:#4caf50;font-size:12px;font-weight:bold;text-align:center;}
QProgressBar::chunk{background:#00e5ff;border-radius:3px;}
QLabel{color:#c8d0e7;font-size:13px;}
QStatusBar{color:#4a5270;font-size:12px;}
QScrollArea{border:none;}
QPlainTextEdit{background:#171c2e;border:1px solid #1e2230;border-radius:4px;
               color:#c8d0e7;font-size:12px;font-family:Courier New;}
"""

CYL_TYPES = {'IQH_cyl', 'AQH_cyl'}


def _phi_str(v_rad):
    ticks = round(v_rad / np.pi * 100)
    m = {0:'0', 25:'pi/4', 33:'pi/3', 50:'pi/2', 67:'2pi/3',
         75:'3pi/4', 100:'pi', 150:'3pi/2', 200:'2pi'}
    return m.get(ticks, f'{v_rad:.4f} rad')


# =====================================================================
# DESIGN ARRAY DIALOG
# =====================================================================
class DesignArrayDialog(QDialog):
    def __init__(self, n_steps, label, current_array=None, parent=None):
        super().__init__(parent)
        self.n_steps      = n_steps
        self.result_array = current_array.copy() if current_array is not None else None
        self.setWindowTitle(f'Design schedule:  {label}')
        self.setMinimumSize(520, 440)
        self.setStyleSheet(STYLE)

        lay = QVBoxLayout(self);  lay.setSpacing(8)
        info = QLabel(f'Enter exactly {n_steps} values  (comma-separated or one per line):')
        info.setStyleSheet('color:#c8d0e7;font-size:13px;')
        lay.addWidget(info)

        self.text_edit = QPlainTextEdit()
        if current_array is not None:
            self.text_edit.setPlainText(', '.join(f'{v:.4f}' for v in current_array))
        else:
            self.text_edit.setPlaceholderText(
                'Example:  1.8, 1.75, 1.70, ...\nor one value per line')
        lay.addWidget(self.text_edit)

        self.fig_p    = Figure(figsize=(5, 2), facecolor=DARK_BG, tight_layout=False)
        self.canvas_p = FigureCanvas(self.fig_p)
        self.canvas_p.setFixedHeight(160)
        self.ax_p = self.fig_p.add_subplot(111)
        self.ax_p.set_facecolor(PANEL_BG)
        for sp in self.ax_p.spines.values(): sp.set_edgecolor('#3a4560')
        self.ax_p.tick_params(colors=TEXT_COL, labelsize=9)
        self.ax_p.set_xlabel('Step', color=TEXT_DIM, fontsize=10)
        self.ax_p.grid(True, color=GRID_COL, lw=0.4)
        self.fig_p.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.22)
        lay.addWidget(self.canvas_p)

        self.lbl_status = QLabel('');  self.lbl_status.setStyleSheet('font-size:12px;')
        lay.addWidget(self.lbl_status)

        btn_row = QHBoxLayout()
        btn_prev = QPushButton('Preview');  btn_prev.clicked.connect(self._preview)
        btn_ok   = QPushButton('OK');       btn_ok.clicked.connect(self._accept)
        btn_can  = QPushButton('Cancel');   btn_can.clicked.connect(self.reject)
        btn_row.addWidget(btn_prev);  btn_row.addStretch()
        btn_row.addWidget(btn_ok);    btn_row.addWidget(btn_can)
        lay.addLayout(btn_row)

        if current_array is not None:
            self._preview()

    def _parse(self):
        text = self.text_edit.toPlainText().strip()
        text = text.replace('\n', ',').replace(';', ',')
        parts = [p.strip() for p in text.split(',') if p.strip()]
        try:    return [float(p) for p in parts]
        except Exception: return None

    def _preview(self):
        vals = self._parse()
        if vals is None:
            self.lbl_status.setStyleSheet('color:#ff4a6e;font-size:12px;')
            self.lbl_status.setText('Error: could not parse — check for non-numeric entries')
            return
        color = '#a8ff78' if len(vals)==self.n_steps else '#ffcc00'
        self.lbl_status.setStyleSheet(f'color:{color};font-size:12px;')
        self.lbl_status.setText(f'{len(vals)} values parsed' +
            ('' if len(vals)==self.n_steps
             else f'  (need {self.n_steps} — will interpolate on OK)'))
        self.ax_p.cla();  self.ax_p.set_facecolor(PANEL_BG)
        self.ax_p.plot(np.arange(len(vals)), vals, color=ACCENT, lw=1.4)
        self.ax_p.grid(True, color=GRID_COL, lw=0.4)
        for sp in self.ax_p.spines.values(): sp.set_edgecolor('#3a4560')
        self.ax_p.tick_params(colors=TEXT_COL, labelsize=9)
        self.canvas_p.draw_idle()

    def _accept(self):
        vals = self._parse()
        if vals is None:
            self.lbl_status.setStyleSheet('color:#ff4a6e;font-size:12px;')
            self.lbl_status.setText('Error: could not parse values');  return
        arr = np.array(vals, dtype=float)
        if len(arr) != self.n_steps:
            x_old = np.linspace(0, 1, len(arr));  x_new = np.linspace(0, 1, self.n_steps)
            arr   = np.interp(x_new, x_old, arr)
        self.result_array = arr;  self.accept()


# =====================================================================
# PARAMETER ROW WIDGET
# =====================================================================
class ParameterRow(QWidget):
    MODES = ['Fixed', 'Sweep', 'Design']

    def __init__(self, label, pkey,
                 default_fixed=0.0, default_from=1.8, default_to=1.2,
                 removable=False, get_nsteps=None, default_mode='Fixed', parent=None):
        super().__init__(parent)
        self.pkey          = pkey
        self.removable     = removable
        self._get_nsteps   = get_nsteps or (lambda: 300)
        self._design_array = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1);  lay.setSpacing(5)

        lbl = QLabel(label);  lbl.setFixedWidth(155)
        lbl.setStyleSheet('color:#c8d0e7;font-size:13px;')
        lay.addWidget(lbl)

        self.mode_combo = QComboBox();  self.mode_combo.addItems(self.MODES)
        self.mode_combo.setFixedWidth(88)
        self.mode_combo.currentTextChanged.connect(self._on_mode)
        lay.addWidget(self.mode_combo)

        self.stack = QStackedWidget();  self.stack.setFixedHeight(30)

        # page 0: Fixed
        p0 = QWidget();  l0 = QHBoxLayout(p0)
        l0.setContentsMargins(0,0,0,0);  l0.setSpacing(4)
        self.spn_fixed = QDoubleSpinBox()
        self.spn_fixed.setDecimals(4);  self.spn_fixed.setRange(-200, 200)
        self.spn_fixed.setValue(default_fixed);  self.spn_fixed.setSingleStep(0.1)
        self.spn_fixed.setFixedWidth(95)
        l0.addWidget(self.spn_fixed);  l0.addStretch()

        # page 1: Sweep
        p1 = QWidget();  l1 = QHBoxLayout(p1)
        l1.setContentsMargins(0,0,0,0);  l1.setSpacing(4)
        self.spn_from = QDoubleSpinBox()
        self.spn_from.setDecimals(4);  self.spn_from.setRange(-200, 200)
        self.spn_from.setValue(default_from);  self.spn_from.setFixedWidth(90)
        arr_lbl = QLabel('→');  arr_lbl.setStyleSheet('color:#4a5270;font-size:13px;')
        self.spn_to = QDoubleSpinBox()
        self.spn_to.setDecimals(4);  self.spn_to.setRange(-200, 200)
        self.spn_to.setValue(default_to);  self.spn_to.setFixedWidth(90)
        l1.addWidget(self.spn_from);  l1.addWidget(arr_lbl)
        l1.addWidget(self.spn_to);    l1.addStretch()

        # page 2: Design
        p2 = QWidget();  l2 = QHBoxLayout(p2)
        l2.setContentsMargins(0,0,0,0);  l2.setSpacing(6)
        self.btn_design = QPushButton('Edit array...')
        self.btn_design.setFixedHeight(26);  self.btn_design.setMinimumWidth(110)
        self.lbl_design = QLabel('not set')
        self.lbl_design.setStyleSheet('color:#4a5270;font-size:12px;')
        self.btn_design.clicked.connect(self._open_design)
        l2.addWidget(self.btn_design);  l2.addWidget(self.lbl_design);  l2.addStretch()

        self.stack.addWidget(p0);  self.stack.addWidget(p1);  self.stack.addWidget(p2)
        lay.addWidget(self.stack, stretch=1)

        if removable:
            self.btn_rm = QPushButton('×')
            self.btn_rm.setFixedSize(24, 24)
            self.btn_rm.setStyleSheet(
                'color:#ff4a6e;font-weight:bold;border:none;background:transparent;font-size:15px;')
            lay.addWidget(self.btn_rm)

        # apply default mode (triggers _on_mode which switches stack page)
        if default_mode != 'Fixed':
            self.mode_combo.setCurrentText(default_mode)

    def _on_mode(self, mode):
        self.stack.setCurrentIndex(self.MODES.index(mode))

    def _open_design(self):
        n   = self._get_nsteps()
        dlg = DesignArrayDialog(n, self.pkey, self._design_array, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._design_array = dlg.result_array
            self.lbl_design.setText(f'{len(self._design_array)} values set')
            self.lbl_design.setStyleSheet('color:#a8ff78;font-size:12px;')

    def get_array(self, n_steps) -> np.ndarray:
        mode = self.mode_combo.currentText()
        if mode == 'Fixed':
            return np.full(n_steps, self.spn_fixed.value())
        elif mode == 'Sweep':
            return np.linspace(self.spn_from.value(), self.spn_to.value(), n_steps)
        else:
            if self._design_array is None:
                return np.zeros(n_steps)
            if len(self._design_array) != n_steps:
                x_old = np.linspace(0, 1, len(self._design_array))
                x_new = np.linspace(0, 1, n_steps)
                return np.interp(x_new, x_old, self._design_array)
            return self._design_array.copy()

    def is_active(self):
        return self.mode_combo.currentText() != 'Fixed'


# =====================================================================
# NONLINEAR WINDOW
# =====================================================================
# =====================================================================
# SLOW TIME WINDOW
# =====================================================================
class SlowTimeWindow(QMainWindow):
    """
    Takes the field at a chosen sweep step, runs NIter_extended more
    iterations saving every single one, then shows 4 heatmaps:
      |a_T|²        — output ring     (x=slow-time iteration, y=θ index)
      |a_T|² summed — all rings       (x=slow-time iteration, y=θ index)
      10·log|a_W|²  — output ring     (x=slow-time iteration, y=FSR μ)
      10·log|a_W|² summed — all rings (x=slow-time iteration, y=FSR μ)
    """
    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f'Slow Time Analysis  —  seeded from sweep step {params["step_idx"]}')
        self.setMinimumSize(1600, 840)
        self.setStyleSheet(STYLE)

        self._p          = params
        self._q          = queue.Queue()
        self._stop       = threading.Event()
        self._plotting   = False   # flag for interrupting _draw_all
        # full history arrays built after run: (NLattice, NumFSRs, NIter_ext)
        self._hist       = None
        self._poll_timer = QTimer(self); self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll)

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        p = self._p
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setSpacing(4)
        root.setContentsMargins(6, 6, 6, 6)

        # Title
        psi_str = f'  ψ={p["psi"]/np.pi:.4f}π' if p.get('psi', 0.0) != 0.0 else ''
        t = QLabel(f'SLOW TIME ANALYSIS  —  seeded from sweep step {p["step_idx"]}  '
                   f'|  Δ={p["detuning"]:.5f}  F={p["F"]:.4f}{psi_str}')
        t.setFont(QFont('Courier New', 11, QFont.Bold))
        t.setStyleSheet(f'color:{ACCENT};')
        root.addWidget(t)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet('color:#1e2230;'); root.addWidget(sep)

        # ── 8 independent figures in a 2×4 QGridLayout ───────────────────
        fig_grid = QWidget()
        gl = QGridLayout(fig_grid)
        gl.setSpacing(4); gl.setContentsMargins(0, 0, 0, 0)
        for r in range(2): gl.setRowStretch(r, 1)
        for c in range(4): gl.setColumnStretch(c, 1)

        def _make_fig(rows=1, cols=1, projection=None):
            fig = Figure(facecolor=DARK_BG, tight_layout=False)
            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            if projection:
                ax = fig.add_subplot(111, projection=projection)
            elif rows == 1 and cols == 1:
                ax = fig.add_subplot(111)
            else:
                ax = None
            return fig, canvas, ax

        # [0,0] Sweep indicator — 2×2 subplots
        self.fig_ind = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_ind = FigureCanvas(self.fig_ind)
        self.canvas_ind.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs0 = self.fig_ind.add_gridspec(2, 2, hspace=0.55, wspace=0.35,
              left=0.10, right=0.97, top=0.95, bottom=0.12)
        ax_pp  = self.fig_ind.add_subplot(gs0[0, 0])
        ax_ppa = self.fig_ind.add_subplot(gs0[0, 1])
        ax_sp  = self.fig_ind.add_subplot(gs0[1, 0])
        ax_spa = self.fig_ind.add_subplot(gs0[1, 1])
        self.ax_ind = [ax_pp, ax_ppa, ax_sp, ax_spa]
        gl.addWidget(self.canvas_ind, 0, 0)

        # [0,1] 4 heatmaps
        self.fig_heat4 = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_heat4 = FigureCanvas(self.fig_heat4)
        self.canvas_heat4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs1 = self.fig_heat4.add_gridspec(2, 2, hspace=0.50, wspace=0.35,
              left=0.10, right=0.97, top=0.95, bottom=0.10)
        self.ax_aT_ring = self.fig_heat4.add_subplot(gs1[0, 0])
        self.ax_aT_all  = self.fig_heat4.add_subplot(gs1[0, 1])
        self.ax_aW_ring = self.fig_heat4.add_subplot(gs1[1, 0])
        self.ax_aW_all  = self.fig_heat4.add_subplot(gs1[1, 1])
        gl.addWidget(self.canvas_heat4, 0, 1)

        # [0,2] Comb spectrum
        self.fig_comb = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_comb = FigureCanvas(self.fig_comb)
        self.canvas_comb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs2 = self.fig_comb.add_gridspec(2, 1, hspace=0.55,
              left=0.10, right=0.97, top=0.95, bottom=0.10)
        self.ax_comb_dB   = self.fig_comb.add_subplot(gs2[0])
        self.ax_comb_zoom = self.fig_comb.add_subplot(gs2[1])
        self.canvas_comb.mpl_connect('button_press_event', self._on_comb_click)
        gl.addWidget(self.canvas_comb, 0, 2)

        # [0,3] 2D Power
        self.fig_2dpow = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_2dpow = FigureCanvas(self.fig_2dpow)
        self.canvas_2dpow.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fig_2dpow.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
        self.ax_vis_2d = self.fig_2dpow.add_subplot(111)
        self.ax_vis_2d.set_facecolor('black')
        self.ax_vis_2d.set_xticks([]); self.ax_vis_2d.set_yticks([])
        gl.addWidget(self.canvas_2dpow, 0, 3)

        # [1,0] ESA + oscilloscope
        self.fig_esa = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_esa = FigureCanvas(self.fig_esa)
        self.canvas_esa.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs3 = self.fig_esa.add_gridspec(3, 1, hspace=0.65,
              left=0.12, right=0.97, top=0.95, bottom=0.10)
        self.ax_esa_dB  = self.fig_esa.add_subplot(gs3[0])
        self.ax_esa_lin = self.fig_esa.add_subplot(gs3[1])
        self.ax_osc     = self.fig_esa.add_subplot(gs3[2])
        gl.addWidget(self.canvas_esa, 1, 0)

        # [1,1] 2D spectrum + y-cuts
        self.fig_spec2d = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_spec2d = FigureCanvas(self.fig_spec2d)
        self.canvas_spec2d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs4 = self.fig_spec2d.add_gridspec(2, 2, hspace=0.55, wspace=0.35,
              height_ratios=[1.8, 1], left=0.10, right=0.97, top=0.95, bottom=0.10)
        self.ax_hmap     = self.fig_spec2d.add_subplot(gs4[0, :])
        self.ax_ycut_pwr = self.fig_spec2d.add_subplot(gs4[1, 0])
        self.ax_ycut_phs = self.fig_spec2d.add_subplot(gs4[1, 1])
        self._cbar = None; self.ax_cbar = None
        gl.addWidget(self.canvas_spec2d, 1, 1)

        # [1,2] Photon Flow
        self.fig_flow = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_flow = FigureCanvas(self.fig_flow)
        self.canvas_flow.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fig_flow.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
        self.ax_vis_flow = self.fig_flow.add_subplot(111)
        self.ax_vis_flow.set_facecolor('black')
        self.ax_vis_flow.set_xticks([]); self.ax_vis_flow.set_yticks([])
        gl.addWidget(self.canvas_flow, 1, 2)

        # [1,3] 3D Power
        self.fig_3d = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_3d = FigureCanvas(self.fig_3d)
        self.canvas_3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fig_3d.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
        self.ax_vis_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.ax_vis_3d.set_facecolor('black')
        gl.addWidget(self.canvas_3d, 1, 3)

        # Style shared axes
        for ax, col, ttl, yl in [
            (self.ax_aT_ring, '#a8ff78', '|a_T|²  ring',      'θ'),
            (self.ax_aT_all,  '#a8ff78', '|a_T|²  all (sum)', 'θ'),
            (self.ax_aW_ring, '#ff8c42', 'log|a_W|²  ring',   'FSR μ'),
            (self.ax_aW_all,  '#ff8c42', 'log|a_W|²  all',    'FSR μ'),
        ]:
            ax.set_facecolor(PANEL_BG)
            ax.set_title(ttl, color=col, fontsize=8, pad=2)
            ax.set_xlabel('Iteration', color=TEXT_DIM, fontsize=7)
            ax.set_ylabel(yl, color=TEXT_DIM, fontsize=7)
            ax.tick_params(colors=TEXT_COL, labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')

        for ax, col, ttl in [
            (self.ax_comb_dB,   '#c8d0e7', 'Comb slow-time spectrum (dB)  — click to select tooth'),
            (self.ax_comb_zoom, '#4a9eff', 'Zoomed spectrum — selected tooth (dB)'),
        ]:
            ax.set_facecolor(PANEL_BG)
            ax.set_title(ttl, color=col, fontsize=8, pad=2)
            ax.set_xlabel('Frequency  (FSR = ? J)', color=TEXT_DIM, fontsize=7)
            ax.set_ylabel('Power (dB)', color=TEXT_DIM, fontsize=7)
            ax.tick_params(colors=TEXT_COL, labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.grid(True, color=GRID_COL, lw=0.4, ls='--', alpha=0.5)

        for ax, col, ttl, xl, yl in [
            (self.ax_esa_dB,  '#a8ff78', 'ESA spectrum (dB)',          'Angular freq (J)', 'Power (dB)'),
            (self.ax_esa_lin, '#a8ff78', 'ESA spectrum (linear norm.)', 'Angular freq (J)', 'Norm. power'),
            (self.ax_osc,     '#c8d0e7', 'Total power vs slow time',   'Iteration',        'Norm. power'),
        ]:
            ax.set_facecolor(PANEL_BG)
            ax.set_title(ttl, color=col, fontsize=8, pad=2)
            ax.set_xlabel(xl, color=TEXT_DIM, fontsize=7)
            ax.set_ylabel(yl, color=TEXT_DIM, fontsize=7)
            ax.tick_params(colors=TEXT_COL, labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.grid(True, color=GRID_COL, lw=0.4, alpha=0.4)

        self.ax_hmap.set_facecolor(PANEL_BG)
        self.ax_hmap.set_title('2D slow-time spectrum  (all modes × freq)', color='#ff8c42', fontsize=8, pad=2)
        self.ax_hmap.set_xlabel('Slow-time freq (J)', color=TEXT_DIM, fontsize=7)
        self.ax_hmap.set_ylabel('Mode index μ', color=TEXT_DIM, fontsize=7)
        self.ax_hmap.tick_params(colors=TEXT_COL, labelsize=6)
        for sp in self.ax_hmap.spines.values(): sp.set_edgecolor('#3a4560')
        for ax, col, ttl, xl, yl in [
            (self.ax_ycut_pwr, '#c8d0e7', 'Y-cut power  (freq=0)',  'Mode μ', 'Power (dB)'),
            (self.ax_ycut_phs, '#c8d0e7', 'Y-cut phase  (freq=0)',  'Mode μ', 'Phase (rad)'),
        ]:
            ax.set_facecolor(PANEL_BG)
            ax.set_title(ttl, color=col, fontsize=8, pad=2)
            ax.set_xlabel(xl, color=TEXT_DIM, fontsize=7)
            ax.set_ylabel(yl, color=TEXT_DIM, fontsize=7)
            ax.tick_params(colors=TEXT_COL, labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.grid(True, color=GRID_COL, lw=0.4, alpha=0.4)

        # Sweep indicator static plots (drawn once)
        _pump_col = '#4a9eff';  _side_col = '#ff4a6e'
        _step_idx = p['step_idx']
        def _norm(arr):
            if arr is None or len(arr) == 0: return np.zeros(1)
            mx = float(np.max(arr)) or 1.
            return np.asarray(arr, float) / mx
        for ax, arr, col, ttl in [
            (ax_pp,  p.get('sweep_pump_out'), _pump_col, 'Pump power  (output ring)'),
            (ax_ppa, p.get('sweep_pump_all'), _pump_col, 'Pump power  (all rings)'),
            (ax_sp,  p.get('sweep_side_out'), _side_col, 'Comb power excl. pump  (output ring)'),
            (ax_spa, p.get('sweep_side_all'), _side_col, 'Comb power excl. pump  (all rings)'),
        ]:
            ax.set_facecolor(PANEL_BG)
            y = _norm(arr)
            ax.plot(y, color=col, lw=0.9)
            ax.axvline(_step_idx, color='white', lw=0.8, ls='--', alpha=0.7)
            ax.set_xlim(0, max(len(y) - 1, 1)); ax.set_ylim(0, 1.05)
            ax.set_title(ttl, color=col, fontsize=7, pad=2)
            ax.set_xlabel('Step', color=TEXT_DIM, fontsize=6)
            ax.set_ylabel('Norm. power', color=TEXT_DIM, fontsize=6)
            ax.tick_params(colors=TEXT_COL, labelsize=5)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.grid(True, color=GRID_COL, lw=0.3, alpha=0.4)

        root.addWidget(fig_grid, stretch=1)

        # ── Controls ──────────────────────────────────────────────────
        ctrl = QWidget(); cl = QHBoxLayout(ctrl)
        cl.setSpacing(8); cl.setContentsMargins(0, 0, 0, 0)

        # Inherited params
        gi = QGroupBox('Inherited parameters'); gi.setMaximumWidth(500)
        il = QGridLayout(gi); il.setSpacing(4)
        def _ro(txt, col='#c8d0e7'):
            lb = QLabel(txt); lb.setStyleSheet(f'color:{col};font-size:12px;'); return lb
        il.addWidget(_ro(f'Δ = {p["detuning"]:.6f} J'),   0, 0)
        il.addWidget(_ro(f'F = {p["F"]:.4f}'),             0, 1)
        il.addWidget(_ro(f'kin = {p["kin"]:.4f}'),         0, 2)
        il.addWidget(_ro(f'kex = {p["kex"]:.4f}'),         1, 0)
        il.addWidget(_ro(f'D2 = {p["D2"]:.4f}'),           1, 1)
        il.addWidget(_ro(f'dt = {p["TStep"]:.3f} J⁻¹'),   1, 2)
        if p.get('psi'):
            il.addWidget(_ro(f'ψ = {p["psi"]/np.pi:.4f} π'), 2, 0)
        for ci, (k, v) in enumerate(p.get('heaters', {}).items()):
            il.addWidget(_ro(f'H{k.split("_")[-1]}={v:+.3f}J', '#ffcc00'), 2, ci)
        cl.addWidget(gi)

        # Run controls
        gr = QGroupBox('Run'); gr.setFixedWidth(380)
        rl = QGridLayout(gr); rl.setSpacing(4)

        def _lbl(txt):
            l = QLabel(txt); l.setStyleSheet('color:#4a5270;font-size:12px;'); return l

        self.spn_niter = QSpinBox()
        self.spn_niter.setRange(2000, 500000); self.spn_niter.setValue(p['NIter_more'])
        self.spn_niter.setSingleStep(1000)
        self.spn_every = QSpinBox()
        self.spn_every.setRange(1, 1000); self.spn_every.setValue(1)
        self.spn_every.setSingleStep(1)
        self.spn_every.setToolTip('Save every N-th iteration (1 = every iteration)')

        self.btn_run  = QPushButton('▶  Run');   self.btn_run.setFixedHeight(28)
        self.btn_stop = QPushButton('■  Stop');  self.btn_stop.setFixedHeight(28)
        self.btn_stop.setEnabled(False)
        self.btn_save = QPushButton('💾  Save'); self.btn_save.setFixedHeight(28)
        self.prog     = QProgressBar(); self.prog.setRange(0, 100); self.prog.setValue(0)
        self.lbl_eta  = QLabel(''); self.lbl_eta.setStyleSheet('color:#4a5270;font-size:12px;')

        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop.clicked.connect(lambda: self._stop.set())
        self.btn_save.clicked.connect(self._on_save)

        rl.addWidget(_lbl('NIter:'),      0, 0); rl.addWidget(self.spn_niter, 0, 1)
        rl.addWidget(_lbl('Save every:'), 0, 2); rl.addWidget(self.spn_every, 0, 3)
        rl.addWidget(self.btn_run,        1, 0); rl.addWidget(self.btn_stop,  1, 1)
        rl.addWidget(self.btn_save,       1, 2, 1, 2)
        rl.addWidget(self.prog,           2, 0, 1, 3)
        rl.addWidget(self.lbl_eta,        2, 3)
        cl.addWidget(gr)

        # Analysis parameters (set before running)
        ga = QGroupBox('Analysis Parameters'); ga.setFixedWidth(420)
        al = QGridLayout(ga); al.setSpacing(4)

        self.spn_fsr_shift = QDoubleSpinBox()
        self.spn_fsr_shift.setRange(0.001, 1000.0); self.spn_fsr_shift.setValue(30.0)
        self.spn_fsr_shift.setSingleStep(1.0); self.spn_fsr_shift.setDecimals(3)
        self.spn_fsr_shift.setToolTip('FSR of a single ring in units of J (shift_factor)')
        self.spn_fsr_shift.valueChanged.connect(lambda _: self._replot_all())

        self.spn_mode_idx = QSpinBox()
        self.spn_mode_idx.setRange(-(p['one_side_fsr']), p['one_side_fsr'])
        self.spn_mode_idx.setValue(4)
        self.spn_mode_idx.setToolTip('Mode index relative to pump to zoom into')
        self.spn_mode_idx.valueChanged.connect(lambda _: self._replot_all())

        self.spn_nest_range = QDoubleSpinBox()
        self.spn_nest_range.setRange(0.01, 100.0); self.spn_nest_range.setValue(2.0)
        self.spn_nest_range.setSingleStep(0.5); self.spn_nest_range.setDecimals(2)
        self.spn_nest_range.setToolTip('Frequency zoom range ±X J for zoomed spectrum and heatmap')
        self.spn_nest_range.valueChanged.connect(lambda _: self._replot_zoom_and_hmap())

        self.spn_ycut_freq = QDoubleSpinBox()
        self.spn_ycut_freq.setRange(-100.0, 100.0); self.spn_ycut_freq.setValue(0.0)
        self.spn_ycut_freq.setSingleStep(0.05); self.spn_ycut_freq.setDecimals(3)
        self.spn_ycut_freq.setToolTip('Frequency at which to take the Y-cut through the 2D heatmap')
        self.spn_ycut_freq.valueChanged.connect(lambda _: self._replot_ycut())

        self.btn_replot = QPushButton('⟳  Replot')
        self.btn_replot.setFixedHeight(28)
        self.btn_replot.setStyleSheet(
            'QPushButton{color:#00e5ff;border-color:#00e5ff;}'
            'QPushButton:hover{background:#1e2a40;}')
        self.btn_replot.clicked.connect(self._replot_all)

        al.addWidget(_lbl('FSR / J (shift):'),  0, 0); al.addWidget(self.spn_fsr_shift,  0, 1)
        al.addWidget(_lbl('Zoom mode index:'),   0, 2); al.addWidget(self.spn_mode_idx,   0, 3)
        al.addWidget(_lbl('Nest range ±J:'),     1, 0); al.addWidget(self.spn_nest_range, 1, 1)
        al.addWidget(_lbl('Y-cut freq (J):'),    1, 2); al.addWidget(self.spn_ycut_freq,  1, 3)
        al.addWidget(self.btn_replot,            2, 0, 1, 4)
        cl.addWidget(ga)

        # Visualization controls
        gv = QGroupBox('Visualization'); gv.setFixedWidth(340)
        vl = QGridLayout(gv); vl.setSpacing(4)

        self.spn_vis_step = QSpinBox()
        self.spn_vis_step.setRange(0, 99999); self.spn_vis_step.setValue(0)
        self.spn_vis_step.setToolTip('Slow-time iteration index to visualize')

        self.btn_vis_plot  = QPushButton('▶  Plot');   self.btn_vis_plot.setFixedHeight(26)
        self.btn_vis_save  = QPushButton('💾  Save');  self.btn_vis_save.setFixedHeight(26)
        self.btn_vis_gif   = QPushButton('🎞  GIF');   self.btn_vis_gif.setFixedHeight(26)
        self.btn_vis_mp4   = QPushButton('🎬  MP4');   self.btn_vis_mp4.setFixedHeight(26)
        self.lbl_vis_status = QLabel('')
        self.lbl_vis_status.setStyleSheet('color:#4a5270;font-size:11px;')

        self.btn_vis_plot.clicked.connect(self._vis_plot)
        self.btn_vis_save.clicked.connect(self._vis_save_step)
        self.btn_vis_gif.clicked.connect(self._vis_export_gif)
        self.btn_vis_mp4.clicked.connect(self._vis_export_mp4)

        vl.addWidget(_lbl('Step:'),        0, 0); vl.addWidget(self.spn_vis_step,  0, 1, 1, 3)
        vl.addWidget(self.btn_vis_plot,    1, 0, 1, 2); vl.addWidget(self.btn_vis_save, 1, 2, 1, 2)
        vl.addWidget(self.btn_vis_gif,     2, 0, 1, 2); vl.addWidget(self.btn_vis_mp4,  2, 2, 1, 2)
        vl.addWidget(self.lbl_vis_status,  3, 0, 1, 4)
        cl.addWidget(gv)

        # Exit / navigation panel
        ge = QGroupBox('Exit'); ge.setFixedWidth(160)
        ge.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        el = QGridLayout(ge); el.setSpacing(4)
        el.setRowStretch(0, 1); el.setRowStretch(1, 1)

        self.btn_vis_exit = QPushButton('✕  Exit')
        self.btn_vis_exit.setStyleSheet('color:#ff4a6e;border-color:#ff4a6e;')
        self.btn_vis_exit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.btn_vis_prev = QPushButton('◀ Prev')
        self.btn_vis_prev.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.btn_vis_next = QPushButton('Next ▶')
        self.btn_vis_next.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.btn_vis_exit.clicked.connect(self._on_exit)
        self.btn_vis_prev.clicked.connect(lambda: self._vis_nav(-1))
        self.btn_vis_next.clicked.connect(lambda: self._vis_nav(+1))

        el.addWidget(self.btn_vis_exit, 0, 0, 1, 2)
        el.addWidget(self.btn_vis_prev, 1, 0)
        el.addWidget(self.btn_vis_next, 1, 1)
        cl.addWidget(ge)

        cl.addStretch()
        root.addWidget(ctrl)

        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.status.showMessage('Ready — press Run to start slow-time integration')

    # ------------------------------------------------------------------
    def _on_run(self):
        import time
        p      = self._p
        niter  = self.spn_niter.value()
        every  = self.spn_every.value()
        n_snap = niter // every

        self.btn_run.setEnabled(False); self.btn_stop.setEnabled(True)
        self.prog.setValue(0); self._stop.clear(); self._hist = None

        DispMat, LM, FSRVec, PumpFSR, NumFSRs = build_disp_loss(
            p['NLattice'], p['one_side_fsr'], p['D2'],
            p['kin'], p['kex'], p['ISite'], p['OSite'])

        # For cylinder lattice: always rebuild H with the correct swept ψ at this step.
        psi_val = p.get('psi', 0.0)
        ls      = p.get('linear_state', {})
        if ls.get('is_cyl') and build_hamiltonian is not None:
            H_full = build_hamiltonian(
                ls['h_str'], ls['nx0'], ls['ny0'], ls['nx1'], ls['ny1'],
                ls.get('j1', 0.3), ls.get('phi_iqh0', np.pi/2), ls.get('phi_iqh1', np.pi/2),
                ls.get('phi_aqh0', np.pi/4), ls.get('phi_aqh1', np.pi/4), psi_val)
            H = H_full[1:, 1:].copy()
        else:
            H = p['H_mat'].copy()
        for k, v in p.get('heaters', {}).items():
            n = int(k.split('_')[1]); H[n-1, n-1] += v

        # Log all parameters for verification
        h_trace = f'H[0,0]={H[0,0]:.4f}' + (f'  H[1,0]={H[1,0]:.4f}' if H.shape[0] > 1 else '')
        self.status.showMessage(
            f'Running — Δ={p["detuning"]:.5f}  F={p["F"]:.4f}  '
            f'ψ={psi_val/np.pi:.4f}π  kin={p["kin"]:.4f}  kex={p["kex"]:.4f}  '
            f'D2={p["D2"]:.4f}  dt={p["TStep"]:.3f}  {h_trace}')

        props, fc_arr, _ = build_propagators(
            H, DispMat, LM, p['detuning'], p['TStep'],
            p['F'], PumpFSR, p['ISite'])
        props_j = jnp.array(props); fc_j = jnp.array(fc_arr)
        # run 1 iteration at a time so we can save every snapshot
        stepper1 = get_stepper(1)

        a_T = jnp.array(p['a_T_init'])   # (NLattice, NumFSRs)
        _q  = self._q

        def worker():
            nonlocal a_T
            # pre-allocate history: (NLattice, NumFSRs, n_snap)
            NL, NF = p['NLattice'], NumFSRs
            hist = np.empty((NL, NF, n_snap), dtype=np.complex128)
            t0   = time.time()
            snap = 0
            try:
                for i in range(niter):
                    if self._stop.is_set():
                        _q.put(('stopped', hist[:, :, :snap], snap)); return
                    a_T = stepper1(a_T, props_j, fc_j, p['TStep'])
                    if (i + 1) % every == 0:
                        hist[:, :, snap] = np.array(a_T)
                        snap += 1
                    if i % max(1, niter // 200) == 0:
                        elapsed = time.time() - t0
                        _q.put(('progress', i + 1, niter, snap, elapsed))
                _q.put(('done', hist, snap))
            except Exception as exc:
                import traceback; traceback.print_exc()
                _q.put(('stopped', hist[:, :, :snap], snap))

        threading.Thread(target=worker, daemon=True).start()
        self._poll_timer.start()

    # ------------------------------------------------------------------
    def _poll(self):
        try:
            while True:
                msg  = self._q.get_nowait()
                kind = msg[0]
                if kind == 'progress':
                    _, done, total, snap, elapsed = msg
                    pct = int(100 * done / total)
                    self.prog.setValue(pct)
                    rem = elapsed / done * (total - done) if done else 0
                    em, es = divmod(int(rem), 60)
                    self.lbl_eta.setText(f'{em}m{es:02d}s')
                    self.status.showMessage(
                        f'Iteration {done}/{total}  —  {snap} snapshots saved  —  ETA {em}m{es:02d}s')
                elif kind in ('done', 'stopped'):
                    _, hist, snap = msg
                    self._poll_timer.stop()
                    self.btn_run.setEnabled(True); self.btn_stop.setEnabled(False)
                    self.prog.setValue(100 if kind == 'done' else self.prog.value())
                    txt = f'Done — {snap} snapshots.' if kind == 'done' else f'Stopped — {snap} snapshots.'
                    self.status.showMessage(txt)
                    if snap > 0:
                        self._hist = hist[:, :, :snap]
                        self.spn_vis_step.setRange(0, snap - 1)
                        self.spn_vis_step.setValue(snap - 1)
                        self.status.showMessage('Plotting…')
                        QApplication.processEvents()
                        self._draw_all()
        except queue.Empty:
            pass


# =====================================================================
# SLOW TIME WINDOW (continued methods — same class via monkey-patching)
# =====================================================================
class _SlowTimeWindowMethods:
    def _draw_all(self):
        # Set interrupt flag for Exit button
        self._plotting = True

        # Collect all interactive widgets to disable
        self._plot_widgets = [
            self.btn_run, self.btn_stop, self.btn_save,
            self.btn_replot, self.btn_vis_plot, self.btn_vis_save,
            self.btn_vis_gif, self.btn_vis_mp4,
            self.btn_vis_prev, self.btn_vis_next,
            self.spn_niter, self.spn_every,
            self.spn_fsr_shift, self.spn_mode_idx,
            self.spn_nest_range, self.spn_ycut_freq,
            self.spn_fsr_shift, self.spn_vis_step,
        ]
        for w in self._plot_widgets:
            w.setEnabled(False)
        # Exit always enabled — clicking it sets _plotting=False and closes
        self.btn_vis_exit.setEnabled(True)

        steps_total = 6
        def _progress(step, label):
            if not self._plotting: return   # interrupted by Exit
            pct = int(100 * step / steps_total)
            self.prog.setValue(pct)
            self.status.showMessage(f'Plotting… {label}')
            QApplication.processEvents()

        # Show "Plotting…" on all panels except the static sweep indicator
        plotting_panels = [
            ([self.ax_aT_ring, self.ax_aT_all, self.ax_aW_ring, self.ax_aW_all], self.canvas_heat4),
            ([self.ax_comb_dB, self.ax_comb_zoom],                                self.canvas_comb),
            ([self.ax_vis_2d],                                                     self.canvas_2dpow),
            ([self.ax_esa_dB, self.ax_esa_lin, self.ax_osc],                      self.canvas_esa),
            ([self.ax_hmap, self.ax_ycut_pwr, self.ax_ycut_phs],                  self.canvas_spec2d),
            ([self.ax_vis_flow],                                                   self.canvas_flow),
        ]
        for axes, canvas in plotting_panels:
            for ax in axes:
                ax.cla(); ax.set_facecolor(PANEL_BG)
                ax.text(0.5, 0.5, 'Plotting…', transform=ax.transAxes,
                        ha='center', va='center', color=TEXT_COL, fontsize=11)
            canvas.draw_idle()
        # 3D axes needs x,y,z coords, not transAxes
        self.ax_vis_3d.cla(); self.ax_vis_3d.set_facecolor('black')
        self.ax_vis_3d.text(0.5, 0.5, 0.5, 'Plotting…', ha='center', va='center',
                            color=TEXT_COL, fontsize=11)
        self.canvas_3d.draw_idle()
        QApplication.processEvents()

        hist     = self._hist                       # (NLattice, NumFSRs, NS)
        p        = self._p
        OSite_i  = p['OSite'] - 1
        FSRVec   = p['FSRVec']
        FSR_s    = np.sort(FSRVec)
        sort_i   = np.argsort(FSRVec)
        one_side = p['one_side_fsr']
        PumpFSR  = one_side                         # 0-indexed
        NumFSRs  = len(FSRVec)
        NL, NF, NS = hist.shape
        every    = self.spn_every.value()
        dt       = p['TStep'] * every
        x_iters  = np.arange(NS) * every

        shift_factor  = self.spn_fsr_shift.value()
        mode_index    = self.spn_mode_idx.value()   # relative to pump
        nest_range    = self.spn_nest_range.value()
        index         = mode_index + PumpFSR        # absolute array index
        mode_indices  = np.arange(NumFSRs) - PumpFSR

        # ── shared data ───────────────────────────────────────────────
        # time-domain
        aT_ring = np.abs(hist[OSite_i, :, :])**2                    # (NF, NS)
        aT_all  = np.sum(np.abs(hist)**2, axis=0)                   # (NF, NS)

        # frequency-domain (FFT along FSR axis, sort FSR bins)
        aW_full     = np.fft.fft(hist, axis=1, norm='ortho')        # (NL, NF, NS)
        aW_ring_dB  = 10*np.log10(np.abs(aW_full[OSite_i, sort_i, :])**2 + 1e-200)
        aW_all_dB   = 10*np.log10(np.sum(np.abs(aW_full[:, sort_i, :])**2, axis=0) + 1e-200)

        theta_idx = np.arange(NF)

        # slow-time FFT of each comb tooth's complex amplitude
        a_W_ring    = aW_full[OSite_i, :, :]                        # (NF, NS)
        freq_J      = np.fft.fftshift(np.fft.fftfreq(NS, d=dt)) * 2*np.pi
        A_freq      = np.fft.fftshift(np.fft.fft(a_W_ring, axis=1), axes=1)
        power_slow  = np.abs(A_freq)**2
        power_slow_dB = 10*np.log10(power_slow + 1e-200)

        freq_mask     = (freq_J >= -nest_range) & (freq_J <= nest_range)
        freq_trimmed  = freq_J[freq_mask]
        pwr_dB_trim   = power_slow_dB[:, freq_mask]
        pwr_trim      = power_slow[:, freq_mask]

        # y-cut at freq=0
        freq_idx      = np.argmin(np.abs(freq_trimmed))
        pwr_ycut_dB   = pwr_dB_trim[:, freq_idx]
        A_ycut        = A_freq[:, freq_mask][:, freq_idx]
        phase_ycut    = np.unwrap(np.angle(A_ycut))

        # ESA: FFT of intensity |a_T|² over slow time
        a_T_ring_raw  = hist[OSite_i, :, :]                         # (NF, NS) complex
        I_t_modes     = np.abs(a_T_ring_raw)**2
        freq_esa      = np.fft.rfftfreq(NS, d=dt) * 2*np.pi
        spec_esa      = np.fft.rfft(I_t_modes, axis=1)
        pwr_esa       = np.abs(spec_esa)**2
        total_pwr_esa = np.sum(pwr_esa, axis=0)
        total_pwr_esa_dB = 10*np.log10(total_pwr_esa + 1e-20)

        # oscilloscope: total |a_W|² vs slow time (sum over all modes)
        I_w_ring = np.abs(a_W_ring)**2                              # (NF, NS)
        pulse    = np.sum(I_w_ring, axis=0)
        pulse    = pulse / (pulse.max() or 1.)

        # ── helper ────────────────────────────────────────────────────
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset_axes

        def _pcm(ax, Z, xv, yv, cmap, col, ttl, xl, yl):
            ax.cla(); ax.set_facecolor(PANEL_BG)
            vmax = float(Z.max())
            vmin = vmax - 60 if cmap == 'hot' else 0.
            Zn   = Z if cmap == 'hot' else Z / (Z.max() or 1.)
            dx   = (xv[1]-xv[0]) if len(xv) > 1 else max(every, 1)
            dy   = yv[1]-yv[0] if len(yv) > 1 else 1.
            xe   = np.append(xv, xv[-1]+dx)
            ye   = np.append(yv, yv[-1]+dy)
            mesh = ax.pcolormesh(xe, ye, Zn, cmap=cmap,
                          vmin=vmin if cmap=='hot' else 0,
                          vmax=vmax if cmap=='hot' else 1,
                          shading='flat')
            ax.set_title(ttl, color=col, fontsize=8, pad=2)
            ax.set_xlabel(xl, color=TEXT_DIM, fontsize=7)
            ax.set_ylabel(yl, color=TEXT_DIM, fontsize=7)
            ax.tick_params(colors=TEXT_COL, labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            # x-axis: leftmost and rightmost only (iteration/step axis)
            ax.set_xticks([xv[0], xv[-1]])
            ax.set_xticklabels([f'{int(xv[0])}', f'{int(xv[-1])}'], fontsize=6)
            # y-axis ticks
            if cmap == 'hot':   # a_W: 3 ticks at -FSR, 0, +FSR
                fsr = one_side
                ax.set_yticks([-fsr, 0, fsr])
                ax.set_yticklabels([f'{-fsr}', '0', f'{fsr}'], fontsize=6)
            else:               # a_T: 2 ticks at 0 and 2π
                ax.set_yticks([yv[0], yv[-1]])
                ax.set_yticklabels(['0', '2π'], fontsize=6)
            return mesh

        def _add_cbar(ax, mesh):
            """Add a slim inset colorbar with only min and max ticks.
            Remove any previously added inset axes first to avoid pileup on re-run."""
            # Remove old inset axes parented to this ax (ax.cla() does not remove them)
            for child in list(ax.child_axes if hasattr(ax, 'child_axes') else []):
                child.remove()
            cax = _inset_axes(ax, width='4%', height='90%',
                              loc='center right',
                              bbox_to_anchor=(0, 0, 1.06, 1),
                              bbox_transform=ax.transAxes,
                              borderpad=0)
            cb = self.fig_heat4.colorbar(mesh, cax=cax)
            clim = mesh.get_clim()
            cb.set_ticks([clim[0], clim[1]])
            cb.set_ticklabels([f'{clim[0]:.0f}', f'{clim[1]:.0f}'])
            cb.ax.tick_params(colors=TEXT_COL, labelsize=5, pad=2)
            for lbl in cb.ax.get_yticklabels():
                lbl.set_color(TEXT_COL); lbl.set_fontsize(5); lbl.set_clip_on(False)

        # cache everything needed for live replot (no recalc)
        self._slow_cache = dict(
            freq_J=freq_J, power_slow=power_slow, power_slow_dB=power_slow_dB,
            mode_indices=mode_indices, one_side=one_side,
            pwr_dB_trim=pwr_dB_trim, freq_trimmed=freq_trimmed,
            A_freq=A_freq,
        )

        # ── Block 1: comb dB + zoomed tooth ──────────────────────────
        _progress(1, 'comb spectrum')
        if not self._plotting: return
        self._draw_comb()
        self._draw_zoom()

        # ── Block 2: ESA dB + ESA linear + oscilloscope ───────────────
        _progress(2, 'ESA & oscilloscope')
        if not self._plotting: return
        df = 2*np.pi / (NS * dt)
        self.ax_esa_dB.cla(); self.ax_esa_dB.set_facecolor(PANEL_BG)
        self.ax_esa_dB.plot(freq_esa, total_pwr_esa_dB, color='#a8ff78', lw=1.2)
        self.ax_esa_dB.set_xlim([0, 4]); self.ax_esa_dB.set_xlabel('Angular freq (J)', color=TEXT_DIM, fontsize=7)
        self.ax_esa_dB.set_ylabel('Power (dB)', color=TEXT_DIM, fontsize=7)
        self.ax_esa_dB.set_title(f'ESA (dB)  Δf≈{df:.3f} J', color='#a8ff78', fontsize=8, pad=2)
        self.ax_esa_dB.tick_params(colors=TEXT_COL, labelsize=6)
        for sp in self.ax_esa_dB.spines.values(): sp.set_edgecolor('#3a4560')
        self.ax_esa_dB.grid(True, color=GRID_COL, lw=0.4, alpha=0.4)

        self.ax_esa_lin.cla(); self.ax_esa_lin.set_facecolor(PANEL_BG)
        self.ax_esa_lin.plot(freq_esa, total_pwr_esa/(total_pwr_esa.max() or 1.),
                             color='#a8ff78', lw=1.2)
        self.ax_esa_lin.set_xlim([0, 4]); self.ax_esa_lin.set_ylim([0, 0.5])
        self.ax_esa_lin.set_xlabel('Angular freq (J)', color=TEXT_DIM, fontsize=7)
        self.ax_esa_lin.set_ylabel('Norm. power', color=TEXT_DIM, fontsize=7)
        self.ax_esa_lin.set_title('ESA (linear)', color='#a8ff78', fontsize=8, pad=2)
        self.ax_esa_lin.tick_params(colors=TEXT_COL, labelsize=6)
        for sp in self.ax_esa_lin.spines.values(): sp.set_edgecolor('#3a4560')
        self.ax_esa_lin.grid(True, color=GRID_COL, lw=0.4, alpha=0.4)

        self.ax_osc.cla(); self.ax_osc.set_facecolor(PANEL_BG)
        self.ax_osc.plot(x_iters, pulse, color='#c8d0e7', lw=1.0)
        self.ax_osc.set_xlim([0, max(x_iters[-1], 1)]); self.ax_osc.set_ylim([0, 1.1])
        self.ax_osc.set_xlabel('Iteration', color=TEXT_DIM, fontsize=7)
        self.ax_osc.set_ylabel('Norm. power', color=TEXT_DIM, fontsize=7)
        self.ax_osc.set_title(f'Total |a_W|² vs slow time  (osite)', color='#c8d0e7', fontsize=8, pad=2)
        self.ax_osc.tick_params(colors=TEXT_COL, labelsize=6)
        for sp in self.ax_osc.spines.values(): sp.set_edgecolor('#3a4560')
        self.ax_osc.grid(True, color=GRID_COL, lw=0.4, alpha=0.4)
        self.canvas_esa.draw_idle()

        # ── Block 3: 2D heatmap + y-cuts ─────────────────────────────
        _progress(3, '2D slow-time spectrum')
        if not self._plotting: return
        # Reset colorbar ref so _draw_hmap_and_ycuts recreates the inset axes fresh.
        # ax_hmap.cla() (called inside) destroys the old inset axes, leaving _cbar
        # pointing at a dead Axes — update_normal() on it crashes on second run.
        self._cbar = None; self.ax_cbar = None
        self._draw_hmap_and_ycuts()

        # ── Block 4: 4 heatmaps (slowest) ────────────────────────────
        _progress(4, '|a_T|² / |a_W|² heatmaps')
        if not self._plotting: return
        _pcm(self.ax_aT_ring, aT_ring,   x_iters, theta_idx, 'viridis',
             '#a8ff78', '|a_T|²  ring',      'Iteration', 'θ')
        m_aT = _pcm(self.ax_aT_all,  aT_all,    x_iters, theta_idx, 'viridis',
             '#a8ff78', '|a_T|²  all (sum)', 'Iteration', 'θ')
        _pcm(self.ax_aW_ring, aW_ring_dB, x_iters, FSR_s, 'hot',
             '#ff8c42', '10·log|a_W|²  ring',      'Iteration', 'FSR μ')
        m_aW = _pcm(self.ax_aW_all,  aW_all_dB,  x_iters, FSR_s, 'hot',
             '#ff8c42', '10·log|a_W|²  all (sum)',  'Iteration', 'FSR μ')
        _add_cbar(self.ax_aT_all, m_aT)
        _add_cbar(self.ax_aW_all, m_aW)
        self.canvas_heat4.draw_idle()

        # show last snapshot in the visualization panels
        _progress(5, 'visualization panels')
        if not self._plotting: return
        last_step = self._hist.shape[2] - 1
        self._vis_frame(last_step)

        # Done — restore all widgets
        self._plotting = False
        _progress(6, 'done')
        self.prog.setValue(100)
        for w in self._plot_widgets:
            w.setEnabled(True)
        self.btn_stop.setEnabled(False)   # stop only active during a run
        self.status.showMessage(f'✅ Plotting complete — {self._hist.shape[2]} snapshots')

    # ------------------------------------------------------------------
    # Focused redraw helpers — no recalculation, use self._slow_cache
    # ------------------------------------------------------------------
    def _draw_comb(self):
        if not hasattr(self, '_slow_cache'): return
        c = self._slow_cache
        freq_J       = c['freq_J']
        power_slow_dB= c['power_slow_dB']
        mode_indices = c['mode_indices']
        one_side     = c['one_side']
        shift_factor = self.spn_fsr_shift.value()
        mode_index   = self.spn_mode_idx.value()
        PumpFSR      = self._p['one_side_fsr']
        index        = mode_index + PumpFSR

        ax = self.ax_comb_dB
        ax.cla(); ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
        ax.grid(True, color=GRID_COL, lw=0.4, ls='--', alpha=0.5)

        for i, m in enumerate(mode_indices):
            # selected tooth bright cyan, others warm amber — both visible on dark bg
            col = '#00e5ff' if i == index else '#c87020'
            lw  = 1.4       if i == index else 0.7
            ax.plot(freq_J + shift_factor*(m+1), power_slow_dB[i], color=col, lw=lw)

        xlim = [-shift_factor*(one_side+1), shift_factor*(one_side+1)]
        ax.set_xlim(xlim)
        ax.set_ylim([power_slow_dB.max()-80, power_slow_dB.max()+5])
        ax.set_title(f'Comb slow-time spectrum (dB)  — click tooth to zoom  |  μ={mode_index}',
                     color='#c8d0e7', fontsize=8, pad=2)
        ax.set_xlabel(f'Freq  (FSR={shift_factor} J)', color=TEXT_DIM, fontsize=7)
        ax.set_ylabel('Power (dB)', color=TEXT_DIM, fontsize=7)
        self.canvas_comb.draw_idle()

    def _draw_zoom(self):
        if not hasattr(self, '_slow_cache'): return
        c = self._slow_cache
        freq_J        = c['freq_J']
        power_slow_dB = c['power_slow_dB']
        mode_index    = self.spn_mode_idx.value()
        nest_range    = self.spn_nest_range.value()
        PumpFSR       = self._p['one_side_fsr']
        index         = mode_index + PumpFSR

        ax = self.ax_comb_zoom
        ax.cla(); ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
        ax.grid(True, color=GRID_COL, lw=0.4, ls='--', alpha=0.5)

        ax.plot(freq_J, power_slow_dB[index], color='#4a9eff', lw=1.4)
        ax.set_xlim([-nest_range, nest_range])
        ax.set_title(f'Zoomed spectrum — mode μ={mode_index} (dB)', color='#4a9eff', fontsize=8, pad=2)
        ax.set_xlabel('Slow-time freq (J)', color=TEXT_DIM, fontsize=7)
        ax.set_ylabel('Power (dB)', color=TEXT_DIM, fontsize=7)
        self.canvas_comb.draw_idle()

    def _draw_hmap_and_ycuts(self):
        if not hasattr(self, '_slow_cache'): return
        c = self._slow_cache
        nest_range   = self.spn_nest_range.value()
        ycut_freq    = self.spn_ycut_freq.value()
        freq_J       = c['freq_J']
        A_freq       = c['A_freq']
        power_slow_dB= c['power_slow_dB']
        mode_indices = c['mode_indices']

        # recompute trimmed window with current nest_range
        freq_mask    = (freq_J >= -nest_range) & (freq_J <= nest_range)
        freq_trimmed = freq_J[freq_mask]
        pwr_dB_trim  = power_slow_dB[:, freq_mask]

        # y-cut at chosen frequency
        if len(freq_trimmed) > 0:
            freq_idx   = np.argmin(np.abs(freq_trimmed - ycut_freq))
            actual_yf  = freq_trimmed[freq_idx]
            # use A_freq (already fftshifted) for phase — find column in full freq_J
            full_idx   = np.argmin(np.abs(freq_J - ycut_freq))
            pwr_ycut_dB = pwr_dB_trim[:, freq_idx]
            A_ycut      = A_freq[:, full_idx]
            phase_ycut  = np.unwrap(np.angle(A_ycut))
        else:
            actual_yf = ycut_freq
            pwr_ycut_dB = np.zeros(len(mode_indices))
            phase_ycut  = np.zeros(len(mode_indices))

        # heatmap
        self.ax_hmap.cla(); self.ax_hmap.set_facecolor(PANEL_BG)
        if len(freq_trimmed) > 1:
            ext = [freq_trimmed[0], freq_trimmed[-1], mode_indices[0], mode_indices[-1]]
            im  = self.ax_hmap.imshow(pwr_dB_trim[:, ::-1], aspect='auto',
                                      extent=ext, origin='lower', cmap='inferno')
            self.ax_hmap.axvline(actual_yf, color='white', lw=1.0, ls='--', alpha=0.8)
            # horizontal dotted line at the selected mode index
            self.ax_hmap.axhline(self.spn_mode_idx.value(),
                                 color='#87ceeb', lw=2.0, ls=':', alpha=0.9)
            # colorbar: placed as a slim inset INSIDE the right edge of the heatmap
            # bbox_to_anchor=(0.98, 0.02, 0.02, 0.96) keeps it fully within axes bounds
            if self._cbar is None:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset_axes
                self.ax_cbar = _inset_axes(self.ax_hmap,
                                           width='3%', height='90%',
                                           loc='center right',
                                           bbox_to_anchor=(0, 0, 1.06, 1),
                                           bbox_transform=self.ax_hmap.transAxes,
                                           borderpad=0)
                self._cbar = self.fig_spec2d.colorbar(im, cax=self.ax_cbar)
                clim = im.get_clim()
                self._cbar.set_ticks([clim[0], clim[1]])
                self._cbar.set_ticklabels([f'{clim[0]:.0f}', f'{clim[1]:.0f}'])
                self._cbar.ax.tick_params(colors=TEXT_COL, labelsize=6)
                for lbl in self._cbar.ax.get_yticklabels():
                    lbl.set_color(TEXT_COL); lbl.set_fontsize(6); lbl.set_clip_on(False)
            else:
                self._cbar.update_normal(im)
                clim = im.get_clim()
                self._cbar.set_ticks([clim[0], clim[1]])
                self._cbar.set_ticklabels([f'{clim[0]:.0f}', f'{clim[1]:.0f}'])
                self._cbar.ax.tick_params(colors=TEXT_COL, labelsize=6, pad=2)
                for lbl in self._cbar.ax.get_yticklabels():
                    lbl.set_color(TEXT_COL); lbl.set_fontsize(6); lbl.set_clip_on(False)
        self.ax_hmap.set_title('2D slow-time spectrum  (all modes × freq)',
                               color='#ff8c42', fontsize=8, pad=2)
        self.ax_hmap.set_xlabel('Slow-time freq (J)', color=TEXT_DIM, fontsize=7)
        self.ax_hmap.set_ylabel('Mode index μ', color=TEXT_DIM, fontsize=7)
        self.ax_hmap.tick_params(colors=TEXT_COL, labelsize=6)
        for sp in self.ax_hmap.spines.values(): sp.set_edgecolor('#3a4560')

        # y-cut power
        self.ax_ycut_pwr.cla(); self.ax_ycut_pwr.set_facecolor(PANEL_BG)
        self.ax_ycut_pwr.scatter(mode_indices, pwr_ycut_dB, color='#c8d0e7', s=4)
        self.ax_ycut_pwr.set_title(f'Y-cut power  (freq≈{actual_yf:.3f} J)',
                                   color='#c8d0e7', fontsize=8, pad=2)
        self.ax_ycut_pwr.set_xlabel('Mode μ', color=TEXT_DIM, fontsize=7)
        self.ax_ycut_pwr.set_ylabel('Power (dB)', color=TEXT_DIM, fontsize=7)
        self.ax_ycut_pwr.tick_params(colors=TEXT_COL, labelsize=6)
        for sp in self.ax_ycut_pwr.spines.values(): sp.set_edgecolor('#3a4560')
        self.ax_ycut_pwr.grid(True, color=GRID_COL, lw=0.4, alpha=0.4)

        # y-cut phase
        self.ax_ycut_phs.cla(); self.ax_ycut_phs.set_facecolor(PANEL_BG)
        self.ax_ycut_phs.scatter(mode_indices, phase_ycut, color='#c8d0e7', s=4)
        self.ax_ycut_phs.set_title(f'Y-cut phase  (freq≈{actual_yf:.3f} J)',
                                   color='#c8d0e7', fontsize=8, pad=2)
        self.ax_ycut_phs.set_xlabel('Mode μ', color=TEXT_DIM, fontsize=7)
        self.ax_ycut_phs.set_ylabel('Phase (rad)', color=TEXT_DIM, fontsize=7)
        self.ax_ycut_phs.tick_params(colors=TEXT_COL, labelsize=6)
        for sp in self.ax_ycut_phs.spines.values(): sp.set_edgecolor('#3a4560')
        self.ax_ycut_phs.grid(True, color=GRID_COL, lw=0.4, alpha=0.4)

        self.canvas_spec2d.draw_idle()

    # ------------------------------------------------------------------
    # Live replot slots — triggered by spinbox changes, no recalc
    # ------------------------------------------------------------------
    def _replot_all(self):
        """Replot all analysis panels without recalculating — triggered by Replot button."""
        if not hasattr(self, '_slow_cache'): return
        self._draw_comb()
        self._draw_zoom()
        self._draw_hmap_and_ycuts()
        self._update_flow()

    def _replot_zoom(self):
        self._draw_comb(); self._draw_zoom()

    def _replot_zoom_and_hmap(self):
        self._draw_zoom(); self._draw_hmap_and_ycuts()

    def _replot_ycut(self):
        self._draw_hmap_and_ycuts()

    def _replot_comb(self):
        self._draw_comb()

    def _on_comb_click(self, event):
        """Click on the comb plot to select the nearest tooth, then replot."""
        if event.inaxes != self.ax_comb_dB: return
        if not hasattr(self, '_slow_cache'): return
        c            = self._slow_cache
        shift_factor = self.spn_fsr_shift.value()
        mode_indices = c['mode_indices']
        centers      = shift_factor * (mode_indices + 1)
        nearest      = int(mode_indices[np.argmin(np.abs(centers - event.xdata))])
        # blockSignals to avoid double-draw from valueChanged; we draw manually below
        self.spn_mode_idx.blockSignals(True)
        self.spn_mode_idx.setValue(nearest)
        self.spn_mode_idx.blockSignals(False)
        self._draw_comb(); self._draw_zoom()
        self._update_flow()

    def _update_flow(self):
        """Redraw only the photon flow panel with the current mode index and vis step."""
        if self._hist is None: return
        step = int(self.spn_vis_step.value())
        step = max(0, min(step, self._hist.shape[2] - 1))
        self._vis_frame(step)

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------
    def _vis_frame(self, step_idx):
        """Draw 2D power and photon flow — Visualization_Figures style, Linear.py colours."""
        if self._hist is None: return
        hist  = self._hist
        p     = self._p
        ls    = p.get('linear_state', {})
        NL    = p['NLattice']
        ISite = p['ISite']; OSite = p['OSite']
        step_idx = max(0, min(step_idx, hist.shape[2]-1))

        a_T_step = hist[:, :, step_idx]        # (NLattice, NumFSRs) complex
        NumFSRs  = a_T_step.shape[1]

        PumpFSR  = p['one_side_fsr']
        mode_abs = int(np.clip(self.spn_mode_idx.value() + PumpFSR, 0, NumFSRs - 1))
        field_mode = a_T_step[:, mode_abs]

        h_str = ls.get('h_str', 'IQH_IQH')
        nx0   = ls.get('nx0', 4); ny0 = ls.get('ny0', 4)
        nx1   = ls.get('nx1', 1); ny1 = ls.get('ny1', 1)
        coords  = {n: site_xy(n, h_str, nx0, ny0, nx1, ny1) for n in range(1, NL+1)}
        c2n     = {(int(round(v[0])), int(round(v[1]))): k for k, v in coords.items()}

        R_field  = RING_R
        cmap_hot = plt.cm.hot
        norm_2d  = plt.Normalize(0, np.max(np.abs(a_T_step)**2) / 1.3 or 1.)
        phi_fine = np.linspace(0, 2*np.pi, NumFSRs, endpoint=False)
        span_x   = nx1 * nx0;  span_y = ny1 * ny0

        # ── 2D power — ring arc LineCollection traces ──────────────────
        ax = self.ax_vis_2d; ax.cla(); ax.set_facecolor('black')
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'2D Power  |  step {step_idx}', color='#c8d0e7', fontsize=8, pad=2)
        for sp in ax.spines.values(): sp.set_edgecolor('#1e2230')

        if nx1 > 1 or ny1 > 1:
            from matplotlib.patches import Rectangle
            for i1 in range(nx1):
                for j1 in range(ny1):
                    fc_p = '#444444' if (i1 + j1) % 2 == 0 else '#888888'
                    ax.add_patch(Rectangle((i1*nx0 - 0.5, j1*ny0 - 0.5), nx0, ny0,
                                           linewidth=0, facecolor=fc_p, alpha=0.3))

        for i in range(NL):
            n = i + 1
            xn, yn  = coords[n]
            power_n = np.abs(a_T_step[i, :])**2
            x_arc = xn + R_field * np.cos(phi_fine)
            y_arc = yn + R_field * np.sin(phi_fine)
            pts   = np.array([x_arc, y_arc]).T.reshape(-1, 1, 2)
            segs  = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=cmap_hot, norm=norm_2d, linewidth=2.0, zorder=2)
            lc.set_array(power_n[:-1]); lc.set_rasterized(True)
            ax.add_collection(lc)

            # outline: IN=#4a9eff  OUT=#ff4a6e  others=hot cmap
            out_theta = np.linspace(0, 2*np.pi, 200)
            out_power = np.interp(out_theta, phi_fine, power_n)
            x_out = xn + R_field * np.cos(out_theta)
            y_out = yn + R_field * np.sin(out_theta)
            op   = np.array([x_out, y_out]).T.reshape(-1, 1, 2)
            os_  = np.concatenate([op[:-1], op[1:]], axis=1)
            if n == ISite:
                ol = LineCollection(os_, colors='#4a9eff', linewidths=2.5, alpha=0.7, zorder=3)
            elif n == OSite:
                ol = LineCollection(os_, colors='#ff4a6e', linewidths=2.5, alpha=0.7, zorder=3)
            else:
                ol = LineCollection(os_, cmap=cmap_hot, norm=norm_2d, linewidths=0.8, zorder=3)
                ol.set_array(out_power[:-1])
            ol.set_rasterized(True)
            ax.add_collection(ol)

        is_zz  = (h_str == 'A_zigzag')
        is_cyl = (h_str in ('IQH_cyl', 'AQH_cyl'))
        if is_zz or is_cyl:
            all_x = [v[0] for v in coords.values()]
            all_y = [v[1] for v in coords.values()]
            pad = R_field * 2
            ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
            ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
        else:
            ax.set_xlim(-1, span_x); ax.set_ylim(-1, span_y)
        ax = self.ax_vis_flow; ax.cla(); ax.set_facecolor('black')
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'Photon Flow  μ={self.spn_mode_idx.value()}  |  step {step_idx}',
                     color='#c8d0e7', fontsize=8, pad=2)
        for sp in ax.spines.values():
            sp.set_edgecolor('white'); sp.set_linewidth(1.5)

        H_mat_flow = np.zeros((NL+1, NL+1), dtype=complex)
        H_mat_flow[1:, 1:] = p['H_mat']
        fld = np.zeros(NL+1, dtype=complex); fld[1:] = field_mode
        Xq, Yq, U, V, cols_f = compute_flow(
            fld, H_mat_flow, NL, ISite, OSite,
            h_str, nx0, ny0, nx1, ny1)
        ax.quiver(Xq, Yq, U, V, color=cols_f,
                  scale=0.6, scale_units='xy', width=0.01, pivot='middle', zorder=3)

        is_zz  = (h_str == 'A_zigzag')
        is_cyl = (h_str in ('IQH_cyl', 'AQH_cyl'))
        if is_zz:
            ax.set_xlim(-0.5, 2*nx0+0.5); ax.set_ylim(-0.5, 2*ny0+0.5)
        elif is_cyl:
            ax.autoscale_view()
        else:
            ax.set_xlim(-1, span_x); ax.set_ylim(-1, span_y)

        # ── 3D power — ring line traces, black bg, white/colored ───────
        ax3 = self.ax_vis_3d; ax3.cla()
        ax3.set_facecolor('black')
        ax3.xaxis.pane.fill = False; ax3.yaxis.pane.fill = False; ax3.zaxis.pane.fill = False
        ax3.xaxis.pane.set_edgecolor('none')
        ax3.yaxis.pane.set_edgecolor('none')
        ax3.zaxis.pane.set_edgecolor('none')
        ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
        ax3.set_axis_off()
        ax3.set_box_aspect([1, 1, 0.5])
        ax3.view_init(elev=80, azim=270)
        ax3.set_title(f'3D Power  |  step {step_idx}', color='#c8d0e7', fontsize=8, pad=2)

        pmax_global = np.abs(self._hist).max()**2 or 1.
        for i in range(NL):
            n = i + 1
            xn, yn  = coords[n]
            power_n = np.abs(a_T_step[i, :])**2 / pmax_global
            x3 = xn + R_field * np.cos(phi_fine)
            y3 = yn + R_field * np.sin(phi_fine)
            z3 = power_n
            x3 = np.append(x3, x3[0]); y3 = np.append(y3, y3[0]); z3 = np.append(z3, z3[0])
            tc = '#4a9eff' if n==ISite else '#ff4a6e' if n==OSite else 'white'
            line = ax3.plot(x3, y3, z3, color=tc, linewidth=0.8)
            line[0].set_rasterized(True)

        self.canvas_2dpow.draw_idle()
        self.canvas_flow.draw_idle()
        self.canvas_3d.draw_idle()

    def _vis_plot(self):
        if self._hist is None:
            self.lbl_vis_status.setText('Run simulation first.')
            return
        step = self.spn_vis_step.value()
        self.spn_vis_step.setRange(0, self._hist.shape[2]-1)
        step = min(step, self._hist.shape[2]-1)
        self._vis_frame(step)
        self.lbl_vis_status.setText(f'Plotted step {step}.')

    def _vis_save_step(self):
        """Save the three visualization figures for the currently plotted step."""
        if self._hist is None:
            self.lbl_vis_status.setText('Run simulation first.'); return
        folder = self._p.get('session_folder')
        if not folder:
            folder = QFileDialog.getExistingDirectory(
                self, 'Choose save folder', os.path.expanduser('~'))
            if not folder: return
        os.makedirs(folder, exist_ok=True)

        step = self.spn_vis_step.value()
        mu   = self.spn_mode_idx.value()

        for fig, name in [
            (self.fig_2dpow, f'04_2d_power_step{step}'),
            (self.fig_flow,  f'07_photon_flow_step{step}_mu{mu}'),
            (self.fig_3d,    f'08_3d_power_step{step}'),
        ]:
            for fmt, dpi in [('png', 200), ('svg', 150)]:
                buf = io.BytesIO()
                fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight',
                            facecolor=fig.get_facecolor())
                with open(os.path.join(folder, f'{name}.{fmt}'), 'wb') as fh:
                    fh.write(buf.getvalue())

        self.lbl_vis_status.setText(f'Saved step {step} → {os.path.basename(folder)}')

    def _on_exit(self):
        """Interrupt any ongoing plotting and close the window."""
        self._plotting = False
        self._stop.set()   # also stop any running simulation
        self.close()

    def _vis_nav(self, delta):
        """Close this window and open a new SlowTimeWindow seeded from step_idx ± 1."""
        parent = self.parent()
        if parent is None or self._p is None: self.close(); return
        if not hasattr(parent, '_results') or parent._results is None:
            self.close(); return
        current = self._p.get('step_idx', 0)
        n_steps = parent._results['a_T'].shape[2]
        new_idx = int(np.clip(current + delta, 0, n_steps - 1))
        parent.sld_probe.setValue(new_idx)
        parent._open_slow_time()
        self.close()

    def _vis_export_gif(self):
        self._vis_export('gif')

    def _vis_export_mp4(self):
        self._vis_export('mp4')

    def _vis_export(self, fmt):
        if self._hist is None:
            self.lbl_vis_status.setText('Run simulation first.'); return

        NS = self._hist.shape[2]

        # ── Ask user for frame range (no folder dialog) ─────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle('Export range')
        dlg.setStyleSheet(STYLE)
        dlg.setMinimumWidth(300)
        fl = QGridLayout(dlg); fl.setSpacing(6)

        def _lbl(t): lb = QLabel(t); lb.setStyleSheet('color:#c8d0e7;font-size:13px;'); return lb

        spn_s = QSpinBox(); spn_s.setRange(0, NS-1); spn_s.setValue(max(0, NS-1001))
        spn_e = QSpinBox(); spn_e.setRange(0, NS-1); spn_e.setValue(NS-1)
        spn_t = QSpinBox(); spn_t.setRange(1, max(1, NS//2)); spn_t.setValue(10)
        spn_t.setToolTip('Save every N-th snapshot as a frame')

        fl.addWidget(_lbl('Start step:'),    0, 0); fl.addWidget(spn_s, 0, 1)
        fl.addWidget(_lbl('End step:'),      1, 0); fl.addWidget(spn_e, 1, 1)
        fl.addWidget(_lbl('Step interval:'), 2, 0); fl.addWidget(spn_t, 2, 1)

        lbl_n = QLabel('')
        lbl_n.setStyleSheet('color:#4a5270;font-size:12px;')
        fl.addWidget(lbl_n, 3, 0, 1, 2)

        def _update_n(*_):
            s = spn_s.value(); e = spn_e.value(); t = spn_t.value()
            n = max(0, (e - s) // t + 1) if e >= s else 0
            lbl_n.setText(f'→  {n} frames')
        spn_s.valueChanged.connect(_update_n)
        spn_e.valueChanged.connect(_update_n)
        spn_t.valueChanged.connect(_update_n)
        _update_n()

        btn_row = QHBoxLayout()
        btn_ok  = QPushButton('Export'); btn_ok.clicked.connect(dlg.accept)
        btn_can = QPushButton('Cancel'); btn_can.clicked.connect(dlg.reject)
        btn_row.addStretch(); btn_row.addWidget(btn_ok); btn_row.addWidget(btn_can)
        fl.addLayout(btn_row, 4, 0, 1, 2)

        if dlg.exec_() != QDialog.Accepted: return

        s_start = spn_s.value(); s_end = spn_e.value(); s_step = spn_t.value()
        if s_end < s_start: s_start, s_end = s_end, s_start
        steps = list(range(s_start, s_end + 1, s_step))
        if not steps:
            self.lbl_vis_status.setText('No frames in range.'); return

        # Save into session folder (same as the other files)
        folder = self._p.get('session_folder') or os.path.join(os.path.expanduser('~'), 'Documents')
        os.makedirs(folder, exist_ok=True)
        mu    = self.spn_mode_idx.value()
        fname = f'movie_s{s_start}_e{s_end}_step{s_step}_mu{mu}.{fmt}'
        out   = os.path.join(folder, fname)

        import matplotlib.animation as animation
        self.lbl_vis_status.setText(f'Exporting {len(steps)} frames…')
        self.canvas_2dpow.draw_idle(); QApplication.processEvents()

        p       = self._p; ls = p.get('linear_state', {})
        NL      = p['NLattice']; ISite = p['ISite']; OSite = p['OSite']
        PumpFSR = p['one_side_fsr']
        NumFSRs = self._hist.shape[1]
        mode_abs = int(np.clip(self.spn_mode_idx.value() + PumpFSR, 0, NumFSRs - 1))
        h_str = ls.get('h_str', 'IQH_IQH')
        nx0   = ls.get('nx0', 4); ny0 = ls.get('ny0', 4)
        nx1   = ls.get('nx1', 1); ny1 = ls.get('ny1', 1)
        coords  = {n: site_xy(n, h_str, nx0, ny0, nx1, ny1) for n in range(1, NL+1)}
        c2n     = {(int(round(v[0])), int(round(v[1]))): k for k, v in coords.items()}
        H_mat   = np.zeros((NL+1, NL+1), dtype=complex); H_mat[1:, 1:] = p['H_mat']
        phi_e   = np.linspace(0, 2*np.pi, NumFSRs, endpoint=False)
        span_x  = nx1 * nx0;  span_y = ny1 * ny0
        R_e     = RING_R
        cmap_h  = plt.cm.hot

        fig_exp = plt.figure(figsize=(8, 5), facecolor=DARK_BG)
        ax1 = fig_exp.add_subplot(2, 2, 1)
        ax2 = fig_exp.add_subplot(2, 2, 3)
        ax3 = fig_exp.add_subplot(1, 2, 2, projection='3d')
        fig_exp.subplots_adjust(hspace=0.25, wspace=0.15,
                                left=0.02, right=0.98, top=0.94, bottom=0.03)
        pmax_global_e = np.abs(self._hist).max()**2 or 1.

        def _frame(si):
            ax1.cla(); ax2.cla(); ax3.cla()
            a_T = self._hist[:, :, si]
            norm_e = plt.Normalize(0, np.max(np.abs(a_T)**2) / 1.3 or 1.)
            field_m = a_T[:, mode_abs]

            # ── 2D power ─────────────────────────────────────────────
            ax1.set_facecolor('black'); ax1.set_aspect('equal')
            ax1.set_xticks([]); ax1.set_yticks([])
            ax1.set_title(f'2D Power  step={si}', color='#c8d0e7', fontsize=8)
            if nx1 > 1 or ny1 > 1:
                from matplotlib.patches import Rectangle as _R
                for i1 in range(nx1):
                    for j1 in range(ny1):
                        fc_p = '#444444' if (i1+j1)%2==0 else '#888888'
                        ax1.add_patch(_R((i1*nx0-0.5, j1*ny0-0.5), nx0, ny0,
                                         linewidth=0, facecolor=fc_p, alpha=0.3))
            for i in range(NL):
                n = i+1; xn, yn = coords[n]
                pn = np.abs(a_T[i, :])**2
                pts = np.array([xn+R_e*np.cos(phi_e), yn+R_e*np.sin(phi_e)]).T.reshape(-1,1,2)
                seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(seg, cmap=cmap_h, norm=norm_e, linewidth=2.0, zorder=2)
                lc.set_array(pn[:-1]); ax1.add_collection(lc)
                ot = np.linspace(0, 2*np.pi, 200)
                op = np.interp(ot, phi_e, pn)
                osp = np.array([xn+R_e*np.cos(ot), yn+R_e*np.sin(ot)]).T.reshape(-1,1,2)
                oss = np.concatenate([osp[:-1], osp[1:]], axis=1)
                if n == ISite:   ol = LineCollection(oss, colors='#4a9eff', linewidths=2.5, alpha=0.7, zorder=3)
                elif n == OSite: ol = LineCollection(oss, colors='#ff4a6e', linewidths=2.5, alpha=0.7, zorder=3)
                else:
                    ol = LineCollection(oss, cmap=cmap_h, norm=norm_e, linewidths=0.8, zorder=3)
                    ol.set_array(op[:-1])
                ax1.add_collection(ol)
            _is_zz  = (h_str == 'A_zigzag')
            _is_cyl = (h_str in ('IQH_cyl', 'AQH_cyl'))
            if _is_zz or _is_cyl:
                all_x = [v[0] for v in coords.values()]
                all_y = [v[1] for v in coords.values()]
                pad = R_e * 2
                ax1.set_xlim(min(all_x) - pad, max(all_x) + pad)
                ax1.set_ylim(min(all_y) - pad, max(all_y) + pad)
            else:
                ax1.set_xlim(-1, span_x); ax1.set_ylim(-1, span_y)

            # ── Photon flow ──────────────────────────────────────────
            ax2.set_facecolor('black'); ax2.set_aspect('equal')
            ax2.set_xticks([]); ax2.set_yticks([])
            ax2.set_title(f'Photon Flow μ={self.spn_mode_idx.value()}  step={si}', color='#c8d0e7', fontsize=8)
            for sp in ax2.spines.values(): sp.set_edgecolor('white'); sp.set_linewidth(1.5)
            fld_e = np.zeros(NL+1, dtype=complex); fld_e[1:] = field_m
            Xq_, Yq_, U_, V_, cf_ = compute_flow(
                fld_e, H_mat, NL, ISite, OSite, h_str, nx0, ny0, nx1, ny1)
            ax2.quiver(Xq_, Yq_, U_, V_, color=cf_,
                       scale=0.6, scale_units='xy', width=0.01, pivot='middle', zorder=3)
            _is_zz  = (h_str == 'A_zigzag')
            _is_cyl = (h_str in ('IQH_cyl', 'AQH_cyl'))
            if _is_zz:   ax2.set_xlim(-0.5, 2*nx0+0.5); ax2.set_ylim(-0.5, 2*ny0+0.5)
            elif _is_cyl: ax2.autoscale_view()
            else:         ax2.set_xlim(-1, span_x); ax2.set_ylim(-1, span_y)

            # ── 3D ring traces — black bg, white/colored ─────────────
            ax3.set_facecolor('black')
            ax3.xaxis.pane.fill = False; ax3.yaxis.pane.fill = False; ax3.zaxis.pane.fill = False
            ax3.xaxis.pane.set_edgecolor('none')
            ax3.yaxis.pane.set_edgecolor('none')
            ax3.zaxis.pane.set_edgecolor('none')
            ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
            ax3.set_axis_off(); ax3.set_box_aspect([1, 1, 0.5])
            ax3.view_init(elev=80, azim=270)
            ax3.set_title(f'3D Power  step={si}', color='#c8d0e7', fontsize=8)
            for i in range(NL):
                n = i+1; xn, yn = coords[n]
                pn = np.abs(a_T[i, :])**2 / pmax_global_e
                x3 = xn + R_e*np.cos(phi_e); y3 = yn + R_e*np.sin(phi_e); z3 = pn
                x3 = np.append(x3, x3[0]); y3 = np.append(y3, y3[0]); z3 = np.append(z3, z3[0])
                tc = '#4a9eff' if n==ISite else '#ff4a6e' if n==OSite else 'white'
                ax3.plot(x3, y3, z3, color=tc, linewidth=0.8)
            return []

        if fmt == 'gif':
            ani = animation.FuncAnimation(fig_exp, _frame, frames=steps, blit=False)
            ani.save(out, writer='pillow', fps=5)
            plt.close(fig_exp)
        else:
            # Use imageio + imageio-ffmpeg (bundles its own ffmpeg, no system install needed)
            try:
                import imageio
                from PIL import Image as _PILImage
                frames_arr = []
                for si in steps:
                    _frame(si)
                    buf = io.BytesIO()
                    fig_exp.savefig(buf, format='png', dpi=100)
                    buf.seek(0)
                    frames_arr.append(np.array(_PILImage.open(buf).convert('RGB')))
                plt.close(fig_exp)
                imageio.mimwrite(out, frames_arr, fps=10, quality=8)
            except ImportError:
                plt.close(fig_exp)
                self.lbl_vis_status.setText(
                    '❌ MP4 needs imageio: pip install imageio imageio-ffmpeg')
                return
        self.lbl_vis_status.setText(f'Saved → {os.path.basename(out)}')

    # ------------------------------------------------------------------
    def _on_save(self):
        folder = self._p.get('session_folder')
        if not folder:
            folder = QFileDialog.getExistingDirectory(
                self, 'Choose save folder', os.path.expanduser('~'))
            if not folder: return
        os.makedirs(folder, exist_ok=True)

        mu    = self.spn_mode_idx.value()
        fsr   = self.spn_fsr_shift.value()
        nest  = self.spn_nest_range.value()
        ycut  = self.spn_ycut_freq.value()

        def _flt(v): return f'{v:.3g}'.replace('.', 'p').replace('-', 'm')

        mu_str   = f'_mu{mu}'
        fsr_str  = f'_FSR{_flt(fsr)}'
        nest_str = f'_nest{_flt(nest)}'
        ycut_str = f'_ycut{_flt(ycut)}'

        figs = [
            (self.fig_ind,    '01_sweep_indicator'),
            (self.fig_heat4,  '02_heatmaps'),
            (self.fig_comb,   f'03_comb_spectrum{mu_str}{fsr_str}'),
            (self.fig_2dpow,  '04_2d_power'),
            (self.fig_esa,    '05_esa_oscilloscope'),
            (self.fig_spec2d, f'06_2d_spectrum{mu_str}{fsr_str}{nest_str}{ycut_str}'),
            (self.fig_flow,   f'07_photon_flow{mu_str}'),
            (self.fig_3d,     '08_3d_power'),
        ]
        for fig, name in figs:
            # PNG
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            with open(os.path.join(folder, f'{name}.png'), 'wb') as fh:
                fh.write(buf.getvalue())

            # SVG — rasterize artists only for the heatmap figure
            snapshot = []
            if fig is self.fig_heat4:
                for ax in fig.get_axes():
                    for art in ax.get_children():
                        if hasattr(art, 'set_rasterized'):
                            snapshot.append((art, art.get_rasterized()))
                            art.set_rasterized(True)
            buf = io.BytesIO()
            fig.savefig(buf, format='svg', dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            for art, was in snapshot:
                art.set_rasterized(was)
            with open(os.path.join(folder, f'{name}.svg'), 'wb') as fh:
                fh.write(buf.getvalue())

        # ── Data ─────────────────────────────────────────────────────────
        if self._hist is not None:
            p = self._p
            np.savez(os.path.join(folder, 'slowtime_data.npz'),
                     hist     = self._hist,
                     FSRVec   = p['FSRVec'],
                     every    = np.array([self.spn_every.value()]),
                     step_idx = np.array([p['step_idx']]),
                     detuning = np.array([p['detuning']]),
                     F        = np.array([p['F']]),
                     kin      = np.array([p['kin']]),
                     kex      = np.array([p['kex']]),
                     D2       = np.array([p['D2']]),
                     TStep    = np.array([p['TStep']]))

        self.status.showMessage(f'✅ Saved → {folder}')


# Attach continuation methods from mixin onto SlowTimeWindow
for _name, _method in _SlowTimeWindowMethods.__dict__.items():
    if callable(_method) and not _name.startswith('__'):
        setattr(SlowTimeWindow, _name, _method)
del _SlowTimeWindowMethods


# =====================================================================
class NonlinearWindow(QMainWindow):
    def __init__(self, linear_state: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Nonlinear Comb Simulator  --  fed from Linear Explorer')
        self.setMinimumSize(1600, 840)
        self.setStyleSheet(STYLE)

        ls = linear_state;  self._ls = ls
        self._NLattice = ls['NL']
        self._is_cyl   = ls.get('is_cyl', False)
        self._h_str    = ls['h_str']

        self._param_rows = []
        self._h_sweep    = set()
        self._q          = queue.Queue()
        self._stop_flag  = threading.Event()
        self._poll_timer = QTimer(self);  self._poll_timer.setInterval(80)
        self._poll_timer.timeout.connect(self._poll_queue)

        self._xv = self._pump_n = self._side_n = self._spec_dB = None
        self._pump_n_all = self._side_n_all = None
        self._FSRVec = self._PumpFSR = None;  self._n_steps = 0
        self._hmap_img = self._line_pump = self._line_side = None
        self._vline_pump = self._vline_side = None
        self._sched_vline = self._sched_vline_psi = self._sched_vline_h = None
        self._spec_drag = False
        self._results   = None

        self._build_ui()

        # Draw lattice immediately so rings are visible on open
        self._draw_lattice()
        # These are NOT removable since they define the baseline on-site potential.
        # User can switch them to Sweep or Design to ramp them.
        for site, delta in sorted(ls.get('heaters', {}).items()):
            self._add_param_row(
                f'Heater {site}  (inherited)', f'heater_{site}',
                default_fixed=delta,
                default_from=delta, default_to=delta + 3.0,
                removable=False)

    # ------------------------------------------------------------------
    def showEvent(self, event):
        """Redraw schedule preview when window becomes visible (ensures canvas is ready)."""
        super().showEvent(event)
        self._draw_sched_preview()

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget();  self.setCentralWidget(central)
        root = QVBoxLayout(central);  root.setSpacing(4)
        root.setContentsMargins(6,6,6,6)

        t = QLabel('NONLINEAR COMB SIMULATOR  --  fed from Linear Lattice Explorer')
        t.setFont(QFont('Courier New', 13, QFont.Bold))
        t.setStyleSheet(f'color:{ACCENT};')
        root.addWidget(t)
        sep = QFrame();  sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet('color:#1e2230;');  root.addWidget(sep)

        self._splitter = QSplitter(Qt.Horizontal)
        splitter = self._splitter

        # ── Left: schedule preview (full height) ──────────────────────
        lw = QWidget();  ll = QVBoxLayout(lw);  ll.setContentsMargins(0,0,0,0)
        self.fig_sched    = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_sched = FigureCanvas(self.fig_sched)
        self.canvas_sched.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ll.addWidget(self.canvas_sched)

        # ── Middle: pump power (top) + comb power (bottom) ────────────
        mw = QWidget();  ml = QVBoxLayout(mw);  ml.setContentsMargins(0,0,0,0)
        self.fig_plots    = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_plots = FigureCanvas(self.fig_plots)
        self.canvas_plots.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs = self.fig_plots.add_gridspec(2, 2, hspace=0.52, wspace=0.38,
             left=0.09, right=0.97, top=0.93, bottom=0.10)
        self.ax_pump     = self.fig_plots.add_subplot(gs[0, 0])
        self.ax_side     = self.fig_plots.add_subplot(gs[1, 0])
        self.ax_pump_all = self.fig_plots.add_subplot(gs[0, 1])
        self.ax_side_all = self.fig_plots.add_subplot(gs[1, 1])
        for ax,col,ttl in [
            (self.ax_pump,    '#4a9eff','Pump power  (output ring)'),
            (self.ax_side,    '#ff4a6e','Comb power excl. pump  (output ring)'),
            (self.ax_pump_all,'#4a9eff','Pump power  (all rings)'),
            (self.ax_side_all,'#ff4a6e','Comb power excl. pump  (all rings)')]:
            ax.set_facecolor(PANEL_BG);  ax.set_title(ttl,color=col,fontsize=8,pad=3)
            ax.tick_params(colors=TEXT_COL,labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560');  sp.set_linewidth(1.2)
            ax.grid(True,color=GRID_COL,linewidth=0.4)
        self.canvas_plots.mpl_connect('button_press_event',   self._on_spec_press)
        self.canvas_plots.mpl_connect('motion_notify_event',  self._on_spec_move)
        self.canvas_plots.mpl_connect('button_release_event',
                                       lambda e: setattr(self,'_spec_drag',False))
        ml.addWidget(self.canvas_plots)

        # keep canvas_lat alive for _draw_lattice (off-screen, not in layout)
        self.fig_lat    = Figure(facecolor=DARK_BG)
        self.canvas_lat = FigureCanvas(self.fig_lat)
        self.ax_lat     = self.fig_lat.add_subplot(111)
        self.ax_lat.set_facecolor(PANEL_BG)
        self.ax_lat.set_xticks([]);  self.ax_lat.set_yticks([])
        for sp in self.ax_lat.spines.values(): sp.set_edgecolor('#3a4560')
        self.fig_lat.subplots_adjust(left=0.01,right=0.99,top=0.97,bottom=0.01)

        # ── Right: a_T and a_W heatmaps (2×2) ────────────────────────
        rw = QWidget();  rl = QVBoxLayout(rw);  rl.setContentsMargins(0,0,0,0)
        self.fig_heat    = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_heat = FigureCanvas(self.fig_heat)
        self.canvas_heat.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gs_h = self.fig_heat.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
               left=0.10, right=0.97, top=0.93, bottom=0.08)
        self.ax_aT_out = self.fig_heat.add_subplot(gs_h[0, 0])
        self.ax_aT_all = self.fig_heat.add_subplot(gs_h[0, 1])
        self.ax_aW_out = self.fig_heat.add_subplot(gs_h[1, 0])
        self.ax_aW_all = self.fig_heat.add_subplot(gs_h[1, 1])
        for ax, col, ttl in [
            (self.ax_aT_out, '#a8ff78', '|a_T|²  (output ring)'),
            (self.ax_aT_all, '#a8ff78', '|a_T|²  (all rings)'),
            (self.ax_aW_out, '#ff8c42', '10·log|a_W|²  (output ring)'),
            (self.ax_aW_all, '#ff8c42', '10·log|a_W|²  (all rings)'),
        ]:
            ax.set_facecolor(PANEL_BG)
            ax.set_title(ttl, color=col, fontsize=8, pad=3)
            ax.tick_params(colors=TEXT_COL, labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.set_xlabel('Step', color=TEXT_DIM, fontsize=7)
        self.ax_aT_out.set_ylabel('θ index', color=TEXT_DIM, fontsize=7)
        self.ax_aT_all.set_ylabel('θ index', color=TEXT_DIM, fontsize=7)
        self.ax_aW_out.set_ylabel('FSR  μ',  color=TEXT_DIM, fontsize=7)
        self.ax_aW_all.set_ylabel('FSR  μ',  color=TEXT_DIM, fontsize=7)
        self._heat_imgs = {}   # stores imshow objects for fast update
        self._heat_vlines = {}  # vertical line overlays on heatmaps

        # cross-section plots (below heatmaps, update with probe)
        self.fig_xsec    = Figure(facecolor=DARK_BG, tight_layout=False)
        self.canvas_xsec = FigureCanvas(self.fig_xsec)
        self.canvas_xsec.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.canvas_xsec.setMinimumHeight(200)
        gs_x = self.fig_xsec.add_gridspec(1, 2, wspace=0.4,
               left=0.10, right=0.97, top=0.88, bottom=0.18)
        self.ax_xsec_aW = self.fig_xsec.add_subplot(gs_x[0])
        self.ax_xsec_aT = self.fig_xsec.add_subplot(gs_x[1])
        for ax, col, ttl in [
            (self.ax_xsec_aW, '#ff8c42', '10·log|a_W|²  at step  (osite)'),
            (self.ax_xsec_aT, '#a8ff78', '|a_T|²  at step  (osite)'),
        ]:
            ax.set_facecolor(PANEL_BG)
            ax.set_title(ttl, color=col, fontsize=8, pad=3)
            ax.tick_params(colors=TEXT_COL, labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.grid(True, color=GRID_COL, lw=0.4)
        self.ax_xsec_aW.set_xlabel('FSR  μ',  color=TEXT_DIM, fontsize=7)
        self.ax_xsec_aT.set_xlabel('θ index', color=TEXT_DIM, fontsize=7)

        rl.addWidget(self.canvas_heat)
        rl.addWidget(self.canvas_xsec)
        rw.setLayout(rl)

        # ── Far right: slow time analysis panel (hidden until triggered) ──
        splitter.addWidget(lw);  splitter.addWidget(mw)
        splitter.addWidget(rw)
        self._splitter = splitter
        splitter.setSizes([480, 560, 580])
        root.addWidget(splitter,stretch=1)
        root.addWidget(self._build_controls())
        self.status = QStatusBar();  self.setStatusBar(self.status)
        self.status.showMessage(
            f'Ready  |  H_mat inherited  |  ' +
            (f'JAX {jax.__version__} on {jax.default_backend()}' if JAX_AVAILABLE else 'JAX not installed — NumPy fallback active'))

    # ------------------------------------------------------------------
    def _build_controls(self):
        w = QWidget();  row = QHBoxLayout(w)
        row.setSpacing(6);  row.setContentsMargins(0,0,0,0)

        # ── 1. Inherited ──────────────────────────────────────────────
        gi  = QGroupBox('Inherited from Linear Explorer')
        il  = QGridLayout(gi);  il.setSpacing(5)
        ls  = self._ls;  h = ls['h_str']

        def _ro(text, col='#c8d0e7'):
            l = QLabel(text);  l.setStyleSheet(f'color:{col};font-size:12px;')
            return l

        nx0,ny0 = ls['nx0'],ls['ny0'];  nx1,ny1 = ls['nx1'],ls['ny1']
        if h == 'A_zigzag':  size_str = f'{nx0}x{ny0} Zigzag'
        elif h in CYL_TYPES: size_str = f'{nx0}x{ny0} Cylinder'
        else:                 size_str = f'{nx0}x{ny0} x {nx1}x{ny1}'

        il.addWidget(_ro(f'H type:  {h}', ACCENT),       0, 0)
        il.addWidget(_ro(f'Size:  {size_str}'),           0, 1)
        il.addWidget(_ro(f'Sites:  {ls["NL"]}'),          0, 2)
        il.addWidget(_ro(f'kin = {ls["kin"]:.4f}'),       1, 0)
        il.addWidget(_ro(f'kex = {ls["kex"]:.4f}'),       1, 1)
        il.addWidget(_ro(f'IN = {ls["isite"]}   OUT = {ls["osite"]}'), 1, 2)

        parts = []
        if h in ('IQH_IQH','IQH_AQH','IQH_cyl'):
            parts.append(f'phi_IQH_0 = {_phi_str(ls["phi_iqh0"])}')
        if h in ('IQH_IQH','AQH_IQH'):
            parts.append(f'phi_IQH_1 = {_phi_str(ls["phi_iqh1"])}')
        if h in ('AQH_AQH','AQH_IQH','A_zigzag','AQH_cyl'):
            parts.append(f'phi_AQH_0 = {_phi_str(ls["phi_aqh0"])}')
        if h in ('AQH_AQH','IQH_AQH'):
            parts.append(f'phi_AQH_1 = {_phi_str(ls["phi_aqh1"])}')
        if ls.get('is_cyl') and ls.get('psi', 0):
            parts.append(f'psi = {_phi_str(ls["psi"])}')
        il.addWidget(_ro('  |  '.join(parts) if parts else 'no phases'), 2, 0, 1, 3)

        def _collapsible(summary, detail, col):
            """Button that toggles a detail label — click to expand/collapse."""
            cw  = QWidget();  cl = QVBoxLayout(cw)
            cl.setContentsMargins(0,0,0,0);  cl.setSpacing(1)
            btn = QPushButton(f'▶  {summary}')
            btn.setCheckable(True)
            btn.setStyleSheet(
                f'QPushButton{{color:{col};background:transparent;border:none;'
                f'font-size:12px;text-align:left;padding:0px;}}'
                f'QPushButton:hover{{color:#ffffff;}}')
            lbl = QLabel(detail)
            lbl.setStyleSheet(f'color:{col};font-size:11px;padding-left:14px;')
            lbl.setWordWrap(True);  lbl.setVisible(False)
            def _toggle(checked, b=btn, l=lbl, s=summary):
                b.setText(('▼' if checked else '▶') + f'  {s}')
                l.setVisible(checked)
            btn.toggled.connect(_toggle)
            cl.addWidget(btn);  cl.addWidget(lbl)
            return cw

        heaters = ls.get('heaters', {})
        defects = ls.get('defects', set())
        if heaters:
            ht_detail = '\n'.join([f'site {k}: {v:+.2f} J' for k,v in sorted(heaters.items())])
            il.addWidget(_collapsible(f'Heaters ({len(heaters)})', ht_detail, '#ffcc00'), 3, 0, 1, 2)
        else:
            il.addWidget(_ro('Heaters:  none'), 3, 0)
        if defects:
            dt_detail = '\n'.join([f'site {k}' for k in sorted(defects)])
            il.addWidget(_collapsible(f'Defects zeroed ({len(defects)})', dt_detail, '#ff6b6b'), 3, 2)
        else:
            il.addWidget(_ro('Defects:  none'), 3, 2)

        gi.setMaximumWidth(500)
        row.addWidget(gi)

        # ── 2. Comb & Stepper ─────────────────────────────────────────
        ge = QGroupBox('Comb & Stepper');  ge.setFixedWidth(280)
        el = QGridLayout(ge);  el.setSpacing(4)
        self.spn_fsr   = QSpinBox();       self.spn_fsr.setRange(1,512);       self.spn_fsr.setValue(128)
        self.spn_D2    = QDoubleSpinBox(); self.spn_D2.setDecimals(4);         self.spn_D2.setRange(0,1);      self.spn_D2.setValue(0.001);  self.spn_D2.setSingleStep(0.001)
        self.spn_niter = QSpinBox();       self.spn_niter.setRange(100,50000); self.spn_niter.setValue(3000);  self.spn_niter.setSingleStep(500)
        self.spn_dt    = QDoubleSpinBox(); self.spn_dt.setDecimals(3);         self.spn_dt.setRange(0.001,1.0); self.spn_dt.setValue(0.1);   self.spn_dt.setSingleStep(0.01)
        for c,lbl,w_ in [(0,'+-FSR',self.spn_fsr),(1,'D2',self.spn_D2)]:
            el.addWidget(QLabel(lbl),0,c); el.addWidget(w_,1,c)
        for c,lbl,w_ in [(0,'NIter',self.spn_niter),(1,'dt/J',self.spn_dt)]:
            el.addWidget(QLabel(lbl),2,c); el.addWidget(w_,3,c)
        row.addWidget(ge)

        # ── 3. Sweep Schedule ─────────────────────────────────────────
        gf = QGroupBox('Sweep Schedule')
        gf.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        fl = QVBoxLayout(gf);  fl.setSpacing(4)

        n_row = QHBoxLayout()
        n_row.addWidget(QLabel('N steps:'))
        self.spn_nsteps = QSpinBox();  self.spn_nsteps.setRange(2,10000); self.spn_nsteps.setValue(600)
        n_row.addWidget(self.spn_nsteps);  n_row.addStretch()
        fl.addLayout(n_row)

        sep2 = QFrame();  sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet('color:#1e2230;margin:2px 0;');  fl.addWidget(sep2)

        hdr = QHBoxLayout()
        for txt, w_ in [('Parameter', 155), ('Mode', 88), ('Value(s)', 210)]:
            l = QLabel(txt);  l.setFixedWidth(w_)
            l.setStyleSheet('color:#4a5270;font-size:11px;font-weight:bold;')
            hdr.addWidget(l)
        hdr.addStretch();  fl.addLayout(hdr)

        self._rows_container = QWidget()
        self._rows_vlay      = QVBoxLayout(self._rows_container)
        self._rows_vlay.setSpacing(2);  self._rows_vlay.setContentsMargins(0,0,0,0)
        self._rows_vlay.addStretch()

        scroll = QScrollArea();  scroll.setWidget(self._rows_container)
        scroll.setWidgetResizable(True);  scroll.setFixedHeight(110)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet('QScrollArea{background:transparent;border:none;}')
        fl.addWidget(scroll)

        # Always-present external rows
        self._add_param_row('Pump F',     'F',        0.15,  0.3,  10.0, removable=False)
        # Inherit detuning sweep range from Linear tab; enforce start > end so the
        # default sweep direction is from red-detuned down toward resonance.
        _ls        = self._ls
        _sw_s      = float(_ls.get('sw_start', 1.455))
        _sw_e      = float(_ls.get('sw_end',   1.4))
        _det_from  = max(_sw_s, _sw_e)   # start (larger)
        _det_to    = min(_sw_s, _sw_e)   # end   (smaller)
        _det_fixed = _det_from           # Fixed-mode default = start
        self._add_param_row('Detuning δ', 'detuning', _det_fixed, _det_from, _det_to,
                            removable=False, default_mode='Sweep')
        if self._is_cyl:
            self._add_param_row('Ext. flux Psi  (×π)', 'psi', 0.0, 0.0, 1.0, removable=False)
        # Inherited heater rows appended after _build_ui() returns (in __init__)

        self.spn_nsteps.valueChanged.connect(self._draw_sched_preview)
        row.addWidget(gf, stretch=1)
        gf.setMaximumWidth(480)

        # ── 4. Run ────────────────────────────────────────────────────
        gr  = QGroupBox('Run');  gr.setMinimumWidth(360);  gr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        rl2 = QGridLayout(gr);  rl2.setSpacing(4)
        self.spn_dynp = QSpinBox();  self.spn_dynp.setRange(1,200); self.spn_dynp.setValue(10)
        self.btn_run  = QPushButton('Run');      self.btn_run.setFixedHeight(30)
        self.btn_stop = QPushButton('Stop');     self.btn_stop.setFixedHeight(30)
        self.btn_save = QPushButton('💾  Save'); self.btn_save.setFixedHeight(30)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet(
            'QPushButton:enabled{color:#00e5ff;border-color:#00e5ff;}'
            'QPushButton:disabled{color:#2a3050;}')
        self.prog_bar  = QProgressBar();  self.prog_bar.setRange(0,100); self.prog_bar.setValue(0)
        self.lbl_eta   = QLabel('');  self.lbl_eta.setStyleSheet('color:#4a5270;font-size:12px;')
        self.lbl_log   = QLabel('Ready');  self.lbl_log.setStyleSheet('color:#4a5270;font-size:12px;')
        self.sld_probe = QSlider(Qt.Horizontal);  self.sld_probe.setRange(0,299); self.sld_probe.setValue(0)
        self.lbl_probe = QLabel('Probe: step 0');  self.lbl_probe.setStyleSheet('color:#c8d0e7;font-size:12px;')
        self.spn_probe = QSpinBox()
        self.spn_probe.setRange(0, 299);  self.spn_probe.setValue(0)
        self.spn_probe.setFixedWidth(72)
        self.spn_probe.setStyleSheet('font-size:12px;')
        self.sld_probe.valueChanged.connect(self._on_probe_change)
        self.spn_probe.valueChanged.connect(self._on_probe_spinbox_change)
        lbl_dynp = QLabel('Plot every:');  lbl_dynp.setStyleSheet('color:#4a5270;font-size:12px;')
        for c in range(4): rl2.setColumnStretch(c, 1)
        # row 0: Plot every | spinbox | Save (right side)
        rl2.addWidget(lbl_dynp,      0,0); rl2.addWidget(self.spn_dynp, 0,1)
        rl2.addWidget(self.btn_save, 0,2,1,2)
        # row 1: Run and Stop take full width equally
        rl2.addWidget(self.btn_run,  1,0,1,2); rl2.addWidget(self.btn_stop,1,2,1,2)
        rl2.addWidget(self.prog_bar, 2,0,1,3);  rl2.addWidget(self.lbl_eta,2,3)
        rl2.addWidget(self.lbl_probe,3,0,1,2);  rl2.addWidget(self.sld_probe,3,2,1,1); rl2.addWidget(self.spn_probe,3,3)
        rl2.addWidget(self.lbl_log,  4,0,1,4)

        # ── Slow-time analysis (visible after run) ─────────────────
        sep3 = QFrame();  sep3.setFrameShape(QFrame.HLine)
        sep3.setStyleSheet('color:#1e2230;');
        rl2.addWidget(sep3, 5, 0, 1, 4)

        # parameter readout at selected step
        self.lbl_step_params = QLabel('—')
        self.lbl_step_params.setStyleSheet('color:#a8ff78;font-size:11px;')
        self.lbl_step_params.setWordWrap(True)
        rl2.addWidget(self.lbl_step_params, 6, 0, 1, 4)

        # slow time button
        self.btn_slow = QPushButton('▶  Run Slow Time Analysis')
        self.btn_slow.setFixedHeight(30)
        self.btn_slow.setEnabled(False)
        self.btn_slow.setStyleSheet(
            'QPushButton:enabled{color:#00e5ff;border-color:#00e5ff;}'
            'QPushButton:disabled{color:#2a3050;}')
        self.btn_slow.clicked.connect(self._open_slow_time)
        rl2.addWidget(self.btn_slow, 8, 0, 1, 4)

        row.addWidget(gr, stretch=1)
        return w

    # ------------------------------------------------------------------
    # Parameter row management
    # ------------------------------------------------------------------
    def _add_param_row(self, label, pkey,
                       default_fixed=0.0, default_from=1.8, default_to=1.2,
                       removable=False, default_mode='Fixed'):
        pr = ParameterRow(label, pkey,
                          default_fixed=default_fixed,
                          default_from=default_from,
                          default_to=default_to,
                          removable=removable,
                          default_mode=default_mode,
                          get_nsteps=lambda: self.spn_nsteps.value())
        if removable:
            pr.btn_rm.clicked.connect(lambda: self._remove_param_row(pr, pkey))
        count = self._rows_vlay.count()
        self._rows_vlay.insertWidget(count - 1, pr)
        self._param_rows.append(pr)
        pr.mode_combo.currentTextChanged.connect(lambda _: self._draw_sched_preview())
        pr.spn_fixed.valueChanged.connect(lambda _: self._draw_sched_preview())
        pr.spn_from.valueChanged.connect( lambda _: self._draw_sched_preview())
        pr.spn_to.valueChanged.connect(   lambda _: self._draw_sched_preview())
        self._draw_sched_preview()
        return pr

    def _remove_param_row(self, pr, pkey):
        if pkey.startswith('heater_'):
            self._h_sweep.discard(int(pkey.split('_')[1]))
            self._draw_lattice()
        pr.setParent(None)
        if pr in self._param_rows:
            self._param_rows.remove(pr)
        self._draw_sched_preview()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _draw_lattice(self):
        ls = self._ls
        NL = ls['NL'];  Nx = ls['Nxt'];  Ny = ls['Nyt']
        ax = self.ax_lat;  ax.cla();  ax.set_facecolor(PANEL_BG)
        ax.set_aspect('equal');  ax.set_xticks([]);  ax.set_yticks([])
        ax.set_title('Lattice power  --  click to add sweep channel',
                     color=TEXT_COL, fontsize=10, pad=3)
        for sp in ax.spines.values(): sp.set_edgecolor(SPINE_COL)
        if not ls.get('is_cyl'):
            ax.set_xlim(-0.8, Nx-0.2);  ax.set_ylim(-0.8, Ny-0.2)

        coords = {n: ((n-1)%Nx, (n-1)//Nx) for n in range(1, NL+1)}
        for n in range(1, NL+1):
            xn,yn = coords[n];  X=(n-1)%Nx;  Y=(n-1)//Nx
            if X < Nx-1:
                nb=n+1;  x2,y2=coords[nb]
                ax.plot([xn,x2],[yn,y2],color='#1e2a40',lw=0.9,zorder=1)
            if Y < Ny-1:
                nb=n+Nx;  x2,y2=coords[nb]
                ax.plot([xn,x2],[yn,y2],color='#1e2a40',lw=0.9,zorder=1)

        defects = ls.get('defects', set())
        for n in range(1, NL+1):
            xn,yn = coords[n]
            if n in defects:                    ec='#333355'
            elif n==ls['isite']:                ec='#4a9eff'
            elif n==ls['osite']:                ec='#ff4a6e'
            elif n in self._h_sweep:            ec='#ff8c42'
            elif ls.get('heaters',{}).get(n,0): ec='#ffcc00'
            else:                               ec='#2d4a6a'

            if self._results is not None and n not in defects:
                step = int(self.sld_probe.value())
                aW   = np.fft.fft(self._results['a_T'][:,:,step],axis=1,norm='ortho')
                pw   = np.sum(np.abs(aW)**2,axis=1)
                fc   = cm.hot(float(np.clip(pw[n-1]/(pw.max() or 1.),0,1)))
            elif n in defects:  fc = '#0a0a0f'
            else:               fc = '#0e1c2f'

            ax.add_patch(mpatches.Circle((xn,yn),RING_R,edgecolor=ec,facecolor=fc,linewidth=1.3,zorder=2))
            ax.add_patch(mpatches.Circle((xn,yn),RING_R*0.62,edgecolor='none',facecolor=DARK_BG,zorder=3))
            label,lc = None,ec
            if n in defects:                        label,lc='×','#555577'
            elif n==ls['isite']:                    label,lc='IN','#4a9eff'
            elif n==ls['osite']:                    label,lc='OUT','#ff4a6e'
            elif n in self._h_sweep:                label,lc='~','#ff8c42'
            elif ls.get('heaters',{}).get(n,0):     label,lc=f"{ls['heaters'][n]:+.1f}",'#ffcc00'
            if label:
                ax.text(xn,yn,label,ha='center',va='center',
                        fontsize=6,color=lc,fontweight='bold',zorder=4)
        self.canvas_lat.draw_idle()

    def _draw_sched_preview(self, *_):
        n = max(2, self.spn_nsteps.value());  x = np.arange(n)

        # Categorise rows
        f_rows   = [pr for pr in self._param_rows if pr.pkey == 'F']
        det_rows = [pr for pr in self._param_rows if pr.pkey == 'detuning']
        psi_rows = [pr for pr in self._param_rows if pr.pkey == 'psi']
        h_rows   = [pr for pr in self._param_rows if pr.pkey.startswith('heater_')]
        has_psi  = len(psi_rows) > 0
        has_h    = len(h_rows)   > 0

        # Build gridspec dynamically
        n_plots = 1 + (1 if has_psi else 0) + (1 if has_h else 0)
        self.fig_sched.clear()
        gs = self.fig_sched.add_gridspec(n_plots, 1, hspace=0.55,
             left=0.13, right=0.87, top=0.95, bottom=0.06)

        def _style(ax, title, col):
            ax.set_facecolor(PANEL_BG)
            ax.set_title(title, color=col, fontsize=9, pad=3)
            ax.tick_params(colors=TEXT_COL, labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.grid(True, color=GRID_COL, lw=0.4)
            ax.set_xlabel('Step', color=TEXT_DIM, fontsize=8)

        # ── Subplot 0: Pump F (left) + Detuning δ (right) twin axis ──
        ax_f = self.fig_sched.add_subplot(gs[0])
        ax_d = ax_f.twinx()
        _style(ax_f, 'Pump F  &  Detuning δ', TEXT_COL)

        if f_rows:
            yf = f_rows[0].get_array(n)
            ax_f.plot(x, yf, color='#ffcc00', lw=1.6,
                      ls='-' if f_rows[0].is_active() else '--')
        ax_f.set_ylabel('F', color='#ffcc00', fontsize=9)
        ax_f.tick_params(axis='y', labelcolor='#ffcc00', labelsize=8)
        ax_f.yaxis.label.set_color('#ffcc00')

        if det_rows:
            yd = det_rows[0].get_array(n)
            ax_d.plot(x, yd, color='#4a9eff', lw=1.6,
                      ls='-' if det_rows[0].is_active() else '--')
        ax_d.set_ylabel('Δ/J', color='#4a9eff', fontsize=9)
        ax_d.tick_params(axis='y', labelcolor='#4a9eff', labelsize=8)
        ax_d.yaxis.label.set_color('#4a9eff')
        # style twinx spines
        for sp in ax_d.spines.values(): sp.set_edgecolor('#3a4560')

        plot_idx = 1

        # ── Subplot 1 (optional): External flux Psi ───────────────────
        if has_psi:
            ax_psi = self.fig_sched.add_subplot(gs[plot_idx])
            _style(ax_psi, 'Ext. flux  Psi', '#ff8c42')
            yp = psi_rows[0].get_array(n)
            ax_psi.plot(x, yp, color='#ff8c42', lw=1.6,
                        ls='-' if psi_rows[0].is_active() else '--')
            ax_psi.set_ylabel('Psi (×π)', color='#ff8c42', fontsize=9)
            ax_psi.tick_params(axis='y', labelcolor='#ff8c42', labelsize=8)
            plot_idx += 1

        # ── Subplot 2 (optional): All heater channels ─────────────────
        if has_h:
            ax_h = self.fig_sched.add_subplot(gs[plot_idx])
            _style(ax_h, 'Heaters', '#a8ff78')
            COLS = ['#a8ff78','#ffcc00','#ff4a6e','#cc88ff','#ff8c42','#4a9eff']
            for ci, pr in enumerate(h_rows):
                yh = pr.get_array(n)
                site = pr.pkey.split('_')[1]
                ax_h.plot(x, yh, color=COLS[ci%len(COLS)], lw=1.4,
                          ls='-' if pr.is_active() else '--',
                          label=f'H{site}')
            ax_h.set_ylabel('delta (J)', color='#a8ff78', fontsize=9)
            ax_h.tick_params(axis='y', labelcolor='#a8ff78', labelsize=8)
            ax_h.legend(fontsize=7, frameon=False, labelcolor=TEXT_COL,
                        ncol=min(len(h_rows), 4), loc='best')

        # ── Real-time step indicator (moves during simulation) ───────
        n_full = max(2, self.spn_nsteps.value())
        self._sched_vline     = ax_f.axvline(x=0, color='white', lw=1.2, ls=':', alpha=0.85)
        self._sched_vline_psi = ax_psi.axvline(x=0, color='white', lw=1.2, ls=':', alpha=0.85) if has_psi else None
        self._sched_vline_h   = ax_h.axvline(  x=0, color='white', lw=1.2, ls=':', alpha=0.85) if has_h   else None
        ax_f.set_xlim(0, n_full - 1)

        self.canvas_sched.draw_idle()

    def _init_spec_plots(self, x_label):
        for ax,col,ttl in [
            (self.ax_pump,    '#4a9eff','Pump power  (output ring)'),
            (self.ax_side,    '#ff4a6e','Comb power excl. pump  (output ring)'),
            (self.ax_pump_all,'#4a9eff','Pump power  (all rings)'),
            (self.ax_side_all,'#ff4a6e','Comb power excl. pump  (all rings)')]:
            ax.cla();  ax.set_facecolor(PANEL_BG)
            ax.set_title(ttl,color=col,fontsize=8,pad=3)
            ax.set_xlabel(x_label,color=TEXT_DIM,fontsize=8)
            ax.tick_params(colors=TEXT_COL,labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
            ax.grid(True,color=GRID_COL,lw=0.4)
        self._line_pump,    =self.ax_pump.plot([],[],    color='#4a9eff',lw=1.5)
        self._line_side,    =self.ax_side.plot([],[],    color='#ff4a6e',lw=1.5)
        self._line_pump_all,=self.ax_pump_all.plot([],[], color='#4a9eff',lw=1.5)
        self._line_side_all,=self.ax_side_all.plot([],[], color='#ff4a6e',lw=1.5)
        self._vline_pump    =self.ax_pump.axvline(0,    color='white',lw=1,ls='--',alpha=0.5)
        self._vline_side    =self.ax_side.axvline(0,    color='white',lw=1,ls='--',alpha=0.5)
        self._vline_pump_all=self.ax_pump_all.axvline(0,color='white',lw=1,ls='--',alpha=0.5)
        self._vline_side_all=self.ax_side_all.axvline(0,color='white',lw=1,ls='--',alpha=0.5)
        self.canvas_plots.draw_idle()

    def _update_spec_plots(self, n_done, probe_i):
        xsf = self._xv[:n_done]
        for arr, line, ax, vl in [
            (self._pump_n,     self._line_pump,     self.ax_pump,     self._vline_pump),
            (self._side_n,     self._line_side,     self.ax_side,     self._vline_side),
            (self._pump_n_all, self._line_pump_all, self.ax_pump_all, self._vline_pump_all),
            (self._side_n_all, self._line_side_all, self.ax_side_all, self._vline_side_all),
        ]:
            data = arr[:n_done]
            mx   = data.max() or 1.
            line.set_data(xsf, data / mx)
            ax.set_xlim(xsf[0], xsf[-1]);  ax.set_ylim(0, 1.05)
            if n_done > probe_i: vl.set_xdata([xsf[probe_i], xsf[probe_i]])
        self.canvas_plots.draw_idle()

    # ------------------------------------------------------------------
    # Interactions
    # ------------------------------------------------------------------
    def _on_lat_click(self, event):
        if event.inaxes!=self.ax_lat or event.xdata is None: return
        ls=self._ls;  NL=ls['NL'];  Nx=ls['Nxt']
        best_n,best_d=None,float('inf')
        for n in range(1, NL+1):
            X=(n-1)%Nx;  Y=(n-1)//Nx
            d=(event.xdata-X)**2+(event.ydata-Y)**2
            if d<best_d: best_d=d; best_n=n
        if best_d>(RING_R*3.5)**2: return
        # clicking any ring adds it as a sweep channel (if not already present)
        pkey=f'heater_{best_n}'
        existing_pkeys = [pr.pkey for pr in self._param_rows]
        if pkey not in existing_pkeys:
            self._h_sweep.add(best_n)
            self._add_param_row(f'Heater {best_n}', pkey,
                                default_fixed=0.0,
                                default_from=0.0, default_to=3.0,
                                removable=True)
            self.status.showMessage(f'Sweep channel added: {pkey}')
            self._draw_lattice()

    def _on_spec_press(self,e): self._spec_drag=True;  self._probe_at(e)
    def _on_spec_move(self,e):
        if self._spec_drag: self._probe_at(e)

    def _probe_at(self, event):
        if self._xv is None: return
        if event.inaxes not in (self.ax_pump,self.ax_side): return
        if event.xdata is None: return
        idx=int(np.argmin(np.abs(self._xv-event.xdata)))
        self.sld_probe.blockSignals(True);  self.sld_probe.setValue(idx)
        self.sld_probe.blockSignals(False);  self._on_probe_change(idx)

    def _on_probe_change(self, idx):
        self.lbl_probe.setText(f'Probe: step {idx}')
        # sync spinbox without re-triggering
        self.spn_probe.blockSignals(True);  self.spn_probe.setValue(idx);  self.spn_probe.blockSignals(False)
        if self._results is not None:
            self._draw_lattice()
            self._update_cross_sections(idx)
            self._update_heatmap_vlines(idx)
        if self._xv is not None and idx < len(self._xv):
            xp = self._xv[idx]
            for vl in (self._vline_pump, self._vline_side,
                       self._vline_pump_all, self._vline_side_all):
                if vl is not None: vl.set_xdata([xp, xp])
            self.canvas_plots.draw_idle()
        # move schedule preview vlines to current probe step
        if getattr(self, '_sched_vline', None) is not None:
            self._sched_vline.set_xdata([idx, idx])
        if getattr(self, '_sched_vline_psi', None) is not None:
            self._sched_vline_psi.set_xdata([idx, idx])
        if getattr(self, '_sched_vline_h', None) is not None:
            self._sched_vline_h.set_xdata([idx, idx])
        self.canvas_sched.draw_idle()
        # update parameter readout
        self._update_step_params(idx)

    def _on_probe_spinbox_change(self, val):
        self.sld_probe.blockSignals(True);  self.sld_probe.setValue(val);  self.sld_probe.blockSignals(False)
        self._on_probe_change(val)

    def _update_step_params(self, idx):
        """Show schedule parameter values at the selected step."""
        if self._results is None or not hasattr(self, '_sched_snapshot'):
            self.lbl_step_params.setText('—')
            return
        sched = self._sched_snapshot
        parts = []
        for pr in self._param_rows:
            arr = pr.get_array(sched.n_steps)
            if pr.pkey == 'psi':
                val = arr[idx] / np.pi
                parts.append(f'Psi={val:.4f}π')
            elif pr.pkey == 'F':
                parts.append(f'F={arr[idx]:.4f}')
            elif pr.pkey == 'detuning':
                parts.append(f'Δ={arr[idx]:.6f}')
            elif pr.pkey.startswith('heater_'):
                site = pr.pkey.split('_')[1]
                parts.append(f'H{site}={arr[idx]:.3f}J')
        self.lbl_step_params.setText('  '.join(parts) if parts else '—')

    def _update_cross_sections(self, idx):
        """Plot |a_W|² and |a_T|² at the selected step for osite."""
        if self._results is None: return
        a_T     = self._results['a_T']          # (NLattice, NumFSRs, N_steps)
        FSRVec  = self._results['FSRVec']
        osite_i = self._ls['osite'] - 1
        a_T_step = a_T[osite_i, :, idx]        # (NumFSRs,)
        a_W_step = np.fft.fft(a_T_step, norm='ortho')
        # pump is at index NF//2 (=PumpFSR), FSRVec already runs -fsr…+fsr
        # so a_W_step[i] corresponds to FSRVec[i] — no fftshift needed
        pW = np.abs(a_W_step)**2
        pT = np.abs(a_T_step)**2
        sort_idx = np.argsort(FSRVec)

        ax = self.ax_xsec_aW;  ax.cla();  ax.set_facecolor(PANEL_BG)
        pW_log = 10 * np.log10(pW + 1e-200)
        ax.plot(FSRVec[sort_idx], pW_log[sort_idx], color='#ff8c42', lw=1.2)
        ax.set_title(f'10·log|a_W|²  step {idx}  (osite)', color='#ff8c42', fontsize=8, pad=3)
        ax.set_xlabel('FSR  μ', color=TEXT_DIM, fontsize=7)
        ax.tick_params(colors=TEXT_COL, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
        ax.grid(True, color=GRID_COL, lw=0.4)
        fsr = len(FSRVec) // 2
        ax.set_xticks([-fsr, 0, fsr]); ax.set_xticklabels([f'{-fsr}', '0', f'{fsr}'], fontsize=7)

        ax = self.ax_xsec_aT;  ax.cla();  ax.set_facecolor(PANEL_BG)
        ax.plot(pT, color='#a8ff78', lw=1.2)
        ax.set_title(f'|a_T|²  step {idx}  (osite)', color='#a8ff78', fontsize=8, pad=3)
        ax.set_xlabel('θ index', color=TEXT_DIM, fontsize=7)
        ax.tick_params(colors=TEXT_COL, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#3a4560')
        ax.grid(True, color=GRID_COL, lw=0.4)
        ax.set_xticks([0, len(pT)-1]); ax.set_xticklabels(['0', '2π'], fontsize=7)

        self.canvas_xsec.draw_idle()

    def _update_heatmap_vlines(self, idx):
        """Overlay a vertical dashed line at the selected step on all heatmaps."""
        for key, vl_attr in [('aT_out','_hvl_aT_out'),('aT_all','_hvl_aT_all'),
                              ('aW_out','_hvl_aW_out'),('aW_all','_hvl_aW_all')]:
            ax = {'aT_out':self.ax_aT_out,'aT_all':self.ax_aT_all,
                  'aW_out':self.ax_aW_out,'aW_all':self.ax_aW_all}[key]
            vl = getattr(self, vl_attr, None)
            if vl is None:
                vl = ax.axvline(idx, color='cyan', lw=1, ls='--', alpha=0.7)
                setattr(self, vl_attr, vl)
            else:
                vl.set_xdata([idx, idx])
        self.canvas_heat.draw_idle()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _on_run(self):
        n  = self.spn_nsteps.value()
        dp = self.spn_dynp.value()
        ls = self._ls
        self._nl_session_folder = None   # reset so folder name reflects new params

        sched = Schedule(n)
        for pr in self._param_rows:
            arr = pr.get_array(n)
            if pr.pkey == 'psi':
                arr = arr * np.pi   # user enters in units of π
            sched.add(pr.pkey, arr)

        x_lbl        = 'Step'
        primary_pkey = next((pr.pkey for pr in self._param_rows if pr.is_active()), 'detuning')

        # Inherited heaters are baked into H_mat by Linear.py.
        # Subtract them so SweepRunner can re-apply cleanly via
        # the heater_N schedule channels without double-counting.
        H_clean = ls['H_mat'][1:,1:].copy()
        for site, delta in ls.get('heaters', {}).items():
            if delta != 0:
                H_clean[site - 1, site - 1] -= delta

        fp = dict(
            H_mat        = H_clean,
            NLattice     = ls['NL'],
            ISite        = ls['isite'],  OSite = ls['osite'],
            kin          = ls['kin'],    kex   = ls['kex'],
            one_side_fsr = self.spn_fsr.value(),
            D2           = self.spn_D2.value(),
            NIter        = self.spn_niter.value(),
            TStep        = self.spn_dt.value(),
            # lattice params needed for ψ H rebuild
            h_str        = ls['h_str'],
            nx0          = ls['nx0'],    ny0   = ls['ny0'],
            nx1          = ls['nx1'],    ny1   = ls['ny1'],
            j1           = ls.get('j1', 0.3),
            phi_iqh0     = ls.get('phi_iqh0', np.pi/2),
            phi_iqh1     = ls.get('phi_iqh1', np.pi/2),
            phi_aqh0     = ls.get('phi_aqh0', np.pi/4),
            phi_aqh1     = ls.get('phi_aqh1', np.pi/4),
        )

        NumFSRs=2*fp['one_side_fsr']+1
        FSRVec=np.arange(-fp['one_side_fsr'],fp['one_side_fsr']+1)
        PumpFSR=fp['one_side_fsr'];  OSite_i=fp['OSite'];  steps=n

        self._xv          = np.arange(steps, dtype=float)
        self._pump_n      = np.zeros(steps);  self._side_n     = np.zeros(steps)
        self._pump_n_all  = np.zeros(steps);  self._side_n_all = np.zeros(steps)
        self._FSRVec      = FSRVec;  self._PumpFSR=PumpFSR;  self._n_steps=steps

        self._sched_snapshot = sched   # keep for parameter readout at any step
        self._init_spec_plots(x_lbl)
        self.sld_probe.setRange(0,steps-1);  self.sld_probe.setValue(0)
        self.spn_probe.setRange(0,steps-1);  self.spn_probe.setValue(0)
        self.prog_bar.setValue(0)
        self.btn_run.setEnabled(False);  self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False);  self.btn_slow.setEnabled(False)
        self.btn_slow.setEnabled(False)
        self.lbl_log.setText('Running...');  self._stop_flag.clear()

        runner=SweepRunner(fp);  _q=self._q

        def worker():
            import time
            def cb(gg,aT,sp,elapsed):
                # output ring
                aW  = np.fft.fft(aT[OSite_i-1,:],norm='ortho');  P  = np.abs(aW)**2
                pv  = P[PumpFSR]
                mask= np.ones(NumFSRs,bool);  mask[PumpFSR]=False
                sv  = P[mask].sum()
                # all rings summed
                aW_all = np.fft.fft(aT,axis=1,norm='ortho')  # (NLattice, NumFSRs)
                P_all  = np.abs(aW_all)**2
                Psum   = P_all.sum(axis=0)
                pv_all = Psum[PumpFSR]
                sv_all = Psum[mask].sum()
                em,es=divmod(int(elapsed/(gg+1)*(steps-gg-1)) if gg>0 else 0,60)
                _q.put(('step',gg,pv,sv,pv_all,sv_all,em,es))
                if self._stop_flag.is_set(): raise StopIteration
            try:
                res=runner.run(sched,callback=cb)
                if self._stop_flag.is_set():
                    _q.put(('stopped', res))
                else:
                    _q.put(('done', res))
            except Exception as exc:
                _q.put(('error',str(exc)));  import traceback; traceback.print_exc()

        threading.Thread(target=worker,daemon=True).start()
        self._poll_timer.start()

    def _poll_queue(self):
        dp=self.spn_dynp.value();  n=self._n_steps
        try:
            while True:
                msg=self._q.get_nowait(); kind=msg[0]
                if kind=='step':
                    _,gg,pv,sv,pv_all,sv_all,em,es=msg
                    self._pump_n[gg]=pv;      self._side_n[gg]=sv
                    self._pump_n_all[gg]=pv_all; self._side_n_all[gg]=sv_all
                    nd=gg+1
                    # advance the schedule vlines
                    if getattr(self, '_sched_vline', None) is not None:
                        self._sched_vline.set_xdata([gg, gg])
                    if getattr(self, '_sched_vline_psi', None) is not None:
                        self._sched_vline_psi.set_xdata([gg, gg])
                    if getattr(self, '_sched_vline_h', None) is not None:
                        self._sched_vline_h.set_xdata([gg, gg])
                    self.canvas_sched.draw_idle()
                    if gg%dp==0 or gg==n-1:
                        self._update_spec_plots(nd,int(self.sld_probe.value()))
                        self._draw_lattice()
                    self.prog_bar.setValue(int(100*nd/n))
                    self.lbl_eta.setText(f'{em}m{es:02d}s left')
                    self.status.showMessage(f'Step {nd}/{n}  |  ETA {em}m{es:02d}s')
                elif kind=='done':    self._on_sweep_done(msg[1])
                elif kind=='stopped': self._on_sweep_stopped(msg[1])
                elif kind=='error':   self._on_sweep_finished(f'Error: {msg[1]}')
        except queue.Empty:
            pass

    def _on_sweep_done(self, res):
        self._results = res
        self._update_spec_plots(self._n_steps, int(self.sld_probe.value()))
        self._draw_lattice()
        self._draw_heat_maps()
        self._update_cross_sections(int(self.sld_probe.value()))
        self._update_step_params(int(self.sld_probe.value()))
        self.btn_save.setEnabled(True)
        self._on_sweep_finished('Done.')

    def _on_sweep_stopped(self, res):
        """Stop was clicked — store and display whatever steps completed."""
        if res is not None and res.get('n_done', 0) > 0:
            self._results = res
            n_done = res['n_done']
            probe  = min(int(self.sld_probe.value()), n_done - 1)
            self._update_spec_plots(n_done, probe)
            self._draw_lattice()
            self._draw_heat_maps()
            self._update_cross_sections(probe)
            self._update_step_params(probe)
            self.btn_save.setEnabled(True)
            self._on_sweep_finished(f'Stopped — {n_done} steps saved.')
        else:
            self._on_sweep_finished('Stopped.')

    def _draw_heat_maps(self):
        if self._results is None: return
        a_T = self._results['a_T']          # (NLattice, NumFSRs, N_steps)
        a_W = np.fft.fft(a_T, axis=1, norm='ortho')
        OSite_i = self._ls['osite'] - 1    # 0-indexed

        datasets = {
            'aT_out': np.abs(a_T[OSite_i, :, :])**2,          # (NumFSRs, N_steps)
            'aT_all': np.abs(a_T).sum(axis=0)**2,              # sum over rings
            'aW_out': np.abs(a_W[OSite_i, :, :])**2,
            'aW_all': np.abs(a_W).sum(axis=0)**2,
        }

        axes = {
            'aT_out': self.ax_aT_out,
            'aT_all': self.ax_aT_all,
            'aW_out': self.ax_aW_out,
            'aW_all': self.ax_aW_all,
        }
        ylabels = {
            'aT_out': 'θ index', 'aT_all': 'θ index',
            'aW_out': 'FSR  μ',  'aW_all': 'FSR  μ',
        }
        titles = {
            'aT_out': ('|a_T|²  (output ring)', '#a8ff78'),
            'aT_all': ('|a_T|²  (all rings)',    '#a8ff78'),
            'aW_out': ('|a_W|²  (output ring)',  '#ff8c42'),
            'aW_all': ('|a_W|²  (all rings)',    '#ff8c42'),
        }
        FSRVec = self._results['FSRVec']
        steps  = a_T.shape[2]
        NumFSRs = a_T.shape[1]

        for key, data in datasets.items():
            ax  = axes[key]
            ttl, col = titles[key]
            ymin = FSRVec[0]  if 'aW' in key else 0
            ymax = FSRVec[-1] if 'aW' in key else NumFSRs - 1

            if 'aW' in key:
                # log scale: add tiny floor before log to avoid -inf
                data_plot = 10 * np.log10(data + 1e-200)
                vmin_plot = data_plot.max() - 60   # show 60 dB of dynamic range
                vmax_plot = data_plot.max() or 0.
                cmap_key  = 'hot'
            else:
                # linear, normalised 0-1
                mx = data.max() or 1.
                data_plot = data / mx
                vmin_plot, vmax_plot = 0, 1
                cmap_key  = 'viridis'

            if key not in self._heat_imgs:
                img = ax.imshow(data_plot, aspect='auto', origin='lower',
                                extent=[0, steps-1, ymin, ymax],
                                cmap=cmap_key, vmin=vmin_plot, vmax=vmax_plot,
                                interpolation='nearest')
                self._heat_imgs[key] = img
                ax.set_xlim(0, steps-1);  ax.set_ylim(ymin, ymax)
            else:
                self._heat_imgs[key].set_data(data_plot)
                self._heat_imgs[key].set_extent([0, steps-1, ymin, ymax])
                self._heat_imgs[key].set_clim(vmin_plot, vmax_plot)
                self._heat_imgs[key].set_cmap(cmap_key)

            ax.set_title(ttl, color=col, fontsize=8, pad=3)
            ax.set_xlabel('Step', color=TEXT_DIM, fontsize=7)
            ax.set_ylabel(ylabels[key], color=TEXT_DIM, fontsize=7)
            ax.tick_params(colors=TEXT_COL, labelsize=7)
            # x-axis: leftmost and rightmost only
            ax.set_xticks([0, steps-1])
            ax.set_xticklabels(['0', f'{steps-1}'], fontsize=7)
            if 'aW' in key:
                one_side = NumFSRs // 2
                ax.set_yticks([-one_side, 0, one_side])
                ax.set_yticklabels([f'{-one_side}', '0', f'{one_side}'], fontsize=7)
            else:
                ax.set_yticks([0, NumFSRs - 1])
                ax.set_yticklabels(['0', '2π'], fontsize=7)

        self.canvas_heat.draw_idle()

    def _on_sweep_finished(self, msg):
        self._poll_timer.stop()
        self.btn_run.setEnabled(True);  self.btn_stop.setEnabled(False)
        self.lbl_log.setText(msg);  self.status.showMessage(msg)

    def _open_slow_time(self):
        if self._results is None or not hasattr(self, '_sched_snapshot'):
            self.status.showMessage('No simulation data — run first.'); return
        idx      = int(self.sld_probe.value())
        a_T_step = self._results['a_T'][:, :, idx].copy()
        niter    = 10000   # NIter for slow-time run; adjustable inside SlowTimeWindow
        sched    = self._sched_snapshot
        sp       = sched.at(idx)

        # Inherited heaters already baked into H_mat — subtract them so
        # SlowTimeWindow can apply heater_N channels cleanly
        H_clean = self._ls['H_mat'][1:,1:].copy()
        for site, delta in self._ls.get('heaters', {}).items():
            if delta != 0:
                H_clean[site - 1, site - 1] -= delta

        # Create step subfolder inside the NL session folder
        nl_folder  = self._get_nl_folder()
        slow_folder = os.path.join(nl_folder, f'step_{idx}')
        os.makedirs(slow_folder, exist_ok=True)

        params = dict(
            a_T_init     = a_T_step,
            step_idx     = idx,
            NIter_more   = niter,
            detuning     = sp.get('detuning', 1.5),
            F            = sp.get('F',        0.3),
            psi          = sp.get('psi',      0.0),
            heaters      = {k: v for k, v in sp.items() if k.startswith('heater_')},
            H_mat        = H_clean,
            NLattice     = self._ls['NL'],
            ISite        = self._ls['isite'],
            OSite        = self._ls['osite'],
            kin          = self._ls['kin'],
            kex          = self._ls['kex'],
            one_side_fsr = self.spn_fsr.value(),
            D2           = self.spn_D2.value(),
            TStep        = self.spn_dt.value(),
            FSRVec       = self._results['FSRVec'],
            # sweep progress power curves for the static indicator panel
            sweep_pump_out  = self._pump_n.copy()     if self._pump_n     is not None else None,
            sweep_side_out  = self._side_n.copy()     if self._side_n     is not None else None,
            sweep_pump_all  = self._pump_n_all.copy() if self._pump_n_all is not None else None,
            sweep_side_all  = self._side_n_all.copy() if self._side_n_all is not None else None,
            # save path for slow time window
            session_folder  = slow_folder,
            # full linear state for geometry/visualization
            linear_state = self._ls,
        )
        self._slow_win = SlowTimeWindow(params, parent=self)
        self._slow_win.show()

    def _on_stop(self):
        self._stop_flag.set();  self.lbl_log.setText('Stopping...')

    def _get_nl_folder(self):
        """Compute (and create) the NL session folder, caching it in self._nl_session_folder."""
        if getattr(self, '_nl_session_folder', None):
            return self._nl_session_folder

        base  = self._ls.get('session_folder') or os.path.join(os.path.expanduser('~'), 'Documents')
        _flt  = lambda v: f'{v:.4g}'.replace('.', 'p')

        n_planned = self.spn_nsteps.value()
        n_done    = (self._results.get('n_done', n_planned)
                     if self._results is not None else n_planned)
        stopped_early = (self._results is not None and n_done < n_planned)
        sched_obj     = (self._results.get('schedule') if self._results else None)

        def _param_str(pkey, prefix):
            rows = [pr for pr in self._param_rows if pr.pkey == pkey]
            if not rows: return ''
            pr   = rows[0]
            mode = pr.mode_combo.currentText()
            if mode == 'Fixed':
                return f"{prefix}{_flt(pr.spn_fixed.value())}"
            elif mode == 'Sweep':
                v_from = pr.spn_from.value()
                v_to   = pr.spn_to.value()
                if (stopped_early and sched_obj is not None
                        and pkey in sched_obj.channels and n_done > 0):
                    v_to = float(sched_obj.channels[pkey][n_done - 1])
                return f"{prefix}{_flt(v_from)}to{_flt(v_to)}"
            else:
                arr = pr.get_array(n_planned)
                return f"{prefix}{_flt(float(arr.min()))}to{_flt(float(arr.max()))}"

        f_str   = _param_str('F',        'F')
        d_str   = _param_str('detuning', 'δ')
        psi_str = _param_str('psi',      'psi')
        extras  = '_'.join(s for s in [f_str, d_str, psi_str] if s)

        name = (f"FSR{self.spn_fsr.value()}"
                f"_D2_{_flt(self.spn_D2.value())}"
                f"_NIter{self.spn_niter.value()}"
                f"_dt{_flt(self.spn_dt.value())}"
                f"_Nsteps{n_done}"
                + (f"_{extras}" if extras else ''))

        folder = os.path.join(base, name)
        os.makedirs(folder, exist_ok=True)
        self._nl_session_folder = folder
        return folder

    def _on_save(self):
        folder = self._get_nl_folder()

        # ── Figures ───────────────────────────────────────────────────────
        n_planned = self.spn_nsteps.value()
        n_done    = (self._results.get('n_done', n_planned)
                     if self._results is not None else n_planned)
        # ── Figures: schedule, sweep power, heatmaps, cross-sections ──────
        for fig_, fname in [
            (self.fig_sched, 'schedule'),
            (self.fig_plots, 'sweep_power'),
            (self.fig_heat,  'heatmaps'),
            (self.fig_xsec,  'cross_sections'),
        ]:
            for fmt in ('png', 'svg'):
                buf = io.BytesIO()
                fig_.savefig(buf, format=fmt, dpi=200, bbox_inches='tight',
                             facecolor=fig_.get_facecolor())
                with open(os.path.join(folder, f'{fname}.{fmt}'), 'wb') as fh:
                    fh.write(buf.getvalue())

        # ── params.npz ────────────────────────────────────────────────────
        ls  = self._ls
        params = dict(
            h_str        = np.array([ls['h_str']]),
            kin          = np.array([ls['kin']]),
            kex          = np.array([ls['kex']]),
            ISite        = np.array([ls['isite']]),
            OSite        = np.array([ls['osite']]),
            one_side_fsr = np.array([self.spn_fsr.value()]),
            D2           = np.array([self.spn_D2.value()]),
            NIter        = np.array([self.spn_niter.value()]),
            TStep        = np.array([self.spn_dt.value()]),
            N_steps      = np.array([n_done]),
            F_schedule   = next((pr.get_array(n_planned) for pr in self._param_rows
                                 if pr.pkey == 'F'), np.array([]))[:n_done],
            D_schedule   = next((pr.get_array(n_planned) for pr in self._param_rows
                                 if pr.pkey == 'detuning'), np.array([]))[:n_done],
        )
        psi_rows = [pr for pr in self._param_rows if pr.pkey == 'psi']
        if psi_rows:
            # Save in radians (multiply by π) to match the live schedule used during sweep
            params['psi_schedule'] = psi_rows[0].get_array(n_planned)[:n_done] * np.pi
        if self._results is not None:
            params['FSRVec']  = self._results['FSRVec']
            params['PumpFSR'] = np.array([self._results['PumpFSR']])
            params['a_T']     = self._results['a_T']
        np.savez(os.path.join(folder, 'nonlinear_params.npz'), **params)

        self.btn_slow.setEnabled(True)
        self.status.showMessage(f'✅ Saved → {folder}')
        self.lbl_log.setText(f'Saved {os.path.basename(folder)}')