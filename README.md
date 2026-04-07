# Topological Photonic Lattice Explorer — v1.0

A standalone desktop application for simulating and visualising **topological photonic lattices**, including linear transmission spectra and nonlinear frequency-comb dynamics. No Jupyter required.

![Splash](original.png)

---

## Features

### Linear Simulator (`Linear.py`)
- **Multiple Hamiltonian types** — IQH×IQH, AQH×AQH, IQH×AQH, AQH×IQH, AQH zigzag, IQH cylinder, AQH cylinder
- Fully vectorised Hamiltonian builders (NumPy — no Python loops)
- Interactive lattice visualisation with clickable sites — add heaters and defects
- Real-time transmission spectra (through, drop, delay, power)
- φ_IQH and φ_AQH phase sliders
- Photon flow visualisation
- Save sessions to `.npz` + PNG/SVG figures
- **Feed to Nonlinear** button — passes the current linear state directly into the nonlinear simulator

### Nonlinear Simulator (`NonLinear.py`)
- Frequency-comb dynamics via split-step Fourier integration
- JAX-accelerated time stepping (`jax.lax.scan` + `@jit`) — runs on CPU or GPU automatically
- Sweep over pump power (F), detuning, and phase (ψ) with fixed or ramped schedules
- Heatmaps, cross-sections, and slow-time analysis windows
- Save sessions to `.npz` + figures

---

## Requirements

| Package | Purpose |
|---|---|
| `PyQt5` | GUI framework |
| `matplotlib` | Plotting |
| `numpy` | Numerics |
| `scipy` | Linear algebra |
| `jax` + `jaxlib` | Accelerated nonlinear solver (CPU/GPU) |
| `pyinstaller` | Building the `.exe` (build time only) |

---

## Running from source

```bash
# Install dependencies (Anaconda recommended)
pip install PyQt5 matplotlib numpy scipy jax jaxlib

# Run
python Linear.py
```

---

## Building the .exe

All 5 files must be in the same folder:

```
Linear.py
NonLinear.py
original.png
icon.ico
TopologicalPhotonic.spec
```

Then in Anaconda Prompt (or any terminal), navigate to the folder and run:

```bash
pip install pyinstaller
pyinstaller TopologicalPhotonic.spec
```

The finished executable will be at:

```
dist/TopologicalPhotonicExplorer.exe
```

It is fully self-contained — no Python or package installation needed on the target machine.

> **Windows note:** On first launch, Windows may show a *"Windows protected your PC"* warning. Click **More info → Run anyway**.

---

## File Overview

| File | Description |
|---|---|
| `Linear.py` | Main application entry point — linear simulator + UI |
| `NonLinear.py` | Nonlinear frequency-comb window, launched from Linear.py |
| `original.png` | Splash screen image shown on startup |
| `icon.ico` | Application icon |
| `TopologicalPhotonic.spec` | PyInstaller build recipe |

---

## License

© Lida Xu. All rights reserved.
