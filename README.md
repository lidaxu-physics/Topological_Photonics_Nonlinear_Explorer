# Topological Photonic Lattice Explorer

**Author:** Lida Xu  
**License:** Proprietary — see [LICENSE](LICENSE) for terms  
**Version:** 1.0

---

## Overview

The Topological Photonic Lattice Explorer is a standalone desktop application for simulating, visualising, and analysing light transport in topological photonic ring-resonator lattices. It covers both the **linear regime** — eigenspectra, thru/drop transmission, Wigner delay, and photon flow — and the **nonlinear regime** — driven Kerr frequency comb generation with slow-time dynamics, Lyapunov analysis, and modal decomposition.

The app requires no Jupyter, no scripts, and no command-line interaction. Everything is driven through an interactive GUI built with PyQt5 and Matplotlib.

---

## Physics Background

### Topological photonic lattices

The simulator models two-dimensional arrays of coupled optical ring resonators on a square lattice. Each resonator is a site; nearest-neighbour (and diagonal, for AQH) coupling between rings is described by a tight-binding Hamiltonian. Two classes of topology are supported:

**IQH — Integer Quantum Hall analogue.** Horizontal hopping acquires a site-dependent Peierls phase `φ × m_y × σ`, where `m_y` is the row index and `σ = ±1` is a pseudospin. This mimics a charged particle hopping in a uniform magnetic flux per plaquette, breaking time-reversal symmetry and producing chiral edge states whose propagation direction is locked to the edge they occupy. The topological invariant is the Chern number `C = ±1`.

**AQH — Anomalous Quantum Hall analogue.** A staggered phase pattern `φ × (−1)^(m_x + m_y) × σ` is applied to both horizontal and vertical bonds, supplemented by next-nearest-neighbour diagonal hoppings. This implements a Haldane-type model with broken time-reversal but zero net flux per unit cell, producing topological edge states without a net magnetic field. The diagonal hoppings are essential — they connect sites of the same sublattice and open the topological gap.

Both models can be arranged as:

- **Flat lattice** with open boundaries (default): edge states appear at the physical boundary.
- **Superlattice** (layer-0 unit cell tiled into a layer-1 supercell): a second set of hopping phases `φ₁` controls inter-unit-cell coupling, enabling nested topological band structures and topological frequency combs.
- **Cylinder** geometry: periodic boundary condition along `x` with an Aharonov–Bohm flux `ψ` threading the cylinder. This is the natural geometry for computing edge-state dispersion as a function of crystal momentum, and for studying chiral transport around the circumference.
- **AQH zigzag ribbon**: a brick-wall connectivity that produces a narrow ribbon with zigzag edges, used to isolate a single pair of counterpropagating edge channels.

### Hamiltonian structure

The tight-binding Hamiltonian `H` is an `N × N` Hermitian matrix (rows/columns 1-indexed, with row/column 0 unused). The diagonal is modified by:

- **Heaters** (on-site detunings): add a real shift `Δ` to `H[n, n]`, representing a thermo-optic or electro-optic phase tuner applied to ring `n`.
- **Defects**: zero out all rows and columns for site `n`, removing it from the lattice — equivalent to a strongly detuned or physically absent resonator.

### Linear transmission spectrum

The app solves the steady-state input–output equations for a lattice driven at the IN port and collected at the OUT port. For a field amplitude `E_n(ω)` in ring `n`:

```
[κ_in δ_nm + κ_ex (δ_nisite + δ_nosite) δ_nm − i H_nm] E_m = √(2κ_ex) δ_nisite
```

where `κ_in` is the intrinsic loss rate, `κ_ex` is the external (coupling) loss rate, `H` is the Hamiltonian, and `ω` is the probe detuning. The system is diagonalised once (`H = V Λ V⁻¹`) and the response at all frequencies is evaluated in O(N²) per point via element-wise division in the eigenbasis — roughly 3× faster than solving a linear system at each frequency.

Outputs:
- **Thru port** power `|1 − √(2κ_ex) E_isite|²` (normalised)
- **Drop port** power `2κ_ex |E_osite|²` (normalised)
- **Wigner group delay** `τ(ω) = −dφ/dω / (2π)` at the drop port, computed via `numpy.gradient` on the unwrapped phase
- **Ring-field power map** `|E_n(ω)|²` for all sites, used to colour the lattice visualisation at the probe frequency
- **Photon flow arrows** from the imaginary part of the bond currents `J_{mn} = Im(E_m* H_{mn} E_n)`

### Nonlinear frequency comb (Kerr)

The nonlinear window simulates driven Kerr comb generation in the lattice using a split-step Fourier method. The equation of motion for the field array `a[n, k]` (site `n`, FSR mode `k`) is:

```
da/dt = −(κ/2) a + i H a + i g |a|² a + F δ_{n,isite} δ_{k,pump}
```

where `g` is the self-phase modulation coefficient (implicit, absorbed into dimensionless units), `F` is the pump amplitude, and `k` indexes modes spaced by one free spectral range (FSR). The split-step integrator applies the nonlinear phase `exp(i g |a|² dt/2)` in the time domain and the linear evolution `exp((−κ/2 + i λ_j) dt)` in the eigenmode basis, with a half-step sandwich (Strang splitting). When JAX is installed the inner loop is JIT-compiled via `jax.lax.scan` for GPU/XLA acceleration; otherwise a pure-NumPy loop is used automatically.

The pump, detuning `δ`, and external flux `ψ` can each be set to a fixed value or swept linearly over the simulation steps — enabling adiabatic comb turn-on, detuning sweeps through Turing/chaos/soliton regimes, and flux-controlled routing studies.

Post-processing features:
- **Comb heatmaps**: power `|a[n, k]|²` as a function of FSR index and slow time step, for every site or averaged over all sites.
- **Pump vs. sideband photon number** time traces, with a probe-frequency slider.
- **Modal decomposition**: project `a(t)` onto the Bloch supermodes of `H` and track power in each supermode — identifies which edge/bulk modes are populated by the comb.
- **Lyapunov exponent** (Wolf et al. algorithm): estimates the largest finite-time Lyapunov exponent `λ` from the slow-time trajectory to classify the dynamics (fixed point `λ < 0`, limit cycle `λ ≈ 0`, chaos `λ > 0`).
- **Slow-time analysis**: FFT of the comb power envelope, cross-correlation between sites, and animation/export of the comb state evolution.

---

## Application Workflow

### Step 1 — Choose and build the lattice

Open the application. The **Lattice** panel (bottom centre) controls the geometry:

| Control | Meaning |
|---|---|
| `Nx₀`, `Ny₀` | Width and height of the unit cell (or full lattice for flat/cylinder types) |
| `Nx₁`, `Ny₁` | Width and height of the supercell tiling (set to 1×1 for no superlattice) |
| `J₁` | Inter-unit-cell hopping strength (relative to intra-cell `J₀ = 1`) |
| Type dropdown | Hamiltonian type: `IQH_IQH`, `AQH_AQH`, `IQH_AQH`, `AQH_IQH`, `A_zigzag`, `IQH_cyl`, `AQH_cyl` |

Click **⟳ Rebuild** after changing dimensions or type. The lattice diagram (centre panel) and photon-flow panel (right) update immediately.

### Step 2 — Set IN and OUT ports

Click the **Set IN** mode button, then click any ring in the lattice diagram to designate it as the input port. Repeat with **Set OUT** for the output port. The status label confirms the selection. Ports can be changed at any time; doing so clears the computed spectrum.

### Step 3 — Add heaters and defects

Select **Heater** mode (default), click a ring, and drag the **on-site detuning (J)** slider or type into the spinbox. Positive values blue-detune the ring; negative values red-detune. The ring colour changes to orange to indicate a non-zero heater.

Select **Remove** mode and click a ring to toggle a defect (the ring is crossed out and removed from the Hamiltonian). Click again to restore it.

### Step 4 — Configure loss rates and phases

In the **Lattice** panel:
- `κ_in` slider: intrinsic loss rate (units of `J₀`), range 0.001–0.050
- `κ_ex` slider: external coupling rate, range 0.010–0.200

In the **Simulation** panel, the visible phase sliders depend on the Hamiltonian type:
- `φ_IQH_layer_0 / _layer_1`: Peierls phases for IQH layers (0 to 2π)
- `φ_AQH_layer_0 / _layer_1`: Haldane phases for AQH layers (0 to 2π)
- `ψ_ext`: Aharonov–Bohm flux threading the cylinder (cylinder types only)

Set the **sweep window**: Start/End detuning (units of `J₀`) and Step size. A smaller step gives higher frequency resolution but takes longer. Typical values: Start = 1.5, End = −1.5, Step = 0.0001.

### Step 5 — Compute the linear spectrum

Click **▶ Compute**. The computation runs in a background thread; the UI remains responsive. When complete:
- The three spectrum plots (Thru, Drop, Wigner delay) are populated.
- The lattice rings are coloured by field intensity at the probe frequency (white = centre of sweep).
- The photon-flow panel shows bond-current arrows.

Click anywhere on a spectrum plot (or drag) to move the **probe marker** (white dashed line) to that frequency. The lattice colours and flow arrows update in real time to show the field distribution at the selected detuning.

### Step 6 — Save the linear session

Set the **Save to** path (or click **Browse**) and click **💾 Save**. A timestamped session folder is created containing:

```
<session_folder>/
    linear_params.npz     ← all simulation parameters + spectrum arrays
    spectra.png / .svg
    lattice.png  / .svg
    flow.png     / .svg
```

The folder name encodes the key parameters (e.g. `II_6611_kex_0p01_kin_0p001_phi_0p5pi_0p5pi_I_1_O_31`). After saving, the **⇢ Feed to Nonlinear Simulator** button becomes active.

### Step 7 — Feed to nonlinear simulator

Click **⇢ Feed to Nonlinear Simulator**. The NonlinearWindow opens, inheriting the full lattice Hamiltonian, port assignments, loss rates, and phase settings from the linear session. No re-entry of parameters is needed.

---

## Nonlinear Simulator Workflow

### Configure the comb parameters

In the nonlinear window's top panel:

| Control | Meaning |
|---|---|
| FSR half-range | Number of FSR modes on each side of the pump (total modes = 2×FSR+1) |
| D₂ | Second-order dispersion coefficient (GHz/FSR²); shifts mode frequencies quadratically |
| NIter | Number of split-step iterations per slow-time step |
| TStep | Slow-time step size `dt` |
| N steps | Total number of slow-time steps to record |

### Set pump, detuning, and flux schedules

Each of the three driven parameters (pump amplitude `F`, detuning `δ`, and flux `ψ`) has a row with a mode selector:

- **Fixed**: the parameter is constant throughout the simulation.
- **Sweep**: the parameter ramps linearly from `From` to `To` over the N steps — use this for adiabatic comb turn-on or detuning sweeps.

### Run the simulation

Click **▶ Run**. Progress is shown in the progress bar and log label. The simulation can be stopped early; results up to that point are retained. On completion, all analysis panels populate automatically.

### Analyse results

**Spectrum panel** (top): comb power spectrum `|a[k]|²` at the probe step, for the OUT site and summed over all sites. Drag the probe slider to scan through slow time.

**Heatmap panel** (centre): power `|a[n, k, t]|²` as a 2D image — FSR index on the x-axis, slow-time step on the y-axis. One heatmap per lattice site (scrollable), plus a lattice-averaged heatmap.

**Cross-section panel**: horizontal and vertical slices through the heatmap at the probe step.

**Step-params panel**: shows the instantaneous values of `F`, `δ`, and `ψ` at the probe step.

**Modal decomposition** (button): projects the comb field onto the eigenmodes of `H` at each slow-time step, producing a power-per-mode vs. time plot. Identifies edge-mode vs. bulk-mode participation.

**Lyapunov analysis** (button): runs the Wolf et al. finite-time Lyapunov exponent algorithm on the slow-time trajectory of the comb power. A theory window explains the interpretation. Results are overlaid on the power trace.

**Slow-time analysis** (button): opens a dedicated panel with FFT of the comb power envelope, site-to-site cross-correlation, and GIF/MP4 export of the evolving comb state.

### Save the nonlinear session

Click **💾 Save** in the nonlinear window. Results are saved into the same session folder as the linear data:

```
<session_folder>/
    nonlinear_params.npz   ← a_T array, schedules, stepper params
    comb_spectrum.png / .svg
    heatmap.png  / .svg
    ...
```

---

## Loading a Previous Session

In the main linear window, click **📂 Load Existing Sim** and select a `linear_params.npz` file. All parameters, spectrum, and plots are restored. You are then offered the option to also load a `nonlinear_params.npz` from the same folder, which reopens the nonlinear window fully populated with the saved comb simulation.

---

## Keyboard / Mouse Reference

| Action | Gesture |
|---|---|
| Move probe frequency | Click or drag on any spectrum plot |
| Select / heater a ring | Click ring in lattice (Heater mode) |
| Set IN port | Switch to Set IN mode, click ring |
| Set OUT port | Switch to Set OUT mode, click ring |
| Toggle defect | Switch to Remove mode, click ring |
| Reset all parameters | Click **⟳ Reset All** in Simulation panel |

---

## Building the Executable

Install dependencies once:

```bash
pip install pyinstaller PyQt5 matplotlib numpy scipy
pip install jax jaxlib          # optional — CPU JAX
# pip install jax[cuda12]       # optional — GPU JAX (NVIDIA only)
```

Build the `.exe` (all four files must be in the same folder):

```bash
pyinstaller TopologicalPhotonic.spec
```

Output: `dist\TopologicalPhotonicExplorer.exe` — fully self-contained, no Python required on the target machine. If JAX is not installed the nonlinear stepper falls back to a pure-NumPy loop automatically.

---

## File Reference

| File | Role |
|---|---|
| `Linear.py` | Main application entry point — linear lattice explorer |
| `NonLinear.py` | Nonlinear comb simulator window, opened from `Linear.py` |
| `TopologicalPhotonic.spec` | PyInstaller build recipe |
| `original.png` | Splash screen background image |
| `icon.ico` | Application window icon |
| `LICENSE` | Proprietary licence — see file for terms |

---

## License

Copyright © 2026 Lida Xu. All rights reserved.  
This software is proprietary. Access is granted solely for collaboration and review as explicitly authorised by the owner. Copying, redistribution, modification, or use in other projects is strictly prohibited without prior written permission. See [LICENSE](LICENSE) for full terms.
