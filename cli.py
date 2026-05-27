"""
Typer CLI entry point for nanoparticle adsorption scan calculations.

Commands
--------
scan-npdosing            : bound fraction vs receptor density, in vivo, single receptor
                           (parameters from system_variables_invivo, or --config YAML)
scan-npdosing-langmuir   : bound fraction vs receptor density, in vitro SPR / Langmuir
                           (parameters from system_variables_invitro, or --config YAML)
scan-multi-npdosing      : bound fraction vs receptor density, in vivo, multi-receptor
                           (parameters from system_variables_invivo_multi, or --config YAML)
scan-both-polymer-models : gaussian vs Flory-exact polymer model comparison
                           (parameters from system_variables_invivo_multi via _sbpm, or --config YAML)
scan-combinations        : receptor density sweeps for every single binder and pair of binders
                           defined in a CSV binder×receptor KD matrix
scan-dosing-langmuir     : NP dosing sweep for different σ_R values, in vitro SPR / Langmuir
                           (Lennart fraction and surface coverage vs NP concentration)
scan-dosing-combinations : dosing sweep for every single binder and pair from a KD CSV
generate-template        : write a commented YAML parameter template to a file
                           (--context invivo|invitro for focused context-specific templates)

Usage
-----
  nanoads <command> [--output-dir DIR]                         # after: pip install -e .
  nanoads <command> [--output-dir DIR] [--config params.yaml]  # with YAML override
  python cli.py <command> [--output-dir DIR]                   # direct execution

Patchable sweep parameters
--------------------------
Module-level variables prefixed _npdosing_*, _langmuir_*, _multi_* set the receptor
density sweep range, resolution, and NP concentration factors for each command.
They are read at call time (not import time) so test code can monkey-patch them:

  import cli
  cli._npdosing_n_pts = 5
  cli.scan_npdosing(output_dir=Path("/tmp/out"))

For scan-both-polymer-models, sweep parameters live in scan_both_polymer_models.py
(_sbpm.n_pts_1D, _sbpm.sigma_R_range, etc.) and are patched on that module directly.
"""
import os
import copy
import csv
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Annotated, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer
from pathlib import Path
from mpmath import mp
from units import *                           # kT, nm, um2, mL, nM, g, mm2
from adsorption import MultivalentBinding, Nmonomers
import scan_both_polymer_models as _sbpm      # safe: computation guarded in __main__

mp.dps = 50
assert mp.dps >= 30

# ── Per-command patchable parameters (read at call time) ──────────────────────
# Override these at module level before calling a command to change sweep resolution/range.
_npdosing_n_pts     = 200              # number of sigma_R points on the log axis
_npdosing_sigma_min = 1.0 / um2        # minimum receptor density [nm^-2] (= 1 um^-2)
_npdosing_sigma_max = 2000 / um2       # maximum receptor density [nm^-2] (= 2000 um^-2)
_npdosing_factors   = [10**k for k in range(-5, 2)]  # NP_conc multipliers: 1e-5 … 10

_langmuir_n_pts     = 200
_langmuir_sigma_min = 1.0 / um2
_langmuir_sigma_max = 2000 / um2
_langmuir_factors   = [10**k for k in range(-5, 2)]

_multi_n_pts        = 200
_multi_sigma_min    = 1.0 / um2
_multi_sigma_max    = 2000 / um2
_multi_factors      = [10**k for k in range(-5, 2)]

_dosing_n_pts          = 200
_dosing_min_fac        = 1e-5   # dimensionless NP_conc multiplier
_dosing_max_fac        = 10.0
_dosing_sigma_R_curves = [1.0, 10.0, 100.0, 500.0, 1000.0]  # µm⁻²


# ── YAML template (written verbatim by generate-template) ────────────────────
_YAML_TEMPLATE = """\
# Nanoparticle adsorption parameter file
# Use with: nanoads <command> --config <this_file>
#
# Units for every field are stated in inline comments.
# Fields under "In-vivo" and "In-vitro" sections are command-specific:
#   scan-npdosing, scan-multi-npdosing  → use In-vivo fields
#   scan-npdosing-langmuir              → uses In-vitro fields
#   scan-both-polymer-models            → uses only NP / PEG / binding / receptors / ligands

# ── Nanoparticle geometry ──────────────────────────────────────────────────────
R_NP_nm: 35.0           # Core NP radius [nm]
N_ligands: 150          # Number of binding ligands grafted on the NP surface

# ── PEG chain parameters ───────────────────────────────────────────────────────
PEG_monomer_size_nm: 0.28         # Projected monomer length [nm]
PEG_kuhn_length_nm: 0.76          # Kuhn segment length [nm]
PEG_ligand_MW_g_per_mol: 3400.0   # MW of PEG tether carrying the binding ligand [g/mol]
PEG_short_MW_g_per_mol: 2000.0    # MW of inert short PEG spacer chains [g/mol]
PEG_short_to_ligand_ratio: 11.4   # sigma_PEG2K / sigma_ligands (inert-to-ligand grafting ratio)

# ── Binding thermodynamics ─────────────────────────────────────────────────────
KD_nM: 150.0                # Solution ligand–receptor dissociation constant [nM]
binder_linear_size_nm: 3.5  # Linear size of the binding domain [nm] (e.g., Fab antibody)
nonspec_interaction_kT: 0.0 # Nonspecific NP–cell interaction energy [kT]; 0 = none

# ── In-vivo biological target (scan-npdosing, scan-multi-npdosing) ─────────────
# Concentrations are derived from primary biological/dosing parameters:
#   cell_conc = N_lympho × T_cell_fraction / V_spleen
#   NP_conc   = Npdosing_per_mL × Vdosing_mL × fTzone / VTzone_mL
# To override the derived values directly, add NP_conc_per_mL and/or
# cell_conc_per_mL — these take precedence when present.
N_lympho: 7.5e7                # Lymphocytes in mouse spleen
T_cell_fraction: 0.25          # Fraction of lymphocytes that are T cells
V_spleen_mm3: 100.0            # Spleen volume [mm³]
A_cell_um2: 100.0              # Target cell surface area [µm²]
Npdosing_per_mL: 8.0e12        # NP concentration in dosing solution [NP/mL]
Vdosing_mL: 0.1                # Dosing volume per animal [mL]  (5 mL/kg × 0.02 kg)
fTzone: 0.1                    # Fraction of dosed particles that reach the T zone
VTzone_mL: 0.042               # Volume of spleen T zone [mL]

# ── In-vitro / SPR target (scan-npdosing-langmuir, scan-dosing-langmuir, scan-dosing-combinations)
# NP_conc = Npdosing_SPR_per_mL;  cell_conc = 1/V_SPR  (single-chip geometry)
A_SPR_mm2: 1.0                 # SPR chip area [mm²]
V_SPR_mL: 6.0e-5               # Volume of solution above the SPR chip [mL]
Npdosing_SPR_per_mL: 4.0e11    # NP concentration in SPR solution [NP/mL]

# ── Receptor definitions ───────────────────────────────────────────────────────
# List one entry per distinct receptor type.
# For a 1D sweep (one receptor axis): list a single receptor.
# For a 2D sweep (two independent receptor axes): list two receptors with distinct names.
receptors:
  - name: default      # Receptor identifier used in output labels and plot axes

# ── Polymer / ligand configuration ────────────────────────────────────────────
# type: inert   → steric PEG chain, no binding
# type: binding → binding-active chain; must name a receptor defined above
#                 KD_nM can be set per ligand to override the global KD_nM
# All ligands with the same receptor name share one receptor object → 1D sweep.
# Ligands referencing distinct receptor names → independent sweep axes → 2D sweep.
ligands:
  - name: PEG2K
    type: inert
  - name: ligands
    type: binding
    receptor: default
    # KD_nM: 150.0    # uncomment to override global KD_nM for this ligand type


# ── Multi-ligand same-receptor example (uncomment to use) ─────────────────────
# ligands:
#   - name: PEG2K
#     type: inert
#   - name: ligands
#     type: binding
#     receptor: default
#     KD_nM: 10000.0
#   - name: ligands2
#     type: binding
#     receptor: default    # same name → same receptor dict → 1D sweep
#     KD_nM: 10000.0


# ── Two distinct receptors (2D sweep, uncomment to use) ───────────────────────
# receptors:
#   - name: CD44
#   - name: CD8
# ligands:
#   - name: PEG2K
#     type: inert
#   - name: ligands_CD44
#     type: binding
#     receptor: CD44
#     KD_nM: 150.0
#   - name: ligands_CD8
#     type: binding
#     receptor: CD8
#     KD_nM: 500.0


# ── Codependent receptor densities ────────────────────────────────────────────
# Tie one or more secondary receptor densities to a primary receptor's sweep axis.
# sigma_secondary = ratio × sigma_primary at every grid point.
# For scan-both-polymer-models: secondary/primary must match the 'receptors' list above.
# For scan-combinations: secondary/primary must match receptor names in the CSV.
# At most 2 *primary* (independent-axis) receptors are allowed.
# Omit this key or leave it empty to sweep all listed receptors independently.
# codependent_receptors:
#   - secondary: CD19       # receptor whose density is derived (not swept independently)
#     primary:   CD44       # receptor that drives the sweep axis
#     ratio:     0.5        # sigma_CD19 = 0.5 × sigma_CD44 at every grid point


# ── Sweep configuration (scan-both-polymer-models only) ───────────────────────
sigma_R_min_per_um2: 1.0       # lower bound of σ_R sweep [µm⁻²]
sigma_R_max_per_um2: 2000.0    # upper bound of σ_R sweep [µm⁻²]
n_pts_1D: 50                   # grid points for 1D sweep
n_pts_2D: 20                   # grid points per axis for 2D sweep (total = n_pts_2D²)

polymer_models:                # valid names: gaussian, Flory-exact, Flory-approx
  - gaussian
  - Flory-exact

# Reference density markers drawn as vertical/cross-hair lines on plots [µm⁻²].
# If provided, target_sigma_R_labels must have the same length as target_sigma_R.
target_sigma_R: []             # e.g. [100.0, 500.0]
target_sigma_R_labels: []      # e.g. ["healthy tissue", "tumour"]
target_sigma_R_axes: []        # per-line axis: '1' (x), '2' (y), or 'both' (default)
target_sigma_R_panels: []      # per-line panel set for langmuir: '1', '2', '3', or 'all' (default)
npdosing_factors: []           # NP dosing multipliers for scan-npdosing-langmuir (default: 10^-5 … 10)

# ── Dosing sweep (scan-dosing-langmuir, scan-dosing-combinations) ──────────────
dosing_min_factor: 1.0e-5         # minimum NP dosing multiplier (dimensionless)
dosing_max_factor: 10.0           # maximum NP dosing multiplier (dimensionless)
dosing_n_pts: 200                 # grid points for the NP dosing axis

# σ_R curve set — two mutually exclusive ways to specify; explicit list takes priority.
# Option A — explicit list [µm⁻²]:
dosing_sigma_R_curves: []         # e.g. [1.0, 10.0, 100.0, 500.0, 1000.0]
# Option B — range spec (used when dosing_sigma_R_curves is empty):
# dosing_sigma_R_min: 1.0         # lower bound [µm⁻²]
# dosing_sigma_R_max: 1000.0      # upper bound [µm⁻²]
# dosing_sigma_R_n_pts: 5         # number of curves
# dosing_sigma_R_scale: log       # 'log' or 'linear'

dosing_sigma_R_labels: []         # optional labels; same length as resolved curve set
"""


_YAML_TEMPLATE_INVIVO = """\
# Nanoparticle adsorption parameter file — in-vivo context
# Commands: scan-npdosing, scan-multi-npdosing, scan-both-polymer-models, scan-combinations

# ── Nanoparticle geometry ──────────────────────────────────────────────────────
R_NP_nm: 35.0           # Core NP radius [nm]
N_ligands: 150          # Number of binding ligands grafted on the NP surface

# ── PEG chain parameters ───────────────────────────────────────────────────────
PEG_monomer_size_nm: 0.28         # Projected monomer length [nm]
PEG_kuhn_length_nm: 0.76          # Kuhn segment length [nm]
PEG_ligand_MW_g_per_mol: 3400.0   # MW of PEG tether carrying the binding ligand [g/mol]
PEG_short_MW_g_per_mol: 2000.0    # MW of inert short PEG spacer chains [g/mol]
PEG_short_to_ligand_ratio: 11.4   # sigma_PEG2K / sigma_ligands (inert-to-ligand grafting ratio)

# ── Binding thermodynamics ─────────────────────────────────────────────────────
KD_nM: 150.0                # Solution ligand–receptor dissociation constant [nM]
binder_linear_size_nm: 3.5  # Linear size of the binding domain [nm]
nonspec_interaction_kT: 0.0 # Nonspecific NP–cell interaction energy [kT]; 0 = none

# ── In-vivo biological target ──────────────────────────────────────────────────
# cell_conc = N_lympho × T_cell_fraction / V_spleen
# NP_conc   = Npdosing_per_mL × Vdosing_mL × fTzone / VTzone_mL
N_lympho: 7.5e7                # Lymphocytes in mouse spleen
T_cell_fraction: 0.25          # Fraction of lymphocytes that are T cells
V_spleen_mm3: 100.0            # Spleen volume [mm³]
A_cell_um2: 100.0              # Target cell surface area [µm²]
Npdosing_per_mL: 8.0e12        # NP concentration in dosing solution [NP/mL]
Vdosing_mL: 0.1                # Dosing volume per animal [mL]
fTzone: 0.1                    # Fraction of dosed particles that reach the T zone
VTzone_mL: 0.042               # Volume of spleen T zone [mL]

# ── Receptor definitions ───────────────────────────────────────────────────────
receptors:
  - name: default

# ── Polymer / ligand configuration ────────────────────────────────────────────
ligands:
  - name: PEG2K
    type: inert
  - name: ligands
    type: binding
    receptor: default
    # KD_nM: 150.0

# ── Codependent receptor densities ────────────────────────────────────────────
# codependent_receptors:
#   - secondary: CD19
#     primary:   CD44
#     ratio:     0.5

# ── Sweep configuration ────────────────────────────────────────────────────────
sigma_R_min_per_um2: 1.0       # lower bound of σ_R sweep [µm⁻²]
sigma_R_max_per_um2: 2000.0    # upper bound of σ_R sweep [µm⁻²]
n_pts_1D: 50                   # grid points for 1D sweep
n_pts_2D: 20                   # grid points per axis for 2D sweep

polymer_models:
  - gaussian
  - Flory-exact

target_sigma_R: []             # reference density markers [µm⁻²]
target_sigma_R_labels: []      # labels for reference markers
target_sigma_R_axes: []        # per-line axis: '1' (x), '2' (y), or 'both'
"""

_YAML_TEMPLATE_INVITRO = """\
# Nanoparticle adsorption parameter file — in-vitro / SPR context
# Commands: scan-npdosing-langmuir, scan-dosing-langmuir, scan-dosing-combinations

# ── Nanoparticle geometry ──────────────────────────────────────────────────────
R_NP_nm: 35.0           # Core NP radius [nm]
N_ligands: 150          # Number of binding ligands grafted on the NP surface

# ── PEG chain parameters ───────────────────────────────────────────────────────
PEG_monomer_size_nm: 0.28         # Projected monomer length [nm]
PEG_kuhn_length_nm: 0.76          # Kuhn segment length [nm]
PEG_ligand_MW_g_per_mol: 3400.0   # MW of PEG tether carrying the binding ligand [g/mol]
PEG_short_MW_g_per_mol: 2000.0    # MW of inert short PEG spacer chains [g/mol]
PEG_short_to_ligand_ratio: 11.4   # sigma_PEG2K / sigma_ligands (inert-to-ligand grafting ratio)

# ── Binding thermodynamics ─────────────────────────────────────────────────────
KD_nM: 150.0                # Solution ligand–receptor dissociation constant [nM]
binder_linear_size_nm: 3.5  # Linear size of the binding domain [nm]
nonspec_interaction_kT: 0.0 # Nonspecific NP–cell interaction energy [kT]; 0 = none

# ── In-vitro / SPR geometry ────────────────────────────────────────────────────
# NP_conc = Npdosing_SPR_per_mL;  cell_conc = 1/V_SPR  (single-chip geometry)
A_SPR_mm2: 1.0                 # SPR chip area [mm²]
V_SPR_mL: 6.0e-5               # Volume of solution above the SPR chip [mL]
Npdosing_SPR_per_mL: 4.0e11    # NP concentration in SPR solution [NP/mL]

# ── Receptor definitions ───────────────────────────────────────────────────────
receptors:
  - name: default

# ── Polymer / ligand configuration ────────────────────────────────────────────
ligands:
  - name: PEG2K
    type: inert
  - name: ligands
    type: binding
    receptor: default
    # KD_nM: 150.0

# ── Sweep configuration (scan-npdosing-langmuir) ──────────────────────────────
sigma_R_min_per_um2: 1.0       # lower bound of σ_R sweep [µm⁻²]
sigma_R_max_per_um2: 2000.0    # upper bound of σ_R sweep [µm⁻²]
n_pts_1D: 50                   # grid points for σ_R axis

polymer_models:
  - Flory-exact

target_sigma_R: []             # reference density markers [µm⁻²]
target_sigma_R_labels: []      # labels for reference markers
target_sigma_R_panels: []      # per-line panel: '1', '2', '3', or 'all'
npdosing_factors: []           # NP dosing multipliers (default: 10^-5 … 10)

# ── Dosing sweep (scan-dosing-langmuir, scan-dosing-combinations) ──────────────
dosing_min_factor: 1.0e-5         # minimum NP dosing multiplier (dimensionless)
dosing_max_factor: 10.0           # maximum NP dosing multiplier (dimensionless)
dosing_n_pts: 200                 # grid points for the NP dosing axis

# σ_R curve set — two mutually exclusive ways to specify; explicit list takes priority.
# Option A — explicit list [µm⁻²]:
dosing_sigma_R_curves: []         # e.g. [1.0, 10.0, 100.0, 500.0, 1000.0]
# Option B — range spec (used when dosing_sigma_R_curves is empty):
# dosing_sigma_R_min: 1.0         # lower bound [µm⁻²]
# dosing_sigma_R_max: 1000.0      # upper bound [µm⁻²]
# dosing_sigma_R_n_pts: 5         # number of curves
# dosing_sigma_R_scale: log       # 'log' or 'linear'

dosing_sigma_R_labels: []         # optional labels; same length as resolved curve set
"""


def _load_system_vars_yaml(path: Path) -> dict:
    """Load system variables from a YAML config file.

    Returns a flat dict of all variables in internal (nm-based) units.
    Receptor dict objects are shared by identity for all ligands naming the same receptor,
    preserving the sweep-dimensionality semantics of scan_both_polymer_models.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml is required for --config support: pip install pyyaml")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    def _req(key):
        if key not in cfg:
            raise ValueError(f"YAML config missing required key: '{key}'")
        return cfg[key]

    # NP geometry
    R_NP    = _req("R_NP_nm") * nm
    N_lig   = int(_req("N_ligands"))
    sigma_L = N_lig / (4.0 * np.pi * R_NP**2)

    # PEG parameters
    amono     = _req("PEG_monomer_size_nm") * nm
    akuhn     = _req("PEG_kuhn_length_nm") * nm
    NmonoL    = Nmonomers(_req("PEG_ligand_MW_g_per_mol") * g)
    NmonoS    = Nmonomers(_req("PEG_short_MW_g_per_mol") * g)
    ratio     = _req("PEG_short_to_ligand_ratio")
    sigma_P2K = sigma_L * ratio

    # Binding thermodynamics
    # KD_nM_global is optional: only required when a binding ligand in 'ligands:' has no
    # per-ligand KD_nM.  scan-combinations reads KDs from the CSV, so it can omit this key.
    KD_nM_global = cfg.get("KD_nM", None)
    binder_size  = _req("binder_linear_size_nm") * nm
    nonspec      = _req("nonspec_interaction_kT") * kT

    # Receptor dicts — one Python object per unique receptor name (shared-reference semantics).
    # 'receptors' is optional; scan-combinations reads receptors from the CSV instead.
    rec_list: list = cfg.get("receptors", []) or []
    _rec_dicts: dict = {rec["name"]: {"name": rec["name"]} for rec in rec_list}

    # Build data_polymers from the 'ligands' section.
    # 'ligands' is optional; scan-combinations builds data_polymers per run from the CSV.
    lig_list: list = cfg.get("ligands", []) or []
    data_polymers: dict = {}
    for lig in lig_list:
        lname = lig["name"]
        ltype = lig.get("type", "binding")
        if ltype == "inert":
            data_polymers[lname] = {
                "N": NmonoS, "a": amono, "sigma": sigma_P2K,
                "name": lname, "akuhn": akuhn,
            }
        else:
            rec_name = lig.get("receptor")
            if rec_name is None:
                raise ValueError(f"Binding ligand '{lname}' has no 'receptor' key")
            if rec_name not in _rec_dicts:
                raise ValueError(
                    f"Ligand '{lname}' references receptor '{rec_name}' "
                    f"not found in 'receptors' list"
                )
            KD_lig = lig.get("KD_nM", KD_nM_global)
            if KD_lig is None:
                raise ValueError(
                    f"Binding ligand '{lname}' has no KD_nM and no global KD_nM is set. "
                    "Add KD_nM globally or per ligand."
                )
            data_polymers[lname] = {
                "N": NmonoL, "a": amono, "sigma": sigma_L,
                "name": lname, "akuhn": akuhn,
                "K_bind_0": (KD_lig * nM) ** (-1),
                "receptor": _rec_dicts[rec_name],
                "binder_linear_size": binder_size,
            }

    first_rec = _rec_dicts[rec_list[0]["name"]] if rec_list else None

    # ── In-vivo context (scan-npdosing, scan-multi-npdosing) ─────────────────
    # cell_conc = N_lympho × T_cell_fraction / V_spleen
    # NP_conc   = Npdosing × Vdosing × fTzone / VTzone
    # If the direct-override keys NP_conc_per_mL / cell_conc_per_mL are present
    # they take precedence over the derived values.
    _N_lympho       = float(cfg.get("N_lympho",        7.5e7))
    _T_cell_frac    = float(cfg.get("T_cell_fraction", 0.25))
    _V_spleen       = float(cfg.get("V_spleen_mm3",    100.0)) * mm3
    _cell_conc_bio  = _N_lympho * _T_cell_frac / _V_spleen
    A_cell          = float(cfg.get("A_cell_um2",      100.0)) * um2
    cell_conc = (float(cfg["cell_conc_per_mL"]) / mL
                 if "cell_conc_per_mL" in cfg else _cell_conc_bio)

    _Npdosing       = float(cfg.get("Npdosing_per_mL", 8e12))  / mL   # [nm⁻³]
    _Vdosing        = float(cfg.get("Vdosing_mL",      0.1))   * mL   # [nm³]
    _fTzone         = float(cfg.get("fTzone",           0.1))
    VTzone          = float(cfg.get("VTzone_mL",        0.042)) * mL
    _NP_conc_bio    = _Npdosing * _Vdosing * _fTzone / VTzone
    NP_conc = (float(cfg["NP_conc_per_mL"]) / mL
               if "NP_conc_per_mL" in cfg else _NP_conc_bio)

    # ── In-vitro / SPR context (scan-npdosing-langmuir) ──────────────────────
    # NP_conc = Npdosing_SPR_per_mL directly; cell_conc = 1/V_SPR (single chip).
    # Accepts both 'Npdosing_SPR_per_mL' (new) and 'NP_conc_SPR_per_mL' (old) for
    # backwards compatibility with existing YAML files.
    A_SPR         = float(cfg.get("A_SPR_mm2",  1.0   )) * mm2
    V_SPR         = float(cfg.get("V_SPR_mL",   6.0e-5)) * mL
    _npdosing_spr = cfg.get("Npdosing_SPR_per_mL", cfg.get("NP_conc_SPR_per_mL", 4.0e11))
    NP_conc_spr   = float(_npdosing_spr) / mL
    cell_conc_spr = 1.0 / V_SPR

    # ── Codependent receptor densities ────────────────────────────────────────
    # Parsed without name validation here; each command validates against its own
    # receptor source (scan-both-polymer-models: YAML receptors list via _sbpm;
    # scan-combinations: CSV receptor names in Step D).
    # Exception: if 'receptors' was provided in this YAML, validate against it now
    # to catch typos at load time for scan-both-polymer-models workflows.
    raw_codep = cfg.get("codependent_receptors", []) or []
    codep: dict = {}
    for entry in raw_codep:
        sec = entry["secondary"]
        pri = entry["primary"]
        if _rec_dicts:
            if sec not in _rec_dicts:
                raise ValueError(
                    f"codependent_receptors secondary '{sec}' not in receptors list"
                )
            if pri not in _rec_dicts:
                raise ValueError(
                    f"codependent_receptors primary '{pri}' not in receptors list"
                )
        codep[sec] = (pri, float(entry["ratio"]))

    # Sweep-control parameters (scan-both-polymer-models)
    # target_sigma_R is in µm⁻² (display units, used directly by axvline and _ref_label).
    # sigma_R_min/max are in nm⁻² (internal units); convert / um2.
    sweep_target_sigma_R        = [float(x) for x in cfg.get("target_sigma_R", []) or []]
    sweep_target_sigma_R_labels = list(cfg.get("target_sigma_R_labels", []) or [])
    sweep_target_sigma_R_axes   = list(cfg.get("target_sigma_R_axes",   []) or [])
    sweep_target_sigma_R_panels = list(cfg.get("target_sigma_R_panels", []) or [])
    sweep_npdosing_factors      = list(cfg.get("npdosing_factors",      []) or [])
    sweep_models                = list(cfg.get("polymer_models", ["gaussian", "Flory-exact"]) or [])
    sweep_n_pts_1D              = int(cfg.get("n_pts_1D", 50))
    sweep_n_pts_2D              = int(cfg.get("n_pts_2D", 20))
    sweep_sigma_R_min           = float(cfg.get("sigma_R_min_per_um2", 1.0   )) / um2
    sweep_sigma_R_max           = float(cfg.get("sigma_R_max_per_um2", 2000.0)) / um2

    # Dosing sweep parameters (scan-dosing-langmuir, scan-dosing-combinations)
    dosing_min_factor     = float(cfg.get("dosing_min_factor",  1e-5))
    dosing_max_factor     = float(cfg.get("dosing_max_factor",  10.0))
    dosing_n_pts          = int(cfg.get("dosing_n_pts",          200))
    dosing_sigma_R_labels = list(cfg.get("dosing_sigma_R_labels", []) or [])

    # σ_R curve set: explicit list takes priority; range spec used when list is absent/empty.
    dosing_sigma_R_curves = [float(x) for x in cfg.get("dosing_sigma_R_curves", []) or []]
    if not dosing_sigma_R_curves and (
        "dosing_sigma_R_min" in cfg or "dosing_sigma_R_max" in cfg
    ):
        _sr_min   = float(cfg.get("dosing_sigma_R_min",   1.0))
        _sr_max   = float(cfg.get("dosing_sigma_R_max", 1000.0))
        _sr_npts  = int(cfg.get("dosing_sigma_R_n_pts",     5))
        _sr_scale = str(cfg.get("dosing_sigma_R_scale",  "log"))
        if _sr_scale == "log":
            dosing_sigma_R_curves = list(
                np.logspace(np.log10(_sr_min), np.log10(_sr_max), _sr_npts)
            )
        else:
            dosing_sigma_R_curves = list(np.linspace(_sr_min, _sr_max, _sr_npts))
    # If still empty, the calling command falls back to its module global _dosing_sigma_R_curves.

    return {
        "R_NP":                  R_NP,
        "data_polymers":         data_polymers,
        "nonspec_interaction":   nonspec,
        "receptor":              first_rec,
        "A_cell":                A_cell,
        "cell_conc":             cell_conc,
        "NP_conc":               NP_conc,
        "VTzone":                VTzone,
        "A_SPR":                 A_SPR,
        "V_SPR":                 V_SPR,
        "NP_conc_spr":           NP_conc_spr,
        "cell_conc_spr":         cell_conc_spr,
        "codependent_receptors": codep,
        "target_sigma_R":        sweep_target_sigma_R,
        "target_sigma_R_labels": sweep_target_sigma_R_labels,
        "target_sigma_R_axes":   sweep_target_sigma_R_axes,
        "target_sigma_R_panels": sweep_target_sigma_R_panels,
        "npdosing_factors":      sweep_npdosing_factors,
        "polymer_models":        sweep_models,
        "n_pts_1D":              sweep_n_pts_1D,
        "n_pts_2D":              sweep_n_pts_2D,
        "sigma_R_min":           sweep_sigma_R_min,
        "sigma_R_max":           sweep_sigma_R_max,
        # NP construction parameters — used by scan-combinations to build data_polymers per run
        "sigma_L":    sigma_L,       # ligand grafting density [nm⁻²]
        "sigma_P2K":  sigma_P2K,     # inert PEG grafting density [nm⁻²]
        "amono":      amono,         # monomer size [nm]
        "akuhn":      akuhn,         # Kuhn segment length [nm]
        "NmonoL":     NmonoL,        # number of segments in ligand PEG chain
        "NmonoS":     NmonoS,        # number of segments in inert PEG chain
        "binder_size": binder_size,  # binder linear size [nm]
        # Dosing sweep keys
        "dosing_min_factor":     dosing_min_factor,
        "dosing_max_factor":     dosing_max_factor,
        "dosing_n_pts":          dosing_n_pts,
        "dosing_sigma_R_curves": dosing_sigma_R_curves,
        "dosing_sigma_R_labels": dosing_sigma_R_labels,
    }


def _read_binders_csv(path: Path) -> dict:
    """Read a binder×receptor KD matrix from a CSV file.

    Rows are receptors; columns are binder IDs. Empty cells mean no binding.
    Returns {binder_id: {receptor_name: kd_nM}} — sparse, missing entries omitted.
    Duplicate column IDs are renamed with suffix _a, _b, ... for robustness.
    Binders with no binding data are silently dropped.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Build unique binder ID list, renaming duplicates in order of occurrence
        seen_ids: dict = {}
        binder_ids: list = []
        for raw_id in [cell.strip() for cell in header[1:]]:
            if raw_id in seen_ids:
                seen_ids[raw_id] += 1
                binder_ids.append(f"{raw_id}_{'abcdefghij'[seen_ids[raw_id] - 1]}")
            else:
                seen_ids[raw_id] = 0
                binder_ids.append(raw_id)

        # Accumulate KD values: binders[binder_id][receptor_name] = kd_nM
        binders: dict = {binder_id: {} for binder_id in binder_ids}
        for row in reader:
            if not row or not row[0].strip():
                continue  # skip blank rows
            receptor_name = row[0].strip()
            for col_index, value in enumerate(row[1:]):
                if col_index < len(binder_ids) and value.strip():
                    binders[binder_ids[col_index]][receptor_name] = float(value)

    # Drop binders that have no binding data at all
    return {binder_id: kd_map for binder_id, kd_map in binders.items() if kd_map}


def _build_data_polymers_for_run(
    binder_ids: list,
    binders_data: dict,
    nanoparticle_params: dict,
    ligand_ratio: float = 0.5,
) -> tuple:
    """Build a data_polymers dict for one sweep run.

    Parameters
    ----------
    binder_ids          : list of 1 or 2 binder ID strings
    binders_data        : {binder_id: {receptor_name: kd_nM}}
    nanoparticle_params : dict with keys sigma_L, sigma_P2K, amono, akuhn,
                          NmonoL, NmonoS, binder_size
    ligand_ratio        : in a 2-binder run, fraction of sigma_L for binder_ids[0];
                          binder_ids[1] gets 1 - ligand_ratio. Ignored for single runs.

    Returns
    -------
    (data_polymers, receptor_registry)
    receptor_registry : {receptor_name: receptor_dict}
        Shared Python dict objects — two ligand entries pointing to the SAME dict
        contribute to the SAME sweep axis (the topology mechanism used by _make_system).
    """
    # Receptor registry: one shared dict per unique receptor name across all binders in this run.
    receptor_registry: dict = {}
    for binder_id in binder_ids:
        for receptor_name in binders_data[binder_id]:
            if receptor_name not in receptor_registry:
                receptor_registry[receptor_name] = {"name": receptor_name}

    # Equal split by default for pairs; full sigma for single-binder runs
    sigma_fractions = [1.0] if len(binder_ids) == 1 else [ligand_ratio, 1.0 - ligand_ratio]

    data_polymers: dict = {}

    # Inert PEG spacer (same composition for every run)
    data_polymers["short"] = {
        "N":     nanoparticle_params["NmonoS"],
        "a":     nanoparticle_params["amono"],
        "sigma": nanoparticle_params["sigma_P2K"],
        "name":  "PEG2K",
        "akuhn": nanoparticle_params["akuhn"],
    }

    # One ligand entry per (binder, receptor) pair.
    # Each entry gets the binder's FULL allocated sigma — not divided by number of receptor types.
    # This matches _expand_multireceptor_ligands semantics: all sigma_L chains of type X are
    # independently available to bind any receptor they have affinity for.
    for binder_id, sigma_fraction in zip(binder_ids, sigma_fractions):
        binder_sigma = nanoparticle_params["sigma_L"] * sigma_fraction
        for receptor_name, kd_nM in binders_data[binder_id].items():
            entry_key = f"lig_{binder_id}_{receptor_name}"
            data_polymers[entry_key] = {
                "N":                  nanoparticle_params["NmonoL"],
                "a":                  nanoparticle_params["amono"],
                "sigma":              binder_sigma,
                "akuhn":              nanoparticle_params["akuhn"],
                "name":               binder_id,
                "K_bind_0":           1.0 / (kd_nM * nM),
                "receptor":           receptor_registry[receptor_name],
                "binder_linear_size": nanoparticle_params["binder_size"],
            }

    return data_polymers, receptor_registry


def _save_results(path: str, results: dict, primary_names: list) -> None:
    """Persist a sweep results dict to a compressed .npz archive."""
    arrays: dict = {
        "_primary_names": np.array(primary_names),
        "_model_names":   np.array(sorted(results.keys())),
    }
    for model, res in results.items():
        for key, val in res.items():
            arrays[f"{model}__{key}"] = np.array(val)
    np.savez_compressed(path, **arrays)


def _load_results(path: str) -> tuple:
    """Load a sweep results dict from a .npz archive.
    Returns (results, primary_names)."""
    data          = np.load(path, allow_pickle=False)
    primary_names = data["_primary_names"].tolist()
    model_names   = data["_model_names"].tolist()
    results: dict = {}
    for model in model_names:
        prefix = f"{model}__"
        res: dict = {}
        for key in data.files:
            if not key.startswith(prefix):
                continue
            subkey = key[len(prefix):]
            val    = data[key]
            res[subkey] = val.tolist() if subkey == "rec_names" else val
        results[model] = res
    return results, primary_names


def _save_langmuir_results(path: str, factor_keys: list, results: dict) -> None:
    """Persist langmuir sweep results preserving insertion order of factor keys."""
    arrays = {"_factor_keys": np.array(factor_keys, dtype=object)}
    for fkey, res in results.items():
        for k, v in res.items():
            arrays[f"{fkey}__{k}"] = np.asarray(v)
    np.savez_compressed(path, **arrays)


def _load_langmuir_results(path: str) -> tuple:
    """Load langmuir sweep results from .npz, restoring insertion order."""
    data        = np.load(path, allow_pickle=True)
    factor_keys = data["_factor_keys"].tolist()
    results     = {}
    for fkey in factor_keys:
        prefix = f"{fkey}__"
        results[fkey] = {k[len(prefix):]: data[k]
                         for k in data.files if k.startswith(prefix)}
    return factor_keys, results


def _write_and_plot_sweep(results: dict, output_dir, primary_names: list,
                          target_axes=None):
    """Write .dat files and PNG plots for a completed receptor-density sweep.

    results      : {model_name: sweep_dict} returned by sweep_1axis or sweep_2axes
    output_dir   : directory where files are written (must already exist)
    primary_names: list of 1 or 2 receptor name strings (determines 1D vs 2D output paths)
    target_axes  : list of '1', '2', or 'both' per entry in _sbpm.target_sigma_R;
                   controls which receptor axis each reference line appears on in 2D plots.
                   Ignored for 1D plots (single axis). Defaults to 'both' for any missing entry.
    """
    n_independent = len(primary_names)
    n_models      = len(results)

    # theta = bound_fraction × NP_conc × (2·R_NP)² / (cell_conc · A_cell)
    # Dimensionless fraction of cell surface physically occupied by adsorbed NPs.
    # _sbpm.* module-level variables are patched by each CLI command before this function runs.
    _NP_excl     = (2.0 * _sbpm.R_NP) ** 2                               # NP footprint [nm²]
    _theta_scale = _sbpm.NP_conc * _NP_excl / (_sbpm.cell_conc * _sbpm.A_cell)  # dimensionless

    for model_name, res in results.items():
        safe_name = model_name.replace("-", "_")
        if n_independent == 1:
            fname = f"adsorption_{safe_name}.dat"
            with open(os.path.join(output_dir, fname), "w") as f:
                f.write("# sigma_R(um^-2)  bound_fraction  theta\n")
                for j in range(len(res["sigma_R_um2"])):
                    f.write(
                        f"{res['sigma_R_um2'][j]:9.4e}  "
                        f"{res['bound_fraction'][j]:9.4e}  "
                        f"{float(res['bound_fraction'][j]) * _theta_scale:9.4e}\n"
                    )
        else:
            fname = f"adsorption_2rec_{safe_name}.dat"
            nr1 = len(res["sigma_R1_um2"])
            nr2 = len(res["sigma_R2_um2"])
            with open(os.path.join(output_dir, fname), "w") as f:
                f.write(
                    f"# {res['rec_names'][0]}(um^-2)  {res['rec_names'][1]}(um^-2)  "
                    f"bound_fraction  theta\n"
                )
                for i in range(nr1):
                    for j in range(nr2):
                        f.write(
                            f"{res['sigma_R1_um2'][i]:9.4e}  "
                            f"{res['sigma_R2_um2'][j]:9.4e}  "
                            f"{res['bound_fraction'][i, j]:9.4e}  "
                            f"{float(res['bound_fraction'][i, j]) * _theta_scale:9.4e}\n"
                        )
        print(f"Written {os.path.join(output_dir, fname)}")

    if n_independent == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        for model_name, res in results.items():
            theta_arr = res["bound_fraction"] * _theta_scale
            ax1.plot(res["sigma_R_um2"], res["bound_fraction"],
                     linestyle="solid", label=model_name)
            ax2.plot(res["sigma_R_um2"], theta_arr,
                     linestyle="solid", label=model_name)
        for k, v in enumerate(_sbpm.target_sigma_R):
            lbl = _sbpm._ref_label(k, v)
            ax1.axvline(x=v, color="gray", linestyle="dashed", linewidth=0.9, label=lbl)
            ax2.axvline(x=v, color="gray", linestyle="dashed", linewidth=0.9, label=lbl)
        ax1.set_xscale("log")
        ax1.set_xlabel(_sbpm._axis_label(primary_names[0]))
        ax1.set_ylabel("Fraction of adsorbed NPs")
        ax1.legend(fontsize="small")
        ax2.set_xscale("log")
        ax2.set_xlabel(_sbpm._axis_label(primary_names[0]))
        ax2.set_ylabel("Fraction of occupied surface")
        ax2.legend(fontsize="small")
        plt.tight_layout()
        out_png = os.path.join(output_dir, "adsorption_polymer_models.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved {out_png}")
    else:
        # Split target_sigma_R by receptor axis for per-axis reference-line control.
        # Entries without an axis tag (or with tag 'both') go to both axes.
        def _axis_split(side):
            axes     = target_axes or []
            orig_sr  = _sbpm.target_sigma_R
            orig_lbl = _sbpm.target_sigma_R_labels
            keep = [i for i in range(len(orig_sr))
                    if (axes[i] if i < len(axes) else "both") in (side, "both")]
            return ([orig_sr[i]  for i in keep],
                    [orig_lbl[i] for i in keep] if orig_lbl else [])

        sr1, sr1_lbl = _axis_split("1")
        sr2, sr2_lbl = _axis_split("2")

        _sbpm._plot_case_b(
            "bound_fraction", "Fraction of adsorbed NPs",
            os.path.join(output_dir, "adsorption_polymer_models_3D.png"),
            os.path.join(output_dir, "adsorption_polymer_models_2D_proj.png"),
            results, n_models,
            _sr1=sr1, _sr1_labels=sr1_lbl,
            _sr2=sr2, _sr2_labels=sr2_lbl,
        )
        # Inject theta under the "n_ads" key so _plot_case_b reads it unchanged.
        results_theta = {
            model: {**res, "n_ads": res["bound_fraction"] * _theta_scale}
            for model, res in results.items()
        }
        _sbpm._plot_case_b(
            "n_ads", "Fraction of occupied surface",
            os.path.join(output_dir, "adsorption_polymer_models_3D_theta.png"),
            os.path.join(output_dir, "adsorption_polymer_models_2D_proj_theta.png"),
            results_theta, n_models,
            _sr1=sr1, _sr1_labels=sr1_lbl,
            _sr2=sr2, _sr2_labels=sr2_lbl,
        )


def _execute_run(run_spec: dict):
    """Execute one scan_combinations run; designed for dispatch to a worker process.

    Receives all required state via run_spec (picklable). Sets _sbpm globals from
    run_spec so each worker process owns its own independent physics state.
    Returns (run_name, None) on success or (None, skip_message) when > 2 axes.
    """
    _sbpm.R_NP                  = run_spec["R_NP"]
    _sbpm.A_cell                = run_spec["A_cell"]
    _sbpm.NP_conc               = run_spec["NP_conc"]
    _sbpm.cell_conc             = run_spec["cell_conc"]
    _sbpm.nonspec_interaction   = run_spec["nonspec_interaction"]
    _sbpm.n_pts_1D              = run_spec["n_pts_1D"]
    _sbpm.n_pts_2D              = run_spec["n_pts_2D"]
    _sbpm.sigma_R_min           = run_spec["sigma_R_min"]
    _sbpm.sigma_R_max           = run_spec["sigma_R_max"]
    _sbpm.target_sigma_R        = run_spec["target_sigma_R"]
    _sbpm.target_sigma_R_labels = run_spec["target_sigma_R_labels"]
    _run_target_axes            = list(run_spec.get("target_sigma_R_axes", []))

    data_polymers, receptor_map = _build_data_polymers_for_run(
        list(run_spec["run_binder_ids"]),
        run_spec["binders_data"],
        run_spec["nanoparticle_params"],
        run_spec["ligand_ratio"],
    )
    _sbpm.data_polymers = data_polymers

    run_codependent = {
        sec: (pri, ratio)
        for sec, (pri, ratio) in run_spec["codependent_map"].items()
        if sec in receptor_map and pri in receptor_map
    }
    _sbpm.codependent_receptors = run_codependent

    secondary_receptors    = set(run_codependent.keys())
    primary_receptor_names = [n for n in receptor_map if n not in secondary_receptors]
    n_primary_axes         = len(primary_receptor_names)
    run_name               = "+".join(run_spec["run_binder_ids"])

    if n_primary_axes > 2:
        return None, (f"SKIP  {run_name}: {n_primary_axes} independent receptor axes "
                      f"({primary_receptor_names}). Use --codependent to reduce axes.")

    run_output_dir = run_spec["run_output_dir"]
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"=== {run_name}  |  receptors: {list(receptor_map)}  "
          f"|  axes: {primary_receptor_names} ===", flush=True)

    cache_path = os.path.join(run_output_dir, "run_results.npz")
    if run_spec.get("replot", False) and os.path.exists(cache_path):
        results, primary_receptor_names = _load_results(cache_path)
        _write_and_plot_sweep(results, run_output_dir, primary_receptor_names,
                              target_axes=_run_target_axes)
        print(f"  [cache] {run_name}", flush=True)
        return run_name, None

    n_cores = run_spec.get("n_cores_per_run", 1)
    results = {}
    for model_name in run_spec["models_to_run"]:
        if n_primary_axes == 1:
            results[model_name] = _sbpm.sweep_1axis(
                model_name, primary_receptor_names[0], n_workers=n_cores)
        else:
            results[model_name] = _sbpm.sweep_2axes(
                model_name, primary_receptor_names[0], primary_receptor_names[1],
                n_workers=n_cores)

    _write_and_plot_sweep(results, run_output_dir, primary_receptor_names,
                          target_axes=_run_target_axes)
    _save_results(cache_path, results, primary_receptor_names)
    return run_name, None


app = typer.Typer(
    help="Nanoparticle adsorption scan calculations.",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command("scan-npdosing")
def scan_npdosing(
    output_dir: Path = typer.Option(
        Path("."), "-o", "--output-dir",
        help="Output directory for data and plot files (created if absent).",
    ),
    config: Annotated[Optional[Path], typer.Option(
        "--config", "-c",
        help="YAML file with system parameters. If omitted, built-in defaults are used.",
    )] = None,
):
    """Bound fraction vs receptor density — in vivo, single receptor type, Flory-exact model.

    Physical context: T cells in mouse spleen (system_variables_invivo). The NP concentration
    is swept over _npdosing_factors multiples of the nominal NP_conc. Receptor fluctuations
    and NP depletion are accounted for via Poisson-averaging.

    Output files
    ------------
    adsorption_Npdosing_x{factor}.dat — columns: sigma_R [um^-2]  bound_fraction  n_ads_VTzone
    adsorption_scan_Npdosing.png      — left: bound_fraction vs sigma_R; right: n_ads (log-log)

    WARNING: output filenames are identical to scan-multi-npdosing; use distinct --output-dir.
    """
    mp.dps = 50
    os.makedirs(output_dir, exist_ok=True)
    if config is not None:
        _v = _load_system_vars_yaml(config)
        R_NP                = _v["R_NP"]
        data_polymers       = _v["data_polymers"]
        receptor            = _v["receptor"]
        A_cell              = _v["A_cell"]
        NP_conc             = _v["NP_conc"]
        cell_conc           = _v["cell_conc"]
        nonspec_interaction = _v["nonspec_interaction"]
        VTzone              = _v["VTzone"]
    else:
        from system_variables_invivo import (
            R_NP, data_polymers, A_cell, NP_conc, cell_conc,
            nonspec_interaction, VTzone, receptor,
        )
    n_sampling_points = _npdosing_n_pts
    sigma_R_min       = _npdosing_sigma_min
    sigma_R_max       = _npdosing_sigma_max
    factors           = list(_npdosing_factors)

    min_exp = np.log10(sigma_R_min)
    max_exp = np.log10(sigma_R_max)
    sigma_R_values = np.logspace(min_exp, max_exp, n_sampling_points)

    system_ref = MultivalentBinding(
        kT=kT, R_NP=R_NP, data_polymers=data_polymers,
        binding_model="exact", polymer_model="Flory-exact",
        A_cell=A_cell, NP_conc=NP_conc, cell_conc=cell_conc,
        nonspec_interaction=nonspec_interaction,
    )

    max_NR_ave = int(mp.pi * R_NP**2 * sigma_R_max)  # mean receptors in NP footprint at sigma_R_max
    # truncate Poisson sum at mean + 4 std; minimum of 50 for low-density regime
    max_n_receptor = max_NR_ave + 4 * (max_NR_ave + 1) + 1 if max_NR_ave > 1 else 50
    print(f"Computing K_bind for NR = 1..{max_n_receptor - 1} (slow step, done once)")
    K_bind_vs_NR = system_ref.calculate_K_bind_vs_receptors(max_n_receptor)
    print("K_bind computation done.")

    M_conc = (A_cell / (2.0 * R_NP)**2) * cell_conc  # available cell surface patches per unit volume [nm^-3]
    results = {}

    for factor in factors:
        NP_conc_i = NP_conc * factor
        label = f"Npdosing x{factor:g}"
        print(f"\n--- {label} (NP_conc = {float(NP_conc_i):.3e} nm^-3) ---")

        sigma_out = np.zeros(n_sampling_points)
        frac_out  = np.zeros(n_sampling_points)
        nads_out  = np.zeros(n_sampling_points)

        for i, sigma_R in enumerate(sigma_R_values):
            receptor["sigma_R"] = sigma_R
            bound_fraction = system_ref.calculate_bound_fraction(
                fluctuations=True, depletion=True,
                K_bind_vs_receptors=K_bind_vs_NR,
                max_n_receptor=max_n_receptor,
                NP_conc=NP_conc_i,
                rho_m=M_conc,
            )
            sigma_out[i] = float(sigma_R / (1 / um2))
            frac_out[i]  = float(bound_fraction)
            nads_out[i]  = float(bound_fraction) * NP_conc_i * VTzone

        results[factor] = (sigma_out, frac_out, nads_out)
        fname = f"adsorption_Npdosing_x{factor:g}.dat"
        with open(os.path.join(output_dir, fname), "w") as f:
            for j in range(n_sampling_points):
                f.write(f"{sigma_out[j]:5.3e} {frac_out[j]:5.3e} {nads_out[j]:5.3e}\n")
        print(f"  Written to {os.path.join(output_dir, fname)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for factor in factors:
        sigma_out, frac_out, nads_out = results[factor]
        ax1.plot(sigma_out, frac_out, linestyle="solid", label=f"Npdosing x{factor:g}")
        ax2.plot(sigma_out, nads_out, linestyle="solid", label=f"Npdosing x{factor:g}")
    ax1.set_xlabel(r"Receptor surface density ($\mu$m$^{-2}$)")
    ax1.set_ylabel("Adsorbed fraction (of amount initially in VTzone volume)")
    ax1.legend()
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel(r"Receptor surface density ($\mu$m$^{-2}$)")
    ax2.set_ylabel("Total # of adsorbed particles in VTzone volume")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "adsorption_scan_Npdosing.png"))
    plt.close()
    print(f"\nPlot saved to {os.path.join(output_dir, 'adsorption_scan_Npdosing.png')}")


@app.command("scan-npdosing-langmuir")
def scan_npdosing_langmuir(
    output_dir: Path = typer.Option(
        Path("."), "-o", "--output-dir",
        help="Output directory for data and plot files (created if absent).",
    ),
    config: Annotated[Optional[Path], typer.Option(
        "--config", "-c",
        help="YAML file with system parameters. If omitted, built-in defaults are used.",
    )] = None,
    n_pts: Annotated[Optional[int], typer.Option(
        "--n-pts",
        help="Grid points for receptor density sweep (overrides YAML/module default).",
    )] = None,
    sigma_r_min: Annotated[Optional[float], typer.Option(
        "--sigma-r-min",
        help="Sweep lower bound [µm⁻²] (overrides YAML/module default).",
    )] = None,
    sigma_r_max: Annotated[Optional[float], typer.Option(
        "--sigma-r-max",
        help="Sweep upper bound [µm⁻²] (overrides YAML/module default).",
    )] = None,
    polymer_model: Annotated[Optional[str], typer.Option(
        "--polymer-model",
        help="Polymer model name (overrides YAML/module default).",
    )] = None,
    npdosing_factor: Annotated[Optional[List[float]], typer.Option(
        "--npdosing-factor",
        help="NP dosing multiplier, repeatable (overrides YAML/module default).",
    )] = None,
    target_sigma_r: Annotated[Optional[List[float]], typer.Option(
        "--target-sigma-r",
        help="Reference density marker [µm⁻²], repeatable.",
    )] = None,
    target_label: Annotated[Optional[List[str]], typer.Option(
        "--target-label",
        help="Label for reference marker (same order as --target-sigma-r), repeatable.",
    )] = None,
    replot: Annotated[bool, typer.Option(
        "--replot",
        help="If run_results.npz already exists, load it and re-plot without recomputing.",
    )] = False,
):
    """Bound fraction vs receptor density — in vitro SPR experiment, Langmuir adsorption.

    Physical context: SPR chip geometry (system_variables_invitro). Unlike the in-vivo commands,
    depletion is disabled (infinite reservoir assumption). An additional Langmuir-limit column is
    written: adsorbed / (solution + adsorbed), as used for SPR signal normalisation.
    Receptor fluctuations are included; the Poisson exclusion limit (1 − exp(−σ_R · A_NP))
    is plotted as a reference curve.

    Output files
    ------------
    adsorption_Npdosing_x{factor}.dat — columns: sigma_R [um^-2]  bound_fraction  n_ads  langmuir_fraction
    adsorption_scan_Npdosing.png      — three panels: bound_fraction, n_ads, langmuir_fraction
    run_results.npz                   — cache; reload with --replot to skip recomputation
    """
    mp.dps = 50
    os.makedirs(output_dir, exist_ok=True)

    # Defaults from module globals
    n_sampling_points = _langmuir_n_pts
    sigma_R_min       = _langmuir_sigma_min
    sigma_R_max       = _langmuir_sigma_max
    _poly_model       = "Flory-exact"
    factors           = list(_langmuir_factors)
    _target_sr:     list = []
    _target_lbl:    list = []
    _target_panels: list = []

    if config is not None:
        _v = _load_system_vars_yaml(config)
        R_NP                = _v["R_NP"]
        data_polymers       = _v["data_polymers"]
        receptor            = _v["receptor"]
        A_SPR               = _v["A_SPR"]
        NP_conc             = _v["NP_conc_spr"]
        cell_conc           = _v["cell_conc_spr"]
        nonspec_interaction = _v["nonspec_interaction"]
        V_SPR               = _v["V_SPR"]
        n_sampling_points   = _v["n_pts_1D"]
        sigma_R_min         = _v["sigma_R_min"]
        sigma_R_max         = _v["sigma_R_max"]
        if len(_v["polymer_models"]) == 1:
            _poly_model     = _v["polymer_models"][0]
        if _v["npdosing_factors"]:
            factors         = list(_v["npdosing_factors"])
        _target_sr          = list(_v["target_sigma_R"])
        _target_lbl         = list(_v["target_sigma_R_labels"])
        _target_panels      = list(_v["target_sigma_R_panels"])
    else:
        from system_variables_invitro import (
            R_NP, data_polymers, A_SPR, NP_conc, cell_conc,
            nonspec_interaction, V_SPR, receptor,
        )

    # CLI overrides (highest priority)
    if n_pts         is not None: n_sampling_points = n_pts
    if sigma_r_min   is not None: sigma_R_min       = sigma_r_min / um2
    if sigma_r_max   is not None: sigma_R_max       = sigma_r_max / um2
    if polymer_model is not None: _poly_model       = polymer_model
    if npdosing_factor:           factors           = list(npdosing_factor)
    if target_sigma_r:            _target_sr        = list(target_sigma_r)
    if target_label:              _target_lbl       = list(target_label)

    cache_path  = os.path.join(str(output_dir), "run_results.npz")
    factor_keys = [f"x{f:g}" for f in factors]

    if replot and os.path.exists(cache_path):
        factor_keys, results = _load_langmuir_results(cache_path)
        print(f"[cache] loaded {cache_path} — re-plotting only.")
    else:
        sigma_R_values = np.logspace(np.log10(sigma_R_min), np.log10(sigma_R_max), n_sampling_points)

        system_ref = MultivalentBinding(
            kT=kT, R_NP=R_NP, data_polymers=data_polymers,
            binding_model="exact", polymer_model=_poly_model,
            A_cell=A_SPR, NP_conc=NP_conc, cell_conc=cell_conc,
            nonspec_interaction=nonspec_interaction,
        )

        max_NR_ave = int((2 * R_NP)**2 * sigma_R_max)
        max_n_receptor = max_NR_ave + 4 * (max_NR_ave + 1) + 1 if max_NR_ave > 1 else 20
        print(f"Computing K_bind for NR = 1..{max_n_receptor - 1} (slow step, done once)")
        K_bind_vs_NR = system_ref.calculate_K_bind_vs_receptors(max_n_receptor)
        print("K_bind computation done.")
        results = {}

        for factor in factors:
            fkey      = f"x{factor:g}"
            NP_conc_i = NP_conc * factor
            print(f"\n--- Npdosing {fkey} (NP_conc = {float(NP_conc_i):.3e} nm^-3) ---")

            bound_vs_receptor = system_ref.calculate_bound_vs_receptors_monodisperse(
                max_n_receptor, depletion=False, verbose=False,
                K_bind_vs_NR=K_bind_vs_NR, NP_conc=NP_conc_i,
            )

            sigma_out             = np.zeros(n_sampling_points)
            frac_out              = np.zeros(n_sampling_points)
            frac_out_max          = np.zeros(n_sampling_points)
            nads_out              = np.zeros(n_sampling_points)
            nads_lennart_fraction = np.zeros(n_sampling_points)

            for i, sigma_R in enumerate(sigma_R_values):
                receptor["sigma_R"] = sigma_R
                bound_fraction = system_ref.calculate_bound_fraction(
                    fluctuations=True, depletion=False,
                    bound_vs_receptor=bound_vs_receptor,
                    max_factor=4,
                )
                max_num_sites   = A_SPR / system_ref.NP_excluded_area
                sigma_out[i]    = float(sigma_R / (1 / um2))
                frac_out[i]     = float(bound_fraction)
                frac_out_max[i] = 1 - np.exp(-sigma_R * system_ref.NP_excluded_area)
                nads_out[i]     = float(bound_fraction) * max_num_sites

            mass_SPR              = V_SPR * NP_conc_i
            nads_lennart_fraction = nads_out / (mass_SPR + nads_out)

            results[fkey] = {
                "sigma_R_um2":       sigma_out,
                "bound_fraction":    frac_out,
                "poisson_limit":     frac_out_max,
                "n_ads":             nads_out,
                "langmuir_fraction": nads_lennart_fraction,
            }

        _save_langmuir_results(cache_path, factor_keys, results)

    # Write .dat files (both normal run and replot)
    for fkey, res in results.items():
        _sr   = res["sigma_R_um2"]
        _bf   = res["bound_fraction"]
        _nads = res["n_ads"]
        _lf   = res["langmuir_fraction"]
        fname = f"adsorption_Npdosing_{fkey}.dat"
        with open(os.path.join(output_dir, fname), "w") as f:
            for j in range(len(_sr)):
                f.write(f"{_sr[j]:5.3e} {_bf[j]:5.3e} {_nads[j]:5.3e} {_lf[j]:5.3e} \n")
        print(f"  Written to {os.path.join(output_dir, fname)}")

    # Plotting
    def _on_panel(k, panel_id):
        tag = (_target_panels[k] if k < len(_target_panels) else "all")
        return tag == "all" or str(panel_id) in tag.split(",")

    first_res = next(iter(results.values()))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    for fkey, res in results.items():
        ax1.plot(res["sigma_R_um2"], res["bound_fraction"],    linestyle="solid", label=fkey)
        ax2.plot(res["sigma_R_um2"], res["n_ads"],             linestyle="solid", label=fkey)
        ax3.plot(res["sigma_R_um2"], res["langmuir_fraction"], linestyle="solid", label=fkey)
    ax1.plot(first_res["sigma_R_um2"], first_res["poisson_limit"],
             linestyle="--", color="black", label="Poisson limit")

    axes_list = [ax1, ax2, ax3]
    for k, v in enumerate(_target_sr):
        lbl = _target_lbl[k] if k < len(_target_lbl) else f"{v:.0f} µm⁻²"
        for panel_id, ax in enumerate(axes_list, start=1):
            if _on_panel(k, panel_id):
                ax.axvline(x=v, color="gray", linestyle="dashed", linewidth=0.9, label=lbl)

    ax1.set_xscale("log")
    ax1.set_xlabel(r"Receptor surface density ($\mu$m$^{-2}$)")
    ax1.set_ylabel("Fraction of adsorbed NPs")
    ax1.legend()
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel(r"Receptor surface density ($\mu$m$^{-2}$)")
    ax2.set_ylabel("Number of adsorbed particles in SPR volume")
    ax2.legend()
    ax3.set_xlabel(r"Receptor surface density ($\mu$m$^{-2}$)")
    ax3.set_ylabel("fraction adsorbed particles (Lennart)")
    ax3.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "adsorption_scan_Npdosing.png"))
    plt.close()
    print(f"\nPlot saved to {os.path.join(output_dir, 'adsorption_scan_Npdosing.png')}")


@app.command("scan-multi-npdosing")
def scan_multi_npdosing(
    output_dir: Path = typer.Option(
        Path("."), "-o", "--output-dir",
        help="Output directory for data and plot files (created if absent).",
    ),
    config: Annotated[Optional[Path], typer.Option(
        "--config", "-c",
        help="YAML file with system parameters. If omitted, built-in defaults are used.",
    )] = None,
):
    """Bound fraction vs receptor density — in vivo, multi-receptor / multi-ligand, Flory-exact.

    Physical context: identical to scan-npdosing but uses system_variables_invivo_multi, which
    can define multiple ligand types binding the same or distinct receptors (see setup A–E
    documentation in system_variables_invivo_multi.py).

    Output files
    ------------
    adsorption_Npdosing_x{factor}.dat — columns: sigma_R [um^-2]  bound_fraction  n_ads_VTzone
    adsorption_scan_Npdosing.png      — left: bound_fraction vs sigma_R; right: n_ads (log-log)

    WARNING: output filenames are identical to scan-npdosing; use distinct --output-dir.
    """
    mp.dps = 50
    os.makedirs(output_dir, exist_ok=True)
    if config is not None:
        _v = _load_system_vars_yaml(config)
        R_NP                = _v["R_NP"]
        data_polymers       = _v["data_polymers"]
        receptor            = _v["receptor"]
        A_cell              = _v["A_cell"]
        NP_conc             = _v["NP_conc"]
        cell_conc           = _v["cell_conc"]
        nonspec_interaction = _v["nonspec_interaction"]
        VTzone              = _v["VTzone"]
    else:
        from system_variables_invivo_multi import (
            R_NP, data_polymers, A_cell, NP_conc, cell_conc,
            nonspec_interaction, VTzone, receptor,
        )
    n_sampling_points = _multi_n_pts
    sigma_R_min       = _multi_sigma_min
    sigma_R_max       = _multi_sigma_max
    factors           = list(_multi_factors)

    min_exp = np.log10(sigma_R_min)
    max_exp = np.log10(sigma_R_max)
    sigma_R_values = np.logspace(min_exp, max_exp, n_sampling_points)

    system_ref = MultivalentBinding(
        kT=kT, R_NP=R_NP, data_polymers=data_polymers,
        binding_model="exact", polymer_model="Flory-exact",
        A_cell=A_cell, NP_conc=NP_conc, cell_conc=cell_conc,
        nonspec_interaction=nonspec_interaction,
    )

    max_NR_ave = int(mp.pi * R_NP**2 * sigma_R_max)  # mean receptors in NP footprint at sigma_R_max
    # truncate Poisson sum at mean + 4 std; minimum of 50 for low-density regime
    max_n_receptor = max_NR_ave + 4 * (max_NR_ave + 1) + 1 if max_NR_ave > 1 else 50
    print(f"Computing K_bind for NR = 1..{max_n_receptor - 1} (slow step, done once)")
    K_bind_vs_NR = system_ref.calculate_K_bind_vs_receptors(max_n_receptor)
    print("K_bind computation done.")

    M_conc = (A_cell / (2.0 * R_NP)**2) * cell_conc  # available cell surface patches per unit volume [nm^-3]
    results = {}

    for factor in factors:
        NP_conc_i = NP_conc * factor
        label = f"Npdosing x{factor:g}"
        print(f"\n--- {label} (NP_conc = {float(NP_conc_i):.3e} nm^-3) ---")

        sigma_out = np.zeros(n_sampling_points)
        frac_out  = np.zeros(n_sampling_points)
        nads_out  = np.zeros(n_sampling_points)

        for i, sigma_R in enumerate(sigma_R_values):
            receptor["sigma_R"] = sigma_R
            bound_fraction = system_ref.calculate_bound_fraction(
                fluctuations=True, depletion=True,
                K_bind_vs_receptors=K_bind_vs_NR,
                max_n_receptor=max_n_receptor,
                NP_conc=NP_conc_i,
                rho_m=M_conc,
            )
            sigma_out[i] = float(sigma_R / (1 / um2))
            frac_out[i]  = float(bound_fraction)
            nads_out[i]  = float(bound_fraction) * NP_conc_i * VTzone

        results[factor] = (sigma_out, frac_out, nads_out)
        fname = f"adsorption_Npdosing_x{factor:g}.dat"
        with open(os.path.join(output_dir, fname), "w") as f:
            for j in range(n_sampling_points):
                f.write(f"{sigma_out[j]:5.3e} {frac_out[j]:5.3e} {nads_out[j]:5.3e}\n")
        print(f"  Written to {os.path.join(output_dir, fname)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for factor in factors:
        sigma_out, frac_out, nads_out = results[factor]
        ax1.plot(sigma_out, frac_out, linestyle="solid", label=f"Npdosing x{factor:g}")
        ax2.plot(sigma_out, nads_out, linestyle="solid", label=f"Npdosing x{factor:g}")
    ax1.set_xlabel(r"Receptor surface density ($\mu$m$^{-2}$)")
    ax1.set_ylabel("Adsorbed fraction (of amount initially in VTzone volume)")
    ax1.legend()
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel(r"Receptor surface density ($\mu$m$^{-2}$)")
    ax2.set_ylabel("Total # of adsorbed particles in VTzone volume")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "adsorption_scan_Npdosing.png"))
    plt.close()
    print(f"\nPlot saved to {os.path.join(output_dir, 'adsorption_scan_Npdosing.png')}")


@app.command("scan-both-polymer-models")
def scan_both_polymer_models_cmd(
    output_dir: Path = typer.Option(
        Path("."), "-o", "--output-dir",
        help="Output directory for data and plot files (created if absent).",
    ),
    config: Annotated[Optional[Path], typer.Option(
        "--config", "-c",
        help="YAML file with system parameters. If omitted, built-in defaults are used.",
    )] = None,
    n_pts_1d: Annotated[Optional[int], typer.Option(
        "--n-pts-1d",
        help="Grid points for 1D receptor density sweep (overrides YAML/module default).",
    )] = None,
    n_pts_2d: Annotated[Optional[int], typer.Option(
        "--n-pts-2d",
        help="Grid points per axis for 2D sweep, total = n_pts_2d² (overrides YAML/module default).",
    )] = None,
    sigma_r_min: Annotated[Optional[float], typer.Option(
        "--sigma-r-min",
        help="Sweep lower bound [µm⁻²] (overrides YAML/module default).",
    )] = None,
    sigma_r_max: Annotated[Optional[float], typer.Option(
        "--sigma-r-max",
        help="Sweep upper bound [µm⁻²] (overrides YAML/module default).",
    )] = None,
    polymer_model: Annotated[Optional[List[str]], typer.Option(
        "--polymer-model",
        help="Polymer model name, repeatable (e.g. --polymer-model gaussian --polymer-model Flory-exact).",
    )] = None,
    codependent: Annotated[Optional[List[str]], typer.Option(
        "--codependent",
        help='Codependent receptor as "secondary:primary:ratio", repeatable.',
    )] = None,
    target_sigma_r: Annotated[Optional[List[float]], typer.Option(
        "--target-sigma-r",
        help="Reference density marker [µm⁻²], repeatable.",
    )] = None,
    target_label: Annotated[Optional[List[str]], typer.Option(
        "--target-label",
        help="Label for reference marker (same order as --target-sigma-r), repeatable.",
    )] = None,
    replot: Annotated[bool, typer.Option(
        "--replot",
        help="If run_results.npz already exists, load it and re-plot without recomputing.",
    )] = False,
):
    """Receptor density sweep comparing gaussian and Flory-exact polymer tether models.

    Parameters are read from scan_both_polymer_models.py (_sbpm), which imports from
    system_variables_invivo_multi. The sweep is 1D (one independent receptor axis) or 2D
    (two independent receptor axes) depending on the receptor configuration and
    _sbpm.codependent_receptors.

    Output files — 1D case
    -----------------------
    adsorption_{model}.dat             — columns: sigma_R [um^-2]  bound_fraction  n_ads [nm^-3]
    adsorption_polymer_models.png      — bound_fraction and n_ads vs sigma_R for all models

    Output files — 2D case
    -----------------------
    adsorption_2rec_{model}.dat        — columns: sigma_R1 [um^-2]  sigma_R2 [um^-2]  bound_fraction  n_ads
    adsorption_polymer_models_3D.png / _2D_proj.png
    adsorption_polymer_models_3D_nads.png / _2D_proj_nads.png
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1 — YAML (if provided): overrides all _sbpm module defaults
    _target_axes: list = []
    if config is not None:
        _v = _load_system_vars_yaml(config)
        _sbpm.data_polymers           = _v["data_polymers"]
        _sbpm.codependent_receptors   = _v["codependent_receptors"]
        _sbpm.target_sigma_R          = _v["target_sigma_R"]
        _sbpm.target_sigma_R_labels   = _v["target_sigma_R_labels"]
        _sbpm.polymer_models          = _v["polymer_models"]
        _sbpm.n_pts_1D                = _v["n_pts_1D"]
        _sbpm.n_pts_2D                = _v["n_pts_2D"]
        _sbpm.sigma_R_min             = _v["sigma_R_min"]
        _sbpm.sigma_R_max             = _v["sigma_R_max"]
        _target_axes                  = list(_v["target_sigma_R_axes"])

    # Step 2 — CLI overrides (only when argument was explicitly supplied)
    if n_pts_1d    is not None:  _sbpm.n_pts_1D               = n_pts_1d
    if n_pts_2d    is not None:  _sbpm.n_pts_2D               = n_pts_2d
    if sigma_r_min is not None:  _sbpm.sigma_R_min            = sigma_r_min / um2
    if sigma_r_max is not None:  _sbpm.sigma_R_max            = sigma_r_max / um2
    if polymer_model:            _sbpm.polymer_models         = list(polymer_model)
    if target_sigma_r:           _sbpm.target_sigma_R         = list(target_sigma_r)
    if target_label:             _sbpm.target_sigma_R_labels  = list(target_label)
    if codependent:
        codep: dict = {}
        for entry in codependent:
            parts = entry.split(":")
            if len(parts) != 3:
                raise typer.BadParameter(
                    f"--codependent must be 'secondary:primary:ratio', got '{entry}'"
                )
            sec, pri, ratio_str = parts
            codep[sec] = (pri, float(ratio_str))
        _sbpm.codependent_receptors = codep

    receptor_map = _sbpm._detect_receptors(_sbpm.data_polymers)

    if _sbpm.target_sigma_R_labels and \
            len(_sbpm.target_sigma_R_labels) != len(_sbpm.target_sigma_R):
        raise typer.BadParameter(
            f"target_sigma_R_labels has {len(_sbpm.target_sigma_R_labels)} entries "
            f"but target_sigma_R has {len(_sbpm.target_sigma_R)}. Lengths must match."
        )

    for sec, (pri, ratio) in _sbpm.codependent_receptors.items():
        if sec not in receptor_map:
            raise typer.BadParameter(
                f"codependent_receptors key '{sec}' not in detected receptors"
            )
        if pri not in receptor_map:
            raise typer.BadParameter(
                f"codependent_receptors primary '{pri}' not in detected receptors"
            )
        if pri in _sbpm.codependent_receptors:
            raise typer.BadParameter(
                f"'{pri}' is listed as both primary and secondary — cycles not allowed"
            )

    secondary_names = set(_sbpm.codependent_receptors.keys())
    primary_names   = [n for n in receptor_map if n not in secondary_names]
    n_independent   = len(primary_names)

    if n_independent > 2:
        raise typer.BadParameter(
            f"{n_independent} independent receptor axes detected "
            f"(primaries: {primary_names}). At most 2 are supported."
        )

    print(f"Detected {len(receptor_map)} receptor type(s): {list(receptor_map.keys())}")
    if _sbpm.codependent_receptors:
        print(f"Codependent receptors: {_sbpm.codependent_receptors}")
    print(f"Independent axes ({n_independent}): {primary_names}")

    models_to_run = list(_sbpm.polymer_models)
    cache_path    = os.path.join(str(output_dir), "run_results.npz")

    if replot and os.path.exists(cache_path):
        results, primary_names = _load_results(cache_path)
        print(f"[cache] loaded {cache_path} — re-plotting only.")
    else:
        results = {}
        for model_name in models_to_run:
            print(f"\n=== Running model: {model_name} ===")
            if n_independent == 1:
                results[model_name] = _sbpm.sweep_1axis(model_name, primary_names[0])
            else:
                results[model_name] = _sbpm.sweep_2axes(
                    model_name, primary_names[0], primary_names[1]
                )
        _save_results(cache_path, results, primary_names)

    print("\nAll sweeps done.")
    _write_and_plot_sweep(results, output_dir, primary_names, target_axes=_target_axes)
    print("\nDone.")


@app.command("scan-combinations")
def scan_combinations_cmd(
    csv_path: Annotated[Path, typer.Option(
        "--csv", "-f",
        help="CSV with binder×receptor KD matrix [nM]. "
             "Rows = receptors, columns = binder IDs. Empty cell = no binding.",
    )],
    output_dir: Path = typer.Option(
        Path("."), "-o", "--output-dir",
        help="Root output directory. Each run writes to its own subdirectory.",
    ),
    config: Annotated[Optional[Path], typer.Option(
        "--config", "-c",
        help="YAML parameter file (NP geometry, PEG, dosing). "
             "If omitted, system_variables_invivo_multi defaults are used.",
    )] = None,
    codependent: Annotated[Optional[List[str]], typer.Option(
        "--codependent",
        help="secondary:primary:ratio — ties a secondary receptor density to a primary. "
             "Repeatable. If > 2 receptors are found and this flag is absent, "
             "an interactive prompt will ask for the mapping.",
    )] = None,
    sigma_r_min: Annotated[Optional[float], typer.Option(
        "--sigma-r-min",
        help="Sweep lower bound [µm⁻²] (overrides YAML/module default).",
    )] = None,
    sigma_r_max: Annotated[Optional[float], typer.Option(
        "--sigma-r-max",
        help="Sweep upper bound [µm⁻²] (overrides YAML/module default).",
    )] = None,
    n_pts_1d: Annotated[Optional[int], typer.Option(
        "--n-pts-1d",
        help="Grid points for 1D sweeps (overrides YAML/module default).",
    )] = None,
    n_pts_2d: Annotated[Optional[int], typer.Option(
        "--n-pts-2d",
        help="Grid points per axis for 2D sweeps (overrides YAML/module default).",
    )] = None,
    polymer_model: Annotated[Optional[List[str]], typer.Option(
        "--polymer-model",
        help="Polymer model name, repeatable (e.g. --polymer-model gaussian).",
    )] = None,
    ligand_ratio: Annotated[float, typer.Option(
        "--ligand-ratio",
        help="In a 2-binder run, fraction of sigma_L given to the first binder "
             "(second binder gets 1 - ratio). Default 0.5.",
    )] = 0.5,
    target_sigma_r: Annotated[Optional[List[float]], typer.Option(
        "--target-sigma-r",
        help="Reference density marker [µm⁻²], repeatable. Overrides YAML value.",
    )] = None,
    target_label: Annotated[Optional[List[str]], typer.Option(
        "--target-label",
        help="Label for reference marker (same order as --target-sigma-r), repeatable. "
             "Overrides YAML value.",
    )] = None,
    n_workers: Annotated[int, typer.Option(
        "--n-workers",
        help="Number of parallel worker processes. Default 1 (serial). "
             "Pass -1 to use all available CPU cores.",
    )] = 1,
    n_cores_per_run: Annotated[int, typer.Option(
        "--n-cores-per-run",
        help="Worker processes used for K_bind precomputation within each combination run "
             "(Level 2 parallelism). Default 4. Set to 1 to disable.",
    )] = 4,
    job_range: Annotated[Optional[str], typer.Option(
        "--job-range",
        help='1-indexed inclusive range "START:END" of non-skipped runs to execute. '
             'Allows splitting a batch across HPC nodes. '
             'Example: --job-range 1:4 runs the first 4 non-skipped runs.',
    )] = None,
    replot: Annotated[bool, typer.Option(
        "--replot",
        help="If run_results.npz already exists for a run, load it and re-plot "
             "without recomputing. Default: always recompute.",
    )] = False,
):
    """Receptor density sweeps for every single binder and pair of binders in a CSV.

    Reads a binder×receptor KD matrix from a CSV file and runs scan_both_polymer_models-
    style sigma_R sweeps for every single binder and every unordered pair of binders.
    Empty cells mean no binding (KD = inf). Results are written to one subdirectory
    per run under --output-dir.

    Output files per run (same as scan-both-polymer-models)
    -------------------------------------------------------
    1D sweep: adsorption_{model}.dat, adsorption_polymer_models.png
    2D sweep: adsorption_2rec_{model}.dat, adsorption_polymer_models_3D.png, _2D_proj.png
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Step A: load NP parameters and keep _sbpm physics globals consistent ──
    yaml_codependent_map: dict = {}
    _target_axes: list = []
    if config is not None:
        config_vars = _load_system_vars_yaml(config)
        yaml_codependent_map = config_vars["codependent_receptors"]
        nanoparticle_params = config_vars  # has sigma_L, amono, akuhn, NmonoL, NmonoS, binder_size
        # Patch _sbpm physics globals so _make_system uses the same NP as nanoparticle_params.
        # _make_system reads R_NP, A_cell, NP_conc, cell_conc, nonspec_interaction by name
        # from _sbpm's module scope — they must match the YAML values used to compute sigma_L.
        _sbpm.R_NP                   = config_vars["R_NP"]
        _sbpm.A_cell                 = config_vars["A_cell"]
        _sbpm.NP_conc                = config_vars["NP_conc"]
        _sbpm.cell_conc              = config_vars["cell_conc"]
        _sbpm.nonspec_interaction    = config_vars["nonspec_interaction"]
        _sbpm.n_pts_1D               = config_vars["n_pts_1D"]
        _sbpm.n_pts_2D               = config_vars["n_pts_2D"]
        _sbpm.sigma_R_min            = config_vars["sigma_R_min"]
        _sbpm.sigma_R_max            = config_vars["sigma_R_max"]
        _sbpm.polymer_models         = config_vars["polymer_models"]
        _sbpm.target_sigma_R         = config_vars["target_sigma_R"]
        _sbpm.target_sigma_R_labels  = config_vars["target_sigma_R_labels"]
        _target_axes                 = list(config_vars["target_sigma_R_axes"])
    else:
        from system_variables_invivo_multi import (
            R_NP, sigma_L, sigma_P2K, amono, akuhn,
            NmonoLigands, NmonoShort, binder_linear_size,
        )
        nanoparticle_params = {
            "R_NP": R_NP, "sigma_L": sigma_L, "sigma_P2K": sigma_P2K,
            "amono": amono, "akuhn": akuhn, "NmonoL": NmonoLigands,
            "NmonoS": NmonoShort, "binder_size": binder_linear_size,
        }
        # _sbpm physics globals are already at these defaults from its module-level import

    # ── Step B: CLI sweep-parameter overrides ────────────────────────────────
    if n_pts_1d      is not None:  _sbpm.n_pts_1D               = n_pts_1d
    if n_pts_2d      is not None:  _sbpm.n_pts_2D               = n_pts_2d
    if sigma_r_min   is not None:  _sbpm.sigma_R_min            = sigma_r_min / um2
    if sigma_r_max   is not None:  _sbpm.sigma_R_max            = sigma_r_max / um2
    if polymer_model:              _sbpm.polymer_models         = list(polymer_model)
    if target_sigma_r:             _sbpm.target_sigma_R        = list(target_sigma_r)
    if target_label:               _sbpm.target_sigma_R_labels = list(target_label)

    # ── Step C: read CSV ──────────────────────────────────────────────────────
    binders_data  = _read_binders_csv(csv_path)
    all_receptors = sorted({r for kd_map in binders_data.values() for r in kd_map})
    binder_list   = sorted(binders_data.keys())
    print(f"Loaded {len(binders_data)} binders across "
          f"{len(all_receptors)} receptor types: {all_receptors}")

    # ── Step D: build codependent receptor map (YAML base + CLI overrides) ────
    # Start from YAML codependencies; --codependent entries override per-key.
    codependent_map: dict = dict(yaml_codependent_map)
    if codependent:
        for entry in codependent:
            parts = entry.split(":")
            if len(parts) != 3:
                raise typer.BadParameter(
                    f"--codependent must be 'secondary:primary:ratio', got '{entry}'"
                )
            secondary, primary, ratio_str = parts
            codependent_map[secondary] = (primary, float(ratio_str))

    # Validate all entries against CSV receptor names.
    if codependent_map:
        for secondary, (primary, _) in codependent_map.items():
            if secondary not in all_receptors:
                raise typer.BadParameter(
                    f"Codependent secondary '{secondary}' not found in CSV receptors: "
                    f"{sorted(all_receptors)}"
                )
            if primary not in all_receptors:
                raise typer.BadParameter(
                    f"Codependent primary '{primary}' not found in CSV receptors: "
                    f"{sorted(all_receptors)}"
                )
    elif len(all_receptors) > 2:
        typer.echo(
            f"\nFound {len(all_receptors)} receptor types: {all_receptors}.\n"
            "Runs with > 2 independent receptor axes will be skipped.\n"
            "You may define codependent receptor densities to reduce the axis count.\n"
            "  Format: secondary:primary:ratio  "
            "(e.g. CD3eg:CD3ed:7.4  means  sigma_CD3eg = 7.4 × sigma_CD3ed)"
        )
        if typer.confirm("Define codependent receptor densities?", default=False):
            while True:
                entry = typer.prompt(
                    "Enter secondary:primary:ratio (empty line to finish)",
                    default="",
                )
                if not entry.strip():
                    break
                parts = entry.split(":")
                if len(parts) != 3:
                    typer.echo("  Invalid format — expected secondary:primary:ratio.")
                    continue
                secondary, primary, ratio_str = parts
                codependent_map[secondary] = (primary, float(ratio_str))
            typer.echo(f"Codependent receptors set: {codependent_map}")

    # ── Step E: build run list, filter skips, dispatch ───────────────────────
    single_binder_runs = [(binder_id,) for binder_id in binder_list]
    pair_runs          = list(itertools.combinations(binder_list, 2))
    all_runs           = single_binder_runs + pair_runs
    print(f"\n{len(single_binder_runs)} single-binder + {len(pair_runs)} pair runs "
          f"= {len(all_runs)} total.\n")

    models_to_run = list(_sbpm.polymer_models)

    # Snapshot physics globals once; each worker receives them by value.
    physics_snapshot = {
        "R_NP":                  _sbpm.R_NP,
        "A_cell":                _sbpm.A_cell,
        "NP_conc":               _sbpm.NP_conc,
        "cell_conc":             _sbpm.cell_conc,
        "nonspec_interaction":   _sbpm.nonspec_interaction,
        "n_pts_1D":              _sbpm.n_pts_1D,
        "n_pts_2D":              _sbpm.n_pts_2D,
        "sigma_R_min":           _sbpm.sigma_R_min,
        "sigma_R_max":           _sbpm.sigma_R_max,
        "target_sigma_R":        list(_sbpm.target_sigma_R),
        "target_sigma_R_labels": list(_sbpm.target_sigma_R_labels),
        "target_sigma_R_axes":   list(_target_axes),
    }

    # Pre-filter: skip runs with > 2 primary axes before dispatching.
    run_specs = []
    for run_binder_ids in all_runs:
        run_name = "+".join(run_binder_ids)
        _, receptor_map_check = _build_data_polymers_for_run(
            list(run_binder_ids), binders_data, nanoparticle_params, ligand_ratio
        )
        run_codependent_check = {
            sec: (pri, ratio)
            for sec, (pri, ratio) in codependent_map.items()
            if sec in receptor_map_check and pri in receptor_map_check
        }
        primary_check = [n for n in receptor_map_check if n not in run_codependent_check]
        if len(primary_check) > 2:
            typer.echo(
                f"SKIP  {run_name}: {len(primary_check)} independent receptor axes "
                f"({primary_check}). Use --codependent to reduce axes."
            )
            continue

        run_specs.append({
            "run_binder_ids":    run_binder_ids,
            "binders_data":      binders_data,
            "nanoparticle_params": nanoparticle_params,
            "ligand_ratio":      ligand_ratio,
            "codependent_map":   codependent_map,
            "run_output_dir":    str(os.path.join(str(output_dir), run_name)),
            "models_to_run":     models_to_run,
            "n_cores_per_run":   n_cores_per_run,
            "replot":     replot,
            **physics_snapshot,
        })

    n_total = len(run_specs)
    print(f"{n_total} non-skipped runs to dispatch.")

    if job_range is not None:
        parts = job_range.split(":")
        if len(parts) != 2:
            raise typer.BadParameter(
                f"--job-range must be 'START:END' (1-indexed, inclusive), got '{job_range}'"
            )
        s, e = int(parts[0]) - 1, int(parts[1])
        if s < 0 or e > n_total or s >= e:
            raise typer.BadParameter(
                f"--job-range {job_range} is out of bounds for {n_total} non-skipped runs "
                f"(valid range: 1:{n_total})"
            )
        run_specs = run_specs[s:e]
        print(f"Job range {job_range}: running {len(run_specs)} of {n_total} runs.")

    actual_workers = os.cpu_count() if n_workers == -1 else n_workers

    if actual_workers == 1:
        for rs in run_specs:
            run_name, skip_msg = _execute_run(rs)
            if skip_msg:
                typer.echo(skip_msg)
    else:
        print(f"Dispatching {len(run_specs)} runs across {actual_workers} workers.")
        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            futures = {pool.submit(_execute_run, rs): rs["run_binder_ids"]
                       for rs in run_specs}
            for future in as_completed(futures):
                run_name, skip_msg = future.result()
                if skip_msg:
                    typer.echo(skip_msg)
                else:
                    typer.echo(f"Completed: {run_name}", err=False)

    print("\nAll runs complete.")


@app.command("scan-dosing-langmuir")
def scan_dosing_langmuir(
    output_dir: Path = typer.Option(
        Path("."), "-o", "--output-dir",
        help="Output directory for data and plot files (created if absent).",
    ),
    config: Annotated[Optional[Path], typer.Option(
        "--config", "-c",
        help="YAML file with system parameters. If omitted, built-in invitro defaults are used.",
    )] = None,
    n_pts: Annotated[Optional[int], typer.Option(
        "--n-pts",
        help="Grid points for NP dosing axis (overrides YAML/module default).",
    )] = None,
    dosing_min: Annotated[Optional[float], typer.Option(
        "--dosing-min",
        help="Minimum NP dosing multiplier, dimensionless (overrides YAML/module default).",
    )] = None,
    dosing_max: Annotated[Optional[float], typer.Option(
        "--dosing-max",
        help="Maximum NP dosing multiplier, dimensionless (overrides YAML/module default).",
    )] = None,
    polymer_model: Annotated[Optional[str], typer.Option(
        "--polymer-model",
        help="Polymer model name (overrides YAML/module default).",
    )] = None,
    sigma_r: Annotated[Optional[List[float]], typer.Option(
        "--sigma-r",
        help="σ_R curve value [µm⁻²], repeatable. Defines the set of σ_R curves to plot. "
             "Overrides YAML dosing_sigma_R_curves and range spec.",
    )] = None,
    sigma_r_label: Annotated[Optional[List[str]], typer.Option(
        "--sigma-r-label",
        help="Label for σ_R curve (same order as --sigma-r), repeatable.",
    )] = None,
    replot: Annotated[bool, typer.Option(
        "--replot",
        help="If run_results.npz already exists, load it and re-plot without recomputing.",
    )] = False,
):
    """NP dosing sweep for different σ_R values — in vitro SPR experiment, Langmuir adsorption.

    Physical context: SPR chip geometry (system_variables_invitro). Sweeps NP concentration
    (x-axis) for a set of receptor surface densities (σ_R; one curve per value). Depletion
    is disabled (infinite reservoir assumption). Receptor fluctuations are included via
    Poisson averaging.

    X-axis: NP concentration [NP mL⁻¹] (logarithmic).
    Panel 1: Lennart fraction = n_ads / (V_SPR × NP_conc + n_ads) — fraction of all NPs
             in the system (solution + surface) that ended up adsorbed.
    Panel 2: Surface coverage = bound_fraction — fraction of SPR chip area occupied by NPs.

    Output files
    ------------
    adsorption_dosing_{key}.dat  columns: NP_conc_per_mL  lennart_fraction  coverage
    adsorption_scan_dosing.png   two panels: lennart_fraction and coverage vs NP_conc
    run_results.npz              cache; reload with --replot to skip recomputation
    """
    mp.dps = 50
    os.makedirs(output_dir, exist_ok=True)

    # ── Parameter resolution: module globals → YAML → CLI overrides ──────────
    n_sampling_points = _dosing_n_pts
    dosing_min_fac    = _dosing_min_fac
    dosing_max_fac    = _dosing_max_fac
    _poly_model       = "Flory-exact"
    sigma_R_curves    = list(_dosing_sigma_R_curves)  # µm⁻², non-empty default
    _curve_labels: list = []

    if config is not None:
        _v = _load_system_vars_yaml(config)
        R_NP                = _v["R_NP"]
        data_polymers       = _v["data_polymers"]
        receptor            = _v["receptor"]
        A_SPR               = _v["A_SPR"]
        NP_conc             = _v["NP_conc_spr"]
        cell_conc           = _v["cell_conc_spr"]
        nonspec_interaction = _v["nonspec_interaction"]
        V_SPR               = _v["V_SPR"]
        n_sampling_points   = _v["dosing_n_pts"]
        dosing_min_fac      = _v["dosing_min_factor"]
        dosing_max_fac      = _v["dosing_max_factor"]
        if len(_v["polymer_models"]) == 1:    # only override when user set exactly one model
            _poly_model     = _v["polymer_models"][0]
        if _v["dosing_sigma_R_curves"]:       # empty list → keep module default
            sigma_R_curves  = _v["dosing_sigma_R_curves"]
        _curve_labels       = list(_v["dosing_sigma_R_labels"])
    else:
        from system_variables_invitro import (
            R_NP, data_polymers, A_SPR, NP_conc, cell_conc,
            nonspec_interaction, V_SPR, receptor,
        )

    if n_pts         is not None: n_sampling_points = n_pts
    if dosing_min    is not None: dosing_min_fac    = dosing_min
    if dosing_max    is not None: dosing_max_fac    = dosing_max
    if polymer_model is not None: _poly_model       = polymer_model
    if sigma_r:                   sigma_R_curves    = list(sigma_r)
    if sigma_r_label:             _curve_labels     = list(sigma_r_label)

    # ── Caching gate ─────────────────────────────────────────────────────────
    cache_path   = os.path.join(str(output_dir), "run_results.npz")
    sigma_R_keys = [f"sigR{v:g}" for v in sigma_R_curves]

    if replot and os.path.exists(cache_path):
        sigma_R_keys, results = _load_langmuir_results(cache_path)
        print(f"[cache] loaded {cache_path} — re-plotting only.")
    else:
        results = _run_dosing_sweep(
            R_NP=R_NP, A_SPR=A_SPR, NP_conc=NP_conc, cell_conc=cell_conc,
            nonspec_interaction=nonspec_interaction, V_SPR=V_SPR,
            data_polymers=data_polymers, receptor_registry={"_single": receptor},
            n_pts=n_sampling_points, dosing_min_fac=dosing_min_fac,
            dosing_max_fac=dosing_max_fac, poly_model=_poly_model,
            sigma_R_curves=sigma_R_curves, sigma_R_keys=sigma_R_keys,
        )
        _save_langmuir_results(cache_path, sigma_R_keys, results)

    # Write .dat files (both normal and --replot paths)
    for key in sigma_R_keys:
        res   = results[key]
        fname = f"adsorption_dosing_{key}.dat"
        with open(os.path.join(str(output_dir), fname), "w") as fh:
            for j in range(len(res["NP_conc_per_mL"])):
                fh.write(
                    f"{res['NP_conc_per_mL'][j]:5.3e} "
                    f"{res['lennart_fraction'][j]:5.3e} "
                    f"{res['coverage'][j]:5.3e}\n"
                )
        print(f"  Written to {os.path.join(str(output_dir), fname)}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    for k_idx, key in enumerate(sigma_R_keys):
        if k_idx < len(_curve_labels):
            lbl = _curve_labels[k_idx]
        else:
            # key = "sigR100" → "100 µm⁻²"; safe for both compute and --replot paths
            lbl = key.removeprefix("sigR") + " µm⁻²"
        res = results[key]
        ax1.plot(res["NP_conc_per_mL"], res["lennart_fraction"], label=lbl)
        ax2.plot(res["NP_conc_per_mL"], res["coverage"],         label=lbl)

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel(r"NP concentration (NP mL$^{-1}$)")
        ax.legend()
    ax1.set_ylabel("Fraction of bound NPs")
    ax2.set_ylabel("Fraction of covered surface")
    plt.tight_layout()
    plot_path = os.path.join(str(output_dir), "adsorption_scan_dosing.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\nPlot saved to {plot_path}")


def _run_dosing_sweep(
    R_NP, A_SPR, NP_conc, cell_conc, nonspec_interaction, V_SPR,
    data_polymers, receptor_registry: dict,
    n_pts: int, dosing_min_fac: float, dosing_max_fac: float,
    poly_model: str, sigma_R_curves: list, sigma_R_keys: list,
) -> dict:
    """Execute the NP dosing sweep and return per-σ_R result arrays.

    Outer loop: NP dosing values (log-spaced between dosing_min_fac × NP_conc and
    dosing_max_fac × NP_conc). Inner loop: σ_R curve values.

    K_bind_vs_NR is computed once (depends only on polymer model, not on σ_R or dosing).
    bound_vs_receptor is computed once per dosing step (depends on NP_conc, not on σ_R).

    Returns {key: {"NP_conc_per_mL": array, "lennart_fraction": array, "coverage": array}}
    for each key in sigma_R_keys.
    """
    dosing_values = np.logspace(
        np.log10(dosing_min_fac), np.log10(dosing_max_fac), n_pts
    )

    # Size the Poisson truncation for the highest σ_R value; floor at 20 for the
    # sparse-receptor regime where calculate_bound_fraction requires at least 20 terms.
    max_sigma_R_nm2 = max(sigma_R_curves) / um2
    max_NR_ave      = int((2 * R_NP)**2 * max_sigma_R_nm2)
    max_n_receptor  = max(max_NR_ave + 4 * (max_NR_ave + 1) + 1, 20)

    system_ref = MultivalentBinding(
        kT=kT, R_NP=R_NP, data_polymers=data_polymers,
        binding_model="exact", polymer_model=poly_model,
        A_cell=A_SPR, NP_conc=NP_conc, cell_conc=cell_conc,
        nonspec_interaction=nonspec_interaction,
    )

    # slow step; depends only on polymer model, not on σ_R or dosing — done once
    print(f"Computing K_bind for NR = 1..{max_n_receptor - 1} (slow step, done once)")
    K_bind_vs_NR = system_ref.calculate_K_bind_vs_receptors(max_n_receptor)
    print("K_bind computation done.")

    acc = {k: {"NP_conc_per_mL": [], "lennart_fraction": [], "coverage": []}
           for k in sigma_R_keys}
    max_num_sites = A_SPR / system_ref.NP_excluded_area

    for dosing_factor in dosing_values:
        NP_conc_i = NP_conc * dosing_factor
        # depends on NP_conc, not on σ_R — reused across all σ_R curves at this dosing step
        bound_vs_receptor = system_ref.calculate_bound_vs_receptors_monodisperse(
            max_n_receptor, depletion=False, verbose=False,
            K_bind_vs_NR=K_bind_vs_NR, NP_conc=NP_conc_i,
        )
        for k_idx, sigma_R_val in enumerate(sigma_R_curves):
            # calculate_bound_fraction reads σ_R from the shared receptor dict(s)
            for rec in receptor_registry.values():
                rec["sigma_R"] = sigma_R_val / um2
            bf = float(system_ref.calculate_bound_fraction(
                fluctuations=True, depletion=False,
                bound_vs_receptor=bound_vs_receptor, max_factor=4,
            ))
            n_ads = bf * max_num_sites
            # adsorbed / total: fraction of all NPs in the system that are surface-bound
            lennart = n_ads / (V_SPR * NP_conc_i + n_ads)
            acc[sigma_R_keys[k_idx]]["NP_conc_per_mL"].append(float(NP_conc_i * mL))
            acc[sigma_R_keys[k_idx]]["lennart_fraction"].append(float(lennart))
            acc[sigma_R_keys[k_idx]]["coverage"].append(float(bf))

    # prevent σ_R residual from leaking into downstream calculations
    for rec in receptor_registry.values():
        rec.pop("sigma_R", None)

    return {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in acc.items()}


def _execute_dosing_run(run_spec: dict):
    """Execute one scan-dosing-combinations run; designed for dispatch to a worker process.

    Receives all required state via run_spec (picklable — no module globals are read).
    Returns (run_name, None) on success or (run_name, error_message) on failure.
    """
    run_binder_ids  = list(run_spec["run_binder_ids"])
    run_output_dir  = run_spec["run_output_dir"]
    run_name        = "+".join(run_binder_ids)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"=== {run_name} ===", flush=True)

    cache_path   = os.path.join(run_output_dir, "run_results.npz")
    sigma_R_keys = [f"sigR{v:g}" for v in run_spec["sigma_R_curves"]]

    if run_spec.get("replot", False) and os.path.exists(cache_path):
        sigma_R_keys, results = _load_langmuir_results(cache_path)
        print(f"  [cache] {run_name}", flush=True)
    else:
        data_polymers, receptor_registry = _build_data_polymers_for_run(
            run_binder_ids,
            run_spec["binders_data"],
            run_spec["nanoparticle_params"],
            run_spec["ligand_ratio"],
        )
        results = _run_dosing_sweep(
            R_NP=run_spec["R_NP"],
            A_SPR=run_spec["A_SPR"],
            NP_conc=run_spec["NP_conc"],
            cell_conc=run_spec["cell_conc"],
            nonspec_interaction=run_spec["nonspec_interaction"],
            V_SPR=run_spec["V_SPR"],
            data_polymers=data_polymers,
            receptor_registry=receptor_registry,
            n_pts=run_spec["n_pts"],
            dosing_min_fac=run_spec["dosing_min_fac"],
            dosing_max_fac=run_spec["dosing_max_fac"],
            poly_model=run_spec["poly_model"],
            sigma_R_curves=run_spec["sigma_R_curves"],
            sigma_R_keys=sigma_R_keys,
        )
        _save_langmuir_results(cache_path, sigma_R_keys, results)

    curve_labels = run_spec.get("curve_labels", [])

    # Write .dat files
    for key in sigma_R_keys:
        res   = results[key]
        fname = f"adsorption_dosing_{key}.dat"
        with open(os.path.join(run_output_dir, fname), "w") as fh:
            for j in range(len(res["NP_conc_per_mL"])):
                fh.write(
                    f"{res['NP_conc_per_mL'][j]:5.3e} "
                    f"{res['lennart_fraction'][j]:5.3e} "
                    f"{res['coverage'][j]:5.3e}\n"
                )

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    for k_idx, key in enumerate(sigma_R_keys):
        lbl = curve_labels[k_idx] if k_idx < len(curve_labels) else key.removeprefix("sigR") + " µm⁻²"
        res = results[key]
        ax1.plot(res["NP_conc_per_mL"], res["lennart_fraction"], label=lbl)
        ax2.plot(res["NP_conc_per_mL"], res["coverage"],         label=lbl)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel(r"NP concentration (NP mL$^{-1}$)")
        ax.legend()
    ax1.set_ylabel("Fraction of bound NPs")
    ax2.set_ylabel("Fraction of covered surface")
    plt.tight_layout()
    plt.savefig(os.path.join(run_output_dir, "adsorption_scan_dosing.png"))
    plt.close()
    print(f"  Done: {run_name}", flush=True)
    return run_name, None


@app.command("scan-dosing-combinations")
def scan_dosing_combinations_cmd(
    csv_path: Annotated[Path, typer.Option(
        "--csv", "-f",
        help="CSV with binder×receptor KD matrix [nM]. "
             "Rows = receptors, columns = binder IDs. Empty cell = no binding.",
    )],
    output_dir: Path = typer.Option(
        Path("."), "-o", "--output-dir",
        help="Root output directory. Each run writes to its own subdirectory.",
    ),
    config: Annotated[Optional[Path], typer.Option(
        "--config", "-c",
        help="YAML parameter file (NP geometry, PEG, SPR geometry, dosing). "
             "If omitted, system_variables_invitro defaults are used.",
    )] = None,
    n_pts: Annotated[Optional[int], typer.Option(
        "--n-pts",
        help="Grid points for NP dosing axis (overrides YAML/module default).",
    )] = None,
    dosing_min: Annotated[Optional[float], typer.Option(
        "--dosing-min",
        help="Minimum NP dosing multiplier (overrides YAML/module default).",
    )] = None,
    dosing_max: Annotated[Optional[float], typer.Option(
        "--dosing-max",
        help="Maximum NP dosing multiplier (overrides YAML/module default).",
    )] = None,
    polymer_model: Annotated[Optional[str], typer.Option(
        "--polymer-model",
        help="Polymer model name (overrides YAML/module default).",
    )] = None,
    sigma_r: Annotated[Optional[List[float]], typer.Option(
        "--sigma-r",
        help="σ_R curve value [µm⁻²], repeatable. Overrides YAML/module default.",
    )] = None,
    sigma_r_label: Annotated[Optional[List[str]], typer.Option(
        "--sigma-r-label",
        help="Label for σ_R curve (same order as --sigma-r), repeatable.",
    )] = None,
    ligand_ratio: Annotated[float, typer.Option(
        "--ligand-ratio",
        help="In a 2-binder run, fraction of sigma_L given to the first binder "
             "(second binder gets 1 - ratio). Default 0.5.",
    )] = 0.5,
    n_workers: Annotated[int, typer.Option(
        "--n-workers",
        help="Number of parallel worker processes. Default 1 (serial). "
             "Pass -1 to use all available CPU cores.",
    )] = 1,
    job_range: Annotated[Optional[str], typer.Option(
        "--job-range",
        help='1-indexed inclusive range "START:END" of runs to execute. '
             'Allows splitting a batch across HPC nodes. '
             'Example: --job-range 1:4 runs the first 4 runs.',
    )] = None,
    replot: Annotated[bool, typer.Option(
        "--replot",
        help="If run_results.npz already exists for a run, load it and re-plot "
             "without recomputing. Default: always recompute.",
    )] = False,
):
    """NP dosing sweeps for every single binder and pair of binders in a CSV — SPR Langmuir.

    Reads a binder×receptor KD matrix from a CSV file and runs scan-dosing-langmuir-style
    sweeps for every single binder and every unordered pair of binders. Results are written
    to one subdirectory per run under --output-dir.

    Physical context: identical to scan-dosing-langmuir (SPR geometry, depletion=False,
    Poisson receptor fluctuations). For pairs with multiple receptor types, all receptor
    densities are set to the same σ_R value at each curve point (isotropic approximation).

    Output files per run
    --------------------
    adsorption_dosing_{key}.dat  columns: NP_conc_per_mL  lennart_fraction  coverage
    adsorption_scan_dosing.png   two panels: lennart_fraction and coverage vs NP_conc
    run_results.npz              cache; reload with --replot to skip recomputation
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Parameter resolution: module globals → YAML → CLI overrides ──────────
    n_sampling_points = _dosing_n_pts
    dosing_min_fac    = _dosing_min_fac
    dosing_max_fac    = _dosing_max_fac
    _poly_model       = "Flory-exact"
    sigma_R_curves    = list(_dosing_sigma_R_curves)
    _curve_labels: list = []

    if config is not None:
        _v = _load_system_vars_yaml(config)
        R_NP                = _v["R_NP"]
        A_SPR               = _v["A_SPR"]
        NP_conc             = _v["NP_conc_spr"]
        cell_conc           = _v["cell_conc_spr"]
        nonspec_interaction = _v["nonspec_interaction"]
        V_SPR               = _v["V_SPR"]
        n_sampling_points   = _v["dosing_n_pts"]
        dosing_min_fac      = _v["dosing_min_factor"]
        dosing_max_fac      = _v["dosing_max_factor"]
        if len(_v["polymer_models"]) == 1:
            _poly_model     = _v["polymer_models"][0]
        if _v["dosing_sigma_R_curves"]:
            sigma_R_curves  = _v["dosing_sigma_R_curves"]
        _curve_labels       = list(_v["dosing_sigma_R_labels"])
        nanoparticle_params = _v  # has sigma_L, amono, akuhn, NmonoL, NmonoS, binder_size
    else:
        from system_variables_invitro import (
            R_NP, sigma_L, sigma_P2K, amono, akuhn,
            NmonoLigands, NmonoShort, binder_linear_size,
            A_SPR, NP_conc, cell_conc, nonspec_interaction, V_SPR,
        )
        nanoparticle_params = {
            "R_NP": R_NP, "sigma_L": sigma_L, "sigma_P2K": sigma_P2K,
            "amono": amono, "akuhn": akuhn, "NmonoL": NmonoLigands,
            "NmonoS": NmonoShort, "binder_size": binder_linear_size,
        }

    if n_pts         is not None: n_sampling_points = n_pts
    if dosing_min    is not None: dosing_min_fac    = dosing_min
    if dosing_max    is not None: dosing_max_fac    = dosing_max
    if polymer_model is not None: _poly_model       = polymer_model
    if sigma_r:                   sigma_R_curves    = list(sigma_r)
    if sigma_r_label:             _curve_labels     = list(sigma_r_label)

    # ── Build run list ────────────────────────────────────────────────────────
    binders_data = _read_binders_csv(csv_path)
    binder_list  = sorted(binders_data.keys())
    single_runs  = [(bid,)          for bid in binder_list]
    pair_runs    = list(itertools.combinations(binder_list, 2))
    all_runs     = single_runs + pair_runs
    print(f"Loaded {len(binders_data)} binders → {len(all_runs)} runs "
          f"({len(single_runs)} single, {len(pair_runs)} pairs)")

    if job_range is not None:
        try:
            start_str, end_str = job_range.split(":")
            start_idx = int(start_str) - 1
            end_idx   = int(end_str)
        except ValueError:
            raise typer.BadParameter(
                f"Invalid --job-range '{job_range}'. Expected format: START:END (e.g. 1:4).",
                param_hint="--job-range",
            )
        all_runs = all_runs[start_idx:end_idx]
        print(f"  job-range {job_range}: executing {len(all_runs)} runs")

    if n_workers == -1:
        import multiprocessing
        n_workers = multiprocessing.cpu_count()

    # Build run specs
    run_specs = []
    for binder_ids in all_runs:
        run_name       = "+".join(binder_ids)
        run_output_dir = os.path.join(str(output_dir), run_name)
        run_specs.append({
            "R_NP": R_NP, "A_SPR": A_SPR, "NP_conc": NP_conc,
            "cell_conc": cell_conc, "nonspec_interaction": nonspec_interaction,
            "V_SPR": V_SPR,
            "n_pts": n_sampling_points,
            "dosing_min_fac": dosing_min_fac, "dosing_max_fac": dosing_max_fac,
            "poly_model": _poly_model,
            "sigma_R_curves": sigma_R_curves,
            "curve_labels": _curve_labels,
            "run_binder_ids": list(binder_ids),
            "binders_data": binders_data,
            "nanoparticle_params": nanoparticle_params,
            "ligand_ratio": ligand_ratio,
            "run_output_dir": run_output_dir,
            "replot": replot,
        })

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if n_workers == 1:
        for spec in run_specs:
            run_name, err = _execute_dosing_run(spec)
            if err:
                typer.echo(err)
            else:
                typer.echo(f"Completed: {run_name}", err=False)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_execute_dosing_run, spec): spec for spec in run_specs}
            for future in as_completed(futures):
                run_name, err = future.result()
                if err:
                    typer.echo(err)
                else:
                    typer.echo(f"Completed: {run_name}", err=False)

    print("\nAll runs complete.")


@app.command("generate-template")
def generate_template(
    output: Path = typer.Option(
        Path("system_params.yaml"), "--output", "-o",
        help="Path for the output YAML template file.",
    ),
    context: Annotated[Optional[str], typer.Option(
        "--context",
        help="Parameter context: 'invivo' or 'invitro'. "
             "invivo: scan-npdosing / scan-combinations fields only. "
             "invitro: SPR / dosing sweep fields only. "
             "Omit for a combined template with all fields.",
    )] = None,
):
    """Write a commented YAML parameter template to a file.

    The generated file contains all recognised fields with their default values
    and inline comments explaining each parameter's physical meaning and units.
    Edit the file and pass it to any command via --config to override the built-in
    Python system_variables defaults.

    Example
    -------
      nanoads generate-template --context invitro --output spr_params.yaml
      # edit spr_params.yaml …
      nanoads scan-dosing-langmuir --config spr_params.yaml --output-dir results/
    """
    if context == "invivo":
        template = _YAML_TEMPLATE_INVIVO
    elif context == "invitro":
        template = _YAML_TEMPLATE_INVITRO
    elif context is None:
        template = _YAML_TEMPLATE
    else:
        raise typer.BadParameter(
            f"Unknown context '{context}'. Use 'invivo' or 'invitro'.",
            param_hint="--context",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write(template)
    print(f"Template written to {output}")
    print(f"Edit the file and run: nanoads <command> --config {output}")


if __name__ == "__main__":
    app()
