from units import *
import numpy as np
from adsorption import Nmonomers

# Define parameters describint the target: T cells in the spleen
# This info should be checked, for now it is just a search from chatGPT
N_lympho = 7.5e7 # Average number of lymphocytes in mouse spleen
N_T_cells = 0.25 * N_lympho # Number of T cells in mouse spleen
V_spleen = 100 * mm3 # Volume of mouse spleen
cell_conc = N_T_cells / V_spleen # T cell concentration in the spleen
A_cell = 100 * um2 # Cell area 

#****Dosing particles intravenously to animal******
# These are parameters controlling the dosing of the nanoparticles in a typical experiments
# Provided by Lennart Lindfors
Npdosing=8e12 / mL # Number of particles in dosing solution per mL [mL-1]
Vdosing=5*0.02 * mL # Dosing volume for each animal [mL] (e.g. 5 mL/kg and animal weight 0.02 kg)
fTzone=0.1 # Fraction of dosed particles that ends up in T zone of animal spleen
VTzone=0.5*0.084 * mL # Volume of spleen Tzone in animal [mL] (Assuming 50% of mouse spleen of volume 0.084 mL)
NP_conc=Npdosing*Vdosing*fTzone/VTzone # Number of particles per mL in Tzone of mouse spleen [mL-1]


######################################################
# Here we define the DESIGN OF MULTIVALENT NANOPARTICLE
######################################################

R_NP = 35 * nm # Nanoparticle radius in units of length
N_ligands = 80 # Number of ligands on the nanoparticle
sigma_L = N_ligands / ( 4.0 * np.pi * R_NP**2 ) # surface density of ligands
sigma_P2K = sigma_L * 11.4  # surface density of inert PEG2K chains (11.4× the ligand density)

# We assume the presence of short (inert) PEG chains + additional ligands on the nanoparticles
amono = 0.28 * nm # Monomer size in PEG chain
akuhn = 0.76 * nm  # Kuhn segment length (relevant for Flory-exact and WLC models)
NmonoLigands = Nmonomers( 3400 * g ) # Number of monomers in the PEG to which ligands are attached
NmonoShort = Nmonomers( 2000 * g ) # Number of monomers in th short, inert PEG chains

# Define the binding constant for ligand-receptor binding in solution
KD = 10000.0 * nM # Dissociation constant in solution between ligand-receptor
K_bind_0 = KD**(-1) # Binding constant in solution between ligand-receptor

nonspec_interaction = 0.0 # Strength of nonspecific interaction between whole nanoparticle and surface, in units of kT
binder_linear_size = 3.5 * nm # Linear size of the binder, in this case, an antibody
# receptor dict: identifies a receptor type on the target cell surface.
#   "name"    — string identifier used in sweep output, plot axes, and codependent_receptors.
#   "sigma_R" — surface density [nm⁻²]. If omitted here, it is set by the sweep driver at
#               every grid point. If set here (e.g. for a one-off calculation), it is used
#               directly.
#
# Object identity matters: two ligand entries pointing to the SAME dict object are treated
# as binding the SAME receptor type (contributions additive, → 1 sweep axis).
# Two ligand entries pointing to DIFFERENT dict objects (even with the same name) are
# treated as DISTINCT receptor types (→ triggers an error; use unique names).
receptor = {"name": "default"}

###############################################################################
# data_polymers: dict of polymer/ligand entries on the nanoparticle.
#
# Required keys in every entry:
#   "N"      — number of monomers in the chain (use Nmonomers(MW * g))
#   "a"      — monomer size [nm]
#   "sigma"  — grafting density [nm⁻²]
#   "akuhn"  — Kuhn segment length [nm]
#   "name"   — string label (used in output and plots)
#
# Binding entries additionally require:
#   "K_bind_0"           — solution binding constant [nm³] = 1/KD
#   "receptor"           — receptor dict (see above) OR list of receptor dicts
#   "binder_linear_size" — (optional) linear size of the binding domain [nm]
#
# Inert entries (no "K_bind_0", no "receptor") contribute steric repulsion only.
#
# Object identity of receptor dicts determines the sweep topology:
#   same dict object  → same receptor type (contributions additive) → 1 axis
#   distinct objects  → distinct receptor types → one sweep axis each
#
# SETUP A — one ligand, one receptor → 1D sweep:
#   receptor = {"name": "CD44"}
#   "ligands": {"K_bind_0": K_bind_0, "receptor": receptor, ...}
#
# SETUP B — two ligands, SAME receptor (current default) → 1D sweep:
#   "ligands":  {"K_bind_0": KD_A**(-1), "receptor": receptor, ...}
#   "ligands2": {"K_bind_0": KD_B**(-1), "receptor": receptor}   # same object
#
# SETUP C — two ligands, DIFFERENT receptors → 2D sweep:
#   rec_A, rec_B = {"name": "CD44"}, {"name": "CD8"}   # distinct objects
#   "ligands":  {"K_bind_0": KD_A**(-1), "receptor": rec_A, ...}
#   "ligands2": {"K_bind_0": KD_B**(-1), "receptor": rec_B, ...}
#
# SETUP D — three receptors, one codependent → 2D sweep:
#   rec_A, rec_B, rec_C = {"name":"CD44"}, {"name":"CD8"}, {"name":"CD19"}
#   # In scan_both_polymer_models.py set:
#   #   codependent_receptors = {"CD19": ("CD44", 0.5)}
#   # → sigma_CD19 = 0.5 × sigma_CD44 at every grid point; 2 primary axes.
#
# SETUP E — list-form receptor (one ligand, multiple targets) → axes per target:
#   "lig": {"receptor": [{"name":"CD44","K_bind_0":KD_A**(-1)},
#                         {"name":"CD8", "K_bind_0":KD_B**(-1)}], ...}
###############################################################################
data_polymers = {}
data_polymers['short'] = {"N": NmonoShort, "a": amono, "sigma": sigma_P2K, "name" : "PEG2K", 'akuhn' : akuhn }
data_polymers['ligands'] = {"N": NmonoLigands, "a": amono, "sigma": sigma_L, "name":"ligands", 'akuhn' : akuhn, "K_bind_0": K_bind_0, "receptor": receptor, "binder_linear_size": binder_linear_size }
data_polymers['ligands2'] = {"N": NmonoLigands, "a": amono, "sigma": sigma_L, "name":"ligands2", 'akuhn' : akuhn, "K_bind_0": K_bind_0, "receptor": receptor, "binder_linear_size": binder_linear_size }