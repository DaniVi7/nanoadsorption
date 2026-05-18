from adsorption import *
from units import *
from system_variables_invivo_multi import *
from mpmath import mp
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection

mp.dps = 50
assert mp.dps >= 30

# ── Receptor density markers ─────────────────────────────────────────────────
# Values in µm⁻². Empty list → disabled.
# 1-axis case  : vertical dashed lines on the sigma_R axis.
# 2-axis case  : red lines on axis 1, blue on axis 2, on the 3D floor and as
#                axvline/axhline on the 2D contour map.
target_sigma_R = []           # e.g. [100.0, 500.0]
# One label per entry in target_sigma_R. If non-empty, must have the same length.
# If empty, only the first line gets an auto-generated "σ_R = <value> µm⁻²" label.
target_sigma_R_labels = []    # e.g. ["healthy tissue", "tumour"]

# Maps secondary receptor name → (primary receptor name, ratio).
# sigma_secondary = ratio × sigma_primary at every sweep point.
# A receptor not listed here is a primary (defines its own independent sweep axis).
# At most 2 primaries allowed (more would require 3D plots → error at runtime).
codependent_receptors = {}    # e.g. {"rec_C": ("rec_A", 0.5), "rec_D": ("rec_B", 2.0)}

# ── Polymer model comparison ──────────────────────────────────────────────────
# If True, all entries in polymer_models are run and overlaid on the same plot.
# If False, only polymer_models[0] is run.
compare_polymer_models = True
polymer_models = ["gaussian", "Flory-exact"]   # "Flory-approx" also valid

# ── Sweep resolution ──────────────────────────────────────────────────────────
n_pts_1D = 50     # sigma_R points in the 1-axis sweep
n_pts_2D = 20     # points per axis in the 2-axis sweep (total grid = n_pts_2D²)
                  # Keep ≤ 20 for a first run: each outer call rebuilds Poisson weights
                  # over ~max_NR² grid points using mpmath.
sigma_R_min = 1.0    / um2   # nm⁻²  (= 1 µm⁻²)
sigma_R_max = 2000.0 / um2   # nm⁻²  (= 2000 µm⁻²)


# ── Reference-line label helper ──────────────────────────────────────────────
def _ref_label(k, v, suffix=""):
    """Legend label for the k-th target_sigma_R reference line.

    With user labels: every line gets its label (+ suffix).
    Without user labels: k==0 gets an auto-generated string; others return "_"
    so matplotlib hides them from the legend.
    suffix appends e.g. a receptor name in the 2-axis case.
    """
    if target_sigma_R_labels:
        return target_sigma_R_labels[k] + suffix
    if k == 0:
        return (r"$\sigma_R = " + f"{v:.0f}" + r"\,\mu\mathrm{m}^{-2}$") + suffix
    return "_"


# ── Multi-receptor ligand expansion ──────────────────────────────────────────
def _expand_multireceptor_ligands(dp):
    """Expand ligands whose "receptor" is a list into one virtual ligand per receptor target.

    Each virtual entry is named "{original_key}_{rec_name}" and carries the
    per-receptor K_bind_0 at the top level; the receptor dict is left without it.
    Dict-form ligands pass through unchanged. The input dp is not modified.
    """
    expanded = {}
    for key, poly in dp.items():
        rec = poly.get("receptor")
        if not isinstance(rec, list):
            expanded[key] = poly
            continue
        base = {k: v for k, v in poly.items() if k != "receptor"}
        for rec_entry in rec:
            rec_entry = dict(rec_entry)
            k_bind    = rec_entry.pop("K_bind_0")
            virt_key  = f"{key}_{rec_entry['name']}"
            virt_poly = dict(base, name=virt_key, K_bind_0=k_bind, receptor=rec_entry)
            expanded[virt_key] = virt_poly
    return expanded


# ── Receptor detection ────────────────────────────────────────────────────────
def _detect_receptors(dp):
    """Return {name: rec_dict} for each distinct receptor dict in dp.

    Handles both dict-form (K_bind_0 at ligand level) and list-form receptors
    (K_bind_0 inside each receptor entry). Uses object identity (id) to
    distinguish shared vs distinct receptor dicts.
    """
    seen_id = {}
    named   = {}
    for poly in dp.values():
        rec_field = poly.get("receptor")
        if rec_field is None:
            continue
        if isinstance(rec_field, list):
            for rec in rec_field:
                if id(rec) not in seen_id:
                    seen_id[id(rec)] = rec
                    named[rec["name"]] = rec
        elif "K_bind_0" in poly:
            if id(rec_field) not in seen_id:
                seen_id[id(rec_field)] = rec_field
                named[rec_field["name"]] = rec_field
    return named


# ── System builder ────────────────────────────────────────────────────────────
def _make_system(polymer_model_name, dp_src=None):
    """Deep-copy dp_src, expand multi-receptor ligands, build MultivalentBinding,
    and return (system, rec_refs) where rec_refs maps receptor name → receptor dict
    inside the copy (live references safe to update during sweeps).
    deepcopy preserves shared-object identity within the copy, so ligands that
    originally pointed to the same receptor dict still do after the copy.
    """
    dp_raw = copy.deepcopy(dp_src if dp_src is not None else data_polymers)
    dp = _expand_multireceptor_ligands(dp_raw)

    rec_refs = {}
    for poly in dp.values():
        if "receptor" in poly:
            rec = poly["receptor"]
            if rec["name"] not in rec_refs:
                rec_refs[rec["name"]] = rec

    system = MultivalentBinding(
        kT=kT, R_NP=R_NP,
        data_polymers=dp,
        binding_model="exact",
        polymer_model=polymer_model_name,
        A_cell=A_cell,
        NP_conc=NP_conc,
        cell_conc=cell_conc,
        nonspec_interaction=nonspec_interaction,
    )
    return system, rec_refs


# ── Codependent receptor resolver ────────────────────────────────────────────
def _resolve_sigma_R(primary_sigma_R_map, rec_refs):
    """Set sigma_R for all receptors in rec_refs.

    Primary receptors are set from primary_sigma_R_map.
    Secondary receptors are set via their ratio to the primary (from module-level
    codependent_receptors). Returns the complete {name: sigma_R} map.
    """
    all_sr = dict(primary_sigma_R_map)
    for sec_name, (pri_name, ratio) in codependent_receptors.items():
        all_sr[sec_name] = ratio * primary_sigma_R_map[pri_name]
    for name, sr in all_sr.items():
        rec_refs[name]["sigma_R"] = sr
    return all_sr


# ── Axis label builder ────────────────────────────────────────────────────────
def _axis_label(primary_name, log=False):
    """Return a matplotlib axis label string for the given primary receptor.

    Appends codependent receptor ratios as a second line when present.
    log=True uses log10 notation (for 3D surface plots).
    """
    if log:
        base = (rf"$\log_{{10}}[\sigma_{{R}}\ /\ \mu\mathrm{{m}}^{{-2}}]$"
                + f"\n({primary_name})")
    else:
        base = rf"$\sigma_{{R}}$ ({primary_name}) ($\mu$m$^{{-2}}$)"
    deps = [(sec, ratio) for sec, (pri, ratio) in codependent_receptors.items()
            if pri == primary_name]
    if not deps:
        return base
    dep_str = ",  ".join(f"{s} = {r:.2g}×{primary_name}" for s, r in deps)
    return base + f"\n[{dep_str}]"


# ── Case A: 1-axis sweep ──────────────────────────────────────────────────────
def sweep_1axis(polymer_model_name, primary_name):
    """Sweep sigma_R along one independent receptor axis.

    Codependent receptors are updated at every point via _resolve_sigma_R.
    Returns {"sigma_R_um2", "bound_fraction", "n_ads", "primary_name"}.
    Uses module-level sigma_R_min, sigma_R_max, n_pts_1D.
    """
    system, rec_refs = _make_system(polymer_model_name)
    sigma_R_arr = np.logspace(np.log10(sigma_R_min), np.log10(sigma_R_max), n_pts_1D)

    # Set to maximum before precomputing K_bind table so the table is sized for
    # the worst-case NR count across the whole sweep (and all codependents).
    _resolve_sigma_R({primary_name: sigma_R_max}, rec_refs)
    max_NR_ave     = int(system.NP_excluded_area * sigma_R_max)
    max_N_receptor = max_NR_ave + 4 * (max_NR_ave + 1) + 1

    print(f"  [{polymer_model_name}] Precomputing K_bind table (max NR = {max_N_receptor})...")
    K_bind_data = system.calculate_K_bind_vs_receptors(max_N_receptor)
    is_multi    = isinstance(K_bind_data, tuple)
    if is_multi:
        K_bind_flat, grid_shape, _, rec_names_ordered = K_bind_data

    M_conc = (A_cell / system.NP_excluded_area) * cell_conc

    bf_arr   = np.zeros(n_pts_1D)
    nads_arr = np.zeros(n_pts_1D)

    for i, sigma_R in enumerate(sigma_R_arr):
        _resolve_sigma_R({primary_name: sigma_R}, rec_refs)
        if is_multi:
            NR_aves    = [float(system.NP_excluded_area * rec_refs[n]["sigma_R"])
                          for n in rec_names_ordered]
            K_bind_arg = (K_bind_flat, grid_shape, NR_aves, rec_names_ordered)
        else:
            K_bind_arg = K_bind_data
        bf = system.calculate_bound_fraction(
            fluctuations=True, depletion=True,
            K_bind_vs_receptors=K_bind_arg, rho_m=M_conc,
            **({"max_n_receptor": max_N_receptor} if not is_multi else {}))
        bf_arr[i]   = float(bf)
        nads_arr[i] = float(bf) * NP_conc
        if (i + 1) % 10 == 0:
            print(f"  [{polymer_model_name}] {i + 1}/{n_pts_1D} sigma_R points done")

    return {
        "sigma_R_um2":    sigma_R_arr / (1 / um2),
        "bound_fraction": bf_arr,
        "n_ads":          nads_arr,
        "primary_name":   primary_name,
    }


# ── Case B: 2-axis grid sweep ─────────────────────────────────────────────────
def sweep_2axes(polymer_model_name, primary_name_1, primary_name_2):
    """Sweep (sigma_R1, sigma_R2) on a 2D grid for two independent receptor axes.

    Codependent receptors are updated at every grid point via _resolve_sigma_R.
    K_bind_flat is precomputed once at sigma_R_max (depends only on integer
    receptor counts). Only NR_aves are updated per grid point.
    Returns {"sigma_R1_um2", "sigma_R2_um2", "bound_fraction", "n_ads", "rec_names"}.
    bound_fraction[i, j] = f(sigma_R1[i], sigma_R2[j]).
    """
    system, rec_refs = _make_system(polymer_model_name)
    sigma_R_arr = np.logspace(np.log10(sigma_R_min), np.log10(sigma_R_max), n_pts_2D)

    _resolve_sigma_R({primary_name_1: sigma_R_max, primary_name_2: sigma_R_max}, rec_refs)
    max_NR_ave     = int(system.NP_excluded_area * sigma_R_max)
    max_N_receptor = max_NR_ave + 4 * (max_NR_ave + 1) + 1

    print(f"  [{polymer_model_name}] Precomputing 2D K_bind grid "
          f"(max NR = {max_N_receptor} per axis, {max_N_receptor**2} grid points)...")
    K_bind_flat, grid_shape, _, rec_names_ordered = \
        system.calculate_K_bind_vs_receptors(max_N_receptor)

    M_conc = (A_cell / system.NP_excluded_area) * cell_conc

    n = n_pts_2D
    bf_grid   = np.zeros((n, n))
    nads_grid = np.zeros((n, n))

    for i, s1 in enumerate(sigma_R_arr):
        for j, s2 in enumerate(sigma_R_arr):
            _resolve_sigma_R({primary_name_1: s1, primary_name_2: s2}, rec_refs)
            NR_aves    = [float(system.NP_excluded_area * rec_refs[nm]["sigma_R"])
                          for nm in rec_names_ordered]
            K_bind_arg = (K_bind_flat, grid_shape, NR_aves, rec_names_ordered)
            bf = system.calculate_bound_fraction(
                fluctuations=True, depletion=True,
                K_bind_vs_receptors=K_bind_arg, rho_m=M_conc)
            bf_grid[i, j]   = float(bf)
            nads_grid[i, j] = float(bf) * NP_conc

        if (i + 1) % 5 == 0:
            print(f"  [{polymer_model_name}] {(i + 1) * n}/{n * n} grid points done")

    sigma_R_um2 = sigma_R_arr / (1 / um2)
    return {
        "sigma_R1_um2":   sigma_R_um2,
        "sigma_R2_um2":   sigma_R_um2,
        "bound_fraction": bf_grid,
        "n_ads":          nads_grid,
        "rec_names":      [primary_name_1, primary_name_2],
    }


# ── Case B plotting helper ────────────────────────────────────────────────────
def _plot_case_b(qty_key, qty_label, fname_3d, fname_2d, results, n_models):
    """Generate 3D surface + 2D contour figures for one quantity (bound_fraction or n_ads)."""
    # ── 3D surface ──
    fig3d = plt.figure(figsize=(8 * n_models, 7))
    for col, (model_name, res) in enumerate(results.items()):
        ax = fig3d.add_subplot(1, n_models, col + 1, projection="3d")

        log_s1 = np.log10(res["sigma_R1_um2"])
        log_s2 = np.log10(res["sigma_R2_um2"])
        LogS1, LogS2 = np.meshgrid(log_s1, log_s2, indexing="ij")

        surf = ax.plot_surface(LogS1, LogS2, res[qty_key],
                               cmap="viridis", alpha=0.85,
                               linewidth=0, antialiased=True)
        fig3d.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label=qty_label)

        log_smin, log_smax = log_s1[0], log_s1[-1]
        for k, v in enumerate(target_sigma_R):
            log_v = np.log10(v)
            if log_smin <= log_v <= log_smax:
                lbl1 = _ref_label(k, v, suffix=f" ({res['rec_names'][0]})")
                lbl2 = _ref_label(k, v, suffix=f" ({res['rec_names'][1]})")
                ax.plot([log_v, log_v], [log_smin, log_smax], [0.0, 0.0],
                        color="red",  linestyle="--", linewidth=1.2, label=lbl1)
                ax.plot([log_smin, log_smax], [log_v, log_v], [0.0, 0.0],
                        color="blue", linestyle="--", linewidth=1.2, label=lbl2)

        ax.set_xlabel(_axis_label(res["rec_names"][0], log=True), labelpad=10)
        ax.set_ylabel(_axis_label(res["rec_names"][1], log=True), labelpad=10)
        ax.set_zlabel(qty_label)
        ax.set_title(model_name)
        if target_sigma_R:
            ax.legend(fontsize=7, loc="upper left")

    fig3d.tight_layout()
    fig3d.savefig(fname_3d, dpi=150)
    plt.close(fig3d)
    print(f"Saved {fname_3d}")

    # ── 2D filled-contour projection ──
    fig2d, axes2d = plt.subplots(1, n_models, figsize=(7 * n_models, 6), squeeze=False)
    for col, (model_name, res) in enumerate(results.items()):
        ax = axes2d.flat[col]

        # contourf(X, Y, Z): Z.shape must be (len(Y), len(X)).
        # res[qty_key][i,j] indexed as (sigma_R1_i, sigma_R2_j) → transpose.
        cf = ax.contourf(res["sigma_R1_um2"], res["sigma_R2_um2"],
                         res[qty_key].T, levels=20, cmap="viridis")
        fig2d.colorbar(cf, ax=ax, label=qty_label)

        for k, v in enumerate(target_sigma_R):
            lbl1 = _ref_label(k, v, suffix=f" ({res['rec_names'][0]})")
            lbl2 = _ref_label(k, v, suffix=f" ({res['rec_names'][1]})")
            ax.axvline(x=v, color="red",  linestyle="--", linewidth=1.0, label=lbl1)
            ax.axhline(y=v, color="blue", linestyle="--", linewidth=1.0, label=lbl2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(_axis_label(res["rec_names"][0]))
        ax.set_ylabel(_axis_label(res["rec_names"][1]))
        ax.set_title(model_name)
        if target_sigma_R:
            ax.legend(fontsize=7)

    fig2d.tight_layout()
    fig2d.savefig(fname_2d, dpi=150)
    plt.close(fig2d)
    print(f"Saved {fname_2d}")


# ── Main execution ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    receptor_map = _detect_receptors(data_polymers)

    if target_sigma_R_labels and len(target_sigma_R_labels) != len(target_sigma_R):
        raise ValueError(
            f"target_sigma_R_labels has {len(target_sigma_R_labels)} entries "
            f"but target_sigma_R has {len(target_sigma_R)}. Lengths must match."
        )

    # Validate codependent_receptors: names must exist, no cycles.
    for sec, (pri, ratio) in codependent_receptors.items():
        if sec not in receptor_map:
            raise ValueError(f"codependent_receptors key '{sec}' not in detected receptors")
        if pri not in receptor_map:
            raise ValueError(f"codependent_receptors primary '{pri}' not in detected receptors")
        if pri in codependent_receptors:
            raise ValueError(
                f"'{pri}' is listed as both primary and secondary — cycles not allowed")

    secondary_names = set(codependent_receptors.keys())
    primary_names   = [n for n in receptor_map if n not in secondary_names]
    n_independent   = len(primary_names)

    if n_independent > 2:
        raise ValueError(
            f"{n_independent} independent receptor axes detected "
            f"(primaries: {primary_names}). At most 2 are supported. "
            "Use codependent_receptors to link excess receptors."
        )

    print(f"Detected {len(receptor_map)} receptor type(s): {list(receptor_map.keys())}")
    if codependent_receptors:
        print(f"Codependent receptors: {codependent_receptors}")
    print(f"Independent axes ({n_independent}): {primary_names}")

    models_to_run = polymer_models if compare_polymer_models else [polymer_models[0]]

    results = {}
    for model_name in models_to_run:
        print(f"\n=== Running model: {model_name} ===")
        if n_independent == 1:
            results[model_name] = sweep_1axis(model_name, primary_names[0])
        else:
            results[model_name] = sweep_2axes(model_name, primary_names[0], primary_names[1])

    print("\nAll sweeps done.")

    # ── Write .dat files ──────────────────────────────────────────────────────
    for model_name, res in results.items():
        safe = model_name.replace("-", "_")
        if n_independent == 1:
            fname = f"adsorption_{safe}.dat"
            with open(fname, "w") as f:
                f.write("# sigma_R(um^-2)  bound_fraction  n_ads(nm^-3)\n")
                for j in range(len(res["sigma_R_um2"])):
                    f.write(f"{res['sigma_R_um2'][j]:9.4e}  "
                            f"{res['bound_fraction'][j]:9.4e}  "
                            f"{res['n_ads'][j]:9.4e}\n")
        else:
            fname = f"adsorption_2rec_{safe}.dat"
            nr1 = len(res["sigma_R1_um2"])
            nr2 = len(res["sigma_R2_um2"])
            with open(fname, "w") as f:
                f.write(f"# {res['rec_names'][0]}(um^-2)  {res['rec_names'][1]}(um^-2)  "
                        f"bound_fraction  n_ads(nm^-3)\n")
                for i in range(nr1):
                    for j in range(nr2):
                        f.write(f"{res['sigma_R1_um2'][i]:9.4e}  "
                                f"{res['sigma_R2_um2'][j]:9.4e}  "
                                f"{res['bound_fraction'][i, j]:9.4e}  "
                                f"{res['n_ads'][i, j]:9.4e}\n")
        print(f"Written {fname}")

    # ── Plotting ──────────────────────────────────────────────────────────────
    n_models = len(results)

    if n_independent == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        for model_name, res in results.items():
            ax1.plot(res["sigma_R_um2"], res["bound_fraction"],
                     linestyle="solid", label=model_name)
            ax2.plot(res["sigma_R_um2"], res["n_ads"],
                     linestyle="solid", label=model_name)

        for k, v in enumerate(target_sigma_R):
            lbl = _ref_label(k, v)
            ax1.axvline(x=v, color="gray", linestyle="dashed", linewidth=0.9, label=lbl)
            ax2.axvline(x=v, color="gray", linestyle="dashed", linewidth=0.9, label=lbl)

        ax1.set_xscale("log")
        ax1.set_xlabel(_axis_label(primary_names[0]))
        ax1.set_ylabel("Adsorbed fraction")
        ax1.legend(fontsize="small")

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel(_axis_label(primary_names[0]))
        ax2.set_ylabel(r"Adsorbed NP concentration (nm$^{-3}$)")
        ax2.legend(fontsize="small")

        plt.tight_layout()
        plt.savefig("adsorption_polymer_models.png", dpi=150)
        plt.close()
        print("Saved adsorption_polymer_models.png")

    else:
        _plot_case_b(
            "bound_fraction", "Adsorbed fraction",
            "adsorption_polymer_models_3D.png",
            "adsorption_polymer_models_2D_proj.png",
            results, n_models,
        )
        _plot_case_b(
            "n_ads", r"Adsorbed NP concentration (nm$^{-3}$)",
            "adsorption_polymer_models_3D_nads.png",
            "adsorption_polymer_models_2D_proj_nads.png",
            results, n_models,
        )

    print("\nDone.")
