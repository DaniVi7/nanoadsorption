import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import typer
from pathlib import Path
from mpmath import mp
from units import *                           # kT, nm, um2, mL, nM, g
from adsorption import MultivalentBinding
import scan_both_polymer_models as _sbpm      # safe: computation guarded in __main__

mp.dps = 50
assert mp.dps >= 30

# ── Per-command patchable parameters (read at call time) ──────────────────────
_npdosing_n_pts     = 200
_npdosing_sigma_min = 1.0 / um2
_npdosing_sigma_max = 2000 / um2
_npdosing_factors   = [10**k for k in range(-5, 2)]

_langmuir_n_pts     = 200
_langmuir_sigma_min = 1.0 / um2
_langmuir_sigma_max = 2000 / um2
_langmuir_factors   = [10**k for k in range(-5, 2)]

_multi_n_pts        = 200
_multi_sigma_min    = 1.0 / um2
_multi_sigma_max    = 2000 / um2
_multi_factors      = [10**k for k in range(-5, 2)]

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
):
    """Bound fraction vs NP dosing concentration — in vivo, single receptor, Flory-exact.

    Writes adsorption_Npdosing_x{factor}.dat and adsorption_scan_Npdosing.png.
    WARNING: output filenames are identical to scan-multi-npdosing; use separate --output-dir.
    """
    mp.dps = 50
    os.makedirs(output_dir, exist_ok=True)
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

    max_NR_ave = int(mp.pi * R_NP**2 * sigma_R_max)
    max_n_receptor = max_NR_ave + 4 * (max_NR_ave + 1) + 1 if max_NR_ave > 1 else 50
    print(f"Computing K_bind for NR = 1..{max_n_receptor - 1} (slow step, done once)")
    K_bind_vs_NR = system_ref.calculate_K_bind_vs_receptors(max_n_receptor)
    print("K_bind computation done.")

    M_conc = (A_cell / (2.0 * R_NP)**2) * cell_conc
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
):
    """Bound fraction vs NP dosing concentration — in vitro / Langmuir SPR.

    Writes adsorption_Npdosing_x{factor}.dat and adsorption_scan_Npdosing.png.
    """
    mp.dps = 50
    os.makedirs(output_dir, exist_ok=True)
    from system_variables_invitro import (
        R_NP, data_polymers, A_SPR, NP_conc, cell_conc,
        nonspec_interaction, V_SPR, receptor,
    )
    n_sampling_points = _langmuir_n_pts
    sigma_R_min       = _langmuir_sigma_min
    sigma_R_max       = _langmuir_sigma_max
    factors           = list(_langmuir_factors)

    min_exp = np.log10(sigma_R_min)
    max_exp = np.log10(sigma_R_max)
    sigma_R_values = np.logspace(min_exp, max_exp, n_sampling_points)

    system_ref = MultivalentBinding(
        kT=kT, R_NP=R_NP, data_polymers=data_polymers,
        binding_model="exact", polymer_model="Flory-exact",
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
        NP_conc_i = NP_conc * factor
        label = f"Npdosing x{factor:g}"
        print(f"\n--- {label} (NP_conc = {float(NP_conc_i):.3e} nm^-3) ---")

        bound_vs_receptor = system_ref.calculate_bound_vs_receptors_monodisperse(
            max_n_receptor, depletion=False, verbose=False,
            K_bind_vs_NR=K_bind_vs_NR, NP_conc=NP_conc_i,
        )

        sigma_out           = np.zeros(n_sampling_points)
        frac_out            = np.zeros(n_sampling_points)
        frac_out_max        = np.zeros(n_sampling_points)
        nads_out            = np.zeros(n_sampling_points)
        nads_lennart_fraction = np.zeros(n_sampling_points)

        for i, sigma_R in enumerate(sigma_R_values):
            receptor["sigma_R"] = sigma_R
            bound_fraction = system_ref.calculate_bound_fraction(
                fluctuations=True, depletion=False,
                bound_vs_receptor=bound_vs_receptor,
                max_factor=4,
            )
            max_num_sites = A_SPR / system_ref.NP_excluded_area
            sigma_out[i]    = float(sigma_R / (1 / um2))
            frac_out[i]     = float(bound_fraction)
            frac_out_max[i] = 1 - np.exp(-sigma_R * system_ref.NP_excluded_area)
            nads_out[i]     = float(bound_fraction) * max_num_sites

        mass_SPR = V_SPR * NP_conc_i
        mass_A_SPR = nads_out
        nads_lennart_fraction = mass_A_SPR / (mass_SPR + mass_A_SPR)

        results[factor] = (sigma_out, frac_out, nads_out, nads_lennart_fraction)
        fname = f"adsorption_Npdosing_x{factor:g}.dat"
        with open(os.path.join(output_dir, fname), "w") as f:
            for j in range(n_sampling_points):
                f.write(
                    f"{sigma_out[j]:5.3e} {frac_out[j]:5.3e} "
                    f"{nads_out[j]:5.3e} {nads_lennart_fraction[j]:5.3e} \n"
                )
        print(f"  Written to {os.path.join(output_dir, fname)}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    for factor in factors:
        sigma_out, frac_out, nads_out, nads_lennart_fraction = results[factor]
        ax1.plot(sigma_out, frac_out,              linestyle="solid", label=f"Npdosing x{factor:g}")
        ax2.plot(sigma_out, nads_out,              linestyle="solid", label=f"Npdosing x{factor:g}")
        ax3.plot(sigma_out, nads_lennart_fraction, linestyle="solid", label=f"Npdosing x{factor:g}")
    ax1.plot(sigma_out, frac_out_max, linestyle="--", label="Poisson")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Receptor surface density ($\mu$m$^{-2}$)")
    ax1.set_ylabel("Adsorbed fraction")
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
):
    """Bound fraction vs NP dosing — in vivo, multi-receptor, Flory-exact.

    Writes adsorption_Npdosing_x{factor}.dat and adsorption_scan_Npdosing.png.
    WARNING: output filenames are identical to scan-npdosing; use separate --output-dir.
    """
    mp.dps = 50
    os.makedirs(output_dir, exist_ok=True)
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

    max_NR_ave = int(mp.pi * R_NP**2 * sigma_R_max)
    max_n_receptor = max_NR_ave + 4 * (max_NR_ave + 1) + 1 if max_NR_ave > 1 else 50
    print(f"Computing K_bind for NR = 1..{max_n_receptor - 1} (slow step, done once)")
    K_bind_vs_NR = system_ref.calculate_K_bind_vs_receptors(max_n_receptor)
    print("K_bind computation done.")

    M_conc = (A_cell / (2.0 * R_NP)**2) * cell_conc
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
):
    """Sweep sigma_R comparing gaussian and Flory-exact polymer models.

    Runs sweep_1axis (1 independent receptor) or sweep_2axes (2 independent receptors)
    depending on the configuration in system_variables_invivo_multi.py and the
    codependent_receptors setting in scan_both_polymer_models.py.
    """
    os.makedirs(output_dir, exist_ok=True)

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

    models_to_run = (
        _sbpm.polymer_models if _sbpm.compare_polymer_models else [_sbpm.polymer_models[0]]
    )

    results = {}
    for model_name in models_to_run:
        print(f"\n=== Running model: {model_name} ===")
        if n_independent == 1:
            results[model_name] = _sbpm.sweep_1axis(model_name, primary_names[0])
        else:
            results[model_name] = _sbpm.sweep_2axes(
                model_name, primary_names[0], primary_names[1]
            )

    print("\nAll sweeps done.")
    n_models = len(results)

    # ── Write .dat files ──────────────────────────────────────────────────────
    for model_name, res in results.items():
        safe = model_name.replace("-", "_")
        if n_independent == 1:
            fname = f"adsorption_{safe}.dat"
            with open(os.path.join(output_dir, fname), "w") as f:
                f.write("# sigma_R(um^-2)  bound_fraction  n_ads(nm^-3)\n")
                for j in range(len(res["sigma_R_um2"])):
                    f.write(
                        f"{res['sigma_R_um2'][j]:9.4e}  "
                        f"{res['bound_fraction'][j]:9.4e}  "
                        f"{res['n_ads'][j]:9.4e}\n"
                    )
        else:
            fname = f"adsorption_2rec_{safe}.dat"
            nr1 = len(res["sigma_R1_um2"])
            nr2 = len(res["sigma_R2_um2"])
            with open(os.path.join(output_dir, fname), "w") as f:
                f.write(
                    f"# {res['rec_names'][0]}(um^-2)  {res['rec_names'][1]}(um^-2)  "
                    f"bound_fraction  n_ads(nm^-3)\n"
                )
                for i in range(nr1):
                    for j in range(nr2):
                        f.write(
                            f"{res['sigma_R1_um2'][i]:9.4e}  "
                            f"{res['sigma_R2_um2'][j]:9.4e}  "
                            f"{res['bound_fraction'][i, j]:9.4e}  "
                            f"{res['n_ads'][i, j]:9.4e}\n"
                        )
        print(f"Written {os.path.join(output_dir, fname)}")

    # ── Plotting ──────────────────────────────────────────────────────────────
    if n_independent == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        for model_name, res in results.items():
            ax1.plot(res["sigma_R_um2"], res["bound_fraction"],
                     linestyle="solid", label=model_name)
            ax2.plot(res["sigma_R_um2"], res["n_ads"],
                     linestyle="solid", label=model_name)
        for k, v in enumerate(_sbpm.target_sigma_R):
            lbl = _sbpm._ref_label(k, v)
            ax1.axvline(x=v, color="gray", linestyle="dashed", linewidth=0.9, label=lbl)
            ax2.axvline(x=v, color="gray", linestyle="dashed", linewidth=0.9, label=lbl)
        ax1.set_xscale("log")
        ax1.set_xlabel(_sbpm._axis_label(primary_names[0]))
        ax1.set_ylabel("Adsorbed fraction")
        ax1.legend(fontsize="small")
        ax2.set_xscale("log"); ax2.set_yscale("log")
        ax2.set_xlabel(_sbpm._axis_label(primary_names[0]))
        ax2.set_ylabel(r"Adsorbed NP concentration (nm$^{-3}$)")
        ax2.legend(fontsize="small")
        plt.tight_layout()
        out_png = os.path.join(output_dir, "adsorption_polymer_models.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved {out_png}")
    else:
        _sbpm._plot_case_b(
            "bound_fraction", "Adsorbed fraction",
            os.path.join(output_dir, "adsorption_polymer_models_3D.png"),
            os.path.join(output_dir, "adsorption_polymer_models_2D_proj.png"),
            results, n_models,
        )
        _sbpm._plot_case_b(
            "n_ads", r"Adsorbed NP concentration (nm$^{-3}$)",
            os.path.join(output_dir, "adsorption_polymer_models_3D_nads.png"),
            os.path.join(output_dir, "adsorption_polymer_models_2D_proj_nads.png"),
            results, n_models,
        )

    print("\nDone.")
