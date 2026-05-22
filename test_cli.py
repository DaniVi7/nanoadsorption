"""
Tests for cli.py.

Fast tests (always run): ~30 s. Verify the CLI produces correct output for each
command at reduced sweep resolution (5 points, 2 NP-dosing factors).

Slow regression tests (opt-in): run the original scan scripts in a subprocess at
their hardcoded production parameters (n=200, sigma 1–2000 µm⁻², 7 factors) and
compare every output file byte-for-byte (within floating-point format precision)
against the CLI with the same default parameters.

    # Fast only (default):
    python -m unittest test_cli -v

    # Full regression (several minutes):
    RUN_SLOW_TESTS=1 python -m unittest test_cli -v

Critical design constraint — receptor dict sharing
---------------------------------------------------
All three npdosing commands mutate the module-level receptor dict in-place:
    receptor["sigma_R"] = sigma_R
MultivalentBinding reads sigma_R through this SAME dict object at call time.
Reference computations in fast tests therefore use the original (un-copied)
data_polymers and receptor from the cached module, not a deepcopy.
"""
import os
import sys
import shutil
import tempfile
import subprocess
import unittest

import numpy as np
from pathlib import Path
from mpmath import mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from units import um2, kT, mm2
from adsorption import MultivalentBinding
import typer
import scan_both_polymer_models as _sbpm
import cli

DANDEV = os.path.dirname(os.path.abspath(__file__))
RUN_SLOW = bool(os.environ.get("RUN_SLOW_TESTS"))
FACTORS_ALL = [10**k for k in range(-5, 2)]   # [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10]

# Reduced parameters for fast tests.
# max_NR_ave = int(π × 35² × 200 µm⁻²) = 0 → max_n_receptor hits the minimum (50),
# keeping K_bind precomputation fast.
_N_PTS   = 5
_SMIN    = 10.0  / um2
_SMAX    = 200.0 / um2
_FACTORS = [0.1, 1.0]


# ── Module-level helpers ─────────────────────────────────────────────────────

def _load_dat(path):
    return np.loadtxt(path, comments="#")


def _save_cli_state():
    return {
        "_npdosing_n_pts":     cli._npdosing_n_pts,
        "_npdosing_sigma_min": cli._npdosing_sigma_min,
        "_npdosing_sigma_max": cli._npdosing_sigma_max,
        "_npdosing_factors":   list(cli._npdosing_factors),
        "_langmuir_n_pts":     cli._langmuir_n_pts,
        "_langmuir_sigma_min": cli._langmuir_sigma_min,
        "_langmuir_sigma_max": cli._langmuir_sigma_max,
        "_langmuir_factors":   list(cli._langmuir_factors),
        "_multi_n_pts":        cli._multi_n_pts,
        "_multi_sigma_min":    cli._multi_sigma_min,
        "_multi_sigma_max":    cli._multi_sigma_max,
        "_multi_factors":      list(cli._multi_factors),
    }


def _restore_cli_state(s):
    for k, v in s.items():
        setattr(cli, k, v)


def _save_sbpm_state():
    return (
        _sbpm.data_polymers,
        _sbpm.codependent_receptors,
        _sbpm.sigma_R_min,
        _sbpm.sigma_R_max,
        _sbpm.n_pts_1D,
        _sbpm.n_pts_2D,
        _sbpm.polymer_models,
        _sbpm.target_sigma_R,
        _sbpm.target_sigma_R_labels,
        _sbpm.R_NP,
        _sbpm.A_cell,
        _sbpm.NP_conc,
        _sbpm.cell_conc,
        _sbpm.nonspec_interaction,
    )


def _restore_sbpm_state(state):
    (
        _sbpm.data_polymers,
        _sbpm.codependent_receptors,
        _sbpm.sigma_R_min,
        _sbpm.sigma_R_max,
        _sbpm.n_pts_1D,
        _sbpm.n_pts_2D,
        _sbpm.polymer_models,
        _sbpm.target_sigma_R,
        _sbpm.target_sigma_R_labels,
        _sbpm.R_NP,
        _sbpm.A_cell,
        _sbpm.NP_conc,
        _sbpm.cell_conc,
        _sbpm.nonspec_interaction,
    ) = state


def _write_temp_csv(tmp_dir: Path, content: str) -> Path:
    path = tmp_dir / "binders.csv"
    path.write_text(content)
    return path


# ── Slow regression tests ────────────────────────────────────────────────────

@unittest.skipUnless(RUN_SLOW, "set RUN_SLOW_TESTS=1 to enable")
class TestCLINpdosingVsOldScript(unittest.TestCase):
    """CLI scan-npdosing vs scan_Npdosing.py at production parameters (n=200)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_old = tempfile.mkdtemp()
        cls.tmp_new = tempfile.mkdtemp()
        subprocess.run(
            [sys.executable, os.path.join(DANDEV, "scan_Npdosing.py")],
            cwd=cls.tmp_old, check=True,
        )
        cli.scan_npdosing(output_dir=Path(cls.tmp_new))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_old, ignore_errors=True)
        shutil.rmtree(cls.tmp_new, ignore_errors=True)

    def test_dat_files_match(self):
        for factor in FACTORS_ALL:
            with self.subTest(factor=factor):
                fname = f"adsorption_Npdosing_x{factor:g}.dat"
                old = _load_dat(os.path.join(self.tmp_old, fname))
                new = _load_dat(os.path.join(self.tmp_new, fname))
                np.testing.assert_allclose(new, old, rtol=1e-10)

    def test_png_written(self):
        self.assertTrue(
            os.path.exists(os.path.join(self.tmp_new, "adsorption_scan_Npdosing.png"))
        )


@unittest.skipUnless(RUN_SLOW, "set RUN_SLOW_TESTS=1 to enable")
class TestCLILangmuirVsOldScript(unittest.TestCase):
    """CLI scan-npdosing-langmuir vs scan_Npdosing_langmuir.py at production parameters."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_old = tempfile.mkdtemp()
        cls.tmp_new = tempfile.mkdtemp()
        subprocess.run(
            [sys.executable, os.path.join(DANDEV, "scan_Npdosing_langmuir.py")],
            cwd=cls.tmp_old, check=True,
        )
        cli.scan_npdosing_langmuir(output_dir=Path(cls.tmp_new))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_old, ignore_errors=True)
        shutil.rmtree(cls.tmp_new, ignore_errors=True)

    def test_dat_files_match(self):
        for factor in FACTORS_ALL:
            with self.subTest(factor=factor):
                fname = f"adsorption_Npdosing_x{factor:g}.dat"
                old = _load_dat(os.path.join(self.tmp_old, fname))
                new = _load_dat(os.path.join(self.tmp_new, fname))
                np.testing.assert_allclose(new, old, rtol=1e-10)

    def test_png_written(self):
        self.assertTrue(
            os.path.exists(os.path.join(self.tmp_new, "adsorption_scan_Npdosing.png"))
        )


@unittest.skipUnless(RUN_SLOW, "set RUN_SLOW_TESTS=1 to enable")
class TestCLIMultiNpdosingVsOldScript(unittest.TestCase):
    """CLI scan-multi-npdosing vs scan_multi_Npdosing.py at production parameters."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_old = tempfile.mkdtemp()
        cls.tmp_new = tempfile.mkdtemp()
        subprocess.run(
            [sys.executable, os.path.join(DANDEV, "scan_multi_Npdosing.py")],
            cwd=cls.tmp_old, check=True,
        )
        cli.scan_multi_npdosing(output_dir=Path(cls.tmp_new))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_old, ignore_errors=True)
        shutil.rmtree(cls.tmp_new, ignore_errors=True)

    def test_dat_files_match(self):
        for factor in FACTORS_ALL:
            with self.subTest(factor=factor):
                fname = f"adsorption_Npdosing_x{factor:g}.dat"
                old = _load_dat(os.path.join(self.tmp_old, fname))
                new = _load_dat(os.path.join(self.tmp_new, fname))
                np.testing.assert_allclose(new, old, rtol=1e-10)

    def test_png_written(self):
        self.assertTrue(
            os.path.exists(os.path.join(self.tmp_new, "adsorption_scan_Npdosing.png"))
        )


# ── Fast tests ───────────────────────────────────────────────────────────────
# File format ":5.3e" gives 4 significant figures → maximum relative rounding
# error ≈ 5e-4. Tolerances for file-vs-direct comparisons are set to rtol=1e-3.
# Slow regression tests compare two files written with identical format, so the
# rounding is the same on both sides → rtol=1e-10 is valid there.

class TestCLINpdosingFast(unittest.TestCase):
    """scan-npdosing at reduced resolution: physical correctness and file format.

    """

    @classmethod
    def setUpClass(cls):
        mp.dps = 50
        from system_variables_invivo import (
            R_NP, data_polymers, A_cell, NP_conc,
            cell_conc, nonspec_interaction, VTzone, receptor,
        )
        s = _save_cli_state()
        cli._npdosing_n_pts     = _N_PTS
        cli._npdosing_sigma_min = _SMIN
        cli._npdosing_sigma_max = _SMAX
        cli._npdosing_factors   = list(_FACTORS)
        cls.tmp = tempfile.mkdtemp()
        try:
            cli.scan_npdosing(output_dir=Path(cls.tmp))
        finally:
            _restore_cli_state(s)

        # Build reference using identical code path to scan_Npdosing.py.
        # data_polymers['ligands']['receptor'] IS the same dict object as receptor.
        # Setting receptor["sigma_R"] before each call is visible to sys_ref.
        sigma_R_values = np.logspace(np.log10(_SMIN), np.log10(_SMAX), _N_PTS)
        sys_ref = MultivalentBinding(
            kT=kT, R_NP=R_NP, data_polymers=data_polymers,
            binding_model="exact", polymer_model="Flory-exact",
            A_cell=A_cell, NP_conc=NP_conc, cell_conc=cell_conc,
            nonspec_interaction=nonspec_interaction,
        )
        max_n_receptor = 50   # max_NR_ave = int(π × 35² × 200 µm⁻²) = 0 → hits minimum
        # CLI loop left sigma_R set on receptor dict; pop it so that
        # calculate_K_bind_vs_receptors uses n_pts = max_n_receptor (not truncated).
        receptor.pop("sigma_R", None)
        K_bind_vs_NR = sys_ref.calculate_K_bind_vs_receptors(max_n_receptor)
        M_conc = (A_cell / (2.0 * R_NP)**2) * cell_conc

        cls.expected = {}
        cls.got      = {}
        for factor in _FACTORS:
            NP_conc_i = NP_conc * factor
            sigma_out = np.zeros(_N_PTS)
            frac_out  = np.zeros(_N_PTS)
            nads_out  = np.zeros(_N_PTS)
            for i, sigma_R in enumerate(sigma_R_values):
                receptor["sigma_R"] = sigma_R
                bf = sys_ref.calculate_bound_fraction(
                    fluctuations=True, depletion=True,
                    K_bind_vs_receptors=K_bind_vs_NR,
                    max_n_receptor=max_n_receptor,
                    NP_conc=NP_conc_i,
                    rho_m=M_conc,
                )
                sigma_out[i] = float(sigma_R / (1 / um2))
                frac_out[i]  = float(bf)
                nads_out[i]  = float(bf) * NP_conc_i * VTzone
            cls.expected[factor] = (sigma_out, frac_out, nads_out)
            cls.got[factor] = _load_dat(
                os.path.join(cls.tmp, f"adsorption_Npdosing_x{factor:g}.dat")
            )

    @classmethod
    def tearDownClass(cls):
        from system_variables_invivo import receptor as _invivo_receptor
        _invivo_receptor.pop("sigma_R", None)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_dat_files_exist(self):
        for factor in _FACTORS:
            with self.subTest(factor=factor):
                self.assertTrue(
                    os.path.exists(
                        os.path.join(self.tmp, f"adsorption_Npdosing_x{factor:g}.dat")
                    )
                )

    def test_dat_dimensions(self):
        for factor in _FACTORS:
            with self.subTest(factor=factor):
                self.assertEqual(self.got[factor].shape, (_N_PTS, 3))

    def test_sigma_R_column(self):
        expected_sigma = np.logspace(np.log10(_SMIN), np.log10(_SMAX), _N_PTS) / (1 / um2)
        np.testing.assert_allclose(
            self.got[_FACTORS[0]][:, 0], expected_sigma, rtol=1e-3
        )

    def test_bound_fraction_matches_direct(self):
        for factor in _FACTORS:
            with self.subTest(factor=factor):
                _, exp_frac, _ = self.expected[factor]
                np.testing.assert_allclose(
                    self.got[factor][:, 1], exp_frac, rtol=1e-3, atol=1e-9
                )

    def test_n_ads_matches_direct(self):
        for factor in _FACTORS:
            with self.subTest(factor=factor):
                _, _, exp_nads = self.expected[factor]
                np.testing.assert_allclose(
                    self.got[factor][:, 2], exp_nads, rtol=1e-3, atol=1e-30
                )

    def test_bound_fraction_in_unit_interval(self):
        for factor in _FACTORS:
            bf = self.got[factor][:, 1]
            self.assertTrue(np.all(bf >= 0) and np.all(bf <= 1))

    def test_monotone_nondecreasing(self):
        """Adsorbed fraction must not decrease as sigma_R increases."""
        for factor in _FACTORS:
            with self.subTest(factor=factor):
                bf = self.got[factor][:, 1]
                self.assertTrue(np.all(np.diff(bf) >= -1e-6))


class TestCLILangmuirFast(unittest.TestCase):
    """scan-npdosing-langmuir at reduced resolution: physical correctness and file format."""

    @classmethod
    def setUpClass(cls):
        mp.dps = 50
        from system_variables_invitro import (
            R_NP, data_polymers, A_SPR, NP_conc,
            cell_conc, nonspec_interaction, V_SPR, receptor,
        )
        s = _save_cli_state()
        cli._langmuir_n_pts     = _N_PTS
        cli._langmuir_sigma_min = _SMIN
        cli._langmuir_sigma_max = _SMAX
        cli._langmuir_factors   = list(_FACTORS)
        cls.tmp = tempfile.mkdtemp()
        try:
            cli.scan_npdosing_langmuir(output_dir=Path(cls.tmp))
        finally:
            _restore_cli_state(s)

        # Build reference using identical code path to scan_Npdosing_langmuir.py.
        # Square NP footprint formula: max_NR_ave = int((2R)² × σ_R_max).
        sigma_R_values = np.logspace(np.log10(_SMIN), np.log10(_SMAX), _N_PTS)
        sys_ref = MultivalentBinding(
            kT=kT, R_NP=R_NP, data_polymers=data_polymers,
            binding_model="exact", polymer_model="Flory-exact",
            A_cell=A_SPR, NP_conc=NP_conc, cell_conc=cell_conc,
            nonspec_interaction=nonspec_interaction,
        )
        max_n_receptor = 20   # int((2×35)² × 200 µm⁻²) = 0 → hits minimum of 20
        receptor.pop("sigma_R", None)  # clear CLI residual so n_pts = max_n_receptor
        K_bind_vs_NR = sys_ref.calculate_K_bind_vs_receptors(max_n_receptor)

        cls.expected = {}
        cls.got      = {}
        for factor in _FACTORS:
            NP_conc_i = NP_conc * factor
            # bound_vs_receptor is precomputed outside the sigma_R loop (matches script)
            bound_vs_receptor = sys_ref.calculate_bound_vs_receptors_monodisperse(
                max_n_receptor, depletion=False, verbose=False,
                K_bind_vs_NR=K_bind_vs_NR, NP_conc=NP_conc_i,
            )
            sigma_out = np.zeros(_N_PTS)
            frac_out  = np.zeros(_N_PTS)
            nads_out  = np.zeros(_N_PTS)
            for i, sigma_R in enumerate(sigma_R_values):
                receptor["sigma_R"] = sigma_R
                bf = sys_ref.calculate_bound_fraction(
                    fluctuations=True, depletion=False,
                    bound_vs_receptor=bound_vs_receptor,
                    max_factor=4,
                )
                max_num_sites = A_SPR / sys_ref.NP_excluded_area
                sigma_out[i] = float(sigma_R / (1 / um2))
                frac_out[i]  = float(bf)
                nads_out[i]  = float(bf) * max_num_sites
            lennart = nads_out / (V_SPR * NP_conc_i + nads_out)
            cls.expected[factor] = (sigma_out, frac_out, nads_out, lennart)
            cls.got[factor] = _load_dat(
                os.path.join(cls.tmp, f"adsorption_Npdosing_x{factor:g}.dat")
            )

    @classmethod
    def tearDownClass(cls):
        # Clear sigma_R left on the receptor dict by the sweep loop; without this,
        # calculate_K_bind_vs_receptors in subsequent slow tests returns a truncated array.
        from system_variables_invitro import receptor as _invitro_receptor
        _invitro_receptor.pop("sigma_R", None)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_dat_files_exist(self):
        for factor in _FACTORS:
            with self.subTest(factor=factor):
                self.assertTrue(
                    os.path.exists(
                        os.path.join(self.tmp, f"adsorption_Npdosing_x{factor:g}.dat")
                    )
                )

    def test_dat_dimensions(self):
        for factor in _FACTORS:
            with self.subTest(factor=factor):
                self.assertEqual(self.got[factor].shape, (_N_PTS, 4))

    def test_bound_fraction_matches_direct(self):
        for factor in _FACTORS:
            with self.subTest(factor=factor):
                _, exp_frac, _, _ = self.expected[factor]
                np.testing.assert_allclose(
                    self.got[factor][:, 1], exp_frac, rtol=1e-3, atol=1e-9
                )

    def test_n_ads_matches_direct(self):
        for factor in _FACTORS:
            with self.subTest(factor=factor):
                _, _, exp_nads, _ = self.expected[factor]
                np.testing.assert_allclose(
                    self.got[factor][:, 2], exp_nads, rtol=1e-3, atol=1e-30
                )

    def test_langmuir_fraction_matches_direct(self):
        for factor in _FACTORS:
            with self.subTest(factor=factor):
                _, _, _, exp_lennart = self.expected[factor]
                np.testing.assert_allclose(
                    self.got[factor][:, 3], exp_lennart, rtol=1e-3, atol=1e-9
                )

    def test_bound_fraction_in_unit_interval(self):
        for factor in _FACTORS:
            bf = self.got[factor][:, 1]
            self.assertTrue(np.all(bf >= 0) and np.all(bf <= 1))

    def test_langmuir_fraction_in_unit_interval(self):
        for factor in _FACTORS:
            lf = self.got[factor][:, 3]
            self.assertTrue(np.all(lf >= 0) and np.all(lf <= 1))


class TestCLIMultiNpdosingFast(unittest.TestCase):
    """scan-multi-npdosing at reduced resolution: physical correctness and file format.

    Uses system_variables_invivo_multi (N_ligands=80, KD=10000 nM — different physics
    from scan-npdosing). data_polymers has two ligand entries ('ligands' and 'ligands2')
    both pointing to the SAME receptor dict; setting receptor["sigma_R"] updates both.
    """

    @classmethod
    def setUpClass(cls):
        mp.dps = 50
        from system_variables_invivo_multi import (
            R_NP, data_polymers, A_cell, NP_conc,
            cell_conc, nonspec_interaction, VTzone, receptor,
        )
        s = _save_cli_state()
        cli._multi_n_pts     = _N_PTS
        cli._multi_sigma_min = _SMIN
        cli._multi_sigma_max = _SMAX
        cli._multi_factors   = list(_FACTORS)
        cls.tmp = tempfile.mkdtemp()
        try:
            cli.scan_multi_npdosing(output_dir=Path(cls.tmp))
        finally:
            _restore_cli_state(s)

        sigma_R_values = np.logspace(np.log10(_SMIN), np.log10(_SMAX), _N_PTS)
        sys_ref = MultivalentBinding(
            kT=kT, R_NP=R_NP, data_polymers=data_polymers,
            binding_model="exact", polymer_model="Flory-exact",
            A_cell=A_cell, NP_conc=NP_conc, cell_conc=cell_conc,
            nonspec_interaction=nonspec_interaction,
        )
        max_n_receptor = 50   # max_NR_ave = 0 at 200 µm⁻² → hits minimum
        receptor.pop("sigma_R", None)  # clear CLI residual so n_pts = max_n_receptor
        K_bind_vs_NR = sys_ref.calculate_K_bind_vs_receptors(max_n_receptor)
        M_conc = (A_cell / (2.0 * R_NP)**2) * cell_conc

        cls.expected = {}
        cls.got      = {}
        for factor in _FACTORS:
            NP_conc_i = NP_conc * factor
            sigma_out = np.zeros(_N_PTS)
            frac_out  = np.zeros(_N_PTS)
            nads_out  = np.zeros(_N_PTS)
            for i, sigma_R in enumerate(sigma_R_values):
                receptor["sigma_R"] = sigma_R
                bf = sys_ref.calculate_bound_fraction(
                    fluctuations=True, depletion=True,
                    K_bind_vs_receptors=K_bind_vs_NR,
                    max_n_receptor=max_n_receptor,
                    NP_conc=NP_conc_i,
                    rho_m=M_conc,
                )
                sigma_out[i] = float(sigma_R / (1 / um2))
                frac_out[i]  = float(bf)
                nads_out[i]  = float(bf) * NP_conc_i * VTzone
            cls.expected[factor] = (sigma_out, frac_out, nads_out)
            cls.got[factor] = _load_dat(
                os.path.join(cls.tmp, f"adsorption_Npdosing_x{factor:g}.dat")
            )

    @classmethod
    def tearDownClass(cls):
        from system_variables_invivo_multi import receptor as _multi_receptor
        _multi_receptor.pop("sigma_R", None)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_dat_files_exist(self):
        for factor in _FACTORS:
            with self.subTest(factor=factor):
                self.assertTrue(
                    os.path.exists(
                        os.path.join(self.tmp, f"adsorption_Npdosing_x{factor:g}.dat")
                    )
                )

    def test_dat_dimensions(self):
        for factor in _FACTORS:
            with self.subTest(factor=factor):
                self.assertEqual(self.got[factor].shape, (_N_PTS, 3))

    def test_sigma_R_column(self):
        expected_sigma = np.logspace(np.log10(_SMIN), np.log10(_SMAX), _N_PTS) / (1 / um2)
        np.testing.assert_allclose(
            self.got[_FACTORS[0]][:, 0], expected_sigma, rtol=1e-3
        )

    def test_bound_fraction_matches_direct(self):
        for factor in _FACTORS:
            with self.subTest(factor=factor):
                _, exp_frac, _ = self.expected[factor]
                np.testing.assert_allclose(
                    self.got[factor][:, 1], exp_frac, rtol=1e-3, atol=1e-9
                )

    def test_n_ads_matches_direct(self):
        for factor in _FACTORS:
            with self.subTest(factor=factor):
                _, _, exp_nads = self.expected[factor]
                np.testing.assert_allclose(
                    self.got[factor][:, 2], exp_nads, rtol=1e-3, atol=1e-30
                )

    def test_bound_fraction_in_unit_interval(self):
        for factor in _FACTORS:
            bf = self.got[factor][:, 1]
            self.assertTrue(np.all(bf >= 0) and np.all(bf <= 1))

    def test_monotone_nondecreasing(self):
        """Adsorbed fraction must not decrease as sigma_R increases."""
        for factor in _FACTORS:
            with self.subTest(factor=factor):
                bf = self.got[factor][:, 1]
                self.assertTrue(np.all(np.diff(bf) >= -1e-6))


class TestCLIBothModelsFast(unittest.TestCase):
    """scan-both-polymer-models: file output matches sweep_1axis return value directly.

    The CLI and the reference call both run under the same patched _sbpm state,
    so the K_bind tables and sweep results are deterministic and identical.
    File format is ":9.4e" (5 significant figures); tolerances reflect this.
    """

    @classmethod
    def setUpClass(cls):
        saved = _save_sbpm_state()
        _sbpm.n_pts_1D               = _N_PTS
        _sbpm.sigma_R_min            = _SMIN
        _sbpm.sigma_R_max            = _SMAX
        _sbpm.polymer_models         = ["gaussian"]
        # data_polymers is NOT patched: default (from system_variables_invivo_multi)
        # has receptor name "default" → primary_names = ["default"] → 1D sweep.
        cls.tmp = tempfile.mkdtemp()
        try:
            cli.scan_both_polymer_models_cmd(output_dir=Path(cls.tmp))
            # Reference sweep runs under the same patched state as the CLI call above.
            cls.ref = _sbpm.sweep_1axis("gaussian", "default")
        finally:
            _restore_sbpm_state(saved)
        cls.dat = _load_dat(os.path.join(cls.tmp, "adsorption_gaussian.dat"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_dat_file_written(self):
        self.assertTrue(
            os.path.exists(os.path.join(self.tmp, "adsorption_gaussian.dat"))
        )

    def test_png_written(self):
        self.assertTrue(
            os.path.exists(os.path.join(self.tmp, "adsorption_polymer_models.png"))
        )

    def test_dat_dimensions(self):
        self.assertEqual(self.dat.shape, (_N_PTS, 3))

    def test_dat_header(self):
        with open(os.path.join(self.tmp, "adsorption_gaussian.dat")) as f:
            first_line = f.readline()
        self.assertTrue(first_line.startswith("# sigma_R"))

    def test_bound_fraction_matches_sweep(self):
        # ":9.4e" format → 5 significant figures → max relative error ≈ 5e-5
        np.testing.assert_allclose(
            self.dat[:, 1], self.ref["bound_fraction"], rtol=1e-4, atol=1e-9
        )

    def test_n_ads_matches_sweep(self):
        np.testing.assert_allclose(
            self.dat[:, 2], self.ref["n_ads"], rtol=1e-4, atol=1e-30
        )

    def test_bound_fraction_in_unit_interval(self):
        bf = self.dat[:, 1]
        self.assertTrue(np.all(bf >= 0) and np.all(bf <= 1))


# ── YAML feature tests ────────────────────────────────────────────────────────

def _write_temp_yaml(tmp_dir, content):
    """Write a YAML string to a temp file in tmp_dir; return its Path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, dir=str(tmp_dir))
    f.write(content)
    f.close()
    return Path(f.name)


_YAML_MINIMAL = (
    "R_NP_nm: 35.0\n"
    "N_ligands: 150\n"
    "PEG_monomer_size_nm: 0.28\n"
    "PEG_kuhn_length_nm: 0.76\n"
    "PEG_ligand_MW_g_per_mol: 3400.0\n"
    "PEG_short_MW_g_per_mol: 2000.0\n"
    "PEG_short_to_ligand_ratio: 11.4\n"
    "KD_nM: 150.0\n"
    "binder_linear_size_nm: 3.5\n"
    "nonspec_interaction_kT: 0.0\n"
    "receptors:\n"
    "  - name: default\n"
    "ligands:\n"
    "  - name: PEG2K\n"
    "    type: inert\n"
    "  - name: ligands\n"
    "    type: binding\n"
    "    receptor: default\n"
)

# Two-receptor YAML with one codependent receptor (recB secondary of recA).
# Without codependent_receptors this would be a 2D sweep; with it → 1D sweep.
_YAML_TWO_RECS_CODEP = (
    "R_NP_nm: 35.0\nN_ligands: 80\n"
    "PEG_monomer_size_nm: 0.28\nPEG_kuhn_length_nm: 0.76\n"
    "PEG_ligand_MW_g_per_mol: 3400.0\nPEG_short_MW_g_per_mol: 2000.0\n"
    "PEG_short_to_ligand_ratio: 11.4\nKD_nM: 150.0\n"
    "binder_linear_size_nm: 3.5\nnonspec_interaction_kT: 0.0\n"
    "receptors:\n  - name: recA\n  - name: recB\n"
    "ligands:\n  - name: PEG2K\n    type: inert\n"
    "  - name: lig_A\n    type: binding\n    receptor: recA\n"
    "  - name: lig_B\n    type: binding\n    receptor: recB\n"
    "codependent_receptors:\n"
    "  - secondary: recB\n    primary: recA\n    ratio: 1.0\n"
    "n_pts_1D: 3\n"
)


class TestGenerateTemplate(unittest.TestCase):
    """generate_template() writes a parseable YAML with all required fields."""

    @classmethod
    def setUpClass(cls):
        import yaml as _yaml
        cls.tmp_dir = Path(tempfile.mkdtemp())
        cli.generate_template(output=cls.tmp_dir / "template.yaml")
        with open(cls.tmp_dir / "template.yaml") as f:
            cls.parsed = _yaml.safe_load(f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_file_exists(self):
        self.assertTrue((self.tmp_dir / "template.yaml").exists())

    def test_is_valid_yaml(self):
        self.assertIsNotNone(self.parsed)

    def test_required_keys_present(self):
        for key in [
            "R_NP_nm", "N_ligands", "PEG_monomer_size_nm", "PEG_kuhn_length_nm",
            "PEG_ligand_MW_g_per_mol", "PEG_short_MW_g_per_mol", "PEG_short_to_ligand_ratio",
            "KD_nM", "binder_linear_size_nm", "nonspec_interaction_kT",
            "receptors", "ligands",
        ]:
            with self.subTest(key=key):
                self.assertIn(key, self.parsed)

    def test_sweep_keys_present(self):
        for key in [
            "sigma_R_min_per_um2", "sigma_R_max_per_um2",
            "n_pts_1D", "n_pts_2D",
            "polymer_models",
            "target_sigma_R", "target_sigma_R_labels",
        ]:
            with self.subTest(key=key):
                self.assertIn(key, self.parsed)

    def test_receptors_is_nonempty_list(self):
        self.assertIsInstance(self.parsed["receptors"], list)
        self.assertGreater(len(self.parsed["receptors"]), 0)

    def test_ligands_is_nonempty_list(self):
        self.assertIsInstance(self.parsed["ligands"], list)
        self.assertGreater(len(self.parsed["ligands"]), 0)


class TestLoadYaml(unittest.TestCase):
    """Unit tests for cli._load_system_vars_yaml() in isolation.

    Uses minimal hand-crafted YAML strings; never invokes a CLI command.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = Path(tempfile.mkdtemp())
        cls.base_path = _write_temp_yaml(cls.tmp_dir, _YAML_MINIMAL)
        cls.base_vars = cli._load_system_vars_yaml(cls.base_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_missing_required_key_raises(self):
        bad = _YAML_MINIMAL.replace("R_NP_nm: 35.0\n", "")
        with self.assertRaises(ValueError):
            cli._load_system_vars_yaml(_write_temp_yaml(self.tmp_dir, bad))

    def test_unknown_receptor_raises(self):
        bad = _YAML_MINIMAL.replace("    receptor: default\n", "    receptor: ghost\n")
        with self.assertRaises(ValueError):
            cli._load_system_vars_yaml(_write_temp_yaml(self.tmp_dir, bad))

    def test_binding_without_receptor_key_raises(self):
        bad = _YAML_MINIMAL.replace(
            "  - name: ligands\n    type: binding\n    receptor: default\n",
            "  - name: ligands\n    type: binding\n",
        )
        with self.assertRaises(ValueError):
            cli._load_system_vars_yaml(_write_temp_yaml(self.tmp_dir, bad))

    def test_inert_ligand_has_no_k_bind(self):
        self.assertNotIn("K_bind_0", self.base_vars["data_polymers"]["PEG2K"])

    def test_binding_ligand_k_bind_correct(self):
        from units import nM
        np.testing.assert_allclose(
            self.base_vars["data_polymers"]["ligands"]["K_bind_0"],
            1.0 / (150.0 * nM), rtol=1e-12,
        )

    def test_receptor_identity_shared(self):
        """Two binding ligands with the same receptor name share one Python dict object."""
        yaml_multi = (
            _YAML_MINIMAL
            .replace("N_ligands: 150\n", "N_ligands: 80\n")
            .replace(
                "  - name: ligands\n    type: binding\n    receptor: default\n",
                (
                    "  - name: ligands\n    type: binding\n"
                    "    receptor: default\n    KD_nM: 10000.0\n"
                    "  - name: ligands2\n    type: binding\n"
                    "    receptor: default\n    KD_nM: 10000.0\n"
                ),
            )
        )
        v = cli._load_system_vars_yaml(_write_temp_yaml(self.tmp_dir, yaml_multi))
        self.assertIs(
            v["data_polymers"]["ligands"]["receptor"],
            v["data_polymers"]["ligands2"]["receptor"],
        )

    def test_receptor_identity_distinct(self):
        """Two binding ligands with different receptor names get distinct dict objects."""
        yaml_two = (
            "R_NP_nm: 35.0\nN_ligands: 80\n"
            "PEG_monomer_size_nm: 0.28\nPEG_kuhn_length_nm: 0.76\n"
            "PEG_ligand_MW_g_per_mol: 3400.0\nPEG_short_MW_g_per_mol: 2000.0\n"
            "PEG_short_to_ligand_ratio: 11.4\nKD_nM: 150.0\n"
            "binder_linear_size_nm: 3.5\nnonspec_interaction_kT: 0.0\n"
            "receptors:\n  - name: CD44\n  - name: CD8\n"
            "ligands:\n  - name: PEG2K\n    type: inert\n"
            "  - name: lig_CD44\n    type: binding\n    receptor: CD44\n    KD_nM: 150.0\n"
            "  - name: lig_CD8\n    type: binding\n    receptor: CD8\n    KD_nM: 500.0\n"
        )
        v = cli._load_system_vars_yaml(_write_temp_yaml(self.tmp_dir, yaml_two))
        r44 = v["data_polymers"]["lig_CD44"]["receptor"]
        r8  = v["data_polymers"]["lig_CD8"]["receptor"]
        self.assertIsNot(r44, r8)
        self.assertEqual(r44["name"], "CD44")
        self.assertEqual(r8["name"], "CD8")

    def test_receptor_is_first_rec_in_data_polymers(self):
        """Returned 'receptor' is the same object as the first binding ligand's receptor."""
        self.assertIs(
            self.base_vars["receptor"],
            self.base_vars["data_polymers"]["ligands"]["receptor"],
        )

    def test_string_sci_notation_handled(self):
        """cell_conc_per_mL without explicit sign (e.g. 1.875e8) is parsed as string
        by PyYAML; the float() cast in _load_system_vars_yaml must handle it correctly."""
        yaml_str = _YAML_MINIMAL + "cell_conc_per_mL: 1.875e8\n"
        v = cli._load_system_vars_yaml(_write_temp_yaml(self.tmp_dir, yaml_str))
        from units import mL
        np.testing.assert_allclose(v["cell_conc"], 1.875e8 / mL, rtol=1e-12)

    def test_per_ligand_kd_override(self):
        """Per-ligand KD_nM overrides the global KD_nM for that ligand only."""
        yaml_override = _YAML_MINIMAL.replace(
            "  - name: ligands\n    type: binding\n    receptor: default\n",
            "  - name: ligands\n    type: binding\n    receptor: default\n    KD_nM: 42.0\n",
        )
        v = cli._load_system_vars_yaml(_write_temp_yaml(self.tmp_dir, yaml_override))
        from units import nM
        np.testing.assert_allclose(
            v["data_polymers"]["ligands"]["K_bind_0"], 1.0 / (42.0 * nM), rtol=1e-12,
        )

    def test_r_np_converted_to_nm(self):
        from units import nm
        np.testing.assert_allclose(self.base_vars["R_NP"], 35.0 * nm, rtol=1e-12)

    def test_a_cell_converted_to_nm2(self):
        from units import um2
        np.testing.assert_allclose(self.base_vars["A_cell"], 100.0 * um2, rtol=1e-12)


class TestYamlVsPythonInvivo(unittest.TestCase):
    """scan-npdosing with a YAML built from exact invivo.py values produces output
    files identical (within 4-sig-fig format precision) to the no-config run."""

    @classmethod
    def setUpClass(cls):
        mp.dps = 50
        from system_variables_invivo import (
            R_NP, N_ligands, KD, A_cell, NP_conc, cell_conc,
            VTzone, nonspec_interaction, binder_linear_size, receptor,
        )
        from units import nm, nM, um2, mL

        yaml_content = (
            f"R_NP_nm: {repr(float(R_NP / nm))}\n"
            f"N_ligands: {int(N_ligands)}\n"
            "PEG_monomer_size_nm: 0.28\n"
            "PEG_kuhn_length_nm: 0.76\n"
            "PEG_ligand_MW_g_per_mol: 3400.0\n"
            "PEG_short_MW_g_per_mol: 2000.0\n"
            "PEG_short_to_ligand_ratio: 11.4\n"
            f"KD_nM: {repr(float(KD / nM))}\n"
            f"binder_linear_size_nm: {repr(float(binder_linear_size / nm))}\n"
            f"nonspec_interaction_kT: {repr(float(nonspec_interaction))}\n"
            f"A_cell_um2: {repr(float(A_cell / um2))}\n"
            f"cell_conc_per_mL: {repr(float(cell_conc * mL))}\n"
            f"NP_conc_per_mL: {repr(float(NP_conc * mL))}\n"
            f"VTzone_mL: {repr(float(VTzone / mL))}\n"
            "A_SPR_mm2: 1.0\n"
            "V_SPR_mL: 6.0e-5\n"
            "NP_conc_SPR_per_mL: 4.0e+11\n"
            "receptors:\n  - name: default\n"
            "ligands:\n  - name: PEG2K\n    type: inert\n"
            "  - name: ligands\n    type: binding\n    receptor: default\n"
            f"    KD_nM: {repr(float(KD / nM))}\n"
        )
        cls.tmp_dir = Path(tempfile.mkdtemp())
        yaml_path = cls.tmp_dir / "invivo_exact.yaml"
        yaml_path.write_text(yaml_content)
        dir_py   = cls.tmp_dir / "py"
        dir_yaml = cls.tmp_dir / "yaml"
        dir_py.mkdir(); dir_yaml.mkdir()

        s = _save_cli_state()
        cli._npdosing_n_pts     = _N_PTS
        cli._npdosing_sigma_min = _SMIN
        cli._npdosing_sigma_max = _SMAX
        cli._npdosing_factors   = list(_FACTORS)
        try:
            receptor.pop("sigma_R", None)
            cli.scan_npdosing(output_dir=dir_py)
            receptor.pop("sigma_R", None)
            cli.scan_npdosing(output_dir=dir_yaml, config=yaml_path)
        finally:
            _restore_cli_state(s)

        cls.got_py   = {f: _load_dat(dir_py   / f"adsorption_Npdosing_x{f:g}.dat")
                        for f in _FACTORS}
        cls.got_yaml = {f: _load_dat(dir_yaml / f"adsorption_Npdosing_x{f:g}.dat")
                        for f in _FACTORS}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_sigma_R_match(self):
        for f in _FACTORS:
            with self.subTest(factor=f):
                np.testing.assert_allclose(
                    self.got_yaml[f][:, 0], self.got_py[f][:, 0], rtol=1e-3
                )

    def test_bound_fraction_match(self):
        for f in _FACTORS:
            with self.subTest(factor=f):
                np.testing.assert_allclose(
                    self.got_yaml[f][:, 1], self.got_py[f][:, 1], rtol=1e-3, atol=1e-9
                )

    def test_n_ads_match(self):
        for f in _FACTORS:
            with self.subTest(factor=f):
                np.testing.assert_allclose(
                    self.got_yaml[f][:, 2], self.got_py[f][:, 2], rtol=1e-3, atol=1e-30
                )


class TestYamlVsPythonLangmuir(unittest.TestCase):
    """scan-npdosing-langmuir with a YAML built from exact invitro.py values matches
    the no-config run (all four columns: sigma_R, bf, n_ads, langmuir_fraction)."""

    @classmethod
    def setUpClass(cls):
        mp.dps = 50
        from system_variables_invitro import (
            R_NP, N_ligands, KD, A_SPR, NP_conc, V_SPR,
            nonspec_interaction, binder_linear_size, receptor,
        )
        from units import nm, nM, mm2, mL

        yaml_content = (
            f"R_NP_nm: {repr(float(R_NP / nm))}\n"
            f"N_ligands: {int(N_ligands)}\n"
            "PEG_monomer_size_nm: 0.28\n"
            "PEG_kuhn_length_nm: 0.76\n"
            "PEG_ligand_MW_g_per_mol: 3400.0\n"
            "PEG_short_MW_g_per_mol: 2000.0\n"
            "PEG_short_to_ligand_ratio: 11.4\n"
            f"KD_nM: {repr(float(KD / nM))}\n"
            f"binder_linear_size_nm: {repr(float(binder_linear_size / nm))}\n"
            f"nonspec_interaction_kT: {repr(float(nonspec_interaction))}\n"
            "A_cell_um2: 100.0\n"
            "cell_conc_per_mL: 1.875e+8\n"
            "NP_conc_per_mL: 1.905e+12\n"
            "VTzone_mL: 0.042\n"
            f"A_SPR_mm2: {repr(float(A_SPR / mm2))}\n"
            f"V_SPR_mL: {repr(float(V_SPR / mL))}\n"
            f"NP_conc_SPR_per_mL: {repr(float(NP_conc * mL))}\n"
            "receptors:\n  - name: default\n"
            "ligands:\n  - name: PEG2K\n    type: inert\n"
            "  - name: ligands\n    type: binding\n    receptor: default\n"
            f"    KD_nM: {repr(float(KD / nM))}\n"
        )
        cls.tmp_dir = Path(tempfile.mkdtemp())
        yaml_path = cls.tmp_dir / "invitro_exact.yaml"
        yaml_path.write_text(yaml_content)
        dir_py   = cls.tmp_dir / "py"
        dir_yaml = cls.tmp_dir / "yaml"
        dir_py.mkdir(); dir_yaml.mkdir()

        s = _save_cli_state()
        cli._langmuir_n_pts     = _N_PTS
        cli._langmuir_sigma_min = _SMIN
        cli._langmuir_sigma_max = _SMAX
        cli._langmuir_factors   = list(_FACTORS)
        try:
            receptor.pop("sigma_R", None)
            cli.scan_npdosing_langmuir(output_dir=dir_py)
            receptor.pop("sigma_R", None)
            cli.scan_npdosing_langmuir(output_dir=dir_yaml, config=yaml_path)
        finally:
            _restore_cli_state(s)

        cls.got_py   = {f: _load_dat(dir_py   / f"adsorption_Npdosing_x{f:g}.dat")
                        for f in _FACTORS}
        cls.got_yaml = {f: _load_dat(dir_yaml / f"adsorption_Npdosing_x{f:g}.dat")
                        for f in _FACTORS}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_sigma_R_match(self):
        for f in _FACTORS:
            with self.subTest(factor=f):
                np.testing.assert_allclose(
                    self.got_yaml[f][:, 0], self.got_py[f][:, 0], rtol=1e-3
                )

    def test_bound_fraction_match(self):
        for f in _FACTORS:
            with self.subTest(factor=f):
                np.testing.assert_allclose(
                    self.got_yaml[f][:, 1], self.got_py[f][:, 1], rtol=1e-3, atol=1e-9
                )

    def test_n_ads_match(self):
        for f in _FACTORS:
            with self.subTest(factor=f):
                np.testing.assert_allclose(
                    self.got_yaml[f][:, 2], self.got_py[f][:, 2], rtol=1e-3, atol=1e-30
                )

    def test_langmuir_fraction_match(self):
        for f in _FACTORS:
            with self.subTest(factor=f):
                np.testing.assert_allclose(
                    self.got_yaml[f][:, 3], self.got_py[f][:, 3], rtol=1e-3, atol=1e-9
                )


class TestYamlVsPythonMulti(unittest.TestCase):
    """scan-multi-npdosing with a YAML (two ligands, same receptor, KD=10000 nM, N=80)
    matches the no-config run from system_variables_invivo_multi.py.

    This is the primary test for receptor dict object identity: both YAML ligand entries
    reference 'default', so the loader must assign them the same Python dict object.
    """

    @classmethod
    def setUpClass(cls):
        mp.dps = 50
        from system_variables_invivo_multi import (
            R_NP, N_ligands, KD, A_cell, NP_conc, cell_conc,
            VTzone, nonspec_interaction, binder_linear_size, receptor,
        )
        from units import nm, nM, um2, mL

        yaml_content = (
            f"R_NP_nm: {repr(float(R_NP / nm))}\n"
            f"N_ligands: {int(N_ligands)}\n"
            "PEG_monomer_size_nm: 0.28\n"
            "PEG_kuhn_length_nm: 0.76\n"
            "PEG_ligand_MW_g_per_mol: 3400.0\n"
            "PEG_short_MW_g_per_mol: 2000.0\n"
            "PEG_short_to_ligand_ratio: 11.4\n"
            f"KD_nM: {repr(float(KD / nM))}\n"
            f"binder_linear_size_nm: {repr(float(binder_linear_size / nm))}\n"
            f"nonspec_interaction_kT: {repr(float(nonspec_interaction))}\n"
            f"A_cell_um2: {repr(float(A_cell / um2))}\n"
            f"cell_conc_per_mL: {repr(float(cell_conc * mL))}\n"
            f"NP_conc_per_mL: {repr(float(NP_conc * mL))}\n"
            f"VTzone_mL: {repr(float(VTzone / mL))}\n"
            "A_SPR_mm2: 1.0\n"
            "V_SPR_mL: 6.0e-5\n"
            "NP_conc_SPR_per_mL: 4.0e+11\n"
            "receptors:\n  - name: default\n"
            "ligands:\n  - name: PEG2K\n    type: inert\n"
            "  - name: ligands\n    type: binding\n    receptor: default\n"
            f"    KD_nM: {repr(float(KD / nM))}\n"
            "  - name: ligands2\n    type: binding\n    receptor: default\n"
            f"    KD_nM: {repr(float(KD / nM))}\n"
        )
        cls.tmp_dir = Path(tempfile.mkdtemp())
        yaml_path = cls.tmp_dir / "invivo_multi_exact.yaml"
        yaml_path.write_text(yaml_content)
        dir_py   = cls.tmp_dir / "py"
        dir_yaml = cls.tmp_dir / "yaml"
        dir_py.mkdir(); dir_yaml.mkdir()

        s = _save_cli_state()
        cli._multi_n_pts     = _N_PTS
        cli._multi_sigma_min = _SMIN
        cli._multi_sigma_max = _SMAX
        cli._multi_factors   = list(_FACTORS)
        try:
            receptor.pop("sigma_R", None)
            cli.scan_multi_npdosing(output_dir=dir_py)
            cli.scan_multi_npdosing(output_dir=dir_yaml, config=yaml_path)
        finally:
            _restore_cli_state(s)

        cls.got_py   = {f: _load_dat(dir_py   / f"adsorption_Npdosing_x{f:g}.dat")
                        for f in _FACTORS}
        cls.got_yaml = {f: _load_dat(dir_yaml / f"adsorption_Npdosing_x{f:g}.dat")
                        for f in _FACTORS}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_sigma_R_match(self):
        for f in _FACTORS:
            with self.subTest(factor=f):
                np.testing.assert_allclose(
                    self.got_yaml[f][:, 0], self.got_py[f][:, 0], rtol=1e-3
                )

    def test_bound_fraction_match(self):
        for f in _FACTORS:
            with self.subTest(factor=f):
                np.testing.assert_allclose(
                    self.got_yaml[f][:, 1], self.got_py[f][:, 1], rtol=1e-3, atol=1e-9
                )

    def test_n_ads_match(self):
        for f in _FACTORS:
            with self.subTest(factor=f):
                np.testing.assert_allclose(
                    self.got_yaml[f][:, 2], self.got_py[f][:, 2], rtol=1e-3, atol=1e-30
                )


class TestLoadYamlSweepControl(unittest.TestCase):
    """Unit tests for the sweep-control fields added to _load_system_vars_yaml.

    All tests call _load_system_vars_yaml directly; no sweep computation runs.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = Path(tempfile.mkdtemp())
        cls.base = cli._load_system_vars_yaml(
            _write_temp_yaml(cls.tmp_dir, _YAML_MINIMAL)
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    # ── Defaults when sweep keys are absent ───────────────────────────────────

    def test_sweep_defaults_no_keys(self):
        b = self.base
        self.assertEqual(b["n_pts_1D"], 50)
        self.assertEqual(b["n_pts_2D"], 20)
        self.assertEqual(b["polymer_models"], ["gaussian", "Flory-exact"])
        self.assertEqual(b["target_sigma_R"], [])
        self.assertEqual(b["target_sigma_R_labels"], [])
        self.assertEqual(b["codependent_receptors"], {})

    # ── Unit conversion for sigma_R_min / sigma_R_max ─────────────────────────

    def test_sigma_r_min_converted(self):
        v = cli._load_system_vars_yaml(
            _write_temp_yaml(self.tmp_dir, _YAML_MINIMAL + "sigma_R_min_per_um2: 5.0\n")
        )
        self.assertTrue(np.isclose(v["sigma_R_min"], 5.0 / um2))

    def test_sigma_r_max_converted(self):
        v = cli._load_system_vars_yaml(
            _write_temp_yaml(self.tmp_dir, _YAML_MINIMAL + "sigma_R_max_per_um2: 500.0\n")
        )
        self.assertTrue(np.isclose(v["sigma_R_max"], 500.0 / um2))

    # ── Scalar sweep parameters ───────────────────────────────────────────────

    def test_n_pts_1d_parsed(self):
        v = cli._load_system_vars_yaml(
            _write_temp_yaml(self.tmp_dir, _YAML_MINIMAL + "n_pts_1D: 7\n")
        )
        self.assertEqual(v["n_pts_1D"], 7)

    def test_n_pts_2d_parsed(self):
        v = cli._load_system_vars_yaml(
            _write_temp_yaml(self.tmp_dir, _YAML_MINIMAL + "n_pts_2D: 8\n")
        )
        self.assertEqual(v["n_pts_2D"], 8)

    def test_polymer_models_single(self):
        v = cli._load_system_vars_yaml(
            _write_temp_yaml(self.tmp_dir, _YAML_MINIMAL + "polymer_models:\n  - gaussian\n")
        )
        self.assertEqual(v["polymer_models"], ["gaussian"])

    # ── target_sigma_R stays in µm⁻² (no unit conversion) ────────────────────

    def test_target_sigma_r_no_unit_conversion(self):
        v = cli._load_system_vars_yaml(
            _write_temp_yaml(self.tmp_dir, _YAML_MINIMAL + "target_sigma_R: [100.0, 500.0]\n")
        )
        self.assertEqual(v["target_sigma_R"], [100.0, 500.0])

    def test_target_sigma_r_labels_parsed(self):
        v = cli._load_system_vars_yaml(
            _write_temp_yaml(
                self.tmp_dir,
                _YAML_MINIMAL + "target_sigma_R_labels: [healthy, tumour]\n",
            )
        )
        self.assertEqual(v["target_sigma_R_labels"], ["healthy", "tumour"])

    # ── codependent_receptors ─────────────────────────────────────────────────

    def test_codependent_default_empty(self):
        self.assertEqual(self.base["codependent_receptors"], {})

    def test_codependent_parsed(self):
        yaml_2rec = (
            "R_NP_nm: 35.0\nN_ligands: 80\n"
            "PEG_monomer_size_nm: 0.28\nPEG_kuhn_length_nm: 0.76\n"
            "PEG_ligand_MW_g_per_mol: 3400.0\nPEG_short_MW_g_per_mol: 2000.0\n"
            "PEG_short_to_ligand_ratio: 11.4\nKD_nM: 150.0\n"
            "binder_linear_size_nm: 3.5\nnonspec_interaction_kT: 0.0\n"
            "receptors:\n  - name: recA\n  - name: recB\n"
            "ligands:\n  - name: PEG2K\n    type: inert\n"
            "  - name: lig_A\n    type: binding\n    receptor: recA\n"
            "  - name: lig_B\n    type: binding\n    receptor: recB\n"
            "codependent_receptors:\n"
            "  - secondary: recB\n    primary: recA\n    ratio: 0.5\n"
        )
        v = cli._load_system_vars_yaml(_write_temp_yaml(self.tmp_dir, yaml_2rec))
        self.assertEqual(v["codependent_receptors"], {"recB": ("recA", 0.5)})

    def test_codependent_secondary_not_in_receptors_raises(self):
        yaml_bad = (
            "R_NP_nm: 35.0\nN_ligands: 80\n"
            "PEG_monomer_size_nm: 0.28\nPEG_kuhn_length_nm: 0.76\n"
            "PEG_ligand_MW_g_per_mol: 3400.0\nPEG_short_MW_g_per_mol: 2000.0\n"
            "PEG_short_to_ligand_ratio: 11.4\nKD_nM: 150.0\n"
            "binder_linear_size_nm: 3.5\nnonspec_interaction_kT: 0.0\n"
            "receptors:\n  - name: recA\n  - name: recB\n"
            "ligands:\n  - name: PEG2K\n    type: inert\n"
            "  - name: lig_A\n    type: binding\n    receptor: recA\n"
            "  - name: lig_B\n    type: binding\n    receptor: recB\n"
            "codependent_receptors:\n"
            "  - secondary: ghost\n    primary: recA\n    ratio: 0.5\n"
        )
        with self.assertRaises(ValueError):
            cli._load_system_vars_yaml(_write_temp_yaml(self.tmp_dir, yaml_bad))

    def test_codependent_primary_not_in_receptors_raises(self):
        yaml_bad = (
            "R_NP_nm: 35.0\nN_ligands: 80\n"
            "PEG_monomer_size_nm: 0.28\nPEG_kuhn_length_nm: 0.76\n"
            "PEG_ligand_MW_g_per_mol: 3400.0\nPEG_short_MW_g_per_mol: 2000.0\n"
            "PEG_short_to_ligand_ratio: 11.4\nKD_nM: 150.0\n"
            "binder_linear_size_nm: 3.5\nnonspec_interaction_kT: 0.0\n"
            "receptors:\n  - name: recA\n  - name: recB\n"
            "ligands:\n  - name: PEG2K\n    type: inert\n"
            "  - name: lig_A\n    type: binding\n    receptor: recA\n"
            "  - name: lig_B\n    type: binding\n    receptor: recB\n"
            "codependent_receptors:\n"
            "  - secondary: recB\n    primary: ghost\n    ratio: 0.5\n"
        )
        with self.assertRaises(ValueError):
            cli._load_system_vars_yaml(_write_temp_yaml(self.tmp_dir, yaml_bad))


class TestLoadYamlPhysicalParams(unittest.TestCase):
    """Unit tests for optional KD_nM, optional receptors/ligands, and the physical
    parameter derivation for cell_conc / NP_conc added to _load_system_vars_yaml."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    # ── KD_nM optional ────────────────────────────────────────────────────────

    def test_kd_optional_inert_only(self):
        """YAML without KD_nM but with only inert ligands must not raise."""
        yaml_no_kd = (
            "R_NP_nm: 35.0\nN_ligands: 150\n"
            "PEG_monomer_size_nm: 0.28\nPEG_kuhn_length_nm: 0.76\n"
            "PEG_ligand_MW_g_per_mol: 3400.0\nPEG_short_MW_g_per_mol: 2000.0\n"
            "PEG_short_to_ligand_ratio: 11.4\nbinder_linear_size_nm: 3.5\n"
            "nonspec_interaction_kT: 0.0\n"
            "receptors:\n  - name: default\n"
            "ligands:\n  - name: PEG2K\n    type: inert\n"
        )
        v = cli._load_system_vars_yaml(_write_temp_yaml(self.tmp_dir, yaml_no_kd))
        self.assertIsNotNone(v)

    def test_kd_optional_no_receptors_no_ligands(self):
        """YAML without KD_nM, receptors, or ligands (scan-combinations use case) must not raise."""
        yaml_bare = (
            "R_NP_nm: 35.0\nN_ligands: 150\n"
            "PEG_monomer_size_nm: 0.28\nPEG_kuhn_length_nm: 0.76\n"
            "PEG_ligand_MW_g_per_mol: 3400.0\nPEG_short_MW_g_per_mol: 2000.0\n"
            "PEG_short_to_ligand_ratio: 11.4\nbinder_linear_size_nm: 3.5\n"
            "nonspec_interaction_kT: 0.0\n"
        )
        v = cli._load_system_vars_yaml(_write_temp_yaml(self.tmp_dir, yaml_bare))
        self.assertEqual(v["data_polymers"], {})
        self.assertIsNone(v["receptor"])

    def test_kd_optional_binding_ligand_no_kd_raises(self):
        """Binding ligand with no per-ligand KD_nM and no global KD_nM must raise ValueError."""
        yaml_bad = (
            "R_NP_nm: 35.0\nN_ligands: 150\n"
            "PEG_monomer_size_nm: 0.28\nPEG_kuhn_length_nm: 0.76\n"
            "PEG_ligand_MW_g_per_mol: 3400.0\nPEG_short_MW_g_per_mol: 2000.0\n"
            "PEG_short_to_ligand_ratio: 11.4\nbinder_linear_size_nm: 3.5\n"
            "nonspec_interaction_kT: 0.0\n"
            "receptors:\n  - name: default\n"
            "ligands:\n  - name: lig\n    type: binding\n    receptor: default\n"
        )
        with self.assertRaises(ValueError):
            cli._load_system_vars_yaml(_write_temp_yaml(self.tmp_dir, yaml_bad))

    # ── Physical derivation for in-vivo ──────────────────────────────────────

    def test_cell_conc_derived_from_biology(self):
        """cell_conc derived from N_lympho, T_cell_fraction, V_spleen_mm3."""
        from units import mL
        yaml_bio = (
            "R_NP_nm: 35.0\nN_ligands: 150\n"
            "PEG_monomer_size_nm: 0.28\nPEG_kuhn_length_nm: 0.76\n"
            "PEG_ligand_MW_g_per_mol: 3400.0\nPEG_short_MW_g_per_mol: 2000.0\n"
            "PEG_short_to_ligand_ratio: 11.4\nbinder_linear_size_nm: 3.5\n"
            "nonspec_interaction_kT: 0.0\n"
            "N_lympho: 7.5e7\nT_cell_fraction: 0.25\nV_spleen_mm3: 100.0\n"
        )
        v = cli._load_system_vars_yaml(_write_temp_yaml(self.tmp_dir, yaml_bio))
        np.testing.assert_allclose(v["cell_conc"], 1.875e8 / mL, rtol=1e-10)

    def test_np_conc_derived_from_dosing(self):
        """NP_conc derived from Npdosing_per_mL, Vdosing_mL, fTzone, VTzone_mL."""
        from units import mL
        yaml_dose = (
            "R_NP_nm: 35.0\nN_ligands: 150\n"
            "PEG_monomer_size_nm: 0.28\nPEG_kuhn_length_nm: 0.76\n"
            "PEG_ligand_MW_g_per_mol: 3400.0\nPEG_short_MW_g_per_mol: 2000.0\n"
            "PEG_short_to_ligand_ratio: 11.4\nbinder_linear_size_nm: 3.5\n"
            "nonspec_interaction_kT: 0.0\n"
            "Npdosing_per_mL: 8.0e12\nVdosing_mL: 0.1\nfTzone: 0.1\nVTzone_mL: 0.042\n"
        )
        v = cli._load_system_vars_yaml(_write_temp_yaml(self.tmp_dir, yaml_dose))
        expected = 8e12 * 0.1 * 0.1 / 0.042 / mL
        np.testing.assert_allclose(v["NP_conc"], expected, rtol=1e-10)

    def test_np_conc_override_wins(self):
        """NP_conc_per_mL direct override takes precedence over derived value."""
        from units import mL
        yaml_override = (
            "R_NP_nm: 35.0\nN_ligands: 150\n"
            "PEG_monomer_size_nm: 0.28\nPEG_kuhn_length_nm: 0.76\n"
            "PEG_ligand_MW_g_per_mol: 3400.0\nPEG_short_MW_g_per_mol: 2000.0\n"
            "PEG_short_to_ligand_ratio: 11.4\nbinder_linear_size_nm: 3.5\n"
            "nonspec_interaction_kT: 0.0\n"
            "Npdosing_per_mL: 8.0e12\nVdosing_mL: 0.1\nfTzone: 0.1\nVTzone_mL: 0.042\n"
            "NP_conc_per_mL: 1.0e10\n"
        )
        v = cli._load_system_vars_yaml(_write_temp_yaml(self.tmp_dir, yaml_override))
        np.testing.assert_allclose(v["NP_conc"], 1.0e10 / mL, rtol=1e-10)

    def test_spr_old_key_alias(self):
        """Old key NP_conc_SPR_per_mL is accepted as alias for Npdosing_SPR_per_mL."""
        from units import mL
        yaml_spr = (
            "R_NP_nm: 35.0\nN_ligands: 150\n"
            "PEG_monomer_size_nm: 0.28\nPEG_kuhn_length_nm: 0.76\n"
            "PEG_ligand_MW_g_per_mol: 3400.0\nPEG_short_MW_g_per_mol: 2000.0\n"
            "PEG_short_to_ligand_ratio: 11.4\nbinder_linear_size_nm: 3.5\n"
            "nonspec_interaction_kT: 0.0\n"
            "NP_conc_SPR_per_mL: 4.0e11\n"
        )
        v = cli._load_system_vars_yaml(_write_temp_yaml(self.tmp_dir, yaml_spr))
        np.testing.assert_allclose(v["NP_conc_spr"], 4.0e11 / mL, rtol=1e-10)


class TestScanCombinations_CodependentFromYaml(unittest.TestCase):
    """Functional test: codependent_receptors in YAML reduces a 2-receptor binder to 1D sweep."""

    @classmethod
    def setUpClass(cls):
        cls.saved = _save_sbpm_state()
        cls.tmp   = Path(tempfile.mkdtemp())
        cls.out   = cls.tmp / "out"
        # YAML with codependent R2→R1 and no KD_nM (taken from CSV)
        yaml_content = (
            "R_NP_nm: 35.0\nN_ligands: 150\n"
            "PEG_monomer_size_nm: 0.28\nPEG_kuhn_length_nm: 0.76\n"
            "PEG_ligand_MW_g_per_mol: 3400.0\nPEG_short_MW_g_per_mol: 2000.0\n"
            "PEG_short_to_ligand_ratio: 11.4\nbinder_linear_size_nm: 3.5\n"
            "nonspec_interaction_kT: 0.0\n"
            "codependent_receptors:\n"
            "  - secondary: R2\n    primary: R1\n    ratio: 1.0\n"
            "n_pts_1D: 3\n"
        )
        yaml_path = _write_temp_yaml(cls.tmp, yaml_content)
        # Binder A targets R1 and R2; without codep this would be 2D → with codep, 1D.
        csv_content = "Target,A\nR1,100.0\nR2,200.0\n"
        cli.scan_combinations_cmd(
            csv_path=_write_temp_csv(cls.tmp, csv_content),
            output_dir=cls.out,
            config=yaml_path,
            polymer_model=["gaussian"],
            n_workers=1,
            n_cores_per_run=1,
        )

    @classmethod
    def tearDownClass(cls):
        _restore_sbpm_state(cls.saved)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_1d_output_exists(self):
        self.assertTrue((self.out / "A" / "adsorption_gaussian.dat").exists())

    def test_2d_output_absent(self):
        self.assertFalse((self.out / "A" / "adsorption_2rec_gaussian.dat").exists())


class TestScanCombinations_CodependentBadReceptorName(unittest.TestCase):
    """YAML codependent with a receptor name absent from the CSV must raise BadParameter."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        yaml_content = (
            "R_NP_nm: 35.0\nN_ligands: 150\n"
            "PEG_monomer_size_nm: 0.28\nPEG_kuhn_length_nm: 0.76\n"
            "PEG_ligand_MW_g_per_mol: 3400.0\nPEG_short_MW_g_per_mol: 2000.0\n"
            "PEG_short_to_ligand_ratio: 11.4\nbinder_linear_size_nm: 3.5\n"
            "nonspec_interaction_kT: 0.0\n"
            "codependent_receptors:\n"
            "  - secondary: ghost\n    primary: R1\n    ratio: 1.0\n"
        )
        cls.yaml_path = _write_temp_yaml(cls.tmp, yaml_content)
        cls.csv_content = "Target,A\nR1,100.0\n"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_bad_secondary_raises(self):
        with self.assertRaises(SystemExit):
            cli.scan_combinations_cmd(
                csv_path=_write_temp_csv(self.tmp, self.csv_content),
                output_dir=self.tmp / "out",
                config=self.yaml_path,
                polymer_model=["gaussian"],
                n_workers=1,
                n_cores_per_run=1,
            )


class TestBothModelsSweepParamsViaYaml(unittest.TestCase):
    """YAML sweep fields (n_pts_1D, sigma_R range, polymer_models) propagate correctly
    to the scan_both_polymer_models_cmd output."""

    @classmethod
    def setUpClass(cls):
        mp.dps = 50
        saved = _save_sbpm_state()
        cls.tmp = tempfile.mkdtemp()
        yaml_content = (
            _YAML_MINIMAL
            + "n_pts_1D: 3\n"
            + "polymer_models:\n  - gaussian\n"
            + "sigma_R_min_per_um2: 10.0\n"
            + "sigma_R_max_per_um2: 100.0\n"
        )
        yaml_path = _write_temp_yaml(Path(cls.tmp), yaml_content)
        try:
            cli.scan_both_polymer_models_cmd(
                output_dir=Path(cls.tmp), config=yaml_path
            )
        finally:
            _restore_sbpm_state(saved)
        cls.dat = _load_dat(os.path.join(cls.tmp, "adsorption_gaussian.dat"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_dat_has_3_rows(self):
        self.assertEqual(self.dat.shape[0], 3)

    def test_sigma_r_in_range(self):
        sigma = self.dat[:, 0]
        self.assertTrue(np.all(sigma >= 10.0))
        self.assertTrue(np.all(sigma <= 100.0))

    def test_only_gaussian_output(self):
        self.assertTrue(
            os.path.exists(os.path.join(self.tmp, "adsorption_gaussian.dat"))
        )
        self.assertFalse(
            os.path.exists(os.path.join(self.tmp, "adsorption_Flory_exact.dat"))
        )

    def test_bound_fraction_in_unit_interval(self):
        bf = self.dat[:, 1]
        self.assertTrue(np.all(bf >= 0) and np.all(bf <= 1))


class TestBothModelsCodependentViaYaml(unittest.TestCase):
    """2-receptor YAML with codependent_receptors reduces the sweep to 1D.

    Without codependent_receptors the same YAML would produce a 2D grid and write
    adsorption_2rec_gaussian.dat. With recB as a secondary, only 1 primary axis
    (recA) remains → adsorption_gaussian.dat is written instead.
    """

    @classmethod
    def setUpClass(cls):
        mp.dps = 50
        saved = _save_sbpm_state()
        cls.tmp = tempfile.mkdtemp()
        yaml_path = _write_temp_yaml(Path(cls.tmp), _YAML_TWO_RECS_CODEP)
        try:
            cli.scan_both_polymer_models_cmd(
                output_dir=Path(cls.tmp), config=yaml_path
            )
        finally:
            _restore_sbpm_state(saved)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_1d_output_exists(self):
        self.assertTrue(
            os.path.exists(os.path.join(self.tmp, "adsorption_gaussian.dat"))
        )

    def test_2d_output_absent(self):
        self.assertFalse(
            os.path.exists(os.path.join(self.tmp, "adsorption_2rec_gaussian.dat"))
        )

    def test_bound_fraction_in_unit_interval(self):
        dat = _load_dat(os.path.join(self.tmp, "adsorption_gaussian.dat"))
        bf = dat[:, 1]
        self.assertTrue(np.all(bf >= 0) and np.all(bf <= 1))


class TestBothModelsCliOverrides(unittest.TestCase):
    """CLI arguments override module defaults for scan_both_polymer_models_cmd."""

    @classmethod
    def setUpClass(cls):
        mp.dps = 50
        saved = _save_sbpm_state()
        cls.tmp = tempfile.mkdtemp()
        try:
            cli.scan_both_polymer_models_cmd(
                output_dir=Path(cls.tmp),
                n_pts_1d=3,
                sigma_r_min=10.0,
                sigma_r_max=100.0,
                polymer_model=["gaussian"],
            )
        finally:
            _restore_sbpm_state(saved)
        cls.dat = _load_dat(os.path.join(cls.tmp, "adsorption_gaussian.dat"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_dat_has_3_rows(self):
        self.assertEqual(self.dat.shape[0], 3)

    def test_sigma_r_in_range(self):
        sigma = self.dat[:, 0]
        self.assertTrue(np.all(sigma >= 10.0))
        self.assertTrue(np.all(sigma <= 100.0))

    def test_only_gaussian_output(self):
        self.assertTrue(
            os.path.exists(os.path.join(self.tmp, "adsorption_gaussian.dat"))
        )
        self.assertFalse(
            os.path.exists(os.path.join(self.tmp, "adsorption_Flory_exact.dat"))
        )

    def test_codependent_malformed_format_raises(self):
        with self.assertRaises(typer.BadParameter):
            cli.scan_both_polymer_models_cmd(
                output_dir=Path(self.tmp),
                codependent=["missing_ratio"],
            )

    def test_labels_length_mismatch_raises(self):
        saved = _save_sbpm_state()
        _sbpm.target_sigma_R        = [100.0]
        _sbpm.target_sigma_R_labels = ["a", "b"]
        try:
            with self.assertRaises(typer.BadParameter):
                cli.scan_both_polymer_models_cmd(output_dir=Path(self.tmp))
        finally:
            _restore_sbpm_state(saved)


# ── scan-combinations tests ──────────────────────────────────────────────────

class TestScanCombinationsUnit(unittest.TestCase):
    """Unit tests for _read_binders_csv and _build_data_polymers_for_run.
    No sweeps are run; no _sbpm state management needed.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.np_params = {
            "sigma_L": 0.01, "sigma_P2K": 0.114, "amono": 0.28, "akuhn": 0.76,
            "NmonoL": 77, "NmonoS": 45, "binder_size": 3.5,
        }

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # _read_binders_csv tests

    def test_read_counts_and_values(self):
        csv_path = _write_temp_csv(self.tmp, "Target,A,B,C\nR1,1.0,2.0,\nR2,,3.0,4.0\n")
        result = cli._read_binders_csv(csv_path)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["A"], {"R1": 1.0})
        self.assertEqual(result["B"], {"R1": 2.0, "R2": 3.0})
        self.assertEqual(result["C"], {"R2": 4.0})

    def test_empty_binder_dropped(self):
        csv_path = _write_temp_csv(self.tmp, "Target,A,B\nR1,1.0,\nR2,2.0,\n")
        result = cli._read_binders_csv(csv_path)
        self.assertIn("A", result)
        self.assertNotIn("B", result)

    def test_duplicate_id_renamed(self):
        csv_path = _write_temp_csv(self.tmp, "Target,A,A\nR1,1.0,2.0\n")
        result = cli._read_binders_csv(csv_path)
        self.assertIn("A", result)
        self.assertIn("A_a", result)
        self.assertAlmostEqual(result["A"]["R1"], 1.0)
        self.assertAlmostEqual(result["A_a"]["R1"], 2.0)

    # _build_data_polymers_for_run tests

    def test_single_binder_structure(self):
        binders_data = {"lig1": {"R1": 150.0}}
        data_polymers, _ = cli._build_data_polymers_for_run(
            ["lig1"], binders_data, self.np_params
        )
        self.assertIn("short", data_polymers)
        self.assertIn("lig_lig1_R1", data_polymers)
        self.assertIn("K_bind_0", data_polymers["lig_lig1_R1"])
        self.assertEqual(data_polymers["lig_lig1_R1"]["receptor"]["name"], "R1")

    def test_shared_receptor_same_object(self):
        binders_data = {"A": {"R1": 100.0}, "B": {"R1": 200.0}}
        data_polymers, _ = cli._build_data_polymers_for_run(
            ["A", "B"], binders_data, self.np_params
        )
        self.assertIs(
            data_polymers["lig_A_R1"]["receptor"],
            data_polymers["lig_B_R1"]["receptor"],
        )

    def test_pair_sigma_split(self):
        binders_data = {"A": {"R1": 100.0}, "B": {"R2": 200.0}}
        data_polymers, _ = cli._build_data_polymers_for_run(
            ["A", "B"], binders_data, self.np_params, ligand_ratio=0.5
        )
        expected = self.np_params["sigma_L"] * 0.5
        self.assertAlmostEqual(data_polymers["lig_A_R1"]["sigma"], expected)
        self.assertAlmostEqual(data_polymers["lig_B_R2"]["sigma"], expected)


class TestScanCombinations_BasicOutput(unittest.TestCase):
    """Functional tests: single-binder and pair output files are created correctly."""

    @classmethod
    def setUpClass(cls):
        cls.saved = _save_sbpm_state()
        cls.tmp   = Path(tempfile.mkdtemp())
        cls.out   = cls.tmp / "out"
        csv_content = "Target,lig1,lig2\nR1,150.0,300.0\n"
        cli.scan_combinations_cmd(
            csv_path=_write_temp_csv(cls.tmp, csv_content),
            output_dir=cls.out,
            n_pts_1d=3,
            polymer_model=["gaussian"],
            n_cores_per_run=1,
        )

    @classmethod
    def tearDownClass(cls):
        _restore_sbpm_state(cls.saved)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_single_binder_dirs_exist(self):
        self.assertTrue((self.out / "lig1").is_dir())
        self.assertTrue((self.out / "lig2").is_dir())

    def test_single_binder_dat_rows(self):
        dat = _load_dat(self.out / "lig1" / "adsorption_gaussian.dat")
        self.assertEqual(dat.shape[0], 3)

    def test_pair_bound_fraction_valid(self):
        dat = _load_dat(self.out / "lig1+lig2" / "adsorption_gaussian.dat")
        bound_fraction = dat[:, 1]
        self.assertTrue(np.all(bound_fraction >= 0.0))
        self.assertTrue(np.all(bound_fraction <= 1.0))


class TestScanCombinations_SkipLogic(unittest.TestCase):
    """Functional test: runs with > 2 primary receptor axes are skipped."""

    @classmethod
    def setUpClass(cls):
        cls.saved = _save_sbpm_state()
        cls.tmp   = Path(tempfile.mkdtemp())
        cls.out   = cls.tmp / "out"
        # A targets R1 + R2; B targets R3 + R4.
        # With codependent R2->R1: single A reduces to 1 primary, single B has 2 primaries,
        # but the pair A+B has primaries R1, R3, R4 (3 axes) → skipped.
        csv_content = (
            "Target,A,B\n"
            "R1,100.0,\n"
            "R2,200.0,\n"
            "R3,,300.0\n"
            "R4,,400.0\n"
        )
        cli.scan_combinations_cmd(
            csv_path=_write_temp_csv(cls.tmp, csv_content),
            output_dir=cls.out,
            codependent=["R2:R1:1.0"],
            n_pts_1d=3,
            n_pts_2d=3,
            polymer_model=["gaussian"],
            n_cores_per_run=1,
        )

    @classmethod
    def tearDownClass(cls):
        _restore_sbpm_state(cls.saved)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_single_A_ran(self):
        self.assertTrue((self.out / "A" / "adsorption_gaussian.dat").exists())

    def test_pair_AB_skipped(self):
        self.assertFalse((self.out / "A+B").exists())


class TestScanCombinations_CodepReduces2Recs_To1D(unittest.TestCase):
    """Functional test: codependent receptor reduces a 2-receptor run to a 1D sweep."""

    @classmethod
    def setUpClass(cls):
        cls.saved = _save_sbpm_state()
        cls.tmp   = Path(tempfile.mkdtemp())
        cls.out   = cls.tmp / "out"
        # Binder A targets R1 and R2. Without codep this would be 2D; with it, 1D.
        csv_content = "Target,A\nR1,100.0\nR2,200.0\n"
        cli.scan_combinations_cmd(
            csv_path=_write_temp_csv(cls.tmp, csv_content),
            output_dir=cls.out,
            codependent=["R2:R1:1.0"],
            n_pts_1d=3,
            polymer_model=["gaussian"],
            n_cores_per_run=1,
        )

    @classmethod
    def tearDownClass(cls):
        _restore_sbpm_state(cls.saved)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_1d_file_present(self):
        self.assertTrue((self.out / "A" / "adsorption_gaussian.dat").exists())

    def test_2d_file_absent(self):
        self.assertFalse((self.out / "A" / "adsorption_2rec_gaussian.dat").exists())


class TestScanCombinations_ParallelEquality(unittest.TestCase):
    """n_workers=2 must produce bit-for-bit identical .dat output as n_workers=1."""

    @classmethod
    def setUpClass(cls):
        cls.saved = _save_sbpm_state()
        cls.tmp   = Path(tempfile.mkdtemp())
        csv_content = "Target,lig1,lig2\nR1,150.0,300.0\n"
        cls.csv_path = _write_temp_csv(cls.tmp, csv_content)
        cls.out_serial   = cls.tmp / "serial"
        cls.out_parallel = cls.tmp / "parallel"

        cli.scan_combinations_cmd(
            csv_path=cls.csv_path,
            output_dir=cls.out_serial,
            n_pts_1d=3,
            polymer_model=["gaussian"],
            n_workers=1,
            n_cores_per_run=1,
        )
        _restore_sbpm_state(cls.saved)

        saved2 = _save_sbpm_state()
        cli.scan_combinations_cmd(
            csv_path=cls.csv_path,
            output_dir=cls.out_parallel,
            n_pts_1d=3,
            polymer_model=["gaussian"],
            n_workers=2,
            n_cores_per_run=1,
        )
        _restore_sbpm_state(saved2)

    @classmethod
    def tearDownClass(cls):
        _restore_sbpm_state(cls.saved)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _check_subdir(self, subdir):
        s = _load_dat(self.out_serial   / subdir / "adsorption_gaussian.dat")
        p = _load_dat(self.out_parallel / subdir / "adsorption_gaussian.dat")
        np.testing.assert_array_almost_equal(
            s, p, decimal=10,
            err_msg=f"serial vs parallel mismatch in {subdir}/adsorption_gaussian.dat",
        )

    def test_lig1_identical(self):
        self._check_subdir("lig1")

    def test_lig2_identical(self):
        self._check_subdir("lig2")

    def test_pair_identical(self):
        self._check_subdir("lig1+lig2")


class TestScanCombinations_NcoresParallelEquality(unittest.TestCase):
    """n_cores_per_run=2 (parallel K_bind) must produce the same .dat output as n_cores_per_run=1.

    CSV has lig1→R1 only and lig2→R2 only, so the pair run exercises sweep_2axes
    (2D K_bind grid), which is the code path that _parallel_calculate_k_bind accelerates.
    """

    @classmethod
    def setUpClass(cls):
        cls.saved = _save_sbpm_state()
        cls.tmp   = Path(tempfile.mkdtemp())
        csv_content = "Target,lig1,lig2\nR1,150.0,\nR2,,300.0\n"
        cls.csv_path = _write_temp_csv(cls.tmp, csv_content)
        cls.out_serial   = cls.tmp / "serial"
        cls.out_parallel = cls.tmp / "parallel"

        cli.scan_combinations_cmd(
            csv_path=cls.csv_path,
            output_dir=cls.out_serial,
            n_pts_1d=3,
            n_pts_2d=3,
            polymer_model=["gaussian"],
            n_workers=1,
            n_cores_per_run=1,
        )
        _restore_sbpm_state(cls.saved)

        saved2 = _save_sbpm_state()
        cli.scan_combinations_cmd(
            csv_path=cls.csv_path,
            output_dir=cls.out_parallel,
            n_pts_1d=3,
            n_pts_2d=3,
            polymer_model=["gaussian"],
            n_workers=1,
            n_cores_per_run=2,
        )
        _restore_sbpm_state(saved2)

    @classmethod
    def tearDownClass(cls):
        _restore_sbpm_state(cls.saved)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _check_subdir(self, subdir, fname):
        s = _load_dat(self.out_serial   / subdir / fname)
        p = _load_dat(self.out_parallel / subdir / fname)
        np.testing.assert_array_almost_equal(
            s, p, decimal=10,
            err_msg=f"serial vs parallel K_bind mismatch in {subdir}/{fname}",
        )

    def test_lig1_identical(self):
        self._check_subdir("lig1", "adsorption_gaussian.dat")

    def test_lig2_identical(self):
        self._check_subdir("lig2", "adsorption_gaussian.dat")

    def test_pair_identical(self):
        self._check_subdir("lig1+lig2", "adsorption_2rec_gaussian.dat")


class TestScanCombinations_JobRange(unittest.TestCase):
    """--job-range 1:2 executes only the first two non-skipped runs."""

    @classmethod
    def setUpClass(cls):
        cls.saved = _save_sbpm_state()
        cls.tmp   = Path(tempfile.mkdtemp())
        cls.out   = cls.tmp / "out"
        # 3 binders all targeting R1 → 3 single + 3 pair runs = 6 non-skipped total.
        # Sorted run order: lig1, lig2, lig3, lig1+lig2, lig1+lig3, lig2+lig3.
        # job_range="1:2" → only lig1 and lig2 execute.
        csv_content = "Target,lig1,lig2,lig3\nR1,150.0,300.0,450.0\n"
        cli.scan_combinations_cmd(
            csv_path=_write_temp_csv(cls.tmp, csv_content),
            output_dir=cls.out,
            n_pts_1d=3,
            polymer_model=["gaussian"],
            n_workers=1,
            n_cores_per_run=1,
            job_range="1:2",
        )

    @classmethod
    def tearDownClass(cls):
        _restore_sbpm_state(cls.saved)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_lig1_ran(self):
        self.assertTrue((self.out / "lig1" / "adsorption_gaussian.dat").exists())

    def test_lig2_ran(self):
        self.assertTrue((self.out / "lig2" / "adsorption_gaussian.dat").exists())

    def test_lig3_absent(self):
        self.assertFalse((self.out / "lig3").exists())

    def test_pair_lig1_lig2_absent(self):
        self.assertFalse((self.out / "lig1+lig2").exists())

    def test_pair_lig1_lig3_absent(self):
        self.assertFalse((self.out / "lig1+lig3").exists())

    def test_pair_lig2_lig3_absent(self):
        self.assertFalse((self.out / "lig2+lig3").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
