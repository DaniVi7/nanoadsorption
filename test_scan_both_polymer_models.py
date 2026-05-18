"""Tests for scan_both_polymer_models.py

Imports scan_both_polymer_models as sbpm (safe because all execution is
guarded under `if __name__ == "__main__":`).

Module-level config vars are overridden at import time for speed.
Synthetic data_polymers dicts are injected into sbpm where needed.
"""
import sys
import os
import copy
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from units import nm, um, um2, nM, g, mL, kT
from adsorption import MultivalentBinding, Nmonomers

import scan_both_polymer_models as sbpm

# ── Speed overrides (must happen before any sweep is called) ─────────────────
sbpm.n_pts_1D    = 5
sbpm.n_pts_2D    = 3
sbpm.sigma_R_min = 10.0  / um2
sbpm.sigma_R_max = 200.0 / um2

# ── Synthetic 2-receptor data_polymers ───────────────────────────────────────
_amono      = 0.28 * nm
_akuhn      = 0.76 * nm
_NmonoLig   = Nmonomers(3400 * g)
_NmonoShort = Nmonomers(2000 * g)
_R_NP       = sbpm.R_NP            # 35 nm, from system_variables_invivo_multi
_sigma_L    = 80 / (4 * np.pi * _R_NP**2)
_sigma_P2K  = _sigma_L * 11.4
_KD         = 10000.0 * nM

# Two distinct receptor dicts with the same parameters → symmetry test is valid.
_rec_A = {"name": "receptor_A", "sigma_R": 50.0 / um2}
_rec_B = {"name": "receptor_B", "sigma_R": 50.0 / um2}

dp_2rec = {
    "short":    {"N": _NmonoShort, "a": _amono, "sigma": _sigma_P2K,
                 "name": "PEG2K",    "akuhn": _akuhn},
    "ligands":  {"N": _NmonoLig,   "a": _amono, "sigma": _sigma_L,
                 "name": "ligands",  "akuhn": _akuhn,
                 "K_bind_0": _KD**(-1), "receptor": _rec_A},
    "ligands2": {"N": _NmonoLig,   "a": _amono, "sigma": _sigma_L,
                 "name": "ligands2", "akuhn": _akuhn,
                 "K_bind_0": _KD**(-1), "receptor": _rec_B},
}

# 3-receptor dp: dp_2rec + a third ligand targeting receptor_C
# Used in TestSweep2AxesWithCodependent.
_rec_C = {"name": "receptor_C", "sigma_R": 50.0 / um2}
dp_3rec = dict(dp_2rec)
dp_3rec["ligands3"] = {
    "N": _NmonoLig, "a": _amono, "sigma": _sigma_L, "akuhn": _akuhn,
    "name": "ligands3", "K_bind_0": _KD**(-1), "receptor": _rec_C,
}

# dp with ONE list-form ligand targeting two receptors.
# Used in TestSweep1AxisListFormLigand.
dp_list_form = {
    "short": {"N": _NmonoShort, "a": _amono, "sigma": _sigma_P2K,
              "name": "PEG2K", "akuhn": _akuhn},
    "lig": {
        "N": _NmonoLig, "a": _amono, "sigma": _sigma_L, "akuhn": _akuhn,
        "receptor": [
            {"name": "rec_X", "sigma_R": 50.0 / um2, "K_bind_0": _KD**(-1)},
            {"name": "rec_Y", "sigma_R": 50.0 / um2, "K_bind_0": _KD**(-1)},
        ],
    },
}


# ────────────────────────────────────────────────────────────────────────────
class TestDetectReceptors(unittest.TestCase):
    """Pure unit tests for _detect_receptors — no MultivalentBinding needed."""

    def test_single_shared_receptor(self):
        rec = {"name": "shared", "sigma_R": 100 / um2}
        dp = {
            "lig_a": {"K_bind_0": 1.0, "receptor": rec},
            "lig_b": {"K_bind_0": 1.0, "receptor": rec},
        }
        result = sbpm._detect_receptors(dp)
        self.assertEqual(len(result), 1)
        self.assertIn("shared", result)

    def test_two_distinct_receptors(self):
        rec_a = {"name": "receptor_A", "sigma_R": 100 / um2}
        rec_b = {"name": "receptor_B", "sigma_R": 200 / um2}
        dp = {
            "lig_a": {"K_bind_0": 1.0, "receptor": rec_a},
            "lig_b": {"K_bind_0": 1.0, "receptor": rec_b},
        }
        result = sbpm._detect_receptors(dp)
        self.assertEqual(len(result), 2)
        self.assertIn("receptor_A", result)
        self.assertIn("receptor_B", result)

    def test_inert_polymer_not_counted(self):
        """Polymer without K_bind_0 must not be counted."""
        rec = {"name": "rec", "sigma_R": 100 / um2}
        dp = {
            "inert":  {"receptor": rec},               # no K_bind_0 → skip
            "ligand": {"K_bind_0": 1.0, "receptor": rec},
        }
        result = sbpm._detect_receptors(dp)
        self.assertEqual(len(result), 1)

    def test_empty_dp_returns_empty(self):
        dp = {"inert": {"N": 10, "a": 0.3}}
        result = sbpm._detect_receptors(dp)
        self.assertEqual(result, {})


# ────────────────────────────────────────────────────────────────────────────
class TestDetectReceptorsListForm(unittest.TestCase):
    """_detect_receptors with list-form receptor entries."""

    def test_list_form_two_receptors_detected(self):
        """List-form ligand with 2 distinct receptor dicts → 2 receptors detected."""
        dp = {
            "lig": {
                "N": 10,
                "receptor": [
                    {"name": "rec_P", "sigma_R": 100 / um2, "K_bind_0": 1.0},
                    {"name": "rec_Q", "sigma_R": 200 / um2, "K_bind_0": 2.0},
                ],
            }
        }
        result = sbpm._detect_receptors(dp)
        self.assertEqual(len(result), 2)
        self.assertIn("rec_P", result)
        self.assertIn("rec_Q", result)

    def test_mixed_dict_and_list_form(self):
        """dp with one dict-form ligand + one list-form ligand → 3 distinct receptors."""
        rec_single = {"name": "rec_single", "sigma_R": 50 / um2}
        dp = {
            "lig_dict": {"K_bind_0": 1.0, "receptor": rec_single},
            "lig_list": {
                "N": 10,
                "receptor": [
                    {"name": "rec_P", "sigma_R": 100 / um2, "K_bind_0": 1.0},
                    {"name": "rec_Q", "sigma_R": 200 / um2, "K_bind_0": 2.0},
                ],
            },
        }
        result = sbpm._detect_receptors(dp)
        self.assertEqual(len(result), 3)
        self.assertIn("rec_single", result)
        self.assertIn("rec_P", result)
        self.assertIn("rec_Q", result)


# ────────────────────────────────────────────────────────────────────────────
class TestExpandMultiReceptor(unittest.TestCase):
    """Unit tests for _expand_multireceptor_ligands."""

    def test_dict_form_passes_through_unchanged(self):
        rec = {"name": "rec_A", "sigma_R": 100 / um2}
        dp = {"lig": {"N": 10, "K_bind_0": 1.0, "receptor": rec}}
        result = sbpm._expand_multireceptor_ligands(dp)
        self.assertIn("lig", result)
        self.assertNotIn("lig_rec_A", result)
        self.assertIs(result["lig"], dp["lig"])

    def test_list_form_expands_to_two_virtual_ligands(self):
        dp = {
            "lig": {
                "N": 10, "a": 0.3, "sigma": 0.01,
                "receptor": [
                    {"name": "rec_A", "sigma_R": 100 / um2, "K_bind_0": 1.0},
                    {"name": "rec_B", "sigma_R": 200 / um2, "K_bind_0": 2.0},
                ],
            }
        }
        result = sbpm._expand_multireceptor_ligands(dp)
        self.assertIn("lig_rec_A", result)
        self.assertIn("lig_rec_B", result)
        self.assertNotIn("lig", result)

    def test_virtual_ligand_has_correct_k_bind(self):
        dp = {
            "lig": {
                "N": 10, "a": 0.3, "sigma": 0.01,
                "receptor": [
                    {"name": "rec_A", "sigma_R": 100 / um2, "K_bind_0": 1.5},
                    {"name": "rec_B", "sigma_R": 200 / um2, "K_bind_0": 3.0},
                ],
            }
        }
        result = sbpm._expand_multireceptor_ligands(dp)
        self.assertAlmostEqual(result["lig_rec_A"]["K_bind_0"], 1.5)
        self.assertAlmostEqual(result["lig_rec_B"]["K_bind_0"], 3.0)

    def test_virtual_ligand_receptor_has_no_k_bind(self):
        dp = {
            "lig": {
                "N": 10,
                "receptor": [
                    {"name": "rec_A", "sigma_R": 100 / um2, "K_bind_0": 1.5},
                ],
            }
        }
        result = sbpm._expand_multireceptor_ligands(dp)
        self.assertNotIn("K_bind_0", result["lig_rec_A"]["receptor"])


# ────────────────────────────────────────────────────────────────────────────
class TestExpandMultiReceptorExtended(unittest.TestCase):
    """Additional coverage for _expand_multireceptor_ligands."""

    def test_original_dp_not_mutated_by_list_expansion(self):
        dp = {
            "lig": {
                "N": 10,
                "receptor": [
                    {"name": "rec_A", "sigma_R": 100 / um2, "K_bind_0": 1.0},
                    {"name": "rec_B", "sigma_R": 200 / um2, "K_bind_0": 2.0},
                ],
            }
        }
        original_receptor = dp["lig"]["receptor"]
        sbpm._expand_multireceptor_ligands(dp)
        self.assertIs(dp["lig"]["receptor"], original_receptor,
                      "Original dp receptor field was replaced by expansion.")
        self.assertIsInstance(dp["lig"]["receptor"], list,
                              "Original dp receptor field must still be a list.")

    def test_polymer_params_preserved_in_virtual_ligands(self):
        dp = {
            "lig": {
                "N": 42, "a": 0.3, "sigma": 0.05, "akuhn": 0.7,
                "receptor": [
                    {"name": "rec_A", "sigma_R": 100 / um2, "K_bind_0": 1.0},
                    {"name": "rec_B", "sigma_R": 200 / um2, "K_bind_0": 2.0},
                ],
            }
        }
        result = sbpm._expand_multireceptor_ligands(dp)
        for virt_key in ("lig_rec_A", "lig_rec_B"):
            self.assertEqual(result[virt_key]["N"],     42)
            self.assertAlmostEqual(result[virt_key]["a"],     0.3)
            self.assertAlmostEqual(result[virt_key]["sigma"],  0.05)
            self.assertAlmostEqual(result[virt_key]["akuhn"], 0.7)

    def test_two_list_form_ligands_in_same_dp(self):
        """Two list-form entries in the same dp both get expanded."""
        dp = {
            "lig1": {
                "N": 10,
                "receptor": [
                    {"name": "rec_A", "sigma_R": 100 / um2, "K_bind_0": 1.0},
                    {"name": "rec_B", "sigma_R": 200 / um2, "K_bind_0": 2.0},
                ],
            },
            "lig2": {
                "N": 20,
                "receptor": [
                    {"name": "rec_C", "sigma_R": 300 / um2, "K_bind_0": 3.0},
                    {"name": "rec_D", "sigma_R": 400 / um2, "K_bind_0": 4.0},
                ],
            },
        }
        result = sbpm._expand_multireceptor_ligands(dp)
        self.assertEqual(len(result), 4)
        for key in ("lig1", "lig2"):
            self.assertNotIn(key, result)
        for key in ("lig1_rec_A", "lig1_rec_B", "lig2_rec_C", "lig2_rec_D"):
            self.assertIn(key, result)


# ────────────────────────────────────────────────────────────────────────────
class TestMakeSystem(unittest.TestCase):

    def test_deepcopy_isolation(self):
        """Modifying rec_refs must not propagate into the original data_polymers."""
        _, rec_refs = sbpm._make_system("gaussian")
        for rec in rec_refs.values():
            rec["sigma_R"] = 999.0 / um2
        for poly in sbpm.data_polymers.values():
            if "receptor" in poly:
                self.assertNotEqual(
                    poly["receptor"].get("sigma_R"), 999.0 / um2,
                    "Original data_polymers receptor was mutated by _make_system.",
                )

    def test_polymer_model_attribute(self):
        system, _ = sbpm._make_system("gaussian")
        self.assertEqual(system.polymer_model, "gaussian")

    def test_dp_src_override_gives_two_receptor_refs(self):
        """dp_src with two distinct receptor dicts → rec_refs has 2 entries."""
        _, rec_refs = sbpm._make_system("gaussian", dp_src=dp_2rec)
        self.assertEqual(len(rec_refs), 2)
        self.assertIn("receptor_A", rec_refs)
        self.assertIn("receptor_B", rec_refs)


# ────────────────────────────────────────────────────────────────────────────
class TestMakeSystemListForm(unittest.TestCase):
    """_make_system with a list-form receptor dp_src."""

    def _make_list_dp(self):
        return {
            "short": {"N": _NmonoShort, "a": _amono, "sigma": _sigma_P2K,
                      "name": "PEG2K", "akuhn": _akuhn},
            "lig": {
                "N": _NmonoLig, "a": _amono, "sigma": _sigma_L, "akuhn": _akuhn,
                "receptor": [
                    {"name": "rec_P", "sigma_R": 50 / um2, "K_bind_0": _KD**(-1)},
                    {"name": "rec_Q", "sigma_R": 50 / um2, "K_bind_0": _KD**(-1)},
                ],
            },
        }

    def test_list_form_dp_expanded_in_make_system(self):
        """List-form dp → expansion → rec_refs contains both receptors."""
        dp_list = self._make_list_dp()
        _, rec_refs = sbpm._make_system("gaussian", dp_src=dp_list)
        self.assertEqual(len(rec_refs), 2)
        self.assertIn("rec_P", rec_refs)
        self.assertIn("rec_Q", rec_refs)

    def test_list_form_original_dp_not_mutated_by_make_system(self):
        """deepcopy+expansion inside _make_system must not touch the original dp."""
        dp_list = self._make_list_dp()
        original_rec = dp_list["lig"]["receptor"]
        sbpm._make_system("gaussian", dp_src=dp_list)
        self.assertIs(dp_list["lig"]["receptor"], original_rec,
                      "Original dp_list receptor field was replaced by _make_system.")
        self.assertIsInstance(dp_list["lig"]["receptor"], list)


# ────────────────────────────────────────────────────────────────────────────
class TestResolveSigmaR(unittest.TestCase):
    """Unit tests for _resolve_sigma_R."""

    def test_no_codependents_sets_only_primary(self):
        rec_refs = {"rec_A": {"name": "rec_A"}, "rec_B": {"name": "rec_B"}}
        saved = sbpm.codependent_receptors
        try:
            sbpm.codependent_receptors = {}
            sbpm._resolve_sigma_R({"rec_A": 100.0 / um2}, rec_refs)
            self.assertAlmostEqual(rec_refs["rec_A"]["sigma_R"], 100.0 / um2)
            self.assertNotIn("sigma_R", rec_refs["rec_B"])
        finally:
            sbpm.codependent_receptors = saved

    def test_codependent_ratio_applied(self):
        rec_refs = {"rec_A": {"name": "rec_A"}, "rec_B": {"name": "rec_B"}}
        saved = sbpm.codependent_receptors
        try:
            sbpm.codependent_receptors = {"rec_B": ("rec_A", 0.5)}
            sbpm._resolve_sigma_R({"rec_A": 200.0 / um2}, rec_refs)
            self.assertAlmostEqual(rec_refs["rec_A"]["sigma_R"], 200.0 / um2)
            self.assertAlmostEqual(rec_refs["rec_B"]["sigma_R"], 100.0 / um2)
        finally:
            sbpm.codependent_receptors = saved


# ────────────────────────────────────────────────────────────────────────────
class TestResolveSigmaRExtended(unittest.TestCase):
    """Additional coverage for _resolve_sigma_R."""

    def test_multiple_codependents_same_primary(self):
        rec_refs = {
            "A": {"name": "A"},
            "B": {"name": "B"},
            "C": {"name": "C"},
        }
        saved = sbpm.codependent_receptors
        try:
            sbpm.codependent_receptors = {"B": ("A", 2.0), "C": ("A", 0.5)}
            sbpm._resolve_sigma_R({"A": 100.0 / um2}, rec_refs)
            self.assertAlmostEqual(rec_refs["A"]["sigma_R"], 100.0 / um2)
            self.assertAlmostEqual(rec_refs["B"]["sigma_R"], 200.0 / um2)
            self.assertAlmostEqual(rec_refs["C"]["sigma_R"],  50.0 / um2)
        finally:
            sbpm.codependent_receptors = saved

    def test_return_value_contains_primary_and_secondaries(self):
        rec_refs = {"A": {"name": "A"}, "B": {"name": "B"}}
        saved = sbpm.codependent_receptors
        try:
            sbpm.codependent_receptors = {"B": ("A", 0.5)}
            result = sbpm._resolve_sigma_R({"A": 100.0 / um2}, rec_refs)
            self.assertIn("A", result)
            self.assertIn("B", result)
        finally:
            sbpm.codependent_receptors = saved


# ────────────────────────────────────────────────────────────────────────────
class TestAxisLabel(unittest.TestCase):
    """Unit tests for _axis_label."""

    def test_no_codependents_single_line(self):
        saved = sbpm.codependent_receptors
        try:
            sbpm.codependent_receptors = {}
            lbl = sbpm._axis_label("rec_A")
            self.assertIn("rec_A", lbl)
            self.assertNotIn("\n[", lbl)
        finally:
            sbpm.codependent_receptors = saved

    def test_with_codependent_shows_ratio_and_both_names(self):
        saved = sbpm.codependent_receptors
        try:
            sbpm.codependent_receptors = {"rec_B": ("rec_A", 2.0)}
            lbl = sbpm._axis_label("rec_A")
            self.assertIn("rec_A", lbl)
            self.assertIn("rec_B", lbl)
            self.assertIn("2", lbl)
            self.assertIn("\n[", lbl)
        finally:
            sbpm.codependent_receptors = saved


# ────────────────────────────────────────────────────────────────────────────
class TestAxisLabelExtended(unittest.TestCase):
    """Additional coverage for _axis_label."""

    def test_log_form_contains_log_text(self):
        saved = sbpm.codependent_receptors
        try:
            sbpm.codependent_receptors = {}
            lbl = sbpm._axis_label("rec_A", log=True)
            self.assertIn("rec_A", lbl)
            self.assertTrue("log" in lbl.lower(), f"'log' not in label: {lbl!r}")
        finally:
            sbpm.codependent_receptors = saved

    def test_multiple_codependents_all_appear(self):
        saved = sbpm.codependent_receptors
        try:
            sbpm.codependent_receptors = {"B": ("A", 2.0), "C": ("A", 0.5)}
            lbl = sbpm._axis_label("A")
            self.assertIn("B", lbl)
            self.assertIn("C", lbl)
        finally:
            sbpm.codependent_receptors = saved

    def test_label_for_other_primary_not_polluted(self):
        """Codependents of primary A must not appear in the label for primary X."""
        saved = sbpm.codependent_receptors
        try:
            sbpm.codependent_receptors = {"B": ("A", 2.0)}
            lbl = sbpm._axis_label("X")
            self.assertNotIn("B", lbl)
            self.assertNotIn("A", lbl)
        finally:
            sbpm.codependent_receptors = saved


# ────────────────────────────────────────────────────────────────────────────
class TestSweep1Axis(unittest.TestCase):
    """Run sweep_1axis once; all subtests share the result."""

    @classmethod
    def setUpClass(cls):
        cls.res = sbpm.sweep_1axis("gaussian", "default")

    def test_array_lengths(self):
        n = sbpm.n_pts_1D
        self.assertEqual(len(self.res["sigma_R_um2"]),    n)
        self.assertEqual(len(self.res["bound_fraction"]), n)
        self.assertEqual(len(self.res["n_ads"]),          n)

    def test_bound_fraction_in_unit_interval(self):
        bf = self.res["bound_fraction"]
        self.assertTrue(np.all(bf >= 0.0), f"Negative bf values: {bf}")
        self.assertTrue(np.all(bf <= 1.0), f"bf > 1 values: {bf}")

    def test_n_ads_consistency(self):
        expected = self.res["bound_fraction"] * sbpm.NP_conc
        np.testing.assert_allclose(self.res["n_ads"], expected, rtol=1e-10)

    def test_monotone_nondecreasing(self):
        """Adsorbed fraction must not decrease as sigma_R increases."""
        bf = self.res["bound_fraction"]
        diffs = np.diff(bf)
        self.assertTrue(
            np.all(diffs >= -1e-6),
            f"Non-monotone bound_fraction, diffs: {diffs}",
        )

    def test_flory_exact_returns_valid_result(self):
        res = sbpm.sweep_1axis("Flory-exact", "default")
        self.assertEqual(len(res["bound_fraction"]), sbpm.n_pts_1D)
        self.assertTrue(np.all(res["bound_fraction"] >= 0.0))
        self.assertTrue(np.all(res["bound_fraction"] <= 1.0))

    def test_primary_name_in_result(self):
        self.assertEqual(self.res["primary_name"], "default")


# ────────────────────────────────────────────────────────────────────────────
class TestSweep2Axes(unittest.TestCase):
    """Run sweep_2axes with dp_2rec; all subtests share the result."""

    @classmethod
    def setUpClass(cls):
        saved_dp  = sbpm.data_polymers
        saved_cod = sbpm.codependent_receptors
        sbpm.data_polymers         = dp_2rec
        sbpm.codependent_receptors = {}
        try:
            cls.res = sbpm.sweep_2axes("gaussian", "receptor_A", "receptor_B")
        finally:
            sbpm.data_polymers         = saved_dp
            sbpm.codependent_receptors = saved_cod

    def test_grid_shapes(self):
        n = sbpm.n_pts_2D
        self.assertEqual(self.res["bound_fraction"].shape, (n, n))
        self.assertEqual(self.res["n_ads"].shape,          (n, n))
        self.assertEqual(len(self.res["sigma_R1_um2"]),    n)
        self.assertEqual(len(self.res["sigma_R2_um2"]),    n)

    def test_bound_fraction_in_unit_interval(self):
        bf = self.res["bound_fraction"]
        self.assertTrue(np.all(bf >= 0.0))
        self.assertTrue(np.all(bf <= 1.0))

    def test_n_ads_consistency(self):
        expected = self.res["bound_fraction"] * sbpm.NP_conc
        np.testing.assert_allclose(self.res["n_ads"], expected, rtol=1e-10)

    def test_rec_names_length(self):
        self.assertEqual(len(self.res["rec_names"]), 2)

    def test_rec_names_match_primaries(self):
        self.assertEqual(self.res["rec_names"], ["receptor_A", "receptor_B"])

    def test_symmetry(self):
        """With identical ligands, bf[i,j] must equal bf[j,i]."""
        bf = self.res["bound_fraction"]
        np.testing.assert_allclose(
            bf, bf.T, rtol=1e-4,
            err_msg="bound_fraction matrix not symmetric under receptor swap",
        )


# ────────────────────────────────────────────────────────────────────────────
class TestSweep1AxisCodependent(unittest.TestCase):
    """Run sweep_1axis with one codependent receptor (ratio 0.5)."""

    @classmethod
    def setUpClass(cls):
        saved_dp  = sbpm.data_polymers
        saved_cod = sbpm.codependent_receptors
        sbpm.data_polymers         = dp_2rec
        sbpm.codependent_receptors = {"receptor_B": ("receptor_A", 0.5)}
        try:
            cls.res = sbpm.sweep_1axis("gaussian", "receptor_A")
        finally:
            sbpm.data_polymers         = saved_dp
            sbpm.codependent_receptors = saved_cod

    def test_array_length(self):
        self.assertEqual(len(self.res["bound_fraction"]), sbpm.n_pts_1D)

    def test_bound_fraction_in_unit_interval(self):
        bf = self.res["bound_fraction"]
        self.assertTrue(np.all(bf >= 0.0))
        self.assertTrue(np.all(bf <= 1.0))

    def test_primary_name_in_result(self):
        self.assertEqual(self.res["primary_name"], "receptor_A")


# ────────────────────────────────────────────────────────────────────────────
class TestSweep1AxisListFormLigand(unittest.TestCase):
    """sweep_1axis with a list-form ligand (triggers is_multi=True code path)."""

    @classmethod
    def setUpClass(cls):
        saved_dp  = sbpm.data_polymers
        saved_cod = sbpm.codependent_receptors
        sbpm.data_polymers         = dp_list_form
        sbpm.codependent_receptors = {"rec_Y": ("rec_X", 1.0)}
        try:
            cls.res = sbpm.sweep_1axis("gaussian", "rec_X")
        finally:
            sbpm.data_polymers         = saved_dp
            sbpm.codependent_receptors = saved_cod

    def test_result_array_length(self):
        self.assertEqual(len(self.res["bound_fraction"]), sbpm.n_pts_1D)

    def test_bound_fraction_in_unit_interval(self):
        bf = self.res["bound_fraction"]
        self.assertTrue(np.all(bf >= 0.0))
        self.assertTrue(np.all(bf <= 1.0))

    def test_monotone_nondecreasing(self):
        """With ratio=1 both densities increase together; bf must not decrease."""
        bf = self.res["bound_fraction"]
        diffs = np.diff(bf)
        self.assertTrue(
            np.all(diffs >= -1e-6),
            f"Non-monotone bound_fraction, diffs: {diffs}",
        )


# ────────────────────────────────────────────────────────────────────────────
class TestSweep2AxesWithCodependent(unittest.TestCase):
    """sweep_2axes with 3 receptors: 2 primaries + 1 codependent."""

    @classmethod
    def setUpClass(cls):
        saved_dp  = sbpm.data_polymers
        saved_cod = sbpm.codependent_receptors
        sbpm.data_polymers         = dp_3rec
        sbpm.codependent_receptors = {"receptor_C": ("receptor_A", 0.5)}
        try:
            cls.res = sbpm.sweep_2axes("gaussian", "receptor_A", "receptor_B")
        finally:
            sbpm.data_polymers         = saved_dp
            sbpm.codependent_receptors = saved_cod

    def test_grid_shape(self):
        n = sbpm.n_pts_2D
        self.assertEqual(self.res["bound_fraction"].shape, (n, n))

    def test_bound_fraction_in_unit_interval(self):
        bf = self.res["bound_fraction"]
        self.assertTrue(np.all(bf >= 0.0))
        self.assertTrue(np.all(bf <= 1.0))


# ────────────────────────────────────────────────────────────────────────────
class TestRefLabel(unittest.TestCase):

    def test_with_user_labels(self):
        saved = sbpm.target_sigma_R_labels
        try:
            sbpm.target_sigma_R_labels = ["cell A", "cell B"]
            self.assertEqual(sbpm._ref_label(0, 100.0), "cell A")
            self.assertEqual(sbpm._ref_label(1, 500.0), "cell B")
            self.assertEqual(sbpm._ref_label(0, 100.0, suffix=" (rec)"), "cell A (rec)")
        finally:
            sbpm.target_sigma_R_labels = saved

    def test_without_labels(self):
        saved = sbpm.target_sigma_R_labels
        try:
            sbpm.target_sigma_R_labels = []
            lbl0 = sbpm._ref_label(0, 100.0)
            self.assertIn("100", lbl0)
            self.assertNotEqual(lbl0, "_")
            self.assertEqual(sbpm._ref_label(1, 500.0), "_")
            self.assertEqual(sbpm._ref_label(2, 1000.0), "_")
        finally:
            sbpm.target_sigma_R_labels = saved


# ────────────────────────────────────────────────────────────────────────────
# Helpers shared across regression test classes
# ────────────────────────────────────────────────────────────────────────────
_R_NP_reg   = 35.0 * nm
_amono_reg  = 0.28 * nm
_akuhn_reg  = 0.76 * nm
_NmonoS_reg = Nmonomers(2000 * g)
_NmonoL_reg = Nmonomers(3400 * g)


def _save_sbpm_state():
    return (sbpm.data_polymers, sbpm.codependent_receptors,
            sbpm.sigma_R_min, sbpm.sigma_R_max, sbpm.n_pts_1D, sbpm.n_pts_2D,
            sbpm.NP_conc, sbpm.cell_conc, sbpm.A_cell, sbpm.R_NP)


def _restore_sbpm_state(state):
    (sbpm.data_polymers, sbpm.codependent_receptors,
     sbpm.sigma_R_min, sbpm.sigma_R_max, sbpm.n_pts_1D, sbpm.n_pts_2D,
     sbpm.NP_conc, sbpm.cell_conc, sbpm.A_cell, sbpm.R_NP) = state


# ────────────────────────────────────────────────────────────────────────────
class TestRegressionSingleLigandSingleReceptor(unittest.TestCase):
    """Replicates multi/test_multi.py System 1 (ligand A only, gaussian).

    Reference bf is computed directly via MultivalentBinding.
    Sweep must produce bit-identical results since adsorption.py is identical.
    """

    @classmethod
    def setUpClass(cls):
        sigma_R_reg   = 200.0 / um2
        N_lig_A       = 100
        sigma_L_A     = N_lig_A / (4.0 * np.pi * _R_NP_reg**2)
        sigma_P2K_A   = sigma_L_A * 11.4
        KD_A          = 150.0 * nM
        NP_conc_reg   = 1e10 / mL
        cell_conc_reg = 1e5  / mL
        A_cell_reg    = 100  * um2   # same as module default

        rec_ref = {"name": "rec_A_reg", "sigma_R": sigma_R_reg}
        dp_ref  = {
            "short": {"N": _NmonoS_reg, "a": _amono_reg, "sigma": sigma_P2K_A,
                      "name": "PEG2K", "akuhn": _akuhn_reg},
            "lig_A": {"N": _NmonoL_reg, "a": _amono_reg, "sigma": sigma_L_A,
                      "name": "lig_A", "akuhn": _akuhn_reg,
                      "K_bind_0": KD_A**(-1), "receptor": rec_ref},
        }

        # ── Reference: direct MultivalentBinding call ──────────────────────
        sys_ref = MultivalentBinding(
            kT=kT, R_NP=_R_NP_reg, data_polymers=dp_ref,
            binding_model="exact", polymer_model="gaussian",
            A_cell=A_cell_reg, NP_conc=NP_conc_reg, cell_conc=cell_conc_reg,
        )
        max_NR    = int(sys_ref.NP_excluded_area * sigma_R_reg)
        max_N     = max_NR + 4 * (max_NR + 1) + 1
        K_vs_NR   = sys_ref.calculate_K_bind_vs_receptors(max_N)
        M_conc    = (A_cell_reg / sys_ref.NP_excluded_area) * cell_conc_reg
        cls.bf_ref = float(sys_ref.calculate_bound_fraction(
            fluctuations=True, depletion=True,
            K_bind_vs_receptors=K_vs_NR, rho_m=M_conc,
            max_n_receptor=max_N))
        cls.NP_conc_reg = NP_conc_reg

        # ── Sweep: single point at sigma_R_reg ────────────────────────────
        saved = _save_sbpm_state()
        sbpm.data_polymers         = dp_ref
        sbpm.codependent_receptors = {}
        sbpm.sigma_R_min           = sigma_R_reg
        sbpm.sigma_R_max           = sigma_R_reg
        sbpm.n_pts_1D              = 1
        sbpm.NP_conc               = NP_conc_reg
        sbpm.cell_conc             = cell_conc_reg
        sbpm.A_cell                = A_cell_reg   # same as default, explicit for clarity
        sbpm.R_NP                  = _R_NP_reg    # same as default, explicit for clarity
        try:
            cls.res = sbpm.sweep_1axis("gaussian", "rec_A_reg")
        finally:
            _restore_sbpm_state(saved)

    def test_bound_fraction_matches_direct(self):
        self.assertAlmostEqual(
            self.res["bound_fraction"][0], self.bf_ref,
            delta=1e-10,
            msg=f"sweep={self.res['bound_fraction'][0]:.15e}, ref={self.bf_ref:.15e}",
        )

    def test_n_ads_matches_direct(self):
        expected = self.bf_ref * self.NP_conc_reg
        self.assertAlmostEqual(
            self.res["n_ads"][0], expected,
            delta=1e-10 * self.NP_conc_reg,
        )


# ────────────────────────────────────────────────────────────────────────────
class TestRegressionTwoLigandsSameReceptor(unittest.TestCase):
    """Replicates multi/test_multi_single_receptor.py System 3.

    Two ligands sharing a single receptor dict, Flory-exact, fluctuations+depletion.
    """

    @classmethod
    def setUpClass(cls):
        sigma_R_test  = 700.0 / um2
        N_lig         = 50
        sigma_L_reg   = N_lig / (4.0 * np.pi * _R_NP_reg**2)
        sigma_P2K_reg = sigma_L_reg * 11.4
        KD_A          = 150.0 * nM
        KD_B          = 50.0  * nM
        NP_conc_reg   = 1e8  / mL
        cell_conc_reg = 1e5  / mL
        A_cell_reg    = 100  * um2

        rec_shared = {"name": "shared_rec_reg", "sigma_R": sigma_R_test}
        dp_ref = {
            "short":  {"N": _NmonoS_reg, "a": _amono_reg, "sigma": sigma_P2K_reg,
                       "name": "PEG2K", "akuhn": _akuhn_reg},
            "lig_A":  {"N": _NmonoL_reg, "a": _amono_reg, "sigma": sigma_L_reg,
                       "name": "lig_A", "akuhn": _akuhn_reg,
                       "K_bind_0": KD_A**(-1), "receptor": rec_shared},
            "lig_B":  {"N": _NmonoL_reg, "a": _amono_reg, "sigma": sigma_L_reg,
                       "name": "lig_B", "akuhn": _akuhn_reg,
                       "K_bind_0": KD_B**(-1), "receptor": rec_shared},  # same dict object
        }

        # ── Reference ─────────────────────────────────────────────────────
        sys_ref = MultivalentBinding(
            kT=kT, R_NP=_R_NP_reg, data_polymers=dp_ref,
            binding_model="exact", polymer_model="Flory-exact",
            A_cell=A_cell_reg, NP_conc=NP_conc_reg, cell_conc=cell_conc_reg,
        )
        max_NR    = int(sys_ref.NP_excluded_area * sigma_R_test)
        max_N     = max_NR + 4 * (max_NR + 1) + 1
        K_vs_NR   = sys_ref.calculate_K_bind_vs_receptors(max_N)
        M_conc    = (A_cell_reg / sys_ref.NP_excluded_area) * cell_conc_reg
        cls.bf_ref = float(sys_ref.calculate_bound_fraction(
            fluctuations=True, depletion=True,
            K_bind_vs_receptors=K_vs_NR, rho_m=M_conc,
            max_n_receptor=max_N))
        cls.NP_conc_reg = NP_conc_reg

        # ── Sweep: single point at sigma_R_test ───────────────────────────
        saved = _save_sbpm_state()
        sbpm.data_polymers         = dp_ref
        sbpm.codependent_receptors = {}
        sbpm.sigma_R_min           = sigma_R_test
        sbpm.sigma_R_max           = sigma_R_test
        sbpm.n_pts_1D              = 1
        sbpm.NP_conc               = NP_conc_reg
        sbpm.cell_conc             = cell_conc_reg
        sbpm.A_cell                = A_cell_reg
        sbpm.R_NP                  = _R_NP_reg
        try:
            cls.res = sbpm.sweep_1axis("Flory-exact", "shared_rec_reg")
        finally:
            _restore_sbpm_state(saved)

    def test_bound_fraction_matches_direct(self):
        self.assertAlmostEqual(
            self.res["bound_fraction"][0], self.bf_ref,
            delta=1e-10,
            msg=f"sweep={self.res['bound_fraction'][0]:.15e}, ref={self.bf_ref:.15e}",
        )

    def test_n_ads_matches_direct(self):
        expected = self.bf_ref * self.NP_conc_reg
        self.assertAlmostEqual(
            self.res["n_ads"][0], expected,
            delta=1e-10 * self.NP_conc_reg,
        )


# ────────────────────────────────────────────────────────────────────────────
class TestRegressionTwoLigandsTwoReceptors(unittest.TestCase):
    """Replicates multi/test_multi.py System 3 at a symmetric sigma_R point.

    Both receptors at the same density → 1×1 grid → check bf_grid[0,0].
    """

    @classmethod
    def setUpClass(cls):
        sigma_R_sym   = 150.0 / um2   # within speed-override range 10–200 µm⁻²
        N_lig_A       = 100
        N_lig_B       = 50
        sigma_L_A_r   = N_lig_A / (4.0 * np.pi * _R_NP_reg**2)
        sigma_L_B_r   = N_lig_B / (4.0 * np.pi * _R_NP_reg**2)
        sigma_P2K_r   = sigma_L_A_r * 11.4
        KD_A          = 150.0 * nM
        KD_B          = 50.0  * nM
        NP_conc_reg   = 1e10 / mL
        cell_conc_reg = 1e5  / mL
        A_cell_reg    = 100  * um2

        rec_A = {"name": "rec_A_reg", "sigma_R": sigma_R_sym}
        rec_B = {"name": "rec_B_reg", "sigma_R": sigma_R_sym}
        dp_ref = {
            "short":  {"N": _NmonoS_reg, "a": _amono_reg, "sigma": sigma_P2K_r,
                       "name": "PEG2K", "akuhn": _akuhn_reg},
            "lig_A":  {"N": _NmonoL_reg, "a": _amono_reg, "sigma": sigma_L_A_r,
                       "name": "lig_A", "akuhn": _akuhn_reg,
                       "K_bind_0": KD_A**(-1), "receptor": rec_A},
            "lig_B":  {"N": _NmonoL_reg, "a": _amono_reg, "sigma": sigma_L_B_r,
                       "name": "lig_B", "akuhn": _akuhn_reg,
                       "K_bind_0": KD_B**(-1), "receptor": rec_B},
        }

        # ── Reference: direct 2-receptor MultivalentBinding call ──────────
        sys_ref = MultivalentBinding(
            kT=kT, R_NP=_R_NP_reg, data_polymers=dp_ref,
            binding_model="exact", polymer_model="gaussian",
            A_cell=A_cell_reg, NP_conc=NP_conc_reg, cell_conc=cell_conc_reg,
        )
        max_NR   = int(sys_ref.NP_excluded_area * sigma_R_sym)
        max_N    = max_NR + 4 * (max_NR + 1) + 1
        K_flat, grid_shape, _, rec_names = sys_ref.calculate_K_bind_vs_receptors(max_N)
        NR_ave   = float(sys_ref.NP_excluded_area * sigma_R_sym)
        M_conc   = (A_cell_reg / sys_ref.NP_excluded_area) * cell_conc_reg
        cls.bf_ref = float(sys_ref.calculate_bound_fraction(
            fluctuations=True, depletion=True, rho_m=M_conc,
            K_bind_vs_receptors=(K_flat, grid_shape,
                                 [NR_ave, NR_ave], rec_names)))
        cls.NP_conc_reg = NP_conc_reg

        # ── Sweep: 1×1 grid at sigma_R_sym ───────────────────────────────
        saved = _save_sbpm_state()
        sbpm.data_polymers         = dp_ref
        sbpm.codependent_receptors = {}
        sbpm.sigma_R_min           = sigma_R_sym
        sbpm.sigma_R_max           = sigma_R_sym
        sbpm.n_pts_2D              = 1
        sbpm.NP_conc               = NP_conc_reg
        sbpm.cell_conc             = cell_conc_reg
        sbpm.A_cell                = A_cell_reg
        sbpm.R_NP                  = _R_NP_reg
        try:
            cls.res = sbpm.sweep_2axes("gaussian", "rec_A_reg", "rec_B_reg")
        finally:
            _restore_sbpm_state(saved)

    def test_bound_fraction_matches_direct(self):
        self.assertAlmostEqual(
            self.res["bound_fraction"][0, 0], self.bf_ref,
            delta=1e-10,
            msg=(f"sweep={self.res['bound_fraction'][0,0]:.15e}, "
                 f"ref={self.bf_ref:.15e}"),
        )

    def test_n_ads_matches_direct(self):
        expected = self.bf_ref * self.NP_conc_reg
        self.assertAlmostEqual(
            self.res["n_ads"][0, 0], expected,
            delta=1e-10 * self.NP_conc_reg,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
