import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from xds_cred.config import ScoreConfig
from xds_cred.geometry import (
    GeometryBatch,
    XdsGeometry,
    XdsGeometryModel,
    calculate_observation_risk,
    excitation_at_z,
    hkl_vectors,
    predict_z_by_newton,
    reciprocal_from_unit_cell,
    reciprocal_matrix_at_z,
    spindle_angle_at_z,
)
from xds_cred.risk import (
    enumerate_neighbor_offsets,
    score_dataframe_chunked,
    score_dataframe_serial,
    score_integrate_file,
)
from xds_cred.statistics import add_leave_one_out_disagreement, add_within_class_ranks, spearman
from xds_cred.symmetry import canonical_hkl, require_symmetry_backend
from xds_cred.xds_parser import parse_integrate_header


TEST_DIR = Path(__file__).resolve().parent
SCRATCH_ROOT = TEST_DIR / "scratch"


def synthetic_geometry(
    *,
    source: str = "synthetic",
    direct_scale: float = 10.0,
    starting_angle: float = 0.0,
    oscillation_range: float = 1.0,
    angle_offset_frames: float = 1.0,
    incident_beam_direction: np.ndarray | None = None,
    wavelength: float = 1.0,
) -> XdsGeometry:
    direct = np.eye(3) * direct_scale
    return XdsGeometry(
        source=source,
        starting_frame=1,
        starting_angle=starting_angle,
        oscillation_range=oscillation_range,
        rotation_axis=np.array([0.0, 0.0, 1.0]),
        wavelength=wavelength,
        incident_beam_direction=(
            np.array([0.0, 0.0, 1.0])
            if incident_beam_direction is None
            else np.asarray(incident_beam_direction, dtype=float)
        ),
        space_group_number=1,
        unit_cell=(direct_scale, direct_scale, direct_scale, 90.0, 90.0, 90.0),
        direct_matrix=direct,
        detector_segment=1,
        nx=100,
        ny=100,
        qx=0.1,
        qy=0.1,
        orgx=50.0,
        orgy=50.0,
        detector_distance=100.0,
        detector_x_axis=np.array([1.0, 0.0, 0.0]),
        detector_y_axis=np.array([0.0, 1.0, 0.0]),
        detector_normal=np.array([0.0, 0.0, 1.0]),
        angle_offset_frames=angle_offset_frames,
    )


def bragg_crossing_geometry() -> XdsGeometry:
    return synthetic_geometry(
        source="bragg-crossing",
        direct_scale=1.0,
        incident_beam_direction=np.array([1.0, 0.0, 0.0]),
        angle_offset_frames=1.0,
    )


def batch_geometry_model() -> XdsGeometryModel:
    base = synthetic_geometry(source="base")
    refined = base.direct_matrix.copy()
    refined[0, 0] = 8.0
    refined[1, 1] = 12.0
    return XdsGeometryModel(
        source="synthetic INTEGRATE.LP batches",
        base=base,
        batches=(
            GeometryBatch(start_image=1, end_image=2, direct_matrix=base.direct_matrix),
            GeometryBatch(start_image=3, end_image=5, direct_matrix=refined),
        ),
        angle_offset_frames=1.0,
    )


def synthetic_observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_id": [0, 1, 2, 3],
            "H": [1, 1, 0, 0],
            "K": [0, 0, 1, 1],
            "L": [0, 0, 0, 0],
            "IOBS": [10.0, 12.0, 20.0, 18.0],
            "SIGMA": [1.0, 1.0, 2.0, 2.0],
            "XCAL": [1.0, 1.0, 1.0, 1.0],
            "YCAL": [1.0, 1.0, 1.0, 1.0],
            "ZCAL": [1.0, 2.0, 3.0, 4.0],
            "XOBS": [1.0, 1.0, 1.0, 1.0],
            "YOBS": [1.0, 1.0, 1.0, 1.0],
            "ZOBS": [1.0, 2.0, 3.0, 4.0],
            "PEAK": [100.0, 100.0, 100.0, 100.0],
            "CORR": [90.0, 90.0, 90.0, 90.0],
        }
    )


def assert_columns_equal(testcase: unittest.TestCase, left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> None:
    assert_frame_equal(
        left[columns].reset_index(drop=True),
        right[columns].reset_index(drop=True),
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )
    testcase.assertEqual(left[columns].shape, right[columns].shape)


class XdsCredWorkflowTests(unittest.TestCase):
    def test_xds_header_parsing(self) -> None:
        path = TEST_DIR / "data" / "minimal_integrate.HKL"
        header = parse_integrate_header(path)
        self.assertEqual(header.columns, ["H", "K", "L", "IOBS", "SIGMA"])
        self.assertEqual(header.metadata["NUMBER_OF_ITEMS_IN_EACH_DATA_RECORD"], 5)

    def test_reciprocal_basis_construction(self) -> None:
        reciprocal = reciprocal_from_unit_cell((10.0, 10.0, 10.0, 90.0, 90.0, 90.0))
        self.assertTrue(np.allclose(np.diag(reciprocal), [0.1, 0.1, 0.1]))

    def test_fractional_zcal_to_angle_uses_configured_one_frame_offset(self) -> None:
        geometry = synthetic_geometry(starting_angle=10.0, oscillation_range=0.03, angle_offset_frames=1.0)
        angle = spindle_angle_at_z(geometry, 3.5)
        self.assertAlmostEqual(angle, 10.0 + 0.03 * (3.5 - 1.0 + 1.0), places=12)

    def test_matrix_rotation_convention_rotates_unrotated_xds_basis(self) -> None:
        geometry = synthetic_geometry(angle_offset_frames=1.0)
        reciprocal = reciprocal_matrix_at_z(geometry, 90.0)
        q_vector = hkl_vectors(np.array([1, 0, 0]), reciprocal)[0]
        self.assertTrue(np.allclose(q_vector, [0.0, 0.1, 0.0], atol=1e-12))

    def test_synthetic_known_bragg_crossing_predicts_fractional_z(self) -> None:
        geometry = bragg_crossing_geometry()
        z_pred, status = predict_z_by_newton((1, 0, 0), geometry, 119.8, max_window_frames=5.0)
        self.assertEqual(status, "converged")
        self.assertAlmostEqual(z_pred, 120.0, places=6)

    def test_target_excitation_is_zero_at_synthetic_bragg_crossing(self) -> None:
        geometry = bragg_crossing_geometry()
        score = ScoreConfig(s0=0.005, sigma_c=0.03, r_cut=0.2)
        result = calculate_observation_risk((1, 0, 0), 120.0, geometry, score, np.empty((0, 3), dtype=int))
        self.assertAlmostEqual(excitation_at_z((1, 0, 0), geometry, 120.0), 0.0, places=12)
        self.assertAlmostEqual(result.s_target, 0.0, places=12)
        self.assertAlmostEqual(result.E_target, 1.0, places=12)
        self.assertAlmostEqual(result.z_residual_frames, 0.0, places=12)

    def test_direct_observation_level_matches_batch_dataframe_calculation(self) -> None:
        geometry = batch_geometry_model()
        score = ScoreConfig(s0=0.05, sigma_c=0.05, r_cut=0.16)
        observations = synthetic_observations().iloc[[2]].copy()
        offsets, _distances, _coupling = enumerate_neighbor_offsets(
            geometry.base.reciprocal_matrix_zero,
            score.r_cut,
            score.sigma_c,
        )
        direct = calculate_observation_risk((0, 1, 0), 3.0, geometry, score, offsets)
        table = score_dataframe_serial(observations, geometry, score).iloc[0]
        self.assertEqual(direct.geometry_batch, "3-5")
        self.assertEqual(table["geometry_batch"], "3-5")
        self.assertAlmostEqual(table["s_target"], direct.s_target, places=12)
        self.assertAlmostEqual(table["E_target"], direct.E_target, places=12)
        self.assertAlmostEqual(table["R_env"], direct.R_env, places=12)
        self.assertAlmostEqual(table["S_risk"], direct.S_risk, places=12)
        self.assertEqual(np.isfinite(table["z_residual_frames"]), np.isfinite(direct.z_residual_frames))
        if np.isfinite(direct.z_residual_frames):
            self.assertAlmostEqual(table["z_residual_frames"], direct.z_residual_frames, places=12)

    def test_brute_force_neighbor_summation_matches_enumerated_offsets(self) -> None:
        geometry = synthetic_geometry()
        score = ScoreConfig(s0=0.05, sigma_c=0.05, r_cut=0.11)
        offsets, _distances, _coupling = enumerate_neighbor_offsets(
            geometry.reciprocal_matrix_zero,
            score.r_cut,
            score.sigma_c,
        )
        result = calculate_observation_risk((1, 0, 0), 1.0, geometry, score, offsets)
        brute_force_r_env, brute_force_count = brute_force_neighbors((1, 0, 0), 1.0, geometry, score, limit=2)
        self.assertEqual(result.neighbor_count, brute_force_count)
        self.assertAlmostEqual(result.R_env, brute_force_r_env, places=12)
        self.assertAlmostEqual(result.S_risk, result.E_target * brute_force_r_env, places=12)

    def test_serial_and_parallel_file_scoring_are_equal(self) -> None:
        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="parallel_", dir=SCRATCH_ROOT) as tmp:
            tmp_path = Path(tmp)
            integrate_path = tmp_path / "synthetic_integrate.HKL"
            columns = write_synthetic_integrate(integrate_path, synthetic_observations())
            geometry = synthetic_geometry()
            score = ScoreConfig(s0=0.05, sigma_c=0.05, r_cut=0.11)
            serial_path = tmp_path / "serial.csv"
            parallel_path = tmp_path / "parallel.csv"

            score_integrate_file(integrate_path, columns, geometry, score, 2, 1, serial_path)
            score_integrate_file(integrate_path, columns, geometry, score, 2, 2, parallel_path)

            serial = pd.read_csv(serial_path)
            parallel = pd.read_csv(parallel_path)
            assert_frame_equal(serial, parallel, check_exact=False, atol=1e-12, rtol=1e-12)

    def test_chunk_size_invariance(self) -> None:
        geometry = synthetic_geometry()
        score = ScoreConfig(s0=0.05, sigma_c=0.05, r_cut=0.11)
        observations = pd.concat([synthetic_observations()] * 3, ignore_index=True)
        observations["observation_id"] = np.arange(len(observations))
        chunk_one = score_dataframe_chunked(observations, geometry, score, chunk_size=1)
        chunk_four = score_dataframe_chunked(observations, geometry, score, chunk_size=4)
        assert_columns_equal(self, chunk_one, chunk_four, ["S_risk", "R_env", "E_target", "z_residual_frames"])

    def test_symmetry_canonicalization_keeps_friedel_policy_backend(self) -> None:
        try:
            require_symmetry_backend()
        except RuntimeError as exc:
            raise unittest.SkipTest("gemmi/cctbx symmetry backend is not installed") from exc
        self.assertEqual(canonical_hkl((1, 2, 3), 1), (1, 2, 3))
        self.assertEqual(canonical_hkl((-1, -2, -3), 1), (-1, -2, -3))

    def test_leave_one_out_median_and_disagreement_eligibility(self) -> None:
        observations = pd.DataFrame(
            {
                "observation_id": [0, 1, 2, 3],
                "h": [1, 1, 1, 2],
                "k": [0, 0, 0, 0],
                "l": [0, 0, 0, 0],
                "symmetry_id": ["a", "a", "a", "b"],
                "multiplicity": [3, 3, 3, 1],
                "IOBS": [10.0, 20.0, 30.0, 50.0],
                "I_over_SIGMA": [10.0, 20.0, 30.0, 25.0],
                "S_risk": [0.1, 0.2, 0.3, 0.4],
            }
        )
        out, excluded = add_leave_one_out_disagreement(observations, min_multiplicity=3)
        self.assertAlmostEqual(out.loc[0, "reference_intensity"], 25.0)
        self.assertAlmostEqual(out.loc[0, "relative_disagreement"], 0.6)
        self.assertFalse(bool(out.loc[3, "disagreement_valid"]))
        reason = excluded.loc[excluded["observation_id"] == 3, "exclusion_reason"].iloc[0]
        self.assertIn("symmetry_multiplicity_lt_3", reason)

    def test_disagreement_invalid_when_reference_nonpositive(self) -> None:
        observations = pd.DataFrame(
            {
                "observation_id": [0, 1, 2],
                "h": [1, 1, 1],
                "k": [0, 0, 0],
                "l": [0, 0, 0],
                "symmetry_id": ["a", "a", "a"],
                "multiplicity": [3, 3, 3],
                "IOBS": [-1.0, -2.0, -3.0],
                "I_over_SIGMA": [-1.0, -2.0, -3.0],
                "S_risk": [0.1, 0.2, 0.3],
            }
        )
        out, _excluded = add_leave_one_out_disagreement(observations, min_multiplicity=3)
        self.assertFalse(out["disagreement_valid"].any())
        self.assertEqual(set(out["exclusion_reason"]), {"nonpositive_reference_intensity"})

    def test_within_class_rank_transformation_and_spearman(self) -> None:
        observations = pd.DataFrame(
            {
                "symmetry_id": ["a", "a", "a"],
                "disagreement_valid": [True, True, True],
                "S_risk": [10.0, 10.0, 30.0],
                "relative_disagreement": [1.0, 3.0, 2.0],
            }
        )
        out = add_within_class_ranks(observations)
        self.assertTrue(np.allclose(out["risk_rank_within_class"], [0.25, 0.25, 1.0]))
        result = spearman(out["risk_rank_within_class"], out["disagreement_rank_within_class"])
        self.assertEqual(result["n"], 3)


def brute_force_neighbors(
    hkl: tuple[int, int, int],
    zcal: float,
    geometry: XdsGeometry,
    score: ScoreConfig,
    limit: int,
) -> tuple[float, int]:
    reciprocal = reciprocal_matrix_at_z(geometry, zcal)
    target_hkl = np.asarray(hkl, dtype=int)
    target_q = hkl_vectors(target_hkl, reciprocal)[0]
    total = 0.0
    count = 0
    for dh in range(-limit, limit + 1):
        for dk in range(-limit, limit + 1):
            for dl in range(-limit, limit + 1):
                if (dh, dk, dl) == (0, 0, 0):
                    continue
                neighbor_hkl = target_hkl + np.asarray([dh, dk, dl], dtype=int)
                neighbor_q = hkl_vectors(neighbor_hkl, reciprocal)[0]
                distance = float(np.linalg.norm(neighbor_q - target_q))
                if distance > score.r_cut + 1e-12:
                    continue
                s_neighbor = excitation_at_z(tuple(int(value) for value in neighbor_hkl), geometry, zcal)
                e_neighbor = math.exp(-((s_neighbor / score.s0) ** 2))
                c_gh = math.exp(-((distance / score.sigma_c) ** 2))
                total += e_neighbor * c_gh
                count += 1
    return total, count


def write_synthetic_integrate(path: Path, observations: pd.DataFrame) -> list[str]:
    columns = ["H", "K", "L", "IOBS", "SIGMA", "XCAL", "YCAL", "ZCAL", "XOBS", "YOBS", "ZOBS", "PEAK", "CORR"]
    lines = ["!END_OF_HEADER"]
    for row in observations.itertuples(index=False):
        values = [getattr(row, column) for column in columns]
        lines.append(" ".join(str(value) for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return columns


if __name__ == "__main__":
    unittest.main()
