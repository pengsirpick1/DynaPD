from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dmmp.encoders.prefix import extract_prefix_condition
from dmmp.projection.padding import PaddingTemplate, render_trace_variable
from dmmp.target_policy.candidate_generator import generate_candidates_for_trace
from dmmp.target_policy.config import TargetPolicyConfig
from dmmp.target_policy.candidate_generator import CandidatePolicy
from dmmp.target_policy.constraint_checker import ConstraintReport, check_counts
from dmmp.target_policy.representation import (
    allocation_to_x0_star,
    largest_remainder_rounding,
    masked_softmax,
    x0_star_to_counts,
)
from dmmp.target_policy.target_pool import TargetPolicyPool, validate_target_policy_pool, write_target_policy_pool
from dmmp.target_policy.target_selector import select_targets


class TargetPolicyCoreTests(unittest.TestCase):
    def test_largest_remainder_exact_budget_and_mask(self) -> None:
        mask = np.zeros((2, 8), dtype=np.float32)
        mask[:, 2:6] = 1.0
        utility = np.arange(16, dtype=np.float32).reshape(2, 8)
        counts = largest_remainder_rounding(utility, 11, mask)
        self.assertEqual(int(counts.sum()), 11)
        self.assertEqual(int(counts[mask <= 0].sum()), 0)
        self.assertTrue(np.all(counts >= 0))

    def test_masked_clr_round_trip_preserves_support_and_budget(self) -> None:
        mask = np.ones((2, 10), dtype=np.float32)
        mask[:, :2] = 0.0
        allocation = masked_softmax(np.linspace(-1.0, 1.0, 20, dtype=np.float32).reshape(2, 10), mask)
        x0 = allocation_to_x0_star(allocation, mask)
        counts, allocation_pred = x0_star_to_counts(x0, mask, 17)
        self.assertEqual(int(counts.sum()), 17)
        self.assertEqual(int(counts[mask <= 0].sum()), 0)
        self.assertAlmostEqual(float(allocation_pred.sum()), 1.0, places=5)

    def test_constraint_checker_rejects_mask_violation(self) -> None:
        mask = np.ones((2, 6), dtype=np.float32)
        mask[0, 0] = 0.0
        counts = np.zeros((2, 6), dtype=np.int32)
        counts[0, 0] = 1
        counts[1, 1] = 2
        report = check_counts(counts, mask, 3)
        self.assertFalse(report.valid)
        self.assertEqual(report.allowed_violation_count, 1)

    def test_renderer_preserves_real_packet_order(self) -> None:
        clean = np.asarray([1.0, -2.0, 3.0, -4.0, 0.0, 0.0], dtype=np.float32)
        counts = np.zeros((2, 8), dtype=np.int32)
        counts[0, 2] = 2
        counts[1, 3] = 1
        trace, origin, stats = render_trace_variable(clean, PaddingTemplate(counts, 3, 3, 0.5))
        retained = trace[origin]
        self.assertTrue(np.array_equal(retained, clean[clean != 0]))
        self.assertEqual(stats["original_retained_count"], 4)
        self.assertEqual(stats["dummy_count"], 3)

    def test_tam_obfuscation_renderer_preserves_real_packets(self) -> None:
        clean = np.asarray([0.10, -0.20, 0.45, -0.75, 0.0, 0.0], dtype=np.float32)
        counts = np.zeros((2, 4), dtype=np.int32)
        counts[0, 0] = 2
        counts[1, 1] = 2
        counts[0, 2] = 1
        trace, origin, stats = render_trace_variable(
            clean,
            PaddingTemplate(counts, 5, 5, 0.5),
            coordinate="tam_obfuscation",
            max_load_time=1.0,
            tam_obfuscation_strategy="hybrid_clustered",
            tam_local_run_max=2,
        )
        retained = trace[origin]
        self.assertTrue(np.array_equal(retained, clean[clean != 0]))
        self.assertEqual(stats["original_retained_count"], 4)
        self.assertEqual(stats["dummy_count"], 5)
        self.assertTrue(np.all(np.isin(np.sign(trace), [-1.0, 1.0])))
        self.assertTrue(np.all(np.abs(trace) > 0.0))

    def test_multi_view_split_renderer_splits_budget_and_preserves_real_packets(self) -> None:
        clean = np.asarray([0.10, -0.20, 0.45, -0.75, 1.10, -1.30, 0.0, 0.0], dtype=np.float32)
        counts = np.zeros((2, 6), dtype=np.int32)
        counts[0, 0] = 3
        counts[1, 1] = 2
        counts[0, 3] = 2
        counts[1, 4] = 3
        trace, origin, stats = render_trace_variable(
            clean,
            PaddingTemplate(counts, 10, 10, 0.5),
            coordinate="multi_view",
            multi_view_mode="split",
            coordinate_length=12,
            max_load_time=2.0,
            multi_view_df_share=0.40,
            multi_view_awf_share=0.30,
            multi_view_rf_share=0.30,
            tam_local_run_max=2,
        )
        retained = trace[origin]
        branch_total = (
            stats["multi_view_df_dummy_count"]
            + stats["multi_view_awf_dummy_count"]
            + stats["multi_view_rf_dummy_count"]
        )
        self.assertTrue(np.array_equal(retained, clean[clean != 0]))
        self.assertEqual(stats["dummy_count"], 10)
        self.assertEqual(branch_total, 10)
        self.assertEqual(stats["original_retained_count"], 6)

    def test_multi_view_fused_renderer_uses_shared_budget_and_preserves_real_packets(self) -> None:
        clean = np.asarray([0.10, -0.20, 0.45, -0.75, 1.10, -1.30, 0.0, 0.0], dtype=np.float32)
        counts = np.zeros((2, 6), dtype=np.int32)
        counts[0, 0] = 3
        counts[1, 1] = 2
        counts[0, 3] = 2
        counts[1, 4] = 3
        trace, origin, stats = render_trace_variable(
            clean,
            PaddingTemplate(counts, 10, 10, 0.5),
            coordinate="multi_view",
            multi_view_mode="fused",
            coordinate_length=12,
            max_load_time=2.0,
            multi_view_df_share=0.40,
            multi_view_awf_share=0.30,
            multi_view_rf_share=0.30,
            tam_local_run_max=2,
        )
        retained = trace[origin]
        self.assertTrue(np.array_equal(retained, clean[clean != 0]))
        self.assertEqual(stats["dummy_count"], 10)
        self.assertEqual(stats["multi_view_mode"], "fused")
        self.assertEqual(stats["multi_view_shared_dummy_count"], 10)
        self.assertGreaterEqual(stats["multi_view_mean_slot_score"], 0.0)
        self.assertEqual(stats["original_retained_count"], 6)
        self.assertTrue(np.all(np.diff(np.abs(trace)) >= -1.0e-7))

    def test_candidate_generation_and_pool_loader_omit_labels(self) -> None:
        trace = np.asarray([(-1.0) ** i * (i + 1) * 0.01 for i in range(80)], dtype=np.float32)
        cfg = TargetPolicyConfig(
            prefix_length=40,
            strategy_horizon=20,
            budgets=(0.10,),
            max_budget=0.10,
            num_candidates=4,
            target_count=2,
            quality_target_count=1,
            diverse_target_count=1,
        )
        condition = extract_prefix_condition(trace, prefix_n=cfg.prefix_length, patch_num=cfg.strategy_horizon)
        candidates = generate_candidates_for_trace(trace, cfg=cfg, prefix_condition=condition, rng=np.random.default_rng(7))
        selected, _ = select_targets(candidates, target_count=2, quality_target_count=1, diverse_target_count=1)
        self.assertEqual(len(selected), 2)
        with tempfile.TemporaryDirectory() as tmp:
            records = [(0, idx, candidate, condition.vector) for idx, candidate in enumerate(selected)]
            write_target_policy_pool(Path(tmp), records, metadata={"seed": 7})
            report = validate_target_policy_pool(Path(tmp))
            self.assertTrue(report["valid"], report)
            arrays = TargetPolicyPool(Path(tmp)).load_training_arrays()
            self.assertNotIn("y", arrays)
            self.assertNotIn("label", arrays)
            self.assertIn("x0_star", arrays)

    def test_training_loader_filters_label_like_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(
                root / "policies.npz",
                x0_star=np.zeros((1, 2, 4), dtype=np.float32),
                true_labels=np.asarray([3], dtype=np.int64),
                target_label=np.asarray([4], dtype=np.int64),
                site_id=np.asarray([5], dtype=np.int64),
                safe_feature=np.asarray([1.0], dtype=np.float32),
            )
            arrays = TargetPolicyPool(root).load_training_arrays()

        self.assertIn("x0_star", arrays)
        self.assertIn("safe_feature", arrays)
        self.assertNotIn("true_labels", arrays)
        self.assertNotIn("target_label", arrays)
        self.assertNotIn("site_id", arrays)

    def test_selector_prefers_deployable_candidates(self) -> None:
        shape = (2, 4)

        def candidate(quality: float, deployable: bool) -> CandidatePolicy:
            report = ConstraintReport(
                valid=True,
                deployable=deployable,
                allowed_violation_count=0,
                negative_count=0,
                actual_count=2,
                target_count=2,
                budget_error=0,
                max_slot_count=1,
                max_consecutive_dummy_run=1,
                max_local_density=0.25,
                tail_extension_ratio=0.0,
            )
            return CandidatePolicy(
                x0_star=np.zeros(shape, dtype=np.float32),
                allocation=np.full(shape, 1.0 / np.prod(shape), dtype=np.float32),
                counts=np.ones(shape, dtype=np.int32),
                allowed_mask=np.ones(shape, dtype=np.float32),
                budget_ratio=0.1,
                budget_count=2,
                family_weights=np.full(5, 0.2, dtype=np.float32),
                primitive_weights=np.full(5, 0.2, dtype=np.float32),
                family_indices=np.asarray([0], dtype=np.int64),
                primitive_indices=np.asarray([0], dtype=np.int64),
                effect_map=np.ones(shape, dtype=np.float32),
                action_rank=np.zeros((1, 2), dtype=np.int64),
                marginal_gain=np.ones(1, dtype=np.float32),
                quality_score=quality,
                proxy_score_df=quality,
                proxy_score_rf=quality,
                proxy_score_attack=quality,
                latency_cost=0.0,
                fallback_flag=not deployable,
                constraint_report=report,
                construction_seed=1,
            )

        selected, fallback_count = select_targets(
            [candidate(10.0, False), candidate(1.0, True)],
            target_count=1,
            quality_target_count=1,
            diverse_target_count=0,
        )
        self.assertTrue(selected[0].constraint_report.deployable)
        self.assertEqual(fallback_count, 0)


if __name__ == "__main__":
    unittest.main()
