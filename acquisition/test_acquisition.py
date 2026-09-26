import json
import tempfile
import unittest
from pathlib import Path

import torch

from acquisition.evaluate import evaluate_bundle
from acquisition.audit import audit_report
from acquisition.audit_r2_reviews import BASE_FIELDS, PAIR_FIELDS, required_fields
from acquisition.fixture import write_fixture
from acquisition.metrics import (bootstrap_cluster, caption_necessity, expected_random_metrics,
                                 material_switch, pair_regret)
from acquisition.observe_conditioned import cache_key as conditioned_cache_key
from acquisition.planner_baseline import parse_choice
from acquisition.screen_entailment import parse_label as parse_entailment
from acquisition.screen_independent_visual import parse_independent_label
from acquisition.screen_visual_support import parse_label as parse_visual_support
from acquisition.set_valued_selector import (SetValuedSelector,
                                              multi_positive_listwise_loss,
                                              paraphrase_consistency_loss)
from acquisition.schema import (
    GoldStore, PublicStore, SchemaError, derive_action_labels, observation_cache_key,
    validate_bundle,
)


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = write_fixture(Path(self.temp.name) / "bundle")
        public = PublicStore(self.root)
        self.context = {row["initial_caption"]: row["context_id"] for row in public.contexts.values()}
        self.candidate = {(row["image_id"], tuple(map(tuple, row["boxes"]))): row["candidate_id"]
                          for row in public.candidates.values()}
        self.image = {Path(row["image_path"]).stem: row["image_id"] for row in public.images.values()}

    def tearDown(self):
        self.temp.cleanup()

    def test_fixture_audits_and_atomic_actions_rederive(self):
        report = validate_bundle(self.root)
        self.assertEqual(report["status"], "pass", report["errors"])
        public, gold = PublicStore(self.root), GoldStore(self.root)
        derived = derive_action_labels(public, gold)
        supplied = {(row["context_id"], row["candidate_id"]):
                    {key: value for key, value in row.items() if key != "_source_line"}
                    for row in gold.tables["action_labels"]}
        self.assertEqual(derived, supplied)
        sparse = self.context["A person rides a bicycle."]
        enriched = self.context["A person rides with a child behind them."]
        rear = self.candidate[(self.image["im1"], ((0, 5, 10, 10),))]
        self.assertEqual(derived[(sparse, rear)]["new_supported_fact_ids"],
                         ["f1_child", "f1_rear"])
        self.assertEqual(derived[(enriched, rear)]["new_supported_fact_ids"], [])

    def test_selector_state_never_contains_gold_or_unexecuted_observations(self):
        public = PublicStore(self.root)
        sparse = self.context["A person rides a bicycle."]
        clothes = self.candidate[(self.image["im1"], ((0, 0, 5, 5),))]
        other = self.candidate[(self.image["im2"], ((0, 0, 5, 10),))]
        state = public.selector_state(sparse, [clothes])
        rendered = json.dumps(state, sort_keys=True)
        for forbidden in ("fact_id", "entity_id", "new_supported", "utility", "f1_red"):
            self.assertNotIn(forbidden, rendered)
        self.assertEqual([row["candidate_id"] for row in state["history"]], [clothes])
        self.assertNotIn("The rider wears red", json.dumps(
            public.selector_state(sparse), sort_keys=True))
        with self.assertRaises(SchemaError):
            public.selector_state(sparse, [other])

    def test_public_gold_key_is_hard_failure(self):
        path = self.root / "public" / "contexts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["target_fact_id"] = "f1_red"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        report = validate_bundle(self.root)
        self.assertEqual(report["status"], "fail")
        self.assertIn("gold-like public key", report["errors"][0])

    def test_duplicate_group_cross_split_is_hard_failure(self):
        path = self.root / "public" / "images.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["duplicate_group_id"] = rows[0]["duplicate_group_id"]
        rows[1]["split"] = "test"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        report = validate_bundle(self.root)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("crosses splits" in error for error in report["errors"]))

    def test_cache_key_ignores_context_and_gold(self):
        public = PublicStore(self.root)
        image = public.images[self.image["im1"]]
        candidate = public.candidates[self.candidate[(self.image["im1"], ((0, 0, 5, 5),))]]
        key = observation_cache_key(image["image_sha256"], candidate, "obs-v1", "prompt-v1")
        self.assertEqual(key, observation_cache_key(
            image["image_sha256"], candidate, "obs-v1", "prompt-v1"))
        self.assertNotEqual(key, observation_cache_key(
            image["image_sha256"], candidate, "obs-v2", "prompt-v1"))

    def test_conditioned_cache_key_retains_equal_text_context_rows(self):
        candidate = {"candidate_id": "cand", "kind": "box", "boxes": [[0, 0, 1, 1]]}
        common = ("image-sha", candidate, "region_only", "prompt-sha", "revision", 48, 50176)
        first = {"context_id": "ctx-a", "context_mode": "conditioned", "caption": "Same."}
        second = {"context_id": "ctx-b", "context_mode": "conditioned", "caption": "Same."}
        self.assertNotEqual(conditioned_cache_key(common[0], first, *common[1:]),
                            conditioned_cache_key(common[0], second, *common[1:]))

    def test_r2_review_pair_requires_both_claim_label_sets(self):
        self.assertEqual(required_fields({"alternate_claim": ""}), BASE_FIELDS)
        self.assertEqual(required_fields({"alternate_claim": "Another claim."}),
                         BASE_FIELDS + PAIR_FIELDS)

    def test_set_valued_losses_and_permutation_equivariance(self):
        logits = torch.tensor([[2.0, 1.0, -3.0]], requires_grad=True)
        positive = torch.tensor([[True, True, False]])
        valid = torch.tensor([[True, True, False]])
        loss = multi_positive_listwise_loss(logits, positive, valid)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertAlmostEqual(float(paraphrase_consistency_loss(
            logits.detach(), logits.detach(), valid)), 0.0, places=6)
        model = SetValuedSelector(3, 2, hidden_dim=8, heads=2, layers=1, dropout=0).eval()
        candidates = torch.randn(2, 4, 3)
        contexts = torch.randn(2, 2)
        mask = torch.ones(2, 4, dtype=torch.bool)
        permutation = torch.tensor([2, 0, 3, 1])
        with torch.no_grad():
            original = model(candidates, contexts, mask)
            permuted = model(candidates[:, permutation], contexts, mask[:, permutation])
        self.assertTrue(torch.allclose(original[:, permutation], permuted, atol=1e-6))

    def test_e1_e2_report_is_diagnostic_without_e0_and_strong_baseline(self):
        report = evaluate_bundle(self.root, "proposal_score", bootstrap_samples=200, seed=7)
        self.assertEqual(report["protocol"]["frozen_baseline_method"], "proposal_score")
        self.assertGreater(report["e1"]["methods"]["recognition_oracle"]
                           ["delta_coverage"]["mean"],
                           report["e1"]["methods"]["proposal_score"]
                           ["delta_coverage"]["mean"])
        self.assertGreater(report["e2"]["caption_necessity_gap"]["mean"], 0)
        self.assertEqual(report["e2"]["paraphrase_gold_action_regret_data_check"]["mean"], 0)
        self.assertFalse(report["protocol"]["evidence_eligible"])
        self.assertFalse(report["e1"]["gate_pass"])
        self.assertFalse(report["e2"]["gate_pass"])

    def test_missing_atomic_matrix_row_is_hard_failure(self):
        path = self.root / "gold" / "context_fact_labels.jsonl"
        rows = path.read_text().splitlines()
        path.write_text("\n".join(rows[:-1]) + "\n")
        report = validate_bundle(self.root)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("exactly cover" in error for error in report["errors"]))

    def test_single_vote_and_unadjudicated_label_are_hard_failures(self):
        path = self.root / "gold" / "candidate_fact_labels.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["annotator_votes"] = rows[0]["annotator_votes"][:1]
        rows[0]["adjudication"] = None
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        report = validate_bundle(self.root)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("two independent" in error for error in report["errors"]))

    def test_public_context_has_no_experimental_condition(self):
        public = PublicStore(self.root)
        rendered = json.dumps(public.tables["contexts"])
        self.assertNotIn("context_kind", rendered)
        self.assertNotIn("reference_context_id", rendered)

    def test_semantic_public_identifier_is_hard_failure(self):
        path = self.root / "public" / "contexts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["context_id"] = "ctx_sparse_caption"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        report = validate_bundle(self.root)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("opaque hashed" in error for error in report["errors"]))

    def test_duplicate_observation_and_repeated_history_are_hard_failures(self):
        path = self.root / "public" / "observations.jsonl"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines + [lines[0]]) + "\n")
        self.assertEqual(validate_bundle(self.root)["status"], "fail")

    def test_caption_metrics_are_tie_aware_and_order_invariant(self):
        scores = {
            "a": {"x": 0.4, "y": 0.2, "stop": 0.0},
            "b": {"x": 0.0, "y": 0.2, "stop": 0.0},
        }
        gap = caption_necessity(scores)
        self.assertAlmostEqual(gap["caption_necessity_gap"], 0.1)
        self.assertTrue(material_switch(scores["a"], scores["b"], epsilon=0.01,
                                        material_loss=0.05))
        self.assertEqual(pair_regret({"x": 0.2, "y": 0.2}, {"x": 0.3, "y": 0.3}), 0)
        reversed_scores = {context: dict(reversed(list(values.items())))
                           for context, values in scores.items()}
        self.assertEqual(caption_necessity(scores), caption_necessity(reversed_scores))

    def test_bootstrap_resamples_duplicate_groups_and_is_reproducible(self):
        a = bootstrap_cluster([0.0] * 20 + [1.0], ["one"] * 20 + ["two"],
                              samples=500, seed=11)
        b = bootstrap_cluster([0.0] * 20 + [1.0], ["one"] * 20 + ["two"],
                              samples=500, seed=11)
        self.assertEqual(a, b)
        self.assertEqual(a["clusters"], 2)
        self.assertAlmostEqual(a["mean"], 0.5)

    def test_fixture_can_never_pass_e0_gate(self):
        report = audit_report(self.root)
        self.assertEqual(report["status"], "not_eligible")
        self.assertTrue(report["e0_gate"]["synthetic_or_fixture"])
        self.assertEqual(set(report["e0_gate"]["table_checks"]), {
            "context_status", "candidate_visibility", "candidate_sufficiency",
            "observation_claim_status",
        })

    def test_model_output_parsers_accept_only_declared_aliases(self):
        candidates = [{"kind": "bbox"}, {"kind": "stop"}]
        self.assertEqual(parse_choice("1", candidates), 1)
        self.assertEqual(parse_choice("STOP", candidates), 1)
        self.assertEqual(parse_entailment("NOT ENTAILED."), "NOT_ENTAILED")
        self.assertEqual(parse_visual_support("SUPPORTIVE"), "SUPPORTED")
        self.assertEqual(parse_independent_label("Answer: INVALID."), "INVALID")
        self.assertIsNone(parse_independent_label("SUPPORTED or INVALID"))
        self.assertIsNone(parse_entailment("probably new"))

    def test_random_rate_is_ratio_of_expected_counts(self):
        actions = {
            "a": {"new_supported_fact_ids": ["x", "y"], "contradicted_claim_ids": [],
                  "repeated_fact_ids": [], "eligible_missing_fact_ids": ["x", "y"],
                  "unknown_claim_ids": [], "pixel_cost": 1},
            "b": {"new_supported_fact_ids": [], "contradicted_claim_ids": ["bad"],
                  "repeated_fact_ids": [], "eligible_missing_fact_ids": ["x", "y"],
                  "unknown_claim_ids": [], "pixel_cost": 1},
        }
        result = expected_random_metrics(["a", "b"], actions, fact_count=2)
        self.assertAlmostEqual(result["invalid_claim_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
