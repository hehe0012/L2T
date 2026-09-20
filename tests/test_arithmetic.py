import unittest

from l2t.arithmetic import (
    ANCHOR_PROGRAMS,
    apply_exponent_program,
    apply_factor_program,
    enumerate_programs,
    exponents_to_value,
    generate_examples,
    split_programs,
    value_to_exponents,
)
from l2t.arithmetic_models import build_arithmetic_model


class ArithmeticTests(unittest.TestCase):
    def test_program_execution(self):
        program = (0, 1, 2, 3)
        self.assertEqual(apply_factor_program(1, program), 210)
        self.assertEqual(apply_exponent_program((0, 0, 0, 0), program), (1, 1, 1, 1))

    def test_exponent_roundtrip(self):
        exponents = (3, 2, 1, 4)
        self.assertEqual(value_to_exponents(exponents_to_value(exponents)), exponents)

    def test_program_enumeration_count(self):
        self.assertEqual(len(enumerate_programs(min_len=1, max_len=3)), 4 + 16 + 64)
        self.assertEqual(len(enumerate_programs(min_len=4, max_len=6)), 4**4 + 4**5 + 4**6)

    def test_program_split_includes_arithmetic_anchors(self):
        train, comp = split_programs(alpha=0.33, seed=0)
        for anchor in ANCHOR_PROGRAMS:
            self.assertIn(anchor, train)
        self.assertGreater(len(comp), 0)

    def test_generated_example_transfer_rule(self):
        examples = generate_examples(n=20, split="train", seed=0, alpha=1.0)
        for item in examples:
            self.assertEqual(apply_factor_program(item.support_x, item.program), item.support_y)
            self.assertEqual(apply_factor_program(item.query_x, item.program), item.query_y)
            self.assertEqual(apply_exponent_program(item.support_x_exp, item.program), item.support_y_exp)
            self.assertEqual(apply_exponent_program(item.query_x_exp, item.program), item.query_y_exp)

    def test_arithmetic_neo_forward_shapes(self):
        import torch

        model = build_arithmetic_model(
            "neo",
            state_dim=8,
            max_exponent=8,
            codebook_size=16,
            policy_layers=1,
            transition_layers=1,
            max_steps=4,
        )
        support_x = torch.tensor([[0, 0, 0, 0], [1, 0, 2, 0]])
        support_y = torch.tensor([[1, 0, 0, 0], [1, 1, 2, 0]])
        query_x = torch.tensor([[0, 1, 0, 0], [2, 0, 0, 1]])
        query_y = torch.tensor([[1, 1, 0, 0], [2, 1, 0, 1]])

        output = model(support_x, support_y, query_x, query_y, rollout_steps=3, gumbel_tau=0.3)

        self.assertEqual(tuple(output.support_logits.shape), (2, 4, 9))
        self.assertEqual(tuple(output.query_logits.shape), (2, 4, 9))
        self.assertEqual(tuple(output.code_logits.shape), (2, 3, 16))
        self.assertIsNotNone(output.vq_loss)
        self.assertIsNotNone(output.grounding_loss)

    def test_arithmetic_neo_s_sampled_forward_shapes(self):
        import torch

        model = build_arithmetic_model(
            "neo_s",
            state_dim=8,
            max_exponent=8,
            codebook_size=16,
            policy_layers=1,
            transition_layers=1,
            max_steps=4,
        )
        support_x = torch.tensor([[0, 0, 0, 0], [1, 0, 2, 0]])
        support_y = torch.tensor([[1, 0, 0, 0], [1, 1, 2, 0]])
        query_x = torch.tensor([[0, 1, 0, 0], [2, 0, 0, 1]])
        query_y = torch.tensor([[1, 1, 0, 0], [2, 1, 0, 1]])

        output = model(
            support_x,
            support_y,
            query_x,
            query_y,
            hard=True,
            sample_count=3,
            rollout_steps=4,
        )

        self.assertEqual(tuple(output.support_logits.shape), (2, 4, 9))
        self.assertEqual(tuple(output.query_logits.shape), (2, 4, 9))
        self.assertEqual(tuple(output.chosen_lengths.shape), (2,))
        self.assertTrue(torch.all(output.chosen_lengths >= 1))
        self.assertTrue(torch.all(output.chosen_lengths <= 4))

    def test_arithmetic_neo_s_majority_exact_reports_selection_stats(self):
        import torch

        model = build_arithmetic_model(
            "neo_s",
            state_dim=8,
            max_exponent=8,
            codebook_size=16,
            policy_layers=1,
            transition_layers=1,
            max_steps=4,
            neo_s_selection="majority_exact",
        )
        support_x = torch.tensor([[0, 0, 0, 0], [1, 0, 2, 0]])
        support_y = torch.tensor([[1, 0, 0, 0], [1, 1, 2, 0]])
        query_x = torch.tensor([[0, 1, 0, 0], [2, 0, 0, 1]])
        query_y = torch.tensor([[1, 1, 0, 0], [2, 1, 0, 1]])

        output = model(
            support_x,
            support_y,
            query_x,
            query_y,
            hard=True,
            sample_count=3,
            rollout_steps=4,
        )

        self.assertIsNotNone(output.exact_candidate_rate)
        self.assertIsNotNone(output.fallback_rate)
        self.assertIsNotNone(output.majority_count)
        self.assertGreaterEqual(float(output.exact_candidate_rate), 0.0)
        self.assertLessEqual(float(output.exact_candidate_rate), 1.0)
        self.assertGreaterEqual(float(output.fallback_rate), 0.0)
        self.assertLessEqual(float(output.fallback_rate), 1.0)


if __name__ == "__main__":
    unittest.main()
