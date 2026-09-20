import unittest

from l2t.gridworld import (
    GridState,
    apply_action,
    apply_program,
    enumerate_programs,
    generate_examples,
    parse_program,
    split_programs,
)
from l2t.models import build_model


class GridWorldTests(unittest.TestCase):
    def test_state_roundtrip(self):
        for index in [0, 7, 42, 99]:
            state = GridState.from_index(index, size=10)
            self.assertEqual(state.to_index(size=10), index)

    def test_wrap_boundary(self):
        self.assertEqual(apply_action(GridState(0, 0), "up").to_index(), 90)
        self.assertEqual(apply_action(GridState(0, 0), "left").to_index(), 9)

    def test_clamp_boundary(self):
        self.assertEqual(apply_action(GridState(0, 0), "up", boundary="clamp").to_index(), 0)
        self.assertEqual(apply_action(GridState(0, 0), "left", boundary="clamp").to_index(), 0)

    def test_program_execution(self):
        state = apply_program(GridState(5, 5), (0, 2, 1, 3))
        self.assertEqual(state, GridState(5, 5))

    def test_program_enumeration_count(self):
        self.assertEqual(len(enumerate_programs(min_len=1, max_len=3)), 4 + 16 + 64)

    def test_program_split(self):
        train, comp = split_programs(alpha=0.5)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(comp), 0)
        self.assertEqual(len(train) + len(comp), 84)

    def test_program_split_includes_gridworld_anchors(self):
        train, _ = split_programs(alpha=0.33, seed=0)
        self.assertIn((0, 0, 0), train)
        self.assertIn((1, 1, 1), train)
        self.assertIn((2, 2, 2), train)
        self.assertIn((3, 3, 3), train)

    def test_paper_program_split_matches_listed_gridworld_programs(self):
        train, comp = split_programs(alpha=0.33, split_mode="paper")

        self.assertEqual(
            train,
            [
                parse_program(spec)
                for spec in ("UUU", "DDD", "LLL", "RRR", "U", "LU", "DL", "DR", "DD", "DDL", "DRR")
            ],
        )
        self.assertEqual(
            comp,
            [
                parse_program(spec)
                for spec in ("L", "R", "D", "LL", "RR", "UU", "RU", "RRL", "LLU", "DLL", "RUU", "DDR", "RRU")
            ],
        )

    def test_paper_singleton_swap_moves_u_to_comp_and_l_to_train(self):
        train, comp = split_programs(alpha=0.33, split_mode="paper", paper_singleton="L")
        self.assertIn(parse_program("L"), train)
        self.assertNotIn(parse_program("U"), train)
        self.assertIn(parse_program("U"), comp)
        self.assertNotIn(parse_program("L"), comp)

    def test_paper_all_singletons_moves_all_length_one_programs_to_train(self):
        train, comp = split_programs(alpha=0.33, split_mode="paper", paper_singleton="ALL")
        for spec in ("U", "D", "L", "R"):
            self.assertIn(parse_program(spec), train)
            self.assertNotIn(parse_program(spec), comp)

    def test_generated_example_transfer_rule(self):
        examples = generate_examples(n=20, split="train", seed=0, alpha=1.0)
        for item in examples:
            sx = GridState.from_index(item.support_x)
            qx = GridState.from_index(item.query_x)
            self.assertEqual(apply_program(sx, item.program).to_index(), item.support_y)
            self.assertEqual(apply_program(qx, item.program).to_index(), item.query_y)

    def test_custom_length_splits(self):
        long_train = generate_examples(
            n=20,
            split="train",
            seed=0,
            alpha=1.0,
            train_min_len=4,
            train_max_len=8,
        )
        short_eval = generate_examples(
            n=20,
            split="length_ood",
            seed=1,
            alpha=1.0,
            train_min_len=4,
            train_max_len=8,
            length_ood_min_len=1,
            length_ood_max_len=3,
        )

        self.assertTrue(all(4 <= len(item.program) <= 8 for item in long_train))
        self.assertTrue(all(1 <= len(item.program) <= 3 for item in short_eval))

    def test_neo_s_sampled_forward_shapes(self):
        import torch

        model = build_model("neo_s", grid_size=10, codebook_size=6, hidden_dim=16, max_steps=4)
        support_x = torch.tensor([0, 11, 22])
        support_y = torch.tensor([1, 12, 23])
        query_x = torch.tensor([30, 41, 52])
        query_y = torch.tensor([31, 42, 53])

        output = model(
            support_x,
            support_y,
            query_x,
            query_y,
            hard=True,
            sample_count=3,
            rollout_steps=4,
        )

        self.assertEqual(tuple(output.support_logits.shape), (3, 100))
        self.assertEqual(tuple(output.query_logits.shape), (3, 100))
        self.assertEqual(tuple(output.chosen_lengths.shape), (3,))
        self.assertTrue(torch.all(output.chosen_lengths >= 1))
        self.assertTrue(torch.all(output.chosen_lengths <= 4))

    def test_neo_s_majority_exact_reports_selection_stats(self):
        import torch

        model = build_model(
            "neo_s",
            grid_size=10,
            codebook_size=6,
            hidden_dim=16,
            max_steps=4,
            neo_s_selection="majority_exact",
        )
        support_x = torch.tensor([0, 11, 22])
        support_y = torch.tensor([1, 12, 23])
        query_x = torch.tensor([30, 41, 52])
        query_y = torch.tensor([31, 42, 53])

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

    def test_neo_forward_reports_vq_and_grounding_losses(self):
        import torch

        model = build_model("neo", grid_size=10, codebook_size=6, hidden_dim=16, max_steps=4)
        support_x = torch.tensor([0, 11])
        support_y = torch.tensor([1, 12])
        query_x = torch.tensor([30, 41])
        query_y = torch.tensor([31, 42])

        output = model(
            support_x,
            support_y,
            query_x,
            query_y,
            rollout_steps=4,
            gumbel_tau=0.3,
        )

        self.assertIsNotNone(output.vq_loss)
        self.assertIsNotNone(output.grounding_loss)
        self.assertGreaterEqual(float(output.vq_loss.detach()), 0.0)
        self.assertGreaterEqual(float(output.grounding_loss.detach()), 0.0)

    def test_cont_mono_forward_uses_continuous_single_step_action(self):
        import torch

        model = build_model("cont_mono", grid_size=10, hidden_dim=16, action_dim=8, ff_dim=32)
        support_x = torch.tensor([0, 11])
        support_y = torch.tensor([1, 12])
        query_x = torch.tensor([30, 41])
        query_y = torch.tensor([31, 42])

        output = model(
            support_x,
            support_y,
            query_x,
            query_y,
            rollout_steps=1,
        )

        self.assertEqual(tuple(output.support_logits.shape), (2, 100))
        self.assertEqual(tuple(output.query_logits.shape), (2, 100))
        self.assertEqual(tuple(output.chosen_lengths.shape), (2,))
        self.assertIsNotNone(output.vq_loss)
        self.assertGreaterEqual(float(output.vq_loss.detach()), 0.0)

    def test_recurrent_policy_supports_longer_eval_than_train_rollout(self):
        import torch

        model = build_model("neo", grid_size=10, codebook_size=6, hidden_dim=16, max_steps=10)
        support_x = torch.tensor([0, 11])
        support_y = torch.tensor([1, 12])
        query_x = torch.tensor([30, 41])
        query_y = torch.tensor([31, 42])

        train_output = model(
            support_x,
            support_y,
            query_x,
            query_y,
            rollout_steps=4,
        )
        length_output = model(
            support_x,
            support_y,
            query_x,
            query_y,
            hard=True,
            rollout_steps=10,
        )

        self.assertEqual(tuple(train_output.code_logits.shape), (2, 4, 6))
        self.assertEqual(tuple(length_output.code_logits.shape), (2, 10, 6))


if __name__ == "__main__":
    unittest.main()
