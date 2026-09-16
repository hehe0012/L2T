"""Initial PyTorch models for the GridWorld reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelOutput:
    loss: object
    support_loss: object
    query_loss: object
    chosen_lengths: object
    code_logits: object
    support_logits: object
    query_logits: object


class NEOGridWorld:
    """A compact NEO-style latent program learner for GridWorld.

    This model infers a K-step discrete latent program from a support pair and
    executes the same inferred program on a query input. It is intentionally
    small so the first reproduction milestone can be run quickly.
    """

    def __init__(
        self,
        *,
        grid_size: int = 10,
        codebook_size: int = 6,
        hidden_dim: int = 128,
        max_steps: int = 4,
        mdl_weight: float = 0.01,
    ) -> None:
        import torch
        from torch import nn

        self.torch = torch
        self.nn = nn
        self.grid_size = grid_size
        self.num_states = grid_size * grid_size
        self.codebook_size = codebook_size
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.mdl_weight = mdl_weight

        class _Module(nn.Module):
            def __init__(inner_self, outer: "NEOGridWorld") -> None:
                super().__init__()
                inner_self.outer = outer
                inner_self.state_emb = nn.Embedding(outer.num_states, hidden_dim)
                inner_self.code_emb = nn.Embedding(codebook_size, hidden_dim)
                inner_self.programmer = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, max_steps * codebook_size),
                )
                inner_self.executor = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, outer.num_states),
                )

            def forward(
                inner_self,
                support_x,
                support_y,
                query_x,
                query_y=None,
                *,
                hard: bool = False,
                sample_count: int = 0,
            ):
                return inner_self.outer._forward_module(
                    inner_self,
                    support_x,
                    support_y,
                    query_x,
                    query_y,
                    hard=hard,
                    sample_count=sample_count,
                )

        self.module = _Module(self)

    def parameters(self):
        return self.module.parameters()

    def to(self, device):
        self.module.to(device)
        return self

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _step_distribution(self, module, state_dist, code_probs):
        torch = self.torch
        state_emb = state_dist @ module.state_emb.weight
        code_emb = module.code_emb.weight
        batch_size = state_emb.shape[0]
        state_expanded = state_emb[:, None, :].expand(batch_size, self.codebook_size, self.hidden_dim)
        code_expanded = code_emb[None, :, :].expand(batch_size, self.codebook_size, self.hidden_dim)
        logits_per_code = module.executor(
            torch.cat([state_expanded, code_expanded], dim=-1)
        )
        probs_per_code = logits_per_code.softmax(dim=-1)
        next_dist = (code_probs[:, :, None] * probs_per_code).sum(dim=1)
        next_logits = torch.log(next_dist.clamp_min(1e-8))
        return next_dist, next_logits

    def _forward_module(
        self,
        module,
        support_x,
        support_y,
        query_x,
        query_y=None,
        *,
        hard: bool = False,
        sample_count: int = 0,
    ):
        torch = self.torch
        nn = self.nn
        batch_size = support_x.shape[0]
        sx_emb = module.state_emb(support_x)
        sy_emb = module.state_emb(support_y)
        code_logits = module.programmer(torch.cat([sx_emb, sy_emb], dim=-1))
        code_logits = code_logits.view(batch_size, self.max_steps, self.codebook_size)
        code_probs = code_logits.softmax(dim=-1)
        if sample_count > 0:
            return self._sampled_forward(
                module,
                support_x,
                support_y,
                query_x,
                query_y,
                code_logits,
                code_probs,
                sample_count=sample_count,
            )
        if hard:
            hard_codes = code_probs.argmax(dim=-1)
            code_probs = nn.functional.one_hot(hard_codes, num_classes=self.codebook_size).float()

        support_dist = nn.functional.one_hot(support_x, num_classes=self.num_states).float()
        query_dist = nn.functional.one_hot(query_x, num_classes=self.num_states).float()
        support_logits_by_step = []
        query_logits_by_step = []

        for step in range(self.max_steps):
            support_dist, support_logits = self._step_distribution(module, support_dist, code_probs[:, step])
            query_dist, query_logits = self._step_distribution(module, query_dist, code_probs[:, step])
            support_logits_by_step.append(support_logits)
            query_logits_by_step.append(query_logits)

        support_losses = torch.stack(
            [nn.functional.nll_loss(logits, support_y, reduction="none") for logits in support_logits_by_step],
            dim=1,
        )
        length_penalty = self.mdl_weight * torch.arange(
            1,
            self.max_steps + 1,
            device=support_x.device,
            dtype=support_losses.dtype,
        )
        chosen_lengths = (support_losses + length_penalty[None, :]).argmin(dim=1)
        row_index = torch.arange(batch_size, device=support_x.device)
        support_loss = support_losses[row_index, chosen_lengths].mean()

        chosen_support_logits = torch.stack(support_logits_by_step, dim=1)[row_index, chosen_lengths]
        chosen_query_logits = torch.stack(query_logits_by_step, dim=1)[row_index, chosen_lengths]
        if query_y is None:
            query_loss = support_loss.new_zeros(())
        else:
            query_loss = nn.functional.nll_loss(chosen_query_logits, query_y)

        loss = support_loss + query_loss
        return ModelOutput(
            loss=loss,
            support_loss=support_loss,
            query_loss=query_loss,
            chosen_lengths=chosen_lengths + 1,
            code_logits=code_logits,
            support_logits=chosen_support_logits,
            query_logits=chosen_query_logits,
        )

    def _sampled_forward(
        self,
        module,
        support_x,
        support_y,
        query_x,
        query_y,
        code_logits,
        code_probs,
        *,
        sample_count: int,
    ):
        """NEO-S inference: sample candidate programs and keep the best support explanation."""

        if sample_count < 1:
            raise ValueError("sample_count must be positive")

        torch = self.torch
        nn = self.nn
        batch_size = support_x.shape[0]

        sampled = torch.multinomial(
            code_probs.reshape(batch_size * self.max_steps, self.codebook_size),
            num_samples=sample_count,
            replacement=True,
        )
        sampled = sampled.view(batch_size, self.max_steps, sample_count).permute(0, 2, 1)
        greedy = code_probs.argmax(dim=-1)[:, None, :]
        candidate_codes = torch.cat([greedy, sampled], dim=1)
        candidate_count = candidate_codes.shape[1]

        flat_codes = candidate_codes.reshape(batch_size * candidate_count, self.max_steps)
        flat_support_x = support_x[:, None].expand(batch_size, candidate_count).reshape(-1)
        flat_query_x = query_x[:, None].expand(batch_size, candidate_count).reshape(-1)
        flat_support_y = support_y[:, None].expand(batch_size, candidate_count).reshape(-1)

        support_dist = nn.functional.one_hot(flat_support_x, num_classes=self.num_states).float()
        query_dist = nn.functional.one_hot(flat_query_x, num_classes=self.num_states).float()
        support_logits_by_step = []
        query_logits_by_step = []

        for step in range(self.max_steps):
            step_probs = nn.functional.one_hot(
                flat_codes[:, step],
                num_classes=self.codebook_size,
            ).float()
            support_dist, support_logits = self._step_distribution(module, support_dist, step_probs)
            query_dist, query_logits = self._step_distribution(module, query_dist, step_probs)
            support_logits_by_step.append(support_logits)
            query_logits_by_step.append(query_logits)

        support_losses = torch.stack(
            [
                nn.functional.nll_loss(logits, flat_support_y, reduction="none")
                for logits in support_logits_by_step
            ],
            dim=1,
        )
        support_losses = support_losses.view(batch_size, candidate_count, self.max_steps)
        length_penalty = self.mdl_weight * torch.arange(
            1,
            self.max_steps + 1,
            device=support_x.device,
            dtype=support_losses.dtype,
        )
        scores = support_losses + length_penalty[None, None, :]
        best_flat = scores.reshape(batch_size, candidate_count * self.max_steps).argmin(dim=1)
        chosen_candidate = best_flat // self.max_steps
        chosen_lengths = best_flat % self.max_steps
        row_index = torch.arange(batch_size, device=support_x.device)

        support_loss = support_losses[row_index, chosen_candidate, chosen_lengths].mean()
        support_logits_stacked = torch.stack(support_logits_by_step, dim=1)
        query_logits_stacked = torch.stack(query_logits_by_step, dim=1)
        support_logits_stacked = support_logits_stacked.view(
            batch_size,
            candidate_count,
            self.max_steps,
            self.num_states,
        )
        query_logits_stacked = query_logits_stacked.view(
            batch_size,
            candidate_count,
            self.max_steps,
            self.num_states,
        )
        chosen_support_logits = support_logits_stacked[row_index, chosen_candidate, chosen_lengths]
        chosen_query_logits = query_logits_stacked[row_index, chosen_candidate, chosen_lengths]

        if query_y is None:
            query_loss = support_loss.new_zeros(())
        else:
            query_loss = nn.functional.nll_loss(chosen_query_logits, query_y)

        return ModelOutput(
            loss=support_loss + query_loss,
            support_loss=support_loss,
            query_loss=query_loss,
            chosen_lengths=chosen_lengths + 1,
            code_logits=code_logits,
            support_logits=chosen_support_logits,
            query_logits=chosen_query_logits,
        )


class DiscMonoGridWorld(NEOGridWorld):
    """One-code monolithic baseline with the same executor interface."""

    def __init__(
        self,
        *,
        grid_size: int = 10,
        codebook_size: int = 32,
        hidden_dim: int = 128,
        max_steps: int = 1,
        mdl_weight: float = 0.0,
    ) -> None:
        super().__init__(
            grid_size=grid_size,
            codebook_size=codebook_size,
            hidden_dim=hidden_dim,
            max_steps=max_steps,
            mdl_weight=mdl_weight,
        )


def build_model(name: Literal["neo", "neo_s", "disc_mono"], **kwargs):
    if name in {"neo", "neo_s"}:
        return NEOGridWorld(**kwargs)
    if name == "disc_mono":
        kwargs.setdefault("max_steps", 1)
        kwargs.setdefault("codebook_size", 32)
        return DiscMonoGridWorld(**kwargs)
    raise ValueError(f"unknown model: {name}")
