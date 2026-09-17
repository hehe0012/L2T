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
        hidden_dim: int = 32,
        max_steps: int = 4,
        mdl_weight: float = 0.95,
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
                    nn.Linear(hidden_dim, codebook_size),
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
                rollout_steps: int | None = None,
                gumbel_tau: float | None = None,
            ):
                return inner_self.outer._forward_module(
                    inner_self,
                    support_x,
                    support_y,
                    query_x,
                    query_y,
                    hard=hard,
                    sample_count=sample_count,
                    rollout_steps=rollout_steps,
                    gumbel_tau=gumbel_tau,
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

    def _programmer_logits(self, module, state_dist, target_y):
        state_emb = state_dist @ module.state_emb.weight
        target_emb = module.state_emb(target_y)
        return module.programmer(self.torch.cat([state_emb, target_emb], dim=-1))

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
        rollout_steps: int | None = None,
        gumbel_tau: float | None = None,
    ):
        torch = self.torch
        nn = self.nn
        batch_size = support_x.shape[0]
        rollout_steps = self.max_steps if rollout_steps is None else rollout_steps
        if rollout_steps < 1 or rollout_steps > self.max_steps:
            raise ValueError(f"rollout_steps must be in [1, {self.max_steps}]")
        if sample_count > 0:
            return self._sampled_forward(
                module,
                support_x,
                support_y,
                query_x,
                query_y,
                sample_count=sample_count,
                rollout_steps=rollout_steps,
            )

        support_dist = nn.functional.one_hot(support_x, num_classes=self.num_states).float()
        query_dist = nn.functional.one_hot(query_x, num_classes=self.num_states).float()
        code_logits_by_step = []
        support_logits_by_step = []
        query_logits_by_step = []

        for _ in range(rollout_steps):
            step_logits = self._programmer_logits(module, support_dist, support_y)
            if hard:
                hard_codes = step_logits.argmax(dim=-1)
                code_probs = nn.functional.one_hot(hard_codes, num_classes=self.codebook_size).float()
            elif gumbel_tau is not None:
                code_probs = nn.functional.gumbel_softmax(step_logits, tau=gumbel_tau, hard=True)
            else:
                code_probs = step_logits.softmax(dim=-1)

            support_dist, support_logits = self._step_distribution(module, support_dist, code_probs)
            query_dist, query_logits = self._step_distribution(module, query_dist, code_probs)
            code_logits_by_step.append(step_logits)
            support_logits_by_step.append(support_logits)
            query_logits_by_step.append(query_logits)

        support_losses = torch.stack(
            [nn.functional.nll_loss(logits, support_y, reduction="none") for logits in support_logits_by_step],
            dim=1,
        )
        length_penalty = self.mdl_weight * torch.arange(
            1,
            rollout_steps + 1,
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
            code_logits=torch.stack(code_logits_by_step, dim=1),
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
        *,
        sample_count: int,
        rollout_steps: int,
    ):
        """NEO-S inference: sample candidate programs and keep the best support explanation."""

        if sample_count < 1:
            raise ValueError("sample_count must be positive")

        torch = self.torch
        nn = self.nn
        batch_size = support_x.shape[0]

        candidate_count = sample_count + 1
        flat_support_x = support_x[:, None].expand(batch_size, candidate_count).reshape(-1)
        flat_query_x = query_x[:, None].expand(batch_size, candidate_count).reshape(-1)
        flat_support_y = support_y[:, None].expand(batch_size, candidate_count).reshape(-1)

        support_dist = nn.functional.one_hot(flat_support_x, num_classes=self.num_states).float()
        query_dist = nn.functional.one_hot(flat_query_x, num_classes=self.num_states).float()
        code_logits_by_step = []
        support_logits_by_step = []
        query_logits_by_step = []

        for _ in range(rollout_steps):
            step_logits = self._programmer_logits(module, support_dist, flat_support_y)
            step_logits_view = step_logits.view(batch_size, candidate_count, self.codebook_size)
            greedy_codes = step_logits_view[:, :1].argmax(dim=-1)
            sampled_codes = torch.multinomial(
                step_logits_view[:, 1:].reshape(batch_size * sample_count, self.codebook_size).softmax(dim=-1),
                num_samples=1,
                replacement=True,
            ).view(batch_size, sample_count)
            step_codes = torch.cat([greedy_codes, sampled_codes], dim=1).reshape(-1)
            step_probs = nn.functional.one_hot(step_codes, num_classes=self.codebook_size).float()
            support_dist, support_logits = self._step_distribution(module, support_dist, step_probs)
            query_dist, query_logits = self._step_distribution(module, query_dist, step_probs)
            code_logits_by_step.append(step_logits_view[:, 0])
            support_logits_by_step.append(support_logits)
            query_logits_by_step.append(query_logits)

        support_losses = torch.stack(
            [
                nn.functional.nll_loss(logits, flat_support_y, reduction="none")
                for logits in support_logits_by_step
            ],
            dim=1,
        )
        support_losses = support_losses.view(batch_size, candidate_count, rollout_steps)
        length_penalty = self.mdl_weight * torch.arange(
            1,
            rollout_steps + 1,
            device=support_x.device,
            dtype=support_losses.dtype,
        )
        scores = support_losses + length_penalty[None, None, :]
        best_flat = scores.reshape(batch_size, candidate_count * rollout_steps).argmin(dim=1)
        chosen_candidate = best_flat // rollout_steps
        chosen_lengths = best_flat % rollout_steps
        row_index = torch.arange(batch_size, device=support_x.device)

        support_loss = support_losses[row_index, chosen_candidate, chosen_lengths].mean()
        support_logits_stacked = torch.stack(support_logits_by_step, dim=1)
        query_logits_stacked = torch.stack(query_logits_by_step, dim=1)
        support_logits_stacked = support_logits_stacked.view(
            batch_size,
            candidate_count,
            rollout_steps,
            self.num_states,
        )
        query_logits_stacked = query_logits_stacked.view(
            batch_size,
            candidate_count,
            rollout_steps,
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
            code_logits=torch.stack(code_logits_by_step, dim=1),
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
        hidden_dim: int = 32,
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
