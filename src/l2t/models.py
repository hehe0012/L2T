"""PyTorch models for the GridWorld reproduction."""

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
    vq_loss: object | None = None
    grounding_loss: object | None = None


class NEOGridWorld:
    """NEO-style latent program learner for GridWorld."""

    def __init__(
        self,
        *,
        grid_size: int = 10,
        codebook_size: int = 6,
        hidden_dim: int = 32,
        action_dim: int = 16,
        ff_dim: int = 128,
        max_steps: int = 4,
        mdl_weight: float = 0.95,
        commitment_cost: float = 0.25,
        vq_loss_weight: float = 1.0,
        grounding_loss_weight: float = 0.1,
    ) -> None:
        import torch
        from torch import nn

        self.torch = torch
        self.nn = nn
        self.grid_size = grid_size
        self.num_states = grid_size * grid_size
        self.codebook_size = codebook_size
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ff_dim = ff_dim
        self.max_steps = max_steps
        self.mdl_weight = mdl_weight
        self.commitment_cost = commitment_cost
        self.vq_loss_weight = vq_loss_weight
        self.grounding_loss_weight = grounding_loss_weight

        class FiLMMLP(nn.Module):
            def __init__(inner_self, input_dim: int, cond_dim: int, hidden: int, output_dim: int) -> None:
                super().__init__()
                inner_self.input_proj = nn.Linear(input_dim, hidden)
                inner_self.film = nn.Linear(cond_dim, hidden * 2)
                inner_self.output = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, output_dim),
                )

            def forward(inner_self, x, cond):
                hidden = inner_self.input_proj(x)
                gamma, beta = inner_self.film(cond).chunk(2, dim=-1)
                hidden = hidden * (1.0 + gamma) + beta
                return inner_self.output(hidden)

        class _Module(nn.Module):
            def __init__(inner_self, outer: "NEOGridWorld") -> None:
                super().__init__()
                inner_self.outer = outer
                inner_self.state_encoder = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(32 * grid_size * grid_size, ff_dim),
                    nn.ReLU(),
                    nn.Linear(ff_dim, hidden_dim),
                )
                inner_self.state_decoder = nn.Sequential(
                    nn.Linear(hidden_dim, ff_dim),
                    nn.ReLU(),
                    nn.Linear(ff_dim, 32 * grid_size * grid_size),
                    nn.ReLU(),
                    nn.Unflatten(1, (32, grid_size, grid_size)),
                    nn.Conv2d(32, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 1, kernel_size=1),
                )
                inner_self.policy = FiLMMLP(hidden_dim, hidden_dim, ff_dim, action_dim)
                inner_self.action_codebook = nn.Embedding(codebook_size, action_dim)
                inner_self.transition = FiLMMLP(hidden_dim, action_dim, ff_dim, hidden_dim)

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

    def _state_image(self, indices):
        nn = self.nn
        image = nn.functional.one_hot(indices, num_classes=self.num_states).float()
        return image.view(indices.shape[0], 1, self.grid_size, self.grid_size)

    def _decode_logits(self, module, state_latent):
        return module.state_decoder(state_latent).flatten(1)

    def _encode_decoded_state(self, module, state_latent):
        decoded_probs = self._decode_logits(module, state_latent).softmax(dim=-1)
        decoded_image = decoded_probs.view(-1, 1, self.grid_size, self.grid_size)
        return module.state_encoder(decoded_image)

    def _action_from_policy(self, module, state_latent, target_latent, *, hard: bool, gumbel_tau: float | None):
        torch = self.torch
        nn = self.nn
        action_pre = module.policy(state_latent, target_latent)
        distances = torch.cdist(action_pre[:, None, :], module.action_codebook.weight[None, :, :]).squeeze(1)
        code_logits = -distances

        if gumbel_tau is not None:
            code_probs = nn.functional.gumbel_softmax(code_logits, tau=gumbel_tau, hard=True)
        elif hard:
            hard_codes = code_logits.argmax(dim=-1)
            code_probs = nn.functional.one_hot(hard_codes, num_classes=self.codebook_size).float()
        else:
            code_probs = code_logits.softmax(dim=-1)

        action_quantized = code_probs @ module.action_codebook.weight
        nearest_codes = code_logits.argmax(dim=-1)
        nearest = module.action_codebook(nearest_codes)
        codebook_loss = (nearest - action_pre.detach()).pow(2).mean()
        commitment_loss = (action_pre - nearest.detach()).pow(2).mean()
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        return action_quantized, code_logits, vq_loss

    def _execute_step(self, module, state_latent, action_quantized):
        delta = module.transition(state_latent, action_quantized)
        return state_latent + delta

    def _grounding_loss(self, module, states):
        if not states:
            return None
        losses = [
            (state - self._encode_decoded_state(module, state).detach()).pow(2).mean()
            for state in states
        ]
        return self.torch.stack(losses).mean()

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

        support_state = module.state_encoder(self._state_image(support_x))
        query_state = module.state_encoder(self._state_image(query_x))
        target_state = module.state_encoder(self._state_image(support_y))
        code_logits_by_step = []
        support_logits_by_step = []
        query_logits_by_step = []
        vq_losses = []
        intermediate_states = []

        for _ in range(rollout_steps):
            action, code_logits, vq_loss = self._action_from_policy(
                module,
                support_state,
                target_state,
                hard=hard,
                gumbel_tau=gumbel_tau,
            )
            support_state = self._execute_step(module, support_state, action)
            query_state = self._execute_step(module, query_state, action)
            support_logits = self._decode_logits(module, support_state).log_softmax(dim=-1)
            query_logits = self._decode_logits(module, query_state).log_softmax(dim=-1)
            code_logits_by_step.append(code_logits)
            support_logits_by_step.append(support_logits)
            query_logits_by_step.append(query_logits)
            vq_losses.append(vq_loss)
            intermediate_states.append(support_state)

        support_losses = torch.stack(
            [nn.functional.nll_loss(logits, support_y, reduction="none") for logits in support_logits_by_step],
            dim=1,
        )
        length_penalty = self.mdl_weight ** torch.arange(
            1,
            rollout_steps + 1,
            device=support_x.device,
            dtype=support_losses.dtype,
        )
        chosen_lengths = (support_losses * length_penalty[None, :]).argmin(dim=1)
        row_index = torch.arange(batch_size, device=support_x.device)
        support_loss = support_losses[row_index, chosen_lengths].mean()

        chosen_support_logits = torch.stack(support_logits_by_step, dim=1)[row_index, chosen_lengths]
        chosen_query_logits = torch.stack(query_logits_by_step, dim=1)[row_index, chosen_lengths]
        if query_y is None:
            query_loss = support_loss.new_zeros(())
        else:
            query_loss = nn.functional.nll_loss(chosen_query_logits, query_y)

        vq_loss = torch.stack(vq_losses).mean()
        grounding_loss = self._grounding_loss(module, intermediate_states)
        if grounding_loss is None:
            grounding_loss = support_loss.new_zeros(())
        loss = (
            support_loss
            + query_loss
            + self.vq_loss_weight * vq_loss
            + self.grounding_loss_weight * grounding_loss
        )
        return ModelOutput(
            loss=loss,
            support_loss=support_loss,
            query_loss=query_loss,
            chosen_lengths=chosen_lengths + 1,
            code_logits=torch.stack(code_logits_by_step, dim=1),
            support_logits=chosen_support_logits,
            query_logits=chosen_query_logits,
            vq_loss=vq_loss,
            grounding_loss=grounding_loss,
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
        if sample_count < 1:
            raise ValueError("sample_count must be positive")

        torch = self.torch
        nn = self.nn
        batch_size = support_x.shape[0]
        candidate_count = sample_count + 1

        flat_support_x = support_x[:, None].expand(batch_size, candidate_count).reshape(-1)
        flat_query_x = query_x[:, None].expand(batch_size, candidate_count).reshape(-1)
        flat_support_y = support_y[:, None].expand(batch_size, candidate_count).reshape(-1)

        support_state = module.state_encoder(self._state_image(flat_support_x))
        query_state = module.state_encoder(self._state_image(flat_query_x))
        target_state = module.state_encoder(self._state_image(flat_support_y))
        code_logits_by_step = []
        support_logits_by_step = []
        query_logits_by_step = []

        for _ in range(rollout_steps):
            action_pre = module.policy(support_state, target_state)
            distances = torch.cdist(action_pre[:, None, :], module.action_codebook.weight[None, :, :]).squeeze(1)
            step_logits = -distances
            step_logits_view = step_logits.view(batch_size, candidate_count, self.codebook_size)
            greedy_codes = step_logits_view[:, :1].argmax(dim=-1)
            sampled_codes = torch.multinomial(
                step_logits_view[:, 1:].reshape(batch_size * sample_count, self.codebook_size).softmax(dim=-1),
                num_samples=1,
                replacement=True,
            ).view(batch_size, sample_count)
            step_codes = torch.cat([greedy_codes, sampled_codes], dim=1).reshape(-1)
            action = module.action_codebook(step_codes)
            support_state = self._execute_step(module, support_state, action)
            query_state = self._execute_step(module, query_state, action)
            support_logits_by_step.append(self._decode_logits(module, support_state).log_softmax(dim=-1))
            query_logits_by_step.append(self._decode_logits(module, query_state).log_softmax(dim=-1))
            code_logits_by_step.append(step_logits_view[:, 0])

        support_losses = torch.stack(
            [
                nn.functional.nll_loss(logits, flat_support_y, reduction="none")
                for logits in support_logits_by_step
            ],
            dim=1,
        )
        support_losses = support_losses.view(batch_size, candidate_count, rollout_steps)
        length_penalty = self.mdl_weight ** torch.arange(
            1,
            rollout_steps + 1,
            device=support_x.device,
            dtype=support_losses.dtype,
        )
        scores = support_losses * length_penalty[None, None, :]
        best_flat = scores.reshape(batch_size, candidate_count * rollout_steps).argmin(dim=1)
        chosen_candidate = best_flat // rollout_steps
        chosen_lengths = best_flat % rollout_steps
        row_index = torch.arange(batch_size, device=support_x.device)

        support_loss = support_losses[row_index, chosen_candidate, chosen_lengths].mean()
        support_logits_stacked = torch.stack(support_logits_by_step, dim=1).view(
            batch_size,
            candidate_count,
            rollout_steps,
            self.num_states,
        )
        query_logits_stacked = torch.stack(query_logits_by_step, dim=1).view(
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
            vq_loss=None,
            grounding_loss=None,
        )


class DiscMonoGridWorld(NEOGridWorld):
    """One-code monolithic baseline with the same executor interface."""

    def __init__(
        self,
        *,
        grid_size: int = 10,
        codebook_size: int = 36,
        hidden_dim: int = 32,
        action_dim: int = 16,
        ff_dim: int = 128,
        max_steps: int = 1,
        mdl_weight: float = 0.0,
        commitment_cost: float = 0.25,
        vq_loss_weight: float = 1.0,
        grounding_loss_weight: float = 0.0,
    ) -> None:
        super().__init__(
            grid_size=grid_size,
            codebook_size=codebook_size,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            ff_dim=ff_dim,
            max_steps=max_steps,
            mdl_weight=mdl_weight,
            commitment_cost=commitment_cost,
            vq_loss_weight=vq_loss_weight,
            grounding_loss_weight=grounding_loss_weight,
        )


def build_model(name: Literal["neo", "neo_s", "disc_mono"], **kwargs):
    if name in {"neo", "neo_s"}:
        return NEOGridWorld(**kwargs)
    if name == "disc_mono":
        kwargs.setdefault("max_steps", 1)
        kwargs.setdefault("codebook_size", 36)
        kwargs.setdefault("grounding_loss_weight", 0.0)
        return DiscMonoGridWorld(**kwargs)
    raise ValueError(f"unknown model: {name}")
