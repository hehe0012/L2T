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
    exact_candidate_rate: object | None = None
    fallback_rate: object | None = None
    majority_count: object | None = None


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
        mdl_score_mode: Literal["multiplicative", "additive"] = "multiplicative",
        mdl_lambda: float = 0.0,
        state_path: Literal["cnn", "mlp"] = "cnn",
        state_dropout: float = 0.1,
        action_mode: Literal["discrete", "continuous"] = "discrete",
        query_loss_weight: float = 0.0,
        neo_s_selection: Literal["min_loss", "majority_exact"] = "min_loss",
        commitment_cost: float = 0.25,
        vq_loss_weight: float = 1.0,
        grounding_loss_weight: float = 0.1,
        grounding_transition_only: bool = False,
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
        self.mdl_score_mode = mdl_score_mode
        self.mdl_lambda = mdl_lambda
        self.state_path = state_path
        self.state_dropout = state_dropout
        self.action_mode = action_mode
        self.query_loss_weight = query_loss_weight
        self.neo_s_selection = neo_s_selection
        self.commitment_cost = commitment_cost
        self.vq_loss_weight = vq_loss_weight
        self.grounding_loss_weight = grounding_loss_weight
        self.grounding_transition_only = grounding_transition_only

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
                if state_path == "cnn":
                    encoder_layers: list[nn.Module] = [
                        nn.Conv2d(1, 16, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(16, 32, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(32 * grid_size * grid_size, ff_dim),
                        nn.ReLU(),
                    ]
                    if state_dropout > 0:
                        encoder_layers.append(nn.Dropout(state_dropout))
                    encoder_layers.append(nn.Linear(ff_dim, hidden_dim))
                    inner_self.state_encoder = nn.Sequential(*encoder_layers)
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
                elif state_path == "mlp":
                    encoder_layers = [
                        nn.Linear(outer.num_states, ff_dim),
                        nn.ReLU(),
                        nn.Linear(ff_dim, ff_dim),
                        nn.ReLU(),
                    ]
                    if state_dropout > 0:
                        encoder_layers.append(nn.Dropout(state_dropout))
                    encoder_layers.append(nn.Linear(ff_dim, hidden_dim))
                    inner_self.state_encoder = nn.Sequential(*encoder_layers)
                    inner_self.state_decoder = nn.Sequential(
                        nn.Linear(hidden_dim, ff_dim),
                        nn.ReLU(),
                        nn.Linear(ff_dim, ff_dim),
                        nn.ReLU(),
                        nn.Linear(ff_dim, outer.num_states),
                    )
                else:
                    raise ValueError(f"unknown state_path: {state_path}")
                policy_output_dim = action_dim * 2 if action_mode == "continuous" else action_dim
                inner_self.policy = FiLMMLP(hidden_dim, hidden_dim, ff_dim, policy_output_dim)
                if action_mode == "discrete":
                    inner_self.action_codebook = nn.Embedding(codebook_size, action_dim)
                elif action_mode != "continuous":
                    raise ValueError(f"unknown action_mode: {action_mode}")
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

    def _state_features(self, indices):
        nn = self.nn
        return nn.functional.one_hot(indices, num_classes=self.num_states).float()

    def _state_image(self, indices):
        return self._state_features(indices).view(indices.shape[0], 1, self.grid_size, self.grid_size)

    def _encode_indices(self, module, indices):
        if self.state_path == "cnn":
            return module.state_encoder(self._state_image(indices))
        return module.state_encoder(self._state_features(indices))

    def _decode_logits(self, module, state_latent):
        return module.state_decoder(state_latent).flatten(1)

    def _encode_decoded_state(self, module, state_latent):
        decoded_probs = self._decode_logits(module, state_latent).softmax(dim=-1)
        if self.state_path == "cnn":
            return module.state_encoder(decoded_probs.view(-1, 1, self.grid_size, self.grid_size))
        return module.state_encoder(decoded_probs)

    def _length_scores(self, support_losses, rollout_steps: int):
        torch = self.torch
        lengths = torch.arange(
            1,
            rollout_steps + 1,
            device=support_losses.device,
            dtype=support_losses.dtype,
        )
        if self.mdl_score_mode == "multiplicative":
            length_penalty = self.mdl_weight ** lengths
            return support_losses * length_penalty.reshape((1,) * (support_losses.dim() - 1) + (-1,))
        if self.mdl_score_mode == "additive":
            return support_losses + self.mdl_lambda * lengths.reshape((1,) * (support_losses.dim() - 1) + (-1,))
        raise ValueError(f"unknown mdl_score_mode: {self.mdl_score_mode}")

    def _action_from_policy(self, module, state_latent, target_latent, *, hard: bool, gumbel_tau: float | None):
        torch = self.torch
        nn = self.nn
        action_pre = module.policy(state_latent, target_latent)
        if self.action_mode == "continuous":
            mean, logvar = action_pre.chunk(2, dim=-1)
            logvar = logvar.clamp(min=-20.0, max=20.0)
            if hard:
                action = mean
            else:
                action = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
            kl_loss = -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1).mean()
            return action, mean, kl_loss

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
        # The paper sums grounding over every intermediate step k. Each term
        # is already averaged over batch and latent dimensions, so do not
        # average over the rollout dimension here.
        return self.torch.stack(losses).sum()

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

        support_state = self._encode_indices(module, support_x)
        query_state = self._encode_indices(module, query_x)
        target_state = self._encode_indices(module, support_y)
        code_logits_by_step = []
        support_logits_by_step = []
        query_logits_by_step = []
        vq_losses = []
        intermediate_states = []
        grounding_states = []
        grounding_state = support_state.detach()

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
            if self.grounding_transition_only:
                # Keep a separate state chain so grounding updates transition
                # but cannot flow through policy or the action codebook.
                grounding_state = self._execute_step(module, grounding_state, action.detach())
                grounding_states.append(grounding_state)
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
        chosen_lengths = self._length_scores(support_losses, rollout_steps).argmin(dim=1)
        row_index = torch.arange(batch_size, device=support_x.device)
        support_loss = support_losses[row_index, chosen_lengths].mean()

        chosen_support_logits = torch.stack(support_logits_by_step, dim=1)[row_index, chosen_lengths]
        chosen_query_logits = torch.stack(query_logits_by_step, dim=1)[row_index, chosen_lengths]
        if query_y is None:
            query_loss = support_loss.new_zeros(())
        else:
            query_loss = nn.functional.nll_loss(chosen_query_logits, query_y)

        vq_loss = torch.stack(vq_losses).mean()
        grounding_loss = self._grounding_loss(
            module,
            grounding_states if self.grounding_transition_only else intermediate_states,
        )
        if grounding_loss is None:
            grounding_loss = support_loss.new_zeros(())
        loss = (
            support_loss
            + self.query_loss_weight * query_loss
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
            exact_candidate_rate=None,
            fallback_rate=None,
            majority_count=None,
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

        support_state = self._encode_indices(module, flat_support_x)
        query_state = self._encode_indices(module, flat_query_x)
        target_state = self._encode_indices(module, flat_support_y)
        code_logits_by_step = []
        support_logits_by_step = []
        query_logits_by_step = []
        step_codes_by_step = []
        grounding_states = []
        grounding_state = support_state.detach()

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
            if self.grounding_transition_only:
                grounding_state = self._execute_step(module, grounding_state, action.detach())
                grounding_states.append(grounding_state)
            support_logits_by_step.append(self._decode_logits(module, support_state).log_softmax(dim=-1))
            query_logits_by_step.append(self._decode_logits(module, query_state).log_softmax(dim=-1))
            code_logits_by_step.append(step_logits_view[:, 0])
            step_codes_by_step.append(step_codes.view(batch_size, candidate_count))

        support_losses = torch.stack(
            [
                nn.functional.nll_loss(logits, flat_support_y, reduction="none")
                for logits in support_logits_by_step
            ],
            dim=1,
        )
        support_losses = support_losses.view(batch_size, candidate_count, rollout_steps)
        scores = self._length_scores(support_losses, rollout_steps)
        best_flat = scores.reshape(batch_size, candidate_count * rollout_steps).argmin(dim=1)
        chosen_candidate = best_flat // rollout_steps
        chosen_lengths = best_flat % rollout_steps
        exact_candidate_rate = None
        fallback_rate = None
        majority_count = None

        if self.neo_s_selection == "majority_exact":
            support_logits_stacked_for_select = torch.stack(support_logits_by_step, dim=1).view(
                batch_size,
                candidate_count,
                rollout_steps,
                self.num_states,
            )
            support_predictions = support_logits_stacked_for_select.argmax(dim=-1)
            exact_mask = support_predictions == support_y[:, None, None]
            step_codes_stacked = torch.stack(step_codes_by_step, dim=2)
            selected_candidates = []
            selected_lengths = []
            majority_counts = []
            fallback_flags = []
            for batch_index in range(batch_size):
                exact_positions = exact_mask[batch_index].nonzero(as_tuple=False)
                if exact_positions.numel() == 0:
                    selected_candidates.append(chosen_candidate[batch_index])
                    selected_lengths.append(chosen_lengths[batch_index])
                    majority_counts.append(torch.zeros((), device=support_x.device, dtype=support_losses.dtype))
                    fallback_flags.append(torch.ones((), device=support_x.device, dtype=support_losses.dtype))
                    continue

                counts: dict[tuple[int, ...], int] = {}
                first_position: dict[tuple[int, ...], tuple[int, int]] = {}
                for position in exact_positions.tolist():
                    candidate_index, length_index = position
                    program = tuple(
                        int(code)
                        for code in step_codes_stacked[batch_index, candidate_index, : length_index + 1]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    counts[program] = counts.get(program, 0) + 1
                    first_position.setdefault(program, (candidate_index, length_index))

                best_program = max(counts, key=lambda program: (counts[program], -len(program)))
                candidate_index, length_index = first_position[best_program]
                selected_candidates.append(torch.tensor(candidate_index, device=support_x.device, dtype=torch.long))
                selected_lengths.append(torch.tensor(length_index, device=support_x.device, dtype=torch.long))
                majority_counts.append(torch.tensor(counts[best_program], device=support_x.device, dtype=support_losses.dtype))
                fallback_flags.append(torch.zeros((), device=support_x.device, dtype=support_losses.dtype))

            chosen_candidate = torch.stack(selected_candidates)
            chosen_lengths = torch.stack(selected_lengths)
            exact_candidate_rate = exact_mask.float().mean()
            fallback_rate = torch.stack(fallback_flags).mean()
            majority_count = torch.stack(majority_counts).mean()
        elif self.neo_s_selection != "min_loss":
            raise ValueError(f"unknown neo_s_selection: {self.neo_s_selection}")

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

        grounding_loss = self._grounding_loss(module, grounding_states)
        if grounding_loss is None:
            grounding_loss = support_loss.new_zeros(())

        return ModelOutput(
            loss=(
                support_loss
                + self.query_loss_weight * query_loss
                + self.grounding_loss_weight * grounding_loss
            ),
            support_loss=support_loss,
            query_loss=query_loss,
            chosen_lengths=chosen_lengths + 1,
            code_logits=torch.stack(code_logits_by_step, dim=1),
            support_logits=chosen_support_logits,
            query_logits=chosen_query_logits,
            vq_loss=None,
            grounding_loss=grounding_loss,
            exact_candidate_rate=exact_candidate_rate,
            fallback_rate=fallback_rate,
            majority_count=majority_count,
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
        mdl_score_mode: Literal["multiplicative", "additive"] = "multiplicative",
        mdl_lambda: float = 0.0,
        state_path: Literal["cnn", "mlp"] = "cnn",
        state_dropout: float = 0.1,
        query_loss_weight: float = 0.0,
        neo_s_selection: Literal["min_loss", "majority_exact"] = "min_loss",
        commitment_cost: float = 0.25,
        vq_loss_weight: float = 1.0,
        grounding_loss_weight: float = 0.0,
        grounding_transition_only: bool = False,
    ) -> None:
        super().__init__(
            grid_size=grid_size,
            codebook_size=codebook_size,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            ff_dim=ff_dim,
            max_steps=max_steps,
            mdl_weight=mdl_weight,
            mdl_score_mode=mdl_score_mode,
            mdl_lambda=mdl_lambda,
            state_path=state_path,
            state_dropout=state_dropout,
            query_loss_weight=query_loss_weight,
            neo_s_selection=neo_s_selection,
            commitment_cost=commitment_cost,
            vq_loss_weight=vq_loss_weight,
            grounding_loss_weight=grounding_loss_weight,
            grounding_transition_only=grounding_transition_only,
        )


class ContMonoGridWorld(NEOGridWorld):
    """Continuous monolithic baseline with one inferred transition action."""

    def __init__(
        self,
        *,
        grid_size: int = 10,
        hidden_dim: int = 32,
        action_dim: int = 16,
        ff_dim: int = 128,
        max_steps: int = 1,
        mdl_weight: float = 0.0,
        mdl_score_mode: Literal["multiplicative", "additive"] = "multiplicative",
        mdl_lambda: float = 0.0,
        state_path: Literal["cnn", "mlp"] = "cnn",
        state_dropout: float = 0.1,
        query_loss_weight: float = 0.0,
        neo_s_selection: Literal["min_loss", "majority_exact"] = "min_loss",
        action_kl_weight: float = 0.01,
        grounding_loss_weight: float = 0.0,
        grounding_transition_only: bool = False,
    ) -> None:
        super().__init__(
            grid_size=grid_size,
            codebook_size=1,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            ff_dim=ff_dim,
            max_steps=max_steps,
            mdl_weight=mdl_weight,
            mdl_score_mode=mdl_score_mode,
            mdl_lambda=mdl_lambda,
            state_path=state_path,
            state_dropout=state_dropout,
            action_mode="continuous",
            query_loss_weight=query_loss_weight,
            neo_s_selection=neo_s_selection,
            commitment_cost=0.0,
            vq_loss_weight=action_kl_weight,
            grounding_loss_weight=grounding_loss_weight,
            grounding_transition_only=grounding_transition_only,
        )


def build_model(name: Literal["neo", "neo_s", "disc_mono", "cont_mono"], **kwargs):
    if name in {"neo", "neo_s"}:
        return NEOGridWorld(**kwargs)
    if name == "disc_mono":
        kwargs.setdefault("max_steps", 1)
        kwargs.setdefault("codebook_size", 36)
        kwargs.setdefault("grounding_loss_weight", 0.0)
        return DiscMonoGridWorld(**kwargs)
    if name == "cont_mono":
        kwargs.setdefault("max_steps", 1)
        kwargs.setdefault("grounding_loss_weight", 0.0)
        return ContMonoGridWorld(**kwargs)
    raise ValueError(f"unknown model: {name}")
