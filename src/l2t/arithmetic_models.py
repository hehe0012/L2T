"""NEO-style models for Arithmetic Factorization Reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ArithmeticModelOutput:
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


class NEOArithmetic:
    """NEO/NEO-S model for prime-exponent arithmetic observations."""

    def __init__(
        self,
        *,
        state_dim: int = 8,
        max_exponent: int = 12,
        codebook_size: int = 16,
        action_dim: int = 16,
        policy_d_model: int = 64,
        policy_ff_dim: int = 256,
        policy_heads: int = 4,
        policy_layers: int = 6,
        transition_d_model: int = 32,
        transition_ff_dim: int = 128,
        transition_heads: int = 2,
        transition_layers: int = 4,
        max_steps: int = 6,
        mdl_weight: float = 0.99,
        query_loss_weight: float = 0.0,
        neo_s_selection: Literal["min_loss", "majority_exact"] = "min_loss",
        commitment_cost: float = 0.25,
        vq_loss_weight: float = 1.0,
        grounding_loss_weight: float = 0.5,
    ) -> None:
        import torch
        from torch import nn

        self.torch = torch
        self.nn = nn
        self.num_factors = 4
        self.state_dim = state_dim
        self.max_exponent = max_exponent
        self.codebook_size = codebook_size
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.mdl_weight = mdl_weight
        self.query_loss_weight = query_loss_weight
        self.neo_s_selection = neo_s_selection
        self.commitment_cost = commitment_cost
        self.vq_loss_weight = vq_loss_weight
        self.grounding_loss_weight = grounding_loss_weight

        class CrossAttentionBlock(nn.Module):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.attn = nn.MultiheadAttention(
                    transition_d_model,
                    transition_heads,
                    batch_first=True,
                )
                inner_self.attn_norm = nn.LayerNorm(transition_d_model)
                inner_self.ff = nn.Sequential(
                    nn.Linear(transition_d_model, transition_ff_dim),
                    nn.ReLU(),
                    nn.Linear(transition_ff_dim, transition_d_model),
                )
                inner_self.ff_norm = nn.LayerNorm(transition_d_model)

            def forward(inner_self, state_token, action_token):
                attended, _ = inner_self.attn(state_token, action_token, action_token, need_weights=False)
                state_token = inner_self.attn_norm(state_token + attended)
                return inner_self.ff_norm(state_token + inner_self.ff(state_token))

        class _Module(nn.Module):
            def __init__(inner_self, outer: "NEOArithmetic") -> None:
                super().__init__()
                inner_self.outer = outer
                inner_self.state_encoder = nn.Sequential(
                    nn.Linear(outer.num_factors, policy_d_model),
                    nn.ReLU(),
                    nn.Linear(policy_d_model, state_dim),
                )
                inner_self.state_decoder = nn.Sequential(
                    nn.Linear(state_dim, policy_d_model),
                    nn.ReLU(),
                    nn.Linear(policy_d_model, outer.num_factors * (max_exponent + 1)),
                )
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=policy_d_model,
                    nhead=policy_heads,
                    dim_feedforward=policy_ff_dim,
                    batch_first=True,
                    activation="relu",
                )
                inner_self.policy_input = nn.Linear(state_dim, policy_d_model)
                inner_self.policy_type = nn.Parameter(torch.zeros(2, policy_d_model))
                inner_self.policy = nn.TransformerEncoder(encoder_layer, num_layers=policy_layers)
                inner_self.policy_out = nn.Linear(policy_d_model, action_dim)
                inner_self.action_codebook = nn.Embedding(codebook_size, action_dim)
                inner_self.transition_state = nn.Linear(state_dim, transition_d_model)
                inner_self.transition_action = nn.Linear(action_dim, transition_d_model)
                inner_self.transition_blocks = nn.ModuleList(
                    [CrossAttentionBlock() for _ in range(transition_layers)]
                )
                inner_self.transition_out = nn.Linear(transition_d_model, state_dim)

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

    def _normalize_exponents(self, exponents):
        return exponents.float().clamp(0, self.max_exponent) / float(self.max_exponent)

    def _decode_logits(self, module, state_latent):
        logits = module.state_decoder(state_latent)
        return logits.view(state_latent.shape[0], self.num_factors, self.max_exponent + 1)

    def _decode_expected_exponents(self, logits):
        indices = self.torch.arange(
            self.max_exponent + 1,
            device=logits.device,
            dtype=logits.dtype,
        )
        return (logits.softmax(dim=-1) * indices).sum(dim=-1)

    def _state_loss(self, logits, targets, *, reduction: str = "mean"):
        nn = self.nn
        flat_logits = logits.reshape(-1, self.max_exponent + 1).log_softmax(dim=-1)
        flat_targets = targets.clamp(0, self.max_exponent).reshape(-1)
        loss = nn.functional.nll_loss(flat_logits, flat_targets, reduction="none")
        loss = loss.view(targets.shape[0], self.num_factors).sum(dim=1)
        if reduction == "none":
            return loss
        if reduction == "mean":
            return loss.mean()
        raise ValueError(f"unknown reduction: {reduction}")

    def _encode(self, module, exponents):
        return module.state_encoder(self._normalize_exponents(exponents))

    def _encode_decoded_state(self, module, logits):
        expected = self._decode_expected_exponents(logits)
        return module.state_encoder(expected / float(self.max_exponent))

    def _action_from_policy(self, module, state_latent, target_latent, *, hard: bool, gumbel_tau: float | None):
        torch = self.torch
        nn = self.nn
        tokens = torch.stack(
            [module.policy_input(state_latent), module.policy_input(target_latent)],
            dim=1,
        )
        tokens = tokens + module.policy_type[None, :, :]
        encoded = module.policy(tokens)
        action_pre = module.policy_out(encoded[:, 0])
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
        state_token = module.transition_state(state_latent)[:, None, :]
        action_token = module.transition_action(action_quantized)[:, None, :]
        for block in module.transition_blocks:
            state_token = block(state_token, action_token)
        return state_latent + module.transition_out(state_token[:, 0])

    def _grounding_loss(self, module, logits_by_step, states_by_step):
        if not states_by_step:
            return None
        losses = [
            (state - self._encode_decoded_state(module, logits).detach()).pow(2).mean()
            for logits, state in zip(logits_by_step, states_by_step)
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

        batch_size = support_x.shape[0]
        support_state = self._encode(module, support_x)
        query_state = self._encode(module, query_x)
        target_state = self._encode(module, support_y)
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
            support_logits = self._decode_logits(module, support_state)
            query_logits = self._decode_logits(module, query_state)
            code_logits_by_step.append(code_logits)
            support_logits_by_step.append(support_logits)
            query_logits_by_step.append(query_logits)
            vq_losses.append(vq_loss)
            intermediate_states.append(support_state)

        support_losses = torch.stack(
            [self._state_loss(logits, support_y, reduction="none") for logits in support_logits_by_step],
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
        query_loss = support_loss.new_zeros(()) if query_y is None else self._state_loss(chosen_query_logits, query_y)

        vq_loss = torch.stack(vq_losses).mean()
        grounding_loss = self._grounding_loss(module, support_logits_by_step, intermediate_states)
        if grounding_loss is None:
            grounding_loss = support_loss.new_zeros(())
        loss = (
            support_loss
            + self.query_loss_weight * query_loss
            + self.vq_loss_weight * vq_loss
            + self.grounding_loss_weight * grounding_loss
        )
        return ArithmeticModelOutput(
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

    def _sampled_forward(self, module, support_x, support_y, query_x, query_y, *, sample_count: int, rollout_steps: int):
        if sample_count < 1:
            raise ValueError("sample_count must be positive")

        torch = self.torch
        batch_size = support_x.shape[0]
        candidate_count = sample_count + 1
        flat_support_x = support_x[:, None, :].expand(batch_size, candidate_count, self.num_factors).reshape(-1, self.num_factors)
        flat_query_x = query_x[:, None, :].expand(batch_size, candidate_count, self.num_factors).reshape(-1, self.num_factors)
        flat_support_y = support_y[:, None, :].expand(batch_size, candidate_count, self.num_factors).reshape(-1, self.num_factors)

        support_state = self._encode(module, flat_support_x)
        query_state = self._encode(module, flat_query_x)
        target_state = self._encode(module, flat_support_y)
        code_logits_by_step = []
        support_logits_by_step = []
        query_logits_by_step = []
        step_codes_by_step = []

        for _ in range(rollout_steps):
            action_pre = module.policy_out(
                module.policy(
                    torch.stack(
                        [module.policy_input(support_state), module.policy_input(target_state)],
                        dim=1,
                    )
                    + module.policy_type[None, :, :]
                )[:, 0]
            )
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
            support_logits_by_step.append(self._decode_logits(module, support_state))
            query_logits_by_step.append(self._decode_logits(module, query_state))
            code_logits_by_step.append(step_logits_view[:, 0])
            step_codes_by_step.append(step_codes.view(batch_size, candidate_count))

        support_losses = torch.stack(
            [self._state_loss(logits, flat_support_y, reduction="none") for logits in support_logits_by_step],
            dim=1,
        ).view(batch_size, candidate_count, rollout_steps)
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
        exact_candidate_rate = None
        fallback_rate = None
        majority_count = None

        if self.neo_s_selection == "majority_exact":
            support_logits_stacked_for_select = torch.stack(support_logits_by_step, dim=1).view(
                batch_size,
                candidate_count,
                rollout_steps,
                self.num_factors,
                self.max_exponent + 1,
            )
            support_predictions = support_logits_stacked_for_select.argmax(dim=-1)
            exact_mask = (support_predictions == support_y[:, None, None, :]).all(dim=-1)
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
            self.num_factors,
            self.max_exponent + 1,
        )
        query_logits_stacked = torch.stack(query_logits_by_step, dim=1).view(
            batch_size,
            candidate_count,
            rollout_steps,
            self.num_factors,
            self.max_exponent + 1,
        )
        chosen_support_logits = support_logits_stacked[row_index, chosen_candidate, chosen_lengths]
        chosen_query_logits = query_logits_stacked[row_index, chosen_candidate, chosen_lengths]
        query_loss = support_loss.new_zeros(()) if query_y is None else self._state_loss(chosen_query_logits, query_y)

        return ArithmeticModelOutput(
            loss=support_loss + self.query_loss_weight * query_loss,
            support_loss=support_loss,
            query_loss=query_loss,
            chosen_lengths=chosen_lengths + 1,
            code_logits=torch.stack(code_logits_by_step, dim=1),
            support_logits=chosen_support_logits,
            query_logits=chosen_query_logits,
            vq_loss=None,
            grounding_loss=None,
            exact_candidate_rate=exact_candidate_rate,
            fallback_rate=fallback_rate,
            majority_count=majority_count,
        )


class DiscMonoArithmetic(NEOArithmetic):
    """Single-code Arithmetic baseline matching the Disc-Mono interface."""

    def __init__(
        self,
        *,
        codebook_size: int = 40,
        max_steps: int = 1,
        grounding_loss_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            codebook_size=codebook_size,
            max_steps=max_steps,
            grounding_loss_weight=grounding_loss_weight,
            **kwargs,
        )


def build_arithmetic_model(name: Literal["neo", "neo_s", "disc_mono"], **kwargs):
    if name in {"neo", "neo_s"}:
        return NEOArithmetic(**kwargs)
    if name == "disc_mono":
        kwargs.setdefault("codebook_size", 40)
        kwargs.setdefault("max_steps", 1)
        kwargs.setdefault("grounding_loss_weight", 0.0)
        return DiscMonoArithmetic(**kwargs)
    raise ValueError(f"unknown model: {name}")
