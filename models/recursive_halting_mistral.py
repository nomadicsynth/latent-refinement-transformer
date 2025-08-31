from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.mistral.modeling_mistral import MistralForCausalLM


@dataclass
class HaltingOutputs:
    halting_prob: torch.Tensor  # [batch, steps]
    expected_steps: torch.Tensor  # [batch]
    avg_steps: float


class StopHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [batch, seq, hidden] -> per-token stop probability
        # return shape: [batch, seq]
        return torch.sigmoid(self.proj(h)).squeeze(-1)


class StepFiLM(nn.Module):
    """
    Lightweight per-step FiLM: h -> h * (1 + gamma_t) + beta_t,
    where gamma_t, beta_t come from a learned embedding of step index t.
    """

    def __init__(self, hidden_size: int, k_max: int, rank: int = 128):
        super().__init__()
        self.step_emb = nn.Embedding(k_max, rank)
        self.to_gamma = nn.Linear(rank, hidden_size, bias=True)
        self.to_beta = nn.Linear(rank, hidden_size, bias=True)

    def forward(self, h: torch.Tensor, step_idx: torch.LongTensor) -> torch.Tensor:
        # h: [b, s, d], step_idx: [b]
        e = self.step_emb(step_idx)  # [b, r]
        gamma = self.to_gamma(e).unsqueeze(1)  # [b, 1, d]
        beta = self.to_beta(e).unsqueeze(1)  # [b, 1, d]
        return h * (1.0 + torch.tanh(gamma)) + beta


class RecursiveHaltingMistralForCausalLM(MistralForCausalLM):
    """
    Adaptive Computation Time (ACT) style halting around Mistral.

    - Unroll up to K_max inner steps per token.
    - At each inner step t, compute p_t via a stop head on hidden states.
    - Compute halting weights w_t and expected steps.
    - Loss = token CE + lambda_ponder * E[steps].
    - In training, we form a convex combination of logits over inner steps using w_t.
    """

    def __init__(self, config, k_max: int = 4, tau: float = 0.99, lambda_ponder: float = 0.001,
                 halting_mass_scale: float = 1.0, use_step_film: bool = True, film_rank: int = 128,
                 lambda_deep_supervision: float = 0.0):
        super().__init__(config)
        assert k_max >= 1
        self.k_max = k_max
        self.tau = tau
        self.lambda_ponder = lambda_ponder
        self.halting_mass_scale = halting_mass_scale
        self.stop_head = StopHead(config.hidden_size)
        # Initialize stop bias so initial p_t ~ 0.55 (gentle early-halt prior)
        with torch.no_grad():
            b = math.log(0.55 / 0.45)
            if hasattr(self.stop_head, "proj") and hasattr(self.stop_head.proj, "bias"):
                self.stop_head.proj.bias.fill_(b)
        self.use_step_film = use_step_film
        self.lambda_deep_supervision = lambda_deep_supervision

        # Tiny step-conditioned FiLM so later passes have distinct compute
        if self.use_step_film:
            self.step_film = StepFiLM(hidden_size=config.hidden_size, k_max=k_max, rank=film_rank)
        else:
            self.step_film = None

        # Residual across inner steps
        self.use_residual_across_steps = True
        # Start small (sigmoid(-2) ≈ 0.12) to bias toward gentle updates
        self.step_gates = nn.Parameter(torch.full((k_max,), -2.0))

        # Exposed telemetry for callbacks/logging
        self._last_inner_steps = 1
        self._last_expected_steps_mean = float(self.k_max)
        self._last_expected_steps = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # First inner step
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        h = outputs.last_hidden_state  # [b, s, d]

        batch_size, seq_len = h.size(0), h.size(1)
        # Valid token mask [b, s] (1 for valid tokens, 0 for padding). If None, all ones.
        if attention_mask is None:
            valid_mask = torch.ones((batch_size, seq_len), device=h.device, dtype=torch.bool)
        else:
            # ensure boolean mask
            valid_mask = attention_mask.to(dtype=torch.bool)

        # Halting mass tracked per token [b, s]
        halting_mass = torch.zeros((batch_size, seq_len), device=h.device, dtype=h.dtype)
        weights = []
        logits_list = []
        p_list = []
        ce_per_step = []  # deep supervision

        for t in range(self.k_max):
            p_t = self.stop_head(h)  # [b, s]
            p_list.append(p_t)

            # Compute masks for tokens still running and those that will newly halt
            new_halt = ((halting_mass + p_t) > self.tau) & valid_mask
            still_running = ((halting_mass < self.tau) & valid_mask).to(dtype=h.dtype)

            # Compute weight for this step
            w_t = torch.where(new_halt, 1.0 - halting_mass, p_t)
            w_t = w_t * still_running
            weights.append(w_t)
            halting_mass = (halting_mass + w_t) * self.halting_mass_scale

            # Logits from this step
            logits_t = self.lm_head(h)
            logits_list.append(logits_t)

            # Optional: per-step CE to encourage refinement
            if labels is not None and self.training and self.k_max > 1 and self.lambda_deep_supervision > 0.0:
                shift_logits_t = logits_t[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                ce_t = CrossEntropyLoss(ignore_index=-100)(
                    shift_logits_t.view(-1, shift_logits_t.size(-1)), shift_labels.view(-1)
                )
                ce_per_step.append(ce_t)

            # Early exit if all halted
            if torch.all((halting_mass >= self.tau) | (~valid_mask)):
                break

            # Next inner step input: step-conditioned FiLM to change the function
            if self.use_step_film and self.step_film is not None:
                # use current step index t to prepare the next pass (t in [0..])
                step_idx = torch.full((batch_size,), t, device=h.device, dtype=torch.long)
                h_in = self.step_film(h, step_idx)  # [b, s, d]
            else:
                h_in = h

            # Next inner step compute
            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=h_in,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            h_next = outputs.last_hidden_state  # proposed refinement

            # Gated residual refinement (EMA toward h_next), masked to still-running tokens
            if self.use_residual_across_steps:
                eta = torch.sigmoid(self.step_gates[min(t, self.k_max - 1)])  # scalar
                # h <- h + eta * (h_next - h) on running tokens; keep halted tokens unchanged
                update = eta * (h_next - h)
                h = h + update * still_running.unsqueeze(-1)
            else:
                h = h_next

        # Normalize weights so sum_t w_t ~ 1 for running samples; clamp for numerical stability
        W = torch.stack(weights, dim=0)  # [T, b, s]
        W = torch.clamp(W, min=0.0, max=1.0)
        # Zero-out any weights on invalid tokens explicitly for safety
        W = W * valid_mask.to(dtype=W.dtype).unsqueeze(0)
        W_sum = torch.clamp(W.sum(dim=0, keepdim=True), min=1e-6)
        W_norm = W / W_sum

        # Combine logits: convex combination over steps
        logits = sum(w.unsqueeze(-1) * L for w, L in zip(W_norm, logits_list))  # [b, s, vocab]

        # Expected steps (per batch) for ponder loss
        steps = torch.arange(1, W.size(0) + 1, device=h.device, dtype=h.dtype).view(-1, 1, 1)  # [T,1,1]
        expected_steps_tokens = (W_norm * steps).sum(dim=0)  # [b, s]
        # Compute masked mean per batch element
        valid_counts = valid_mask.to(dtype=h.dtype).sum(dim=1).clamp(min=1.0)  # [b]
        expected_steps = (expected_steps_tokens * valid_mask.to(dtype=h.dtype)).sum(dim=1) / valid_counts  # [b]

        # Expose last-step telemetry
        try:
            self._last_inner_steps = int(W.size(0))
            self._last_expected_steps = expected_steps.detach()
            self._last_expected_steps_mean = float(expected_steps.mean().item())
        except Exception:
            pass

        loss = None
        if labels is not None:
            # Final mixed CE (primary)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce = CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            ponder = expected_steps.mean()
            loss = ce + (self.lambda_ponder * ponder if self.training else 0.0)

            # Deep supervision (later steps heavier)
            if self.training and self.lambda_deep_supervision > 0.0 and len(ce_per_step) > 0:
                # Use average halting weights across batch/seq to weight per-step CE
                with torch.no_grad():
                    w_mean = W_norm.mean(dim=(1, 2))  # [T]
                    alphas = w_mean / (w_mean.sum() + 1e-8)
                aux = sum(a * c for a, c in zip(alphas, ce_per_step))
                loss = loss + self.lambda_deep_supervision * aux

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )

    # Make Trainer's FLOPs estimation reflect inner-step recursion
    def floating_point_ops(self, inputs: Optional[dict] = None, exclude_embeddings: bool = True) -> int:
        base = 0
        # Try parent implementation if available
        try:
            base = super().floating_point_ops(inputs, exclude_embeddings=exclude_embeddings)  # type: ignore[arg-type]
        except Exception:
            base = 0
        if not isinstance(base, (int, float)) or base == 0:
            # Fallback: parameter-count based rough proxy
            try:
                base = sum(p.numel() for p in self.parameters())
            except Exception:
                base = 0
        steps = getattr(self, "_last_inner_steps", self.k_max)
        try:
            return int(base * max(1, int(steps)))
        except Exception:
            return int(base)
