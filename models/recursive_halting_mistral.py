from dataclasses import dataclass
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
        # h: [batch, seq, hidden] → pool over seq
        pooled = h.mean(dim=1)
        return torch.sigmoid(self.proj(pooled)).squeeze(-1)  # [batch]


class RecursiveHaltingMistralForCausalLM(MistralForCausalLM):
    """
    Adaptive Computation Time (ACT) style halting around Mistral.

    - Unroll up to K_max inner steps per token.
    - At each inner step t, compute p_t via a stop head on hidden states.
    - Compute halting weights w_t and expected steps.
    - Loss = token CE + lambda_ponder * E[steps].
    - In training, we form a convex combination of logits over inner steps using w_t.
    """

    def __init__(self, config, k_max: int = 4, tau: float = 0.99, lambda_ponder: float = 0.001, halting_mass_scale: float = 1.0):
        super().__init__(config)
        assert k_max >= 1
        self.k_max = k_max
        self.tau = tau
        self.lambda_ponder = lambda_ponder
        self.halting_mass_scale = halting_mass_scale
        self.stop_head = StopHead(config.hidden_size)
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

        batch_size = h.size(0)
        halting_mass = torch.zeros(batch_size, device=h.device, dtype=h.dtype)
        weights = []
        logits_list = []
        p_list = []

        for t in range(self.k_max):
            p_t = self.stop_head(h)  # [b]
            p_list.append(p_t)

            new_halt = (halting_mass + p_t) > self.tau
            still_running = (halting_mass < self.tau).float()

            # Compute weight for this step
            w_t = torch.where(
                new_halt,
                1.0 - halting_mass,
                p_t,
            )
            w_t = w_t * still_running
            weights.append(w_t)
            halting_mass = halting_mass + w_t
            halting_mass = halting_mass * self.halting_mass_scale

            # Logits from this step
            logits_t = self.lm_head(h)
            logits_list.append(logits_t)

            # Early exit if all halted
            if torch.all(halting_mass >= self.tau):
                break

            # Next inner step
            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=h,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            h = outputs.last_hidden_state

        # Normalize weights so sum_t w_t ~ 1 for running samples; clamp for numerical stability
        W = torch.stack(weights, dim=0)  # [T, b]
        W = torch.clamp(W, min=0.0, max=1.0)
        W_sum = torch.clamp(W.sum(dim=0, keepdim=True), min=1e-6)
        W_norm = W / W_sum

        # Combine logits: convex combination over steps
        # logits_list: list[T] of [b, s, vocab]
        logits = sum(w.view(-1, 1, 1) * L for w, L in zip(W_norm, logits_list))

        # Expected steps (per batch) for ponder loss
        steps = torch.arange(1, W.size(0) + 1, device=h.device, dtype=h.dtype).view(-1, 1)
        expected_steps = (W_norm * steps).sum(dim=0)  # [b]

        # Expose last-step telemetry
        try:
            self._last_inner_steps = int(W.size(0))
            self._last_expected_steps = expected_steps.detach()
            self._last_expected_steps_mean = float(expected_steps.mean().item())
        except Exception:
            pass

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce = CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            ponder = expected_steps.mean()
            loss = ce + self.lambda_ponder * ponder

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
