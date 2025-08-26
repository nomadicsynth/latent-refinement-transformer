import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.mistral.modeling_mistral import MistralForCausalLM


class RecursiveMistralForCausalLM(MistralForCausalLM):
    """
    A minimal fixed-K recursive wrapper around MistralForCausalLM.

    It runs the decoder stack multiple times per forward call by feeding the
    last hidden states back as inputs_embeds for additional inner iterations.

    Notes:
    - For simplicity, we disable/use no KV caching across inner iterations.
    - Rotary/positional encoding is re-applied each inner step (not ideal),
      but sufficient as a proof-of-concept. A refined version would bypass
      re-applying RoPE on inner steps.
    """

    def __init__(self, config, inner_steps: int = 2):
        super().__init__(config)
        assert inner_steps >= 1, "inner_steps must be >= 1"
        self.inner_steps = inner_steps

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

        # Caching across inner steps isn’t supported in this minimal PoC.
        if use_cache:
            warnings.warn("RecursiveMistralForCausalLM: use_cache disabled for recursive inner steps.")
        use_cache = False

        # First pass: standard model forward (input_ids or inputs_embeds)
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

        hidden_states = outputs.last_hidden_state

        # Additional inner iterations: feed back as inputs_embeds
        for _ in range(max(0, self.inner_steps - 1)):
            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                # Recompute positions internally for safety when using inputs_embeds
                position_ids=None,
                past_key_values=None,
                inputs_embeds=hidden_states,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

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
