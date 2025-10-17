from transformers.models.modernbert.modeling_modernbert import _unpad_modernbert_input, _pad_modernbert_output, ModernBertForSequenceClassification, BaseModelOutput, SequenceClassifierOutput, _prepare_4d_attention_mask, is_triton_available
from typing import Optional, Union, List
import torch
from loguru import logger
import torch.nn as nn

class MultiHeadModernBert(nn.Module):
    def __init__(self, models: List[ModernBertForSequenceClassification]):
        super().__init__()
        # Find the last common layer by comparing parameters across all models
        last_common_layer = 0
        num_layers = len(models[0].model.layers)
        
        for layer_idx in range(num_layers):
            layer_is_common = True
            reference_layer = models[0].model.layers[layer_idx]
            
            # Compare this layer across all models
            for model_idx, model in enumerate(models[1:], 1):
                current_layer = model.model.layers[layer_idx]
                
                # Compare all parameters in this layer
                for (ref_name, ref_param), (cur_name, cur_param) in zip(
                    reference_layer.named_parameters(), 
                    current_layer.named_parameters()
                ):
                    if ref_name != cur_name or not torch.equal(ref_param, cur_param):
                        layer_is_common = False
                        break
                
                if not layer_is_common:
                    break
            
            if layer_is_common:
                last_common_layer = layer_idx + 1
            else:
                break
        
        logger.info(f"Found {last_common_layer} common layers out of {num_layers} total layers")
        self.base_model_layers = models[0].model.layers[:last_common_layer]

        self.classifiers = nn.ModuleList(
            [model.classifier for model in models]
        )
        self.class_layers = nn.ModuleList(
            [model.model.layers[last_common_layer:] for model in models]
        )
        self.heads = nn.ModuleList(
            [model.head for model in models]
        )
        self.embeddings = models[0].model.embeddings
        self.final_norm = models[0].model.final_norm
        self.config = models[0].config
        self.dtype = models[0].dtype
        self.device = models[0].device
        self.warn_if_padding_and_no_attention_mask = models[0].warn_if_padding_and_no_attention_mask

    def _update_attention_mask(self, attention_mask: torch.Tensor, output_attentions: bool) -> torch.Tensor:
        if output_attentions:
            if self.config._attn_implementation == "sdpa":
                logger.warning_once(
                    "Outputting attentions is only supported with the 'eager' attention implementation, "
                    'not with "sdpa". Falling back to `attn_implementation="eager"`.'
                )
                self.config._attn_implementation = "eager"
            elif self.config._attn_implementation != "eager":
                logger.warning_once(
                    "Outputting attentions is only supported with the eager attention implementation, "
                    f'not with {self.config._attn_implementation}. Consider setting `attn_implementation="eager"`.'
                    " Setting `output_attentions=False`."
                )

        global_attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)

        # Create position indices
        rows = torch.arange(global_attention_mask.shape[2]).unsqueeze(0)
        # Calculate distance between positions
        distance = torch.abs(rows - rows.T)

        # Create sliding window mask (1 for positions within window, 0 outside)
        window_mask = (
            (distance <= self.config.local_attention // 2).unsqueeze(0).unsqueeze(0).to(attention_mask.device)
        )
        # Combine with existing mask
        sliding_window_mask = global_attention_mask.masked_fill(window_mask.logical_not(), torch.finfo(self.dtype).min)

        return global_attention_mask, sliding_window_mask

    def _maybe_set_compile(self):
        if self.config.reference_compile is False:
            return

        if hasattr(self, "hf_device_map") and len(self.hf_device_map) > 1:
            if self.config.reference_compile:
                logger.warning_once(
                    "If `accelerate` split the model across devices, `torch.compile` will not work. "
                    "Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        if self.device.type == "mps":
            if self.config.reference_compile:
                logger.warning_once(
                    "Compiling the model with `torch.compile` and using a `torch.mps` device is not supported. "
                    "Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        if self.device.type == "cpu":
            if self.config.reference_compile:
                logger.warning_once(
                    "Compiling the model with `torch.compile` and using a `torch.cpu` device is not supported. "
                    "Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        if self.config.reference_compile is None:
            self.config.reference_compile = is_triton_available()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, ...], SequenceClassifierOutput]:
        r"""
        sliding_window_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
            perform global attention, while the rest perform local attention. This mask is used to avoid attending to
            far-away tokens in the local attention layers when not using Flash Attention.
        indices (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*):
            Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
        cu_seqlens (`torch.Tensor` of shape `(batch + 1,)`, *optional*):
            Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
        max_seqlen (`int`, *optional*):
            Maximum sequence length in the batch excluding padding tokens. Used to unpad input_ids and pad output tensors.
        batch_size (`int`, *optional*):
            Batch size of the input sequences. Used to pad the output tensors.
        seq_len (`int`, *optional*):
            Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
        """


        self._maybe_set_compile()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        self._maybe_set_compile()

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)

        if batch_size is None and seq_len is None:
            if inputs_embeds is not None:
                batch_size, seq_len = inputs_embeds.shape[:2]
            else:
                batch_size, seq_len = input_ids.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

        original_attention_mask = attention_mask

        repad = False
        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                repad = True
                if inputs_embeds is None:
                    with torch.no_grad():
                        input_ids, indices, cu_seqlens, max_seqlen, *_ = _unpad_modernbert_input(
                            inputs=input_ids, attention_mask=attention_mask
                        )
                else:
                    inputs_embeds, indices, cu_seqlens, max_seqlen, *_ = _unpad_modernbert_input(
                        inputs=inputs_embeds, attention_mask=attention_mask
                    )
        else:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            attention_mask, sliding_window_mask = self._update_attention_mask(
                attention_mask, output_attentions=output_attentions
            )

        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

        for encoder_layer in self.base_model_layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        common_hidden_states = hidden_states
        output_logits = torch.zeros(batch_size, len(self.class_layers), device=device)
        for i in range(len(self.class_layers)):
            hidden_states = common_hidden_states

            for encoder_layer in self.class_layers[i]:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    sliding_window_mask=sliding_window_mask,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]
            hidden_states = self.final_norm(hidden_states)

            if repad:
                hidden_states = _pad_modernbert_output(
                    inputs=hidden_states, indices=indices, batch=batch_size, seqlen=seq_len
                )

            if self.config.classifier_pooling == "cls":
                last_hidden_state = hidden_states[:, 0]
            elif self.config.classifier_pooling == "mean":
                last_hidden_state = (hidden_states * original_attention_mask.unsqueeze(-1)).sum(dim=1) / original_attention_mask.sum(
                    dim=1, keepdim=True
                )
        
            pooled_output = self.heads[i](last_hidden_state)
            logits = self.classifiers[i](pooled_output)
            output_logits[:, i] = logits[:, 0]

        if not return_dict:
            return tuple(v for v in [output_logits, all_hidden_states, all_self_attentions] if v is not None)

        return SequenceClassifierOutput(
            logits=output_logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )