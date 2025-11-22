# Derived and modified from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

from transformers.models.llama.modeling_llama import *
from transformers.models.llama.modeling_llama import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

from .graber import graber


def new_LlamaAttention_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    hallu_heads: Optional[list[int]] = None,  # AllPath: Added
    good_heads: Optional[list[int]] = None,  # AllPath: Added
    in_scale: Optional[float] = None,  # AllPath: Added
    de_scale: Optional[float] = None,  # AllPath: Added
    output_log_prob_increase: bool = False,  # AllPath: Added
    norm_distribution: Optional[bool] = None,  # AllPath: Added
    heads: Optional[dict[int, list[int]]] = None,  # AllPath: Added
    adhh_threshold: Optional[float] = None,  # AllPath: Added
    enh_para: Optional[float] = None,  # AllPath: Added
    sup_para: Optional[float] = None,  # AllPath: Added
    use_vaf: bool = False,  # AllPath: Added
    pai_alpha: Optional[float] = None,  # AllPath: Added
    is_sdpa_available: bool = False,  # AllPath: Added
    **kwargs,
) -> Tuple[
    torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]], torch.Tensor
]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # AllPath: Modification Starts =====
    # PAI
    if pai_alpha is not None:
        image_start_pos = graber["image_start_pos"]
        image_end_pos = graber["image_end_pos"]
        attn_weights[:, :, -1, image_start_pos:image_end_pos] = (
            attn_weights[:, :, -1, image_start_pos:image_end_pos].abs() * pai_alpha
            + attn_weights[:, :, -1, image_start_pos:image_end_pos]
        )

    # VAF from ClearSight:
    if enh_para is not None and sup_para is not None:
        image_start_pos = graber["image_start_pos"]
        image_end_pos = graber["image_end_pos"]
        if q_len > image_end_pos:
            attn_weights[:, :, image_end_pos:, image_start_pos:image_end_pos] = (
                enh_para
                * attn_weights[:, :, image_end_pos:, image_start_pos:image_end_pos]
            )
            attn_weights[:, :, image_end_pos:, :image_start_pos] = (
                sup_para * attn_weights[:, :, image_end_pos:, :image_start_pos]
            )
        else:
            attn_weights[:, :, :, image_start_pos:image_end_pos] = (
                enh_para * attn_weights[:, :, :, image_start_pos:image_end_pos]
            )
            attn_weights[:, :, :, :image_start_pos] = (
                sup_para * attn_weights[:, :, :, :image_start_pos]
            )
    # AllPath: Modification Ends =====

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    # AllPath: Modification Starts =====
    # AD-HH:
    if adhh_threshold is not None and heads:
        image_end_pos = graber["image_end_pos"]
        for head in heads:
            aggre_attention = torch.sum(attn_weights[:, head, -1, image_end_pos:])
            if aggre_attention >= adhh_threshold:
                attn_weights[:, head, -1, image_end_pos:] = 0

    # AllPath: Modification Ends =====

    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    # AllPath: Modification Starts =====
    modify_heads = bool(hallu_heads or good_heads)
    if modify_heads:
        if in_scale is None:
            in_scale = 2
        if de_scale is None:
            de_scale = 0

    if modify_heads or output_log_prob_increase:
        raw_attn_output = attn_output

        bsz, num_head, q_len, head_dim = raw_attn_output.shape
        out_features = self.o_proj.out_features
        o_proj_slices = self.o_proj.weight.T
        o_proj_slices = o_proj_slices.reshape(num_head, head_dim, out_features)
        raw_attn_output = raw_attn_output @ o_proj_slices
    else:
        raw_attn_output = None

    if modify_heads:
        scale = torch.ones(
            bsz,
            num_head,
            q_len,
            dtype=raw_attn_output.dtype,
            device=raw_attn_output.device,
        )
        scale[:, hallu_heads, :] = de_scale
        scale[:, good_heads, :] = in_scale

        if norm_distribution:
            scale = scale * (num_head / scale.sum(dim=-2, keepdim=True))

        scale = scale.reshape(bsz, num_head, q_len, 1)
        attn_output = scale * raw_attn_output
        attn_output = attn_output.sum(dim=1)
    else:
        # AllPath: Modification Ends =====
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value, raw_attn_output


def new_LlamaSdpaAttention_forward(
    self: LlamaSdpaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    hallu_heads: Optional[list[int]] = None,  # AllPath: Added
    good_heads: Optional[list[int]] = None,  # AllPath: Added
    in_scale: Optional[float] = None,  # AllPath: Added
    de_scale: Optional[float] = None,  # AllPath: Added
    use_cache: bool = False,
    output_log_prob_increase: bool = False,  # AllPath: Added
    norm_distribution: Optional[bool] = None,  # AllPath: Added
    heads: Optional[dict[int, list[int]]] = None,  # AllPath: Added
    adhh_threshold: Optional[float] = None,  # AllPath: Added
    enh_para: Optional[float] = None,  # AllPath: Added
    sup_para: Optional[float] = None,  # AllPath: Added
    use_vaf: bool = False,  # AllPath: Added
    pai_alpha: Optional[float] = None,  # AllPath: Added
    is_sdpa_available: bool = False,  # AllPath: Added
) -> Tuple[
    torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]], torch.Tensor
]:
    # AllPath: Modification Starts
    if not is_sdpa_available:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "Falling back to the manual attention implementation. The reason may be shown below."
        )
        if output_attentions:
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        if adhh_threshold is not None:
            logger.warning_once(
                "Method AD-HH neads to modify attention weights, but it's not supported by `torch.nn.functional.scaled_dot_product_attention`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        if use_vaf:
            logger.warning_once(
                "Method VAF from ClearSight neads to modify attention weights, but it's not supported by `torch.nn.functional.scaled_dot_product_attention`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        if pai_alpha is not None:
            logger.warning_once(
                "Method PAI neads to modify attention weights, but it's not supported by `torch.nn.functional.scaled_dot_product_attention`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        # AllPath: Modification Ends
        return super(LlamaSdpaAttention, self).forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            hallu_heads=hallu_heads,  # AllPath: Added
            good_heads=good_heads,  # AllPath: Added
            in_scale=in_scale,  # AllPath: Added
            de_scale=de_scale,  # AllPath: Added
            output_log_prob_increase=output_log_prob_increase,  # AllPath: Added
            norm_distribution=norm_distribution,  # AllPath: Added
            heads=heads,  # AllPath: Added
            adhh_threshold=adhh_threshold,  # AllPath: Added
            enh_para=enh_para,  # AllPath: Added
            sup_para=sup_para,  # AllPath: Added
            use_vaf=use_vaf,  # AllPath: Added
            pai_alpha=pai_alpha,  # AllPath: Added
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal=self.is_causal and attention_mask is None and q_len > 1,
    )

    # AllPath: Modification Starts =====
    assert adhh_threshold is None
    assert not use_vaf

    modify_heads = bool(hallu_heads or good_heads)
    if in_scale is None:
        in_scale = 2
    if de_scale is None:
        de_scale = 0

    if modify_heads or output_log_prob_increase:
        raw_attn_output = attn_output

        bsz, num_head, q_len, head_dim = raw_attn_output.shape
        out_features = self.o_proj.out_features
        o_proj_slices = self.o_proj.weight.T
        o_proj_slices = o_proj_slices.reshape(num_head, head_dim, out_features)
        raw_attn_output = raw_attn_output @ o_proj_slices
    else:
        raw_attn_output = None

    if modify_heads:
        scale = torch.ones(
            bsz,
            num_head,
            q_len,
            dtype=raw_attn_output.dtype,
            device=raw_attn_output.device,
        )
        scale[:, hallu_heads, :] = de_scale
        scale[:, good_heads, :] = in_scale

        if norm_distribution:
            scale = scale * (num_head / scale.sum(dim=-2, keepdim=True))

        scale = scale.reshape(bsz, num_head, q_len, 1)
        attn_output = scale * raw_attn_output
        attn_output = attn_output.sum(dim=1)
    else:
        # AllPath: Modification Ends =====
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value, raw_attn_output


def new_LlamaFlashAttention2_forward(
    self: LlamaFlashAttention2,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    hallu_heads: Optional[list[int]] = None,  # AllPath: Added
    good_heads: Optional[list[int]] = None,  # AllPath: Added
    in_scale: Optional[float] = None,  # AllPath: Added
    de_scale: Optional[float] = None,  # AllPath: Added
    output_log_prob_increase: bool = False,  # AllPath: Added
    norm_distribution: Optional[bool] = None,  # AllPath: Added
    heads: Optional[dict[int, list[int]]] = None,  # AllPath: Added
    adhh_threshold: float = None,  # AllPath: Added
    enh_para: float = None,  # AllPath: Added
    sup_para: float = None,  # AllPath: Added
    use_vaf: bool = False,  # AllPath: Added
    pai_alpha: Optional[float] = None,  # AllPath: Added
    is_sdpa_available: bool = False,  # AllPath: Added
    **kwargs,
) -> Tuple[
    torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]], torch.Tensor
]:
    # LlamaFlashAttention2 attention does not support output_attentions
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = self._flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
    )

    # AllPath: Modification Starts =====
    raise NotImplementedError("Use sdpa or naive attention instead.")
    raw_attn_output = attn_output.transpose(1, 2)
    # AllPath: Modification Ends =====

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    # AllPath: Modification Starts =====
    bsz, num_head, q_len, head_dim = raw_attn_output.shape
    out_features = self.o_proj.out_features
    o_proj_slices = self.o_proj.weight.T
    o_proj_slices = o_proj_slices.reshape(num_head, head_dim, out_features)
    raw_attn_output = raw_attn_output @ o_proj_slices

    # new_attn_output = raw_attn_output.sum(dim=1)
    # print(raw_attn_output.shape)
    # print(new_attn_output - attn_output)
    # AllPath: Modification Ends =====

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value, raw_attn_output


def new_LlamaDecoderLayer_forward(
    self: LlamaDecoderLayer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    lm_head: Optional[nn.Linear] = None,  # AllPath: Added
    hallu_heads: Optional[list[int]] = None,  # AllPath: Added
    good_heads: Optional[list[int]] = None,  # AllPath: Added
    in_scale: Optional[float] = None,  # AllPath: Added
    de_scale: Optional[float] = None,  # AllPath: Added
    output_log_prob_increase: bool = False,  # AllPath: Added
    norm_distribution: Optional[bool] = None,  # AllPath: Added
    heads: Optional[dict[int, list[int]]] = None,  # AllPath: Added
    adhh_threshold: float = None,  # AllPath: Added
    enh_para: float = None,  # AllPath: Added
    sup_para: float = None,  # AllPath: Added
    use_vaf: bool = False,  # AllPath: Added
    pai_alpha: Optional[float] = None,  # AllPath: Added
    is_sdpa_available: bool = False,  # AllPath: Added
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*):
            attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
            query_sequence_length, key_sequence_length)` if default attention is used.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    # AllPath: Modification Starts =====
    (
        hidden_states,
        self_attn_weights,
        present_key_value,
        hidden_states_per_head,
    ) = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        hallu_heads=hallu_heads,
        good_heads=good_heads,
        in_scale=in_scale,
        de_scale=de_scale,
        output_log_prob_increase=output_log_prob_increase,
        norm_distribution=norm_distribution,
        heads=heads,
        adhh_threshold=adhh_threshold,
        enh_para=enh_para,
        sup_para=sup_para,
        use_vaf=use_vaf,
        pai_alpha=pai_alpha,
        is_sdpa_available=is_sdpa_available,
        **kwargs,
    )
    # AllPath: Modification Ends =====

    hidden_states = residual + hidden_states

    # AllPath: Modification Starts =====
    if output_log_prob_increase:
        residual_per_head = residual.unsqueeze(1)
        hidden_states_per_head = residual_per_head + hidden_states_per_head

        # get hidden states only for the last token input
        residual_hs = residual_per_head[:, :, -1, :]
        hs_per_head = hidden_states_per_head[:, :, -1, :]
    # AllPath: Modification Ends =====

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    # AllPath: Modification Starts =====
    if output_log_prob_increase:
        outputs += (residual_hs, hs_per_head)
    # AllPath: Modification Ends =====

    return outputs


def new_LlamaModel_forward(
    self: LlamaModel,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    # AllPath: Start of Addition =====
    lm_head: Optional[nn.Linear] = None,
    hallu_heads: Optional[dict[int, list[int]]] = None,
    good_heads: Optional[dict[int, list[int]]] = None,
    in_scale: Optional[float] = None,
    de_scale: Optional[float] = None,
    norm_distribution: Optional[bool] = None,
    heads: Optional[dict[int, list[int]]] = None,
    adhh_threshold: float = None,
    enh_para: float = None,
    sup_para: float = None,
    use_vaf: bool = False,
    pai_alpha: Optional[float] = None,
    # AllPath: End of Addition =====
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # AllPath: Modified Here
    is_sdpa_available = bool(
        not output_attentions
        and adhh_threshold is None
        and not use_vaf
        and pai_alpha is None
    )

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    # AllPath: Modified Here
    elif self._use_sdpa and is_sdpa_available:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    # AllPath: Modification Starts =====
    output_log_prob_increase = "residual_hs" in graber and "hs_per_head" in graber
    all_residual_hs = () if output_log_prob_increase else None
    all_hs_per_head = () if output_log_prob_increase else None

    for layer_idx, decoder_layer in enumerate(self.layers):
        cur_hallu_heads = None if hallu_heads is None else hallu_heads[layer_idx]
        cur_good_heads = None if good_heads is None else good_heads[layer_idx]
        cur_heads = None if heads is None else heads[layer_idx]

        if cur_hallu_heads is not None and cur_good_heads is not None:
            cur_hallu_heads = set(cur_hallu_heads)
            cur_good_heads = set(cur_good_heads)

            cur_overlap_heads = cur_hallu_heads & cur_good_heads

            cur_hallu_heads = cur_hallu_heads - cur_overlap_heads
            cur_good_heads = cur_good_heads - cur_overlap_heads

            cur_hallu_heads = list(cur_hallu_heads)
            cur_good_heads = list(cur_good_heads)
        # AllPath: Modification Ends =====
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                lm_head=lm_head,  # AllPath: Added
                hallu_heads=cur_hallu_heads,  # AllPath: Added
                good_heads=cur_good_heads,  # AllPath: Added
                in_scale=in_scale,  # AllPath: Added
                de_scale=de_scale,  # AllPath: Added
                output_log_prob_increase=output_log_prob_increase,  # AllPath: Added
                norm_distribution=norm_distribution,  # AllPath: Added
                heads=cur_heads,  # AllPath: Added
                adhh_threshold=adhh_threshold,  # AllPath: Added
                enh_para=enh_para if 8 < layer_idx < 15 else None,  # AllPath: Added
                sup_para=sup_para if 8 < layer_idx < 15 else None,  # AllPath: Added
                use_vaf=use_vaf,  # AllPath: Added
                pai_alpha=pai_alpha if layer_idx > 2 else None,  # AllPath: Added
                is_sdpa_available=is_sdpa_available,  # AllPath: Added
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                lm_head=lm_head,  # AllPath: Added
                hallu_heads=cur_hallu_heads,  # AllPath: Added
                good_heads=cur_good_heads,  # AllPath: Added
                in_scale=in_scale,  # AllPath: Added
                de_scale=de_scale,  # AllPath: Added
                output_log_prob_increase=output_log_prob_increase,  # AllPath: Added
                norm_distribution=norm_distribution,  # AllPath: Added
                heads=cur_heads,  # AllPath: Added
                adhh_threshold=adhh_threshold,  # AllPath: Added
                enh_para=enh_para if 8 < layer_idx < 15 else None,  # AllPath: Added
                sup_para=sup_para if 8 < layer_idx < 15 else None,  # AllPath: Added
                use_vaf=use_vaf,  # AllPath: Added
                pai_alpha=pai_alpha if layer_idx >= 2 else None,  # AllPath: Added
                is_sdpa_available=is_sdpa_available,  # AllPath: Added
            )

        # AllPath: Modification Starts =====
        if output_log_prob_increase:
            residual_hs, hs_per_head = layer_outputs[-2], layer_outputs[-1]
            all_residual_hs += (residual_hs,)
            all_hs_per_head += (hs_per_head,)
        # AllPath: Modification Ends =====

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    # AllPath: Modification Starts =====
    if output_log_prob_increase:
        graber["residual_hs"] += (all_residual_hs,)
        graber["hs_per_head"] += (all_hs_per_head,)
    # AllPath: Modification Ends =====

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if use_legacy_cache
            else next_decoder_cache
        )
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def new_LlamaForCausalLM_forward(
    self: LlamaForCausalLM,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # AllPath: Modification Starts =====
    hallu_heads = graber.get("hallu_heads", None)
    good_heads = graber.get("good_heads", None)
    in_scale = graber.get("in_scale", None)
    de_scale = graber.get("de_scale", None)
    norm_distribution = graber.get("norm_distribution", None)

    # PAI:
    pai_alpha = graber.get("pai_alpha", None)

    # AD-HH:
    heads = graber.get("heads", None)  # heads of other methods
    adhh_threshold = graber.get("adhh_threshold", None)

    # VAF from ClearSight:
    enh_para = graber.get("enh_para", None)
    sup_para = graber.get("sup_para", None)
    use_vaf = enh_para is not None and sup_para is not None
    # AllPath: Modification Ends =====

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        lm_head=self.lm_head,  # AllPath: Added
        hallu_heads=hallu_heads,  # AllPath: Added
        good_heads=good_heads,  # AllPath: Added
        in_scale=in_scale,  # AllPath: Added
        de_scale=de_scale,  # AllPath: Added
        norm_distribution=norm_distribution,  # AllPath: Added
        heads=heads,  # AllPath: Added
        adhh_threshold=adhh_threshold,  # AllPath: Added
        enh_para=enh_para,  # AllPath: Added
        sup_para=sup_para,  # AllPath: Added
        use_vaf=use_vaf,  # AllPath: Added
        pai_alpha=pai_alpha,  # AllPath: Added
    )

    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(
            self.vocab_size // self.config.pretraining_tp, dim=0
        )
        logits = [
            F.linear(hidden_states, lm_head_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def register():
    LlamaAttention.forward = new_LlamaAttention_forward
    LlamaSdpaAttention.forward = new_LlamaSdpaAttention_forward
    LlamaFlashAttention2.forward = new_LlamaFlashAttention2_forward
    LlamaDecoderLayer.forward = new_LlamaDecoderLayer_forward
    LlamaModel.forward = new_LlamaModel_forward
    LlamaForCausalLM.forward = new_LlamaForCausalLM_forward
