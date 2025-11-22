# Derived and modified from Tongyi Qianwen Materials

from .graber import graber
from .qwen_vl_chat.modeling_qwen import *
from .qwen_vl_chat.modeling_qwen import (
    _ERROR_BAD_CHAT_FORMAT,
    _ERROR_STREAM_IN_CHAT,
    _SENTINEL,
)
from .vcd_add_noise import add_diffusion_noise


def new_QWenAttention__attn(
    self: QWenAttention,
    query,
    key,
    value,
    registered_causal_mask,
    attention_mask=None,
    head_mask=None,
    pai_alpha: Optional[float] = None,  # AllPath: Added
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if self.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [],
            value.size(-1) ** 0.5,
            dtype=attn_weights.dtype,
            device=attn_weights.device,
        )

    query_length, key_length = query.size(-2), key.size(-2)
    # causal_mask = self.bias[
    #     :, :, key_length - query_length : key_length, :key_length
    # ]
    # mask_value = torch.finfo(attn_weights.dtype).min
    # mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
    #     attn_weights.device
    # )
    # attn_weights = torch.where(
    #     causal_mask, attn_weights.to(attn_weights.dtype), mask_value
    # )
    attn_weights = attn_weights + attention_mask

    # AllPath: Modification Starts =====
    img_start_idx = graber["image_start_pos"]
    img_end_idx = graber["image_end_pos"]
    if pai_alpha is not None:
        attn_weights[:, :, -1, img_start_idx:img_end_idx] = (
            attn_weights[:, :, -1, img_start_idx:img_end_idx].abs() * pai_alpha
            + attn_weights[:, :, -1, img_start_idx:img_end_idx]
        )
    # AllPath: Modification Ends =====

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights


def new_QWenAttention_forward(
    self: QWenAttention,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    rotary_pos_emb: Optional[List[torch.Tensor]] = None,
    registered_causal_mask: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    hallu_heads: Optional[list[int]] = None,  # AllPath: Added
    good_heads: Optional[list[int]] = None,  # AllPath: Added
    in_scale: Optional[float] = None,  # AllPath: Added
    de_scale: Optional[float] = None,  # AllPath: Added
    output_log_prob_increase: bool = False,  # AllPath: Added
    pai_alpha: Optional[float] = None,  # AllPath: Added
):
    mixed_x_layer = self.c_attn(hidden_states)

    query, key, value = mixed_x_layer.split(self.split_size, dim=2)

    query = self._split_heads(query, self.num_heads, self.head_dim)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    value = self._split_heads(value, self.num_heads, self.head_dim)

    if rotary_pos_emb is not None:
        cur_len = query.shape[1]
        rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
        rotary_pos_emb = (rotary_pos_emb,) * 2
        q_pos_emb, k_pos_emb = rotary_pos_emb
        # Slice the pos emb for current inference
        query = apply_rotary_pos_emb(query, q_pos_emb)
        key = apply_rotary_pos_emb(key, k_pos_emb)

    if layer_past is not None:
        past_key, past_value = layer_past[0], layer_past[1]
        key = torch.cat((past_key, key), dim=1)
        value = torch.cat((past_value, value), dim=1)

    if use_cache:
        present = (key, value)
    else:
        present = None

    if self.use_logn_attn and not self.training:
        if (
            self.logn_tensor.device != query.device
            or self.logn_tensor.dtype != query.dtype
        ):
            self.logn_tensor = self.logn_tensor.to(query.device).type_as(query)
        seq_start = key.size(1) - query.size(1)
        seq_end = key.size(1)
        logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :]
        query = query * logn_tensor.expand_as(query)

    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    attn_output, attn_weight = new_QWenAttention__attn(
        self,
        query,
        key,
        value,
        registered_causal_mask,
        attention_mask,
        head_mask,
        pai_alpha,
    )

    # AllPath: Modification Starts =====
    modify_heads = bool(hallu_heads or good_heads)
    if in_scale is None:
        in_scale = 2
    if de_scale is None:
        de_scale = 0

    if modify_heads or output_log_prob_increase:
        raw_attn_output = attn_output.permute(0, 2, 1, 3)

        bsz, num_head, q_len, head_dim = raw_attn_output.shape
        out_features = self.c_proj.out_features
        c_proj_slices = self.c_proj.weight.T
        c_proj_slices = c_proj_slices.reshape(num_head, head_dim, out_features)
        raw_attn_output = raw_attn_output @ c_proj_slices
    else:
        raw_attn_output = None

    # new_attn_output = raw_attn_output.sum(dim=1)
    # breakpoint()

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
        scale = scale.reshape(bsz, num_head, q_len, 1)
        attn_output = scale * raw_attn_output
        attn_output = attn_output.sum(dim=1)
    else:
        # AllPath: Modification Ends =====
        context_layer = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output: torch.Tensor = self.c_proj(context_layer)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weight,)

    return outputs, raw_attn_output


def new_QWenBlock_forward(
    self: QWenBlock,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    rotary_pos_emb: Optional[List[torch.Tensor]] = None,
    registered_causal_mask: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    lm_head: Optional[nn.Linear] = None,  # AllPath: Added
    hallu_heads: Optional[list[int]] = None,  # AllPath: Added
    good_heads: Optional[list[int]] = None,  # AllPath: Added
    in_scale: Optional[float] = None,  # AllPath: Added
    de_scale: Optional[float] = None,  # AllPath: Added
    output_log_prob_increase: bool = False,  # AllPath: Added
    pai_alpha: Optional[float] = None,  # AllPath: Added
):
    layernorm_output = self.ln_1(hidden_states)

    attn_outputs, hidden_states_per_head = self.attn(
        layernorm_output,
        rotary_pos_emb,
        registered_causal_mask=registered_causal_mask,
        layer_past=layer_past,
        attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        hallu_heads=hallu_heads,  # AllPath: Added
        good_heads=good_heads,  # AllPath: Added
        in_scale=in_scale,  # AllPath: Added
        de_scale=de_scale,  # AllPath: Added
        output_log_prob_increase=output_log_prob_increase,  # AllPath: Added
        pai_alpha=pai_alpha,  # AllPath: Added
    )
    attn_output = attn_outputs[0]

    outputs = attn_outputs[1:]

    residual = hidden_states
    layernorm_input = attn_output + residual

    # AllPath: Modification Starts =====
    if output_log_prob_increase:
        residual_per_head = residual.unsqueeze(1)
        hidden_states_per_head = residual_per_head + hidden_states_per_head

        # get logit only for the last token input
        residual_hs = residual_per_head[:, :, -1, :]
        hs_per_head = hidden_states_per_head[:, :, -1, :]
    # AllPath: Modification Ends =====

    layernorm_output = self.ln_2(layernorm_input)

    residual = layernorm_input
    mlp_output = self.mlp(layernorm_output)
    hidden_states = residual + mlp_output

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    # AllPath: Modification Starts =====
    if output_log_prob_increase:
        outputs += (residual_hs, hs_per_head)
    # AllPath: Modification Ends =====

    return outputs


def new_QWenModel_forward(
    self: QWenModel,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    if past_key_values is None and torch.any(
        input_ids == self.config.visual["image_start_id"]
    ):
        bos_pos = torch.where(input_ids == self.config.visual["image_start_id"])
        eos_pos = torch.where(input_ids == self.config.visual["image_start_id"] + 1)
        assert (bos_pos[0] == eos_pos[0]).all()
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
        images = []
        for i, a, b in img_pos:
            image = input_ids[i][a + 1 : b - 1].tolist()
            image = image[: image.index(self.config.visual["image_start_id"] + 2)]
            images.append(bytes(image).decode("utf-8"))

        images = self.visual.encode(images)
        if graber.get("is_vcd_branch", False):  # AllPath: Added
            images = add_diffusion_noise(images, graber["noise_step"])
        assert images.shape[0] == len(images)
        fake_images = None
    elif self.training:
        fake_images = torch.zeros(1, 3, 224, 224).to(
            dtype=self.visual.conv1.weight.dtype, device=self.visual.conv1.weight.device
        )
        images = self.visual(fake_images)
    else:
        fake_images = None
        images = None

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

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        past_length = past_key_values[0][0].size(-2)

    if position_ids is None:
        position_ids = torch.arange(
            past_length,
            input_shape[-1] + past_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    encoder_attention_mask = None
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

    if batch_size <= 0:
        raise ValueError("batch_size has to be defined and > 0")
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_length
    )

    hidden_states = inputs_embeds

    kv_seq_len = hidden_states.size()[1]
    if past_key_values[0] is not None:
        # past key values[0][0] shape: bs * seq_len * head_num * dim
        kv_seq_len += past_key_values[0][0].shape[1]
    if (
        self.use_dynamic_ntk
        and kv_seq_len == hidden_states.size()[1]
        and not self.training
    ):
        context_value = math.log(kv_seq_len / self.seq_length, 2) + 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)
    else:
        ntk_alpha = self.rotary_emb._ntk_alpha_cached

    rotary_pos_emb = self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha)
    for idx in range(len(rotary_pos_emb)):
        rotary_pos_emb[idx] = rotary_pos_emb[idx].to(hidden_states.device)

    hidden_states = self.drop(hidden_states).clone()
    if fake_images is not None:
        hidden_states = hidden_states + images.mean() * 0
    elif images is not None:
        for idx, (i, a, b) in enumerate(img_pos):
            hidden_states[i][a + 1 : b] = images[idx]
    output_shape = input_shape + (hidden_states.size(-1),)

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None

    # AllPath: Modification Starts =====
    hallu_heads = graber.get("hallu_heads", None)
    good_heads = graber.get("good_heads", None)
    in_scale = graber.get("in_scale", None)
    de_scale = graber.get("de_scale", None)
    pai_alpha = graber.get("pai_alpha", None)

    output_log_prob_increase = "residual_hs" in graber and "hs_per_head" in graber
    all_residual_hs = () if output_log_prob_increase else None
    all_hs_per_head = () if output_log_prob_increase else None
    lm_head = graber["lm_head"]
    # AllPath: Modification Ends =====
    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
        # AllPath: Modification Starts =====
        cur_hallu_heads = None if hallu_heads is None else hallu_heads[i]
        cur_good_heads = None if good_heads is None else good_heads[i]

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
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                rotary_pos_emb,
                self.registered_causal_mask,
                None,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                lm_head=lm_head,  # AllPath: Added
                hallu_heads=cur_hallu_heads,  # AllPath: Added
                good_heads=cur_good_heads,  # AllPath: Added
                in_scale=in_scale,  # AllPath: Added
                de_scale=de_scale,  # AllPath: Added
                output_log_prob_increase=output_log_prob_increase,  # AllPath: Added
                pai_alpha=pai_alpha if i > 2 else None,  # AllPath: Added
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                rotary_pos_emb=rotary_pos_emb,
                registered_causal_mask=self.registered_causal_mask,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                lm_head=lm_head,  # AllPath: Added
                hallu_heads=cur_hallu_heads,  # AllPath: Added
                good_heads=cur_good_heads,  # AllPath: Added
                in_scale=in_scale,  # AllPath: Added
                de_scale=de_scale,  # AllPath: Added
                output_log_prob_increase=output_log_prob_increase,  # AllPath: Added
                pai_alpha=pai_alpha if i >= 2 else None,  # AllPath: Added
            )

        # AllPath: Modification Starts =====
        if output_log_prob_increase:
            residual_hs, hs_per_head = outputs[-2], outputs[-1]
            all_residual_hs += (residual_hs,)
            all_hs_per_head += (hs_per_head,)
        # AllPath: Modification Ends =====

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (
                outputs[2 if use_cache else 1],
            )

    # AllPath: Modification Starts =====
    if output_log_prob_increase:
        graber["residual_hs"] += (all_residual_hs,)
        graber["hs_per_head"] += (all_hs_per_head,)
    # AllPath: Modification Ends =====

    hidden_states = self.ln_f(hidden_states)
    hidden_states = hidden_states.view(output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, presents, all_hidden_states] if v is not None
        )

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def new_QWenLMHeadModel_forward(
    self: QWenLMHeadModel,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    graber["lm_head"] = self.lm_head

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = transformer_outputs[0]

    lm_logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        labels = labels.to(lm_logits.device)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )


def new_QWenLMHeadModel_chat(
    self: QWenLMHeadModel,
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: Optional[HistoryType],
    system: str = "You are a helpful assistant.",
    append_history: bool = True,
    stream: Optional[bool] = _SENTINEL,  # type: ignore
    stop_words_ids: Optional[List[List[int]]] = None,
    generation_config: Optional[GenerationConfig] = None,
    **kwargs,
) -> Tuple[str, HistoryType]:
    from mods.new_qwen_generation_utils import make_context  # AllPath: Added

    generation_config = (
        generation_config if generation_config is not None else self.generation_config
    )
    assert generation_config is not None

    assert stream is _SENTINEL, _ERROR_STREAM_IN_CHAT
    assert generation_config.chat_format == "chatml", _ERROR_BAD_CHAT_FORMAT
    if history is None:
        history = []
    if stop_words_ids is None:
        stop_words_ids = []

    max_window_size = kwargs.get("max_window_size", None)
    if max_window_size is None:
        max_window_size = generation_config.max_window_size
    raw_text, context_tokens = make_context(
        tokenizer,
        query,
        history=history,
        system=system,
        max_window_size=max_window_size,
        chat_format=generation_config.chat_format,
    )

    pai_cfg = graber.pop("pai_cfg", None)
    if pai_cfg is not None:
        from transformers.generation.logits_process import LogitsProcessorList

        from mods.pai_cfg import init_cfg_processor

        kwargs["logits_processor"] = LogitsProcessorList(
            [
                init_cfg_processor(
                    tokenizer, self, [raw_text], pai_cfg, 1, 2, 32, "Qwen-VL-Chat"
                )
            ]
        )  # start layer and end layer of PAI is hardcoded

    input_ids = torch.tensor([context_tokens]).to(self.device)

    # AllPath: Modification Starts =====
    use_cd = graber.get("use_cd", None)
    if use_cd is not None:
        if use_cd == "icd":
            raw_text_cd, context_tokens_cd = make_context(
                tokenizer,
                query,
                history=history,
                system=system,
                max_window_size=max_window_size,
                chat_format=generation_config.chat_format,
                use_cd=use_cd,
            )
            input_ids_cd = torch.tensor([context_tokens_cd]).to(self.device)
        elif use_cd == "vcd":
            input_ids_cd = input_ids.clone()
        else:
            raise ValueError(f"Unknown use_cd value: {use_cd}")
    else:
        input_ids_cd = None
    graber["input_ids_cd"] = input_ids_cd
    # AllPath: Modification Ends =====

    stop_words_ids.extend(get_stop_words_ids(generation_config.chat_format, tokenizer))

    # AllPath: Modification Starts =====
    graber["input_ids"] = input_ids
    bs, _ = input_ids.shape
    assert bs == 1, f"Batch size for input should be 1, got {bs}."
    graber["image_start_pos"] = torch.where(input_ids[0] == 151857)[0][0].item() + 1
    graber["image_end_pos"] = torch.where(input_ids[0] == 151858)[0][0].item()
    # AllPath: Modification Ends =====

    outputs = self.generate(
        input_ids,
        stop_words_ids=stop_words_ids,
        # return_dict_in_generate=False,  # AllPath: removed
        generation_config=generation_config,
        **kwargs,
    )

    # AllPath: Modification Starts =====
    if "return_dict_in_generate" in kwargs.keys() and kwargs["return_dict_in_generate"]:
        output_ids = outputs["sequences"]
    else:
        output_ids = outputs
        outputs = None
    # AllPath: Modification Ends =====

    response = decode_tokens(
        output_ids[0],  # AllPath: Modified
        tokenizer,
        raw_text_len=len(raw_text),
        context_length=len(context_tokens),
        chat_format=generation_config.chat_format,
        verbose=False,
        errors="replace",
    )

    if append_history:
        history.append((query, response))

    return response, outputs  # AllPath: Modified
