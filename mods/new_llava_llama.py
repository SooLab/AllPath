# Derived and modified from: https://github.com/haotian-liu/LLaVA/blob/main/llava/model/language_model/llava_llama.py

from llava.model.language_model.llava_llama import *

from .graber import graber
from .new_llava_arch import new_prepare_inputs_labels_for_multimodal


def new_LlavaLlamaForCausalLM_forward(
    self: LlavaLlamaForCausalLM,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    images: Optional[torch.FloatTensor] = None,
    image_sizes: Optional[List[List[int]]] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    if inputs_embeds is None:
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            image_sizes,
        )

    return super(LlavaLlamaForCausalLM, self).forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )


@torch.no_grad()
def new_LlavaLlamaForCausalLM_generate(
    self: LlavaLlamaForCausalLM,
    inputs: Optional[torch.Tensor] = None,
    images: Optional[torch.Tensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    with torch.no_grad():
        position_ids = kwargs.pop("position_ids", None)
        position_ids_cd = kwargs.pop("position_ids_cd", None)
        attention_mask = kwargs.pop("attention_mask", None)
        attention_mask_cd = kwargs.pop("attention_mask_cd", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if "inputs_embeds_cd" in kwargs:
            raise NotImplementedError("`inputs_embeds_cd` is not supported")

        # AllPath: Modification Here
        use_cd = graber.get("use_cd", None) is not None
        input_ids_cd = graber.pop("input_ids_cd", None)
        images_cd = graber.pop("images_cd", None)
        input_ids = graber["input_ids"]

        if use_cd and input_ids_cd is None and input_ids is not None:
            input_ids_cd = input_ids.clone()
        if use_cd and images_cd is None and images is not None:
            images_cd = images.clone()
        # AllPath: Modification End

        if images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = new_prepare_inputs_labels_for_multimodal(
                self,
                input_ids,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        # AllPath: Modification Here
        if use_cd:
            (
                input_ids_cd,
                position_ids_cd,
                attention_mask_cd,
                _,
                inputs_embeds_cd,
                _,
            ) = new_prepare_inputs_labels_for_multimodal(
                self,
                input_ids_cd,
                position_ids_cd,
                attention_mask_cd,
                None,
                None,
                images_cd,
                image_sizes=image_sizes,
            )
        else:
            inputs_embeds_cd = None
        # AllPath: Modification End

        graber["position_ids_cd"] = position_ids_cd
        graber["attention_mask_cd"] = attention_mask_cd
        graber["inputs_embeds_cd"] = inputs_embeds_cd

        return super(LlavaLlamaForCausalLM, self).generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


def register():
    LlavaLlamaForCausalLM.forward = new_LlavaLlamaForCausalLM_forward
    LlavaLlamaForCausalLM.generate = new_LlavaLlamaForCausalLM_generate
