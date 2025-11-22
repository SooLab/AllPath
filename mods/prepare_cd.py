from copy import deepcopy

import torch

from .graber import graber

transformers_dynamic_cache_available = True
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    transformers_dynamic_cache_available = False


def prepare_kwargs_for_cd(input_ids, model_kwargs):
    input_ids_cd = graber.get("input_ids_cd", None)
    use_cd: bool = graber["use_cd"] is not None

    if use_cd:
        if input_ids_cd is None:
            input_ids_cd = input_ids.clone()
        cd_type = graber["cd_type"]
        model_kwargs_cd = model_kwargs.copy()

        if "inputs_embeds_cd" in graber.keys():
            model_kwargs_cd["inputs_embeds"] = graber["inputs_embeds_cd"]

        if "attention_mask_cd" in graber.keys():
            model_kwargs_cd["attention_mask"] = graber["attention_mask_cd"]

        if "input_scaling_cd" in graber.keys():
            model_kwargs_cd["input_scaling"] = graber["input_scaling_cd"]

        if "position_ids_cd" in graber.keys():
            model_kwargs_cd["position_ids"] = graber["position_ids_cd"]

        if "pixel_values_cd" in graber.keys():
            model_kwargs_cd["pixel_values"] = graber["pixel_values_cd"]

        if "pixel_values_videos_cd" in graber.keys():
            model_kwargs_cd["pixel_values_videos"] = graber["pixel_values_videos_cd"]

        if "image_grid_thw_cd" in graber.keys():
            model_kwargs_cd["image_grid_thw"] = graber["image_grid_thw_cd"]

        if "video_grid_thw_cd" in graber.keys():
            model_kwargs_cd["video_grid_thw"] = graber["video_grid_thw_cd"]

        # AllPath: Hardcode Here
        if "inputs_embeds" in model_kwargs_cd.keys():
            bs, ls = model_kwargs_cd["inputs_embeds"].shape[:2]
        else:
            bs, ls = input_ids_cd.shape[:2]

        model_kwargs_cd["attention_mask"] = torch.ones(
            bs, ls, dtype=torch.int64, device=model_kwargs["attention_mask"].device
        )
        if "cache_position" in model_kwargs_cd.keys():
            model_kwargs_cd["cache_position"] = torch.arange(
                ls,
                dtype=model_kwargs_cd["cache_position"].dtype,
                device=model_kwargs_cd["cache_position"].device,
            )
        if (
            transformers_dynamic_cache_available
            and "past_key_values" in model_kwargs_cd.keys()
            and isinstance(model_kwargs_cd["past_key_values"], DynamicCache)
        ):
            model_kwargs_cd["past_key_values"] = deepcopy(
                model_kwargs["past_key_values"]
            )

    else:
        model_kwargs_cd = None
        cd_type = None

    if "inputs_embeds_cd" in graber.keys():
        del graber["inputs_embeds_cd"]
    if "attention_mask_cd" in graber.keys():
        del graber["attention_mask_cd"]
    if "input_scaling_cd" in graber.keys():
        del graber["input_scaling_cd"]
    if "position_ids_cd" in graber.keys():
        del graber["position_ids_cd"]
    if "pixel_values_cd" in graber.keys():
        del graber["pixel_values_cd"]
    if "image_grid_thw_cd" in graber.keys():
        del graber["image_grid_thw_cd"]
    if "pixel_values_videos_cd" in graber.keys():
        del graber["pixel_values_videos_cd"]
    if "video_grid_thw_cd" in graber.keys():
        del graber["video_grid_thw_cd"]

    return input_ids, input_ids_cd, model_kwargs, model_kwargs_cd, use_cd, cd_type
