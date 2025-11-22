import argparse
import json
import random
import re
import tempfile
import types
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from pycocotools.coco import COCO
from tqdm import tqdm

import get_scores
from mods.graber import graber
from more_benchmarks import ResampledCHAIR, ResampledMCQPOPE, ResampledPOPE
from playground import get_eval_benchmark_from_args, seed_everything
from playground._utils._colors import print_note, print_warning
from playground._utils._path import safe_open
from playground.benchmarks import BenchBase
from playground.chair.chair import CHAIR  # do not remove
from playground.models import LM, LLaVA, QwenVL
from playground.path_table import get_path_from_table
from token_utils import (
    AlignedTokens,
    evaluator,
    get_overlap_tokens,
    get_token_indices,
    get_tokens_position,
    new_caption_to_words,
)

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import PreTrainedTokenizerBase


output_score_file = None
delete_exist = False
debug = False
coco = COCO(get_path_from_table("COCO annotation"))
categories = coco.loadCats(coco.getCatIds())
category_inv_map = {cat["name"]: cat["id"] for cat in categories}
grid_h: int
grid_w: int


def get_objects(qs: str) -> tuple[list[str], list[tuple[int, int]]]:
    # POPE
    match_obj = re.match(
        r"Is there an? (.*) in the iman?ge\? Please answer this question with one word\.",
        qs,
    )
    if match_obj is not None:
        return [match_obj.group(1)], [match_obj.span(1)]

    # MCQ_POPE
    match_obj = re.match(
        r"Which of the following (?:objects )?(?:appears |does not appear |is (?:not )?)in the image\?(?:\n| )A\. (.*)(?:\n| )B\. (.*)(?:\n| )C\. (.*)(?:\n| )D\. (.*)(?:\n| )(?:Please answer this question with one word\.|Answer with the option\'s letter from the given choices directly\.)",
        qs,
    )
    if match_obj is not None:
        return [match_obj.group(i) for i in range(1, 5)], [
            match_obj.span(i) for i in range(1, 5)
        ]

    raise ValueError()


def new_eval_model_pretrained(self, args, disable_conv_mode_warning=False, **kwargs):
    # Copied and modified from LLaVA: llava/eval/run_llava.py
    # Major changes:
    # 1. Support 0 image input;
    # 2. Support `return_dict_in_generate=True` for dev usage.

    import re

    import torch
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_PLACEHOLDER,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.utils import disable_torch_init

    # Model
    disable_torch_init()

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if args.image_file is not None:
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in self.model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in self.model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in self.model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in self.model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in self.model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if (
        conv_mode == "llava_v0"
        and args.conv_mode is None
        and not disable_conv_mode_warning
    ):
        print_warning(
            "The auto inferred conversation mode 'llava_v0' is currently being used for the LLaVA model. However, this is uncommon. This warning may appear because your model name does not match certain expected keywords. Using the incorrect conversation mode could result in performance decrease. Therefore, it is recommended to do a double-check. To disable this warning, you can pass `disable_conv_mode_warning=True` to this function."
        )

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        if not disable_conv_mode_warning:
            print_warning(
                "The auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}. To disable this warning, you can pass `disable_conv_mode_warning=True` to this function.".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if args.image_file is None:
        images_tensor = None
        image_sizes = None
    else:
        image_files = self.image_parser(args)
        images = self.load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images, self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .cuda()
    )

    graber["input_ids"] = input_ids

    with torch.inference_mode():
        output = self.model.generate(
            input_ids, images=images_tensor, image_sizes=image_sizes, **kwargs
        )

    if not isinstance(output, torch.Tensor):
        output_ids = output.sequences
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
    else:
        response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[
            0
        ].strip()
        output = None

    return response, output


def submit(self, prompt, image=None, question_id=None, **kwargs):
    if image is None:
        query = prompt
    else:
        query = self.tokenizer.from_list_format(
            [
                {"image": image},
                {"text": prompt},
            ]
        )
    response, output = self.model.chat(
        self.tokenizer, query=query, history=None, **kwargs
    )
    return response, output, None


def get_coco_seg_mask(image_id, query_object):
    ann_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(ann_ids)
    query_id = category_inv_map[query_object]
    global_mask = None

    for ann in annotations:
        category_id = ann["category_id"]
        if category_id == query_id:
            mask = coco.annToMask(ann)
            if global_mask is None:
                global_mask = mask
            else:
                global_mask |= mask

    if global_mask is None:
        return None

    h, w = global_mask.shape

    i_points = np.linspace(0, h, grid_h + 1)
    i_points = np.round(i_points).astype(int)
    j_points = np.linspace(0, w, grid_w + 1)
    j_points = np.round(j_points).astype(int)

    def is_overlap(mask, i1, i2, j1, j2):
        bbox_mask = np.zeros_like(mask)
        bbox_mask[i1:i2, j1:j2] = 1
        overlap_mask = np.logical_and(mask, bbox_mask)
        overlap_area = np.sum(overlap_mask)
        return overlap_area > 0

    tokens_mask = []
    for i1, i2 in zip(i_points, i_points[1:]):
        for j1, j2 in zip(j_points, j_points[1:]):
            tokens_mask.append(is_overlap(global_mask, i1, i2, j1, j2))

    assert len(tokens_mask) == grid_h * grid_w

    tokens_mask = np.array(tokens_mask, dtype=int)

    return tokens_mask


def get_image_attention(output, positions):
    assert output is not None
    sel_text_token = 0

    attns = output["attentions"][sel_text_token]
    attns: torch.Tensor = torch.stack(attns, dim=0)  # type:ignore

    attns = attns[:, 0, :, positions, :]
    attns = attns.mean(dim=2)

    image_start_pos = graber["image_start_pos"]
    image_end_pos = graber["image_end_pos"]

    image_attns = attns[:, :, image_start_pos:image_end_pos]

    return image_attns


def get_data_pope(
    model: LM, benchmark: "BenchBase", prompt: str, image_id: int, output, gt: str
) -> list[dict[Any, Any]]:
    objects, query_positions = get_objects(prompt)
    tokens_mask = [get_coco_seg_mask(image_id, o) for o in objects]

    if isinstance(benchmark, ResampledPOPE):
        assert gt in ("yes", "no")
        assert len(tokens_mask) == 1
        if gt == "yes" and tokens_mask[0] is None:
            return []
        if gt == "no" and tokens_mask[0] is not None:
            return []
    elif isinstance(benchmark, ResampledMCQPOPE):
        if all(x is None for x in tokens_mask) or all(
            x is not None for x in tokens_mask
        ):
            return []
    else:
        raise ValueError()

    input_ids = graber["input_ids"]

    if isinstance(model, LLaVA):
        match_offset = torch.where(input_ids[0] == -200)[-1].item() + 3
        match_ids = get_tokens_position(
            input_ids[0][match_offset:-5], prompt, model.tokenizer
        )

        object_tokens = [
            torch.tensor(
                get_overlap_tokens(match_ids, [qp]),
                dtype=torch.int,
                device=input_ids.device,
            )
            + match_offset
            + 575
            for qp in query_positions
        ]
    elif isinstance(model, QwenVL):
        match_offset = torch.where(input_ids[0] == 151858)[-1].item() + 2
        match_ids = get_tokens_position(
            input_ids[0][match_offset:-5], prompt, model.tokenizer
        )

        object_tokens = [
            torch.tensor(
                get_overlap_tokens(match_ids, [qp]),
                dtype=torch.int,
                device=input_ids.device,
            )
            + match_offset
            for qp in query_positions
        ]
    else:
        raise ValueError()

    image_attns = [get_image_attention(output, p) for p in object_tokens]
    image_attns = torch.stack(image_attns)

    yes_cases = [i for i, tm in enumerate(tokens_mask) if tm is not None]
    no_cases = [i for i, tm in enumerate(tokens_mask) if tm is None]

    out_list: list[dict[Any, Any]] = []

    if yes_cases:
        yes_image_attns = image_attns[yes_cases]
        segmentation_masks = (
            torch.tensor(
                [tm for tm in tokens_mask if tm is not None],
                dtype=torch.int,
                device=input_ids.device,
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )
        pos_scores = torch.sum(yes_image_attns * segmentation_masks, dim=-1)
        for item in pos_scores:
            dd = {}
            for layer_n, layer in enumerate(item):
                d = {}
                for head_n, score in enumerate(layer):
                    d[head_n] = score.item()
                dd[layer_n] = d
            out_list.append({"is_safe": True, "data": dd})

    if no_cases:
        no_image_attns = image_attns[no_cases]
        neg_scores = torch.sum(no_image_attns, dim=-1)
        for item in neg_scores:
            dd = {}
            for layer_n, layer in enumerate(item):
                d = {}
                for head_n, score in enumerate(layer):
                    d[head_n] = score.item()
                dd[layer_n] = d
            out_list.append({"is_safe": False, "data": dd})

    return out_list


def get_data_chair(
    tokenizer: "PreTrainedTokenizerBase",
    output_ids: "Tensor",
    attentions: tuple[tuple["Tensor", ...], ...],
    coco_id: int,
) -> list[dict[Any, Any]]:
    assert len(attentions) == len(output_ids)

    caption = tokenizer.decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    output_tokens = [
        tokenizer.decode(
            token_id, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        for token_id in output_ids
    ]
    output_tokens = get_token_indices(caption, output_tokens)

    _, coco_words, coco_words_2_chair_tokens_ptr, chair_tokens = new_caption_to_words(
        evaluator, caption
    )

    chair_tokens = get_token_indices(caption.lower(), chair_tokens)
    chair_tokens = AlignedTokens(
        start=[chair_tokens.start[i] for i in coco_words_2_chair_tokens_ptr],
        end=[chair_tokens.end[i] for i in coco_words_2_chair_tokens_ptr],
        tokens=[chair_tokens.tokens[i] for i in coco_words_2_chair_tokens_ptr],
        caption=chair_tokens.caption,
    )

    output_tokens_2_chair_tokens_ptr: list[Optional[int]] = []

    if chair_tokens.start:
        i = 0
        for _, s, e in zip(
            output_tokens.tokens, output_tokens.start, output_tokens.end
        ):
            start_i = chair_tokens.start[i]
            end_i = chair_tokens.end[i]
            if s is None or e is None:
                output_tokens_2_chair_tokens_ptr.append(None)
            elif start_i is None or end_i is None:
                output_tokens_2_chair_tokens_ptr.append(None)
            elif start_i <= e:
                output_tokens_2_chair_tokens_ptr.append(i)
                i += 1
                if i >= len(chair_tokens.start):
                    break
            else:
                output_tokens_2_chair_tokens_ptr.append(None)

    gt_objs = evaluator.imid_to_objects[coco_id]
    tqdm.write(f"COCO {coco_id}, CHAIR-GT: {gt_objs}")

    output_list = []
    appeared_tokens = set()
    for sel_text_token, ptr in enumerate(output_tokens_2_chair_tokens_ptr):
        if ptr is None:
            continue
        output_id = output_ids[sel_text_token]
        output_token = output_tokens.tokens[sel_text_token]
        chair_token = chair_tokens.tokens[ptr]
        coco_token = coco_words[ptr]
        # is_safe = coco_token in gt_objs

        if coco_token in appeared_tokens:
            continue
        appeared_tokens.add(coco_token)

        attn = attentions[sel_text_token]
        attn = torch.stack(attn).to(torch.float32)
        segmentation = get_coco_seg_mask(coco_id, coco_token)
        if segmentation is None:
            segmentation = torch.ones(
                grid_h * grid_w, dtype=torch.int, device=attn.device
            )
            is_safe = False
        else:
            segmentation = torch.tensor(
                segmentation, dtype=torch.int, device=attn.device
            )
            is_safe = True

        image_start_pos = graber["image_start_pos"]
        image_end_pos = graber["image_end_pos"]

        image_attns = attn[:, 0, :, -1, image_start_pos:image_end_pos]

        tqdm.write(
            f"[{sel_text_token:>3}] {output_id} -> {output_token} -> {chair_token} -> {coco_token} ({'Safe' if is_safe else 'Hallucination'})"
        )

        s = torch.sum(image_attns * segmentation, dim=-1)

        ddd = {}
        for layer_id, layer in enumerate(s):
            dd = {}
            for head_id, head in enumerate(layer):
                dd[head_id] = head.item()
            ddd[layer_id] = dd

        output_list.append(
            {
                "coco_id": coco_id,
                "sel_text_token": sel_text_token,
                "output_id": output_id.item(),
                "output_token": output_token,
                "chair_token": chair_token,
                "coco_token": coco_token,
                "is_safe": is_safe,
                "data": ddd,
            }
        )

    return output_list


def new_eval(self: LM, benchmark: "BenchBase", shuffle=False, **kwargs) -> None:
    global output_score_file, delete_exist, debug

    indices = list(range(len(benchmark)))
    if shuffle:
        random.shuffle(indices)

    log_list = []

    assert output_score_file is not None

    with (
        tempfile.TemporaryFile("w")
        if debug
        else safe_open(output_score_file, mode="w" if delete_exist else "x")
    ) as file_handle:
        for original_idx in tqdm(indices):
            prompt, image_path, user_log_dict = benchmark[original_idx]
            graber.clear()
            _, output, item = self(
                prompt,
                image_path,
                user_log_dict=user_log_dict,
                question_id=original_idx,
                **kwargs,
            )
            log_list.append(item)
            assert user_log_dict is not None
            assert output is not None
            assert item is not None

            if isinstance(benchmark, ResampledPOPE) or isinstance(
                benchmark, ResampledMCQPOPE
            ):
                datalist = get_data_pope(
                    self,
                    benchmark,
                    prompt,
                    int(Path(image_path).stem.split("_")[-1]),
                    output,
                    user_log_dict["GT"],  # type:ignore
                )
            elif isinstance(benchmark, ResampledCHAIR):
                if isinstance(self, LLaVA):
                    start_pos = 1
                elif isinstance(self, QwenVL):
                    start_pos = len(graber["input_ids"][0])
                else:
                    raise ValueError()
                datalist = get_data_chair(
                    model.tokenizer,
                    output.sequences[0, start_pos:],
                    output.attentions,  # type:ignore
                    item["COCO_id"],
                )
            else:
                raise ValueError()

            for d in datalist:
                file_handle.write(json.dumps(d) + "\n")

    if self.log_file_path is not None:
        log_file_name = Path(self.log_file_path).stem
    else:
        time_str = datetime.now().strftime("%Y%m%dT%H%M%S")
        log_file_name = Path(f"{self.name}--{time_str}")

    log_file_path = f"./eval_responses/{log_file_name}.json"

    benchmark.get_score(log_list, log_file_path)


if __name__ == "__main__":
    LM.eval = new_eval

    # load model and benchmarks
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("-D", "--overwrite", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    args, remain_args = parser.parse_known_args()

    model_name: str = args.model.lower()

    print_note(f"Using model {model_name}.")

    if model_name == "llava":
        from mods import new_llava_llama

        new_llava_llama.register()
        LLaVA.new_eval_model_pretrained = new_eval_model_pretrained
        model = LLaVA()
        grid_h = 24
        grid_w = 24
    elif model_name == "qwenvl":
        model = QwenVL()
        model.submit = types.MethodType(submit, model)
        from mods.new_modeling_qwen import (
            new_QWenAttention_forward,
            new_QWenBlock_forward,
            new_QWenLMHeadModel_chat,
            new_QWenLMHeadModel_forward,
            new_QWenModel_forward,
        )

        model.model.chat = types.MethodType(  # type: ignore
            new_QWenLMHeadModel_chat, model.model
        )
        model.model.forward = types.MethodType(new_QWenLMHeadModel_forward, model.model)
        model.model.transformer.forward = types.MethodType(
            new_QWenModel_forward, model.model.transformer
        )
        for block in model.model.transformer.h:
            block.forward = types.MethodType(new_QWenBlock_forward, block)
            block.attn.forward = types.MethodType(new_QWenAttention_forward, block.attn)
        grid_h = 16
        grid_w = 16
    else:
        raise ValueError()

    seed_everything(args.seed)

    if args.overwrite:
        delete_exist = True
    if args.debug:
        debug = True

    bench, remain_args = get_eval_benchmark_from_args(remain_args)

    if isinstance(bench, ResampledPOPE):
        kwargs = {"do_sample": False, "max_new_tokens": 1}
        bench_name = f"resampled-{bench.dataset}-{bench.split}"
    elif isinstance(bench, ResampledMCQPOPE):
        kwargs = {"do_sample": False, "max_new_tokens": 1}
        bench_name = f"resampled-mcq-{bench.dataset}-{bench.split}"
    elif isinstance(bench, ResampledCHAIR):
        kwargs = {"do_sample": False, "max_new_tokens": 512}
        bench_name = "resampled-chair"
    else:
        raise ValueError()

    output_score_file = f"./heads_ours/{model.name}/details-{bench_name}-image.jsonl"

    if remain_args:
        print_warning(f"Cli args {remain_args} are not used.")

    # Inference
    with torch.inference_mode():
        model.eval(
            bench, return_dict_in_generate=True, output_attentions=True, **kwargs
        )

    # Get heads list
    if isinstance(model, QwenVL):
        get_scores.main(Path(output_score_file), 0)
    else:
        get_scores.main(Path(output_score_file))
