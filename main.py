import argparse
import random
import re
import types
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Optional, Sequence

from tqdm import tqdm

from mods.graber import graber
from more_benchmarks import MCQPOPE
from playground import (
    get_eval_benchmark_from_args,
    get_generation_params_from_args,
    load_structured_file,
    seed_everything,
)
from playground._utils._colors import print_note, print_warning
from playground.benchmarks import CHAIR as ChairBench
from playground.benchmarks import MME, POPE, BenchBase
from playground.models import LM, LLaVA, QwenVL


def new_eval_model_pretrained(
    self: LLaVA, args, disable_conv_mode_warning=False, **kwargs
):
    # Copied and modified from LLaVA: llava/eval/run_llava.py
    # Major changes:
    # 1. Support 0 image input;
    # 2. Support `return_dict_in_generate=True` for dev usage.

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

    # AllPath: Modification starts here =====
    # CD
    input_ids_cd = None
    images_tensor_cd = None
    use_cd = graber["use_cd"]
    if use_cd is not None:
        if use_cd == "vcd":
            noise_step = graber.pop("noise_step", None)

            from mods.vcd_add_noise import add_diffusion_noise

            images_tensor_cd = add_diffusion_noise(images_tensor, noise_step)
            input_ids_cd = None

        elif use_cd == "icd":
            qs_cd = (
                "You are a confused objects detector to provide a fuzzy overview or impression of the image. "
                + args.query
            )
            if args.image_file is not None:
                if IMAGE_PLACEHOLDER in qs:
                    if self.model.config.mm_use_im_start_end:
                        qs_cd = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs_cd)
                    else:
                        qs_cd = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs_cd)
                else:
                    if self.model.config.mm_use_im_start_end:
                        qs_cd = image_token_se + "\n" + qs_cd
                    else:
                        qs_cd = DEFAULT_IMAGE_TOKEN + "\n" + qs_cd

            if args.conv_mode is not None and conv_mode != args.conv_mode:
                pass
            else:
                args.conv_mode = conv_mode

            conv_cd = conv_templates[args.conv_mode].copy()
            conv_cd.append_message(conv.roles[0], qs_cd)
            conv_cd.append_message(conv.roles[1], None)
            prompt_cd = conv_cd.get_prompt()

            input_ids_cd = (
                tokenizer_image_token(
                    prompt_cd, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )
            images_tensor_cd = None

        else:
            raise ValueError()

    # PAI
    pai_cfg = graber.pop("pai_cfg", None)
    if pai_cfg is not None:
        from transformers.generation.logits_process import LogitsProcessorList

        from mods.pai_cfg import init_cfg_processor

        kwargs["logits_processor"] = LogitsProcessorList(
            [
                init_cfg_processor(
                    self.tokenizer, self.model, [prompt], pai_cfg, 1, 2, 32, "llava-1.5"
                )
            ]
        )  # start layer and end layer of PAI is hardcoded
    # AllPath: Modification ends here =====

    graber["input_ids_cd"] = input_ids_cd
    graber["images_cd"] = images_tensor_cd
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


def prepare_heads(
    heads_list: Sequence[tuple[tuple[int, int], int]]
) -> defaultdict[int, list[int]]:
    formatted = defaultdict(list)
    for (layer, head), _ in heads_list:
        formatted[layer].append(head)
    return formatted


def prepare_adhh_heads(heads_list: Sequence[list[int]]) -> defaultdict[int, list[int]]:
    formatted = defaultdict(list)
    for layer, head in heads_list:
        formatted[layer].append(head)
    return formatted


def new_eval(
    self: "LM",
    benchmark: "BenchBase",
    shuffle: bool = False,
    # AllPath: Start of Addition =====
    hallu_heads: Optional[dict[int, list[int]]] = None,
    good_heads: Optional[dict[int, list[int]]] = None,
    in_scale: Optional[float] = None,
    de_scale: Optional[float] = None,
    heads: Optional[dict[int, list[int]]] = None,
    adhh_threshold: Optional[float] = None,
    pai_alpha: Optional[float] = None,
    pai_cfg: Optional[float] = None,
    use_cd: Optional[str] = None,
    cd_alpha: Optional[float] = None,
    cd_beta: Optional[float] = None,
    noise_step: Optional[int] = None,
    cd_type: Optional[str] = None,
    # AllPath: End of Addition =====
    **kwargs,
) -> None:
    indices = list(range(len(benchmark)))
    if shuffle:
        random.shuffle(indices)

    log_list = []

    for original_idx in tqdm(indices):
        prompt, image_path, user_log_dict = benchmark[original_idx]

        graber.clear()

        graber["hallu_heads"] = hallu_heads
        graber["good_heads"] = good_heads
        graber["in_scale"] = in_scale
        graber["de_scale"] = de_scale
        graber["heads"] = heads
        graber["adhh_threshold"] = adhh_threshold
        graber["pai_alpha"] = pai_alpha
        graber["pai_cfg"] = pai_cfg
        graber["use_cd"] = use_cd
        graber["cd_alpha"] = cd_alpha
        graber["cd_beta"] = cd_beta
        graber["noise_step"] = noise_step
        graber["cd_type"] = cd_type

        _, _, item = self(
            prompt,
            image_path,
            user_log_dict=user_log_dict,
            question_id=original_idx,
            **kwargs,
        )
        log_list.append(item)

    if self.log_file_path is not None:
        log_file_name = Path(self.log_file_path).stem
    else:
        time_str = datetime.now().strftime("%Y%m%dT%H%M%S")
        log_file_name = Path(f"{self.name}--{time_str}")

    log_file_path = f"./eval_responses/{log_file_name}.json"

    benchmark.get_score(log_list, log_file_path)


if __name__ == "__main__":
    # load model and benchmarks
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava")
    parser.add_argument("--method", type=str, default="allpath")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-path", type=Path, default=None)

    # AllPath
    parser.add_argument("--good-format-heads", type=int, default=None)
    parser.add_argument("--hallu-format-heads", type=int, default=None)
    parser.add_argument("--good-image-heads", type=int, default=None)
    parser.add_argument("--hallu-image-heads", type=int, default=None)
    parser.add_argument("--format-heads-list-path", type=Path, default=None)
    parser.add_argument("--image-heads-list-path", type=Path, default=None)
    parser.add_argument("--in-scale", type=float, default=2.0)
    parser.add_argument("--de-scale", type=float, default=0.0)

    # General CD algorithms' parameters (VCD & ICD here)
    parser.add_argument("--cd-alpha", type=float, default=1.0)
    parser.add_argument("--cd-beta", type=float, default=0.1)

    # VCD
    parser.add_argument("--noise-step", type=int, default=500)

    # PAI
    parser.add_argument("--pai-alpha", type=float, default=None)
    parser.add_argument("--pai-cfg", type=float, default=1.1)

    # AD-HH
    parser.add_argument("--adhh-threshold", type=float, default=0.4)

    args, remain_args = parser.parse_known_args()

    seed_everything(args.seed)

    method: str = args.method.lower()
    if method == "ours":
        method = "allpath"

    model_name: str = args.model.lower()

    assert model_name in ["llava", "qwenvl"]
    assert method in ["baseline", "vcd", "icd", "pai", "adhh", "allpath"]

    print_note(f"Using model {model_name}.")
    print_note(f"Using method {method}.")

    LM.eval = new_eval
    if model_name == "llava":
        from transformers.generation.utils import GenerationMixin

        from mods import new_llava_llama, new_modeling_llama
        from mods.search_methods_4_37_2 import new_greedy_search, new_sample

        new_llava_llama.register()
        new_modeling_llama.register()
        LLaVA.new_eval_model_pretrained = new_eval_model_pretrained

        GenerationMixin.greedy_search = new_greedy_search
        GenerationMixin.sample = new_sample

        model = LLaVA()
    elif model_name == "qwenvl":
        from transformers.generation.utils import GenerationMixin

        from mods.new_modeling_qwen import (
            new_QWenAttention_forward,
            new_QWenBlock_forward,
            new_QWenLMHeadModel_chat,
            new_QWenLMHeadModel_forward,
            new_QWenModel_forward,
        )
        from mods.search_methods_4_32_0 import new_greedy_search, new_sample

        # CD Algorithm
        GenerationMixin.greedy_search = new_greedy_search
        GenerationMixin.sample = new_sample

        model = QwenVL()

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
    else:
        raise ValueError()

    kwargs, remain_args = get_generation_params_from_args(remain_args)
    bench, remain_args = get_eval_benchmark_from_args(remain_args)

    if bench is None:
        raise ValueError(
            "You should use --eval <benchmark name> to assign a benchmark."
        )

    if remain_args:
        print_warning(f"Cli args {remain_args} are not used.")

    if method == "allpath":
        # Select hyperparameter for AllPath
        good_format_heads: Optional[int] = args.good_format_heads
        hallu_format_heads: Optional[int] = args.hallu_format_heads
        good_image_heads: Optional[int] = args.good_image_heads
        hallu_image_heads: Optional[int] = args.hallu_image_heads
        format_heads_list_path: Optional[Path] = args.format_heads_list_path
        image_heads_list_path: Optional[Path] = args.image_heads_list_path
        if isinstance(bench, ChairBench) and model_name == "llava":
            if good_format_heads is None:
                good_format_heads = 40
            if hallu_format_heads is None:
                hallu_format_heads = 40
            if good_image_heads is None:
                good_image_heads = 50
            if hallu_image_heads is None:
                hallu_image_heads = 0
        else:
            if good_format_heads is None:
                good_format_heads = 20
            if hallu_format_heads is None:
                hallu_format_heads = 20
            if good_image_heads is None:
                good_image_heads = 10
            if hallu_image_heads is None:
                hallu_image_heads = 0
        if isinstance(bench, POPE) or isinstance(bench, MME):
            bench_name = f"resampled-coco-adversarial"
        elif isinstance(bench, MCQPOPE):
            bench_name = f"resampled-mcq-coco-adversarial"
        elif isinstance(bench, ChairBench):
            bench_name = "resampled-chair"
        else:
            raise ValueError()

        if format_heads_list_path is None:
            format_heads_list_path = Path(
                f"./heads_ours/{model.name}/heads-{bench_name}-format.jsonl"
            )
        if image_heads_list_path is None:
            image_heads_list_path = Path(
                f"./heads_ours/{model.name}/heads-{bench_name}-image.jsonl"
            )

        # Get heads
        format_heads_list = load_structured_file(format_heads_list_path)
        image_heads_list = load_structured_file(image_heads_list_path)

        hallu_format_heads_list = format_heads_list[:hallu_format_heads]
        hallu_image_heads_list = image_heads_list[:hallu_image_heads]

        format_heads_list.reverse()
        image_heads_list.reverse()

        good_format_heads_list = format_heads_list[:good_format_heads]
        good_image_heads_list = image_heads_list[:good_image_heads]

        hallu_heads = prepare_heads(hallu_format_heads_list + hallu_image_heads_list)
        good_heads = prepare_heads(good_format_heads_list + good_image_heads_list)

        in_scale: float = args.in_scale
        de_scale: float = args.de_scale

        print_method_kwargs = pformat(
            {
                "format_heads": format_heads_list_path,
                "image_heads": image_heads_list_path,
                "hallu_format_heads": hallu_format_heads,
                "good_format_heads": good_format_heads,
                "hallu_image_heads": hallu_image_heads,
                "good_image_heads": good_image_heads,
                "in_scale": in_scale,
                "de_scale": de_scale,
            },
            sort_dicts=False,
        )

        method_kwargs = {
            "hallu_heads": hallu_heads,
            "good_heads": good_heads,
            "in_scale": in_scale,
            "de_scale": de_scale,
        }
    elif method == "baseline":
        print_method_kwargs = "baseline"
        method_kwargs = {}
    elif method == "vcd":
        method_kwargs = {
            "use_cd": "vcd",
            "cd_alpha": args.cd_alpha,
            "cd_beta": args.cd_beta,
            "noise_step": args.noise_step,
            "cd_type": "contrastive",
        }
        print_method_kwargs = pformat(method_kwargs)
    elif method == "icd":
        method_kwargs = {
            "use_cd": "icd",
            "cd_alpha": args.cd_alpha,
            "cd_beta": args.cd_beta,
            "cd_type": "contrastive",
        }
        print_method_kwargs = pformat(method_kwargs)
    elif method == "pai":
        pai_alpha: Optional[float] = args.pai_alpha
        pai_cfg: float = args.pai_cfg
        if isinstance(bench, ChairBench):
            if pai_alpha is None:
                pai_alpha = 0.5
        else:
            if pai_alpha is None:
                pai_alpha = 0.2

        print_method_kwargs = pformat(
            {"pai_alpha": pai_alpha, "pai_cfg": pai_cfg}, sort_dicts=False
        )
        method_kwargs: dict[str, Any] = {"pai_alpha": pai_alpha, "pai_cfg": pai_cfg}
    elif method == "adhh":
        if model_name == "llava":
            if model.name == "llava-v1.5-7b":
                adhh_heads_list = [
                    [16, 29],
                    [26, 9],
                    [13, 31],
                    [15, 10],
                    [20, 12],
                    [30, 9],
                    [19, 18],
                    [17, 0],
                    [18, 9],
                    [26, 28],
                    [19, 27],
                    [18, 26],
                    [15, 25],
                    [14, 16],
                    [31, 26],
                    [15, 24],
                    [31, 3],
                    [22, 20],
                    [27, 29],
                    [17, 28],
                ]
            elif model.name == "llava-v1.5-13b":
                adhh_heads_list = [
                    [0, 8],
                    [29, 27],
                    [23, 18],
                    [20, 11],
                    [36, 26],
                    [19, 37],
                    [22, 16],
                    [22, 34],
                    [21, 31],
                    [20, 34],
                    [37, 11],
                    [17, 25],
                    [35, 10],
                    [17, 5],
                    [15, 26],
                    [0, 22],
                    [19, 5],
                    [19, 0],
                    [14, 1],
                    [23, 20],
                    [21, 6],
                    [30, 24],
                    [26, 27],
                    [21, 32],
                    [15, 28],
                    [15, 31],
                    [19, 30],
                    [20, 8],
                    [19, 14],
                    [14, 9],
                    [39, 26],
                    [25, 1],
                    [18, 32],
                    [17, 27],
                    [39, 32],
                ]
            else:
                raise ValueError()
        elif model_name == "qwenvl":
            raise ValueError()

        adhh_threshold: float = args.adhh_threshold
        adhh_heads = prepare_adhh_heads(adhh_heads_list)

        print_method_kwargs = pformat({"adhh_threshold": adhh_threshold})
        method_kwargs = {"heads": adhh_heads, "adhh_threshold": adhh_threshold}
    else:
        raise ValueError()

    print_note(f"Got method parameters:\n{print_method_kwargs}")

    if args.save_path is not None:
        model.log_file_path = args.save_path

    print_note(f"Results will be saved to {model.log_file_path}")

    # eval benchmark
    model.eval(bench, **kwargs, **method_kwargs)
