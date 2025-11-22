import argparse
import json
import random
import tempfile
import types
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.nn import functional as F
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
from token_utils import (
    AlignedTokens,
    evaluator,
    get_token_indices,
    new_caption_to_words,
)

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import PreTrainedTokenizerBase

output_score_file = None
delete_exist = False
debug = False


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


def get_data_pope(
    model: LM,
    lm_head: torch.nn.Linear,
    residual_hs: tuple[tuple["Tensor", ...], ...],
    hs_per_head: tuple[tuple["Tensor", ...], ...],
    gt: str,
) -> list[dict[Any, Any]]:
    good_dd = {}
    bad_dd = {}

    assert len(residual_hs) == 1 and len(hs_per_head) == 1  # bs==1
    residual_hs_cur = residual_hs[0]
    hs_per_head_cur = hs_per_head[0]

    if isinstance(model, LLaVA):
        if gt in ["yes", "no"]:
            good_token_ids = set()
            bad_token_ids = set()
            # LLAMA:
            # Yes: 3869
            # No: 1939
            if gt == "yes":
                good_token_ids.add(3869)
                bad_token_ids.add(1939)
            elif gt == "no":
                good_token_ids.add(1939)
                bad_token_ids.add(3869)
        elif gt in ["A", "B", "C", "D"]:
            good_token_ids = set()
            bad_token_ids = {319, 350, 315, 360}
            # LLAMA:
            # A: 319
            # B: 350
            # C: 315
            # D: 360
            if gt == "A":
                bad_token_ids.remove(319)
                good_token_ids.add(319)
            elif gt == "B":
                bad_token_ids.remove(350)
                good_token_ids.add(350)
            elif gt == "C":
                bad_token_ids.remove(315)
                good_token_ids.add(315)
            elif gt == "D":
                bad_token_ids.remove(360)
                good_token_ids.add(360)
            assert len(good_token_ids) == 1
            assert len(bad_token_ids) == 3
        else:
            raise ValueError()
    elif isinstance(model, QwenVL):
        if gt in ["yes", "no"]:
            good_token_ids = set()
            bad_token_ids = set()
            # Qwen:
            # yes: 9693
            # no: 2152
            if gt == "yes":
                good_token_ids.add(9693)
                bad_token_ids.add(2152)
            elif gt == "no":
                good_token_ids.add(2152)
                bad_token_ids.add(9693)
        elif gt in ["A", "B", "C", "D"]:
            good_token_ids = set()
            bad_token_ids = {32, 33, 34, 35}
            # Qwen:
            # A: 32
            # B: 33
            # C: 34
            # D: 35
            if gt == "A":
                bad_token_ids.remove(32)
                good_token_ids.add(32)
            elif gt == "B":
                bad_token_ids.remove(33)
                good_token_ids.add(33)
            elif gt == "C":
                bad_token_ids.remove(34)
                good_token_ids.add(34)
            elif gt == "D":
                bad_token_ids.remove(35)
                good_token_ids.add(35)
            assert len(good_token_ids) == 1
            assert len(bad_token_ids) == 3
        else:
            raise ValueError()
    else:
        raise ValueError()

    for layer_id, (hs, hs_) in enumerate(zip(residual_hs_cur, hs_per_head_cur)):
        L = lm_head(hs)
        L_ = lm_head(hs_)

        good_d = {}
        bad_d = {}

        assert len(L) == len(L_) == 1  # bs==1
        L = L[0]  # type: ignore[reportConstantRedefinition]
        L_ = L_[0]  # type: ignore[reportConstantRedefinition]

        cur_good_token_ids = good_token_ids.copy()
        cur_bad_token_ids = bad_token_ids.copy()

        assert len(cur_good_token_ids & cur_bad_token_ids) == 0

        cur_good_token_ids = list(cur_good_token_ids)
        cur_bad_token_ids = list(cur_bad_token_ids)

        prob = F.softmax(L.to(torch.float32), dim=-1)
        prob_ = F.softmax(L_.to(torch.float32), dim=-1)

        good_prob = prob[:, cur_good_token_ids].sum(dim=-1)
        good_prob_ = prob_[:, cur_good_token_ids].sum(dim=-1)
        bad_prob = prob[:, cur_bad_token_ids].sum(dim=-1)
        bad_prob_ = prob_[:, cur_bad_token_ids].sum(dim=-1)

        good_log_prob_i = good_prob_.log() - good_prob.log()
        bad_log_prob_i = bad_prob_.log() - bad_prob.log()

        for head_id, score in enumerate(good_log_prob_i):
            good_d[head_id] = score.item()

        for head_id, score in enumerate(bad_log_prob_i):
            bad_d[head_id] = score.item()

        good_dd[layer_id] = good_d
        bad_dd[layer_id] = bad_d

    return [
        {"is_safe": True, "data": good_dd},
        {"is_safe": False, "data": bad_dd},
    ]


def get_data_chair(
    tokenizer: "PreTrainedTokenizerBase",
    lm_head: torch.nn.Linear,
    output_ids: "Tensor",
    residual_hs: tuple[tuple["Tensor", ...], ...],
    hs_per_head: tuple[tuple["Tensor", ...], ...],
    coco_id: int,
) -> list[dict[Any, Any]]:
    assert len(residual_hs) == len(hs_per_head) == len(output_ids)

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
    tqdm.write(f"COCO {coco_id}, GT: {gt_objs}")

    output_list = []

    for sel_text_token, ptr in enumerate(output_tokens_2_chair_tokens_ptr):
        if ptr is None:
            continue
        output_id = output_ids[sel_text_token]
        output_token = output_tokens.tokens[sel_text_token]
        chair_token = chair_tokens.tokens[ptr]
        coco_token = coco_words[ptr]
        is_safe = coco_token in gt_objs
        tqdm.write(
            f"[{sel_text_token:>3}] {output_id} -> {output_token} -> {chair_token} -> {coco_token} ({'Safe' if is_safe else 'Hallucination'})"
        )

        rp = F.softmax(lm_head(torch.stack(residual_hs[sel_text_token])), dim=-1).to(
            torch.float32
        )
        pph = F.softmax(lm_head(torch.stack(hs_per_head[sel_text_token])), dim=-1).to(
            torch.float32
        )
        log_p_i = torch.log(pph) - torch.log(rp)

        del rp, pph

        ddd = {}
        for layer_id, layer in enumerate(log_p_i):
            dd = {}
            assert layer.shape[0] == 1  # bs == 1
            layer = layer[0, :, output_id]
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

    lm_head: torch.nn.Linear = self.model.lm_head

    with (
        tempfile.TemporaryFile("w")
        if debug
        else safe_open(output_score_file, mode="w" if delete_exist else "x")
    ) as file_handle:
        for original_idx in tqdm(indices):
            prompt, image_path, user_log_dict = benchmark[original_idx]
            graber.clear()
            graber["residual_hs"] = ()
            graber["hs_per_head"] = ()
            _, output, item = self(
                prompt,
                image_path,
                user_log_dict=user_log_dict,
                question_id=original_idx,
                **kwargs,
            )
            log_list.append(item)
            residual_hs = graber.pop("residual_hs")
            hs_per_head = graber.pop("hs_per_head")
            assert user_log_dict is not None
            assert output is not None
            assert item is not None

            if isinstance(benchmark, ResampledPOPE) or isinstance(
                benchmark, ResampledMCQPOPE
            ):
                datalist = get_data_pope(
                    self, lm_head, residual_hs, hs_per_head, user_log_dict["GT"]  # type: ignore
                )
            elif isinstance(benchmark, ResampledCHAIR):
                if isinstance(self, LLaVA):
                    start_pos = 1
                elif isinstance(self, QwenVL):
                    start_pos = len(graber["input_ids"][0])
                else:
                    raise ValueError()
                datalist = get_data_chair(
                    self.tokenizer,
                    lm_head,
                    output.sequences[0, start_pos:],
                    residual_hs,
                    hs_per_head,
                    item["COCO_id"],
                )
            else:
                raise ValueError()

            del residual_hs, hs_per_head  # do not remove

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
        from mods import new_modeling_llama

        new_modeling_llama.register()
        model = LLaVA()
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

    output_score_file = f"./heads_ours/{model.name}/details-{bench_name}-format.jsonl"

    if remain_args:
        print_warning(f"Cli args {remain_args} are not used.")

    # Inference
    with torch.inference_mode():
        model.eval(bench, return_dict_in_generate=True, **kwargs)

    # Get heads list
    get_scores.main(Path(output_score_file))
