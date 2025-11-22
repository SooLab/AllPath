import argparse
import math
from collections import defaultdict
from pathlib import Path

from playground import load_structured_file, save_structured_file


def main(lfp: Path, scale: float = 1.0):
    safe_total_numbers = defaultdict(int)
    safe_total_scores = defaultdict(float)

    hallu_total_numbers = defaultdict(int)
    hallu_total_scores = defaultdict(float)

    scores = defaultdict(float)

    sfp = lfp.with_name(lfp.name.replace("details", "heads"))

    lf = load_structured_file(lfp)

    for line in lf:
        is_safe = line["is_safe"]

        line = line["data"]
        for layer_id, item in line.items():
            layer_id = int(layer_id)
            for head_id, score in item.items():
                head_id = int(head_id)
                key = (layer_id, head_id)

                if not math.isnan(score) and not math.isinf(score):
                    if is_safe:
                        safe_total_numbers[key] += 1
                        safe_total_scores[key] += score
                    else:
                        hallu_total_numbers[key] += 1
                        hallu_total_scores[key] += score

    safe_total_numbers = dict(safe_total_numbers)
    safe_total_scores = dict(safe_total_scores)

    hallu_total_numbers = dict(hallu_total_numbers)
    hallu_total_scores = dict(hallu_total_scores)

    assert len(safe_total_numbers) == len(safe_total_scores)
    assert len(hallu_total_numbers) == len(hallu_total_scores)
    for key, total_number in safe_total_numbers.items():
        total_score = safe_total_scores[key]
        score = total_score / total_number
        scores[key] = score

    for key, total_number in hallu_total_numbers.items():
        total_score = hallu_total_scores[key]
        score = total_score / total_number
        scores[key] -= score * scale

    scores = sorted(scores.items(), key=lambda item: item[1], reverse=False)

    save_structured_file(scores, sfp, "w")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "load_file_path",
        type=Path,
    )
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    lfp = args.load_file_path
    scale = args.scale
    main(lfp, scale)
