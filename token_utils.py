import pickle
from dataclasses import dataclass
from typing import Optional

from playground.chair.chair import CHAIR, WordNetLemmatizer, nltk, wordnet

evaluator: CHAIR = pickle.load(open("./playground/chair/chair.pkl", "rb"))
assert type(evaluator) is CHAIR


@dataclass
class AlignedTokens:
    start: list[Optional[int]]
    end: list[Optional[int]]
    tokens: list[str]
    caption: str


def get_tokens_position(
    input_ids, qs, tokenizer
) -> list[tuple[Optional[int], Optional[int]]]:
    output: list[tuple[Optional[int], Optional[int]]] = []
    end_pos = 0

    for token in input_ids:
        decoded_token = tokenizer.decode(
            token, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if decoded_token == "<0x0A>":  # LLAMA
            decoded_token = "\n"
        start_pos = qs.find(decoded_token, end_pos)
        if start_pos == -1:
            output.append((None, None))
        else:
            end_pos = start_pos + len(decoded_token)
            output.append((start_pos, end_pos))

    return output


def has_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def get_overlap_tokens(
    token_slices: list[tuple[Optional[int], Optional[int]]],
    query_slices: list[tuple[int, int]],
) -> list[int]:
    output = []
    for i, (token_start, token_end) in enumerate(token_slices):
        if token_start is None or token_end is None:
            continue
        for query_start, query_end in query_slices:
            if has_overlap((token_start, token_end), (query_start, query_end)):
                output.append(i)
                break
    return output


def get_token_indices(caption: str, tokens: list[str]) -> AlignedTokens:
    start_poses: list[Optional[int]] = []
    end_poses: list[Optional[int]] = []
    end_pos = 0
    for token in tokens:
        start_pos = caption.find(token, end_pos)
        if start_pos == -1:
            start_poses.append(None)
            end_poses.append(None)
        else:
            end_pos = start_pos + len(token)
            start_poses.append(start_pos)
            end_poses.append(end_pos)

    assert len(start_poses) == len(end_poses) == len(tokens)

    return AlignedTokens(
        start=start_poses, end=end_poses, tokens=tokens, caption=caption
    )


def new_caption_to_words(self: CHAIR, caption: str):
    # Adapted from https://github.com/Maxlinn/CHAIR-metric-standalone/blob/main/chair.py

    # standard preprocessing
    words = nltk.word_tokenize(caption.lower())
    tagged_sent = nltk.pos_tag(words)
    lemmas_sent = []
    wnl = WordNetLemmatizer()
    for tag in tagged_sent:
        wordnet_pos = self.get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
    # words = [singularize(w) for w in words]
    origin_words = words
    words = lemmas_sent

    # replace double words
    i = 0
    double_words = []
    origin_double_words = []  # AllPath: Added
    idxs = []
    while i < len(words):
        idxs.append(i)
        double_word = " ".join(words[i : i + 2])
        origin_double_word = " ".join(origin_words[i : i + 2])  # AllPath: Added
        if double_word in self.double_word_dict:
            double_words.append(self.double_word_dict[double_word])
            origin_double_words.append(origin_double_word)  # AllPath: Added
            i += 2
        else:
            double_words.append(words[i])
            origin_double_words.append(origin_words[i])  # AllPath: Added
            i += 1
    words = double_words
    double_words = origin_double_words  # AllPath: Added

    # toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
    if ("toilet" in words) & ("seat" in words):
        words = [word for word in words if word != "seat"]

    # get synonyms for all words in the caption
    # TODO: check what if then?
    idxs = [idx for idx, word in enumerate(words) if word in set(self.mscoco_objects)]
    # idxs = [idxs[idx] for idx, word in enumerate(words) if word in set(self.mscoco_objects)]
    words = [word for word in words if word in set(self.mscoco_objects)]
    node_words = []
    for word in words:
        node_words.append(self.inverse_synonym_dict[word])
    # return all the MSCOCO objects in the caption
    return words, node_words, idxs, double_words
