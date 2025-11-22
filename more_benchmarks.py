import os

from sklearn.metrics import accuracy_score, classification_report

from playground import load_structured_file
from playground.benchmarks import BenchBase, get_path_from_table


class MCQPOPE(BenchBase):
    def __init__(self, dataset: str, split: str) -> None:
        super().__init__()

        dataset = dataset.lower()
        split = split.lower()

        self.dataset = dataset
        self.split = split

        if dataset in ["coco", "aokvqa"]:
            self.image_dirpath = get_path_from_table("COCO path")
        elif dataset == "gqa":
            self.image_dirpath = get_path_from_table("GQA path")
        else:
            raise ValueError(
                f"POPE: Dataset should be in 'coco', 'aokvqa' or 'gqa', got {repr(dataset)}."
            )

        if split not in ["random", "popular", "adversarial"]:
            raise ValueError(
                f"POPE: Dataset should be in 'random', 'popular' or 'adversarial', got {repr(split)}."
            )

        self.pope_path = f"./benchs/mcq_pope/{dataset}/{dataset}_mcq_{split}.jsonl"

        self.pope_questions = load_structured_file(self.pope_path)

    def __len__(self):
        return len(self.pope_questions)

    def __getitem__(self, index):
        item = self.pope_questions[index]
        image = item["image"]
        text = item["text"]

        return (
            f"{text}Please answer this question with one word.",
            os.path.join(self.image_dirpath, image),
            {
                "question_id": item["question_id"],
                "pope_dataset": self.dataset,
                "pope_split": self.split,
            },
        )

    def get_score(self, log_list, log_file_path) -> None:
        y_true = []
        y_pred = []

        for log_data, question_data in zip(log_list, self.pope_questions):
            idx = log_data["question_id"]

            assert idx == question_data["question_id"]
            assert log_data["pope_dataset"] == self.dataset
            assert log_data["pope_split"] == self.split

            try:
                response = log_data["response"].lower()[0]
            except:
                response = ""
            answer = question_data["label"].lower()[0]

            y_pred.append(response)
            y_true.append(answer)

        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        # Classification Report
        print()
        print(
            classification_report(
                y_true, y_pred, labels=["a", "b", "c", "d"], zero_division=0, digits=4
            )
        )


class ResampledPOPE(BenchBase):
    name = "resampled_pope"

    def __init__(self, dataset: str, split: str) -> None:
        super().__init__()

        dataset = dataset.lower()
        split = split.lower()

        self.dataset = dataset
        self.split = split

        if dataset in ["coco", "aokvqa"]:
            self.image_dirpath = get_path_from_table("COCO path")
        elif dataset == "gqa":
            self.image_dirpath = get_path_from_table("GQA path")
        else:
            raise ValueError(
                f"POPE: Dataset should be in 'coco', 'aokvqa' or 'gqa', got {repr(dataset)}."
            )

        if split not in ["random", "popular", "adversarial"]:
            raise ValueError(
                f"POPE: Dataset should be in 'random', 'popular' or 'adversarial', got {repr(split)}."
            )

        self.pope_path = os.path.join(
            "./benchs/pope", dataset, f"resampled_{dataset}_pope_{split}.jsonl"
        )

        self.pope_questions = load_structured_file(self.pope_path)

    def __len__(self):
        return len(self.pope_questions)

    def __getitem__(self, index):
        item = self.pope_questions[index]
        image = item["image"]
        text = item["text"]

        return (
            f"{text} Please answer this question with one word.",
            os.path.join(self.image_dirpath, image),
            {
                "question_id": item["question_id"],
                "pope_dataset": self.dataset,
                "pope_split": self.split,
                "GT": item["label"],  # Should pass GT to model when getting scores
            },
        )

    def get_score(self, log_list, log_file_path) -> None:
        # Since the resampled POPE is only used for head identification, to
        # prevent the results from being printed, this section was left blank
        # intentionally.
        pass


class ResampledMCQPOPE(BenchBase):
    def __init__(self, dataset: str, split: str) -> None:
        super().__init__()

        dataset = dataset.lower()
        split = split.lower()

        self.dataset = dataset
        self.split = split

        if dataset in ["coco", "aokvqa"]:
            self.image_dirpath = get_path_from_table("COCO path")
        elif dataset == "gqa":
            self.image_dirpath = get_path_from_table("GQA path")
        else:
            raise ValueError(
                f"POPE: Dataset should be in 'coco', 'aokvqa' or 'gqa', got {repr(dataset)}."
            )

        if split not in ["random", "popular", "adversarial"]:
            raise ValueError(
                f"POPE: Dataset should be in 'random', 'popular' or 'adversarial', got {repr(split)}."
            )

        self.pope_path = (
            f"./benchs/mcq_pope/{dataset}/resampled_{dataset}_mcq_{split}.jsonl"
        )

        self.pope_questions = load_structured_file(self.pope_path)

    def __len__(self):
        return len(self.pope_questions)

    def __getitem__(self, index):
        item = self.pope_questions[index]
        image = item["image"]
        text = item["text"]

        return (
            f"{text}Please answer this question with one word.",
            os.path.join(self.image_dirpath, image),
            {
                "question_id": item["question_id"],
                "pope_dataset": self.dataset,
                "pope_split": self.split,
                "GT": item["label"],  # Should pass GT to model when getting scores
            },
        )

    def get_score(self, log_list, log_file_path) -> None:
        # Since the resampled MCQ POPE is only used for head identification, to
        # prevent the results from being printed, this section was left blank
        # intentionally.
        pass


class ResampledCHAIR(BenchBase):
    fixed_500 = [
        38572,
        113246,
        133821,
        221555,
        243075,
        433993,
        488387,
        415882,
        482784,
        154670,
        166142,
        37044,
        504414,
        188318,
        426283,
        184330,
        162256,
        467511,
        97679,
        134807,
        331280,
        307034,
        235747,
        88859,
        330053,
        347558,
        512334,
        144784,
        544322,
        215608,
        175825,
        410724,
        397613,
        125936,
        147835,
        124759,
        332074,
        467000,
        456519,
        499622,
        484404,
        564743,
        260370,
        485172,
        59393,
        441874,
        158084,
        205435,
        48674,
        298372,
        201723,
        403657,
        36487,
        409572,
        308235,
        114745,
        560879,
        481239,
        110559,
        267812,
        344368,
        492608,
        79070,
        531798,
        17449,
        542959,
        133596,
        193405,
        523394,
        312509,
        81081,
        225731,
        378712,
        282346,
        333772,
        300124,
        461172,
        395749,
        461759,
        515577,
        68525,
        265552,
        144162,
        241453,
        469635,
        488997,
        248238,
        567253,
        258322,
        76249,
        411685,
        264730,
        499266,
        223362,
        162257,
        262228,
        86946,
        501420,
        370842,
        267363,
        104666,
        119194,
        314495,
        211891,
        330986,
        244215,
        395283,
        267314,
        66320,
        136988,
        175331,
        90891,
        474272,
        445298,
        56724,
        46526,
        203690,
        39480,
        443101,
        148911,
        421970,
        8612,
        418471,
        313337,
        183155,
        281573,
        519703,
        45248,
        322922,
        486864,
        158384,
        89432,
        82085,
        310538,
        375128,
        542257,
        191240,
        534456,
        507966,
        370413,
        183437,
        378940,
        215982,
        500965,
        571657,
        221872,
        17909,
        192905,
        448053,
        185802,
        215693,
        523033,
        142337,
        352671,
        340007,
        40468,
        41550,
        43961,
        325885,
        105234,
        276971,
        134160,
        267408,
        273329,
        188414,
        278226,
        80737,
        361103,
        209041,
        107481,
        106712,
        126257,
        260510,
        276260,
        158227,
        356351,
        291597,
        116848,
        62423,
        518177,
        357501,
        26609,
        364703,
        93040,
        149364,
        125071,
        561311,
        192763,
        67419,
        10707,
        302089,
        284641,
        569001,
        110765,
        419386,
        127119,
        54517,
        245453,
        304827,
        6614,
        365745,
        318837,
        182240,
        155035,
        291412,
        384527,
        62706,
        192670,
        498537,
        20541,
        425762,
        20254,
        352937,
        326667,
        229358,
        399966,
        84113,
        544456,
        72811,
        202865,
        110621,
        152120,
        519542,
        468917,
        157370,
        571865,
        198312,
        464906,
        272889,
        112160,
        383163,
        316700,
        400851,
        391889,
        314026,
        37688,
        534336,
        573667,
        292140,
        403792,
        148785,
        79121,
        365663,
        323119,
        88142,
        240023,
        319726,
        35105,
        80017,
        312343,
        413666,
        153692,
        530384,
        283012,
        14547,
        262325,
        504342,
        421499,
        238431,
        51984,
        579056,
        555696,
        176193,
        219848,
        84312,
        142879,
        102625,
        77479,
        266847,
        391253,
        362545,
        114861,
        560272,
        269721,
        251119,
        405068,
        3865,
        288399,
        91359,
        102411,
        480657,
        570116,
        514787,
        118344,
        290700,
        139917,
        453724,
        326698,
        19157,
        468996,
        106140,
        395644,
        258789,
        225133,
        357432,
        575081,
        34691,
        303731,
        140129,
        78032,
        492084,
        318426,
        165638,
        440043,
        281598,
        484695,
        490794,
        325078,
        180563,
        211775,
        315432,
        84749,
        44029,
        526536,
        253607,
        1955,
        346140,
        554037,
        100489,
        27768,
        404805,
        389804,
        90572,
        232262,
        338531,
        281397,
        173907,
        287469,
        49984,
        334083,
        296759,
        294992,
        279149,
        402499,
        500940,
        545826,
        515247,
        188386,
        331799,
        412094,
        425870,
        81782,
        391862,
        569917,
        5673,
        477867,
        25743,
        105658,
        278313,
        533137,
        270222,
        383448,
        310103,
        83217,
        87144,
        534736,
        252929,
        407945,
        83235,
        574297,
        268044,
        538330,
        480122,
        200770,
        348251,
        308512,
        535519,
        138075,
        489687,
        515727,
        431364,
        526728,
        511654,
        232592,
        8589,
        301494,
        34418,
        576354,
        11537,
        221561,
        458325,
        227413,
        8292,
        528458,
        567657,
        4840,
        413435,
        503772,
        232460,
        74181,
        427782,
        569931,
        199442,
        8923,
        88652,
        125656,
        308194,
        224368,
        166524,
        471205,
        260238,
        447043,
        199977,
        331395,
        59152,
        278203,
        384983,
        358149,
        228344,
        120767,
        490491,
        547338,
        173532,
        293794,
        445602,
        31717,
        228144,
        269662,
        22103,
        219485,
        458275,
        385057,
        421822,
        382214,
        357044,
        77270,
        461634,
        120129,
        546160,
        175908,
        580621,
        138054,
        309371,
        318209,
        69106,
        165675,
        375765,
        397354,
        17003,
        173379,
        463151,
        2157,
        158548,
        221693,
        39961,
        452881,
        76942,
        141651,
        320425,
        284991,
        426172,
        386189,
        385238,
        303413,
        115765,
        138755,
        559474,
        17769,
        421010,
        289960,
        7952,
        342711,
        536292,
        249969,
        296136,
        369470,
        61647,
        306603,
        189427,
        46616,
        221044,
        381658,
        577590,
        285773,
        530146,
        508218,
        152749,
        401157,
        322125,
        115709,
        241837,
        543985,
        71072,
        86848,
        464089,
        561698,
        146986,
        227952,
        141574,
        291680,
        454195,
        109653,
        253770,
        281365,
        296433,
        60623,
        226256,
        397857,
        543192,
        572510,
    ]

    def __init__(self, fixed: bool = True) -> None:
        super().__init__()
        self.COCO_PATH = get_path_from_table("COCO path")

        self.coco_dirpath = self.COCO_PATH
        self.sample_n = len(self.fixed_500)
        # img_files = os.listdir(self.coco_dirpath)
        self.img_files = [
            os.path.join(self.coco_dirpath, f"COCO_val2014_{coco_id:012}.jpg")
            for coco_id in self.fixed_500
        ]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        img_id = int(img_file.split(".jpg")[0][-6:])

        return (
            "Please help me describe the image in detail.",
            os.path.join(self.coco_dirpath, img_file),
            {"COCO_id": img_id},
        )

    def get_score(self, log_list, log_file_path) -> None:
        # Since the resampled CHAIR is only used for head identification, to
        # prevent the results from being printed, this section was left blank
        # intentionally.
        pass


def register():
    # The benchmarks are registered automatically after importing this file.
    pass
