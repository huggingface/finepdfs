from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.templates.utils.formulation import CFFormulation, HybridFormulation, MCFFormulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.continuation import get_continuation_prompt_function
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.metrics.dynamic_metrics import loglikelihood_acc_metric, probability_metric
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.multilingual.adapters import winogrand_adapter
from lighteval.metrics.dynamic_metrics import multilingual_quasi_exact_match_metric, multilingual_quasi_f1_score_metric
from lighteval.tasks.multilingual.tasks import xcsqa_tasks
from lighteval.utils.language import Language
from functools import partial
from lighteval.metrics.metrics import Metrics

# ENGLISH

MMLU_SUBSETS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

arcs = [LightevalTaskConfig(
    name=f"arc_cf:{subset}",
    prompt_function=get_mcq_prompt_function(
        Language.ENGLISH,
        lambda line: {
            "question": line["question"],
            "choices": line["choices"]["text"],
            "gold_idx": line["choices"]["label"].index(line["answerKey"]),
        },
        formulation=CFFormulation(),
    ),
    suite=("lighteval",),
    hf_repo="allenai/ai2_arc",
    hf_subset="ARC-Easy" if subset == "easy" else "ARC-Challenge",
    evaluation_splits=("test",),
    few_shots_split="validation",
    metric=get_metrics_for_formulation(
        CFFormulation(),
        [
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
            probability_metric(normalization=LogProbTokenNorm()),
        ],
        ),
    )
    for subset in ["easy", "challenge"]
]
hellaswag_task = [
    LightevalTaskConfig(
        name=f"hellaswag_cf",
        suite=["lighteval"],
        prompt_function=get_hellaswag_prompt_function(
            language=Language.ENGLISH,
            adapter=lambda line: {
                # We don't use activity_label as they are not available
                "ctx_a": line["ctx_a"],
                "ctx_b": line["ctx_b"],
                "continuations": line["endings"],
                "gold_idx": int(line["label"]),
            },
            formulation=CFFormulation(),
        ),
        hf_repo="Rowan/hellaswag",
        hf_subset="default",
        evaluation_splits=["validation"],
        hf_avail_splits=["train", "validation"],
        metric=get_metrics_for_formulation(
            CFFormulation(),
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                probability_metric(normalization=LogProbTokenNorm()),
            ],
        ),
    )
]

def mmlu_redux_prompt(line):
    if line["error_type"] not in ["wrong_groundtruth", "ok"]:
        return None
    gold_idx = line["answer"] if line["error_type"] != "wrong_groundtruth" else line["correct_answer"]
    if gold_idx is None:
        return None

    try:
        gold_idx = LETTER_INDICES.index(gold_idx)
    except ValueError:
        gold_idx = int(gold_idx)

    
    choices = line["choices"]
    return {
        "question": line["question"],
        "choices": choices,
        "gold_idx": int(gold_idx),
    }

mmlu_redux = [
    LightevalTaskConfig(
        name=f"mmlu_redux_cf:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            mmlu_redux_prompt,
            formulation=CFFormulation(),
        ),
        suite=("lighteval",),
        hf_repo="edinburgh-dawg/mmlu-redux-2.0",
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        hf_subset=subset,
        metric=get_metrics_for_formulation(
            CFFormulation(),
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
                probability_metric(normalization=LogProbTokenNorm()),
            ],
        ),
    )
    for subset in MMLU_SUBSETS
]

gsm8k = LightevalTaskConfig(
    name="gsm8k",
    suite=["lighteval"],
    prompt_function=get_qa_prompt_function(
        Language.ENGLISH,
        lambda line: {
            "question": line["question"],
            "choices": [f"{line['answer'].split('####')[-1].strip()}"],
        },
    ),
    hf_repo="gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    generation_size=256,
    metric=[
        # Metrics.expr_gold_metric,
        probability_metric(normalization=LogProbTokenNorm()),
    ],
    stop_sequence=["Question:", "Question", "question", "question:", "\n"],
)


def get_drop_date(x):
    components = [x["day"], x["month"], x["year"]]
    components = list(filter(lambda x: x, components))
    return " ".join(components)

drop_qa = LightevalTaskConfig(
    name="drop",
    prompt_function=get_qa_prompt_function(
        Language.ENGLISH,
        lambda line: {
            "context": line["passage"],
            "question": line["question"],
            "choices": list(
                filter(
                    lambda x: x,
                    [line["answer"].get("number")] + line["answer"]["spans"] + [get_drop_date(line["answer"].get("date"))],
                )
            ),
        },
    ),
    suite=("lighteval",),
    hf_repo="lighteval/drop_harness",
    hf_subset="default",
    hf_filter=lambda line: list(
        filter(
            lambda x: x,
            [line["answer"].get("number")] + line["answer"]["spans"] + [get_drop_date(line["answer"].get("date"))],
        )
    ),
    evaluation_splits=("validation",),
    few_shots_split="train",
    generation_size=300,
    stop_sequence=["Question:", "Question", "question", "question:", "\n"],
    metric=(
        probability_metric(normalization=LogProbTokenNorm()),
        # multilingual_quasi_exact_match_metric(Language.ENGLISH, "prefix"),
        # multilingual_quasi_f1_score_metric(Language.ENGLISH),
    ),
)

squad_v2 = LightevalTaskConfig(
    name="squad_v2",
    prompt_function=get_qa_prompt_function(
        Language.ENGLISH,
        lambda line: {
            "question": line["question"],
            "context": line["context"],
            "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
        },
    ),
    suite=("lighteval",),
    hf_repo="lighteval/squad_v2",
    hf_subset="default",
    hf_filter=lambda line: any(ans for ans in line["answers"]["text"] if len(ans) > 0),
    evaluation_splits=("validation",),
    few_shots_split="train",
    stop_sequence=["\n", "Question:", "Question", "question", "question:"],
    generation_size=300,
    metric=(
        probability_metric(normalization=LogProbTokenNorm()),
        # multilingual_quasi_exact_match_metric(Language.ENGLISH, "prefix"),
        # multilingual_quasi_f1_score_metric(Language.ENGLISH),
    ),
)

en_xcsqa = [
    LightevalTaskConfig(
        name=f"xcsqa_cf",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"]["stem"],
                "choices": line["question"]["choices"]["text"],
                "gold_idx": line["question"]["choices"]["label"].index(line["answerKey"]),
            },
            formulation=CFFormulation(),
        ),
        suite=("lighteval",),
        hf_repo="INK-USC/xcsr",
        hf_subset="X-CSQA-en",
        hf_filter=lambda x: all(
            len(x["question"]["choices"]["text"][i].strip()) > 0 for i in range(len(x["question"]["choices"]["text"]))
        ),
        evaluation_splits=("validation",),
        hf_avail_splits=["validation"],
        metric=get_metrics_for_formulation(
            CFFormulation(),
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
                probability_metric(normalization=LogProbTokenNorm()),
            ],
        ),
    )
]

openbook_qa_tasks = [
    LightevalTaskConfig(
        name=f"openbookqa_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question_stem"],
                "choices": line["choices"]["text"],
                "gold_idx": LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=["lighteval"],
        hf_repo="allenai/openbookqa",
        hf_subset="main",
        hf_revision="388097ea7776314e93a529163e0fea805b8a6454",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=get_metrics_for_formulation(formulation, [
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
            loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            probability_metric(normalization=LogProbTokenNorm()),
        ]),
    )
    for formulation in [CFFormulation()]
]

winogrande_tasks = [
    LightevalTaskConfig(
        name=f"winogrande_{formulation.name.lower()}",
        suite=("lighteval",),
        prompt_function=get_continuation_prompt_function(
            Language.ENGLISH,
            partial(winogrand_adapter, Language.ENGLISH),
            formulation=formulation,
        ),
        hf_repo="allenai/winogrande",
        hf_subset="winogrande_xl",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=get_metrics_for_formulation(formulation, [
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
            loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            probability_metric(normalization=LogProbTokenNorm()),
        ]),
    )
    for formulation in [CFFormulation()]
]

piqa_tasks = [
    LightevalTaskConfig(
        name=f"piqa_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["goal"],
                "choices": [line["sol1"], line["sol2"]],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        suite=["lighteval"],
        hf_repo="lighteval/piqa",
        hf_subset="plain_text",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        few_shots_split="train",
        metric=get_metrics_for_formulation(formulation, [
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
            loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            probability_metric(normalization=LogProbTokenNorm()),
        ]),
    )
    for formulation in [CFFormulation()]
]


treb_qa = LightevalTaskConfig(
    name="treb_qa",
    prompt_function=get_qa_prompt_function(
        Language.ENGLISH,
        lambda line: {
            "context": line["Table_markdown"][0],
            "question": line["question"],
            "choices": [line["answer"]],
        },
    ),
    suite=("lighteval",),
    hf_repo="lighteval/treb_table_retrieval",
    hf_subset="default",
    hf_filter=lambda line: len(line["Table_markdown"]) == 1,
    evaluation_splits=("test",),
    generation_size=256,
    metric=(
        probability_metric(normalization=LogProbTokenNorm()),
        # multilingual_quasi_exact_match_metric(Language.ENGLISH, "prefix"),
        # multilingual_quasi_f1_score_metric(Language.ENGLISH),
    ),
    stop_sequence=["\n", "Question:", "Question", "question", "question:"],
)


wikitablequestions = LightevalTaskConfig(
    name="wikitablequestions",
    prompt_function=get_qa_prompt_function(
        Language.ENGLISH,
        lambda line: {
            "context": line["table_md"],
            "question": line["question"],
            "choices": line["answers"],
        },
    ),
    suite=("lighteval",),
    hf_repo="lighteval/wikitablequestions",
    hf_subset="default",
    evaluation_splits=("test",),
    generation_size=256,
    metric=(
        probability_metric(normalization=LogProbTokenNorm()),
        # multilingual_quasi_exact_match_metric(Language.ENGLISH, "prefix"),
        # multilingual_quasi_f1_score_metric(Language.ENGLISH),

    ),
    stop_sequence=["\n", "Question:", "Question", "question", "question:"],
)




TASKS_TABLE = [
    *arcs,
    *hellaswag_task,
    *mmlu_redux,
    *openbook_qa_tasks,
    *winogrande_tasks,
    *piqa_tasks,
    gsm8k,
    drop_qa,
    squad_v2,
    *en_xcsqa,
    treb_qa,
    wikitablequestions,
]