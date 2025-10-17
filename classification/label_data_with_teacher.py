"""
Basic inference pipeline example without chunking.

This example shows how to run inference on documents using the InferenceRunner
without chunking. Documents are processed and saved to a simple output structure.
"""

import argparse     
from datatrove.data import Document
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.readers import JsonlReader
from datatrove.executor.local import LocalPipelineExecutor


async def async_query_builder(runner, document: Document):

    FW_EDU = """\
Below is an extract from a PDF file. Evaluate whether the extract has a high educational
value and could be useful in an educational setting for teaching from primary school to
grade school levels using the additive 5-point scoring system described below. Points are
accumulated based on the satisfaction of each criterion:
- Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and
promotional material.
- Add another point if the extract addresses certain elements pertinent to education but
does not align closely with educational standards. It might mix educational content with
non-educational material, offering a superficial overview of potentially useful topics, or
presenting information in a disorganized manner and incoherent writing style.
- Award a third point if the extract is appropriate for educational use and introduces key
concepts relevant to school curricula. It is coherent though it may not be comprehensive
or could include some extraneous information. It may resemble an introductory section of
a textbook or a basic tutorial that is suitable for learning but has notable limitations like
treating concepts that are too complex for grade school students.
- Grant a fourth point if the extract highly relevant and beneficial for educational purposes
for a level not higher than grade school, exhibiting a clear and consistent writing style. It
could be similar to a chapter from a textbook or a tutorial, offering substantial educational
content, including exercises and solutions, with minimal irrelevant information, and the
concepts aren’t too advanced for grade school students. The content is coherent, focused,
and valuable for structured learning.
- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for
teaching either at primary school or grade school. It follows detailed reasoning, the writing
style is easy to follow and offers profound and thorough insights into the subject matter,
devoid of any non-educational or complex content.
The extract: {example}.
After examining the extract:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Educational score: <total points>"\
"""

    FW_EDU_2 = """\
Below is an extract from a PDF file. Evaluate whether the extract exhibits properties suitable for educational training data using the 6-point scoring system described below. Select the single score that best represents the extract's educational quality level:

**Score 0: No Educational Value**
- Award 0 points for content with zero educational merit including spam, promotional material, garbled text, random sequences, severely corrupted formatting, or content that provides no learning opportunities whatsoever.

**Score 1: Minimal Educational Content**
- Award 1 point for content with very limited educational value such as basic data listings, simple contact information, minimal factual statements without context, brief announcements, or content that presents isolated facts without meaningful educational framework.

**Score 2: Basic Informational Content**
- Award 2 points for content that provides basic information but lacks depth, context, or clear educational structure. This includes simple news items, basic product descriptions, brief summaries, casual observations, or informational content that states facts without explanation or educational development.

**Score 3: Moderate Educational Value**
- Award 3 points for content that offers solid educational information with some context and explanation. This includes informative articles with background information, basic explanatory content, introductory-level material, general knowledge content, or well-written informational pieces that provide context and some depth.

**Score 4: Strong Educational Content**
- Award 4 points for content with clear educational merit featuring detailed explanations, multiple perspectives, analytical depth, or comprehensive coverage of topics. This includes academic articles, detailed tutorials, in-depth analyses, research-based content, or material that demonstrates critical thinking and provides substantial learning value.

**Score 5: Exceptional Educational Value**
- Award 5 points for content with outstanding educational merit that demonstrates expert-level knowledge, sophisticated analysis, comprehensive understanding, and significant pedagogical value. This includes advanced academic research, expert commentary with deep insights, comprehensive educational material with multiple learning dimensions, or content that advances understanding through original thinking and thorough exploration.

## Evaluation Process
The extract: {example}

After examining the extract:
- Briefly justify your total score, focusing on the educational depth, context provided, and learning potential, up to 100 words.
- Conclude with the score using the format: "Educational value score: <total points>"\
"""


    DCLM="""\
Below is an extract from a PDF file. Evaluate whether the extract exhibits properties suitable for instruction-following or question-answering training data using the 6-point scoring system described below. Select the single score that best represents the extract's quality level:

**Score 0: Spam, Garbled, or Completely Unusable Content**
- Award 0 points for SEO spam content, promotional material with no educational value, completely garbled/corrupted text that is unreadable, random character sequences, or severely corrupted formatting that makes the content incomprehensible.

**Score 1: Simple Lists, Forms, or Minimal-Value Content**
- Award 1 point for content that has basic readable formatting but consists primarily of simple lists without context, forms, contact information, schedules, basic data tables without explanation, or other minimal-value structured content that lacks meaningful narrative or educational substance.

**Score 2: Cohesive Text Without Educational Value**
- Award 2 points if the extract contains cohesive, well-structured text that flows logically but lacks educational or instructional value. This includes meeting reports, business correspondence, letters, basic manual descriptions, administrative documents, or narrative content that doesn't teach or explain concepts.

**Score 3: Educational Content Without Q&A Structure**
- Award 3 points if the extract contains educational or informational content that could be useful for learning but doesn't follow a clear instructional format. This includes Wikipedia-style articles, research papers, academic content, encyclopedic entries, or explanatory text that presents information without explicit teaching structure.

**Score 4: Instructional Manuals and Structured Q&A**
- Award 4 points if the extract demonstrates clear instructional format with identifiable structure such as how-to guides, instruction manuals, structured question-answer pairs, problem-solution formats, or other organized pedagogical patterns. The content should be well-organized and follow recognizable educational conventions.

**Score 5: High-Quality Instructional Content with Explanations**
- Award 5 points if the extract exhibits exemplary instruction-response or question-answer properties with clear reasoning and detailed explanations. It should demonstrate thoughtful, step-by-step reasoning found in high-quality educational content like comprehensive tutorials, detailed explanations with context and reasoning, or expert-level instructional material that provides not just answers but explanatory reasoning and educational depth.

## Evaluation Process

The extract: {example}

After examining the extract:
- Briefly justify your total score, focusing on the content type and instructional/explanatory qualities, up to 100 words.
- Conclude with the score using the format: "Instruction/Q&A score: <total points>\
"""

    OCR_QUALITY = """\
Below is an extract from a PDF file. Evaluate the quality of the document extraction using the 4-point string system described below. Select the single score that best represents the extraction quality level:

**Score 0: Garbage Text Present**
- Award 0 points if there are any garbage artifacts present in the text, regardless of how much legitimate content surrounds them. This includes OCR corruption like random character sequences (e.g., "7*/3./ +*/ 6- 4603"), unreadable symbol combinations, corrupted encoding artifacts, or any form of garbled text that renders portions of the document incomprehensible. Even if 90% of the text is perfectly readable, the presence of any garbage characters results in a score of 0.

**Score 1: Clear Formatting Issues**
- Award 1 point if there are no garbage characters but clear formatting problems are present. This includes broken mathematical equations or formulas that are unreadable, excessive or irregular spacing that disrupts readability, malformed tables or lists, severely corrupted line breaks, or other structural formatting issues that significantly impact the document's usability while keeping the text itself readable.

**Score 2: Minor Formatting Problems**
- Award 2 points if there are no garbage characters but minor formatting issues exist. This includes scattered extra spaces within words or sentences (e.g., "A t t h e S how"), inconsistent spacing, minor alignment issues, occasional broken line formatting, or small structural problems that don't severely impact readability but indicate imperfect extraction quality.

**Score 3: Clean Extraction**
- Award 3 points if there are no OCR garbage artifacts, no significant formatting issues, and the text extraction preserves the document's structure and readability effectively. The content should be clean, properly formatted, and easily readable with minimal to no extraction artifacts.

## Evaluation Process
The extract: {example}

After examining the extract:
- Briefly justify your score, focusing specifically on the presence of garbage text, formatting issues, and overall extraction quality, up to 100 words.
- Conclude with the score using the format: "Document extraction score: <total points>"\
"""
    document.metadata = {
        "origin": document.metadata["origin"],
        "chunks": document.metadata["chunks"],
    }
    document.text = ""
    document.media = []
    for prompt_type in [
        # As we found only fw_edu to be the most effective
        FW_EDU,
    ]:
        for chunk in document.metadata["chunks"]:
            prompt = prompt_type.format(example=chunk)
            yield {
                "messages": [
                    {
                        "role": "user", 
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
                "max_tokens": 512,
            }

def postprocess(document: Document):
    import re
    from datatrove.pipeline.inference.run_inference import InferenceError, InferenceSuccess
    def parse_fw_edu_score(result: InferenceSuccess | InferenceError):
        if isinstance(result, InferenceError):
            return None
        match = re.search(r"Educational score:\s*(\d+)", result.text)
        if match:
            return int(match.group(1))
        return None
    n_chunks = len(document.metadata["chunks"])
    fw_edu_scores = [
        [parse_fw_edu_score(result) for result in document.metadata["inference_results"][:n_chunks]],
    ]
    document.metadata["fw_edu_scores"] = fw_edu_scores

    del document.metadata["inference_results"]
    return document



INPUT_DIR = "./finepdfs/data/exact_dedup/per_language/output"
OUTPUT_DIR = "./finepdfs/data/classification_only_top_bottom_300k/teacher"


def run_pipeline_for_tokens():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    parser.add_argument("--languages", type=str, default="eng_Latn")
    parser.add_argument("--gpus", type=int, default=8)
    args = parser.parse_args()
    model_name = args.model_name
    model_kwargs = {}
    # model_kwargs = {
    #     "limit-mm-per-prompt.image": 0,
    #     "limit-mm-per-prompt.video": 0,
    # }
    if "oss" not in model_name and "FP8" not in model_name:
        model_kwargs["quantization"] = "fp8"

    if "mistral" in model_name:
        model_kwargs["tokenizer-mode"] = "mistral"

    sampling_rate = 1.0 # Compute so that you get ~300k samples
    

    config = InferenceConfig(
        server_type="vllm",
        model_name_or_path=model_name,
        temperature=0.0,
        model_max_context=16384,
        max_concurrent_requests=400,
        max_concurrent_tasks=400,
        model_kwargs=model_kwargs,
        tp=args.gpus,
    )

    languages = args.languages.split(",")

    for lang in languages:
        pipeline=[
            JsonlReader(
                data_folder=f"{INPUT_DIR}/{lang}",
                glob_pattern=f"*.jsonl.gz",
                shuffle_files=True,
                doc_progress=True,

            ),
            #
            SamplerFilter(
                rate=sampling_rate,
            ),
            InferenceRunner(
                query_builder=async_query_builder,
                postprocess_fn=postprocess,
                config=config,
                output_writer=JsonlWriter(
                    output_folder=f"{OUTPUT_DIR}/{lang}/{args.model_name.replace('/', '_')}",
                    output_filename="${rank}_chunk_${chunk_index}.jsonl.gz",  # Chunked filename pattern
                ),
            ),
        ]
        executor = LocalPipelineExecutor(pipeline)
        executor.run()
    

if __name__ == "__main__":
    run_pipeline_for_tokens()




